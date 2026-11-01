import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


DATA_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/data")
TRAIN_PRICES = DATA_DIR / "train_files/stock_prices.csv"
STOCK_LIST = DATA_DIR / "stock_list.csv"
OUTPUT_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/outputs")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_prices(usecols: Optional[List[str]] = None) -> pd.DataFrame:
    default_cols = [
        "Date",
        "SecuritiesCode",
        "Close",
        "Volume",
        "AdjustmentFactor",
        "SupervisionFlag",
        "Target",
    ]
    cols = usecols or default_cols
    dtypes = {
        "SecuritiesCode": np.int32,
        "Close": np.float32,
        "Volume": np.float64,
        "AdjustmentFactor": np.float32,
        "Target": np.float32,
    }
    parse_dates = ["Date"] if "Date" in cols else None
    df = pd.read_csv(
        TRAIN_PRICES,
        usecols=cols,
        dtype=dtypes,
        parse_dates=parse_dates,
        low_memory=False,
    )
    if "SupervisionFlag" in df.columns:
        # Convert to boolean if it came in as object
        if df["SupervisionFlag"].dtype == object:
            df["SupervisionFlag"] = df["SupervisionFlag"].astype(str).str.lower().map({"true": True, "false": False})
        df["SupervisionFlag"] = df["SupervisionFlag"].fillna(False).astype(bool)
    return df


def read_universe() -> pd.DataFrame:
    cols = ["SecuritiesCode", "Universe0"]
    df = pd.read_csv(STOCK_LIST, usecols=cols, dtype={"SecuritiesCode": np.int32})
    # Universe0 can be TRUE/FALSE strings
    if df["Universe0"].dtype == object:
        df["Universe0"] = df["Universe0"].astype(str).str.lower().map({"true": True, "false": False})
    return df


def add_adjusted_close(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["SecuritiesCode", "Date"]).copy()
    # Cumulative adjustment factor per code; forward-adjust series
    prices["cum_adj"] = prices.groupby("SecuritiesCode")["AdjustmentFactor"].cumprod()
    prices["AdjClose"] = prices["Close"] * prices["cum_adj"]
    return prices


def add_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["SecuritiesCode", "Date"]).copy()
    # Daily adjusted return
    prices["adj_ret1"] = prices.groupby("SecuritiesCode")["AdjClose"].pct_change()
    # Momentum features using only information up to t (no lookahead)
    prices["mom5"] = prices.groupby("SecuritiesCode")["AdjClose"].pct_change(5)
    prices["mom20"] = prices.groupby("SecuritiesCode")["AdjClose"].pct_change(20)
    # Volatility (past 20-day realized volatility)
    prices["vol20"] = (
        prices.groupby("SecuritiesCode")["adj_ret1"].rolling(20).std().reset_index(level=0, drop=True)
    )
    # Combine into a simple score: favor momentum, penalize high volatility
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    prices["score_raw"] = 0.5 * prices["mom5"] + 0.5 * prices["mom20"]
    prices["score"] = prices["score_raw"] / (prices["vol20"].replace(0.0, np.nan) + eps)
    return prices


def rank_and_select(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    top_n: int = 200,
    bottom_n: int = 200,
) -> pd.DataFrame:
    df = prices.merge(universe, on="SecuritiesCode", how="left")
    df = df[(df["Universe0"] == True) & (~df["SupervisionFlag"])]  # noqa: E712
    # Keep rows where score is finite
    df = df[np.isfinite(df["score"])].copy()

    # Rank per day (high score = bullish)
    def select_per_day(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("score", ascending=False)
        long = g.head(top_n).copy()
        short = g.tail(bottom_n).copy()
        long["Side"] = "long"
        short["Side"] = "short"
        return pd.concat([long, short], axis=0)

    selected = df.groupby("Date", group_keys=False).apply(select_per_day)
    return selected.sort_values(["Date", "Side", "score"], ascending=[True, True, False])


def evaluate_long_short(selected: pd.DataFrame) -> pd.DataFrame:
    # On date t, evaluation uses Target (defined on t) as future (t+2 vs t+1) return
    eval_df = selected.dropna(subset=["Target"]).copy()
    # Mean long and short by day
    daily = (
        eval_df.pivot_table(
            index="Date",
            columns="Side",
            values="Target",
            aggfunc="mean",
        )
        .rename(columns={"long": "long_mean", "short": "short_mean"})
        .reset_index()
    )
    if "long_mean" not in daily.columns:
        daily["long_mean"] = np.nan
    if "short_mean" not in daily.columns:
        daily["short_mean"] = np.nan
    daily["ls_return"] = daily["long_mean"] - daily["short_mean"]
    return daily.sort_values("Date")


def compute_sharpe(daily_ls: pd.DataFrame) -> float:
    r = daily_ls["ls_return"].dropna()
    if r.empty:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(mu / sd)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    # Load data
    prices = read_prices()
    universe = read_universe()

    # Engineering
    prices = add_adjusted_close(prices)
    prices = add_features(prices)

    # Selection
    selected = rank_and_select(prices, universe, top_n=200, bottom_n=200)

    # Evaluation
    daily_ls = evaluate_long_short(selected)
    sharpe = compute_sharpe(daily_ls)

    # Write outputs
    selected_out = OUTPUT_DIR / "daily_selection.csv"
    daily_ls_out = OUTPUT_DIR / "daily_long_short_returns.csv"
    summary_out = OUTPUT_DIR / "summary.txt"

    selected[["Date", "SecuritiesCode", "Side", "score"]].to_csv(selected_out, index=False)
    daily_ls.to_csv(daily_ls_out, index=False)
    with open(summary_out, "w", encoding="utf-8") as f:
        f.write(f"Sharpe: {sharpe:.6f}\n")
        f.write(f"Days: {daily_ls['ls_return'].notna().sum()}\n")
        f.write(f"Start: {daily_ls['Date'].min()}\n")
        f.write(f"End: {daily_ls['Date'].max()}\n")

    print(f"Sharpe: {sharpe:.6f}")
    print(f"Saved: {selected_out}")
    print(f"Saved: {daily_ls_out}")
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()



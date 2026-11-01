from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


DATA_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/data")
TRAIN_PRICES = DATA_DIR / "train_files/stock_prices.csv"
STOCK_LIST = DATA_DIR / "stock_list.csv"
OUTPUT_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/outputs") / "mlp"


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
        if df["SupervisionFlag"].dtype == object:
            df["SupervisionFlag"] = (
                df["SupervisionFlag"].astype(str).str.lower().map({"true": True, "false": False})
            )
        df["SupervisionFlag"] = df["SupervisionFlag"].fillna(False).astype(bool)
    return df


def read_universe() -> pd.DataFrame:
    cols = ["SecuritiesCode", "Universe0"]
    df = pd.read_csv(STOCK_LIST, usecols=cols, dtype={"SecuritiesCode": np.int32})
    if df["Universe0"].dtype == object:
        df["Universe0"] = df["Universe0"].astype(str).str.lower().map({"true": True, "false": False})
    return df


def add_adjusted_close(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["SecuritiesCode", "Date"]).copy()
    prices["cum_adj"] = prices.groupby("SecuritiesCode")["AdjustmentFactor"].cumprod()
    prices["AdjClose"] = prices["Close"] * prices["cum_adj"]
    return prices


def add_multi_scale_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["SecuritiesCode", "Date"]).copy()
    prices["ret1"] = prices.groupby("SecuritiesCode")["AdjClose"].pct_change()
    for n in [1, 2, 5, 10, 20]:
        prices[f"mom{n}"] = prices.groupby("SecuritiesCode")["AdjClose"].pct_change(n)
    for n in [5, 10, 20]:
        prices[f"vol{n}"] = (
            prices.groupby("SecuritiesCode")["ret1"].rolling(n).std().reset_index(level=0, drop=True)
        )
    prices["dv"] = prices["Close"] * prices["Volume"]
    for n in [5, 10, 20]:
        prices[f"dv_mean{n}"] = (
            prices.groupby("SecuritiesCode")["dv"].rolling(n).mean().reset_index(level=0, drop=True)
        )
        prices[f"vol_mean{n}"] = (
            prices.groupby("SecuritiesCode")["Volume"].rolling(n).mean().reset_index(level=0, drop=True)
        )
    prices["log_vol"] = np.log1p(prices["Volume"])
    prices["log_dv"] = np.log1p(prices["dv"])
    return prices


def cross_sectional_winsorize(df: pd.DataFrame, cols: List[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    def _clip(g: pd.DataFrame) -> pd.DataFrame:
        for c in cols:
            x = g[c]
            ql = x.quantile(lower)
            qu = x.quantile(upper)
            g[c] = x.clip(ql, qu)
        return g
    return df.groupby("Date", group_keys=False).apply(_clip)


def cross_sectional_standardize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    def _z(g: pd.DataFrame) -> pd.DataFrame:
        for c in cols:
            x = g[c]
            mu = x.mean()
            sd = x.std(ddof=1)
            g[c] = (x - mu) / (sd + 1e-6)
        return g
    return df.groupby("Date", group_keys=False).apply(_z)


def build_feature_matrix(prices: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    prices = prices.copy()
    feature_cols: List[str] = []
    for n in [1, 2, 5, 10, 20]:
        feature_cols.append(f"mom{n}")
    for n in [5, 10, 20]:
        feature_cols.append(f"vol{n}")
    feature_cols += [
        "log_vol",
        "log_dv",
        "dv_mean5",
        "dv_mean10",
        "dv_mean20",
        "vol_mean5",
        "vol_mean10",
        "vol_mean20",
    ]

    base = prices[(prices["Close"] > 0) & (prices["Volume"] >= 0)].copy()
    base = cross_sectional_winsorize(base, feature_cols, lower=0.01, upper=0.99)
    base = cross_sectional_standardize(base, feature_cols)
    return base, feature_cols


def select_outlier_training_samples(df: pd.DataFrame, target_col: str, per_side: int = 300) -> pd.DataFrame:
    def _pick(g: pd.DataFrame) -> pd.DataFrame:
        g = g.dropna(subset=[target_col]).sort_values(target_col)
        if g.shape[0] <= per_side * 2:
            return g.assign(label_bin=0)
        bottom = g.head(per_side).assign(label_bin=0)
        top = g.tail(per_side).assign(label_bin=1)
        return pd.concat([bottom, top], axis=0)
    return df.groupby("Date", group_keys=False).apply(_pick)


def split_train_valid(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = np.sort(df["Date"].unique())
    cut_idx = int(len(dates) * 0.8)
    cutoff_date = pd.to_datetime(dates[cut_idx])
    train = df[df["Date"] < cutoff_date].copy()
    valid = df[df["Date"] >= cutoff_date].copy()
    return train, valid, cutoff_date


def train_mlp_classifier(
    X: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 100,
    batch_size: int = 4096,
    hidden_dims: List[int] = [384, 192, 96],
    lr: float = 2e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers: List[nn.Module] = []
            d_prev = in_dim
            for d in hidden_dims:
                layers.extend([nn.Linear(d_prev, d), nn.ReLU(inplace=True), nn.BatchNorm1d(d), nn.Dropout(0.1)])
                d_prev = d
            layers.append(nn.Linear(d_prev, 1))  # logits
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = MLP().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(num_epochs):
        total = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"epoch {epoch+1}: bce_loss={total/max(n,1):.6f}")
    model.eval()
    return model


def predict_mlp_proba(model, X: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            xb = torch.from_numpy(X[i:i+8192]).to(device)
            logits = model(xb)
            pb = torch.sigmoid(logits).detach().cpu().numpy()
            preds.append(pb.reshape(-1))
    return np.concatenate(preds, axis=0)


def rank_and_select(df: pd.DataFrame, universe: pd.DataFrame, top_n: int = 200, bottom_n: int = 200) -> pd.DataFrame:
    merged = df.merge(universe, on="SecuritiesCode", how="left")
    tradable = merged[(merged["Universe0"] == True) & (~merged["SupervisionFlag"]) & (merged["Close"] > 0) & (merged["Volume"] > 0)]  # noqa: E712
    tradable = tradable[np.isfinite(tradable["score"])].copy()
    def _pick(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("score", ascending=False)
        long = g.head(top_n).copy()
        short = g.tail(bottom_n).copy()
        long["Side"] = "long"
        short["Side"] = "short"
        return pd.concat([long, short], axis=0)
    selected = tradable.groupby("Date", group_keys=False).apply(_pick)
    return selected.sort_values(["Date", "Side", "score"], ascending=[True, True, False])


def evaluate_long_short(selected: pd.DataFrame, target_col: str = "Target") -> pd.DataFrame:
    eval_df = selected.dropna(subset=[target_col]).copy()
    daily = (
        eval_df.pivot_table(index="Date", columns="Side", values=target_col, aggfunc="mean")
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

    # Load and features
    prices = read_prices()
    universe = read_universe()
    prices = add_adjusted_close(prices)
    prices = add_multi_scale_features(prices)
    feat_df, feature_cols = build_feature_matrix(prices)

    # Split and outlier selection
    train_df, valid_df, cutoff_date = split_train_valid(feat_df)
    train_extreme = select_outlier_training_samples(train_df, target_col="Target", per_side=300)

    # Build tabular X/y (classification)
    X_tr = train_extreme[feature_cols].fillna(0).values.astype(np.float32)
    y_tr = train_extreme["label_bin"].values.astype(np.float32)

    X_all = feat_df[feature_cols].fillna(0).values.astype(np.float32)
    base_all = feat_df[["Date", "SecuritiesCode", "Close", "Volume", "SupervisionFlag", "Target"]].copy()

    # Train MLP classifier
    model = train_mlp_classifier(X_tr, y_tr, num_epochs=5, batch_size=4096, hidden_dims=[384,192,96], lr=2e-3)

    # Predict probabilities -> per-day z-score as scores
    proba = predict_mlp_proba(model, X_all)
    scored = base_all.copy()
    scored["proba"] = proba
    scored["score"] = scored.groupby("Date")["proba"].transform(lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-6))

    # Select long/short
    selected = rank_and_select(scored, universe, top_n=200, bottom_n=200)

    # Evaluate OOS only
    selected_oos = selected[selected["Date"] >= cutoff_date]
    daily_ls = evaluate_long_short(selected_oos, target_col="Target")
    sharpe = compute_sharpe(daily_ls)

    # Write outputs
    out_sel = OUTPUT_DIR / "daily_selection.csv"
    out_ls = OUTPUT_DIR / "daily_long_short_returns.csv"
    out_summary = OUTPUT_DIR / "summary.txt"
    out_scored = OUTPUT_DIR / "scored.csv"
    selected[["Date", "SecuritiesCode", "Side", "score"]].to_csv(out_sel, index=False)
    daily_ls.to_csv(out_ls, index=False)
    scored[["Date", "SecuritiesCode", "score", "Target", "Close", "Volume", "SupervisionFlag"]].to_csv(out_scored, index=False)
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write(f"Sharpe (OOS): {sharpe:.6f}\n")
        f.write(f"OOS Days: {daily_ls['ls_return'].notna().sum()}\n")
        f.write(f"OOS Start: {daily_ls['Date'].min()}\n")
        f.write(f"OOS End: {daily_ls['Date'].max()}\n")

    print(f"Sharpe (OOS): {sharpe:.6f}")
    print(f"Saved: {out_sel}")
    print(f"Saved: {out_ls}")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_scored}")


if __name__ == "__main__":
    main()




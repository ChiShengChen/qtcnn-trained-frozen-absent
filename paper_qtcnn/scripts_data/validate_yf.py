"""Cross-validate a Yahoo Finance feed against the Kaggle JPX panel.

The extension is only worth building if returns reconstructed from the new
source line up with the returns the study already uses. The supplemental file
(2021-12-06 to 2022-06-24) overlaps both, so it is the natural test bed.

Conventions differ and must be reconciled before comparing:
  * Kaggle gives a raw traded Close plus a separate AdjustmentFactor, which we
    accumulate backwards (see data.add_adjusted_close).
  * Yahoo's Close under auto_adjust=False is already split-adjusted but not
    dividend-adjusted, which matches the JPX convention; Adj Close is not what
    we want because it also folds in dividends.
So the comparison is between daily returns, not price levels.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qtcnn_research import data as D  # noqa: E402

OVERLAP_START = pd.Timestamp("2021-12-06")
OVERLAP_END = pd.Timestamp("2022-06-24")


def kaggle_returns(data_dir: Path, codes) -> pd.DataFrame:
    prices = D.load_prices(data_dir, include_supplemental=True)
    prices = prices[prices["SecuritiesCode"].isin(codes)]
    prices = D.add_adjusted_close(prices)
    prices = prices.sort_values(["SecuritiesCode", "Date"])
    prices["ret"] = prices.groupby("SecuritiesCode")["AdjClose"].pct_change(fill_method=None)
    window = prices[(prices["Date"] >= OVERLAP_START) & (prices["Date"] <= OVERLAP_END)]
    return window[["Date", "SecuritiesCode", "Close", "Volume", "ret"]]


def yahoo_returns(codes, pause: float = 0.4) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    missing = []
    # Batch rather than one request per code, and pause between batches.
    codes = list(codes)
    for i in range(0, len(codes), 20):
        batch = codes[i:i + 20]
        tickers = [f"{c}.T" for c in batch]
        raw = yf.download(tickers, start="2021-11-01", end="2022-07-05",
                          auto_adjust=False, progress=False, threads=False,
                          group_by="ticker")
        for code, ticker in zip(batch, tickers):
            try:
                sub = raw[ticker] if len(tickers) > 1 else raw
            except KeyError:
                missing.append(code)
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                missing.append(code)
                continue
            frame = pd.DataFrame({
                "Date": pd.to_datetime(sub.index).tz_localize(None).normalize(),
                "SecuritiesCode": code,
                "yf_close": sub["Close"].to_numpy(),
                "yf_volume": sub["Volume"].to_numpy(),
            })
            frame["yf_ret"] = frame["yf_close"].pct_change()
            frames.append(frame)
        time.sleep(pause)
    if missing:
        print(f"  no data for {len(missing)} codes: {missing[:10]}", flush=True)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), missing


def main() -> None:
    n_codes = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    data_dir = ROOT.parent / "data"

    universe = D.load_universe(data_dir)
    codes = universe.loc[universe["Universe0"], "SecuritiesCode"].to_numpy()
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(codes, min(n_codes, len(codes)), replace=False))
    print(f"validating {len(sample)} of {len(codes)} Universe0 securities", flush=True)

    kag = kaggle_returns(data_dir, sample)
    print(f"  kaggle rows in overlap: {len(kag):,}", flush=True)

    yah, missing = yahoo_returns(sample)
    if yah.empty:
        print("  yahoo returned nothing; extension not feasible from this source")
        return
    yah = yah[(yah["Date"] >= OVERLAP_START) & (yah["Date"] <= OVERLAP_END)]
    print(f"  yahoo rows in overlap: {len(yah):,}", flush=True)

    merged = kag.merge(yah, on=["Date", "SecuritiesCode"], how="inner")
    print(f"  matched rows: {len(merged):,}", flush=True)
    if merged.empty:
        print("  no overlap after merge -- check date alignment")
        return

    ok = merged.dropna(subset=["ret", "yf_ret"])
    diff = (ok["ret"] - ok["yf_ret"]).abs()
    print("\n=== daily return agreement ===")
    print(f"  n            : {len(ok):,}")
    print(f"  corr         : {ok['ret'].corr(ok['yf_ret']):.6f}")
    print(f"  median |diff|: {diff.median():.2e}")
    print(f"  p99   |diff| : {diff.quantile(0.99):.2e}")
    print(f"  max   |diff| : {diff.max():.2e}")
    print(f"  frac >1e-4   : {(diff > 1e-4).mean():.4f}")
    print(f"  frac >1e-2   : {(diff > 1e-2).mean():.4f}")

    per_code = ok.groupby("SecuritiesCode").apply(
        lambda g: pd.Series({"corr": g["ret"].corr(g["yf_ret"]),
                             "med_abs_diff": (g["ret"] - g["yf_ret"]).abs().median(),
                             "n": len(g)})
    )
    bad = per_code[per_code["corr"] < 0.99].sort_values("corr")
    print(f"\n  securities with return corr < 0.99: {len(bad)} of {len(per_code)}")
    if len(bad):
        print(bad.head(10).round(4).to_string())

    print("\n=== coverage ===")
    print(f"  codes requested : {len(sample)}")
    print(f"  codes with data : {yah['SecuritiesCode'].nunique()}")
    print(f"  missing         : {len(missing)}")

    out = ROOT / "results" / "yf_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    per_code.to_csv(out)
    print(f"\nper-security detail -> {out}")


if __name__ == "__main__":
    main()

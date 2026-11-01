"""Turn the fetched US panel into the same feature cache the JPX pipeline consumes.

Universe selection (point-in-time, first calendar year only): among symbols with at
least 200 trading days in 2017 and a median 2017 close >= $2, the 2000 with the
highest median 2017 dollar volume. Rows from all other symbols are dropped so the
cross-sectional normalisation sees exactly the universe, as in the JPX panel.
Target is C_{t+2}/C_{t+1} - 1 per symbol, the JPX definition. Integer
SecuritiesCode ids are assigned alphabetically; the map is written next to the cache.

Point-in-time price floor: security-days whose close is below --min-price are dropped.
This is the standard penny-stock screen, and it is also what removes firms in
bankruptcy reorganisation, for which the source splices the post-emergence entity onto
the pre-emergence ticker and its adjusted prices are unusable. Without it a single
security-day (CHRD, 2020-11-18, close $0.12) carries a forward return of +25733%,
which alone supplies two thirds of the panel's mean and inflates its daily standard
deviation more than fourfold. JPX needs no such screen: its worst forward return over
2.48M security-days is 62%.
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtcnn_research import data as D  # noqa: E402

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="cache/us/by_symbol")
    p.add_argument("--out", default="cache/us/features.parquet")
    p.add_argument("--n-universe", type=int, default=2000)
    p.add_argument("--select-year", type=int, default=2017)
    p.add_argument("--min-days", type=int, default=200)
    p.add_argument("--min-price", type=float, default=2.0,
                   help="minimum median close in the selection year")
    p.add_argument("--floor-price", type=float, default=1.0,
                   help="point-in-time floor: drop security-days below this close")
    p.add_argument("--start", default="2016-11-01")
    p.add_argument("--end", default="2022-06-24")
    a = p.parse_args()
    files = sorted(glob.glob(f"{a.src}/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df["Date"] >= a.start) & (df["Date"] <= a.end)]
    df = df[np.isfinite(df["Close"]) & (df["Close"] > 0)]
    print(f"raw: {len(files)} symbols, {len(df):,} rows, {df.Date.nunique()} dates", flush=True)
    y = df[df["Date"].dt.year == a.select_year]
    stats = y.groupby("symbol").agg(days=("Close", "size"), px=("Close", "median"),
                                    dv=("Close", lambda s: float(np.median(s * y.loc[s.index, "Volume"]))))
    elig = stats[(stats["days"] >= a.min_days) & (stats["px"] >= a.min_price)].sort_values("dv", ascending=False)
    uni = elig.head(a.n_universe).index.tolist()
    print(f"eligible {len(elig)} -> universe {len(uni)} (min median $vol {elig['dv'].iloc[len(uni)-1]:,.0f})", flush=True)
    df = df[df["symbol"].isin(uni)].copy()
    codes = {s: i + 1 for i, s in enumerate(sorted(uni))}
    pd.DataFrame({"SecuritiesCode": list(codes.values()), "symbol": list(codes.keys())}).merge(
        stats.reset_index(), on="symbol").to_csv(Path(a.out).with_name("universe.csv"), index=False)
    df["SecuritiesCode"] = df["symbol"].map(codes).astype(np.int32)
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    df["AdjClose"] = df["Close"]; df["AdjustmentFactor"] = 1.0
    df["SupervisionFlag"] = False; df["Universe0"] = True
    g = df.groupby("SecuritiesCode")["Close"]
    df["Target"] = g.shift(-2) / g.shift(-1) - 1.0
    df = D.add_features(df, D.FeatureSpec())
    feat = D.FeatureSpec().columns()
    keep = ["Date", "SecuritiesCode", "Close", "Volume", "AdjClose", "SupervisionFlag", "Universe0", "Target"] + feat
    df = df[keep]
    df = df[np.isfinite(df[feat]).all(axis=1) & df["Target"].notna()]
    n_before = len(df)
    df = df[df["Close"] >= a.floor_price]
    print(f"point-in-time price floor ${a.floor_price:g}: dropped {n_before - len(df):,} "
          f"security-days ({100 * (n_before - len(df)) / n_before:.2f}%); "
          f"max |Target| now {df['Target'].abs().max():.3f}", flush=True)
    df = D.cross_sectional_normalise(df, feat)
    df = df.sort_values(["Date", "SecuritiesCode"], ignore_index=True)
    df.to_parquet(a.out, index=False)
    print(f"cache: {len(df):,} rows, {df.Date.nunique()} dates ({df.Date.min().date()}..{df.Date.max().date()}), "
          f"{df.SecuritiesCode.nunique()} securities, {len(feat)} features -> {a.out}", flush=True)

if __name__ == "__main__":
    main()

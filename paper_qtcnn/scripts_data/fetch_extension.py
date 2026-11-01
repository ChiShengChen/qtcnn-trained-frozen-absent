"""Fetch JPX daily prices for 2021-11 onwards and shape them like the Kaggle panel.

Validated first (scripts_data/validate_yf.py): over the 2021-12-06..2022-06-24
overlap, daily returns from this source and from the Kaggle panel agree to 1e-8
with a correlation of 1.000000. Yahoo's Close under auto_adjust=False is already
split-adjusted and not dividend-adjusted, which is the JPX convention, so it is
written straight into AdjClose and AdjustmentFactor is set to 1.0; the existing
feature pipeline then runs unchanged.

The first version of this script cached failures as if they were results: it
wrote one shard per batch and skipped existing shards on re-run, so a transient
timeout became a permanent hole. It reported 73% of securities missing, and the
gaps included securities that fetch fine in isolation. This version tracks
status per security in a manifest and only marks a code done when it either
returned rows or came back empty on several separate passes, so a code is never
written off on a single failure.

Known limitations, both reported in the paper:
  * The universe is the 2000 securities listed as of 2021. Names that delist
    simply stop appearing, which is what a portfolio would experience, but
    listings after 2022 are never added, so the universe ages.
  * SupervisionFlag has no equivalent in this feed and is set False, making the
    tradability filter looser over the extension than over the original period.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qtcnn_research import data as D  # noqa: E402

MAX_ATTEMPTS = 5  # a code is only written off after this many empty passes


def fetch_batch(codes, start: str, end: str) -> dict:
    """Return {code: DataFrame} for whichever codes actually came back."""
    import yfinance as yf

    tickers = [f"{c}.T" for c in codes]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                      progress=False, threads=False, group_by="ticker")
    out = {}
    for code, ticker in zip(codes, tickers):
        try:
            sub = raw[ticker] if len(tickers) > 1 else raw
        except KeyError:
            continue
        sub = sub.dropna(subset=["Close"])
        if sub.empty:
            continue
        out[int(code)] = pd.DataFrame({
            "Date": pd.to_datetime(sub.index).tz_localize(None).normalize(),
            "SecuritiesCode": np.int32(code),
            "Close": sub["Close"].to_numpy(dtype=np.float64),
            "Volume": sub["Volume"].to_numpy(dtype=np.float64),
        })
    return out


def load_manifest(path: Path, codes) -> pd.DataFrame:
    if path.exists():
        man = pd.read_csv(path)
        known = set(man["SecuritiesCode"])
        extra = [c for c in codes if c not in known]
        if extra:
            man = pd.concat([man, pd.DataFrame({"SecuritiesCode": extra,
                                                "status": "pending",
                                                "attempts": 0, "rows": 0})],
                            ignore_index=True)
        return man
    return pd.DataFrame({"SecuritiesCode": codes, "status": "pending",
                         "attempts": 0, "rows": 0})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-11-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--batch-size", type=int, default=5,
                   help="batches of 5 fetched reliably in testing; larger batches "
                        "were where the throttling showed up")
    p.add_argument("--pause", type=float, default=1.0)
    p.add_argument("--backoff", type=float, default=20.0,
                   help="seconds to wait after a batch that returned nothing")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--passes", type=int, default=MAX_ATTEMPTS)
    p.add_argument("--verify-pause", type=float, default=4.0,
                   help="pause between single-code fetches in the final verification pass")
    p.add_argument("--out", default="cache/extension")
    args = p.parse_args()

    data_dir = ROOT.parent / "data"
    codes = np.sort(D.load_universe(data_dir)["SecuritiesCode"].unique())
    if args.limit:
        codes = codes[: args.limit]

    out_dir = Path(args.out)
    per_code_dir = out_dir / "by_code"
    per_code_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    man = load_manifest(manifest_path, [int(c) for c in codes])

    for pass_no in range(args.passes):
        todo = man.loc[(man["status"] == "pending") &
                       (man["attempts"] <= pass_no), "SecuritiesCode"].to_numpy()
        if len(todo) == 0:
            break
        print(f"\n=== pass {pass_no + 1}: {len(todo)} securities outstanding ===", flush=True)
        t0 = time.perf_counter()
        batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

        for bi, batch in enumerate(batches):
            try:
                got = fetch_batch(batch, args.start, args.end)
            except Exception as exc:
                print(f"  batch {bi} error: {type(exc).__name__}: {exc}", flush=True)
                got = {}

            for code in batch:
                row = man["SecuritiesCode"] == int(code)
                man.loc[row, "attempts"] += 1
                if int(code) in got:
                    frame = got[int(code)]
                    frame.to_parquet(per_code_dir / f"{int(code)}.parquet", index=False)
                    man.loc[row, ["status", "rows"]] = ["ok", len(frame)]
                elif int(man.loc[row, "attempts"].iloc[0]) >= args.passes:
                    # Only now is it safe to call this a genuine absence.
                    man.loc[row, "status"] = "missing"

            if not got:
                # Whole batch empty: far more likely throttling than a run of
                # delistings, so slow down rather than burn through the list.
                time.sleep(args.backoff)
            else:
                time.sleep(args.pause)

            if bi % 20 == 0:
                man.to_csv(manifest_path, index=False)
                done = bi + 1
                rate = (time.perf_counter() - t0) / done
                ok = int((man["status"] == "ok").sum())
                print(f"  batch {done}/{len(batches)}  ok={ok}/{len(codes)}  "
                      f"eta={rate * (len(batches) - done) / 60:.1f} min", flush=True)

        man.to_csv(manifest_path, index=False)
        ok = int((man["status"] == "ok").sum())
        print(f"  pass {pass_no + 1} complete: ok={ok}, "
              f"pending={int((man['status'] == 'pending').sum())}", flush=True)

    # Final verification: anything still outstanding is retried one code at a
    # time with generous pauses. The batch failures seen in practice arrive in
    # runs of consecutive codes, which is throttling rather than a run of
    # simultaneous delistings, so a code is only called missing once it has also
    # failed on its own. The Kaggle panel says which codes were still trading at
    # the end of the original sample, so a code trading then but returning
    # nothing here is flagged as suspect rather than silently dropped.
    outstanding = man.loc[man["status"] != "ok", "SecuritiesCode"].to_numpy()
    if len(outstanding):
        print(f"\n=== verification pass: {len(outstanding)} codes, one at a time ===",
              flush=True)
        for code in outstanding:
            try:
                got = fetch_batch([int(code)], args.start, args.end)
            except Exception:
                got = {}
            row = man["SecuritiesCode"] == int(code)
            man.loc[row, "attempts"] += 1
            if int(code) in got:
                frame = got[int(code)]
                frame.to_parquet(per_code_dir / f"{int(code)}.parquet", index=False)
                man.loc[row, ["status", "rows"]] = ["ok", len(frame)]
                print(f"  recovered {int(code)} ({len(frame)} rows)", flush=True)
            else:
                man.loc[row, "status"] = "missing"
            time.sleep(args.verify_pause)
        man.to_csv(manifest_path, index=False)

    still_missing = man.loc[man["status"] == "missing", "SecuritiesCode"].to_numpy()
    if len(still_missing):
        kag_late = D.load_prices(data_dir, include_supplemental=True,
                                 usecols=["Date", "SecuritiesCode", "Close"])
        traded_late = set(
            kag_late.loc[kag_late["Date"] >= pd.Timestamp("2022-06-01"), "SecuritiesCode"]
        )
        suspect = [int(c) for c in still_missing if int(c) in traded_late]
        print(f"\n  missing: {len(still_missing)}; of those, {len(suspect)} were still "
              f"trading in the Kaggle panel in June 2022 and are therefore suspect "
              f"fetch failures rather than delistings")
        if suspect:
            print(f"  suspect codes: {suspect[:30]}")

    man.to_csv(manifest_path, index=False)
    print(f"\nmanifest: {man['status'].value_counts().to_dict()}", flush=True)

    files = sorted(per_code_dir.glob("*.parquet"))
    if not files:
        print("nothing fetched")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "SecuritiesCode"], keep="last")
    df = df[df["Close"] > 0].sort_values(["SecuritiesCode", "Date"], ignore_index=True)

    # Yahoo's Close is already split-adjusted, so no further adjustment applies.
    df["AdjustmentFactor"] = 1.0
    df["SupervisionFlag"] = False
    grp = df.groupby("SecuritiesCode", sort=False)["Close"]
    df["Target"] = grp.shift(-2) / grp.shift(-1) - 1.0

    out = out_dir / "prices_extended.parquet"
    df.to_parquet(out, index=False)
    print(f"rows={len(df):,}  securities={df.SecuritiesCode.nunique()}  "
          f"dates={df.Date.nunique()}  {df.Date.min().date()} -> {df.Date.max().date()}")
    print(f"saved -> {out}", flush=True)

    # Strongest available check: the reconstructed Target against the Kaggle one.
    kag = D.load_prices(data_dir, include_supplemental=True)[
        ["Date", "SecuritiesCode", "Target"]].rename(columns={"Target": "kaggle_target"})
    merged = df.merge(kag, on=["Date", "SecuritiesCode"], how="inner").dropna(
        subset=["Target", "kaggle_target"])
    if merged.empty:
        print("  no overlap rows to validate against")
        return
    diff = (merged["Target"] - merged["kaggle_target"]).abs()
    print("\n=== reconstructed Target vs Kaggle Target (overlap) ===")
    print(f"  n            : {len(merged):,}")
    print(f"  corr         : {merged['Target'].corr(merged['kaggle_target']):.8f}")
    print(f"  median |diff|: {diff.median():.3e}")
    print(f"  p99   |diff| : {diff.quantile(0.99):.3e}")
    print(f"  frac >1e-4   : {(diff > 1e-4).mean():.5f}", flush=True)


if __name__ == "__main__":
    main()

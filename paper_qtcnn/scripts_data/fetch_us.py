"""Fetch a US common-stock daily panel over the JPX study window (second market).

Universe: every symbol in the Nasdaq Trader symbol directories (nasdaqlisted.txt,
otherlisted.txt; snapshot saved in cache/us/) on NYSE, NYSE American or Nasdaq,
excluding ETFs, test issues, warrants, units, rights, preferreds, notes, ADRs and
SPAC shells by name pattern. Prices come from Yahoo Finance with
auto_adjust=False, whose Close is split-adjusted but not dividend-adjusted (the
JPX convention validated in scripts_data/validate_yf.py), so it is written straight
into AdjClose with AdjustmentFactor 1.0. SupervisionFlag has no equivalent and is
False.

Known limitation, reported in the paper: the symbol directory is a current
snapshot, so names delisted before the snapshot are absent (survivorship). The
ablation contrasts compare conditions trained on the identical panel, so the bias
is common to every condition; absolute performance levels are not the claim.

Manifest/retry logic mirrors fetch_extension.py: a symbol is only marked missing
after several separate empty passes plus a one-at-a-time verification pass.
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAX_ATTEMPTS = 4
BAD_NAME = re.compile(r"warrant|\bunit\b|units|\bright\b|rights|preferred|preference|depositary|depository|"
                      r"\bnote\b|notes|debenture|acquisition|\btrust\b|\bfund\b|\betf\b|convertible|"
                      r"subordinated|capital securities|closed[- ]end|limited partnership|\bl\.?p\.?\b", re.I)

def build_universe(us_dir: Path) -> pd.DataFrame:
    nq = pd.read_csv(us_dir / "nasdaqlisted.txt", sep="|")
    nq = nq[nq["Symbol"].notna() & ~nq["Symbol"].astype(str).str.startswith("File Creation")]
    nq = nq[(nq["ETF"] == "N") & (nq["Test Issue"] == "N")]
    nq = pd.DataFrame({"symbol": nq["Symbol"], "name": nq["Security Name"], "exchange": "Q"})
    ot = pd.read_csv(us_dir / "otherlisted.txt", sep="|")
    ot = ot[ot["ACT Symbol"].notna() & ~ot["ACT Symbol"].astype(str).str.startswith("File Creation")]
    ot = ot[(ot["ETF"] == "N") & (ot["Test Issue"] == "N") & ot["Exchange"].isin(["N", "A"])]
    ot = pd.DataFrame({"symbol": ot["ACT Symbol"], "name": ot["Security Name"], "exchange": ot["Exchange"]})
    u = pd.concat([nq, ot], ignore_index=True)
    u = u[u["symbol"].str.fullmatch(r"[A-Z]{1,5}")]                 # no $ . ^ suffix classes
    u = u[~u["symbol"].str.fullmatch(r"[A-Z]{4}[WRUP]")]             # 5-letter warrant/right/unit/pref suffixes
    u = u[~u["name"].str.contains(BAD_NAME, na=False)]
    u = u[u["name"].str.contains(r"common|ordinary|class [abc]|shares", case=False, na=False) |
          ~u["name"].str.contains(r"[A-Z]", na=False)]
    return u.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)

def fetch_batch(symbols, start, end) -> dict:
    import yfinance as yf
    raw = yf.download(list(symbols), start=start, end=end, auto_adjust=False,
                      progress=False, threads=False, group_by="ticker")
    out = {}
    for s in symbols:
        try:
            sub = raw[s] if len(symbols) > 1 else raw
        except KeyError:
            continue
        sub = sub.dropna(subset=["Close"])
        if sub.empty:
            continue
        out[s] = pd.DataFrame({"Date": pd.to_datetime(sub.index).tz_localize(None).normalize(),
                               "symbol": s,
                               "Close": sub["Close"].to_numpy(dtype=np.float64),
                               "Volume": sub["Volume"].to_numpy(dtype=np.float64)})
    return out

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2016-09-01")
    p.add_argument("--end", default="2022-07-01")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--pause", type=float, default=1.0)
    p.add_argument("--backoff", type=float, default=20.0)
    p.add_argument("--passes", type=int, default=MAX_ATTEMPTS)
    p.add_argument("--verify-pause", type=float, default=3.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="cache/us")
    a = p.parse_args()
    out = Path(a.out); per = out / "by_symbol"; per.mkdir(parents=True, exist_ok=True)
    uni = build_universe(out); uni.to_csv(out / "universe_candidates.csv", index=False)
    syms = uni["symbol"].tolist()[: a.limit] if a.limit else uni["symbol"].tolist()
    print(f"candidate symbols: {len(syms)}", flush=True)
    mp = out / "manifest.csv"
    if mp.exists():
        man = pd.read_csv(mp)
        new = [s for s in syms if s not in set(man["symbol"])]
        if new: man = pd.concat([man, pd.DataFrame({"symbol": new, "status": "pending", "attempts": 0, "rows": 0})], ignore_index=True)
    else:
        man = pd.DataFrame({"symbol": syms, "status": "pending", "attempts": 0, "rows": 0})
    for pass_no in range(a.passes):
        todo = man.loc[(man["status"] == "pending") & (man["attempts"] <= pass_no), "symbol"].tolist()
        if not todo: break
        print(f"\n=== pass {pass_no+1}: {len(todo)} outstanding ===", flush=True)
        t0 = time.perf_counter(); batches = [todo[i:i+a.batch_size] for i in range(0, len(todo), a.batch_size)]
        for bi, batch in enumerate(batches):
            try: got = fetch_batch(batch, a.start, a.end)
            except Exception as exc: print(f"  batch {bi} error: {type(exc).__name__}: {exc}", flush=True); got = {}
            for s in batch:
                row = man["symbol"] == s; man.loc[row, "attempts"] += 1
                if s in got:
                    got[s].to_parquet(per / f"{s}.parquet", index=False); man.loc[row, ["status", "rows"]] = ["ok", len(got[s])]
                elif int(man.loc[row, "attempts"].iloc[0]) >= a.passes:
                    man.loc[row, "status"] = "missing"
            time.sleep(a.backoff if not got else a.pause)
            if bi % 20 == 0:
                man.to_csv(mp, index=False); done = bi + 1; rate = (time.perf_counter()-t0)/done
                print(f"  batch {done}/{len(batches)} [{done}/{len(batches)}] ok={int((man.status=='ok').sum())}/{len(syms)} eta={rate*(len(batches)-done)/60:.1f} min", flush=True)
        man.to_csv(mp, index=False)
        print(f"  pass {pass_no+1}: ok={int((man.status=='ok').sum())} pending={int((man.status=='pending').sum())}", flush=True)
    outstanding = man.loc[man["status"] != "ok", "symbol"].tolist()
    if outstanding:
        print(f"\n=== verification: {len(outstanding)} symbols one at a time ===", flush=True)
        for s in outstanding:
            try: got = fetch_batch([s], a.start, a.end)
            except Exception: got = {}
            row = man["symbol"] == s; man.loc[row, "attempts"] += 1
            if s in got:
                got[s].to_parquet(per / f"{s}.parquet", index=False); man.loc[row, ["status", "rows"]] = ["ok", len(got[s])]
            else: man.loc[row, "status"] = "missing"
            time.sleep(a.verify_pause)
        man.to_csv(mp, index=False)
    print(f"\nfinal: ok={int((man.status=='ok').sum())} missing={int((man.status=='missing').sum())}", flush=True)

if __name__ == "__main__":
    main()

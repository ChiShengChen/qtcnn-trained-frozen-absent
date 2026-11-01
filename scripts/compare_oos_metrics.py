from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/media/meow/Transcend/kaggle_jpx_stock")
OUT_LGBM = ROOT / "outputs/lgbm_ranker"
OUT_TRF = ROOT / "outputs/transformer"
OUT_LSTM = ROOT / "outputs/lstm"
OUT_MLP = ROOT / "outputs/mlp"
OUT_PQC = ROOT / "outputs/pqc_lgbm"
OUT_QLSTM = ROOT / "outputs/qlstm"
OUT_QSVM = ROOT / "outputs/qsvm"
OUT_DIR = ROOT / "outputs/compare"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_summary(summary_path: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = None
    end = None
    if not summary_path.exists():
        return (pd.NaT, pd.NaT)
    txt = summary_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in txt:
        if "OOS Start:" in line:
            try:
                start = pd.to_datetime(line.split("OOS Start:")[-1].strip())
            except Exception:
                start = pd.NaT
        if "OOS End:" in line:
            try:
                end = pd.to_datetime(line.split("OOS End:")[-1].strip())
            except Exception:
                end = pd.NaT
    return start, end


def daily_ic(df: pd.DataFrame) -> pd.DataFrame:
    # df with columns: Date, SecuritiesCode, score, Target
    def _ic(g: pd.DataFrame) -> float:
        g = g.dropna(subset=["score", "Target"]).copy()
        if g.shape[0] < 3:
            return np.nan
        return g[["score", "Target"]].corr(method="spearman").iloc[0, 1]
    ics = df.groupby("Date").apply(_ic).rename("ic").reset_index()
    return ics


def sharpe_of_monthly(daily_returns: pd.Series) -> Tuple[float, pd.DataFrame]:
    # daily_returns indexed by Date
    mret = (
        (1.0 + daily_returns).groupby([daily_returns.index.year, daily_returns.index.month]).prod() - 1.0
    )
    mret.index = pd.to_datetime([f"{y:04d}-{m:02d}-01" for y, m in mret.index])
    mu = mret.mean()
    sd = mret.std(ddof=1)
    sharpe = float(mu / sd) if sd > 0 else float("nan")
    return sharpe, mret.to_frame("monthly_return")


def main() -> None:
    ensure_dir(OUT_DIR)

    # Load summaries to align OOS
    lgbm_start, lgbm_end = parse_summary(OUT_LGBM / "summary.txt")
    trf_start, trf_end = parse_summary(OUT_TRF / "summary.txt")
    lstm_start, lstm_end = parse_summary(OUT_LSTM / "summary.txt")
    mlp_start, mlp_end = parse_summary(OUT_MLP / "summary.txt")
    pqc_start, pqc_end = parse_summary(OUT_PQC / "summary.txt")
    qlstm_start, qlstm_end = parse_summary(OUT_QLSTM / "summary.txt")
    qsvm_start, qsvm_end = parse_summary(OUT_QSVM / "summary.txt")
    # Align across all models
    aligned_start = max(lgbm_start, trf_start, lstm_start, mlp_start, pqc_start, qlstm_start, qsvm_start)
    aligned_end = min(lgbm_end, trf_end, lstm_end, mlp_end, pqc_end, qlstm_end, qsvm_end)

    # Load scored universes
    lgbm_scored = pd.read_csv(OUT_LGBM / "scored.csv", parse_dates=["Date"])  # Date, SecuritiesCode, score, Target
    trf_scored = pd.read_csv(OUT_TRF / "scored.csv", parse_dates=["Date"])    # Date, SecuritiesCode, score, Target
    lstm_scored = pd.read_csv(OUT_LSTM / "scored.csv", parse_dates=["Date"])  # Date, SecuritiesCode, score, Target
    mlp_scored = pd.read_csv(OUT_MLP / "scored.csv", parse_dates=["Date"])    # Date, SecuritiesCode, score, Target
    pqc_scored = pd.read_csv(OUT_PQC / "scored.csv", parse_dates=["Date"])    # Date, SecuritiesCode, score, Target
    qlstm_scored = pd.read_csv(OUT_QLSTM / "scored.csv", parse_dates=["Date"])  # Date, SecuritiesCode, score, Target
    qsvm_scored = pd.read_csv(OUT_QSVM / "scored.csv", parse_dates=["Date"])  # Date, SecuritiesCode, score, Target
    # Align OOS range
    lgbm_scored = lgbm_scored[(lgbm_scored["Date"] >= aligned_start) & (lgbm_scored["Date"] <= aligned_end)]
    trf_scored = trf_scored[(trf_scored["Date"] >= aligned_start) & (trf_scored["Date"] <= aligned_end)]
    lstm_scored = lstm_scored[(lstm_scored["Date"] >= aligned_start) & (lstm_scored["Date"] <= aligned_end)]
    mlp_scored = mlp_scored[(mlp_scored["Date"] >= aligned_start) & (mlp_scored["Date"] <= aligned_end)]
    pqc_scored = pqc_scored[(pqc_scored["Date"] >= aligned_start) & (pqc_scored["Date"] <= aligned_end)]
    qlstm_scored = qlstm_scored[(qlstm_scored["Date"] >= aligned_start) & (qlstm_scored["Date"] <= aligned_end)]
    qsvm_scored = qsvm_scored[(qsvm_scored["Date"] >= aligned_start) & (qsvm_scored["Date"] <= aligned_end)]

    # Compute daily IC and ICIR
    lgbm_ic = daily_ic(lgbm_scored)
    trf_ic = daily_ic(trf_scored)
    def ic_stats(ics: pd.DataFrame) -> Dict[str, float]:
        x = ics["ic"].dropna()
        return {
            "ic_mean": float(x.mean()),
            "ic_std": float(x.std(ddof=1)),
            "icir": float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else float("nan"),
            "days": int(x.size),
        }
    lgbm_ic_stats = ic_stats(lgbm_ic)
    trf_ic_stats = ic_stats(trf_ic)
    lstm_ic = daily_ic(lstm_scored)
    lstm_ic_stats = ic_stats(lstm_ic)
    mlp_ic = daily_ic(mlp_scored)
    mlp_ic_stats = ic_stats(mlp_ic)
    pqc_ic = daily_ic(pqc_scored)
    pqc_ic_stats = ic_stats(pqc_ic)
    qlstm_ic = daily_ic(qlstm_scored)
    qlstm_ic_stats = ic_stats(qlstm_ic)
    qsvm_ic = daily_ic(qsvm_scored)
    qsvm_ic_stats = ic_stats(qsvm_ic)

    # Load daily long-short returns and align
    lgbm_ls = pd.read_csv(OUT_LGBM / "daily_long_short_returns.csv", parse_dates=["Date"]).set_index("Date")
    trf_ls = pd.read_csv(OUT_TRF / "daily_long_short_returns.csv", parse_dates=["Date"]).set_index("Date")
    lstm_ls = pd.read_csv(OUT_LSTM / "daily_long_short_returns.csv", parse_dates=["Date"]).set_index("Date")
    mlp_ls = pd.read_csv(OUT_MLP / "daily_long_short_returns.csv", parse_dates=["Date"]).set_index("Date")
    pqc_ls = pd.read_csv(OUT_PQC / "daily_long_short_returns.csv", parse_dates=["Date"]).set_index("Date")
    qlstm_ls = pd.read_csv(OUT_QLSTM / "daily_long_short_returns.csv", parse_dates=["Date"]).set_index("Date")
    qsvm_ls = pd.read_csv(OUT_QSVM / "daily_long_short_returns.csv", parse_dates=["Date"]).set_index("Date")
    lgbm_ls = lgbm_ls[(lgbm_ls.index >= aligned_start) & (lgbm_ls.index <= aligned_end)]
    trf_ls = trf_ls[(trf_ls.index >= aligned_start) & (trf_ls.index <= aligned_end)]
    lstm_ls = lstm_ls[(lstm_ls.index >= aligned_start) & (lstm_ls.index <= aligned_end)]
    mlp_ls = mlp_ls[(mlp_ls.index >= aligned_start) & (mlp_ls.index <= aligned_end)]
    pqc_ls = pqc_ls[(pqc_ls.index >= aligned_start) & (pqc_ls.index <= aligned_end)]
    qlstm_ls = qlstm_ls[(qlstm_ls.index >= aligned_start) & (qlstm_ls.index <= aligned_end)]
    qsvm_ls = qsvm_ls[(qsvm_ls.index >= aligned_start) & (qsvm_ls.index <= aligned_end)]

    # Monthly Sharpe from monthly compounded returns
    lgbm_m_sharpe, lgbm_mrets = sharpe_of_monthly(lgbm_ls["ls_return"].dropna())
    trf_m_sharpe, trf_mrets = sharpe_of_monthly(trf_ls["ls_return"].dropna())
    lstm_m_sharpe, lstm_mrets = sharpe_of_monthly(lstm_ls["ls_return"].dropna())
    mlp_m_sharpe, mlp_mrets = sharpe_of_monthly(mlp_ls["ls_return"].dropna())
    pqc_m_sharpe, pqc_mrets = sharpe_of_monthly(pqc_ls["ls_return"].dropna())
    qlstm_m_sharpe, qlstm_mrets = sharpe_of_monthly(qlstm_ls["ls_return"].dropna())
    qsvm_m_sharpe, qsvm_mrets = sharpe_of_monthly(qsvm_ls["ls_return"].dropna())

    # Write outputs
    lgbm_ic.to_csv(OUT_DIR / "lgbm_daily_ic.csv", index=False)
    trf_ic.to_csv(OUT_DIR / "transformer_daily_ic.csv", index=False)
    lgbm_mrets.to_csv(OUT_DIR / "lgbm_monthly_returns.csv")
    trf_mrets.to_csv(OUT_DIR / "transformer_monthly_returns.csv")
    lstm_mrets.to_csv(OUT_DIR / "lstm_monthly_returns.csv")
    mlp_mrets.to_csv(OUT_DIR / "mlp_monthly_returns.csv")
    pqc_mrets.to_csv(OUT_DIR / "pqc_monthly_returns.csv")
    qlstm_mrets.to_csv(OUT_DIR / "qlstm_monthly_returns.csv")
    qsvm_mrets.to_csv(OUT_DIR / "qsvm_monthly_returns.csv")

    # Report
    report = OUT_DIR / "report.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"Aligned OOS: {aligned_start.date()} to {aligned_end.date()}\n")
        f.write("\nIC/ICIR\n")
        f.write(f"LGBM: IC_mean={lgbm_ic_stats['ic_mean']:.6f}, ICIR={lgbm_ic_stats['icir']:.3f}, days={lgbm_ic_stats['days']}\n")
        f.write(f"Transformer: IC_mean={trf_ic_stats['ic_mean']:.6f}, ICIR={trf_ic_stats['icir']:.3f}, days={trf_ic_stats['days']}\n")
        f.write(f"LSTM: IC_mean={lstm_ic_stats['ic_mean']:.6f}, ICIR={lstm_ic_stats['icir']:.3f}, days={lstm_ic_stats['days']}\n")
        f.write(f"MLP: IC_mean={mlp_ic_stats['ic_mean']:.6f}, ICIR={mlp_ic_stats['icir']:.3f}, days={mlp_ic_stats['days']}\n")
        f.write(f"PQC: IC_mean={pqc_ic_stats['ic_mean']:.6f}, ICIR={pqc_ic_stats['icir']:.3f}, days={pqc_ic_stats['days']}\n")
        f.write(f"QLSTM: IC_mean={qlstm_ic_stats['ic_mean']:.6f}, ICIR={qlstm_ic_stats['icir']:.3f}, days={qlstm_ic_stats['days']}\n")
        f.write(f"QSVM: IC_mean={qsvm_ic_stats['ic_mean']:.6f}, ICIR={qsvm_ic_stats['icir']:.3f}, days={qsvm_ic_stats['days']}\n")
        f.write("\nMonthly Sharpe (from monthly compounded returns)\n")
        f.write(f"LGBM monthly Sharpe: {lgbm_m_sharpe:.3f}\n")
        f.write(f"Transformer monthly Sharpe: {trf_m_sharpe:.3f}\n")
        f.write(f"LSTM monthly Sharpe: {lstm_m_sharpe:.3f}\n")
        f.write(f"MLP monthly Sharpe: {mlp_m_sharpe:.3f}\n")
        f.write(f"PQC monthly Sharpe: {pqc_m_sharpe:.3f}\n")
        f.write(f"QLSTM monthly Sharpe: {qlstm_m_sharpe:.3f}\n")
        f.write(f"QSVM monthly Sharpe: {qsvm_m_sharpe:.3f}\n")

    # Save IC series
    lstm_ic.to_csv(OUT_DIR / "lstm_daily_ic.csv", index=False)
    mlp_ic.to_csv(OUT_DIR / "mlp_daily_ic.csv", index=False)
    pqc_ic.to_csv(OUT_DIR / "pqc_daily_ic.csv", index=False)
    qlstm_ic.to_csv(OUT_DIR / "qlstm_daily_ic.csv", index=False)
    qsvm_ic.to_csv(OUT_DIR / "qsvm_daily_ic.csv", index=False)

    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()



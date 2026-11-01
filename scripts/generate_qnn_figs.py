from __future__ import annotations

from pathlib import Path
import argparse
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_daily_ls(input_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(input_dir / "daily_long_short_returns.csv", parse_dates=["Date"])
    return df.sort_values("Date")


def read_selection(input_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(input_dir / "daily_selection.csv", parse_dates=["Date"])
    return df.sort_values(["Date", "SecuritiesCode"])


def read_scored(input_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(input_dir / "scored.csv", parse_dates=["Date"])
    return df.sort_values(["Date", "SecuritiesCode"])


def plot_cum_ls_curve(daily_ls: pd.DataFrame, out_path: Path) -> None:
    df = daily_ls.copy()
    df["cum_ls"] = df["ls_return"].fillna(0).cumsum()
    plt.figure(figsize=(12, 4))
    plt.plot(df["Date"], df["cum_ls"], label="Cumulative long-short")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Cumulative long-short return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_daily_ls_bar(daily_ls: pd.DataFrame, out_path: Path) -> None:
    df = daily_ls.copy()
    plt.figure(figsize=(12, 4))
    colors = np.where(df["ls_return"] >= 0, "#2ca02c", "#d62728")
    plt.bar(df["Date"], df["ls_return"], color=colors, width=3)  # width in days
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Daily long-short returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_long_short_means(daily_ls: pd.DataFrame, out_path: Path) -> None:
    df = daily_ls.copy()
    plt.figure(figsize=(12, 4))
    plt.plot(df["Date"], df["long_mean"], label="Long leg mean")
    plt.plot(df["Date"], df["short_mean"], label="Short leg mean")
    plt.title("Daily mean returns for long and short legs")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_selection_counts(selection: pd.DataFrame, out_path: Path) -> None:
    counts = selection.groupby(["Date", "Side"]).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 4))
    if "long" in counts.columns:
        plt.plot(counts.index, counts["long"], label="Long selection count")
    if "short" in counts.columns:
        plt.plot(counts.index, counts["short"], label="Short selection count")
    plt.title("Daily selection counts by side")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_turnover_per_side(selection: pd.DataFrame) -> pd.DataFrame:
    df = selection[["Date", "SecuritiesCode", "Side"]].copy()
    df = df.sort_values(["Date", "SecuritiesCode"])
    dates = sorted(df["Date"].unique())

    def turnover(side: str) -> List[float]:
        prev: Set[int] = set()
        values: List[float] = []
        for d in dates:
            cur = set(df[(df["Date"] == d) & (df["Side"] == side)]["SecuritiesCode"].astype(int))
            if len(prev) == 0:
                values.append(np.nan)
            else:
                inter = len(prev & cur)
                values.append(1.0 - (inter / max(1, len(prev))))
            prev = cur
        return values

    out = pd.DataFrame({"Date": dates})
    out["turnover_long"] = turnover("long")
    out["turnover_short"] = turnover("short")
    return out


def plot_turnover(selection: pd.DataFrame, out_path: Path) -> None:
    t = compute_turnover_per_side(selection)
    plt.figure(figsize=(12, 4))
    plt.plot(t["Date"], t["turnover_long"], label="Turnover (long)")
    plt.plot(t["Date"], t["turnover_short"], label="Turnover (short)")
    plt.title("Portfolio turnover over time")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_score_hist(scored: pd.DataFrame, out_path: Path) -> None:
    s = scored["score"].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(8, 4))
    plt.hist(s, bins=60, color="#1f77b4", edgecolor="white")
    plt.title("Distribution of scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_daily_score_stats(scored: pd.DataFrame, out_path: Path) -> None:
    g = scored.groupby("Date")["score"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax1.plot(g["Date"], g["mean"], label="Daily mean", color="#1f77b4")
    ax1.set_ylabel("Mean", color="#1f77b4")
    ax1.set_xlabel("Date")
    ax2 = ax1.twinx()
    ax2.plot(g["Date"], g["std"], label="Daily std", color="#ff7f0e")
    ax2.set_ylabel("Std", color="#ff7f0e")
    plt.title("Daily score mean and std")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_target_by_score_decile(scored: pd.DataFrame, out_path: Path) -> None:
    df = scored[["Date", "score", "Target"]].dropna().copy()
    # Cross-sectional deciles per Date; drop days with insufficient samples
    def per_date_bins(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("score")
        if g.shape[0] < 10:
            g["decile"] = np.nan
            return g
        # qcut might fail if too many ties; fallback to rank-based bins
        try:
            g["decile"] = pd.qcut(g["score"], 10, labels=False, duplicates="drop")
        except Exception:
            ranks = g["score"].rank(method="first")
            g["decile"] = pd.qcut(ranks, 10, labels=False, duplicates="drop")
        return g

    df = df.groupby("Date", group_keys=False).apply(per_date_bins)
    df = df.dropna(subset=["decile"])
    df["decile"] = df["decile"].astype(int)
    agg = df.groupby("decile")["Target"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.bar(agg["decile"], agg["Target"], color="#2ca02c")
    plt.xticks(agg["decile"])
    plt.title("Mean Target by score decile (cross-sectional per day)")
    plt.xlabel("Score decile (0=lowest)")
    plt.ylabel("Mean Target")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_target_by_decile_supervision(scored: pd.DataFrame, out_path: Path) -> None:
    df = scored[["score", "Target", "SupervisionFlag"]].dropna().copy()
    # Global deciles (simpler and robust across flags)
    try:
        df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")
    except Exception:
        ranks = df["score"].rank(method="first")
        df["decile"] = pd.qcut(ranks, 10, labels=False, duplicates="drop")
    df = df.dropna(subset=["decile"])
    df["decile"] = df["decile"].astype(int)
    agg = df.groupby(["decile", "SupervisionFlag"])["Target"].mean().reset_index()
    piv = agg.pivot(index="decile", columns="SupervisionFlag", values="Target").fillna(0)
    plt.figure(figsize=(9, 4))
    x = np.arange(len(piv.index))
    w = 0.4
    plt.bar(x - w / 2, piv.get(False, pd.Series(index=piv.index, dtype=float)).values, width=w, label="SupervisionFlag=False")
    plt.bar(x + w / 2, piv.get(True, pd.Series(index=piv.index, dtype=float)).values, width=w, label="SupervisionFlag=True")
    plt.xticks(x, piv.index)
    plt.title("Mean Target by score decile and SupervisionFlag")
    plt.xlabel("Score decile (global)")
    plt.ylabel("Mean Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cum_ls_by_year(daily_ls: pd.DataFrame, out_path: Path) -> None:
    df = daily_ls.copy()
    df["Year"] = df["Date"].dt.year
    plt.figure(figsize=(12, 5))
    for y, g in df.groupby("Year"):
        g = g.sort_values("Date")
        curve = g["ls_return"].fillna(0).cumsum()
        plt.plot(g["Date"], curve, label=str(y))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Cumulative long-short by year (re-based within year)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="/media/meow/Transcend/kaggle_jpx_stock/outputs/qnn_50epoch_stride11")
    p.add_argument("--out-dir", type=str, default=None, help="Optional override for figure output directory")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else input_dir / "figs"
    ensure_dir(out_dir)

    daily_ls = read_daily_ls(input_dir)
    selection = read_selection(input_dir)
    scored = read_scored(input_dir)

    plot_cum_ls_curve(daily_ls, out_dir / "cum_ls_curve.png")
    plot_daily_ls_bar(daily_ls, out_dir / "daily_ls_bar.png")
    plot_long_short_means(daily_ls, out_dir / "long_short_means.png")
    plot_selection_counts(selection, out_dir / "selection_counts.png")
    plot_turnover(selection, out_dir / "turnover.png")
    plot_score_hist(scored, out_dir / "score_hist.png")
    plot_daily_score_stats(scored, out_dir / "daily_score_stats.png")
    plot_target_by_score_decile(scored, out_dir / "target_by_score_decile.png")
    plot_target_by_decile_supervision(scored, out_dir / "target_by_decile_supervision.png")
    plot_cum_ls_by_year(daily_ls, out_dir / "cum_ls_by_year.png")

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()



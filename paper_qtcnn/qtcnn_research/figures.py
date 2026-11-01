"""Manuscript figures. Static PDF for LaTeX, so single light theme, no hover.

Colour follows the reference palette in the dataviz skill: categorical slots
are assigned in fixed order and never cycled, quantum-path conditions and
classical baselines are the only two identities that need distinguishing in
the headline figure, and every multi-series chart stays within the first three
slots, which validate all-pairs. Marks are thin, gridlines recessive, and text
wears ink tokens rather than series colour.
"""

from __future__ import annotations

from pathlib import Path

from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- reference palette (light surface) -------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3

LABELS = {
    "qtcnn_trained": "QTCNN, trained circuit",
    "qtcnn_frozen": "QTCNN, frozen circuit",
    "qtcnn_surrogate": "QTCNN, classical surrogate",
    "qtcnn_noquantum": "QTCNN, no circuit",
    "qtcnn_quantumonly": "QTCNN, circuit only",
    "tcn": "TCN baseline",
    "lstm": "LSTM baseline",
    "transformer": "Transformer baseline",
    "mlp": "MLP baseline",
    "lightgbm": "LightGBM baseline",
}
QUANTUM_PATH = {"qtcnn_trained", "qtcnn_frozen", "qtcnn_surrogate", "qtcnn_noquantum"}


def _style(ax, ylabel: str = "", xlabel: str = ""):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}.pdf", flush=True)


# ---------------------------------------------------------------------------
def fig_ablation(head: pd.DataFrame, out: Path):
    """Headline: Sharpe with 95% bootstrap CI, every condition on 500 shared days."""
    df = head.sort_values("sharpe_ann").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y = np.arange(len(df))
    for i, r in df.iterrows():
        colour = S1 if r["model"] in QUANTUM_PATH else S2
        ax.plot([r["sharpe_ci_low"], r["sharpe_ci_high"]], [i, i], color=colour, lw=2, solid_capstyle="round")
        ax.plot(r["sharpe_ann"], i, "o", color=colour, ms=6, mec=SURFACE, mew=1.2)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in df["model"]], fontsize=8.5, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.yaxis.grid(False)
    _style(ax, xlabel="Annualised Sharpe ratio, 500 walk-forward days (95% stationary-bootstrap CI)")
    ax.yaxis.grid(False)
    # legend for the two identities
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=S1, marker="o", lw=2, label="hybrid QTCNN conditions"),
                       Line2D([], [], color=S2, marker="o", lw=2, label="classical baselines")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              fontsize=8, frameon=False, labelcolor=INK2)
    _save(fig, out, "fig_ablation")


def fig_trained_vs_frozen(cells: pd.DataFrame, out: Path):
    """Per-cell Sharpe, trained against frozen: the identity line is the finding."""
    t = cells[cells["model"] == "qtcnn_trained"].set_index(["fold", "seed"])["sharpe_ann"]
    f = cells[cells["model"] == "qtcnn_frozen"].set_index(["fold", "seed"])["sharpe_ann"]
    joined = pd.concat([t.rename("trained"), f.rename("frozen")], axis=1).dropna()
    lim = float(np.abs(joined.values).max()) * 1.1
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=0.8, ls="--", zorder=0)
    ax.scatter(joined["frozen"], joined["trained"], s=28, color=S1, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    _style(ax, ylabel="Sharpe, trained circuit", xlabel="Sharpe, frozen circuit (same initialisation)")
    r = joined["trained"].corr(joined["frozen"])
    ax.text(0.04, 0.96, f"20 cells (4 folds × 5 seeds)\nr = {r:.3f}", transform=ax.transAxes,
            va="top", fontsize=8, color=INK2)
    _save(fig, out, "fig_trained_vs_frozen")


def fig_seed_spread(cells: pd.DataFrame, out: Path):
    """Seed draw versus architecture: one dot per seed, per fold, two conditions."""
    models = [("qtcnn_trained", S1), ("qtcnn_noquantum", S2)]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    folds = sorted(cells["fold"].unique())
    offs = {"qtcnn_trained": -0.12, "qtcnn_noquantum": 0.12}
    for m, c in models:
        sub = cells[cells["model"] == m]
        ax.scatter(sub["fold"] + offs[m], sub["sharpe_ann"], s=26, color=c,
                   edgecolor=SURFACE, linewidth=0.8, label=LABELS[m], zorder=3)
        means = sub.groupby("fold")["sharpe_ann"].mean()
        ax.hlines(means.values, means.index + offs[m] - 0.09, means.index + offs[m] + 0.09,
                  color=c, lw=2, zorder=4)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"fold {k}" for k in folds], fontsize=8.5, color=INK)
    _style(ax, ylabel="Annualised Sharpe, 125-day test block")
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK2)
    _save(fig, out, "fig_seed_spread")


def fig_cost(costs: pd.DataFrame, out: Path):
    """Net Sharpe against round-trip cost. Linear in cost at fixed turnover."""
    keep = ["qtcnn_trained", "qtcnn_noquantum", "tcn"]
    colours = dict(zip(keep, (S1, S2, S3)))
    grid = [0, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    head = pd.read_csv(out.parent.parent / "results/walkforward/headline.csv").set_index("model")
    for m in keep:
        row = costs.set_index("model").loc[m]
        ys = [head.loc[m, "sharpe_ann"], row["sharpe_ann_5bps"], row["sharpe_ann_10bps"], row["sharpe_ann_20bps"]]
        ax.plot(grid, ys, color=colours[m], lw=2, marker="o", ms=5, mec=SURFACE, mew=1)
        ax.annotate(LABELS[m], (grid[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
        ax.axvline(row["break_even_bps"], color=colours[m], lw=0.8, ls=":", alpha=0.7)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
    ax.set_xticks(grid)
    ax.set_xlim(-0.5, 27)
    _style(ax, ylabel="Annualised Sharpe, net of costs", xlabel="Round-trip cost per rebalance (bps); dotted = break-even")
    _save(fig, out, "fig_cost")


def fig_shots(shots: pd.DataFrame, out: Path, real: Optional[pd.DataFrame] = None):
    """Rank preservation against shot budget, at both input distributions.

    The random-angle curve is what a circuit-level scan measures; the real
    cross-section curve is what the deployed model faces, and the gap between
    them is the point of the figure.
    """
    df = shots[shots["shots"] != "analytic"].copy()
    df["shots"] = df["shots"].astype(int)
    df = df.sort_values("shots")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(df["shots"], df["rank_corr_vs_analytic"], color=S1, lw=2, marker="o", ms=6,
            mec=SURFACE, mew=1.2, label="random angles (sd 0.44)")
    xmax = df["shots"].max()
    if real is not None and not real.empty:
        r = real.groupby("shots")["rank_corr"].mean().sort_index()
        ax.plot(r.index, r.values, color=S2, lw=2, marker="s", ms=6, mec=SURFACE, mew=1.2,
                label="real cross-sections (sd 0.02)")
        xmax = max(xmax, int(r.index.max()))
    ax.axhline(0.99, color=AXIS, lw=0.8, ls="--")
    ax.text(xmax * 1.6, 0.99, "0.99", fontsize=7.5, color=MUTED, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xlim(df["shots"].min() / 1.5, xmax * 2.2)
    ax.set_ylim(0.2, 1.03)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    _style(ax, ylabel="Spearman rank corr. vs analytic", xlabel="Shots per circuit evaluation")
    _save(fig, out, "fig_shots")


def fig_barren(bp: pd.DataFrame, out: Path):
    df = bp.sort_values("n_qubits")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(df["n_qubits"], df["grad_var"], color=S1, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.2)
    ax.set_yscale("log")
    ax.set_xticks(df["n_qubits"])
    if "decay_rate_per_qubit" in df.columns and df["decay_rate_per_qubit"].notna().any():
        b = float(df["decay_rate_per_qubit"].iloc[0])
        ax.text(0.97, 0.95, f"fitted log-var slope = {b:+.3f} per qubit\n(exponential decay would be ≈ −0.69)",
                transform=ax.transAxes, fontsize=8, color=INK2, ha="right", va="top")
    _style(ax, ylabel="Var[∂⟨Z⟩/∂θ] at random init (log)", xlabel="Qubits (3 conv layers, 6 shared params each)")
    _save(fig, out, "fig_barren")


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "manuscript" / "figures"
    wf = root / "results" / "walkforward"
    head = pd.read_csv(wf / "headline.csv")
    cells = pd.concat([pd.read_csv(f) for f in sorted(wf.glob("fold_metrics_*.csv"))], ignore_index=True)
    costs = pd.read_csv(wf / "cost_table.csv")
    diag = root / "results" / "diagnostics"
    print("figures ->", out)
    fig_ablation(head, out)
    fig_trained_vs_frozen(cells, out)
    fig_seed_spread(cells, out)
    fig_cost(costs, out)
    real_lo = diag / "shot_budget_real.csv"
    real_hi = diag / "shot_budget_real_high.csv"
    real = pd.concat([pd.read_csv(f) for f in (real_lo, real_hi) if f.exists()], ignore_index=True) \
        if (real_lo.exists() or real_hi.exists()) else None
    fig_shots(pd.read_csv(diag / "shot_noise.csv"), out, real=real)
    fig_barren(pd.read_csv(diag / "barren_plateau.csv"), out)


if __name__ == "__main__":
    main()

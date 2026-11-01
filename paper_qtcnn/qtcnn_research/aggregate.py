"""Turn per-fold, per-seed results into the tables the manuscript reports.

Two levels of aggregation matter and are easy to conflate:

*Across seeds* the spread measures optimisation noise for a fixed data split,
so seeds are averaged within a fold first.

*Across folds* the walk-forward test blocks are disjoint, so their return series
concatenate into one continuous out-of-sample track. Statistics are computed on
that pooled track rather than by averaging per-fold Sharpe ratios, which would
overweight short folds and understate the variance.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from qtcnn_research import evaluation as ev


def load_returns(returns_dir: Path) -> pd.DataFrame:
    """Read every ``<model>_f<fold>_s<seed>.csv`` into a tidy frame."""
    pattern = re.compile(r"^(?P<model>.+)_f(?P<fold>\d+)_s(?P<seed>\d+)$")
    rows = []
    for path in sorted(Path(returns_dir).glob("*.csv")):
        match = pattern.match(path.stem)
        if match is None:
            continue
        series = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")["ls_return"]
        rows.append(pd.DataFrame({
            "model": match["model"],
            "fold": int(match["fold"]),
            "seed": int(match["seed"]),
            "Date": series.index,
            "ls_return": series.values,
        }))
    if not rows:
        return pd.DataFrame(columns=["model", "fold", "seed", "Date", "ls_return"])
    return pd.concat(rows, ignore_index=True)


def pooled_track(returns: pd.DataFrame, model: str) -> pd.Series:
    """Seed-averaged daily returns concatenated across folds.

    Averaging across seeds on a given date is the return of an equal-weighted
    ensemble of the seeds, which is the quantity a practitioner would actually
    realise, and it removes seed noise without pretending the folds overlap.
    """
    sub = returns[returns["model"] == model]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("Date")["ls_return"].mean().sort_index()


def per_seed_tracks(returns: pd.DataFrame, model: str) -> Dict[int, pd.Series]:
    sub = returns[returns["model"] == model]
    return {
        int(seed): g.groupby("Date")["ls_return"].mean().sort_index()
        for seed, g in sub.groupby("seed")
    }


def headline_table(returns: pd.DataFrame, n_trials: Optional[int] = None,
                   bootstrap: int = 2000) -> pd.DataFrame:
    """Main results table: one row per model over the pooled OOS track."""
    models = sorted(returns["model"].unique())
    n_trials = n_trials or len(models)
    rows = []
    for model in models:
        track = pooled_track(returns, model)
        seeds = per_seed_tracks(returns, model)
        seed_sharpes = [ev.annualised_sharpe(s) for s in seeds.values()]
        report = ev.evaluate_strategy(model, track, n_trials=n_trials, bootstrap=bootstrap)
        row = report.to_row()
        row["sharpe_seed_mean"] = float(np.nanmean(seed_sharpes)) if seed_sharpes else np.nan
        row["sharpe_seed_std"] = float(np.nanstd(seed_sharpes, ddof=1)) if len(seed_sharpes) > 1 else np.nan
        row["n_seeds"] = len(seed_sharpes)
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_contrasts(returns: pd.DataFrame, reference: str = "qtcnn_noquantum",
                       lags: Optional[int] = None) -> pd.DataFrame:
    """Diebold-Mariano contrasts of every model against the no-quantum control.

    This is the test that answers the paper's title question: if the trained
    circuit adds nothing over the same network without it, the difference series
    has a mean indistinguishable from zero.
    """
    ref = pooled_track(returns, reference)
    if ref.empty:
        return pd.DataFrame()
    rows = []
    for model in sorted(returns["model"].unique()):
        if model == reference:
            continue
        track = pooled_track(returns, model)
        aligned = pd.concat([track.rename("a"), ref.rename("b")], axis=1).dropna()
        if aligned.empty:
            continue
        stat = ev.diebold_mariano(aligned["a"], aligned["b"], lags=lags)
        stat.update({"model": model, "reference": reference,
                     "sharpe_model": ev.annualised_sharpe(aligned["a"]),
                     "sharpe_reference": ev.annualised_sharpe(aligned["b"])})
        rows.append(stat)
    cols = ["model", "reference", "sharpe_model", "sharpe_reference",
            "mean_diff", "se_nw", "dm_stat", "p_value", "n"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def sharpe_contrasts(returns: pd.DataFrame, reference: str = "qtcnn_noquantum",
                     n_boot: int = 2000) -> pd.DataFrame:
    """Paired bootstrap test of the Sharpe difference against the control.

    The Diebold-Mariano contrasts test a difference in mean return. The study
    ranks by Sharpe ratio, so the corresponding hypothesis is tested here.
    """
    ref = pooled_track(returns, reference)
    rows = []
    for m in sorted(returns["model"].unique()):
        if m == reference:
            continue
        track = pooled_track(returns, m)
        common = track.index.intersection(ref.index)
        out = ev.sharpe_difference_test(track.reindex(common), ref.reindex(common), n_boot=n_boot)
        rows.append(dict(model=m, reference=reference, **out))
    return pd.DataFrame(rows)


def multiple_comparison(returns: pd.DataFrame, reference: str = "qtcnn_noquantum",
                        n_boot: int = 1000) -> pd.DataFrame:
    """Romano-Wolf stepdown over all models against the control.

    The family is the set of contrasts against the no-circuit control on one
    panel: seven hypotheses. It does not cover the per-model tests against zero,
    nor does it pool the two markets or the holdout, and the manuscript states
    the family explicitly rather than claiming family-wise control over
    everything reported.
    """
    ref = pooled_track(returns, reference)
    candidates = {
        m: pooled_track(returns, m)
        for m in sorted(returns["model"].unique())
        if m != reference
    }
    if ref.empty or not candidates:
        return pd.DataFrame()
    frame = pd.DataFrame(candidates)
    aligned = frame.join(ref.rename("__ref__"), how="inner").dropna()
    return ev.romano_wolf_stepdown(
        {c: aligned[c] for c in frame.columns}, aligned["__ref__"], n_boot=n_boot
    )


def overfitting_table(returns: pd.DataFrame, n_splits: int = 10) -> pd.DataFrame:
    """CSCV probability of backtest overfitting across the model set."""
    tracks = {m: pooled_track(returns, m) for m in sorted(returns["model"].unique())}
    matrix = pd.DataFrame(tracks).dropna()
    result = ev.probability_of_backtest_overfitting(matrix, n_splits=n_splits)
    return pd.DataFrame([result])


def cost_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean turnover and net Sharpe at each cost level, plus break-even.

    ``break_even_bps`` is NaN for any cell whose gross mean return is not
    positive, so a plain mean over cells silently conditions on the cell having
    been profitable and is not comparable across models. It is reported here
    alongside the fraction of cells that defined it, and ``break_even_cells``
    makes that conditioning visible; the pooled-track break-even, which is the
    quantity the manuscript quotes, is computed in :func:`pooled_cost_table`.
    """
    cols = [c for c in fold_metrics.columns
            if c in ("turnover_mean", "break_even_bps") or c.startswith("sharpe_ann_")]
    if not cols:
        return pd.DataFrame()
    out = fold_metrics.groupby("model")[cols].mean()
    if "break_even_bps" in fold_metrics.columns:
        defined = fold_metrics.groupby("model")["break_even_bps"].apply(
            lambda s: f"{int(s.notna().sum())}/{len(s)}")
        out["break_even_cells"] = defined
    return out.reset_index()


def pooled_cost_table(returns: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Break-even cost of the seed-averaged pooled track, per model.

    NaN where the pooled gross mean return is not positive, which is the honest
    answer: a strategy that loses money before costs has no break-even cost.
    """
    rows = []
    for model in sorted(returns["model"].unique()):
        r = pooled_track(returns, model)
        to = fold_metrics.loc[fold_metrics["model"] == model, "turnover_mean"].mean()
        mean_ret = float(r.mean())
        rows.append(dict(model=model, n_days=len(r), gross_mean_daily=mean_ret,
                         turnover_mean=float(to),
                         break_even_bps=(float(mean_ret / (2.0 * to) * 10_000.0)
                                         if to and to > 0 and mean_ret > 0 else float("nan"))))
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


PAPER_LABELS = {
    "qtcnn_trained": r"QTCNN, trained circuit",
    "qtcnn_frozen": r"QTCNN, frozen circuit",
    "qtcnn_surrogate": r"QTCNN, classical surrogate",
    "qtcnn_noquantum": r"QTCNN, no circuit",
    "qtcnn_quantumonly": r"QTCNN, circuit only",
    "tcn": "TCN baseline",
    "lstm": "LSTM baseline",
    "transformer": "Transformer baseline",
    "mlp": "MLP baseline",
    "lightgbm": "LightGBM baseline",
}


def paper_headline_table(head: pd.DataFrame, path: Path, n_days: int, n_trials: int,
                         market: str = "", label: str = "tab:headline") -> None:
    """Publication layout of the headline table: readable headers, sorted."""
    df = head.sort_values("sharpe_ann", ascending=False)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Out-of-sample performance on the pooled walk-forward track"
         + (f" for {market}" if market else "") + " "
         f"(\\num{{{n_days}}} trading days, five seeds per condition). Confidence "
         r"intervals are 95\% stationary bootstrap; DSR is the Deflated Sharpe Ratio "
         f"deflated by the \\num{{{n_trials}}} configurations evaluated.}}"),
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & Sharpe & 95\% CI & $p$ & DSR & Seed sd & Max DD \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        name = PAPER_LABELS.get(r["model"], r["model"])
        lines.append(
            f"{name} & {r['sharpe_ann']:+.3f} & "
            f"[{r['sharpe_ci_low']:+.2f}, {r['sharpe_ci_high']:+.2f}] & "
            f"{r['sharpe_p_value']:.2f} & {r['dsr_dsr']:.2f} & "
            f"{r['sharpe_seed_std']:.2f} & {r['max_drawdown']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str,
             float_fmt: str = "%.3f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(index=False, float_format=float_fmt, escape=True,
                       caption=caption, label=label, position="t")
    path.write_text(body)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/walkforward")
    p.add_argument("--tables", default="manuscript/tables")
    p.add_argument("--n-trials", type=int, default=None,
                   help="configurations to deflate the Sharpe ratio by; "
                        "default is models x seeds, the size of the search actually run")
    p.add_argument("--market", default="", help="market name for table captions")
    p.add_argument("--label-suffix", default="", help="suffix for LaTeX table labels")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()

    res = Path(args.results)
    returns = load_returns(res / "returns")
    if returns.empty:
        print(f"no return files under {res/'returns'}")
        return

    metric_files = sorted(res.glob("fold_metrics*.csv"))
    fold_metrics = pd.concat([pd.read_csv(f) for f in metric_files], ignore_index=True) \
        if metric_files else pd.DataFrame()

    pd.set_option("display.width", 240, "display.max_columns", 60)

    # Rows computed over different date ranges are not comparable, which is easy
    # to miss while a sweep is still filling in. Say so rather than print a table
    # that invites a cross-row reading it cannot support.
    spans = returns.groupby("model")["Date"].agg(["min", "max", "count"])
    if spans["count"].nunique() > 1:
        print("WARNING: models cover different windows; cross-row Sharpe comparison "
              "is not valid until every model has the same folds.")
        print(spans.to_string(), "\n")

    n_models = returns["model"].nunique()
    n_seeds = returns["seed"].nunique() if "seed" in returns.columns else 1
    n_trials = args.n_trials or n_models * n_seeds
    print(f"deflating the Sharpe ratio by {n_trials} configurations "
          f"({n_models} models x {n_seeds} seeds)\n")
    head = headline_table(returns, n_trials=n_trials, bootstrap=args.bootstrap)
    show = ["model", "n_days", "n_seeds", "sharpe_ann", "sharpe_seed_mean", "sharpe_seed_std",
            "sharpe_ci_low", "sharpe_ci_high", "sharpe_p_value", "dsr_dsr", "max_drawdown"]
    print("\n=== headline (pooled OOS track) ===")
    print(head[[c for c in show if c in head.columns]].round(4).to_string(index=False))

    contrasts = ablation_contrasts(returns)
    if not contrasts.empty:
        print("\n=== ablation contrasts vs qtcnn_noquantum (Diebold-Mariano) ===")
        print(contrasts.round(4).to_string(index=False))

    rw = multiple_comparison(returns)
    if not rw.empty:
        print("\n=== Romano-Wolf stepdown (familywise corrected) ===")
        print(rw.round(4).to_string(index=False))

    pbo = overfitting_table(returns)
    print("\n=== backtest overfitting (CSCV) ===")
    print(pbo.round(4).to_string(index=False))

    if not fold_metrics.empty:
        costs = cost_table(fold_metrics)
        if not costs.empty:
            print("\n=== turnover and net-of-cost Sharpe ===")
            print(costs.round(4).to_string(index=False))
            costs.to_csv(res / "cost_table.csv", index=False)
        pooled_costs = pooled_cost_table(returns, fold_metrics)
        print("\n=== break-even on the pooled track (NaN = gross return not positive) ===")
        print(pooled_costs.round(4).to_string(index=False))
        pooled_costs.to_csv(res / "pooled_cost_table.csv", index=False)

    # Equivalence: a non-significant difference is not evidence of no difference.
    # TRAINED against FROZEN is the contrast the design can actually resolve, so it
    # is tested against a tight margin; the contrasts against the no-circuit control
    # are reported with their minimum detectable effect so the reader can see what
    # the design could and could not have found.
    from .equivalence import equivalence_table
    eq_rows = []
    models = set(returns["model"].unique())
    if {"qtcnn_trained", "qtcnn_frozen"} <= models:
        eq_rows.append(equivalence_table(returns, reference="qtcnn_frozen",
                                         models=["qtcnn_trained"],
                                         margins=(0.03, 0.10, 0.25), n_boot=args.bootstrap))
    if "qtcnn_noquantum" in models:
        others = [m for m in ("qtcnn_trained", "qtcnn_frozen", "qtcnn_surrogate") if m in models]
        if others:
            eq_rows.append(equivalence_table(returns, reference="qtcnn_noquantum",
                                             models=others, margins=(0.25, 0.50),
                                             n_boot=args.bootstrap))
    if eq_rows:
        eq = pd.concat(eq_rows, ignore_index=True)
        print("\n=== equivalence (TOST) and minimum detectable effect ===")
        print(eq[["model", "reference", "margin", "diff_sharpe", "se", "ci_low", "ci_high",
                  "p_tost", "equivalent", "mde80"]].round(4).to_string(index=False))
        eq.to_csv(res / "equivalence.csv", index=False)

    head.to_csv(res / "headline.csv", index=False)
    contrasts.to_csv(res / "ablation_contrasts.csv", index=False)
    rw.to_csv(res / "romano_wolf.csv", index=False)

    sc = sharpe_contrasts(returns, n_boot=args.bootstrap) if "qtcnn_noquantum" in set(returns["model"]) else pd.DataFrame()
    if not sc.empty:
        print("\n=== Sharpe-difference contrasts vs qtcnn_noquantum (paired bootstrap) ===")
        print(sc[["model", "diff", "se", "ci_low", "ci_high", "p_value", "n"]].round(4).to_string(index=False))
        sc.to_csv(res / "sharpe_contrasts.csv", index=False)

    tables = Path(args.tables)
    paper_headline_table(head, tables / "headline.tex",
                         n_days=int(head["n_days"].max()),
                         n_trials=n_trials,
                         market=args.market,
                         label="tab:headline" + args.label_suffix)
    if not contrasts.empty:
        to_latex(contrasts, tables / "ablation.tex",
                 "Diebold-Mariano contrasts against the no-quantum control"
                 + (f" for {args.market}" if args.market else "") + ".",
                 "tab:ablation" + args.label_suffix)
    print(f"\ntables -> {tables}")


if __name__ == "__main__":
    main()

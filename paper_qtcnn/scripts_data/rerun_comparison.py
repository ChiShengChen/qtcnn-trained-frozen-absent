"""September runs against the reproducible November rerun, condition by condition.

The September network results were produced before the code was first committed
and could not be reproduced by it. This script tabulates, for every condition
that was re-run, the pooled-track Sharpe ratio of each run and the correlation
of per-cell Sharpe ratios between them, plus the trained-versus-frozen contrast
in each run. Writes manuscript/tables/rerun.tex.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtcnn_research.aggregate import load_returns, pooled_track, PAPER_LABELS  # noqa: E402
from qtcnn_research import evaluation as ev                                     # noqa: E402

PANELS = [  # label, September dir, November dir, models
    ("JPX walk-forward", "results/walkforward_2026-09_original", "results/walkforward",
     ["qtcnn_trained", "qtcnn_frozen", "qtcnn_surrogate", "qtcnn_noquantum", "qtcnn_quantumonly",
      "tcn", "lstm", "transformer", "mlp"]),
    ("JPX, full budget", "results/walkforward_nostop_2026-09_original", "results/walkforward_nostop",
     ["qtcnn_trained", "qtcnn_frozen"]),
    ("US walk-forward", "results/us_walkforward_2026-09_original", "results/us_walkforward",
     ["qtcnn_trained", "qtcnn_frozen", "qtcnn_surrogate", "qtcnn_noquantum", "tcn", "lstm",
      "transformer", "mlp"]),
    ("Holdout", "results/holdout_2026-09_original", "results/holdout",
     ["qtcnn_trained", "qtcnn_frozen", "qtcnn_surrogate", "qtcnn_noquantum", "tcn", "lstm",
      "transformer", "mlp"]),
]


def cells(d: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted(d.glob("fold_metrics_*.csv"))], ignore_index=True)


def tf_contrast(d: Path) -> str:
    f = d / "equivalence.csv"
    if not f.exists():
        return "--"
    e = pd.read_csv(f)
    e = e[(e.model == "qtcnn_trained") & (e.reference == "qtcnn_frozen") & (e.margin == 0.03)]
    if e.empty:
        return "--"
    r = e.iloc[0]
    p = "<0.001" if r.p_tost < 0.001 else (f"{r.p_tost:.4f}" if r.p_tost < 0.01 else f"{r.p_tost:.3f}")
    return f"${r.diff_sharpe:+.3f}$ ({r.se:.4f}), $p_{{0.03}}{'' if p.startswith('<') else '='}{p}$"


def main() -> None:
    lines, summary = [], []
    for label, old, new, models in PANELS:
        o_dir, n_dir = ROOT / old, ROOT / new
        o_ret, n_ret = load_returns(o_dir / "returns"), load_returns(n_dir / "returns")
        o_c, n_c = cells(o_dir), cells(n_dir)
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for m in models:
            so = ev.annualised_sharpe(pooled_track(o_ret, m))
            sn = ev.annualised_sharpe(pooled_track(n_ret, m))
            a = o_c[o_c.model == m].set_index(["fold", "seed"]).sharpe_ann
            b = n_c[n_c.model == m].set_index(["fold", "seed"]).sharpe_ann
            j = pd.concat([a.rename("o"), b.rename("n")], axis=1).dropna()
            r = j.o.corr(j.n) if len(j) > 5 else np.nan
            rs = "--" if not np.isfinite(r) else f"{r:.2f}"
            name = PAPER_LABELS.get(m, m).replace("QTCNN, ", "")
            lines.append(f"\\quad {name} & ${so:+.2f}$ & ${sn:+.2f}$ & {rs} \\\\")
            summary.append((label, m, so, sn, r))
        lines.append(f"\\quad trained $-$ frozen (s.e.) & \\multicolumn{{3}}{{l}}{{Sept.\\ {tf_contrast(o_dir)}; "
                     f"Nov.\\ {tf_contrast(n_dir)}}} \\\\[2pt]")
    body = "\n".join(lines)
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{The first (September 2026) and the reproducible (November 2026) run of every "
        "network condition that was re-run: pooled-track annualised Sharpe ratio of each, and the "
        "correlation between the two runs' per-cell Sharpe ratios (not given for the five-cell "
        "holdout). The last line of each panel is the trained-minus-frozen pooled Sharpe difference "
        "with its paired standard error and the two one-sided test against a 0.03 margin.}\n"
        "\\label{tab:rerun}\n\\small\n\\begin{tabular}{lrrr}\n\\toprule\n"
        "Condition & September & November & Cell corr. \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    out = ROOT / "manuscript/tables/rerun.tex"
    out.write_text(tex)
    print(tex)


if __name__ == "__main__":
    main()

"""Rows of the locked-holdout table (manuscript Table tab:holdout), straight from the CSVs.

Reads results/holdout/{headline,romano_wolf,pooled_cost_table}.csv written by
`python -m qtcnn_research.aggregate --results results/holdout --n-trials 55` and
prints the LaTeX rows in descending Sharpe order, plus the DSR dispersion and
expected-maximum figures quoted in the caption.

    python scripts_data/holdout_table.py            # print rows
    python scripts_data/holdout_table.py --check    # exit 1 if the manuscript differs
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"transformer": "Transformer baseline", "qtcnn_surrogate": "QTCNN, classical surrogate",
          "lstm": "LSTM baseline", "qtcnn_trained": "QTCNN, trained circuit",
          "qtcnn_frozen": "QTCNN, frozen circuit", "qtcnn_noquantum": "QTCNN, no circuit",
          "tcn": "TCN baseline", "lightgbm": "LightGBM baseline", "mlp": "MLP baseline"}


def rows() -> tuple[list[str], str]:
    d = ROOT / "results/holdout"
    h = pd.read_csv(d / "headline.csv").set_index("model")
    rw = pd.read_csv(d / "romano_wolf.csv").set_index("model")
    c = pd.read_csv(d / "pooled_cost_table.csv").set_index("model")
    out = []
    for m in sorted(LABELS, key=lambda m: -h.loc[m, "sharpe_ann"]):
        be = c.loc[m, "break_even_bps"]
        be = "--  " if pd.isna(be) else f"{be:.1f}"
        prw = "--  " if m == "qtcnn_noquantum" else f"{rw.loc[m, 'p_rw']:.2f}"
        out.append(f"{LABELS[m]:26s} & ${h.loc[m, 'sharpe_ann']:+.2f}$ & {h.loc[m, 'sharpe_seed_std']:.2f}"
                   f" & {h.loc[m, 'sharpe_p_value']:.2f} & {prw} & {h.loc[m, 'dsr_dsr']:.2f} & {be} \\\\")
    note = (f"trial Sharpe sd {h['dsr_trial_sd_ann'].iloc[0]:.2f}, "
            f"expected maximum under the null {h['dsr_sr_zero_ann'].iloc[0]:.2f}")
    return out, note


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    r, note = rows()
    if not a.check:
        print("\n".join(r)); print("%", note); return
    tex = (ROOT / "manuscript/sections/06_results_holdout.tex").read_text()
    block = tex[tex.index("\\label{tab:holdout}"):tex.index("\\bottomrule", tex.index("\\label{tab:holdout}"))]
    have = [l for l in block.splitlines() if "&" in l and not l.startswith("Condition")]
    if have != r:
        print("manuscript holdout table differs from the CSVs:")
        for x, y in zip(have, r):
            if x != y: print("  tex:", x, "\n  csv:", y)
        sys.exit(1)
    print(f"holdout table matches the CSVs ({len(r)} rows); {note}")


if __name__ == "__main__":
    main()

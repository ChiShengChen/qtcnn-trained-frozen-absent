"""Circuit against the parameter-matched classical surrogate, on every panel.

aggregate.py contrasts every condition with the no-circuit control; this adds the
contrast the ablation's middle question needs: is the random feature a quantum
circuit computes any different from a classical one of equal size? Writes
results/<panel>/surrogate_contrasts.csv.

    python scripts_data/surrogate_contrasts.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtcnn_research.aggregate import load_returns, pooled_track   # noqa: E402
from qtcnn_research import evaluation as ev                       # noqa: E402
from qtcnn_research.equivalence import equivalence_table          # noqa: E402


def main() -> None:
    for panel in ["walkforward", "us_walkforward", "holdout"]:
        res = ROOT / "results" / panel
        r = load_returns(res / "returns")
        rows = []
        for m in ["qtcnn_trained", "qtcnn_frozen"]:
            a, b = pooled_track(r, m), pooled_track(r, "qtcnn_surrogate")
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            sd = ev.sharpe_difference_test(j.a, j.b)
            dm = ev.diebold_mariano(j.a, j.b)
            eq = equivalence_table(r, reference="qtcnn_surrogate", models=[m],
                                   margins=(0.25, 0.50), n_boot=2000).set_index("margin")
            rows.append(dict(model=m, reference="qtcnn_surrogate", n_days=len(j),
                             sharpe_model=ev.annualised_sharpe(j.a), sharpe_reference=ev.annualised_sharpe(j.b),
                             diff=sd.get("diff"), se=sd.get("se"), ci_low=sd.get("ci_low"), ci_high=sd.get("ci_high"),
                             p_sharpe=sd.get("p_value"), p_dm=dm["p_value"],
                             p_tost_025=eq.loc[0.25, "p_tost"], p_tost_050=eq.loc[0.50, "p_tost"]))
        out = pd.DataFrame(rows)
        out.to_csv(res / "surrogate_contrasts.csv", index=False)
        print(panel); print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

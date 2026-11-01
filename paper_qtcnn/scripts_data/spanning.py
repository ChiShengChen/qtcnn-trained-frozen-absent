"""Is the trained-circuit model's return spanned by the classical models?

Regresses the trained-circuit pooled daily return on the four neural baselines
and, separately, on its own no-circuit twin; blends it equal-weight with each
classical model. Writes spanning_quantum_vs_classical.csv, spanning_vs_twin.csv
and combination_gain.csv next to the results it reads.

    python scripts_data/spanning.py --results results/walkforward
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtcnn_research.aggregate import load_returns, pooled_track          # noqa: E402
from qtcnn_research.generalisation import spanning_regression, combination_gain  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/walkforward")
    p.add_argument("--quantum", default="qtcnn_trained")
    a = p.parse_args()
    res = ROOT / a.results
    returns = load_returns(res / "returns")
    classical = ["tcn", "lstm", "transformer", "mlp"]
    tracks = {m: pooled_track(returns, m) for m in [a.quantum, "qtcnn_noquantum", *classical]}
    span = spanning_regression(tracks[a.quantum], {m: tracks[m] for m in classical})
    twin = spanning_regression(tracks[a.quantum], {"qtcnn_noquantum": tracks["qtcnn_noquantum"]})
    gain = combination_gain(tracks, a.quantum, classical + ["qtcnn_noquantum"])
    pd.DataFrame([span]).to_csv(res / "spanning_quantum_vs_classical.csv", index=False)
    pd.DataFrame([twin]).to_csv(res / "spanning_vs_twin.csv", index=False)
    gain.to_csv(res / "combination_gain.csv", index=False)
    print("vs four neural baselines:", {k: round(v, 4) for k, v in span.items() if not k.startswith(("beta_", "t_"))})
    print("vs no-circuit twin:      ", {k: round(v, 4) for k, v in twin.items()})
    print(gain.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

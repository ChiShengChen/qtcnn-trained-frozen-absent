"""Shot budget measured on the circuit inputs the model actually produces.

The shot-noise scan in `qtcnn_research.diagnostics` feeds the circuit uniform
random angles. Shot requirements scale with the spread of the scores being
ranked, so the number that matters for deployment is the one measured at the
cross-sectional spread the encoder actually creates on a trading day, not at the
spread of random inputs. This script takes real feature windows, pushes them
through the encoder of an initialised model (which is exactly the FROZEN
condition), and repeats the shot scan at that input distribution, one trading day
at a time so the rank correlation is the within-date quantity the strategy uses.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtcnn_research import data as D                      # noqa: E402
from qtcnn_research.models import build_model             # noqa: E402
from qtcnn_research.quantum import CircuitConfig          # noqa: E402
from qtcnn_research.resources import claim_cpu, hide_gpu  # noqa: E402
from scipy import stats                                   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache/features.parquet")
    p.add_argument("--seq-cache", default="cache/sequences")
    p.add_argument("--n-dates", type=int, default=20)
    p.add_argument("--per-date", type=int, default=400, help="securities per cross-section")
    p.add_argument("--shots", type=int, nargs="+", default=[64, 256, 1024, 8192])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="results/diagnostics/shot_budget_real.csv")
    a = p.parse_args()
    claim_cpu(a.threads, niceness=15); hide_gpu()
    import pennylane as qml
    import qtcnn_research.quantum as Q

    X = np.load(Path(a.seq_cache) / "X_seq20.npy", mmap_mode="r")
    idx = pd.read_parquet(Path(a.seq_cache) / "index_seq20.parquet")
    rng = np.random.default_rng(a.seed)
    dates = rng.choice(np.sort(idx["Date"].unique()), size=a.n_dates, replace=False)

    torch.manual_seed(a.seed)
    model = build_model("qtcnn_frozen", n_features=21, seq_len=20).eval()
    cfg: CircuitConfig = model.quantum.cfg
    weights = model.quantum.weights.detach().numpy()

    def run(angles: np.ndarray, shots):
        dev = qml.device("default.qubit", wires=cfg.n_qubits, shots=shots, seed=a.seed)
        @qml.qnode(dev)
        def circuit(xi, w):
            Q._embed(torch.as_tensor(xi), cfg)
            active = list(range(cfg.n_qubits)); cursor = 0
            for layer in range(cfg.effective_layers):
                if len(active) < 2: break
                Q._conv_block(torch.as_tensor(w[layer]), active)
                active = active[::2]
            return qml.expval(qml.PauliZ(active[0] if active else 0))
        return np.array([float(circuit(x, weights)) for x in angles])

    rows = []
    for d in dates:
        sel = idx.index[idx["Date"] == d].to_numpy()
        if len(sel) < 50: continue
        sel = np.sort(rng.choice(sel, size=min(a.per_date, len(sel)), replace=False))
        with torch.no_grad():
            enc = model.encoder(torch.as_tensor(np.asarray(X[sel]), dtype=torch.float32)).numpy()
        angles = np.clip(enc[:, : cfg.n_qubits], -np.pi, np.pi).astype(float)
        exact = run(angles, None)
        for s in a.shots:
            noisy = run(angles, s)
            rows.append(dict(date=pd.Timestamp(d).date(), n=len(sel), shots=s,
                             score_std=float(exact.std()),
                             rank_corr=float(stats.spearmanr(exact, noisy).statistic),
                             rmse=float(np.sqrt(((exact - noisy) ** 2).mean()))))
        print(f"{pd.Timestamp(d).date()}  n={len(sel)}  score sd={exact.std():.4f}  "
              + "  ".join(f"{r['shots']}:{r['rank_corr']:.4f}" for r in rows[-len(a.shots):]), flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n=== mean over dates ===", flush=True)
    print(df.groupby("shots")[["score_std", "rank_corr", "rmse"]].mean().round(4).to_string(), flush=True)
    print(f"\ncross-sectional score sd: mean {df.score_std.mean():.4f}, "
          f"range {df.score_std.min():.4f}-{df.score_std.max():.4f}", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

"""Shot budget measured on the circuit inputs the model actually produces.

The shot-noise scan in `qtcnn_research.diagnostics` feeds the circuit uniform
random angles. Shot requirements scale with the spread of the scores being
ranked, so the number that matters for deployment is the one measured at the
cross-sectional spread the encoder actually creates on a trading day, not at the
spread of random inputs. This script takes real feature windows, pushes them
through an encoder and repeats the shot scan at that input distribution, one
trading day at a time so the rank correlation is the within-date quantity the
strategy uses.

Two encoders are supported. Without ``--states`` the encoder is an initialised,
untrained model (the first measurement, F37, was made this way and the
manuscript mis-described it as trained). With ``--states`` the script loads the
weights saved by ``run_experiment.py --save-state`` for each (model, fold, seed)
cell and measures each cell on trading dates drawn from its own out-of-sample
test window, which is the deployed configuration.
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
    p.add_argument("--states", nargs="*", default=None,
                   help="trained-weight files <model>_f<fold>_s<seed>.pt from run_experiment "
                        "--save-state; dates are drawn from each cell's test window")
    p.add_argument("--clip", action="store_true",
                   help="clip encoder outputs to [-pi, pi] before embedding (the F37 run did; "
                        "the deployed model does not)")
    p.add_argument("--train-days", type=int, default=500)
    p.add_argument("--valid-days", type=int, default=60)
    p.add_argument("--test-days", type=int, default=125)
    p.add_argument("--embargo-days", type=int, default=20)
    a = p.parse_args()
    claim_cpu(a.threads, niceness=15); hide_gpu()
    import pennylane as qml
    import qtcnn_research.quantum as Q

    X = np.load(Path(a.seq_cache) / "X_seq20.npy", mmap_mode="r")
    idx = pd.read_parquet(Path(a.seq_cache) / "index_seq20.parquet")
    rng = np.random.default_rng(a.seed)
    all_dates = np.sort(idx["Date"].unique())

    # (label, model, fold, seed, state or None, candidate dates)
    cells = []
    if a.states:
        import re
        from qtcnn_research.walkforward import make_walk_forward_folds
        # Fold boundaries must come from the feature-cache calendar, exactly as
        # run_experiment builds them; the sequence cache starts 19 days later and
        # would shift every test window by about a month.
        feat_dates = np.sort(pd.read_parquet(a.cache, columns=["Date"])["Date"].unique())
        dev = feat_dates[feat_dates < D.HOLDOUT_START]
        folds = {f.fold: f for f in make_walk_forward_folds(
            dev, train_days=a.train_days, test_days=a.test_days,
            valid_days=a.valid_days, embargo_days=a.embargo_days)}
        for path in a.states:
            m = re.match(r"(.+)_f(\d+)_s(\d+)\.pt$", Path(path).name)
            if not m:
                raise SystemExit(f"cannot parse model/fold/seed from {path}")
            name, fold, seed = m.group(1), int(m.group(2)), int(m.group(3))
            f = folds[fold]
            window = all_dates[(all_dates >= f.test_start) & (all_dates <= f.test_end)]
            cells.append((Path(path).stem, name, fold, seed, torch.load(path, map_location="cpu"), window))
    else:
        cells.append(("untrained_s%d" % a.seed, "qtcnn_frozen", -1, a.seed, None, all_dates))

    def make_runner(model):
        cfg: CircuitConfig = model.quantum.cfg
        weights = model.quantum.weights.detach().numpy()

        def run(angles: np.ndarray, shots):
            dev = qml.device("default.qubit", wires=cfg.n_qubits, shots=shots, seed=a.seed)
            @qml.qnode(dev)
            def circuit(xi, w):
                Q._embed(torch.as_tensor(xi), cfg)
                active = list(range(cfg.n_qubits))
                for layer in range(cfg.effective_layers):
                    if len(active) < 2: break
                    Q._conv_block(torch.as_tensor(w[layer]), active)
                    active = active[::2]
                return qml.expval(qml.PauliZ(active[0] if active else 0))
            return np.array([float(circuit(x, weights)) for x in angles])
        return cfg, run

    rows = []
    for label, name, fold, seed, state, window in cells:
        torch.manual_seed(seed)
        model = build_model(name, n_features=21, seq_len=20, seed=seed).eval()
        if state is not None:
            model.load_state_dict(state)
        cfg, run = make_runner(model)
        dates = rng.choice(window, size=min(a.n_dates, len(window)), replace=False)
        for d in dates:
            sel = idx.index[idx["Date"] == d].to_numpy()
            if len(sel) < 50: continue
            sel = np.sort(rng.choice(sel, size=min(a.per_date, len(sel)), replace=False))
            with torch.no_grad():
                enc = model.encoder(torch.as_tensor(np.asarray(X[sel]), dtype=torch.float32)).numpy()
            angles = enc[:, : cfg.n_qubits].astype(float)
            if a.clip:
                angles = np.clip(angles, -np.pi, np.pi)
            exact = run(angles, None)
            for s_ in a.shots:
                noisy = run(angles, s_)
                rows.append(dict(cell=label, model=name, fold=fold, seed=seed,
                                 date=pd.Timestamp(d).date(), n=len(sel), shots=s_,
                                 angle_std=float(angles.std()),
                                 score_std=float(exact.std()),
                                 rank_corr=float(stats.spearmanr(exact, noisy).statistic),
                                 rmse=float(np.sqrt(((exact - noisy) ** 2).mean()))))
            print(f"{label}  {pd.Timestamp(d).date()}  n={len(sel)}  angle sd={angles.std():.3f}  "
                  f"score sd={exact.std():.4f}  "
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

"""Quantum-specific analyses that a general ML venue would not ask for but a
quantum venue will: circuit scaling, trainability, expressibility, resource
counts, and behaviour under shot and gate noise.

These are what turn "our hybrid model scored X" into a claim about the circuit,
and they are cheap relative to the walk-forward sweep because none of them
require the financial data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Resource counts
# ----------------------------------------------------------------------------
def resource_table(qubit_grid: Sequence[int] = (4, 6, 8, 10, 12, 16),
                   layer_grid: Sequence[int] = (1, 2, 3, 4),
                   shared: Sequence[bool] = (True, False)) -> pd.DataFrame:
    """Gate counts, depth and parameter counts for the hardware-feasibility table."""
    from qtcnn_research.quantum import CircuitConfig

    rows = []
    for q in qubit_grid:
        for L in layer_grid:
            for sh in shared:
                cfg = CircuitConfig(n_qubits=q, n_layers=L, shared_params=sh)
                row = cfg.gate_counts()
                row.update({"requested_layers": L, "shared_params": sh})
                rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Trainability: barren plateaus
# ----------------------------------------------------------------------------
def barren_plateau_scan(qubit_grid: Sequence[int] = (2, 4, 6, 8, 10, 12),
                        n_layers: int = 3, n_samples: int = 120,
                        batch: int = 16, seed: int = 0) -> pd.DataFrame:
    """Gradient variance against qubit count.

    An exponential decay in ``grad_var`` with qubit count is the signature of a
    barren plateau. Shallow, parameter-shared QCNNs are expected to resist it,
    and demonstrating that is a substantive claim about this architecture rather
    than a generic benchmark number.
    """
    from qtcnn_research.quantum import CircuitConfig, gradient_variance

    rows = []
    for q in qubit_grid:
        cfg = CircuitConfig(n_qubits=q, n_layers=n_layers)
        t0 = time.perf_counter()
        row = gradient_variance(cfg, n_samples=n_samples, batch=batch, seed=seed)
        row["seconds"] = time.perf_counter() - t0
        row["n_params"] = cfg.n_circuit_params
        rows.append(row)
        print(f"  qubits={q:3d} grad_var={row['grad_var']:.3e} "
              f"({row['seconds']:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    if len(df) > 2:
        # Fit log(var) = a + b*qubits; b << 0 indicates exponential decay.
        mask = df["grad_var"] > 0
        if mask.sum() > 2:
            b, a = np.polyfit(df.loc[mask, "n_qubits"], np.log(df.loc[mask, "grad_var"]), 1)
            df["decay_rate_per_qubit"] = b
            df["decay_intercept"] = a
    return df


def expressibility_scan(qubit_grid: Sequence[int] = (4, 6, 8),
                        layer_grid: Sequence[int] = (1, 2, 3),
                        n_pairs: int = 400, seed: int = 0) -> pd.DataFrame:
    """Sim et al. (2019) expressibility: KL divergence from the Haar ensemble."""
    from qtcnn_research.quantum import CircuitConfig, expressibility

    rows = []
    for q in qubit_grid:
        for L in layer_grid:
            cfg = CircuitConfig(n_qubits=q, n_layers=L)
            if cfg.effective_layers < L:
                continue  # pooling caps the depth; skip unreachable settings
            t0 = time.perf_counter()
            kl = expressibility(cfg, n_pairs=n_pairs, seed=seed)
            rows.append({"n_qubits": q, "layers": cfg.effective_layers,
                         "expressibility_kl": kl, "n_params": cfg.n_circuit_params,
                         "seconds": time.perf_counter() - t0})
            print(f"  qubits={q} layers={cfg.effective_layers} KL={kl:.4f}", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Noise robustness
# ----------------------------------------------------------------------------
def shot_noise_scan(shot_grid: Sequence[Optional[int]] = (None, 8192, 1024, 256, 64),
                    n_qubits: int = 8, n_layers: int = 3,
                    n_inputs: int = 256, seed: int = 0) -> pd.DataFrame:
    """Deviation of circuit outputs from the analytic value at finite shot counts.

    A model whose scores are only reproducible at 100k shots is not deployable;
    reporting the shot count needed to preserve the ranking is the honest
    version of a hardware-feasibility claim.
    """
    import pennylane as qml
    import torch

    import qtcnn_research.quantum as Q
    from qtcnn_research.quantum import CircuitConfig

    cfg = CircuitConfig(n_qubits=n_qubits, n_layers=n_layers)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi, np.pi, (n_inputs, n_qubits))
    w = rng.uniform(-np.pi, np.pi, (cfg.effective_layers, 6))

    def run(shots: Optional[int]) -> np.ndarray:
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots, seed=seed)

        @qml.qnode(dev)
        def circuit(xi, weights):
            Q._embed(xi, cfg)
            active = list(range(n_qubits))
            for layer in range(cfg.effective_layers):
                if len(active) < 2:
                    break
                Q._conv_block(weights[layer], active)
                active = active[::2]
            return qml.expval(qml.PauliZ(active[0] if active else 0))

        return np.array([float(circuit(x[i], w)) for i in range(n_inputs)])

    exact = run(None)
    rows = []
    for shots in shot_grid:
        vals = exact if shots is None else run(shots)
        # Spearman rank correlation: ranking is what the strategy consumes.
        rank_corr = pd.Series(vals).corr(pd.Series(exact), method="spearman")
        rows.append({
            "shots": "analytic" if shots is None else shots,
            "rmse_vs_analytic": float(np.sqrt(np.mean((vals - exact) ** 2))),
            "rank_corr_vs_analytic": float(rank_corr),
            "output_std": float(vals.std()),
        })
        print(f"  shots={rows[-1]['shots']:>8}  rmse={rows[-1]['rmse_vs_analytic']:.4f} "
              f"rank_corr={rank_corr:.4f}", flush=True)
    return pd.DataFrame(rows)


def depolarising_noise_scan(
    error_grid: Sequence[float] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05),
    n_qubits: int = 8,
    n_layers: int = 3,
    n_inputs: int = 128,
    seed: int = 0,
    heterogeneous: bool = False,
    shots: Optional[int] = None,
) -> pd.DataFrame:
    """Output degradation under per-gate depolarising noise on a mixed-state device.

    Uniform depolarising noise multiplies every expectation value by the same
    factor. That is a monotone transform, so it leaves any ranking exactly
    intact, and a rank correlation of 1.0 from that setting is an identity
    rather than a robustness result. Two settings make the test informative:
    ``heterogeneous=True`` gives each wire its own error rate, breaking the
    uniformity, and ``shots`` adds the sampling noise a real device also has,
    which does reorder scores.
    """
    import pennylane as qml

    import qtcnn_research.quantum as Q
    from qtcnn_research.quantum import CircuitConfig

    cfg = CircuitConfig(n_qubits=n_qubits, n_layers=n_layers)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi, np.pi, (n_inputs, n_qubits))
    w = rng.uniform(-np.pi, np.pi, (cfg.effective_layers, 6))

    # Per-wire multipliers on the nominal rate, drawn once so that the noise
    # profile is a fixed property of the "device" and not resampled per input.
    wire_scale = rng.uniform(0.2, 2.0, n_qubits) if heterogeneous else np.ones(n_qubits)

    def _body(xi, weights, p):
        Q._embed(xi, cfg)
        active = list(range(n_qubits))
        for layer in range(cfg.effective_layers):
            if len(active) < 2:
                break
            Q._conv_block(weights[layer], active)
            if p > 0:
                for wire in active:
                    qml.DepolarizingChannel(min(0.75, p * wire_scale[wire]), wires=wire)
            active = active[::2]
        return qml.expval(qml.PauliZ(active[0] if active else 0))

    def run(p: float, n_shots: Optional[int]) -> np.ndarray:
        dev = qml.device("default.mixed", wires=n_qubits, shots=n_shots)
        circuit = qml.QNode(lambda xi, weights: _body(xi, weights, p), dev)
        return np.array([float(circuit(x[i], w)) for i in range(n_inputs)])

    # Reference is always the noiseless, infinite-shot circuit.
    exact = run(0.0, None)

    rows = []
    for p in error_grid:
        vals = run(p, shots)
        rows.append({
            "depolarising_p": p,
            "heterogeneous": heterogeneous,
            "shots": "analytic" if shots is None else shots,
            "rmse_vs_noiseless": float(np.sqrt(np.mean((vals - exact) ** 2))),
            "rank_corr_vs_noiseless": float(
                pd.Series(vals).corr(pd.Series(exact), method="spearman")
            ),
            "signal_contraction": float(np.std(vals) / (np.std(exact) + 1e-12)),
        })
        print(f"  p={p:<6} het={heterogeneous!s:5s} shots={rows[-1]['shots']:>8} "
              f"rmse={rows[-1]['rmse_vs_noiseless']:.4f} "
              f"rank_corr={rows[-1]['rank_corr_vs_noiseless']:.4f}", flush=True)
    return pd.DataFrame(rows)


def amplitude_damping_scan(
    error_grid: Sequence[float] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05),
    n_qubits: int = 8,
    n_layers: int = 3,
    n_inputs: int = 128,
    seed: int = 0,
    shots: Optional[int] = None,
) -> pd.DataFrame:
    """Degradation under amplitude damping, a non-unital channel.

    This is the control for the depolarising result. Depolarising noise scales a
    Pauli expectation by an input-independent factor, so it cannot reorder
    scores no matter how large the error rate, and a rank correlation of 1.0
    there says nothing about robustness. Amplitude damping biases the state
    towards |0> rather than towards the maximally mixed state, so its effect on
    <Z> depends on the state and it *can* reorder. If this scan also returned
    1.0 the whole measurement would lack power.
    """
    import pennylane as qml

    import qtcnn_research.quantum as Q
    from qtcnn_research.quantum import CircuitConfig

    cfg = CircuitConfig(n_qubits=n_qubits, n_layers=n_layers)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi, np.pi, (n_inputs, n_qubits))
    w = rng.uniform(-np.pi, np.pi, (cfg.effective_layers, 6))

    def _body(xi, weights, gamma):
        Q._embed(xi, cfg)
        active = list(range(n_qubits))
        for layer in range(cfg.effective_layers):
            if len(active) < 2:
                break
            Q._conv_block(weights[layer], active)
            if gamma > 0:
                for wire in active:
                    qml.AmplitudeDamping(gamma, wires=wire)
            active = active[::2]
        return qml.expval(qml.PauliZ(active[0] if active else 0))

    def run(gamma: float, n_shots: Optional[int]) -> np.ndarray:
        dev = qml.device("default.mixed", wires=n_qubits, shots=n_shots)
        circuit = qml.QNode(lambda xi, weights: _body(xi, weights, gamma), dev)
        return np.array([float(circuit(x[i], w)) for i in range(n_inputs)])

    exact = run(0.0, None)
    rows = []
    for gamma in error_grid:
        vals = run(gamma, shots)
        rows.append({
            "amplitude_damping_gamma": gamma,
            "shots": "analytic" if shots is None else shots,
            "rmse_vs_noiseless": float(np.sqrt(np.mean((vals - exact) ** 2))),
            "rank_corr_vs_noiseless": float(
                pd.Series(vals).corr(pd.Series(exact), method="spearman")
            ),
            "mean_shift": float(np.mean(vals) - np.mean(exact)),
            "signal_contraction": float(np.std(vals) / (np.std(exact) + 1e-12)),
        })
        print(f"  gamma={gamma:<6} shots={rows[-1]['shots']:>8} "
              f"rmse={rows[-1]['rmse_vs_noiseless']:.4f} "
              f"rank_corr={rows[-1]['rank_corr_vs_noiseless']:.4f} "
              f"mean_shift={rows[-1]['mean_shift']:+.4f}", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/diagnostics")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--nice", type=int, default=15)
    p.add_argument("--max-load", type=float, default=1.5)
    p.add_argument("--load-wait", type=int, default=1800)
    p.add_argument("--which", nargs="+",
                   default=["resources", "barren", "expressibility", "shots",
                            "depolarising", "damping"])
    p.add_argument("--quick", action="store_true", help="smaller grids for a smoke test")
    args = p.parse_args()

    from qtcnn_research.resources import claim_cpu, hide_gpu, wait_for_cpu

    claim_cpu(args.threads, niceness=args.nice)
    hide_gpu()
    wait_for_cpu(max_load_per_core=args.max_load, max_wait_seconds=args.load_wait)

    import warnings

    warnings.filterwarnings("ignore")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if "resources" in args.which:
        print("resource counts ...", flush=True)
        resource_table().to_csv(out / "resources.csv", index=False)

    if "barren" in args.which:
        print("barren plateau scan ...", flush=True)
        grid = (2, 4, 6, 8) if args.quick else (2, 4, 6, 8, 10, 12)
        n = 40 if args.quick else 120
        barren_plateau_scan(qubit_grid=grid, n_samples=n).to_csv(
            out / "barren_plateau.csv", index=False)

    if "expressibility" in args.which:
        print("expressibility scan ...", flush=True)
        grid = (4, 6) if args.quick else (4, 6, 8)
        n = 150 if args.quick else 400
        expressibility_scan(qubit_grid=grid, n_pairs=n).to_csv(
            out / "expressibility.csv", index=False)

    if "shots" in args.which:
        print("shot noise scan ...", flush=True)
        shot_noise_scan(n_inputs=64 if args.quick else 256).to_csv(
            out / "shot_noise.csv", index=False)

    if "depolarising" in args.which:
        print("depolarising noise scan ...", flush=True)
        n_in = 32 if args.quick else 128
        # Three settings: the uniform case (where rank preservation is an
        # identity, reported for completeness), per-wire heterogeneous rates,
        # and heterogeneous rates combined with a realistic shot budget.
        frames = [
            depolarising_noise_scan(n_inputs=n_in, heterogeneous=False),
            depolarising_noise_scan(n_inputs=n_in, heterogeneous=True),
            depolarising_noise_scan(n_inputs=n_in, heterogeneous=True, shots=1024),
        ]
        pd.concat(frames, ignore_index=True).to_csv(out / "depolarising.csv", index=False)

    if "damping" in args.which:
        print("amplitude damping scan ...", flush=True)
        n_in = 32 if args.quick else 128
        pd.concat(
            [
                amplitude_damping_scan(n_inputs=n_in),
                amplitude_damping_scan(n_inputs=n_in, shots=1024),
            ],
            ignore_index=True,
        ).to_csv(out / "amplitude_damping.csv", index=False)

    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()

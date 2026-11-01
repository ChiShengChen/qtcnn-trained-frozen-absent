"""Benchmark the batched QCNN against the per-sample loop, and measure their agreement.

The manuscript claims a speed-up and an agreement tolerance for the batched
formulation. This script is the artefact behind both numbers: it times the two
implementations on identical inputs and reports the maximum absolute difference
of their outputs, writing results/diagnostics/batching_benchmark.csv.

Run under the scheduler, e.g.
    gq run --cpu-only -c 2 -m 6 -- ./.venv/bin/python scripts_data/bench_batched_circuit.py
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtcnn_research.quantum import CircuitConfig, make_qcnn  # noqa: E402
from qtcnn_research.resources import claim_cpu, describe_environment  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qubits", type=int, default=8)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--batches", type=int, nargs="+", default=[64, 256, 512])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="results/diagnostics/batching_benchmark.csv")
    a = p.parse_args()
    claim_cpu(a.threads, niceness=15)
    torch.manual_seed(0)

    cfg = CircuitConfig(n_qubits=a.qubits, n_layers=a.layers)
    qnode, weight_shape = make_qcnn(cfg)
    n_par = int(np.prod(weight_shape))
    theta = torch.rand(*weight_shape, dtype=torch.float64) * 2 * np.pi - np.pi

    rows = []
    for bs in a.batches:
        x = torch.rand(bs, cfg.n_qubits, dtype=torch.float64) * 2 * np.pi - np.pi
        # warm-up (compilation / caching) excluded from the timings
        _ = qnode(x[:2], theta)
        _ = qnode(x[0], theta)

        t_batch, t_loop = [], []
        for _ in range(a.repeats):
            t0 = time.perf_counter(); out_b = qnode(x, theta); t_batch.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            out_l = torch.stack([qnode(x[i], theta) for i in range(bs)])
            t_loop.append(time.perf_counter() - t0)
        out_b = torch.as_tensor(out_b).reshape(bs, -1)
        out_l = torch.as_tensor(out_l).reshape(bs, -1)
        diff = float((out_b - out_l).abs().max())
        rows.append(dict(n_qubits=cfg.n_qubits, n_layers=cfg.effective_layers, n_params=n_par,
                         batch_size=bs, repeats=a.repeats,
                         loop_seconds=float(np.median(t_loop)),
                         batched_seconds=float(np.median(t_batch)),
                         speedup=float(np.median(t_loop) / np.median(t_batch)),
                         max_abs_diff=diff))
        print(f"batch {bs:4d}: loop {np.median(t_loop):8.3f}s  batched {np.median(t_batch):7.3f}s  "
              f"speedup {np.median(t_loop)/np.median(t_batch):6.1f}x  max|diff| {diff:.3e}", flush=True)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    (out.parent / "batching_environment.txt").write_text(describe_environment())
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()

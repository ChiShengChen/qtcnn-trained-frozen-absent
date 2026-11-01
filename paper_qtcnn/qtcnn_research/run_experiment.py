"""Walk-forward experiment runner.

Designed to share a busy machine: it caps its own threads, runs at low
priority, waits out other people's load spikes, keeps the sequence tensor on
disk as a memmap rather than holding 4 GB of RAM, and checkpoints after every
(model, fold, seed) so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from qtcnn_research.resources import (
    claim_cpu,
    describe_environment,
    hide_gpu,
    wait_for_cpu,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="cache/features.parquet")
    p.add_argument("--seq-cache", default="cache/sequences")
    p.add_argument("--out", default="results/walkforward")
    p.add_argument("--tag", default=None,
                   help="suffix for this job's metrics file. Parallel gq jobs must use "
                        "distinct tags: they share the returns directory (filenames are "
                        "already unique per model/fold/seed) but appending to one CSV from "
                        "several processes races on the header. Defaults to the model list.")
    p.add_argument("--models", nargs="+", default=["qtcnn_trained", "qtcnn_frozen",
                                                   "qtcnn_surrogate", "qtcnn_noquantum", "tcn"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--per-side", type=int, default=200)
    p.add_argument("--train-days", type=int, default=500)
    p.add_argument("--valid-days", type=int, default=60)
    p.add_argument("--test-days", type=int, default=125)
    p.add_argument("--embargo-days", type=int, default=20)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--ansatz", default="qcnn", choices=["qcnn", "reupload_hea"],
                   help="circuit family for the qtcnn_* conditions; reupload_hea is the "
                        "second ansatz (data re-uploading, CNOT ring, multi-wire Z)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-train", type=int, default=60000)
    p.add_argument("--patience", type=int, default=6,
                   help="early-stopping patience on validation AUC; set >= --epochs to disable")
    p.add_argument("--final-weights", action="store_true",
                   help="score the weights of the last epoch rather than restoring the best "
                        "validation epoch; with --patience >= --epochs this is the full-budget control")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--nice", type=int, default=15)
    p.add_argument("--max-load", type=float, default=1.5,
                   help="wait while 1-minute load per core exceeds this")
    p.add_argument("--load-wait", type=int, default=1800,
                   help="seconds to wait for the machine to quieten before each fold")
    p.add_argument("--n-trials", type=int, default=None,
                   help="configurations searched across the whole study, for the "
                        "Deflated Sharpe Ratio. Set it explicitly when the sweep is "
                        "split across parallel jobs.")
    p.add_argument("--save-state", default=None,
                   help="directory in which to save each cell's trained weights as "
                        "<model>_f<fold>_s<seed>.pt, so diagnostics can be run on the "
                        "encoder the model actually deployed rather than on an initialised one")
    p.add_argument("--holdout", action="store_true",
                   help="evaluate the locked 2021-12-06..2022-06-24 window instead of walk-forward")
    p.add_argument("--pilot", action="store_true",
                   help="one fold, reduced epochs: validates the pipeline cheaply")
    p.add_argument("--gpu", action="store_true",
                   help="use the GPU this process was granted. Submit through the gq "
                        "scheduler, which is what actually waits for VRAM: "
                        "gq run -g 4G -- python -m qtcnn_research.run_experiment --gpu ... "
                        "Only helps the classical baselines; the quantum simulator is CPU-bound.")
    return p.parse_args()


def build_sequence_cache(df, feature_cols, seq_len: int, cache_dir: Path):
    """Materialise sliding windows once, on disk, as a float32 memmap."""
    import numpy as np
    import pandas as pd
    from qtcnn_research.data import make_sequences

    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path = cache_dir / f"X_seq{seq_len}.npy"
    idx_path = cache_dir / f"index_seq{seq_len}.parquet"

    if x_path.exists() and idx_path.exists():
        index = pd.read_parquet(idx_path)
        X = np.load(x_path, mmap_mode="r")
        return X, index

    # Two jobs started together (e.g. the quantum and control stages) would
    # otherwise truncate and write the same file at once. Build into a
    # process-private temporary file and publish it atomically; if another
    # process published first, discard ours and use theirs.
    import os
    X, index = make_sequences(df, feature_cols, seq_len=seq_len)
    tmp_x = cache_dir / f"X_seq{seq_len}.{os.getpid()}.tmp.npy"
    tmp_i = cache_dir / f"index_seq{seq_len}.{os.getpid()}.tmp.parquet"
    np.save(tmp_x, X)
    index.to_parquet(tmp_i, index=False)
    del X
    if x_path.exists() and idx_path.exists():
        tmp_x.unlink(); tmp_i.unlink()
    else:
        os.replace(tmp_x, x_path)
        os.replace(tmp_i, idx_path)
    index = pd.read_parquet(idx_path)
    return np.load(x_path, mmap_mode="r"), index


def main() -> None:
    args = parse_args()
    claim_cpu(args.threads, niceness=args.nice)
    if not args.gpu:
        # Default is strictly CPU: a CUDA-enabled torch would otherwise take
        # VRAM that the scheduler has not allocated to us.
        hide_gpu()

    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd

    from qtcnn_research import data as D
    from qtcnn_research.experiment import (
        TrainConfig,
        build_index_frame,
        prepare_fold,
        run_fold,
    )
    from qtcnn_research.walkforward import make_walk_forward_folds, Fold, summarise_folds

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.txt").write_text(describe_environment())

    print("loading feature cache ...", flush=True)
    df = pd.read_parquet(args.cache)
    feature_cols = D.FeatureSpec().columns()
    df["tradable"] = D.tradable_mask(df)
    print(f"  rows={len(df):,} dates={df.Date.nunique()} features={len(feature_cols)}", flush=True)

    print("building sequence cache ...", flush=True)
    t0 = time.perf_counter()
    X, index = build_sequence_cache(df, feature_cols, args.seq_len, Path(args.seq_cache))
    print(f"  X={X.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

    meta = df[["Date", "SecuritiesCode", "Target", "tradable"]]
    print("joining sequence index to metadata ...", flush=True)
    t0 = time.perf_counter()
    frame = build_index_frame(index, meta)
    print(f"  {len(frame):,} rows in {time.perf_counter()-t0:.1f}s", flush=True)

    all_dates = np.sort(df["Date"].unique())
    del df, meta  # the 2.5M-row feature frame is not needed past this point

    dev_dates = all_dates[all_dates < D.HOLDOUT_START]
    if args.holdout:
        # One fold: train on everything before the holdout, test on the holdout.
        hold = all_dates[all_dates >= D.HOLDOUT_START]
        cut = len(dev_dates) - args.valid_days - args.embargo_days
        folds = [Fold(0, pd.Timestamp(dev_dates[0]), pd.Timestamp(dev_dates[cut - 1]),
                      pd.Timestamp(dev_dates[cut]), pd.Timestamp(dev_dates[cut + args.valid_days - 1]),
                      pd.Timestamp(hold[0]), pd.Timestamp(hold[-1]))]
    else:
        folds = make_walk_forward_folds(
            dev_dates, train_days=args.train_days, test_days=args.test_days,
            valid_days=args.valid_days, embargo_days=args.embargo_days,
            max_folds=1 if args.pilot else args.max_folds,
        )
    print(summarise_folds(folds).to_string(index=False), flush=True)

    import torch

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    print(f"training device: {device}; circuit ansatz: {args.ansatz}", flush=True)
    train_cfg = TrainConfig(
        epochs=5 if args.pilot else args.epochs,
        batch_size=args.batch_size,
        max_train_samples=10000 if args.pilot else args.max_train,
        patience=args.patience,
        restore_best=not args.final_weights,
        device=device,
    )
    seeds = args.seeds[:2] if args.pilot else args.seeds

    tag = args.tag or "-".join(args.models)
    rows_path = out_dir / f"fold_metrics_{tag}.csv"
    returns_dir = out_dir / "returns"
    returns_dir.mkdir(exist_ok=True)
    done = set()
    for prior_path in sorted(out_dir.glob("fold_metrics*.csv")):
        prior = pd.read_csv(prior_path)
        done |= set(zip(prior["model"], prior["fold"], prior["seed"]))
    if done:
        print(f"resuming: {len(done)} (model, fold, seed) cells already complete", flush=True)

    # Trial count for the Deflated Sharpe Ratio must cover the whole study, not
    # just this job's slice, or splitting the sweep across jobs would silently
    # weaken the correction. Recomputed for the final tables in aggregate.py.
    n_trials = args.n_trials or (len(args.models) * len(seeds))

    for fold in folds:
        wait_for_cpu(max_load_per_core=args.max_load, max_wait_seconds=args.load_wait)
        pending = [(m, s_) for m in args.models for s_ in seeds
                   if (m, fold.fold, s_) not in done]
        if not pending:
            continue
        fold_data = prepare_fold(fold, X, frame, per_side=args.per_side)
        print(f"  {fold_data.describe()}", flush=True)
        for model_name in args.models:
            for seed in seeds:
                key = (model_name, fold.fold, seed)
                if key in done:
                    continue
                t0 = time.perf_counter()
                try:
                    result = run_fold(
                        model_name, fold_data, seed=seed,
                        train_cfg=train_cfg, per_side=args.per_side,
                        model_kwargs=({"ansatz": args.ansatz} if args.ansatz != "qcnn"
                                      and model_name.startswith("qtcnn") else None),
                    )
                except Exception as exc:  # keep the sweep alive; record the failure
                    print(f"  FAILED {model_name} fold{fold.fold} seed{seed}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    continue

                row = result.metrics(n_trials=n_trials)
                row["wall_seconds"] = time.perf_counter() - t0
                pd.DataFrame([row]).to_csv(
                    rows_path, mode="a", header=not rows_path.exists(), index=False
                )
                result.daily_returns.rename("ls_return").to_csv(
                    returns_dir / f"{model_name}_f{fold.fold}_s{seed}.csv",
                    header=True, index_label="Date",
                )
                if args.save_state and result.state is not None:
                    sdir = Path(args.save_state); sdir.mkdir(parents=True, exist_ok=True)
                    torch.save(result.state, sdir / f"{model_name}_f{fold.fold}_s{seed}.pt")
                del result
                print(
                    f"  {model_name:18s} fold{fold.fold} seed{seed}  "
                    f"sharpe={row.get('sharpe_ann', float('nan')):7.3f}  "
                    f"IC={row.get('ic_mean', float('nan')):+.5f}  "
                    f"auc={row.get('best_valid_auc', float('nan')):.4f}  "
                    f"{row['wall_seconds']:.0f}s",
                    flush=True,
                )

        del fold_data

    print(f"\ndone -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()

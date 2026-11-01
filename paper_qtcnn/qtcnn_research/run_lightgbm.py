"""Gradient-boosted classical baseline on the same walk-forward protocol.

The neural baselines (TCN, LSTM, Transformer, MLP) answer whether a sequence
model somebody would deploy beats the hybrid. A gradient-boosted tree on the
flattened window is the model that actually wins cross-sectional equity
competitions, including the one this dataset comes from, so its absence would
be the first thing a finance reader notices. It runs through the same folds,
labels, subsample, scoring and portfolio rule as every other condition; only
the fit call differs. The seed controls the training subsample and the
booster's own randomness, as it does for the networks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from qtcnn_research.resources import claim_cpu, describe_environment, hide_gpu, wait_for_cpu


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache/features.parquet")
    p.add_argument("--seq-cache", default="cache/sequences")
    p.add_argument("--out", default="results/walkforward")
    p.add_argument("--tag", default="lightgbm")
    p.add_argument("--model-name", default="lightgbm")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--per-side", type=int, default=200)
    p.add_argument("--train-days", type=int, default=500)
    p.add_argument("--valid-days", type=int, default=60)
    p.add_argument("--test-days", type=int, default=125)
    p.add_argument("--embargo-days", type=int, default=20)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--max-train", type=int, default=60000)
    p.add_argument("--n-estimators", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--min-child-samples", type=int, default=200)
    p.add_argument("--early-stopping", type=int, default=100,
                   help="rounds without validation-AUC improvement before stopping; "
                        "the booster's analogue of the networks' patience")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--nice", type=int, default=15)
    p.add_argument("--max-load", type=float, default=1.5)
    p.add_argument("--load-wait", type=int, default=1800)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--pilot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    claim_cpu(args.threads, niceness=args.nice)
    hide_gpu()

    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    from qtcnn_research import data as D
    from qtcnn_research.experiment import (
        FoldResult, build_index_frame, cross_sectional_score, long_short_returns, prepare_fold,
    )
    from qtcnn_research.run_experiment import build_sequence_cache
    from qtcnn_research.walkforward import Fold, make_walk_forward_folds, summarise_folds

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading feature cache ...", flush=True)
    df = pd.read_parquet(args.cache)
    feature_cols = D.FeatureSpec().columns()
    df["tradable"] = D.tradable_mask(df)
    X, index = build_sequence_cache(df, feature_cols, args.seq_len, Path(args.seq_cache))
    meta = df[["Date", "SecuritiesCode", "Target", "tradable"]]
    frame = build_index_frame(index, meta)
    all_dates = np.sort(df["Date"].unique())
    del df, meta

    dev_dates = all_dates[all_dates < D.HOLDOUT_START]
    if args.holdout:
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

    seeds = args.seeds[:2] if args.pilot else args.seeds
    max_train = 10000 if args.pilot else args.max_train
    n_estimators = 200 if args.pilot else args.n_estimators
    rows_path = out_dir / f"fold_metrics_{args.tag}.csv"
    returns_dir = out_dir / "returns"
    returns_dir.mkdir(exist_ok=True)
    done = set()
    for prior_path in sorted(out_dir.glob("fold_metrics*.csv")):
        prior = pd.read_csv(prior_path)
        done |= set(zip(prior["model"], prior["fold"], prior["seed"]))
    n_trials = args.n_trials or len(seeds)

    for fold in folds:
        pending = [s for s in seeds if (args.model_name, fold.fold, s) not in done]
        if not pending:
            continue
        wait_for_cpu(max_load_per_core=args.max_load, max_wait_seconds=args.load_wait)
        data = prepare_fold(fold, X, frame, per_side=args.per_side)
        print(f"  {data.describe()}", flush=True)
        Xtr = np.asarray(data.X_train).reshape(len(data.X_train), -1)
        Xva = np.asarray(data.X_valid).reshape(len(data.X_valid), -1)
        Xte = np.asarray(data.X_test).reshape(len(data.X_test), -1)
        for seed in pending:
            t0 = time.perf_counter()
            rng = np.random.default_rng(seed)
            if len(Xtr) > max_train:
                pick = np.sort(rng.choice(len(Xtr), max_train, replace=False))
                xtr, ytr = Xtr[pick], data.y_train[pick]
            else:
                xtr, ytr = Xtr, data.y_train
            booster = lgb.LGBMClassifier(
                objective="binary", n_estimators=n_estimators, learning_rate=args.learning_rate,
                num_leaves=args.num_leaves, min_child_samples=args.min_child_samples,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=args.threads, random_state=seed, verbose=-1,
            )
            booster.fit(xtr, ytr, eval_set=[(Xva, data.y_valid)], eval_metric="auc",
                        callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)])
            best_iter = int(booster.best_iteration_ or n_estimators)
            valid_auc = float(roc_auc_score(data.y_valid, booster.predict_proba(Xva)[:, 1]))
            train_seconds = time.perf_counter() - t0

            scored = data.test_meta.copy()
            scored["raw_score"] = booster.predict_proba(Xte)[:, 1]
            scored = cross_sectional_score(scored)
            daily, selection = long_short_returns(scored, top_n=args.per_side, bottom_n=args.per_side)
            result = FoldResult(
                model=args.model_name, fold=fold.fold, seed=seed, daily_returns=daily,
                scored=scored[["Date", "SecuritiesCode", "score", "Target"]], selection=selection,
                train_info={
                    # leaves across the kept trees, the closest analogue of a parameter count
                    "n_params": best_iter * args.num_leaves,
                    "train_seconds": train_seconds, "best_valid_auc": valid_auc,
                    "epochs_run": best_iter, "n_train": int(len(xtr)),
                },
            )
            row = result.metrics(n_trials=n_trials)
            row["wall_seconds"] = time.perf_counter() - t0
            pd.DataFrame([row]).to_csv(rows_path, mode="a", header=not rows_path.exists(), index=False)
            result.daily_returns.rename("ls_return").to_csv(
                returns_dir / f"{args.model_name}_f{fold.fold}_s{seed}.csv", header=True, index_label="Date")
            print(f"  {args.model_name:18s} fold{fold.fold} seed{seed}  sharpe={row.get('sharpe_ann', float('nan')):7.3f}  "
                  f"IC={row.get('ic_mean', float('nan')):+.5f}  auc={valid_auc:.4f}  trees={best_iter}  "
                  f"{row['wall_seconds']:.0f}s", flush=True)
        del data
    print(f"\ndone -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()

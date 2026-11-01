"""Training and walk-forward orchestration shared by every model in the study.

One code path trains all conditions. The quantum ablations, the classical
surrogates and the baselines differ only in the model object handed to
:func:`train_model`, which keeps the comparison honest: identical labels,
identical optimiser, identical early-stopping rule, identical portfolio
construction.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from qtcnn_research import evaluation as ev
from qtcnn_research.models import build_model, count_parameters
from qtcnn_research.walkforward import Fold


# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------
def extreme_labels(
    df: pd.DataFrame, per_side: int = 200, target_col: str = "Target"
) -> pd.DataFrame:
    """Label the top and bottom ``per_side`` names of each cross-section.

    The task is ranking, not level prediction, so training on the tails gives a
    cleaner decision boundary than regressing the noisy middle of the
    distribution.
    """
    def _pick(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < 2 * per_side:
            return g.iloc[0:0]
        g = g.sort_values(target_col)
        return pd.concat(
            [g.head(per_side).assign(label=0.0), g.tail(per_side).assign(label=1.0)]
        )

    return df.groupby("Date", group_keys=False).apply(_pick)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6            # early stopping on validation AUC
    # Restore the weights of the best validation epoch when training ends. The
    # full-budget control sets this False so the model scored is the one the
    # last epoch produced; with it True, a larger patience changes nothing
    # unless a later epoch beats the best one.
    restore_best: bool = True
    max_train_samples: int = 60_000
    grad_clip: float = 1.0
    device: str = "cpu"
    verbose: bool = False


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC without a sklearn import in the inner loop."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
    cfg: TrainConfig = TrainConfig(),
    seed: int = 0,
) -> Dict[str, object]:
    """Train with early stopping on validation AUC; restore the best weights."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    if len(X_train) > cfg.max_train_samples:
        pick = rng.choice(len(X_train), cfg.max_train_samples, replace=False)
        X_train, y_train = X_train[pick], y_train[pick]

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    loss_fn = nn.BCEWithLogitsLoss()

    history: List[Dict[str, float]] = []
    # Snapshot the circuit parameters so the run can report how far training
    # actually moved them. A frozen-equals-trained result is only informative if
    # the trained circuit moved at all, so this displacement is reported.
    q0 = {k: v.detach().clone() for k, v in model.state_dict().items() if "quantum" in k}
    best_auc, best_state, best_epoch, stale = -np.inf, None, -1, 0
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        sched.step()

        row = {"epoch": epoch, "train_loss": total / max(n, 1)}
        if X_valid is not None and len(X_valid):
            scores = predict(model, X_valid, cfg.batch_size, device)
            row["valid_auc"] = _auc(y_valid, scores)
            if row["valid_auc"] > best_auc:
                best_auc, best_epoch, stale = row["valid_auc"], epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if cfg.verbose:
            print(f"  epoch {epoch}: {row}")
        if stale >= cfg.patience:
            break

    if best_state is not None and cfg.restore_best:
        model.load_state_dict(best_state)
    model.eval()

    return {
        "history": history,
        "best_valid_auc": float(best_auc) if np.isfinite(best_auc) else float("nan"),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "circuit_shift_l2": float(
            sum(float(((model.state_dict()[k] - v) ** 2).sum()) for k, v in q0.items()) ** 0.5
        ) if q0 else float("nan"),
        "circuit_norm_l2": float(
            sum(float((v ** 2).sum()) for v in q0.values()) ** 0.5
        ) if q0 else float("nan"),
        "train_seconds": time.perf_counter() - t0,
        "n_train": int(len(X_train)),
        "n_params": count_parameters(model),
    }


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 2048,
            device: torch.device = torch.device("cpu")) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device)
        out.append(model(xb).float().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.empty(0, np.float32)


# ----------------------------------------------------------------------------
# Portfolio construction
# ----------------------------------------------------------------------------
def cross_sectional_score(scored: pd.DataFrame, col: str = "raw_score") -> pd.DataFrame:
    """Z-score model output within each date so ranks are comparable across days."""
    scored = scored.copy()
    grp = scored.groupby("Date")[col]
    scored["score"] = ((scored[col] - grp.transform("mean")) / (grp.transform("std") + 1e-8))
    return scored


def long_short_returns(
    scored: pd.DataFrame,
    top_n: int = 200,
    bottom_n: int = 200,
    target_col: str = "Target",
) -> Tuple[pd.Series, pd.DataFrame]:
    """Equal-weighted long/short book rebalanced every day.

    Returns the daily spread and the selection frame needed for turnover.
    """
    rows, daily = [], {}
    for date, g in scored.groupby("Date", sort=True):
        g = g[np.isfinite(g["score"]) & g[target_col].notna()]
        if len(g) < top_n + bottom_n:
            continue
        g = g.sort_values("score", ascending=False)
        long, short = g.head(top_n), g.tail(bottom_n)
        daily[date] = float(long[target_col].mean() - short[target_col].mean())
        rows.append(long.assign(Side="long"))
        rows.append(short.assign(Side="short"))
    if not rows:
        return pd.Series(dtype=float), pd.DataFrame()
    selection = pd.concat(rows, ignore_index=True)[["Date", "SecuritiesCode", "Side", "score"]]
    return pd.Series(daily).sort_index(), selection


# ----------------------------------------------------------------------------
# One fold, one model, one seed
# ----------------------------------------------------------------------------
@dataclass
class FoldResult:
    """Everything one (model, fold, seed) cell produced, plus its metric block."""

    model: str
    fold: int
    seed: int
    daily_returns: pd.Series
    scored: pd.DataFrame
    selection: pd.DataFrame
    train_info: Dict[str, object]
    state: Optional[Dict[str, torch.Tensor]] = None  # trained weights, for post-hoc diagnostics

    def metrics(self, n_trials: int = 1) -> Dict[str, object]:
        report = ev.evaluate_strategy(
            self.model, self.daily_returns, scored=self.scored,
            selection=self.selection, n_trials=n_trials, seed=self.seed, bootstrap=500,
        )
        row = report.to_row()
        row.update({
            "fold": self.fold,
            "seed": self.seed,
            "n_params": self.train_info.get("n_params"),
            "train_seconds": self.train_info.get("train_seconds"),
            "best_valid_auc": self.train_info.get("best_valid_auc"),
            "epochs_run": self.train_info.get("epochs_run"),
            "circuit_shift_l2": self.train_info.get("circuit_shift_l2"),
            "circuit_norm_l2": self.train_info.get("circuit_norm_l2"),
        })
        return row


@dataclass
class FoldData:
    """Materialised tensors for one fold, shared by every model and seed in it.

    Preparing these inside ``run_fold`` cost roughly 570 s per cell, dominated by
    re-merging the 2.4M-row index against the metadata and by gathering
    scattered rows out of the on-disk sequence memmap. Both are identical across
    the models and seeds evaluated on a fold, so they are hoisted here and paid
    once per fold instead of once per cell.
    """

    fold: Fold
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    test_meta: pd.DataFrame
    prep_seconds: float

    def describe(self) -> str:
        return (f"fold {self.fold.fold}: train {len(self.X_train)} valid {len(self.X_valid)} "
                f"test {len(self.X_test)} (prep {self.prep_seconds:.0f}s)")


def build_index_frame(index: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Join sequence positions to their Target and tradability, once per run."""
    frame = index.merge(meta, on=["Date", "SecuritiesCode"], how="left")
    frame["row"] = np.arange(len(frame), dtype=np.int64)
    return frame


def prepare_fold(
    fold: Fold,
    X: np.ndarray,
    frame: pd.DataFrame,
    per_side: int = 200,
) -> FoldData:
    """Gather the fold's train/valid/test tensors from the sequence store.

    Row indices are sorted before gathering: the sequence file is laid out by
    security then date, so a fold's rows are scattered through it, and a sorted
    gather turns random reads into a forward scan.
    """
    t0 = time.perf_counter()
    dates = frame["Date"]

    def _labelled(mask: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        sub = frame[mask]
        if sub.empty:
            return np.empty((0, X.shape[1], X.shape[2]), np.float32), np.empty(0, np.float32)
        labelled = extreme_labels(sub, per_side=per_side)
        if labelled.empty:
            return np.empty((0, X.shape[1], X.shape[2]), np.float32), np.empty(0, np.float32)
        labelled = labelled.sort_values("row")
        rows = labelled["row"].to_numpy()
        return np.asarray(X[rows]), labelled["label"].to_numpy(np.float32)

    X_train, y_train = _labelled(fold.train_mask(dates))
    X_valid, y_valid = _labelled(fold.valid_mask(dates))

    test = frame[fold.test_mask(dates) & frame["tradable"]].sort_values("row").copy()
    X_test = np.asarray(X[test["row"].to_numpy()])

    return FoldData(
        fold=fold, X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid, X_test=X_test,
        test_meta=test[["Date", "SecuritiesCode", "Target"]].reset_index(drop=True),
        prep_seconds=time.perf_counter() - t0,
    )


def run_fold(
    model_name: str,
    data: FoldData,
    seed: int = 0,
    train_cfg: TrainConfig = TrainConfig(),
    per_side: int = 200,
    model_kwargs: Optional[Dict] = None,
) -> FoldResult:
    """Train on the fold's prepared tensors and score its test window."""
    model_kwargs = model_kwargs or {}
    if len(data.X_train) == 0:
        raise RuntimeError(f"fold {data.fold.fold} has no training samples")

    model = build_model(model_name, n_features=data.X_train.shape[2],
                        seq_len=data.X_train.shape[1], seed=seed, **model_kwargs)
    info = train_model(
        model, data.X_train, data.y_train,
        data.X_valid if len(data.X_valid) else None,
        data.y_valid if len(data.y_valid) else None,
        cfg=train_cfg, seed=seed,
    )

    scored = data.test_meta.copy()
    scored["raw_score"] = predict(model, data.X_test, max(train_cfg.batch_size, 2048),
                                  torch.device(train_cfg.device))
    scored = cross_sectional_score(scored)
    daily, selection = long_short_returns(scored, top_n=per_side, bottom_n=per_side)

    return FoldResult(
        model=model_name, fold=data.fold.fold, seed=seed, daily_returns=daily,
        scored=scored[["Date", "SecuritiesCode", "score", "Target"]],
        selection=selection, train_info=info,
        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    )

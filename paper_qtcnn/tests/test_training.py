"""The two training regimes the paper reports: early stopping that restores the
best-validation weights, and the full-budget control that scores the last epoch.

The full-budget control is only a control if it actually differs from the
early-stopped run; with best-weight restoration left on, a larger patience
reproduces the early-stopped model whenever no later epoch beats the best one.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from qtcnn_research.experiment import TrainConfig, train_model


def _data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5, 3)).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(np.float32)   # pure noise: validation AUC peaks early
    return X, y


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(15, 16), nn.ReLU(), nn.Linear(16, 1), nn.Flatten(0))


def _run(restore_best: bool):
    Xt, yt = _data(400, 0)
    Xv, yv = _data(200, 1)
    model = _model()
    cfg = TrainConfig(epochs=12, batch_size=64, patience=12, lr=1e-2, restore_best=restore_best)
    info = train_model(model, Xt, yt, Xv, yv, cfg=cfg, seed=0)
    return model, info


def test_full_budget_runs_every_epoch():
    _, info = _run(restore_best=False)
    assert info["epochs_run"] == 12


def test_last_epoch_weights_differ_from_best_epoch_weights():
    best, info = _run(restore_best=True)
    last, _ = _run(restore_best=False)
    assert info["best_epoch"] < 11, "fixture should peak before the last epoch"
    diff = sum(float((a - b).abs().sum()) for a, b in zip(best.state_dict().values(),
                                                           last.state_dict().values()))
    assert diff > 0


def test_training_is_deterministic_for_a_fixed_seed():
    a, _ = _run(restore_best=False)
    b, _ = _run(restore_best=False)
    for x, y in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(x, y)

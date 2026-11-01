"""Rolling-origin evaluation protocol.

The legacy pipelines used one 80/20 date split and, on the downsampled subsets,
ended up reporting Sharpe ratios computed on 22 or even 2 out-of-sample days.
This module replaces that with anchored or rolling walk-forward folds, an
embargo between train and test to stop feature windows leaking across the
boundary, and a locked final holdout that model selection never sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One train/validation/test date range, expressed in trading days."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: Optional[pd.Timestamp]
    valid_end: Optional[pd.Timestamp]
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> str:
        val = (
            f" valid {self.valid_start.date()}..{self.valid_end.date()}"
            if self.valid_start is not None
            else ""
        )
        return (
            f"fold {self.fold}: train {self.train_start.date()}..{self.train_end.date()}"
            f"{val} test {self.test_start.date()}..{self.test_end.date()}"
        )

    def train_mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.train_start) & (dates <= self.train_end)

    def valid_mask(self, dates: pd.Series) -> pd.Series:
        if self.valid_start is None:
            return pd.Series(False, index=dates.index)
        return (dates >= self.valid_start) & (dates <= self.valid_end)

    def test_mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.test_start) & (dates <= self.test_end)


def make_walk_forward_folds(
    dates: Sequence[pd.Timestamp],
    train_days: int = 500,
    test_days: int = 125,
    valid_days: int = 60,
    embargo_days: int = 20,
    step_days: Optional[int] = None,
    anchored: bool = False,
    max_folds: Optional[int] = None,
) -> List[Fold]:
    """Split a sorted date index into successive train/valid/test blocks.

    ``embargo_days`` drops the days immediately before each test block from
    training. Features use rolling windows up to 20 days long, so without an
    embargo the last training rows overlap the first test rows' input windows.

    ``anchored=True`` grows the training window from a fixed start (more data
    later on); the default rolling window keeps the training size constant, which
    isolates regime effects from sample-size effects.
    """
    unique = pd.DatetimeIndex(pd.Series(pd.to_datetime(list(dates))).drop_duplicates().sort_values())
    n = len(unique)
    step = step_days or test_days
    folds: List[Fold] = []

    # First test block starts once there is enough history for train+valid+embargo.
    min_history = train_days + valid_days + embargo_days
    start = min_history
    fold_id = 0
    while start + test_days <= n:
        test_lo, test_hi = start, start + test_days - 1
        embargo_lo = test_lo - embargo_days
        valid_hi = embargo_lo - 1
        valid_lo = valid_hi - valid_days + 1
        train_hi = valid_lo - 1
        train_lo = 0 if anchored else max(0, train_hi - train_days + 1)
        if train_hi < train_lo:
            break

        folds.append(
            Fold(
                fold=fold_id,
                train_start=unique[train_lo],
                train_end=unique[train_hi],
                valid_start=unique[valid_lo] if valid_days > 0 else None,
                valid_end=unique[valid_hi] if valid_days > 0 else None,
                test_start=unique[test_lo],
                test_end=unique[test_hi],
            )
        )
        fold_id += 1
        start += step
        if max_folds is not None and len(folds) >= max_folds:
            break

    return folds


def split_dev_holdout(
    dates: Sequence[pd.Timestamp], holdout_start: pd.Timestamp
) -> tuple:
    """Separate the development period from the locked final holdout."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).drop_duplicates().sort_values()
    return idx[idx < holdout_start], idx[idx >= holdout_start]


def iter_folds(df: pd.DataFrame, folds: Sequence[Fold], date_col: str = "Date") -> Iterator:
    """Yield ``(fold, train_df, valid_df, test_df)`` views for each fold."""
    dates = df[date_col]
    for fold in folds:
        yield (
            fold,
            df[fold.train_mask(dates)],
            df[fold.valid_mask(dates)],
            df[fold.test_mask(dates)],
        )


def summarise_folds(folds: Sequence[Fold]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": f.fold,
                "train_start": f.train_start.date(),
                "train_end": f.train_end.date(),
                "test_start": f.test_start.date(),
                "test_end": f.test_end.date(),
            }
            for f in folds
        ]
    )


def pool_fold_returns(fold_returns: Sequence[pd.Series]) -> pd.Series:
    """Concatenate per-fold out-of-sample return series into one continuous track.

    Walk-forward test blocks are disjoint by construction, so the pooled series
    is a genuine one-pass backtest rather than an average of overlapping runs.
    """
    if not fold_returns:
        return pd.Series(dtype=float)
    pooled = pd.concat(list(fold_returns)).sort_index()
    return pooled[~pooled.index.duplicated(keep="first")]

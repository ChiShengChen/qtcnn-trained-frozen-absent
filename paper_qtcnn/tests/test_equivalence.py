"""Calibration of the equivalence machinery: it must declare equivalence when two
series really are the same, and must not when they differ by more than the margin."""
import numpy as np
import pandas as pd
import pytest

from qtcnn_research.equivalence import minimum_detectable_effect, tost


def _series(n, mu, sd, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_identical_series_are_declared_equivalent():
    a = _series(500, 0.0002, 0.01, 0)
    r = tost(a, a.copy(), margin=0.10, n_boot=400, seed=1)
    assert r.diff_sharpe == pytest.approx(0.0, abs=1e-9)
    assert r.equivalent


def test_clearly_different_series_are_not_declared_equivalent():
    a = _series(500, 0.0030, 0.01, 2)   # strongly positive
    b = _series(500, -0.0010, 0.01, 3)  # strongly negative
    r = tost(a, b, margin=0.25, n_boot=400, seed=4)
    assert abs(r.diff_sharpe) > 1.0
    assert not r.equivalent


def test_wide_margin_accepts_a_small_difference_a_narrow_one_does_not():
    a = _series(800, 0.0003, 0.01, 5)
    b = _series(800, 0.00025, 0.01, 5)   # same draws, tiny mean shift
    narrow = tost(a, b, margin=0.01, n_boot=400, seed=6)
    wide = tost(a, b, margin=2.0, n_boot=400, seed=6)
    assert wide.equivalent
    assert not narrow.equivalent


def test_minimum_detectable_effect_scales_with_the_standard_error():
    assert minimum_detectable_effect(0.5, 0.80) == pytest.approx(2 * minimum_detectable_effect(0.25, 0.80))
    assert minimum_detectable_effect(0.5, 0.90) > minimum_detectable_effect(0.5, 0.80)
    assert np.isnan(minimum_detectable_effect(0.0))

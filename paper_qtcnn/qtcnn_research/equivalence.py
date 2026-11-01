"""Equivalence tests and minimum detectable effects for the ablation contrasts.

A non-significant difference is not evidence of no difference. This module supplies
the two statistics the manuscript needs to say which of its nulls are established
and which are merely underpowered:

* ``minimum_detectable_effect`` -- the smallest true difference the design could
  have found at a given power, from the paired bootstrap standard error.
* ``tost`` -- two one-sided tests against an equivalence margin. A significant
  TOST is positive evidence that the difference lies inside the margin.

Both operate on the paired per-day difference of two pooled tracks, which is the
right unit here because seeds and folds are matched across conditions by
construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import evaluation as ev


@dataclass(frozen=True)
class EquivalenceResult:
    model: str
    reference: str
    n_days: int
    diff_sharpe: float
    se: float
    ci_low: float
    ci_high: float
    margin: float
    p_tost: float
    equivalent: bool
    mde80: float
    mde90: float


def _sharpe_diff(a: pd.Series, b: pd.Series, periods: int = ev.TRADING_DAYS) -> float:
    return ev.annualised_sharpe(a, periods) - ev.annualised_sharpe(b, periods)


def paired_bootstrap_sharpe_diff(
    a: pd.Series, b: pd.Series, n_boot: int = 2000, block: int = 10,
    seed: int = 0, periods: int = ev.TRADING_DAYS,
) -> np.ndarray:
    """Stationary-bootstrap draws of the Sharpe difference, resampling days jointly.

    Resampling the two series with the same index preserves their pairing, so the
    draws describe the difference rather than the sum of two independent errors.
    """
    a = pd.Series(a, dtype=float).dropna()
    b = pd.Series(b, dtype=float).reindex(a.index).dropna()
    a = a.reindex(b.index)
    n = len(a)
    if n < 20:
        return np.array([])
    rng = np.random.default_rng(seed)
    av, bv = a.to_numpy(), b.to_numpy()
    p = 1.0 / block
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.empty(n, dtype=int)
        j = rng.integers(n)
        for t in range(n):
            idx[t] = j
            j = rng.integers(n) if rng.random() < p else (j + 1) % n
        out[i] = _sharpe_diff(pd.Series(av[idx]), pd.Series(bv[idx]), periods)
    return out


def minimum_detectable_effect(se: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest true effect a two-sided test at this standard error would detect."""
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)


def tost(
    a: pd.Series, b: pd.Series, margin: float, n_boot: int = 2000, block: int = 10,
    seed: int = 0, alpha: float = 0.05, model: str = "", reference: str = "",
) -> EquivalenceResult:
    """Two one-sided tests that |Sharpe(a) - Sharpe(b)| < margin.

    The p-value is the larger of the two one-sided bootstrap tail probabilities;
    equivalence is declared when it falls below alpha, i.e. when the whole
    (1 - 2*alpha) interval lies inside the margin.
    """
    draws = paired_bootstrap_sharpe_diff(a, b, n_boot=n_boot, block=block, seed=seed)
    obs = _sharpe_diff(pd.Series(a, dtype=float).dropna(), pd.Series(b, dtype=float).dropna())
    if draws.size == 0:
        nan = float("nan")
        return EquivalenceResult(model, reference, 0, obs, nan, nan, nan, margin, nan, False, nan, nan)
    se = float(draws.std(ddof=1))
    centred = draws - draws.mean()
    # H0_lower: diff <= -margin, rejected for large obs; the null distribution is
    # the centred draws shifted to -margin, so p = P(centred - margin >= obs).
    # H0_upper: diff >= +margin, rejected for small obs, p = P(centred + margin <= obs).
    p_lower = float(np.mean(centred >= obs + margin))
    p_upper = float(np.mean(centred <= obs - margin))
    p = max(p_lower, p_upper)
    lo, hi = np.quantile(draws, [alpha, 1 - alpha])
    return EquivalenceResult(
        model=model, reference=reference, n_days=int(min(len(pd.Series(a).dropna()), len(pd.Series(b).dropna()))),
        diff_sharpe=obs, se=se, ci_low=float(lo), ci_high=float(hi), margin=margin,
        p_tost=p, equivalent=bool(p < alpha), mde80=minimum_detectable_effect(se, 0.80),
        mde90=minimum_detectable_effect(se, 0.90),
    )


def equivalence_table(
    returns: pd.DataFrame, reference: str, margins: Sequence[float] = (0.10, 0.25, 0.50),
    models: Optional[Sequence[str]] = None, n_boot: int = 2000, seed: int = 0,
) -> pd.DataFrame:
    """TOST of every model against ``reference`` at each margin, plus its MDE."""
    from .aggregate import pooled_track

    ref = pooled_track(returns, reference)
    names = list(models) if models is not None else sorted(returns["model"].unique())
    rows = []
    for m in names:
        if m == reference:
            continue
        track = pooled_track(returns, m)
        common = track.index.intersection(ref.index)
        for margin in margins:
            r = tost(track.reindex(common), ref.reindex(common), margin=margin,
                     n_boot=n_boot, seed=seed, model=m, reference=reference)
            rows.append(r.__dict__)
    return pd.DataFrame(rows)

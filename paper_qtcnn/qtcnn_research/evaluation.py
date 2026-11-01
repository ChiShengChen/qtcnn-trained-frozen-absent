"""Statistically defensible evaluation of cross-sectional stock-ranking models.

The pipelines in ``scripts/`` reported a non-annualised ``mean/std`` Sharpe on a
22-day out-of-sample window with a single seed. Nothing in that setup can
support a claim of one model beating another, so this module supplies the
machinery a finance referee expects:

  * annualised Sharpe/Sortino/Calmar and drawdown;
  * rank IC with Newey-West standard errors (IC series are autocorrelated);
  * stationary-bootstrap confidence intervals for the Sharpe ratio;
  * the Deflated Sharpe Ratio of Bailey and Lopez de Prado, which discounts the
    number of configurations tried;
  * probability of backtest overfitting via combinatorially symmetric CV;
  * Diebold-Mariano tests for pairwise model comparison;
  * turnover and a break-even transaction-cost curve.

Every function takes plain numpy/pandas and is independent of the model, so the
same code scores the quantum model, the classical surrogates and the baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# Return-series statistics
# ----------------------------------------------------------------------------
def annualised_sharpe(returns: Sequence[float], periods: int = TRADING_DAYS) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods))


def annualised_sortino(returns: Sequence[float], periods: int = TRADING_DAYS) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return float("nan")
    downside = r[r < 0]
    dd = np.sqrt((downside**2).mean()) if len(downside) else 0.0
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    return float(r.mean() / dd * np.sqrt(periods))


def max_drawdown(returns: Sequence[float]) -> float:
    """Peak-to-trough drawdown of the compounded equity curve (a negative number).

    A drawdown cannot be worse than -1: the compounded ratio stops being
    interpretable once the equity curve crosses zero, which one return of -100%
    or worse is enough to cause. Rather than print an impossible number, return
    -1 and let the caller see that the book was wiped out. The guard fires only
    on data errors for a dollar-neutral book; it fired on an unadjusted
    bankruptcy-reorganisation price in an early version of the US panel.
    """
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    if (equity <= 0).any():
        return -1.0
    return float(max(-1.0, (equity / equity.cummax() - 1.0).min()))


def calmar(returns: Sequence[float], periods: int = TRADING_DAYS) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return float("nan")
    ann_return = float((1.0 + r).prod() ** (periods / len(r)) - 1.0)
    mdd = max_drawdown(r)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return ann_return / abs(mdd)


def annualised_return(returns: Sequence[float], periods: int = TRADING_DAYS) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0  # the book was wiped out; a fractional power of a negative is not a return
    return float(growth ** (periods / len(r)) - 1.0)


def hit_rate(returns: Sequence[float]) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


# ----------------------------------------------------------------------------
# Newey-West inference
# ----------------------------------------------------------------------------
def _nw_lags(n: int) -> int:
    """Newey-West (1994) automatic lag rule, 4*(n/100)^(2/9)."""
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def newey_west_tstat(x: Sequence[float], lags: Optional[int] = None) -> Tuple[float, float, float]:
    """Mean, HAC standard error and t-statistic of a serially correlated series.

    Daily IC series carry meaningful autocorrelation, so an OLS t-stat overstates
    significance. Lag length defaults to the Newey-West rule 4*(n/100)^(2/9).
    """
    a = np.asarray(pd.Series(x, dtype=float).dropna(), dtype=float)
    n = len(a)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    if lags is None:
        lags = _nw_lags(n)
    lags = max(0, min(lags, n - 2))

    mu = a.mean()
    e = a - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for lag in range(1, lags + 1):
        cov = float(e[lag:] @ e[:-lag]) / n
        var += 2.0 * (1.0 - lag / (lags + 1.0)) * cov
    if var <= 0:
        return float(mu), float("nan"), float("nan")
    se = np.sqrt(var / n)
    return float(mu), float(se), float(mu / se)


# ----------------------------------------------------------------------------
# Information coefficient
# ----------------------------------------------------------------------------
def daily_ic(
    scored: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "Target",
    date_col: str = "Date",
    method: str = "spearman",
) -> pd.Series:
    """Cross-sectional score/return correlation per date."""

    def _corr(g: pd.DataFrame) -> float:
        x = g[score_col]
        y = g[target_col]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
            return np.nan
        return x[ok].corr(y[ok], method=method)

    # Select the columns before applying: passing the grouping column through
    # is deprecated in pandas 2.2 and unnecessary here.
    return scored.groupby(date_col, sort=True)[[score_col, target_col]].apply(_corr).dropna()


def ic_summary(ic: pd.Series, periods: int = TRADING_DAYS) -> Dict[str, float]:
    ic = pd.Series(ic, dtype=float).dropna()
    if ic.empty:
        return {"ic_mean": np.nan, "icir": np.nan, "ic_tstat_nw": np.nan, "ic_days": 0}
    mean, se, tstat = newey_west_tstat(ic)
    sd = ic.std(ddof=1)
    icir = float(mean / sd) if sd and np.isfinite(sd) else float("nan")
    return {
        "ic_mean": float(mean),
        "ic_std": float(sd),
        "icir": icir,
        "icir_ann": icir * np.sqrt(periods) if np.isfinite(icir) else np.nan,
        "ic_se_nw": se,
        "ic_tstat_nw": tstat,
        "ic_pvalue_nw": float(2 * (1 - stats.norm.cdf(abs(tstat)))) if np.isfinite(tstat) else np.nan,
        "ic_positive_rate": float((ic > 0).mean()),
        "ic_days": int(len(ic)),
    }


# ----------------------------------------------------------------------------
# Stationary bootstrap
# ----------------------------------------------------------------------------
def stationary_bootstrap_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric blocks preserve dependence."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=np.int64)
    cur = rng.integers(0, n)
    for i in range(n):
        idx[i] = cur
        if rng.random() < p:
            cur = rng.integers(0, n)
        else:
            cur = (cur + 1) % n
    return idx


def bootstrap_sharpe_ci(
    returns: Sequence[float],
    n_boot: int = 2000,
    mean_block: float = 10.0,
    alpha: float = 0.05,
    periods: int = TRADING_DAYS,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile CI and a two-sided p-value for H0: Sharpe = 0."""
    r = np.asarray(pd.Series(returns, dtype=float).dropna(), dtype=float)
    if len(r) < 20:
        return {"sharpe": annualised_sharpe(r, periods), "ci_low": np.nan,
                "ci_high": np.nan, "p_value": np.nan, "n_boot": 0}
    rng = np.random.default_rng(seed)
    point = annualised_sharpe(r, periods)
    draws = np.empty(n_boot)
    # The null is imposed on the returns, not on the Sharpe draws. Demeaning the
    # series makes its true Sharpe ratio zero, so resampling it gives the
    # sampling distribution of the statistic under H0 directly. Re-centring the
    # Sharpe draws instead assumes that distribution is symmetric, which fails
    # badly under skew: on a panel with skew -23 the two constructions differed
    # by a factor of seven and the p-value contradicted the interval beside it.
    null_r = r - r.mean()
    null_draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(len(r), mean_block, rng)
        draws[b] = annualised_sharpe(r[idx], periods)
        null_draws[b] = annualised_sharpe(null_r[idx], periods)
    draws = draws[np.isfinite(draws)]
    null_draws = null_draws[np.isfinite(null_draws)]
    if draws.size == 0 or null_draws.size == 0:
        return {"sharpe": point, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "n_boot": 0}
    p = float((np.abs(null_draws) >= abs(point)).mean())
    return {
        "sharpe": point,
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "p_value": p,
        "n_boot": int(draws.size),
    }


# ----------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
# ----------------------------------------------------------------------------
def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int,
    periods: int = TRADING_DAYS,
    trial_sharpe_std: Optional[float] = None,
) -> Dict[str, float]:
    """Probability that the observed Sharpe exceeds what selection bias alone gives.

    ``n_trials`` is the number of configurations searched before reporting this
    one; reporting DSR without it is exactly the overfitting a referee suspects.
    """
    r = np.asarray(pd.Series(returns, dtype=float).dropna(), dtype=float)
    n = len(r)
    if n < 20 or n_trials < 1:
        return {"dsr": np.nan, "sr_zero": np.nan, "skew": np.nan, "kurtosis": np.nan}

    sr = float(r.mean() / r.std(ddof=1))  # per-period, not annualised
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    # Expected maximum Sharpe across n_trials independent trials.
    if trial_sharpe_std is None:
        trial_sharpe_std = 1.0 / np.sqrt(n)
    if n_trials == 1:
        # A single trial carries no selection bias, and the expected-maximum
        # formula is undefined there (norm.ppf(0) = -inf).
        e_max = 0.0
    else:
        euler = 0.5772156649015329
        e_max = trial_sharpe_std * (
            (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials)
            + euler * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )

    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr**2)
    if not np.isfinite(denom) or denom <= 0:
        return {"dsr": np.nan, "sr_zero": float(e_max * np.sqrt(periods)),
                "skew": skew, "kurtosis": kurt}
    z = (sr - e_max) * np.sqrt(n - 1) / denom
    return {
        "dsr": float(stats.norm.cdf(z)),
        "trial_sd_ann": float(trial_sharpe_std * np.sqrt(periods)),
        "sr_observed_ann": sr * np.sqrt(periods),
        "sr_zero_ann": float(e_max * np.sqrt(periods)),
        "skew": skew,
        "kurtosis": kurt,
        "n_trials": int(n_trials),
    }


# ----------------------------------------------------------------------------
# Probability of Backtest Overfitting (CSCV)
# ----------------------------------------------------------------------------
def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame, n_splits: int = 10
) -> Dict[str, float]:
    """CSCV-based PBO over a matrix of candidate strategies (columns = configs).

    Splits the sample into ``n_splits`` blocks, and for every balanced in/out
    partition checks where the in-sample winner ranks out of sample. A PBO near
    0.5 means the selection procedure carries no information.
    """
    mat = returns_matrix.dropna(how="any")
    n_obs, n_cfg = mat.shape
    if n_cfg < 2 or n_obs < n_splits * 2:
        return {"pbo": np.nan, "n_partitions": 0}

    blocks = np.array_split(np.arange(n_obs), n_splits)
    half = n_splits // 2
    logits: List[float] = []
    for combo in combinations(range(n_splits), half):
        is_idx = np.concatenate([blocks[b] for b in combo])
        oos_idx = np.concatenate([blocks[b] for b in range(n_splits) if b not in combo])
        is_sr = mat.iloc[is_idx].apply(annualised_sharpe)
        oos_sr = mat.iloc[oos_idx].apply(annualised_sharpe)
        if is_sr.isna().all() or oos_sr.isna().all():
            continue
        best = is_sr.idxmax()
        # Relative rank of the in-sample winner within the OOS Sharpe ranking.
        # Divide by k+1 rather than k so that the median rank of an uninformative
        # selection rule sits at exactly 0.5.
        rank = oos_sr.rank(ascending=True).get(best, np.nan) / (len(oos_sr) + 1.0)
        if not np.isfinite(rank):
            continue
        rank = min(max(float(rank), 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))

    if not logits:
        return {"pbo": np.nan, "n_partitions": 0}
    return {
        "pbo": float(np.mean(np.asarray(logits) <= 0)),
        "logit_mean": float(np.mean(logits)),
        "n_partitions": len(logits),
    }


# ----------------------------------------------------------------------------
# Pairwise model comparison
# ----------------------------------------------------------------------------
def diebold_mariano(
    series_a: Sequence[float], series_b: Sequence[float], lags: Optional[int] = None,
    small_sample: bool = True,
) -> Dict[str, float]:
    """HAC test on the difference of two aligned performance series (e.g. daily IC).

    Positive statistic means ``series_a`` is the better of the two. This tests a
    difference in *mean*, which is not the same hypothesis as a difference in
    Sharpe ratio whenever the two series have different volatilities; use
    :func:`sharpe_difference_test` for the latter.

    ``small_sample`` applies the Harvey--Leybourne--Newbold correction and refers
    the statistic to a t distribution, which matters on windows of a few hundred
    days where the normal approximation over-rejects.
    """
    a = pd.Series(series_a, dtype=float)
    b = pd.Series(series_b, dtype=float)
    diff = (a - b).dropna()
    mean, se, tstat = newey_west_tstat(diff, lags=lags)
    n = int(len(diff))
    if small_sample and n > 2 and np.isfinite(tstat):
        h = max(1, int(lags if lags is not None else _nw_lags(n)))
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        tstat = float(tstat * np.sqrt(max(factor, 1e-9)))
        p = float(2 * (1 - stats.t.cdf(abs(tstat), df=n - 1)))
    else:
        p = float(2 * (1 - stats.norm.cdf(abs(tstat)))) if np.isfinite(tstat) else np.nan
    return {
        "mean_diff": mean,
        "se_nw": se,
        "dm_stat": tstat,
        "p_value": p,
        "n": n,
    }


def sharpe_difference_test(
    series_a: Sequence[float], series_b: Sequence[float], n_boot: int = 2000,
    mean_block: float = 10.0, seed: int = 0, periods: int = TRADING_DAYS,
) -> Dict[str, float]:
    """Paired stationary-bootstrap test of H0: Sharpe(a) = Sharpe(b).

    The headline metric of this study is the Sharpe ratio, and a difference in
    mean return is a different hypothesis: two tracks with the same mean and
    different volatilities have different Sharpe ratios and an identical
    Diebold--Mariano statistic. Days are resampled jointly so the pairing that
    the matched folds and seeds create is preserved, and the null is imposed by
    demeaning each series before resampling.
    """
    a = pd.Series(series_a, dtype=float)
    b = pd.Series(series_b, dtype=float).reindex(a.index)
    common = a.notna() & b.notna()
    av, bv = a[common].to_numpy(float), b[common].to_numpy(float)
    n = len(av)
    if n < 20:
        return {"diff": np.nan, "se": np.nan, "p_value": np.nan, "n": n}
    point = annualised_sharpe(av, periods) - annualised_sharpe(bv, periods)
    rng = np.random.default_rng(seed)
    a0, b0 = av - av.mean(), bv - bv.mean()
    draws = np.empty(n_boot)
    null = np.empty(n_boot)
    for i in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        draws[i] = annualised_sharpe(av[idx], periods) - annualised_sharpe(bv[idx], periods)
        null[i] = annualised_sharpe(a0[idx], periods) - annualised_sharpe(b0[idx], periods)
    draws, null = draws[np.isfinite(draws)], null[np.isfinite(null)]
    if draws.size == 0 or null.size == 0:
        return {"diff": point, "se": np.nan, "p_value": np.nan, "n": n}
    return {
        "diff": float(point),
        "se": float(draws.std(ddof=1)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_value": float((np.abs(null) >= abs(point)).mean()),
        "n": n,
    }


def romano_wolf_stepdown(
    candidate_series: Dict[str, Sequence[float]],
    benchmark: Sequence[float],
    n_boot: int = 1000,
    mean_block: float = 10.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Stepdown multiple-testing correction across many candidate models.

    With a dozen models on one dataset, uncorrected p-values are not credible;
    this controls the familywise error rate while staying far less conservative
    than Bonferroni.
    """
    bench = pd.Series(benchmark, dtype=float)
    frame = pd.DataFrame({k: pd.Series(v, dtype=float) for k, v in candidate_series.items()})
    frame = frame.sub(bench, axis=0).dropna(how="any")
    if frame.empty:
        return pd.DataFrame(columns=["model", "mean_diff", "t_stat", "p_rw"])

    n, k = frame.shape
    obs_t = np.array([newey_west_tstat(frame.iloc[:, j])[2] for j in range(k)])
    rng = np.random.default_rng(seed)
    centred = frame.values - frame.values.mean(axis=0, keepdims=True)

    boot_t = np.empty((n_boot, k))
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        sample = centred[idx]
        for j in range(k):
            boot_t[b, j] = newey_west_tstat(sample[:, j])[2]
    boot_t = np.nan_to_num(boot_t, nan=0.0)

    order = np.argsort(-np.abs(obs_t))
    p_rw = np.ones(k)
    active = list(order)
    prev_p = 0.0
    while active:
        j = active[0]
        max_abs = np.abs(boot_t[:, active]).max(axis=1)
        p = float((max_abs >= abs(obs_t[j])).mean())
        p = max(p, prev_p)  # enforce monotonicity down the stepdown sequence
        p_rw[j] = p
        prev_p = p
        active = active[1:]

    return pd.DataFrame(
        {
            "model": frame.columns,
            "mean_diff": frame.mean(axis=0).values,
            "t_stat": obs_t,
            "p_rw": p_rw,
        }
    ).sort_values("t_stat", ascending=False, ignore_index=True)


# ----------------------------------------------------------------------------
# Quantile portfolios
# ----------------------------------------------------------------------------
def quantile_portfolios(
    scored: pd.DataFrame,
    n_quantiles: int = 10,
    score_col: str = "score",
    target_col: str = "Target",
    date_col: str = "Date",
    periods: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Mean forward return of each score decile, and the spread between the ends.

    A positive information coefficient with a negative top-minus-bottom spread
    is not a contradiction: IC is a rank correlation over the whole
    cross-section, while the long/short book only holds the two tails. When they
    disagree the relationship is non-monotonic, and only a quantile table shows
    where the signal actually lives. Reviewers of factor research expect this
    table, and it is the diagnostic that explains the disagreement.
    """
    frame = scored[[date_col, score_col, target_col]].dropna()
    if frame.empty:
        return pd.DataFrame()

    def _bucket(g: pd.DataFrame) -> pd.Series:
        if len(g) < n_quantiles * 2:
            return pd.Series(dtype=float)
        # rank-based bucketing is robust to the score distribution's shape
        q = pd.qcut(g[score_col].rank(method="first"), n_quantiles,
                    labels=False, duplicates="drop")
        return g[target_col].groupby(q).mean()

    daily = frame.groupby(date_col, sort=True)[[score_col, target_col]].apply(_bucket)
    if isinstance(daily, pd.Series):
        daily = daily.unstack()
    daily = daily.dropna(how="all")
    if daily.empty:
        return pd.DataFrame()

    rows = []
    for q in daily.columns:
        series = daily[q].dropna()
        mean, se, tstat = newey_west_tstat(series)
        rows.append({
            "quantile": int(q) + 1,
            "mean_return": mean,
            "mean_return_ann": mean * periods,
            "t_stat_nw": tstat,
            "sharpe_ann": annualised_sharpe(series, periods),
            "n_days": int(len(series)),
        })
    table = pd.DataFrame(rows)

    lo, hi = daily.columns.min(), daily.columns.max()
    spread = (daily[hi] - daily[lo]).dropna()
    mean, se, tstat = newey_west_tstat(spread)
    # Spearman correlation between quantile index and mean return: 1.0 is a
    # perfectly monotone signal, near 0 means the tails carry it all.
    monotonicity = table["quantile"].corr(table["mean_return"], method="spearman")
    table.attrs["spread"] = {
        "mean_return": mean,
        "t_stat_nw": tstat,
        "sharpe_ann": annualised_sharpe(spread, periods),
        "monotonicity": float(monotonicity) if np.isfinite(monotonicity) else np.nan,
        "n_days": int(len(spread)),
    }
    return table


# ----------------------------------------------------------------------------
# Turnover and transaction costs
# ----------------------------------------------------------------------------
def daily_turnover(
    selection: pd.DataFrame,
    date_col: str = "Date",
    code_col: str = "SecuritiesCode",
    side_col: str = "Side",
) -> pd.Series:
    """Fraction of each leg replaced between consecutive rebalances."""
    turnovers: Dict[pd.Timestamp, float] = {}
    prev: Dict[str, set] = {}
    for date, g in selection.groupby(date_col, sort=True):
        legs = {}
        changed, total = 0.0, 0.0
        for side, gs in g.groupby(side_col):
            legs[side] = set(gs[code_col].tolist())
            if side in prev and prev[side]:
                changed += len(legs[side] - prev[side])
                total += len(legs[side])
        turnovers[date] = changed / total if total else np.nan
        prev = legs
    return pd.Series(turnovers).dropna()


def apply_transaction_costs(
    daily_returns: pd.Series, turnover: pd.Series, cost_bps: float
) -> pd.Series:
    """Charge ``cost_bps`` on the traded fraction of both legs each rebalance."""
    aligned = turnover.reindex(daily_returns.index).fillna(0.0)
    # Long and short legs each trade `turnover` of their notional.
    return daily_returns - aligned * (2.0 * cost_bps / 10_000.0)


def cost_sensitivity_curve(
    daily_returns: pd.Series,
    turnover: pd.Series,
    cost_grid: Iterable[float] = (0, 5, 10, 20, 30, 50),
    periods: int = TRADING_DAYS,
) -> pd.DataFrame:
    rows = []
    for bps in cost_grid:
        net = apply_transaction_costs(daily_returns, turnover, bps)
        rows.append(
            {
                "cost_bps": bps,
                "sharpe_ann": annualised_sharpe(net, periods),
                "return_ann": annualised_return(net, periods),
                "max_drawdown": max_drawdown(net),
            }
        )
    return pd.DataFrame(rows)


def break_even_cost_bps(daily_returns: pd.Series, turnover: pd.Series) -> float:
    """Cost level at which the mean net return hits zero."""
    mean_ret = pd.Series(daily_returns, dtype=float).dropna().mean()
    mean_to = turnover.reindex(pd.Series(daily_returns).dropna().index).fillna(0.0).mean()
    if not np.isfinite(mean_ret) or mean_to <= 0 or mean_ret <= 0:
        return float("nan")
    return float(mean_ret / (2.0 * mean_to) * 10_000.0)


# ----------------------------------------------------------------------------
# One-call summary
# ----------------------------------------------------------------------------
@dataclass
class PerformanceReport:
    model: str
    n_days: int
    start: Optional[str]
    end: Optional[str]
    metrics: Dict[str, float]

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "model": self.model,
            "n_days": self.n_days,
            "start": self.start,
            "end": self.end,
        }
        row.update(self.metrics)
        return row

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_strategy(
    model: str,
    daily_returns: pd.Series,
    scored: Optional[pd.DataFrame] = None,
    selection: Optional[pd.DataFrame] = None,
    n_trials: int = 1,
    seed: int = 0,
    periods: int = TRADING_DAYS,
    bootstrap: int = 2000,
    trial_sharpe_std: Optional[float] = None,
) -> PerformanceReport:
    """Full metric block for one model on one out-of-sample window.

    ``trial_sharpe_std`` is the per-period standard deviation of the Sharpe
    ratios across the ``n_trials`` configurations searched, which is what Bailey
    and Lopez de Prado's expected-maximum formula takes. When it is not supplied
    the sampling standard error ``1/sqrt(n)`` is used instead, which treats the
    trials as noise around a common zero and is the weaker correction.
    """
    r = pd.Series(daily_returns, dtype=float).dropna().sort_index()
    metrics: Dict[str, float] = {
        "sharpe_ann": annualised_sharpe(r, periods),
        "sortino_ann": annualised_sortino(r, periods),
        "return_ann": annualised_return(r, periods),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r, periods),
        "hit_rate": hit_rate(r),
        "mean_daily": float(r.mean()) if len(r) else np.nan,
    }
    metrics.update({f"sharpe_{k}": v for k, v in
                    bootstrap_sharpe_ci(r, n_boot=bootstrap, seed=seed, periods=periods).items()
                    if k != "sharpe"})
    metrics.update({f"dsr_{k}": v for k, v in
                    deflated_sharpe_ratio(r, n_trials=n_trials, periods=periods,
                                          trial_sharpe_std=trial_sharpe_std).items()})

    if scored is not None and not scored.empty:
        ic = daily_ic(scored)
        metrics.update(ic_summary(ic, periods))
        # Decile spread and monotonicity explain any disagreement between the
        # IC (whole cross-section) and the long/short Sharpe (tails only).
        quantiles = quantile_portfolios(scored, periods=periods)
        spread = quantiles.attrs.get("spread", {}) if not quantiles.empty else {}
        for key in ("mean_return", "t_stat_nw", "sharpe_ann", "monotonicity"):
            metrics[f"decile_{key}"] = spread.get(key, np.nan)

    if selection is not None and not selection.empty:
        to = daily_turnover(selection)
        metrics["turnover_mean"] = float(to.mean()) if len(to) else np.nan
        metrics["break_even_bps"] = break_even_cost_bps(r, to)
        for bps in (5, 10, 20):
            metrics[f"sharpe_ann_{bps}bps"] = annualised_sharpe(
                apply_transaction_costs(r, to, bps), periods
            )

    return PerformanceReport(
        model=model,
        n_days=int(len(r)),
        start=str(r.index.min()) if len(r) else None,
        end=str(r.index.max()) if len(r) else None,
        metrics=metrics,
    )

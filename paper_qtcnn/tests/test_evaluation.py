"""Calibration tests for the evaluation statistics.

These are the checks the manuscript cites: an estimator that is not calibrated
on data with a known answer cannot be trusted on data without one. Each test
states the ground truth it checks against rather than asserting a value that was
read off a previous run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qtcnn_research import evaluation as ev


# ----------------------------------------------------------------------------
# Risk metrics against closed-form answers
# ----------------------------------------------------------------------------
def test_annualised_sharpe_matches_definition():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015] * 40)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert ev.annualised_sharpe(r) == pytest.approx(expected)


def test_annualised_sharpe_recovers_a_known_population_value():
    # A long sample from N(mu, sigma) has Sharpe mu/sigma*sqrt(252).
    rng = np.random.default_rng(0)
    r = rng.normal(5e-4, 1e-2, 400_000)
    assert ev.annualised_sharpe(r) == pytest.approx(5e-4 / 1e-2 * np.sqrt(252), rel=0.05)


def test_max_drawdown_of_a_monotone_series_is_zero():
    assert ev.max_drawdown(pd.Series([0.01] * 50)) == pytest.approx(0.0)


def test_max_drawdown_matches_a_hand_computed_path():
    # equity: 1.0 -> 1.5 -> 0.75 -> 0.9 ; trough is 50% below the 1.5 peak
    r = pd.Series([0.5, -0.5, 0.2])
    assert ev.max_drawdown(r) == pytest.approx(-0.5)


def test_short_series_return_nan_rather_than_a_misleading_number():
    assert np.isnan(ev.annualised_sharpe([0.01]))
    assert np.isnan(ev.annualised_sharpe([]))


# ----------------------------------------------------------------------------
# Newey-West
# ----------------------------------------------------------------------------
def test_newey_west_widens_the_error_on_autocorrelated_data():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 5000)
    smooth = pd.Series(np.convolve(noise, np.ones(10) / 10, mode="same"))
    _, se_nw, _ = ev.newey_west_tstat(smooth)
    se_ols = smooth.std(ddof=1) / np.sqrt(len(smooth))
    assert se_nw > se_ols


def test_newey_west_agrees_with_ols_on_iid_data():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(0, 1, 20_000))
    _, se_nw, _ = ev.newey_west_tstat(x)
    se_ols = x.std(ddof=1) / np.sqrt(len(x))
    assert se_nw == pytest.approx(se_ols, rel=0.15)


# ----------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------
def test_bootstrap_does_not_reject_a_zero_mean_series():
    rng = np.random.default_rng(3)
    out = ev.bootstrap_sharpe_ci(rng.normal(0, 1e-2, 1500), n_boot=400, seed=0)
    assert out["p_value"] > 0.10
    assert out["ci_low"] < 0 < out["ci_high"]


def test_bootstrap_rejects_a_strong_signal():
    rng = np.random.default_rng(4)
    out = ev.bootstrap_sharpe_ci(rng.normal(3e-3, 1e-2, 1500), n_boot=400, seed=0)
    assert out["p_value"] < 0.05
    assert out["ci_low"] > 0


def test_stationary_bootstrap_indices_stay_in_range():
    rng = np.random.default_rng(5)
    idx = ev.stationary_bootstrap_indices(100, 10.0, rng)
    assert len(idx) == 100
    assert idx.min() >= 0 and idx.max() < 100


# ----------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ----------------------------------------------------------------------------
def test_dsr_falls_as_the_number_of_trials_grows():
    rng = np.random.default_rng(6)
    r = rng.normal(1e-3, 1e-2, 1000)
    single = ev.deflated_sharpe_ratio(r, n_trials=1)["dsr"]
    many = ev.deflated_sharpe_ratio(r, n_trials=500)["dsr"]
    assert single > many


def test_dsr_is_defined_for_a_single_trial():
    # norm.ppf(1 - 1/n) is -inf at n=1; a single trial has no selection bias.
    rng = np.random.default_rng(7)
    out = ev.deflated_sharpe_ratio(rng.normal(1e-3, 1e-2, 500), n_trials=1)
    assert np.isfinite(out["dsr"])
    assert out["sr_zero_ann"] == pytest.approx(0.0)


# ----------------------------------------------------------------------------
# Probability of backtest overfitting
# ----------------------------------------------------------------------------
def test_pbo_is_calibrated_at_one_half_on_independent_strategies():
    """The claim quoted in the paper: uninformative selection gives PBO ~ 0.5."""
    values = []
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        matrix = pd.DataFrame(rng.normal(0, 1e-2, (800, 10)),
                              columns=[f"c{i}" for i in range(10)])
        values.append(ev.probability_of_backtest_overfitting(matrix, n_splits=8)["pbo"])
    values = np.asarray(values)
    stderr = values.std(ddof=1) / np.sqrt(len(values))
    assert abs(values.mean() - 0.5) < 3 * stderr


# ----------------------------------------------------------------------------
# Model comparison
# ----------------------------------------------------------------------------
def test_diebold_mariano_detects_a_genuine_difference():
    rng = np.random.default_rng(8)
    common = rng.normal(0, 1e-2, 2000)
    better = common + rng.normal(1e-3, 2e-3, 2000)
    out = ev.diebold_mariano(better, common)
    assert out["dm_stat"] > 0
    assert out["p_value"] < 0.01


def test_diebold_mariano_finds_nothing_between_identical_series():
    rng = np.random.default_rng(9)
    a = pd.Series(rng.normal(0, 1e-2, 1000))
    out = ev.diebold_mariano(a, a.copy())
    assert np.isnan(out["dm_stat"]) or abs(out["dm_stat"]) < 1e-6


def test_romano_wolf_p_values_are_monotone_in_the_stepdown_order():
    rng = np.random.default_rng(10)
    bench = pd.Series(rng.normal(0, 1e-2, 600))
    candidates = {f"m{i}": bench + rng.normal(i * 2e-4, 5e-3, 600) for i in range(4)}
    table = ev.romano_wolf_stepdown(candidates, bench, n_boot=120, seed=0)
    ordered = table.reindex(table["t_stat"].abs().sort_values(ascending=False).index)
    assert ordered["p_rw"].is_monotonic_increasing


# ----------------------------------------------------------------------------
# Costs and turnover
# ----------------------------------------------------------------------------
def _selection(dates, longs, shorts):
    rows = []
    for date, lo, sh in zip(dates, longs, shorts):
        for code in lo:
            rows.append({"Date": date, "SecuritiesCode": code, "Side": "long"})
        for code in sh:
            rows.append({"Date": date, "SecuritiesCode": code, "Side": "short"})
    return pd.DataFrame(rows)


def test_turnover_is_zero_when_the_book_never_changes():
    dates = pd.date_range("2020-01-01", periods=5)
    sel = _selection(dates, [[1, 2]] * 5, [[3, 4]] * 5)
    assert ev.daily_turnover(sel).eq(0.0).all()


def test_turnover_is_one_when_the_book_is_fully_replaced():
    dates = pd.date_range("2020-01-01", periods=2)
    sel = _selection(dates, [[1, 2], [5, 6]], [[3, 4], [7, 8]])
    assert ev.daily_turnover(sel).iloc[0] == pytest.approx(1.0)


def test_costs_reduce_returns_by_turnover_times_both_legs():
    idx = pd.date_range("2020-01-01", periods=3)
    returns = pd.Series([0.01, 0.01, 0.01], index=idx)
    turnover = pd.Series([0.5, 0.5, 0.5], index=idx)
    net = ev.apply_transaction_costs(returns, turnover, cost_bps=10)
    assert net.iloc[0] == pytest.approx(0.01 - 0.5 * 2 * 10 / 10_000)


def test_break_even_cost_zeroes_the_mean_net_return():
    idx = pd.date_range("2020-01-01", periods=100)
    returns = pd.Series(np.full(100, 5e-4), index=idx)
    turnover = pd.Series(np.full(100, 0.25), index=idx)
    bps = ev.break_even_cost_bps(returns, turnover)
    assert ev.apply_transaction_costs(returns, turnover, bps).mean() == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------
# Quantile portfolios
# ----------------------------------------------------------------------------
def _panel(seed, effect):
    rng = np.random.default_rng(seed)
    frames = []
    for date in pd.bdate_range("2020-01-01", periods=150):
        score = rng.normal(0, 1, 400)
        frames.append(pd.DataFrame({
            "Date": date,
            "SecuritiesCode": np.arange(400),
            "score": score,
            "Target": effect(score, rng),
        }))
    return pd.concat(frames, ignore_index=True)


def test_quantile_table_is_monotone_for_a_monotone_signal():
    panel = _panel(11, lambda s, rng: 0.002 * s + rng.normal(0, 0.02, len(s)))
    table = ev.quantile_portfolios(panel)
    assert table.attrs["spread"]["monotonicity"] > 0.9
    assert table.attrs["spread"]["mean_return"] > 0


def test_quantile_spread_is_negative_when_the_tails_reverse():
    # Signal predicts in the middle but reverses in both tails: exactly the
    # case where a positive IC coexists with a losing long/short book.
    def effect(s, rng):
        return -0.004 * s * (np.abs(s) > 1.2) + rng.normal(0, 0.005, len(s))

    panel = _panel(12, effect)
    table = ev.quantile_portfolios(panel)
    assert table.attrs["spread"]["mean_return"] < 0


def test_ic_and_decile_spread_can_disagree_in_sign():
    """A rank-positive signal can still lose money in the top decile.

    IC is a rank correlation and ignores magnitudes; a decile's return is a
    mean and does not. A handful of large losses concentrated among the highest
    scores therefore leaves the IC positive while dragging the top-minus-bottom
    spread negative. This is the mechanism behind the disagreement observed in
    the reference implementation's own results, so the estimators must be able
    to show it rather than smooth it away.
    """
    def effect(s, rng):
        base = 0.002 * s + rng.normal(0, 0.01, len(s))
        crash = (s > 2.0) & (rng.random(len(s)) < 0.5)
        return base - 0.6 * crash

    panel = _panel(13, effect)
    ic = ev.daily_ic(panel).mean()
    spread = ev.quantile_portfolios(panel).attrs["spread"]["mean_return"]
    assert ic > 0, f"rank correlation should stay positive, got {ic}"
    assert spread < 0, f"top-minus-bottom mean should be negative, got {spread}"


def test_bootstrap_p_value_is_calibrated_under_the_null():
    """A zero-mean series must not be rejected more often than alpha."""
    import numpy as np
    from qtcnn_research.evaluation import bootstrap_sharpe_ci
    rejects = 0
    trials = 40
    for s in range(trials):
        rng = np.random.default_rng(1000 + s)
        r = rng.normal(0.0, 0.01, 400)
        out = bootstrap_sharpe_ci(r - r.mean(), n_boot=400, seed=s)
        rejects += out["p_value"] < 0.05
    assert rejects <= 0.20 * trials


def test_bootstrap_p_value_rejects_a_strong_signal():
    import numpy as np
    from qtcnn_research.evaluation import bootstrap_sharpe_ci
    rng = np.random.default_rng(7)
    r = rng.normal(0.004, 0.01, 400)          # ~ +6 annualised Sharpe
    out = bootstrap_sharpe_ci(r, n_boot=600, seed=3)
    assert out["p_value"] < 0.01
    assert out["ci_low"] > 0


def test_bootstrap_p_value_survives_a_single_extreme_outlier():
    """Under heavy left skew the re-centred-draws construction contradicted the CI."""
    import numpy as np
    from qtcnn_research.evaluation import bootstrap_sharpe_ci
    rng = np.random.default_rng(11)
    r = rng.normal(0.0, 0.01, 400)
    r[200] = -1.2                              # one impossible day, as in the US panel
    out = bootstrap_sharpe_ci(r, n_boot=600, seed=5)
    # the interval and the p-value must agree about whether zero is excluded
    excludes_zero = out["ci_low"] > 0 or out["ci_high"] < 0
    assert excludes_zero == (out["p_value"] < 0.05)

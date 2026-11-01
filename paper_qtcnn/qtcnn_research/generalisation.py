"""Tier-2 experiments: data efficiency, market regimes, and signal orthogonality.

These carry the paper's positive claims if the headline comparison turns out
flat. Each answers a question that a Sharpe table cannot:

* **Data efficiency** -- daily cross-sections are intrinsically small samples,
  and a low-parameter circuit is exactly the kind of model expected to hold up
  when training data is scarce. Sweeping the training-set size tests that.
* **Regimes** -- an average over five years hides whether a model helps in the
  conditions where help is worth paying for.
* **Orthogonality** -- a signal that is uncorrelated with classical ones can add
  value in combination even when it loses head-to-head, and a spanning
  regression is the standard way to establish that.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from qtcnn_research import evaluation as ev


# ----------------------------------------------------------------------------
# Regime split
# ----------------------------------------------------------------------------
def market_regimes(
    market_returns: pd.Series, vol_window: int = 60, trend_window: int = 60
) -> pd.DataFrame:
    """Label each date by realised-volatility tercile and trailing trend sign.

    Both labels use only trailing data, so conditioning on them is something a
    practitioner could actually do in real time.
    """
    r = pd.Series(market_returns, dtype=float).sort_index()
    vol = r.rolling(vol_window, min_periods=vol_window // 2).std()
    trend = r.rolling(trend_window, min_periods=trend_window // 2).mean()
    frame = pd.DataFrame({"market_return": r, "vol": vol, "trend": trend})
    # Terciles are taken on trailing volatility over the whole sample; this is
    # in-sample labelling used only for reporting, never for model selection.
    q1, q2 = frame["vol"].quantile([1 / 3, 2 / 3])
    frame["vol_regime"] = np.where(frame["vol"] <= q1, "low",
                            np.where(frame["vol"] <= q2, "mid", "high"))
    frame["trend_regime"] = np.where(frame["trend"] >= 0, "up", "down")
    return frame


def regime_performance(
    daily_returns: pd.Series, regimes: pd.DataFrame, regime_col: str = "vol_regime"
) -> pd.DataFrame:
    """Annualised Sharpe within each regime, with a bootstrap CI per bucket."""
    joined = pd.concat([pd.Series(daily_returns, name="r"), regimes[regime_col]], axis=1).dropna()
    rows = []
    for label, g in joined.groupby(regime_col):
        ci = ev.bootstrap_sharpe_ci(g["r"], n_boot=1000, seed=0)
        rows.append({
            "regime": label,
            "n_days": len(g),
            "sharpe_ann": ci["sharpe"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "p_value": ci["p_value"],
            "mean_daily": float(g["r"].mean()),
        })
    return pd.DataFrame(rows).sort_values("regime", ignore_index=True)


# ----------------------------------------------------------------------------
# Signal orthogonality and spanning
# ----------------------------------------------------------------------------
def score_correlation(scored: Dict[str, pd.DataFrame], score_col: str = "score") -> pd.DataFrame:
    """Mean within-date rank correlation between each pair of models' scores.

    Correlations must be computed inside a cross-section: pooling across dates
    would mostly measure the shared market-wide level.
    """
    names = sorted(scored)
    keys = ["Date", "SecuritiesCode"]
    merged = None
    for name in names:
        sub = scored[name][keys + [score_col]].rename(columns={score_col: name})
        merged = sub if merged is None else merged.merge(sub, on=keys, how="inner")
    if merged is None or merged.empty:
        return pd.DataFrame()

    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            per_date = merged.groupby("Date").apply(
                lambda g, x=a, y=b: g[x].corr(g[y], method="spearman")
            )
            value = float(per_date.mean())
            out.loc[a, b] = out.loc[b, a] = value
    return out


def spanning_regression(
    target_returns: pd.Series, factor_returns: Dict[str, pd.Series]
) -> Dict[str, float]:
    """Regress a strategy's returns on other strategies; report alpha and its t.

    A significant intercept means the strategy delivers return that the others
    do not span -- the claim to make when a model does not win outright but is
    not redundant either.
    """
    frame = pd.DataFrame({"y": pd.Series(target_returns, dtype=float)})
    for name, series in factor_returns.items():
        frame[name] = pd.Series(series, dtype=float)
    frame = frame.dropna()
    if len(frame) < 30:
        return {"alpha": np.nan, "alpha_t": np.nan, "n": len(frame)}

    y = frame["y"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in frame.columns[1:]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    out = {
        "alpha": float(beta[0]),
        "alpha_ann": float(beta[0] * ev.TRADING_DAYS),
        "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
        "r_squared": float(1 - resid.var(ddof=0) / y.var(ddof=0)) if y.var() > 0 else np.nan,
        "n": int(len(y)),
    }
    out["alpha_p"] = float(2 * (1 - stats.t.cdf(abs(out["alpha_t"]), dof))) \
        if np.isfinite(out["alpha_t"]) else np.nan
    for name, b, s in zip(frame.columns[1:], beta[1:], se[1:]):
        out[f"beta_{name}"] = float(b)
        out[f"t_{name}"] = float(b / s) if s > 0 else np.nan
    return out


def combination_gain(
    returns: Dict[str, pd.Series], quantum: str, classical: Sequence[str]
) -> pd.DataFrame:
    """Sharpe of each classical model alone vs. an equal-weight blend with the
    quantum model. The practical form of the orthogonality claim."""
    rows = []
    q = pd.Series(returns[quantum], dtype=float)
    for name in classical:
        c = pd.Series(returns[name], dtype=float)
        joined = pd.concat([q.rename("q"), c.rename("c")], axis=1).dropna()
        if joined.empty:
            continue
        blend = 0.5 * joined["q"] + 0.5 * joined["c"]
        rows.append({
            "classical": name,
            "sharpe_classical": ev.annualised_sharpe(joined["c"]),
            "sharpe_quantum": ev.annualised_sharpe(joined["q"]),
            "sharpe_blend": ev.annualised_sharpe(blend),
            "corr": float(joined["q"].corr(joined["c"])),
            "n_days": len(joined),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gain_vs_classical"] = df["sharpe_blend"] - df["sharpe_classical"]
    return df


# ----------------------------------------------------------------------------
# Data efficiency
# ----------------------------------------------------------------------------
def data_efficiency_grid(
    full_size: int, fractions: Sequence[float] = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
) -> List[int]:
    """Training-set sizes to sweep, deduplicated and capped at the full sample."""
    sizes = sorted({max(512, int(round(f * full_size))) for f in fractions})
    return [s for s in sizes if s <= full_size]

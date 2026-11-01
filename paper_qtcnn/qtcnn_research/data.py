"""JPX data loading and cross-sectional feature construction.

Two things here differ deliberately from ``scripts/*_pipeline.py``:

1. **Split adjustment direction.** The legacy pipelines compute
   ``AdjustmentFactor.cumprod()`` in forward date order. In this dataset a
   factor of 0.5 recorded on day *t* means prices up to and including *t* must
   be halved to line up with prices after *t*, so the cumulative factor has to
   be accumulated backwards. Measured over the 610 affected securities, the
   forward version leaves 1418 daily moves above 30% (worse than the 718 in
   completely unadjusted prices), while the backward version leaves 16.

2. **Supplemental period held back.** ``train_files`` ends 2021-12-03 and
   ``supplemental_files`` runs to 2022-06-24. The supplemental window is kept as
   a single locked holdout that no model selection ever touches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

PRICE_COLS = [
    "Date",
    "SecuritiesCode",
    "Close",
    "Volume",
    "AdjustmentFactor",
    "SupervisionFlag",
    "Target",
]

# Split of the sample that every experiment shares.
DEV_END = pd.Timestamp("2021-12-03")       # walk-forward development period
HOLDOUT_START = pd.Timestamp("2021-12-06")  # locked, evaluated once


@dataclass(frozen=True)
class FeatureSpec:
    """Which engineered columns feed the models."""

    momentum: Sequence[int] = (1, 2, 5, 10, 20)
    volatility: Sequence[int] = (5, 10, 20)
    liquidity: Sequence[int] = (5, 10, 20)
    include_rsi: bool = True
    include_position: bool = True

    def columns(self) -> List[str]:
        cols = [f"mom{n}" for n in self.momentum]
        cols += [f"vol{n}" for n in self.volatility]
        cols += [f"ret_z{n}" for n in self.volatility]
        cols += ["log_vol", "log_dv"]
        cols += [f"dv_z{n}" for n in self.liquidity]
        if self.include_position:
            cols += ["pos10", "pos20", "dd20"]
        if self.include_rsi:
            cols += ["rsi7", "rsi14"]
        return cols


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.lower().map({"true": True, "false": False})
    return series.fillna(False).astype(bool)


def load_prices(
    data_dir: Path,
    include_supplemental: bool = True,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Concatenate the training and supplemental price files."""
    data_dir = Path(data_dir)
    cols = usecols or PRICE_COLS
    dtypes = {
        "SecuritiesCode": np.int32,
        "Close": np.float64,
        "Volume": np.float64,
        "AdjustmentFactor": np.float64,
        "Target": np.float64,
    }
    frames = []
    paths = [data_dir / "train_files/stock_prices.csv"]
    if include_supplemental:
        supplemental = data_dir / "supplemental_files/stock_prices.csv"
        if supplemental.exists():
            paths.append(supplemental)
    for path in paths:
        frame = pd.read_csv(path, usecols=cols, dtype=dtypes, parse_dates=["Date"])
        frames.append(frame)
    prices = pd.concat(frames, ignore_index=True)
    if "SupervisionFlag" in prices.columns:
        prices["SupervisionFlag"] = _coerce_bool(prices["SupervisionFlag"])
    prices = prices.drop_duplicates(subset=["Date", "SecuritiesCode"], keep="last")
    return prices.sort_values(["SecuritiesCode", "Date"], ignore_index=True)


def load_universe(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        Path(data_dir) / "stock_list.csv",
        usecols=["SecuritiesCode", "Universe0"],
        dtype={"SecuritiesCode": np.int32},
    )
    df["Universe0"] = _coerce_bool(df["Universe0"])
    return df


# ----------------------------------------------------------------------------
# Split-adjusted prices
# ----------------------------------------------------------------------------
def add_adjusted_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust closes so that returns are free of split artefacts.

    The cumulative factor for date *t* is the product of every AdjustmentFactor
    from *t* onwards, which is a reverse cumulative product within each security.
    """
    prices = prices.sort_values(["SecuritiesCode", "Date"]).copy()
    rev_cum = (
        prices.iloc[::-1]
        .groupby("SecuritiesCode", sort=False)["AdjustmentFactor"]
        .cumprod()
        .iloc[::-1]
    )
    prices["cum_adj"] = rev_cum
    prices["AdjClose"] = prices["Close"] * prices["cum_adj"]
    return prices


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def add_features(prices: pd.DataFrame, spec: FeatureSpec = FeatureSpec()) -> pd.DataFrame:
    """Per-security rolling features. All windows look strictly backwards."""
    df = prices.sort_values(["SecuritiesCode", "Date"]).copy()
    grp = df.groupby("SecuritiesCode", sort=False)["AdjClose"]

    df["ret1"] = grp.pct_change(fill_method=None)
    for n in spec.momentum:
        df[f"mom{n}"] = grp.pct_change(n, fill_method=None)

    ret_grp = df.groupby("SecuritiesCode", sort=False)["ret1"]
    for n in spec.volatility:
        std = ret_grp.rolling(n, min_periods=n).std().reset_index(level=0, drop=True)
        mean = ret_grp.rolling(n, min_periods=n).mean().reset_index(level=0, drop=True)
        df[f"vol{n}"] = std
        # Standardised recent move: how unusual is today relative to its own window.
        df[f"ret_z{n}"] = (df["ret1"] - mean) / (std + 1e-8)

    df["dv"] = df["Close"] * df["Volume"]
    df["log_vol"] = np.log1p(df["Volume"])
    df["log_dv"] = np.log1p(df["dv"])
    dv_grp = df.groupby("SecuritiesCode", sort=False)["log_dv"]
    for n in spec.liquidity:
        mean = dv_grp.rolling(n, min_periods=n).mean().reset_index(level=0, drop=True)
        std = dv_grp.rolling(n, min_periods=n).std().reset_index(level=0, drop=True)
        df[f"dv_z{n}"] = (df["log_dv"] - mean) / (std + 1e-8)

    if spec.include_position:
        for n in (10, 20):
            hi = grp.rolling(n, min_periods=n).max().reset_index(level=0, drop=True)
            lo = grp.rolling(n, min_periods=n).min().reset_index(level=0, drop=True)
            df[f"pos{n}"] = (df["AdjClose"] - lo) / (hi - lo + 1e-8)
        peak = grp.rolling(20, min_periods=20).max().reset_index(level=0, drop=True)
        df["dd20"] = df["AdjClose"] / (peak + 1e-8) - 1.0

    if spec.include_rsi:
        for n in (7, 14):
            df[f"rsi{n}"] = (
                df.groupby("SecuritiesCode", sort=False)["AdjClose"]
                .apply(lambda s, w=n: _rsi(s, w))
                .reset_index(level=0, drop=True)
            )
            df[f"rsi{n}"] = df[f"rsi{n}"] / 100.0 - 0.5

    return df


# ----------------------------------------------------------------------------
# Cross-sectional normalisation
# ----------------------------------------------------------------------------
def cross_sectional_normalise(
    df: pd.DataFrame,
    cols: Sequence[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorise then z-score within each date, vectorised over the whole frame.

    Done per date, this uses no information from other dates and therefore
    introduces no look-ahead, while removing the market-wide level that would
    otherwise dominate a cross-sectional ranking model.
    """
    df = df.copy()
    cols = list(cols)
    grouped = df.groupby("Date", sort=False)[cols]
    lo = grouped.transform(lambda s: s.quantile(lower))
    hi = grouped.transform(lambda s: s.quantile(upper))
    clipped = df[cols].clip(lower=lo, upper=hi)
    mu = clipped.groupby(df["Date"], sort=False).transform("mean")
    sd = clipped.groupby(df["Date"], sort=False).transform("std")
    df[cols] = ((clipped - mu) / (sd + 1e-8)).astype(np.float32)
    return df


def build_dataset(
    data_dir: Path,
    spec: FeatureSpec = FeatureSpec(),
    include_supplemental: bool = True,
    min_price: float = 0.0,
) -> pd.DataFrame:
    """End-to-end: raw CSVs to a normalised, universe-tagged feature frame."""
    prices = load_prices(data_dir, include_supplemental=include_supplemental)
    prices = add_adjusted_close(prices)
    prices = add_features(prices, spec)
    universe = load_universe(data_dir)
    prices = prices.merge(universe, on="SecuritiesCode", how="left")
    prices["Universe0"] = prices["Universe0"].fillna(False).astype(bool)

    feature_cols = spec.columns()
    keep = ["Date", "SecuritiesCode", "Close", "Volume", "AdjClose", "SupervisionFlag",
            "Universe0", "Target"] + feature_cols
    df = prices[keep]
    df = df[np.isfinite(df[feature_cols]).all(axis=1) & df["Target"].notna()]
    if min_price > 0:
        df = df[df["Close"] >= min_price]
    df = cross_sectional_normalise(df, feature_cols)
    return df.sort_values(["Date", "SecuritiesCode"], ignore_index=True)


def tradable_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that a real portfolio could actually hold."""
    return (
        df["Universe0"]
        & (~df["SupervisionFlag"])
        & (df["Close"] > 0)
        & (df["Volume"] > 0)
    )


def make_sequences(
    df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int = 20
) -> tuple:
    """Vectorised sliding windows per security.

    Returns ``(X, index)`` where ``X`` is ``[N, seq_len, F]`` and ``index`` holds
    the Date/SecuritiesCode of each window's final day.
    """
    feature_cols = list(feature_cols)
    df = df.sort_values(["SecuritiesCode", "Date"])
    values = df[feature_cols].to_numpy(dtype=np.float32)
    codes = df["SecuritiesCode"].to_numpy()
    dates = df["Date"].to_numpy()

    # Window boundaries must not straddle two securities.
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]

    blocks, idx_pos = [], []
    for s, e in zip(starts, ends):
        n = e - s
        if n < seq_len:
            continue
        # stride tricks give a view; copy once per security keeps memory bounded
        window = np.lib.stride_tricks.sliding_window_view(
            values[s:e], (seq_len, values.shape[1])
        )[:, 0]
        blocks.append(window)
        idx_pos.append(np.arange(s + seq_len - 1, e))

    if not blocks:
        return np.empty((0, seq_len, len(feature_cols)), np.float32), pd.DataFrame(
            columns=["Date", "SecuritiesCode"]
        )

    X = np.concatenate(blocks, axis=0).astype(np.float32)
    pos = np.concatenate(idx_pos)
    index = pd.DataFrame({"Date": dates[pos], "SecuritiesCode": codes[pos]})
    return X, index

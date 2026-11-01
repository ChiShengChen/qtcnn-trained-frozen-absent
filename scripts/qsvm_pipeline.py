from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/data")
TRAIN_PRICES = DATA_DIR / "train_files/stock_prices.csv"
STOCK_LIST = DATA_DIR / "stock_list.csv"
OUTPUT_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/outputs") / "qsvm"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_prices(usecols: Optional[List[str]] = None) -> pd.DataFrame:
    default_cols = [
        "Date",
        "SecuritiesCode",
        "Close",
        "Volume",
        "AdjustmentFactor",
        "SupervisionFlag",
        "Target",
    ]
    cols = usecols or default_cols
    dtypes = {
        "SecuritiesCode": np.int32,
        "Close": np.float32,
        "Volume": np.float64,
        "AdjustmentFactor": np.float32,
        "Target": np.float32,
    }
    parse_dates = ["Date"] if "Date" in cols else None
    df = pd.read_csv(
        TRAIN_PRICES,
        usecols=cols,
        dtype=dtypes,
        parse_dates=parse_dates,
        low_memory=False,
    )
    if "SupervisionFlag" in df.columns:
        if df["SupervisionFlag"].dtype == object:
            df["SupervisionFlag"] = (
                df["SupervisionFlag"].astype(str).str.lower().map({"true": True, "false": False})
            )
        df["SupervisionFlag"] = df["SupervisionFlag"].fillna(False).astype(bool)
    return df


def read_universe() -> pd.DataFrame:
    cols = ["SecuritiesCode", "Universe0"]
    df = pd.read_csv(STOCK_LIST, usecols=cols, dtype={"SecuritiesCode": np.int32})
    if df["Universe0"].dtype == object:
        df["Universe0"] = df["Universe0"].astype(str).str.lower().map({"true": True, "false": False})
    return df


def add_adjusted_close(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["SecuritiesCode", "Date"]).copy()
    prices["cum_adj"] = prices.groupby("SecuritiesCode")["AdjustmentFactor"].cumprod()
    prices["AdjClose"] = prices["Close"] * prices["cum_adj"]
    return prices


def add_multi_scale_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["SecuritiesCode", "Date"]).copy()
    prices["ret1"] = prices.groupby("SecuritiesCode")["AdjClose"].pct_change(fill_method=None)
    for n in [1, 2, 5, 10, 20]:
        prices[f"mom{n}"] = prices.groupby("SecuritiesCode")["AdjClose"].pct_change(n, fill_method=None)
    for n in [5, 10, 20]:
        prices[f"vol{n}"] = (
            prices.groupby("SecuritiesCode")["ret1"].rolling(n).std().reset_index(level=0, drop=True)
        )
    prices["dv"] = prices["Close"] * prices["Volume"]
    for n in [5, 10, 20]:
        prices[f"dv_mean{n}"] = (
            prices.groupby("SecuritiesCode")["dv"].rolling(n).mean().reset_index(level=0, drop=True)
        )
        prices[f"vol_mean{n}"] = (
            prices.groupby("SecuritiesCode")["Volume"].rolling(n).mean().reset_index(level=0, drop=True)
        )
    prices["log_vol"] = np.log1p(prices["Volume"])
    prices["log_dv"] = np.log1p(prices["dv"])
    return prices


def cross_sectional_winsorize(df: pd.DataFrame, cols: List[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    def _clip(g: pd.DataFrame) -> pd.DataFrame:
        for c in cols:
            x = g[c]
            ql = x.quantile(lower)
            qu = x.quantile(upper)
            g[c] = x.clip(ql, qu)
        return g
    return df.groupby("Date", group_keys=False).apply(_clip)


def cross_sectional_standardize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    def _z(g: pd.DataFrame) -> pd.DataFrame:
        for c in cols:
            x = g[c]
            mu = x.mean()
            sd = x.std(ddof=1)
            g[c] = (x - mu) / (sd + 1e-6)
        return g
    return df.groupby("Date", group_keys=False).apply(_z)


def build_feature_matrix(prices: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    prices = prices.copy()
    feature_cols: List[str] = [
        "mom1", "mom2", "mom5", "vol5", "log_vol", "log_dv",
    ]
    base = prices[(prices["Close"] > 0) & (prices["Volume"] >= 0)].copy()
    base = cross_sectional_winsorize(base, feature_cols, lower=0.01, upper=0.99)
    base = cross_sectional_standardize(base, feature_cols)
    return base, feature_cols


def select_outlier_training_samples(df: pd.DataFrame, target_col: str, per_side: int = 200) -> pd.DataFrame:
    def _pick(g: pd.DataFrame) -> pd.DataFrame:
        g = g.dropna(subset=[target_col]).sort_values(target_col)
        if g.shape[0] <= per_side * 2:
            return g.assign(label_bin=0)
        bottom = g.head(per_side).assign(label_bin=0)
        top = g.tail(per_side).assign(label_bin=1)
        return pd.concat([bottom, top], axis=0)
    return df.groupby("Date", group_keys=False).apply(_pick)


def split_train_valid(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = np.sort(df["Date"].unique())
    cut_idx = int(len(dates) * 0.8)
    cutoff_date = pd.to_datetime(dates[cut_idx])
    train = df[df["Date"] < cutoff_date].copy()
    valid = df[df["Date"] >= cutoff_date].copy()
    return train, valid, cutoff_date


def prepare_feature_space(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    cutoff_date: pd.Timestamp,
    max_train: int = 2000,
    n_qubits: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    # Drop rows with any non-finite features to avoid NaN/Inf in PCA and scaling
    feat_df = feat_df[np.isfinite(feat_df[feature_cols]).all(axis=1)].copy()

    labels_df = select_outlier_training_samples(feat_df[feat_df["Date"] < cutoff_date], target_col="Target", per_side=200)
    labels_df = labels_df[["Date", "SecuritiesCode", "label_bin"]]

    train_idx = feat_df[feat_df["Date"] < cutoff_date][["Date", "SecuritiesCode"]].merge(labels_df, on=["Date", "SecuritiesCode"], how="inner")
    key_all = feat_df["Date"].astype(str) + "_" + feat_df["SecuritiesCode"].astype(str)
    key_tr = set(train_idx["Date"].astype(str) + "_" + train_idx["SecuritiesCode"].astype(str))
    sel_mask = key_all.isin(key_tr).values

    X_all = feat_df[feature_cols].values.astype(np.float32)
    meta = feat_df[["Date", "SecuritiesCode", "Close", "Volume", "SupervisionFlag", "Target"]].copy()

    X_tr = X_all[sel_mask]
    y_tr = train_idx["label_bin"].values.astype(np.int64)

    if X_tr.shape[0] > max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_tr.shape[0], size=max_train, replace=False)
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_all_scaled = scaler.transform(X_all)

    pca = PCA(n_components=min(n_qubits, X_tr_scaled.shape[1]), random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_scaled).astype(np.float32)
    X_all_pca = pca.transform(X_all_scaled).astype(np.float32)

    angles_tr = np.clip(X_tr_pca / 3.0, -1.0, 1.0)
    angles_all = np.clip(X_all_pca / 3.0, -1.0, 1.0)
    angles_tr = (angles_tr + 1.0) * (np.pi / 2.0)
    angles_all = (angles_all + 1.0) * (np.pi / 2.0)

    return angles_tr, y_tr, angles_all, meta, np.array(pca.components_, dtype=np.float32)


def qkernel_matrix(angles_A: np.ndarray, angles_B: Optional[np.ndarray] = None, n_qubits: int = 8) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    def feature_map(angles):
        qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")

    @qml.qnode(dev)
    def kernel_qnode(a, b):
        feature_map(a)
        qml.adjoint(feature_map)(b)
        return qml.probs(wires=range(n_qubits))

    A = angles_A
    B = angles_A if angles_B is None else angles_B
    K = np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            probs = kernel_qnode(A[i], B[j])
            K[i, j] = float(probs[0])
    return K


def train_qsvm(angles_tr: np.ndarray, y_tr: np.ndarray, n_qubits: int = 8):
    K_tr = qkernel_matrix(angles_tr, None, n_qubits=n_qubits)
    model = SVC(kernel="precomputed", C=1.0, probability=False, class_weight="balanced", random_state=42)
    model.fit(K_tr, y_tr)
    return model, angles_tr


def predict_qsvm(model, angles_all: np.ndarray, angles_tr: np.ndarray, n_qubits: int = 8) -> np.ndarray:
    K_te = qkernel_matrix(angles_all, angles_tr, n_qubits=n_qubits)
    scores = model.decision_function(K_te).astype(np.float32)
    return scores


def rank_and_select(df: pd.DataFrame, universe: pd.DataFrame, top_n: int = 200, bottom_n: int = 200) -> pd.DataFrame:
    merged = df.merge(universe, on="SecuritiesCode", how="left")
    tradable = merged[(merged["Universe0"] == True) & (~merged["SupervisionFlag"]) & (merged["Close"] > 0) & (merged["Volume"] > 0)]  # noqa: E712
    tradable = tradable[np.isfinite(tradable["score"])].copy()
    def _pick(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("score", ascending=False)
        long = g.head(top_n).copy()
        short = g.tail(bottom_n).copy()
        long["Side"] = "long"
        short["Side"] = "short"
        return pd.concat([long, short], axis=0)
    selected = tradable.groupby("Date", group_keys=False).apply(_pick)
    return selected.sort_values(["Date", "Side", "score"], ascending=[True, True, False])


def evaluate_long_short(selected: pd.DataFrame, target_col: str = "Target") -> pd.DataFrame:
    eval_df = selected.dropna(subset=[target_col]).copy()
    daily = (
        eval_df.pivot_table(index="Date", columns="Side", values=target_col, aggfunc="mean")
        .rename(columns={"long": "long_mean", "short": "short_mean"})
        .reset_index()
    )
    if "long_mean" not in daily.columns:
        daily["long_mean"] = np.nan
    if "short_mean" not in daily.columns:
        daily["short_mean"] = np.nan
    daily["ls_return"] = daily["long_mean"] - daily["short_mean"]
    return daily.sort_values("Date")


def compute_sharpe(daily_ls: pd.DataFrame) -> float:
    r = daily_ls["ls_return"].dropna()
    if r.empty:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(mu / sd)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    prices = read_prices()
    universe = read_universe()
    prices = add_adjusted_close(prices)
    prices = add_multi_scale_features(prices)
    feat_df, feature_cols = build_feature_matrix(prices)

    train_df, valid_df, cutoff_date = split_train_valid(feat_df)
    angles_tr, y_tr, angles_all, meta, _ = prepare_feature_space(feat_df, feature_cols, cutoff_date, max_train=1500, n_qubits=8)

    model, angles_tr_ref = train_qsvm(angles_tr, y_tr, n_qubits=8)
    raw_scores = predict_qsvm(model, angles_all, angles_tr_ref, n_qubits=8)

    scored = meta.copy()
    scored["raw_score"] = raw_scores
    scored["score"] = scored.groupby("Date")["raw_score"].transform(lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-6))

    selected = rank_and_select(scored, universe, top_n=200, bottom_n=200)
    selected_oos = selected[selected["Date"] >= cutoff_date]
    daily_ls = evaluate_long_short(selected_oos, target_col="Target")
    sharpe = compute_sharpe(daily_ls)

    out_sel = OUTPUT_DIR / "daily_selection.csv"
    out_ls = OUTPUT_DIR / "daily_long_short_returns.csv"
    out_summary = OUTPUT_DIR / "summary.txt"
    out_scored = OUTPUT_DIR / "scored.csv"
    selected[["Date", "SecuritiesCode", "Side", "score"]].to_csv(out_sel, index=False)
    daily_ls.to_csv(out_ls, index=False)
    scored[["Date", "SecuritiesCode", "score", "Target", "Close", "Volume", "SupervisionFlag"]].to_csv(out_scored, index=False)
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write(f"Sharpe (OOS): {sharpe:.6f}\n")
        f.write(f"OOS Days: {daily_ls['ls_return'].notna().sum()}\n")
        f.write(f"OOS Start: {daily_ls['Date'].min()}\n")
        f.write(f"OOS End: {daily_ls['Date'].max()}\n")

    print(f"Sharpe (OOS): {sharpe:.6f}")
    print(f"Saved: {out_sel}")
    print(f"Saved: {out_ls}")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_scored}")


if __name__ == "__main__":
    main()




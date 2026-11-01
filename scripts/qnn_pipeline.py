from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DATA_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/data")
TRAIN_PRICES = DATA_DIR / "train_files/stock_prices.csv"
STOCK_LIST = DATA_DIR / "stock_list.csv"
OUTPUT_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/outputs") / "qnn"


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
    prices["ret1"] = prices.groupby("SecuritiesCode")["AdjClose"].pct_change()
    for n in [1, 2, 5, 10, 20]:
        prices[f"mom{n}"] = prices.groupby("SecuritiesCode")["AdjClose"].pct_change(n)
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


def downsample_prices_by_dates(
    prices: pd.DataFrame,
    date_stride: Optional[int] = None,
    date_fraction: Optional[float] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if date_stride is None and date_fraction is None:
        return prices, np.sort(prices["Date"].unique())
    dates = np.sort(prices["Date"].unique())
    if date_stride is not None and date_stride > 1:
        sampled_dates = dates[::date_stride]
    else:
        frac = float(date_fraction) if date_fraction is not None else 0.1
        rng = np.random.default_rng(seed)
        years = pd.to_datetime(dates).astype("datetime64[Y]")
        buckets = []
        for y in np.unique(years):
            mask = years == y
            d = dates[mask]
            k = max(1, int(round(len(d) * frac)))
            idx = rng.choice(len(d), size=k, replace=False)
            buckets.append(d[np.sort(idx)])
        sampled_dates = np.concatenate(buckets) if buckets else dates
        sampled_dates = np.sort(sampled_dates)
    filtered = prices[prices["Date"].isin(sampled_dates)].copy()
    return filtered, sampled_dates


def write_subset_dataset(subset_dir: Path, prices_subset: pd.DataFrame) -> None:
    subset_dir = Path(subset_dir)
    prices_path = subset_dir / "train_files" / "stock_prices.csv"
    prices_path.parent.mkdir(parents=True, exist_ok=True)
    prices_subset.to_csv(prices_path, index=False)
    # Copy stock_list to keep dataset self-contained
    dest_stock_list = subset_dir / "stock_list.csv"
    dest_stock_list.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(STOCK_LIST, dest_stock_list)
    except Exception:
        pass


def build_feature_matrix(prices: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    prices = prices.copy()
    feature_cols: List[str] = []
    for n in [1, 2, 5, 10, 20]:
        feature_cols.append(f"mom{n}")
    for n in [5, 10, 20]:
        feature_cols.append(f"vol{n}")
    feature_cols += [
        "log_vol",
        "log_dv",
        "dv_mean5",
        "dv_mean10",
        "dv_mean20",
        "vol_mean5",
        "vol_mean10",
        "vol_mean20",
    ]

    base = prices[(prices["Close"] > 0) & (prices["Volume"] >= 0)].copy()
    base = cross_sectional_winsorize(base, feature_cols, lower=0.01, upper=0.99)
    base = cross_sectional_standardize(base, feature_cols)
    return base, feature_cols


def select_outlier_training_samples(df: pd.DataFrame, target_col: str, per_side: int = 300) -> pd.DataFrame:
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


def prepare_qnn_angles(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    cutoff_date: pd.Timestamp,
    n_qubits: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    # Fit on training window only
    train_mask = feat_df["Date"] < cutoff_date
    X_train = feat_df.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=min(n_qubits, X_train_scaled.shape[1]), random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    # Map to angles in [0, pi]
    angles_tr = np.clip(X_train_pca / 3.0, -1.0, 1.0)
    angles_tr = (angles_tr + 1.0) * (np.pi / 2.0)
    # Transform all samples
    X_all = feat_df[feature_cols].fillna(0).values.astype(np.float32)
    X_all_scaled = scaler.transform(X_all)
    X_all_pca = pca.transform(X_all_scaled).astype(np.float32)
    angles_all = np.clip(X_all_pca / 3.0, -1.0, 1.0)
    angles_all = (angles_all + 1.0) * (np.pi / 2.0)
    meta_all = feat_df[["Date", "SecuritiesCode", "Close", "Volume", "SupervisionFlag", "Target"]].copy()
    return angles_tr, angles_all, meta_all, train_mask.values


def train_qnn_classifier(
    angles_tr: np.ndarray,
    y_tr: np.ndarray,
    n_qubits: int = 8,
    n_layers: int = 2,
    num_epochs: int = 50,
    batch_size: int = 512,
    lr: float = 2e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    class QNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = qlayer
            self.head = nn.Linear(n_qubits, 1)
        def forward(self, x):
            z = self.q(x)
            return self.head(z).squeeze(-1)

    model = QNNModel().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(angles_tr)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(num_epochs):
        total = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"epoch {epoch+1}: bce_loss={total/max(n,1):.6f}")
    model.eval()
    return model


def predict_qnn_proba(model, angles_all: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    with torch.no_grad():
        for i in range(0, len(angles_all), batch_size):
            xb = torch.from_numpy(angles_all[i:i+batch_size]).to(device)
            logits = model(xb)
            pb = torch.sigmoid(logits).detach().cpu().numpy()
            preds.append(pb.reshape(-1))
    return np.concatenate(preds, axis=0)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None, help="Override data directory containing train_files/stock_prices.csv and stock_list.csv")
    p.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--date-stride", type=int, default=None, help="Sample every N-th trading day (e.g., 11 for ~9%%)")
    group.add_argument("--date-fraction", type=float, default=None, help="Sample fraction of trading days per year (e.g., 0.1)")
    p.add_argument("--subset-save-dir", type=str, default=None, help="If set, save the downsampled dataset to this directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed for fraction-based sampling")
    return p.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    global DATA_DIR, TRAIN_PRICES, STOCK_LIST, OUTPUT_DIR
    if args is None:
        args = parse_args()
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
        TRAIN_PRICES = DATA_DIR / "train_files/stock_prices.csv"
        STOCK_LIST = DATA_DIR / "stock_list.csv"
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    ensure_output_dir(out_dir)

    # Load and features
    prices = read_prices()
    # Optional downsampling of trading days before feature engineering
    prices, sampled_dates = downsample_prices_by_dates(
        prices,
        date_stride=args.date_stride,
        date_fraction=args.date_fraction,
        seed=args.seed,
    )
    if args.date_stride or args.date_fraction:
        print(f"Downsampled to {len(sampled_dates)} trading days, rows={len(prices)}")
        if args.subset_save_dir:
            write_subset_dataset(Path(args.subset_save_dir), prices)
            print(f"Saved subset dataset to: {args.subset_save_dir}")
    universe = read_universe()
    prices = add_adjusted_close(prices)
    prices = add_multi_scale_features(prices)
    feat_df, feature_cols = build_feature_matrix(prices)

    # Split and labels
    train_df, valid_df, cutoff_date = split_train_valid(feat_df)
    train_extreme = select_outlier_training_samples(train_df, target_col="Target", per_side=300)
    labels_df = train_extreme[["Date", "SecuritiesCode", "label_bin"]].copy()

    # Prepare QNN angles
    n_qubits = 8
    angles_tr_all, angles_all, meta_all, train_mask = prepare_qnn_angles(feat_df, feature_cols, cutoff_date, n_qubits=n_qubits)
    # Pick only labeled training samples
    idx_all = feat_df[["Date", "SecuritiesCode"]].copy()
    idx_train = idx_all[train_mask].merge(labels_df, on=["Date", "SecuritiesCode"], how="inner")
    key_all = idx_all["Date"].astype(str) + "_" + idx_all["SecuritiesCode"].astype(str)
    key_train = set(idx_train["Date"].astype(str) + "_" + idx_train["SecuritiesCode"].astype(str))
    sel_mask = key_all.isin(key_train).values
    angles_tr = angles_tr_all[sel_mask[train_mask]] if angles_tr_all.shape[0] == train_mask.sum() else angles_tr_all
    y_tr = idx_train["label_bin"].values.astype(np.float32)

    # Train true 2-layer QNN
    model = train_qnn_classifier(angles_tr, y_tr, n_qubits=n_qubits, n_layers=2, num_epochs=50, batch_size=512, lr=2e-3)

    # Predict probabilities -> per-day z-score as scores
    proba = predict_qnn_proba(model, angles_all, batch_size=2048)
    scored = meta_all.copy()
    scored["proba"] = proba
    scored["score"] = scored.groupby("Date")["proba"].transform(lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-6))

    # Select long/short
    selected = rank_and_select(scored, universe, top_n=200, bottom_n=200)

    # Evaluate OOS only
    selected_oos = selected[selected["Date"] >= cutoff_date]
    daily_ls = evaluate_long_short(selected_oos, target_col="Target")
    sharpe = compute_sharpe(daily_ls)

    # Write outputs
    out_sel = out_dir / "daily_selection.csv"
    out_ls = out_dir / "daily_long_short_returns.csv"
    out_summary = out_dir / "summary.txt"
    out_scored = out_dir / "scored.csv"
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
    main(parse_args())


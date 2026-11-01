from pathlib import Path
from typing import Any, List, Optional, Tuple
import argparse
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


# ----------------------------
# Paths and IO helpers
# ----------------------------
DATA_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/data")
TRAIN_PRICES = DATA_DIR / "train_files/stock_prices.csv"
STOCK_LIST = DATA_DIR / "stock_list.csv"
OUTPUT_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/outputs") / "qtcnn"


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
    dest_stock_list = subset_dir / "stock_list.csv"
    dest_stock_list.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(STOCK_LIST, dest_stock_list)
    except Exception:
        pass


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


def make_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int = 20) -> Tuple[np.ndarray, pd.DataFrame]:
    seqs = []
    rows = []
    for code, g in df.groupby("SecuritiesCode", sort=False):
        g = g.sort_values("Date")
        values = g[feature_cols].values
        dates = g["Date"].values
        if len(g) < seq_len:
            continue
        for i in range(seq_len - 1, len(g)):
            x = values[i - seq_len + 1 : i + 1]
            if not np.isfinite(x).all():
                continue
            seqs.append(x)
            rows.append((pd.to_datetime(dates[i]), int(code)))
    X = np.asarray(seqs, dtype=np.float32)
    idx = pd.DataFrame(rows, columns=["Date", "SecuritiesCode"])
    return X, idx


# ----------------------------
# QCNN model and training
# ----------------------------
def get_qcnn_device(n_qubits: int) -> Any:
    return qml.device("default.qubit", wires=n_qubits)


def conv_layer(params, wires):
    # Pairwise conv; skip leftover if odd count
    for i in range(0, len(wires) - 1, 2):
        w1, w2 = wires[i], wires[i + 1]
        qml.RY(params[0], wires=w1)
        qml.RY(params[1], wires=w2)
        qml.CNOT(wires=[w1, w2])
        qml.RZ(params[2], wires=w1)
        qml.RZ(params[3], wires=w2)
        qml.CNOT(wires=[w2, w1])
        qml.RY(params[4], wires=w1)
        qml.RY(params[5], wires=w2)


def pool_layer(wires):
    return wires[::2]


def make_qcnn_qnode(n_qubits: int, n_layers: int) -> Tuple[qml.QNode, torch.Tensor]:
    dev = get_qcnn_device(n_qubits)
    max_layers = int(np.floor(np.log2(max(1, n_qubits))))
    effective_layers = max(0, min(n_layers, max_layers))

    def qcnn_block(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        active_wires = list(range(n_qubits))
        # Only apply up to effective_layers conv+pool steps
        for li in range(effective_layers):
            if len(active_wires) < 2:
                break
            w = weights[li]
            conv_layer(w, active_wires)
            active_wires = pool_layer(active_wires)
        # Ensure at least one wire remains to measure
        if len(active_wires) == 0:
            active_wires = [0]
        return active_wires[0]

    @qml.qnode(dev, interface="torch")
    def qnode(x, weights):
        active_wire = qcnn_block(x, weights)
        return qml.expval(qml.PauliZ(active_wire))

    init_weights = 0.01 * torch.randn(n_layers, 6)
    return qnode, init_weights


class QuantumTemporalQCNN(nn.Module):
    def __init__(self, n_qubits: int = 8, n_layers: int = 3, beta: float = 1.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.beta = beta
        self.qnode, init_w = make_qcnn_qnode(n_qubits, n_layers)
        self.weights = nn.Parameter(init_w)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_scores = []
        for i in range(x.shape[0]):
            s = self.qnode(x[i], self.weights)
            batch_scores.append(s)
        scores = torch.stack(batch_scores)
        probs = torch.sigmoid(self.beta * scores)
        # Ensure float32 for downstream losses
        return probs.float(), scores.float()


class HybridQTCNN(nn.Module):
    def __init__(self, n_qubits: int = 8, n_layers: int = 3, beta: float = 1.0, classical_dim: int = 8):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.beta = beta
        self.qnode, init_w = make_qcnn_qnode(n_qubits, n_layers)
        self.weights = nn.Parameter(init_w)
        # small MLP head: concat [quantum_score, classical_features] -> logits
        in_dim = int(classical_dim) + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, n_qubits] classical compressed features (same vector used by QCNN)
        batch_scores = []
        for i in range(x.shape[0]):
            s = self.qnode(x[i], self.weights)  # scalar
            batch_scores.append(s)
        scores = torch.stack(batch_scores).float()  # [B]
        feats = torch.cat([scores.unsqueeze(1), x.float()], dim=1)  # [B, 1+n_qubits]
        logits = self.mlp(feats).squeeze(1)  # [B]
        return logits.float(), scores


def prepare_qtcnn_inputs(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    cutoff_date: pd.Timestamp,
    n_qubits: int = 8,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, StandardScaler, PCA]:
    train_mask = feat_df["Date"] < cutoff_date
    X_train = feat_df.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=min(n_qubits, X_train_scaled.shape[1]), random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_train_pca = np.clip(X_train_pca, -3.0, 3.0)

    X_all = feat_df[feature_cols].fillna(0).values.astype(np.float32)
    X_all_scaled = scaler.transform(X_all)
    X_all_pca = pca.transform(X_all_scaled).astype(np.float32)
    X_all_pca = np.clip(X_all_pca, -3.0, 3.0)

    meta_all = feat_df[["Date", "SecuritiesCode", "Close", "Volume", "SupervisionFlag", "Target"]].copy()
    return X_train_pca, X_all_pca, meta_all, train_mask.values, scaler, pca


def train_qtcnn_classifier(
    X: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 50,
    batch_size: int = 128,
    n_qubits: int = 8,
    n_layers: int = 3,
    beta: float = 1.0,
    lr: float = 1e-3,
) -> HybridQTCNN:
    device = torch.device("cpu")
    model = HybridQTCNN(n_qubits=n_qubits, n_layers=n_layers, beta=beta, classical_dim=int(X.shape[1])).to(device)
    model = model.float()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))

    max_train = 10000
    if X_t.shape[0] > max_train:
        idx = torch.randperm(X_t.shape[0])[:max_train]
        X_t = X_t[idx]
        y_t = y_t[idx]
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(num_epochs):
        total = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad()
            logits, _ = model(xb)
            logits = logits.to(dtype=torch.float32)
            yb = yb.to(dtype=torch.float32)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"epoch {epoch+1}: bce_loss={total / max(n, 1):.6f}")
    model.eval()
    return model


def predict_qtcnn_proba(model: HybridQTCNN, X: np.ndarray) -> np.ndarray:
    device = torch.device("cpu")
    preds = []
    bs = 256
    with torch.no_grad():
        for i in tqdm(range(0, len(X), bs), desc="Predict QTCNN", dynamic_ncols=True):
            xb = torch.from_numpy(X[i : i + bs]).to(device=device, dtype=torch.float32)
            logits, _ = model(xb)
            probs = torch.sigmoid(logits)
            preds.append(probs.to(torch.float32).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


# ----------------------------
# Ranking, evaluation, outputs
# ----------------------------
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
    group.add_argument("--date-stride", type=int, default=None, help="Sample every N-th trading day (e.g., 11 for ~9%)")
    group.add_argument("--date-fraction", type=float, default=None, help="Sample fraction of trading days per year (e.g., 0.1)")
    p.add_argument("--subset-save-dir", type=str, default=None, help="If set, save the downsampled dataset to this directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed for fraction-based sampling")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--n-qubits", type=int, default=8, help="Number of qubits / PCA components")
    p.add_argument("--n-layers", type=int, default=3, help="QCNN convolution layers")
    p.add_argument("--beta", type=float, default=1.0, help="Sigmoid scaling for probabilities")
    return p.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    global DATA_DIR, TRAIN_PRICES, STOCK_LIST, OUTPUT_DIR
    if args is None:
        args = parse_args()
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
        TRAIN_PRICES = DATA_DIR / "train_files/stock_prices.csv"
        STOCK_LIST = DATA_DIR / "stock_list.csv"
    OUTPUT_DIR = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    ensure_output_dir(OUTPUT_DIR)

    prices = read_prices()
    prices, sampled_dates = downsample_prices_by_dates(
        prices,
        date_stride=args.date_stride if hasattr(args, "date_stride") else None,
        date_fraction=args.date_fraction if hasattr(args, "date_fraction") else None,
        seed=args.seed if hasattr(args, "seed") else 42,
    )
    if (hasattr(args, "date_stride") and args.date_stride) or (hasattr(args, "date_fraction") and args.date_fraction):
        print(f"Downsampled to {len(sampled_dates)} trading days, rows={len(prices)}")
        if hasattr(args, "subset_save_dir") and args.subset_save_dir:
            write_subset_dataset(Path(args.subset_save_dir), prices)
            print(f"Saved subset dataset to: {args.subset_save_dir}")

    universe = read_universe()
    prices = add_adjusted_close(prices)
    prices = add_multi_scale_features(prices)
    feat_df, feature_cols = build_feature_matrix(prices)

    train_df, valid_df, cutoff_date = split_train_valid(feat_df)
    labels_df = select_outlier_training_samples(train_df, target_col="Target", per_side=200)
    labels_df = labels_df[["Date", "SecuritiesCode", "label_bin"]].copy()

    # Prepare QCNN inputs via PCA compression to n_qubits
    X_tr_vec, X_all_vec, meta_all, train_mask, scaler, pca = prepare_qtcnn_inputs(
        feat_df, feature_cols, cutoff_date, n_qubits=args.n_qubits
    )
    # Align training vectors with labeled index (Date, SecuritiesCode)
    idx_all = feat_df[["Date", "SecuritiesCode"]].copy()
    idx_train = idx_all[train_mask].merge(labels_df, on=["Date", "SecuritiesCode"], how="inner")
    key_all = idx_all["Date"].astype(str) + "_" + idx_all["SecuritiesCode"].astype(str)
    key_train = set(idx_train["Date"].astype(str) + "_" + idx_train["SecuritiesCode"].astype(str))
    sel_mask = key_all.isin(key_train).values
    X_tr = X_all_vec[sel_mask]
    idx_tr = idx_all[sel_mask].merge(idx_train, on=["Date", "SecuritiesCode"], how="left")
    y_tr = idx_tr["label_bin"].fillna(0).values.astype(np.float32)

    model = train_qtcnn_classifier(
        X_tr,
        y_tr,
        num_epochs=int(args.epochs),
        batch_size=128,
        n_qubits=int(args.n_qubits),
        n_layers=int(args.n_layers),
        beta=float(args.beta),
        lr=1e-3,
    )

    proba = predict_qtcnn_proba(model, X_all_vec)
    scored = meta_all.copy()
    scored["proba"] = proba
    scored["score"] = scored.groupby("Date")["proba"].transform(lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-6))

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
    main(parse_args())

from pathlib import Path
from typing import List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml


DATA_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/data")
TRAIN_PRICES = DATA_DIR / "train_files/stock_prices.csv"
STOCK_LIST = DATA_DIR / "stock_list.csv"
OUTPUT_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/outputs") / "qasa"


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
    prices["log_vol"] = np.log1p(prices["Volume"])  # handle zeros
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
    # Keep a compact set of features for sequence model
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
    # Return X: [N, T, F], and index rows (Date, SecuritiesCode)
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


def train_qasa_classifier(
    X: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 2,
    batch_size: int = 64,
    hidden_dim: int = 64,
    n_qubits: int = 8,
    q_layers: int = 2,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = X.shape[1]
    F = X.shape[2]

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (q_layers, n_qubits)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    class QuantumLayer(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, n_qubits)
            self.norm = nn.LayerNorm(n_qubits)
            self.output_proj = nn.Linear(n_qubits, output_dim)
        def forward(self, x):
            # x: [B*T, H]
            x_proj = torch.tanh(self.input_proj(x))
            x_proj = self.norm(x_proj)
            embs = []
            for i in range(x_proj.shape[0]):
                e = qlayer(x_proj[i])
                embs.append(e)
            E = torch.stack(embs, dim=0)
            return self.output_proj(E)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(F, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
            self.encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dropout=0.1)
            self.q_layer = QuantumLayer(hidden_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            # x: [B,T,F]
            h = torch.tanh(self.embed(x))
            h = self.norm(h)
            h = self.encoder(h)
            B, Tn, H = h.shape
            h_flat = h.reshape(B * Tn, H)
            q_out = self.q_layer(h_flat).view(B, Tn, H)
            last = q_out[:, -1, :]
            logits = self.head(last).squeeze(-1)
            return logits

    model = Model().to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y.astype(np.float32))
    # Subsample for speed
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


def predict_qasa_proba(model, X: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = torch.from_numpy(X[i : i + 64]).to(device)
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

    # Load and feature engineering
    prices = read_prices()
    universe = read_universe()
    prices = add_adjusted_close(prices)
    prices = add_multi_scale_features(prices)
    feat_df, feature_cols = build_feature_matrix(prices)

    # Outlier selection for labels
    train_df, valid_df, cutoff_date = split_train_valid(feat_df)
    labels_df = select_outlier_training_samples(train_df, target_col="Target", per_side=200)
    labels_df = labels_df[["Date", "SecuritiesCode", "label_bin"]].copy()

    # Build sequences for all and join labels
    seq_len = 20
    X_all, idx_all = make_sequences(feat_df, feature_cols, seq_len=seq_len)
    all_with_meta = idx_all.merge(
        feat_df[["Date", "SecuritiesCode", "Close", "Volume", "SupervisionFlag", "Target"]],
        on=["Date", "SecuritiesCode"],
        how="left",
    )

    # Training set: only samples before cutoff and present in labels
    train_mask = (idx_all["Date"] < cutoff_date)
    idx_train = idx_all[train_mask].merge(labels_df, on=["Date", "SecuritiesCode"], how="inner")
    # Align X with idx_train
    # Build a boolean mask for rows present in idx_train
    key = idx_all["Date"].astype(str) + "_" + idx_all["SecuritiesCode"].astype(str)
    key_train = set(idx_train["Date"].astype(str) + "_" + idx_train["SecuritiesCode"].astype(str))
    sel_mask = key.isin(key_train).values
    X_tr = X_all[sel_mask]
    # Fetch labels in the same order
    idx_tr = idx_all[sel_mask].merge(idx_train, on=["Date", "SecuritiesCode"], how="left")
    y_tr = idx_tr["label_bin"].fillna(0).values.astype(np.float32)

    # Train QASA classifier
    model = train_qasa_classifier(X_tr, y_tr, num_epochs=50, batch_size=64, hidden_dim=64, n_qubits=8, q_layers=2, lr=1e-3)

    # Predict probabilities on all sequences
    proba = predict_qasa_proba(model, X_all)
    scored = all_with_meta.copy()
    scored["proba"] = proba
    scored["score"] = scored.groupby("Date")["proba"].transform(lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-6))

    # Select per day
    selected = rank_and_select(scored, universe, top_n=200, bottom_n=200)

    # Evaluate OOS only
    selected_oos = selected[selected["Date"] >= cutoff_date]
    daily_ls = evaluate_long_short(selected_oos, target_col="Target")
    sharpe = compute_sharpe(daily_ls)

    # Write outputs
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



from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import lightgbm as lgb


DATA_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/data")
TRAIN_PRICES = DATA_DIR / "train_files/stock_prices.csv"
STOCK_LIST = DATA_DIR / "stock_list.csv"
OUTPUT_DIR = Path("/media/meow/Transcend/kaggle_jpx_stock/outputs") / "pqc_lgbm"


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
            return g
        return pd.concat([g.head(per_side), g.tail(per_side)], axis=0)
    return df.groupby("Date", group_keys=False).apply(_pick)


def pca_fit_transform(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # Economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:n_components].T  # [D, k]
    Z = Xc @ W  # [N, k]
    return Z, W, mu


def pca_transform(X: np.ndarray, W: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (X - mu) @ W


def train_pqc_embedder(X: np.ndarray, y: np.ndarray, n_wires: int = 8, n_layers: int = 2, num_epochs: int = 3, batch_size: int = 256, lr: float = 5e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        # AngleEmbedding + entangling layers
        qml.AngleEmbedding(inputs, wires=range(n_wires), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_wires))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

    weight_shapes = {"weights": (n_layers, n_wires)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes, init_method=torch.nn.init.uniform_)

    class PQCEmbedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            self.head = nn.Linear(n_wires, 1)
        def forward(self, xb):
            # xb: [B, n_wires]
            E = self.qlayer(xb)
            pred = self.head(E).squeeze(-1)
            return E, pred

    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = PQCEmbedder().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for epoch in range(num_epochs):
        total = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            _, pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"epoch {epoch+1}: huber_loss={total/max(n,1):.6f}")
    model.eval()
    return model


def embed_with_pqc(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    embs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32))
            E, _ = model(xb)
            embs.append(E.detach().cpu().numpy())
    return np.concatenate(embs, axis=0)


def select_outlier_labels(df: pd.DataFrame, per_side: int = 300) -> pd.DataFrame:
    def _pick(g: pd.DataFrame) -> pd.DataFrame:
        g = g.dropna(subset=["Target"]).sort_values("Target")
        if g.shape[0] <= per_side * 2:
            return g.assign(label_int=0)
        bottom = g.head(per_side).assign(label_int=0)
        top = g.tail(per_side).assign(label_int=1)
        return pd.concat([bottom, top], axis=0)
    return df.groupby("Date", group_keys=False).apply(_pick)


def train_lgbm_ranker(train: pd.DataFrame, feature_cols: List[str]):
    X = train[feature_cols]
    y = train["label_int"]
    groups = train.groupby("Date").size().tolist()
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, group=groups)
    return model


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

    # Load & features
    prices = read_prices()
    universe = read_universe()
    prices = add_adjusted_close(prices)
    prices = add_multi_scale_features(prices)
    feat_df, feature_cols = build_feature_matrix(prices)

    # Split train/valid cutoff by time (80/20)
    dates = np.sort(feat_df["Date"].unique())
    cut_idx = int(len(dates) * 0.8)
    cutoff_date = pd.to_datetime(dates[cut_idx])

    # Outlier selection for PQC training
    train_df = feat_df[feat_df["Date"] < cutoff_date].copy()
    train_extreme = select_outlier_training_samples(train_df, target_col="Target", per_side=300)

    # Prepare PCA -> PQC inputs
    X_tr_raw = train_extreme[feature_cols].fillna(0).values.astype(np.float32)
    y_tr = train_extreme["Target"].values.astype(np.float32)
    # Limit training samples for speed
    max_train = 60000
    if X_tr_raw.shape[0] > max_train:
        idx = np.random.RandomState(42).choice(X_tr_raw.shape[0], size=max_train, replace=False)
        X_tr_raw = X_tr_raw[idx]
        y_tr = y_tr[idx]

    d_embed = 8
    Z_tr, W, mu = pca_fit_transform(X_tr_raw, n_components=d_embed)

    # Train PQC embedder (predict Target via Huber; embeddings are intermediate)
    pqc_model = train_pqc_embedder(Z_tr, y_tr, n_wires=d_embed, n_layers=2, num_epochs=3, batch_size=256, lr=5e-3)

    # Build embeddings for full universe
    X_all_raw = feat_df[feature_cols].fillna(0).values.astype(np.float32)
    Z_all = pca_transform(X_all_raw, W, mu).astype(np.float32)
    E_all = embed_with_pqc(pqc_model, Z_all, batch_size=512)
    emb_cols = [f"qemb_{i}" for i in range(E_all.shape[1])]
    emb_df = pd.DataFrame(E_all, columns=emb_cols)

    # Merge embeddings back
    base_all = feat_df[["Date", "SecuritiesCode", "Close", "Volume", "SupervisionFlag", "Target"]].reset_index(drop=True)
    full_emb = pd.concat([base_all, emb_df], axis=1)

    # Train LGBM ranker on embeddings (outlier labels)
    labeled = select_outlier_labels(full_emb[full_emb["Date"] < cutoff_date], per_side=300)
    ranker = train_lgbm_ranker(labeled, emb_cols)

    # Score all
    scores = ranker.predict(full_emb[emb_cols])
    scored = full_emb.copy()
    scored["score"] = scores

    # Select long/short per day
    selected = rank_and_select(scored, universe, top_n=200, bottom_n=200)

    # Evaluate on OOS
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
    main()



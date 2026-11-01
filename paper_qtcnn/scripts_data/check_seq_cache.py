"""Spot-check a sequence memmap against a fresh recomputation for a few securities."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from qtcnn_research import data as D
cache, seq_dir = sys.argv[1], Path(sys.argv[2])
df = pd.read_parquet(cache); feat = D.FeatureSpec().columns()
X = np.load(seq_dir / "X_seq20.npy", mmap_mode="r"); idx = pd.read_parquet(seq_dir / "index_seq20.parquet")
print("X", X.shape, "index", len(idx), "| zero rows in first 5000:", int((np.abs(X[:5000]).sum(axis=(1, 2)) == 0).sum()),
      "| zero rows in last 5000:", int((np.abs(X[-5000:]).sum(axis=(1, 2)) == 0).sum()))
rng = np.random.default_rng(0); codes = rng.choice(df.SecuritiesCode.unique(), 6, replace=False)
sub = df[df.SecuritiesCode.isin(codes)]
Xs, ids = D.make_sequences(sub, feat, seq_len=20)
key = idx.reset_index().merge(ids.reset_index(), on=["Date", "SecuritiesCode"], suffixes=("_c", "_r"))
diff = np.abs(np.asarray(X[key["index_c"].to_numpy()]) - Xs[key["index_r"].to_numpy()]).max()
print(f"checked {len(key)} windows from {len(codes)} securities: max abs diff = {diff:.2e}")
sys.exit(0 if (diff < 1e-6 and len(key) > 0) else 1)

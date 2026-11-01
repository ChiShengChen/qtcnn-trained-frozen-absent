"""Build the normalised feature frame once and cache it as parquet."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from qtcnn_research.resources import claim_cpu, hide_gpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data",
                        help="directory holding the Kaggle JPX competition files (train_files/, stock_list.csv)")
    parser.add_argument("--out", default="cache/features.parquet")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    claim_cpu(args.threads)
    hide_gpu()

    import warnings

    warnings.filterwarnings("ignore")
    from qtcnn_research import data as D

    t0 = time.perf_counter()
    df = D.build_dataset(Path(args.data_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"rows={len(df):,}  cols={df.shape[1]}  secs={df.SecuritiesCode.nunique()}")
    print(f"dates={df.Date.nunique()}  {df.Date.min().date()} -> {df.Date.max().date()}")
    print(f"features={len(D.FeatureSpec().columns())}")
    print(f"elapsed={time.perf_counter()-t0:.1f}s  saved={out}")


if __name__ == "__main__":
    main()

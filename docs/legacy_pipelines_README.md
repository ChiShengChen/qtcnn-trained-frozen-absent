## JPX Stock Pipelines - Subset Workflow and How to Run

### Overview
- This repo contains multiple pipelines that train models on daily cross-sectional features and evaluate a long/short strategy:
  - QNN: `scripts/qnn_pipeline.py`
  - QLSTM: `scripts/qlstm_pipeline.py`
  - Transformer: `scripts/transformer_pipeline.py`

You can optionally downsample trading days to speed up development. The recommended subset for quick iteration is ~8–10% of trading days (every 10–12 days pick 1 day). This preserves within-day cross-sectional ranking behavior while reducing runtime.

### Directory layout (key paths)
- Data (original, full): `/media/meow/Transcend/kaggle_jpx_stock/data`
  - `train_files/stock_prices.csv`
  - `stock_list.csv`
- Example subset data (downsampled): `/media/meow/Transcend/kaggle_jpx_stock/data_subset_stride11`
  - `train_files/stock_prices.csv` (only sampled trading days)
  - `stock_list.csv`
- Outputs (examples):
  - QNN: `/media/meow/Transcend/kaggle_jpx_stock/outputs/qnn_stride11/`
  - QLSTM: `/media/meow/Transcend/kaggle_jpx_stock/outputs/qlstm_stride11/`
  - Transformer: `/media/meow/Transcend/kaggle_jpx_stock/outputs/transformer_stride11/`

Each pipeline writes:
- `daily_selection.csv`
- `daily_long_short_returns.csv`
- `scored.csv`
- `summary.txt` (Sharpe, OOS days, and date range)

### Create a subset dataset (about 9% of trading days)
This will sample every 11th trading day and save a new dataset to a separate folder. It also runs the QNN pipeline on that subset in one go.

```bash
python /media/meow/Transcend/kaggle_jpx_stock/scripts/qnn_pipeline.py \
  --date-stride 11 \
  --subset-save-dir /media/meow/Transcend/kaggle_jpx_stock/data_subset_stride11 \
  --output-dir /media/meow/Transcend/kaggle_jpx_stock/outputs/qnn_stride11
```

Notes:
- You should see a message like: “Downsampled to <N> trading days, rows=<M>” and “Saved subset dataset to: ...”
- The subset folder will contain a self-contained copy:
  - `/media/meow/Transcend/kaggle_jpx_stock/data_subset_stride11/train_files/stock_prices.csv`
  - `/media/meow/Transcend/kaggle_jpx_stock/data_subset_stride11/stock_list.csv`

Alternative sampling:
- Fractional (stratified by year): `--date-fraction 0.10` (≈10% per year)
- Random seed for fraction: `--seed 42`

Show CLI help:
```bash
python /media/meow/Transcend/kaggle_jpx_stock/scripts/qnn_pipeline.py --help
```

### Run pipelines on the subset dataset
Use `--data-dir` to point pipelines to the subset, and choose separate output directories.

- QNN:
```bash
python /media/meow/Transcend/kaggle_jpx_stock/scripts/qnn_pipeline.py \
  --data-dir /media/meow/Transcend/kaggle_jpx_stock/data_subset_stride11 \
  --output-dir /media/meow/Transcend/kaggle_jpx_stock/outputs/qnn_stride11
```

- QLSTM:
```bash
python /media/meow/Transcend/kaggle_jpx_stock/scripts/qlstm_pipeline.py \
  --data-dir /media/meow/Transcend/kaggle_jpx_stock/data_subset_stride11 \
  --output-dir /media/meow/Transcend/kaggle_jpx_stock/outputs/qlstm_stride11
```

- Transformer:
```bash
python /media/meow/Transcend/kaggle_jpx_stock/scripts/transformer_pipeline.py \
  --data-dir /media/meow/Transcend/kaggle_jpx_stock/data_subset_stride11 \
  --output-dir /media/meow/Transcend/kaggle_jpx_stock/outputs/transformer_stride11
```

### About the subset contents
- It keeps all stocks within a sampled day (to preserve cross-sectional statistics and ranking).
- It reduces the number of trading days (e.g., stride 11 ≈ 9%), which cuts runtime substantially.
- Expected `scored.csv` row count decreases roughly in proportion to the sampling ratio (e.g., ~210k rows vs. 2.3M+ full).

### Tips
- If using multiline shell commands, ensure the backslash `\` is the last character on the line (no trailing spaces).
- Review `summary.txt` in each output directory for quick KPIs (Sharpe, OOS days, start/end dates).

### Help for other pipelines
```bash
python /media/meow/Transcend/kaggle_jpx_stock/scripts/qlstm_pipeline.py --help
python /media/meow/Transcend/kaggle_jpx_stock/scripts/transformer_pipeline.py --help
```

# JPX Stock Prediction Pipelines

This repository contains pipelines for JPX Tokyo Stock Exchange prediction, including classical and quantum-inspired models.

## Project Structure

- `scripts/`: Training and evaluation pipelines
  - `lgbm_ranker_pipeline.py`, `lstm_pipeline.py`, `transformer_pipeline.py`, `mlp_pipeline.py`, `pqc_lgbm_pipeline.py`, `qasa_pipeline.py`, `qlstm_pipeline.py`
  - `qsvm_pipeline.py`
- `data/`: Place the JPX dataset here (ignored by git)
  - `train_files/`, `supplemental_files/`, `example_test_files/`, `stock_list.csv`
- `outputs/`: Model outputs, reports, and summaries (ignored by default)

data : https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction 

## Requirements

- Python 3.9+
- Recommended: Conda environment
- Key libraries: pandas, numpy, scikit-learn, torch, lightgbm, pennylane, matplotlib, tqdm

Example (conda):

```bash
conda create -n jpx python=3.10 -y
conda activate jpx
pip install -r requirements.txt  # If provided, otherwise install dependencies manually
```

If no `requirements.txt` is provided, install the basics:

```bash
pip install pandas numpy scikit-learn torch lightgbm pennylane matplotlib tqdm
```

## Data

Download the JPX Tokyo Market Prediction dataset from Kaggle and place files under `data/` as follows (paths used by scripts):

- `data/train_files/stock_prices.csv`
- `data/supplemental_files/stock_prices.csv`
- `data/stock_list.csv`

The entire `data/` directory and common archive formats (`*.zip`, `*.tar.gz`, etc.) are ignored by git via `.gitignore`.

## How to Run

From the project root:

```bash
# Classical baselines
python scripts/lgbm_ranker_pipeline.py
python scripts/lstm_pipeline.py
python scripts/transformer_pipeline.py
python scripts/mlp_pipeline.py

# Quantum-inspired
python scripts/pqc_lgbm_pipeline.py
python scripts/qasa_pipeline.py
python scripts/qlstm_pipeline.py
python scripts/qsvm_pipeline.py
```

Each pipeline reads from `data/` and writes results under `outputs/`. See `summary.txt`, `daily_selection.csv`, `daily_long_short_returns.csv` in each model's folder.

## Compare Models

```bash
python scripts/compare_oos_metrics.py
```

This expects `scored.csv` and daily returns CSVs under each model's `outputs/<model>/` folder.

## Results Comparison

Aligned OOS: 2020-12-21 to 2021-12-03
Training period (for aligned comparison): up to 2020-12-20 (time-based 80% split per model)

| Model | IC mean | ICIR | Monthly Sharpe |
| --- | ---: | ---: | ---: |
| LGBM | 0.006037 | 0.086 | -0.123 |
| Transformer | -0.007999 | -0.119 | 0.145 |
| LSTM | 0.001210 | 0.025 | 0.317 |
| MLP | 0.013016 | 0.117 | -0.325 |
| QNN-LGBM | 0.005736 | 0.126 | -0.221 |

Raw report is at `outputs/compare/report.txt`.

Note: QLSTM is now included in the comparison script. Run `python scripts/compare_oos_metrics.py` to regenerate the report and metrics (IC/ICIR and monthly Sharpe) including QLSTM.

## Metrics: IC and ICIR

- **IC (Information Coefficient)**: Cross-sectional correlation each date between factor scores and next-period returns. Common choices:
  - **Pearson IC**: linear correlation
  - **Spearman IC**: rank correlation (more robust; often preferred in practice)
- **ICIR (IC Information Ratio)**: Stability of IC over time, defined as mean(IC) divided by std(IC); can be annualized.

Formulas:
- Per-date IC: \( \mathrm{IC}_t = \mathrm{corr}(\text{score}_{t-1},\ \text{return}_t) \)
- Non-annualized: \( \mathrm{ICIR} = \dfrac{\mu(\mathrm{IC})}{\sigma(\mathrm{IC})} \)
- Annualized (daily): \( \mathrm{ICIR}_{\text{ann}} = \mathrm{ICIR} \times \sqrt{252} \)

Example (pandas):

```python
# df columns: Date, SecuritiesCode, score, fwd_return
ic_daily = df.groupby('Date').apply(
    lambda g: g['score'].corr(g['fwd_return'], method='spearman')
)
ic = ic_daily.mean()
icir = ic_daily.mean() / ic_daily.std(ddof=1)
icir_annualized = icir * np.sqrt(252)
```

## Notes

- You may see pandas deprecation warnings; functionality is unaffected but APIs may change in future versions.
- Large data and outputs are ignored by git by default. To version outputs, remove `outputs/` from `.gitignore`.
- For GPU training with PyTorch, ensure CUDA is available and install the correct torch build.


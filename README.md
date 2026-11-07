# Trained, Frozen, or Absent

Code, results and manuscript for

> C.-S. Chen, X. Zhang, E.-J. Kuo, R. Fu, Q. Xie and F. Zhang, *Trained, Frozen, or Absent: Identifying What a
> Quantum Layer Contributes to Cross-Sectional Equity Ranking* (2026).

The study trains a hybrid quantum temporal convolutional network for daily
cross-sectional ranking of Tokyo Stock Exchange equities under six control
conditions (trained circuit, frozen circuit, parameter-matched classical
surrogate, no circuit, and the circuit alone, trained and frozen), a second ansatz family, five classical
baselines including LightGBM, a walk-forward protocol with an embargo and a
locked holdout, and an evaluation stack with bootstrap intervals, Deflated
Sharpe Ratios, backtest-overfitting probabilities, Romano-Wolf correction and
equivalence (TOST) tests. It also re-scores the authors' earlier published
QTCNN result over a longer window.

## Layout

| path | contents |
| --- | --- |
| `paper_qtcnn/qtcnn_research/` | the study: data, walk-forward folds, batched quantum layer and ablation conditions, models, training, evaluation statistics, equivalence tests, aggregation, figures |
| `paper_qtcnn/run_all.sh` | every experiment stage, as submitted to the job scheduler |
| `paper_qtcnn/results/` | per-cell metrics and daily return series for every (model, fold, seed) on both markets, the holdout, the full-budget control and the second ansatz; trained weights (`states/`, `states_rerun/`); circuit diagnostics including the shot budget on the trained cells. `*_2026-09_original/` holds the first, non-reproducible September runs that the appendix compares against |
| `paper_qtcnn/manuscript/` | LaTeX source, generated tables, figures and the compiled PDF |
| `paper_qtcnn/notes/findings.md` | the lab notebook, including the corrections made along the way |
| `paper_qtcnn/tests/` | calibration tests for the evaluation statistics |
| `paper_qtcnn/scripts_data/` | US panel construction, the delisting audit, the shot-budget measurement, the spanning analysis and the September-versus-November comparison |
| `scripts/` | the legacy pipelines re-evaluated in the paper, kept as they were |
| `docs/` | notes on the dataset and the legacy pipelines |

`paper_qtcnn/README.md` documents the protocol, the split-adjustment bug in
the legacy pipelines, and the compute policy.

## Data

The Tokyo panel is the public Kaggle competition dataset
[JPX Tokyo Stock Exchange Prediction](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction).
It is not redistributed here; download it from Kaggle and point
`paper_qtcnn/qtcnn_research/build_cache.py --data-dir` at it. The US panel is
rebuilt from public daily prices by `paper_qtcnn/scripts_data/fetch_us.py`.

## Reproducing

```bash
cd paper_qtcnn
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m qtcnn_research.build_cache --data-dir /path/to/jpx
./run_all.sh tier0              # trained and frozen circuit
./run_all.sh controls           # surrogate, no circuit, TCN
./run_all.sh baselines          # LSTM, transformer, MLP
./run_all.sh quantumonly        # circuit only, trained
./run_all.sh quantumonly_frozen # circuit only, frozen
./run_all.sh lightgbm
./run_all.sh nostop             # full-budget control (last-epoch weights)
./run_all.sh reupload           # second ansatz family
./run_all.sh states             # trained weights for the shot budget
./run_all.sh shots_trained      # shot budget on the trained cells
./run_all.sh aggregate          # tables under manuscript/tables
.venv/bin/python -m qtcnn_research.figures
```

Every table and figure in the manuscript is produced by `aggregate.py`,
`figures.py` and the scripts in `scripts_data/` from the files under
`results/`. Training is deterministic: re-running a stage reproduces its
per-cell results bit for bit; the test suite under `tests/`
checks the evaluation statistics against synthetic data with known answers.

## License

MIT (see `LICENSE`).

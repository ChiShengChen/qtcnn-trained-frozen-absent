# QTCNN paper workspace

Target: a submission-grade study of hybrid quantum temporal CNNs for
cross-sectional equity ranking on the JPX Tokyo Stock Exchange data, aimed at a
journal in the IF>4 band (Expert Systems with Applications, Engineering
Applications of AI, Knowledge-Based Systems, EPJ Quantum Technology).

This directory is self-contained. It does not modify anything under
`../scripts/` or `../outputs/`.

## Why a rewrite rather than more runs of the legacy pipelines

Re-scoring the existing `outputs/` with a correct metric stack
(`results/legacy_reeval.csv`) shows the published numbers cannot support any
claim:

| issue | evidence |
| --- | --- |
| Sharpe was never annualised | `compute_sharpe` in `scripts/qtcnn_pipeline.py` returns `mean/std`; qtcnnV2's reported 0.113 is 1.80 annualised |
| headline runs used 22 out-of-sample days | `outputs/qtcnn_subset_*/summary.txt`; some transformer runs used 2 days |
| no model is significant | qtcnnV2 Sharpe 1.80, bootstrap 95% CI [-0.03, 3.79], p = 0.07, DSR = 0.54 |
| costs were never charged | qtcnnV2 break-even is 7.4 bps; at 10 bps its Sharpe is -0.63 |
| IC and Sharpe disagree | qtcnnV2 has the *lowest* IC (t = 0.42) and the *highest* Sharpe of the nine models |
| split adjustment runs the wrong way | see below |

### The split-adjustment bug

`add_adjusted_close` in every legacy pipeline accumulates `AdjustmentFactor`
forward in time. In this dataset a factor of 0.5 on day *t* means prices up to
and including *t* must be halved. Measured over the 610 securities that have a
non-unit factor:

| adjustment | daily moves >30% | >50% | max abs move |
| --- | ---: | ---: | ---: |
| raw, unadjusted | 718 | 565 | 20.1 |
| forward cumprod (legacy code) | 1418 | 1091 | 23.4 |
| **backward cumprod (correct)** | **16** | **3** | **1.12** |

The legacy adjustment is worse than doing nothing, and it contaminates every
momentum and volatility feature.

## What this workspace adds

* `qtcnn_research/resources.py` — CPU thread budgeting, low-priority
  scheduling, a load gate, and `hide_gpu()` so a CUDA-enabled torch cannot take
  VRAM the scheduler never granted.
* `qtcnn_research/data.py` — correct back-adjustment, 21 cross-sectionally
  normalised features, vectorised sliding windows.
* `qtcnn_research/walkforward.py` — rolling-origin folds with an embargo, plus a
  locked 2021-12-06..2022-06-24 holdout that model selection never sees.
* `qtcnn_research/evaluation.py` — annualised risk metrics, Newey-West IC
  inference, stationary-bootstrap Sharpe CIs, Deflated Sharpe Ratio, CSCV
  probability of backtest overfitting, Diebold-Mariano, Romano-Wolf stepdown,
  turnover and cost curves.
* `qtcnn_research/quantum.py` — batched QCNN (33-72x faster than the legacy
  per-sample loop, agreeing to 1e-6) and the four ablation conditions.
* `qtcnn_research/models.py` — hybrid QTCNN and parameter-matched classical
  references.

## Compute policy

One RTX 3090 is shared across many sessions, and 24 CPU cores are shared with
several long-running jobs. So:

1. **GPU work goes through the `gq` scheduler**, never straight to the card.
   `run_all.sh` submits every stage with an honest resource declaration —
   `--cpu-only` for the quantum stages, which are CPU-bound, and `-g 4G` for the
   classical baselines, the only stage that benefits from a GPU. Use
   `QUEUE=1 ./run_all.sh <stage>` to queue and walk away.
2. `claim_cpu(threads)` caps BLAS/torch threads and drops priority.
3. `hide_gpu()` is the default, so a CUDA-enabled torch cannot take VRAM that
   was not allocated to us.
4. `wait_for_cpu()` yields while other people's jobs have the box
   oversubscribed; measured load has reached 57 on 24 cores, under which our
   own throughput collapses by two orders of magnitude.

## Status

- [x] Environment, resource guards, feature cache (2.48M rows, 2017-02..2022-06)
- [x] Evaluation stack, calibration-tested
- [x] Walk-forward protocol (4 folds x 125 test days = 500 OOS days;
      a fifth fold would need 1205 dev days and only 1161 exist)
- [x] Batched quantum layer and ablation conditions
- [x] Tier 0: 8 conditions x 4 folds x 5 seeds, 500 OOS days -- no condition beats zero or the no-circuit control; trained == frozen
- [x] Tier 1: no barren plateau to 12 qubits; rank robust to gate noise; shot budget on the trained cells: 8192 shots for 0.99 rank corr at the median cell, >131072 at the narrowest (F37, F40)
- [x] Tier 2: spanning regression (R^2 0.91, no alpha; `scripts_data/spanning.py`); extension fetched; its 207 missing names all matched to JPX delisting lists (204 acquisitions) — `scripts_data/delisting_audit.py`, F25
- [x] Locked holdout, 135 days, opened three times (Sept networks, Oct LightGBM, Nov reproducible rerun): ablation replicates (trained == frozen), nothing survives RW/DSR/10 bps
- [x] Manuscript: all sections written; intro & related work built from the verified literature corpus (`literature/report.md`, 216 cards, 51 papers); builds to 18 pages, 0 errors
- [x] Second market (US, 2000 equities, 625 OOS days): rerun after the price-floor fix; all eight conditions negative, nothing significant, equivalence replicates at the 0.03 margin (F28, F32, F33)
- [x] Ablation restated as an equivalence test (TOST + minimum detectable effect), `qtcnn_research/equivalence.py`, 4 calibration tests (F33)
- [x] Six-reviewer pass (F40) and reproducible rerun of every network result with the committed code (F41); September runs archived in `results/*_2026-09_original/`
- [x] Circuit-only, no-early-stopping and LightGBM conditions reported (F38); second ansatz family run and reported with its looser equivalence bound (F39); abstract at ~350 words; 27 pages
- [x] Literature search (paper-deep-search): 35 queries + 2 snowball rounds, 2107 ledger rows, 471 included, saturation reached; `manuscript/refs_generated.bib` generated from ledger metadata only

#!/usr/bin/env bash
# Full experiment plan, in the order results are needed for the manuscript.
#
# Every stage goes through the gq scheduler so this shares the machine properly:
# GPU work waits for VRAM instead of competing for it, and CPU-only work is
# declared as such. Resource declarations are what gq admits on, so they are
# sized from measurement, not guessed upwards:
#
#   tier0/holdout  --cpu-only   the quantum simulator is CPU-bound; measured
#                               RSS peaks near 5 GB while the sequence memmap
#                               is written, ~3 GB afterwards.
#   baselines      -g 4G        LSTM/Transformer/MLP at batch 512, T=20, F=21,
#                               d_model 64. Well under 2 GB in practice; 4 G is
#                               the small-model band with headroom for the CUDA
#                               context. Re-check with `gq status` and tighten.
#   diagnostics    --cpu-only   circuit simulation only, no financial data.
#
# Each stage checkpoints per (model, fold, seed) and resumes if interrupted.
#
# Usage:  ./run_all.sh <stage>          run it here, scheduled  (gq run)
#         QUEUE=1 ./run_all.sh <stage>  queue it and return     (gq submit)
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python
THREADS=${THREADS:-4}
NICE=${NICE:-15}
MAXLOAD=${MAXLOAD:-1.5}
PRIO=${PRIO:-0}
QUEUE=${QUEUE:-0}
mkdir -p logs results

# gq_run <name> <gq resource args...> -- <command...>
gq_run() {
  local name=$1; shift
  if [[ "$QUEUE" == "1" ]]; then
    gq submit -n "$name" -p "$PRIO" "$@"
  else
    gq run -n "$name" -p "$PRIO" "$@" 2>&1 | tee "logs/${name}.log"
  fi
}

COMMON=(--threads "$THREADS" --nice "$NICE" --max-load "$MAXLOAD")
ABLATION=(qtcnn_trained qtcnn_frozen qtcnn_surrogate qtcnn_noquantum tcn)

# 25 (fold, seed) cells per model. Measured throughput on this box is
# 1358 samples/s training and 5255 samples/s inference for the circuit, so the
# two conditions that backpropagate through it cost ~15-25 min per cell while
# the other three cost ~2 min. Splitting them into separate gq jobs lets the
# scheduler overlap the cheap ones with the expensive ones instead of running
# 20 hours in series. Jobs share results/walkforward/returns (filenames are
# unique per model/fold/seed) and write separate metrics files via --tag.
#
# n_trials is fixed at 25 = 5 conditions x 5 seeds for every job, so the
# Deflated Sharpe Ratio is deflated by the whole study rather than one slice.
N_TRIALS=55   # eleven conditions x five seeds; aggregate.py is also run with --n-trials 55

stage_tier0() {
  # The two expensive conditions: gradients flow through the circuit.
  gq_run qtcnn-tier0-quantum --cpu-only -c "$THREADS" -m 12 -- \
    $PY -u -m qtcnn_research.run_experiment \
      --models qtcnn_trained qtcnn_frozen --seeds 0 1 2 3 4 \
      --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
      --tag quantum --out results/walkforward
}

stage_tier0_controls() {
  # The three cheap control conditions.
  gq_run qtcnn-tier0-controls --cpu-only -c "$THREADS" -m 12 -- \
    $PY -u -m qtcnn_research.run_experiment \
      --models qtcnn_surrogate qtcnn_noquantum tcn --seeds 0 1 2 3 4 \
      --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
      --tag controls --out results/walkforward
}

stage_baselines() {
  # Classical references. Run on the CPU like everything else: the reported
  # numbers come from CPU training, which is bit-for-bit reproducible; GPU
  # kernels are not, and would give different cells (add --gpu only for speed).
  gq_run qtcnn-baselines --cpu-only -c "$THREADS" -m 12 -- \
    $PY -u -m qtcnn_research.run_experiment \
      --models lstm transformer mlp --seeds 0 1 2 3 4 \
      --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
      --tag baselines --out results/walkforward
}

stage_diagnostics() {
  # Circuit-level analyses: scaling, barren plateaus, expressibility, noise.
  gq_run qtcnn-diagnostics --cpu-only -c "$THREADS" -m 8 -- \
    $PY -u -m qtcnn_research.diagnostics "${COMMON[@]}" \
      --out results/diagnostics
}

stage_holdout() {
  # The locked window, 2021-12-06 to 2022-06-24, for the eight network conditions
  # (LightGBM: stage holdout_lightgbm). Opened three times; see the protocol
  # section. Re-running it reproduces the reported cells exactly.
  gq_run qtcnn-holdout --cpu-only -c "$THREADS" -m 12 -- \
    $PY -u -m qtcnn_research.run_experiment --holdout \
      --models "${ABLATION[@]}" lstm transformer mlp --seeds 0 1 2 3 4 \
      --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
      --tag holdout --out results/holdout
}

stage_aggregate() {
  # Cheap and interactive; no scheduling needed.
  $PY -m qtcnn_research.aggregate --results results/walkforward \
    --tables manuscript/tables --n-trials "$N_TRIALS" | tee logs/aggregate.log
}


# ---- Second market: US common stocks over the same calendar window ---------
# Same protocol, same code path, same n_trials; only the cache/output paths
# differ. Build the cache first (scripts_data/fetch_us.py, build_us_cache.py).
US=(--cache cache/us/features.parquet --seq-cache cache/us/sequences)

stage_us_quantum() {
  gq_run qtcnn-us-quantum --cpu-only -c "$THREADS" -m 12 -- \
    $PY -u -m qtcnn_research.run_experiment "${US[@]}" \
      --models qtcnn_trained qtcnn_frozen --seeds 0 1 2 3 4 \
      --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
      --tag quantum --out results/us_walkforward
}

stage_us_controls() {
  gq_run qtcnn-us-controls --cpu-only -c "$THREADS" -m 12 -- \
    $PY -u -m qtcnn_research.run_experiment "${US[@]}" \
      --models qtcnn_surrogate qtcnn_noquantum tcn --seeds 0 1 2 3 4 \
      --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
      --tag controls --out results/us_walkforward
}

stage_us_baselines() {
  gq_run qtcnn-us-baselines --cpu-only -c "$THREADS" -m 12 -- \
    $PY -u -m qtcnn_research.run_experiment "${US[@]}" \
      --models lstm transformer mlp --seeds 0 1 2 3 4 \
      --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
      --tag baselines --out results/us_walkforward
}

stage_us_aggregate() {
  $PY -m qtcnn_research.aggregate --results results/us_walkforward \
     --tables manuscript/tables/us --market "the US panel" --label-suffix=-us --n-trials "$N_TRIALS" | tee logs/us_aggregate.log
}


stage_nostop() {
  # No-early-stopping control: the full 30-epoch budget for both circuit
  # conditions, so the trained-equals-frozen result cannot be attributed to
  # early stopping having halted training before the circuit moved.
  gq_run qtcnn-nostop --cpu-only -c "$THREADS" -m 12 -- \
    $PY -u -m qtcnn_research.run_experiment \
      --models qtcnn_trained qtcnn_frozen --seeds 0 1 2 3 4 \
      --epochs 30 --patience 30 --final-weights --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
      --tag nostop --out results/walkforward_nostop
}


stage_quantumonly() {
  # The fifth condition declared in the method section: the head sees the circuit
  # output alone, which bounds how much ranking information the measured
  # observable carries by itself.
  gq_run qtcnn-quantumonly --cpu-only -c "$THREADS" -m 12 -- \
    $PY -u -m qtcnn_research.run_experiment \
      --models qtcnn_quantumonly --seeds 0 1 2 3 4 \
      --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
      --tag quantumonly --out results/walkforward
}

stage_quantumonly_frozen() {
  # Circuit-only with the circuit held at its initialisation: the arm that makes
  # QuantumOnly a trained-vs-frozen contrast in its own right (review, 2026-11).
  for seed in 0 1 2 3 4; do
    gq_run qtcnn-qonly-frz-s$seed --cpu-only -c "$THREADS" -m 12 -t 40m -- \
      $PY -u -m qtcnn_research.run_experiment \
        --models qtcnn_quantumonly_frozen --seeds $seed \
        --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
        --tag quantumonly_frozen_s$seed --out results/walkforward
  done
}

stage_states() {
  # Re-train the trained-circuit condition cell by cell and keep the weights, so
  # the shot budget can be measured on the encoder the model actually deployed
  # (F37 measured it on an initialised encoder). Results land in their own
  # directory and are checked against results/walkforward for determinism.
  for seed in 0 1 2 3 4; do
    gq_run qtcnn-states-s$seed --cpu-only -c "$THREADS" -m 12 -t 40m -- \
      $PY -u -m qtcnn_research.run_experiment \
        --models qtcnn_trained --seeds $seed \
        --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
        --tag states_s$seed --out results/walkforward_states --save-state results/states
  done
}

stage_shots_trained() {
  # Shot budget at the deployed input distribution, one measurement per trained cell.
  gq_run qtcnn-shots-trained --cpu-only -c "$THREADS" -m 8 -t 90m -- \
    $PY -u scripts_data/shot_budget_real_scores.py --threads "$THREADS" \
      --states results/states/qtcnn_trained_f*_s*.pt --n-dates 3 --per-date 300 \
      --shots 256 1024 8192 32768 131072 \
      --out results/diagnostics/shot_budget_trained.csv
}

stage_reupload() {
  # Second ansatz family (data re-uploading, CNOT ring, multi-wire Z): the
  # trained-vs-frozen contrast on JPX, split by seed so four jobs share the box.
  # Results go to their own directory; the no-circuit control is copied in at
  # aggregation time so the contrasts against it can be computed there.
  for seed in 0 1 2 3 4; do
    gq_run "qtcnn-reup-s$seed" --cpu-only -c "$THREADS" -m 12 -- \
      $PY -u -m qtcnn_research.run_experiment --ansatz reupload_hea \
        --models qtcnn_trained qtcnn_frozen --seeds "$seed" \
        --epochs 30 --max-train 60000 --n-trials "$N_TRIALS" "${COMMON[@]}" \
        --tag "reup_s$seed" --out results/walkforward_reup
  done
}

stage_lightgbm() {
  # Gradient-boosted baseline on the flattened window, same folds and seeds.
  gq_run qtcnn-lightgbm --cpu-only -c 8 -m 16 -- \
    $PY -u -m qtcnn_research.run_lightgbm --threads 8 --nice "$NICE" --max-load "$MAXLOAD" \
      --n-trials "$N_TRIALS" --out results/walkforward
}

stage_us_lightgbm() {
  gq_run qtcnn-us-lgbm --cpu-only -c 8 -m 16 -- \
    $PY -u -m qtcnn_research.run_lightgbm --threads 8 --nice "$NICE" --max-load "$MAXLOAD" \
      --n-trials "$N_TRIALS" "${US[@]}" --out results/us_walkforward
}

stage_holdout_lightgbm() {
  gq_run qtcnn-ho-lgbm --cpu-only -c 8 -m 16 -- \
    $PY -u -m qtcnn_research.run_lightgbm --threads 8 --nice "$NICE" --max-load "$MAXLOAD" \
      --n-trials "$N_TRIALS" --holdout --out results/holdout
}

case "${1:-all}" in
  reupload)    stage_reupload ;;
  quantumonly_frozen) stage_quantumonly_frozen ;;
  states)      stage_states ;;
  shots_trained) stage_shots_trained ;;
  lightgbm)    stage_lightgbm ;;
  us_lightgbm) stage_us_lightgbm ;;
  holdout_lightgbm) stage_holdout_lightgbm ;;
  tier0)       stage_tier0 ;;
  controls)    stage_tier0_controls ;;
  baselines)   stage_baselines ;;
  diagnostics) stage_diagnostics ;;
  holdout)     stage_holdout ;;
  aggregate)   stage_aggregate ;;
  us_quantum)  stage_us_quantum ;;
  us_controls) stage_us_controls ;;
  us_baselines) stage_us_baselines ;;
  us_aggregate) stage_us_aggregate ;;
  nostop)      stage_nostop ;;
  quantumonly) stage_quantumonly ;;
  us_all)      QUEUE=1 stage_us_quantum; QUEUE=1 stage_us_controls; QUEUE=1 stage_us_baselines; echo "queued; watch with: gq status" ;;
  all)         QUEUE=1 stage_tier0; QUEUE=1 stage_tier0_controls; \
               QUEUE=1 stage_baselines; QUEUE=1 stage_diagnostics; \
               echo "queued; watch with: gq status" ;;
  *) echo "unknown stage: $1"; exit 1 ;;
esac

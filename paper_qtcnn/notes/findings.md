# Findings log

Running record of results that go into the manuscript. Each entry names the
command that produced it so the number can be regenerated.

## F1. The legacy Sharpe was never annualised
`scripts/qtcnn_pipeline.py:437` returns `mean/std` of daily long-short returns.
Multiplying by sqrt(252) turns the reported qtcnnV2 figure of 0.113 into 1.80.
Every summary.txt in `../outputs/` is on the un-annualised scale, so none of
them are comparable with published Sharpe ratios.

## F2. No model in the legacy comparison is statistically distinguishable from zero
Re-scored on the 241-day aligned window (`results/legacy_reeval.csv`):

| model | Sharpe (ann.) | 95% CI | p | DSR | IC t (NW) |
| --- | ---: | --- | ---: | ---: | ---: |
| qtcnnV2 | 1.80 | [-0.03, 3.79] | 0.070 | 0.54 | 0.42 |
| transformer | 0.46 | [-0.88, 2.04] | 0.531 | 0.11 | 0.65 |
| qlstm | 0.13 | [-1.42, 1.75] | 0.854 | 0.06 | 2.37 |
| qnn | -0.43 | [-2.31, 1.42] | 0.664 | 0.02 | 2.10 |
| lstm | -0.59 | [-2.31, 1.27] | 0.495 | 0.01 | 1.05 |
| qcnn_lstm | -2.15 | [-4.22, -0.17] | 0.038 | 0.00 | 0.70 |

DSR is computed with `n_trials=12`, the number of model variants in the repo.
A defensible claim needs DSR > 0.95; the best model reaches 0.54.

## F3. IC and Sharpe point in opposite directions
qtcnnV2 has the *lowest* IC of the nine models (0.0016, t = 0.42) and the
*highest* Sharpe. The models with genuinely significant IC (MLP t = 2.57,
QLSTM t = 2.37, PQC t = 2.21) all have negative or near-zero Sharpe. A ranking
signal that does not survive into the tails of the cross-section is the most
likely explanation, and it is worth a paragraph in the discussion: it means
long/short top-200 selection is not the right readout for this signal strength.

## F4. Nothing survives transaction costs
qtcnnV2 turnover averages 24% of the book per day and its break-even cost is
7.4 bps. At 10 bps its Sharpe falls from +1.80 to -0.63. Turnover for
qcnn_lstm and pqc_lgbm is 76% and 79% per day, giving negative break-even.
This is the single most damaging omission in the original work.

## F5. The split adjustment runs the wrong way
See README. Forward cumulative product leaves 1418 daily moves above 30%
across the 610 adjusted securities, against 718 for raw unadjusted prices and
16 for the correct backward product. The legacy adjustment is worse than no
adjustment, and it feeds every momentum and volatility feature.

Verified directly on security 6861: `AdjustmentFactor` 0.5 is stamped on
2017-01-17 while the close halves between 2017-01-17 (82600) and 2017-01-18
(40920), so the factor applies to prices up to and including the stamped day.

## F6. The quantum layer can be batched for a 46.7x speedup
Parameter broadcasting through `default.qubit` reproduces the per-sample loop
to within 1e-6 and runs 46.7x faster at 8 qubits, 3 layers, batch 256
(15.74 s -> 0.34 s per forward+backward). `lightning.qubit` with adjoint
differentiation is slower here (2.95 s) because the register is small.
This is what makes full-sample walk-forward affordable, and it is the reason
the legacy runs fell back to 9% date subsets and 22-day evaluation windows.

## F7. Machine contention dominates naive timings
A benchmark run at 1-minute load average 57 on 24 cores reported 16.4 s per
training step for the hybrid model; the same step measured 0.13 s once traced
on a quiet machine. Any wall-clock number in the paper must record the load at
measurement time. `resources.wait_for_cpu` gates the experiments on this.

## Open questions to settle with the sweep
1. Does the trained circuit beat the frozen circuit? If not, it is a random
   feature and the paper must say so.
2. Does the parameter-matched classical surrogate match the circuit? If it
   does, there is no quantum contribution to claim.
3. Is `qtcnn_noquantum` distinguishable from any of them at 500 OOS days?
4. Does anything at all survive 10 bps of cost?

## F8. The measured observable needs many shots to preserve a ranking
`gq run --cpu-only -- python -m qtcnn_research.diagnostics --which shots`
at 8 qubits, 3 layers, 256 random inputs:

| shots | RMSE vs analytic | Spearman rank corr. vs analytic |
| ---: | ---: | ---: |
| analytic | 0.0000 | 1.000 |
| 8192 | 0.0109 | 0.986 |
| 1024 | 0.0301 | 0.894 |
| 256 | 0.0618 | 0.834 |
| 64 | 0.1452 | 0.583 |

The strategy consumes a ranking, not a value, so rank correlation is the
quantity that matters. Preserving it above 0.98 costs roughly 8192 shots per
score. At 2000 securities per day that is 1.6e7 circuit executions per trading
day, which is the number a hardware-feasibility claim has to defend --- and it
is far more informative than reporting the analytic-simulator Sharpe alone.

## Scheduling
GPU arbitration on this machine belongs to the `gq` scheduler. All stages are
submitted through `run_all.sh`, which declares `--cpu-only` for the quantum
stages (the simulator is CPU-bound) and `-g 4G` for the classical baselines.
`resources.gpu_queue` is deprecated: polling for free VRAM alongside a
scheduler that admits on declared values would double-book the card.

## F9. The parameter-shared QCNN shows no barren plateau at this scale
`--which barren` (3 layers, 40 random initialisations per width, quick grid):

| qubits | gradient variance |
| ---: | ---: |
| 2 | 3.007e-03 |
| 4 | 1.686e-03 |
| 6 | 1.788e-03 |
| 8 | 1.643e-03 |

Flat, not exponentially decaying. Sharing one 6-parameter block across all
pairs in a layer keeps the parameter count at 6L regardless of register width,
and the measured observable is local, both of which are the conditions under
which barren plateaus are avoided. This is a substantive claim about the
architecture and supports the low-resource framing. It must be re-run on the
full grid (up to 12 qubits, 120 samples) before it goes in the paper, and the
fitted decay rate reported alongside.

## F10. Expressibility rises with width and depth as expected
`--which expressibility` (KL from the Haar fidelity distribution; lower is more
expressive): 4 qubits 0.237 (1 layer) / 0.175 (2 layers); 6 qubits 0.138 / 0.083.
Nothing surprising, but it establishes that the shared-parameter economy does
not come at the cost of a degenerate state distribution.

## F11. Uniform depolarising noise cannot reorder scores --- by construction
The first noise scan returned a Spearman rank correlation of exactly 1.0000 at
every error rate up to p=0.05, with RMSE growing linearly to 0.087. That is not
a robustness finding: uniform depolarising noise multiplies every expectation
value by the same factor, which is a monotone transform, so rank preservation
is an identity. Presenting it as evidence of robustness would be wrong.

The scan was therefore rewritten to cover three settings --- uniform (reported
for completeness, with the analytic explanation), per-wire heterogeneous rates,
and heterogeneous rates at a 1024-shot budget. Only the latter two can reorder
scores, and the shot-noise result (F8) suggests sampling, not gate error, is
the binding constraint on hardware.

## F12. The training-sample draw explains more variance than the architecture
Pilot, fold 0, 125 out-of-sample days, 10k of 192,400 labelled training
sequences drawn per seed:

| seed | qtcnn_trained | qtcnn_noquantum | tcn (matched) |
| ---: | ---: | ---: | ---: |
| 0 | 1.610 | 1.949 | 1.949 |
| 1 | 3.386 | 3.250 | 3.250 |

Within a seed the conditions are nearly indistinguishable; between seeds the
same condition moves by more than a full Sharpe unit. The seed controls only
which subset of the training pool is drawn, so the draw dominates the
architecture. Any single-seed comparison of these models is measuring the draw.

Separately, an earlier build of the same cell returned -0.160 where this one
returns +1.610. The only change was sorting the row indices before gathering
them from the sequence store, for I/O locality, which reorders the training
pool and therefore changes which 10k rows the subsample selects. Nothing about
the model or the data changed.

## F13. The parameter-matched control and the classical baseline are not the same thing
The first pilot reported identical Sharpe, IC, AUC, turnover and parameter
count for `qtcnn_noquantum` and `tcn`, because at the hybrid's width they are
the same network: TemporalConvEncoder(21 -> 8) followed by MLP(8 -> 64 -> 32 -> 1).

They fill different roles and must be configured differently. The matched
control (23,817 parameters) answers whether the circuit adds anything at equal
capacity. The baseline answers whether any of this beats a classical model
someone would actually deploy, and is therefore sized as one would size it for
the task (129,633 parameters: hidden 128, embedding 32, three dilated blocks).
Reporting a deliberately narrowed classical model as the baseline is the
failure mode this separation avoids -- and it is the criticism the legacy
comparison in `../outputs/compare/report.txt` is open to.

## F14. Gate noise cannot reorder the scores; sampling noise is the binding constraint
`--which depolarising` at 8 qubits, 3 layers, 128 random inputs. Spearman rank
correlation against the noiseless, infinite-shot circuit:

| setting | p=0 | p=0.01 | p=0.05 |
| --- | ---: | ---: | ---: |
| uniform depolarising, analytic | 1.0000 | 1.0000 | 1.0000 |
| heterogeneous depolarising, analytic | 1.0000 | 1.0000 | 0.9996 |
| heterogeneous depolarising, 1024 shots | 0.9880 | 0.9780 | 0.9859 |

Two things follow.

First, the analytic rows are an identity, not a robustness result. A
depolarising channel maps a Pauli expectation to $(1-p)^k\\langle Z\\rangle$, and
that factor does not depend on the input, so it rescales every score by the same
amount and preserves the ranking exactly. Giving each wire its own error rate
does not change this, because the factor is a property of the channel rather
than of the state. This is why the heterogeneous row is also 1.0.

Second, at a realistic 1024-shot budget the rank correlation is already 0.988
with *no* gate noise at all, and raising the depolarising rate from 0 to 0.05
moves it only to 0.986. Sampling dominates gate error by an order of magnitude
for this circuit. Combined with F8 --- roughly 8192 shots to hold the rank
correlation above 0.98, hence about 1.6e7 circuit executions per trading day
across 2000 securities --- the deployment barrier for this architecture is the
shot budget, not device fidelity.

The claim is bounded to unital noise and a single-Pauli observable. An
amplitude-damping scan is included as the control: it is non-unital, its effect
on <Z> is state-dependent, and it therefore can reorder. Without that control
the depolarising result would only show that the measurement lacks power.

## F15. The noise measurement has power: amplitude damping does bias the output
Control for F14. Amplitude damping is non-unital, so unlike depolarising it
cannot be written as an input-independent rescaling of <Z>.

| gamma | rank corr (analytic) | mean shift (analytic) | rank corr (1024 shots) |
| ---: | ---: | ---: | ---: |
| 0.000 | 1.0000 | +0.0000 | 0.9785 |
| 0.005 | 0.9996 | +0.0074 | 0.9912 |
| 0.010 | 0.9996 | +0.0148 | 0.9892 |
| 0.020 | 0.9996 | +0.0296 | 0.9857 |
| 0.050 | 0.9996 | +0.0734 | 0.9839 |

The mean shift grows linearly in gamma and is strictly positive: damping pulls
the state towards |0>, so <Z> is pushed towards +1. Depolarising produced no
mean shift at all. The measurement therefore distinguishes unital from
non-unital noise, which is what makes the F14 result meaningful rather than an
artefact of an insensitive test.

Even so, the reordering damping causes is small (0.9996), while 1024 shots
alone give 0.9785 with no gate noise whatsoever. The conclusion stands and is
now properly bounded: for this circuit and observable, ranking is robust to
gate noise of both unital and non-unital kinds, and the deployment constraint
is the sampling budget.

## F16. Walk-forward result: no condition is distinguishable from any other, or from zero
Partial sweep (controls complete over 4 folds / 500 OOS days; the quantum
conditions were still running at 375 and 250 days, so the pooled Sharpe values
below are NOT comparable across rows -- the Diebold-Mariano contrasts, which
align dates first, are).

| model | OOS days | Sharpe | 95% CI | p | DSR |
| --- | ---: | ---: | --- | ---: | ---: |
| qtcnn_frozen | 250 | 0.309 | [-1.28, 1.79] | 0.65 | 0.19 |
| qtcnn_noquantum | 500 | 0.013 | [-1.13, 1.15] | 0.98 | 0.12 |
| qtcnn_surrogate | 500 | -0.040 | [-1.17, 1.04] | 0.95 | 0.11 |
| qtcnn_trained | 375 | -0.276 | [-1.69, 1.01] | 0.66 | 0.07 |
| tcn | 500 | -0.237 | [-1.47, 0.98] | 0.67 | 0.07 |

Every interval spans zero and the best Deflated Sharpe Ratio is 0.19 against
the 0.95 a claim would need.

Diebold-Mariano against the no-quantum control, on aligned dates:
frozen p=0.29, surrogate p=0.79, trained p=0.66, tcn p=0.40. After Romano-Wolf
familywise correction every candidate sits at p=0.511. The trained circuit does
not beat the same network without it.

Two further numbers sharpen the picture.

**PBO = 0.80.** Combinatorially symmetric cross-validation puts the probability
of backtest overfitting at 0.80, well above the 0.5 of an uninformative
selection rule. Choosing the in-sample best among these five conditions is
worse than choosing at random out of sample.

**Break-even cost 3.4-4.2 bps.** Lower than the 7.4 bps of the legacy pipeline.
At 5 bps every condition is already negative; at 10 bps they run -1.2 to -2.9.

## F17. Training the circuit changes almost nothing
Fold 0, per seed, trained versus frozen (identical initialisation, gradients
disabled on the circuit only):

| seed | Sharpe trained | Sharpe frozen | IC trained | IC frozen |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.611 | 0.592 | 0.00840 | 0.00840 |
| 1 | 1.848 | 1.780 | 0.01155 | 0.01156 |
| 2 | 2.399 | 2.356 | 0.01356 | 0.01357 |

The information coefficients agree to five decimal places and the validation
AUCs are identical. Whatever the encoder and head learn, they learn it around a
circuit whose parameters are effectively fixed: the quantum layer is acting as
a random feature. This is the cleanest available answer to the question in the
paper's title, and it is a negative one.

## F18. Final walk-forward result: the quantum layer contributes nothing
Complete sweep. Four rolling-origin folds, five seeds, five conditions, all on
the same 500 out-of-sample trading days (2019-06-18 to 2021-08-05). The
development period holds 1161 trading days; a fifth fold would need 1205, so
four is the maximum this sample supports. Earlier notes in this file said 625
days, which assumed five folds and was wrong.

| model | Sharpe | seed sd | 95% CI | p | DSR |
| --- | ---: | ---: | --- | ---: | ---: |
| qtcnn_noquantum | 0.013 | 0.454 | [-1.14, 1.18] | 0.98 | 0.12 |
| qtcnn_surrogate | -0.040 | 0.324 | [-1.25, 1.05] | 0.95 | 0.11 |
| qtcnn_frozen | -0.071 | 0.304 | [-1.25, 1.04] | 0.90 | 0.10 |
| qtcnn_trained | -0.072 | 0.308 | [-1.25, 1.04] | 0.90 | 0.10 |
| tcn (baseline) | -0.237 | 0.260 | [-1.44, 0.93] | 0.70 | 0.07 |

Three readings, none of which needs a judgement call.

**The trained and frozen circuits are the same model in every respect that
matters.** Sharpe -0.0715 against -0.0707, over identical dates. Learning the
circuit parameters changes nothing, so the layer is a random feature.

**Nothing beats the control.** Diebold-Mariano against `qtcnn_noquantum` gives
p = 0.66, 0.79, 0.66 and 0.40; after Romano-Wolf correction every p is at least
0.69. Nothing is distinguishable from zero either: every confidence interval
spans it and the best Deflated Sharpe Ratio is 0.12 against the 0.95 a claim
would need.

**PBO = 0.75.** Selecting the in-sample best of these five conditions places it
below the out-of-sample median three times in four.

**Costs remove what little is left.** Break-even is 3.4 to 4.0 bps; at 5 bps
every condition returns between -1.0 and -1.5, at 10 bps between -2.3 and -2.9.
Turnover runs 49-57% of the book per day.

The seed standard deviations (0.26 to 0.45) are larger than every difference
between conditions, which is F12 restated on the full sample: on this task the
training draw matters and the architecture does not.

## F19. The extension carries survivorship bias that this source cannot fix
`gq run -- python scripts_data/fetch_extension.py` over 2021-11-01..2025-12-30
returned 3674 of the 4417 codes in `stock_list.csv`, covering **1793 of the
2000 Universe0 securities (89.6%)**. The reconstructed Target agrees with the
Kaggle Target on 283,294 overlapping rows at corr 0.99953, median absolute
difference 5.1e-17, with 0.001% of rows differing by more than 1e-4. The feed
itself is sound.

The 207 missing Universe0 names are the problem. 204 of them were still trading
in the Kaggle panel in June 2022, so they are not stale entries in the universe
list. Spot-checking six of them individually returns HTTP 404 "Quote not found"
rather than the timeouts that caused the earlier false negatives, so they are
genuinely absent from this source: almost certainly firms that delisted or
merged after 2022 and whose pages were withdrawn. (Resolved in F25: all 207 are on the JPX delisting lists; 204 are acquisitions, not failures.)

That makes the extension period a **surviving-firms-only sample**. Names that
left the market between 2022 and 2026 are absent, and delisting is not
independent of performance, so the extension is biased in the optimistic
direction by an amount this source gives no way to measure.

This matters more than usual here. The paper's argument is that quantum-finance
results are reported on protocols too weak to support them; publishing an
extension with a known, unquantified survivorship bias as a headline result
would undercut that argument. Two defensible options:

1. Use the extension only as a secondary robustness check, state the bias, and
   keep the 500-day walk-forward as the primary evidence.
2. Obtain J-Quants credentials. It is the same source the Kaggle panel came
   from, carries delisted securities, and would remove the bias entirely.

Option 2 is preferable and needs an account; option 1 is what the data in hand
supports.

## F20. The quantum track is spanned by the classical ones
Return-level spanning regression on the pooled 500-day track:

    qtcnn_trained ~ tcn + lstm + transformer + mlp
      alpha_ann = +0.024   t = 1.07   p = 0.284   R^2 = 0.936
      beta_tcn = 0.66 (t 14.6), beta_lstm = 0.30 (t 9.5),
      beta_transformer = 0.10 (t 2.1), beta_mlp = -0.07 (t -1.7)

    qtcnn_trained ~ qtcnn_noquantum
      alpha_ann = -0.010   t = -0.48   p = 0.632   R^2 = 0.939

Ninety-four percent of the hybrid's daily return variance is explained by
classical models, and the residual carries no significant alpha. Pairwise
return correlations run 0.89-0.97. A 50/50 blend with any classical model
stays negative. This closes the last positive framing available: the circuit
does not supply an orthogonal signal that would justify keeping it in an
ensemble even if it loses head to head.

## Full 8-model table, 500 OOS days
| model | Sharpe | seed sd | p | DSR |
| --- | ---: | ---: | ---: | ---: |
| qtcnn_noquantum | 0.013 | 0.454 | 0.98 | 0.08 |
| qtcnn_surrogate | -0.040 | 0.324 | 0.95 | 0.07 |
| qtcnn_frozen | -0.071 | 0.304 | 0.90 | 0.06 |
| qtcnn_trained | -0.072 | 0.308 | 0.90 | 0.06 |
| transformer | -0.139 | 0.214 | 0.82 | 0.05 |
| mlp | -0.149 | 0.360 | 0.80 | 0.05 |
| tcn | -0.237 | 0.260 | 0.70 | 0.04 |
| lstm | -0.329 | 0.186 | 0.57 | 0.03 |

DSR is now deflated by 40 trials (8 models x 5 seeds). The larger classical
baselines rank at the bottom, which is consistent with the sample being too
small to reward capacity, not with any model having found signal.

## F21. CORRECTION to F8 and F14: the shot budget was overstated by ~30x
The quick diagnostics (64 random inputs) reported rank correlation 0.894 at
1024 shots and 0.583 at 64. The full run (`results/diagnostics/shot_noise.csv`,
256 inputs, 120-sample grids) gives:

| shots | rank corr | RMSE |
| ---: | ---: | ---: |
| 8192 | 0.9996 | 0.0096 |
| 1024 | 0.9970 | 0.0287 |
| 256 | 0.9914 | 0.0568 |
| 64 | 0.9666 | 0.1173 |

The RMSE values agree between runs; the rank correlations do not, because the
quick run's 64 inputs happened to produce outputs clustered in a narrow band
(the ratio of output spread to shot noise sets the rank correlation, not the
noise alone). The full run is the reference. Consequences:

* "Roughly 8192 shots to hold rank correlation above 0.98" becomes **256
  shots for 0.99, 1024 for 0.997**. The per-day figure of 1.6e7 circuit
  executions for 2000 securities becomes **5e5 at 256 shots or 2e6 at 1024**.
* The qualitative conclusion of F14 stands and is, if anything, sharper: with
  heterogeneous depolarising noise at 1024 shots the rank correlation is
  0.997 flat across p = 0..0.05, so gate error is invisible next to sampling
  at a shot budget that is itself modest.
* F8's number must not appear in the manuscript.

## F22. CORRECTION to F15: the sign of the damping shift is not a finding
The quick run showed a positive mean shift under amplitude damping; the full
run shows a negative one (-0.026 at gamma 0.05). Damping pushes each damped
wire towards |0>, but the wire that is measured sits behind further two-qubit
gates whose parameters were drawn at random, so the sign of the shift on <Z>
depends on the draw. What is robust across both runs is that |mean shift|
grows linearly with gamma while depolarising gives exactly zero shift at every
rate: that is the unital/non-unital distinction, and it is what the control
was for. The manuscript reports magnitude only.

## Full-grid barren-plateau and expressibility results (job #74)
Gradient variance 2->12 qubits: 3.08e-3, 2.16e-3, 1.79e-3, 1.10e-3, 2.32e-3,
1.98e-3. Fitted slope of log-variance -0.036 per qubit; an exponential 2^-n
decay would give -0.69. Expressibility KL at 8 qubits: 0.188 / 0.126 / 0.078
for 1 / 2 / 3 layers.

## F23. Locked holdout (evaluated once): the ablation replicates, the sign does not
2021-12-06 to 2022-06-24, 135 trading days, 8 conditions x 5 seeds, trained on
everything before 2021-08-06 with the usual 60-day validation and 20-day
embargo. This window was not used for any design decision and is not re-run.

| model | Sharpe | seed sd | 95% CI | p (uncorr.) | p (RW vs none) | DSR | break-even bps |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| transformer | 2.34 | 0.30 | [0.31, 4.82] | 0.044 | 0.67 | 0.60 | -- |
| qtcnn_surrogate | 1.70 | 0.69 | [-0.25, 4.07] | 0.12 | 0.67 | 0.42 | 4.8 |
| lstm | 1.60 | 0.81 | [-0.38, 3.67] | 0.13 | 0.34 | 0.38 | 7.7 |
| qtcnn_trained | 1.54 | 0.96 | [-0.49, 3.89] | 0.17 | 0.63 | 0.37 | 5.6 |
| qtcnn_frozen | 1.53 | 0.92 | [-0.52, 3.86] | 0.18 | 0.63 | 0.37 | 5.5 |
| qtcnn_noquantum | 1.12 | 0.36 | [-0.79, 3.07] | 0.27 | -- | 0.26 | 5.9 |
| tcn | 0.72 | 0.29 | [-1.08, 2.70] | 0.47 | 0.63 | 0.18 | 2.0 |
| mlp | -0.31 | 0.69 | [-2.57, 1.88] | 0.79 | 0.14 | 0.05 | 2.8 |

Three readings.

**The ablation replicates out of time.** Trained and frozen differ by 0.02.
The parameter-matched classical surrogate is the best of the four hybrid
conditions. Every DM contrast against the no-circuit control has p > 0.40.

**Gross Sharpe rose for everything, which is the window, not the models.**
Seven of eight conditions are positive on a stretch where the cross-section
rewarded the momentum/liquidity features every model shares. Seed standard
deviations (0.29-0.96) are larger than in the walk-forward because 135 days
is short, and every interval but one spans zero.

**The one nominal significance is a multiple-comparison artefact.** The
transformer's uncorrected p = 0.044 becomes 0.67 after Romano-Wolf across the
eight candidates; its DSR is 0.60 against the 0.95 threshold; at 10 bps it is
negative. It is the best of eight draws on 135 days, which is what the
Deflated Sharpe Ratio exists to discount.

At 5 bps only three conditions remain positive (lstm +0.48, trained +0.13,
frozen +0.11); at 10 bps none.

## F24. CORRECTION to the §7.2 audit wording: the published paper states an annualised Sharpe

The reference paper (P333, arXiv 2512.06630) states "The primary metric is the out-of-sample
annualized Sharpe ratio" with an explicit sqrt(252) factor (papers/P333.txt L1166-1170) and reports
0.538 ± 0.042 over 5 runs (L1447), Ledoit-Wolf p = 0.027 (L1639), on a 22-day test window (L1442)
with a self-acknowledged power caveat (~45 days needed for 90% power, L1651-1652). The
*repository* we re-ran computes mean/std with no annualisation (0.33 on the 22-day subset).
The earlier manuscript text ("the ratio was not annualised") was therefore only true of the code,
not of the paper. §7.2 and the abstract now attribute the un-annualised figure to the public
implementation and keep the four window/seed/adjustment/encoder points, which stand either way.
The paper's own TConv-MLP ablation (0.412 ± 0.039, L1481) is now cited as evidence that most of
the reported edge sits in the classical encoder. Evidence cards P333-c8..c21 (all verified).

## F25. The 207 securities missing from the extension are all accounted for: 204 were acquired, not failed
`scripts_data/delisting_audit.py` matches the 207 Universe0 codes absent from the yfinance
extension (`cache/extension/universe_missing.csv`) against the JPX official delisting lists for
2022-2026 (`cache/extension/jpx_delisted/*.html`, 489 rows). Every one of the 207 is on the list
(0 unmatched), delisted in 2022 (22), 2023 (33), 2024 (44), 2025 (59) and 2026 (49).

| delisting reason | n |
| --- | ---: |
| acquisition / going-private (TOB, MBO, squeeze-out by share consolidation, subsidiarisation, merger) | 204 |
| failure of listing-maintenance standards | 1 |
| exchange-ordered (internal-control) | 1 |
| voluntary application | 1 |

So the survivorship bias in the extension runs in a specific direction: the missing names are
overwhelmingly takeover targets, which typically leave at a premium, not distressed firms that
leave at a loss. Excluding them removes positive terminal returns from whichever side of the book
held them, and since the strategy is a rank-based long-short, the sign of the net effect is not
determined a priori. This reverses the "almost certainly the ones that delisted [and lost]" reading
in F19. The audit does not recover prices, so the extension stays a secondary check; but the bias
is now characterised rather than assumed. Output: `cache/extension/universe_missing_delisted.csv`.

## F26. Second market: US common-stock panel built over the same calendar window
Universe candidates from the Nasdaq Trader symbol directories (snapshot 2026-09-13,
`cache/us/nasdaqlisted.txt`, `otherlisted.txt`): 4826 common stocks on NYSE (1612), NYSE
American (227) and Nasdaq (2987) after removing ETFs, test issues, warrants, units, rights,
preferreds, notes, ADRs, trusts/funds and SPAC shells by name pattern. `gq #878`
(`scripts_data/fetch_us.py`, yfinance, auto_adjust=False) returned 3935 symbols over
2016-09-01..2022-07-01; the 891 missing are overwhelmingly post-2022 listings ("Data doesn't
exist for startDate"), i.e. names that did not trade in the window, not fetch failures.

`scripts_data/build_us_cache.py` (`gq #879`): 4,503,596 raw rows over 1421 dates; 2635 symbols
had >= 200 trading days and a median close >= $2 in 2017; the 2000 with the highest median 2017
dollar volume form the universe (cut-off $413,763/day). Target = C_{t+2}/C_{t+1} - 1, the 21
JPX features, the same cross-sectional winsorise/z-score, all through `qtcnn_research.data`.
Cache: 2,796,919 rows, 1399 dates (2016-11-30..2022-06-22), 2000 securities
(`cache/us/features.parquet`, `cache/us/universe.csv`). Because the US calendar has more
trading days than the JPX one, the development period yields five walk-forward folds
(625 OOS days) instead of four; the holdout window (from 2021-12-06) is identical.

Survivorship: the symbol directory is a current snapshot, so firms delisted before 2026 are
absent. Every condition is trained and scored on the identical panel, so the bias is common to
all eight conditions and does not enter the ablation contrasts; absolute levels are not claimed.
Training queued as `gq #880` (trained/frozen), `#884` (surrogate/noquantum/tcn; #881 was cancelled because it raced #880 on the sequence cache, now written atomically), `#882`
(lstm/transformer/mlp), priority 5, `results/us_walkforward/`.
Sequence cache verified after #880 wrote it (`scripts_data/check_seq_cache.py`): 2,758,919
windows, no zero rows, 8280 windows from 6 securities recomputed with max abs diff 0. Folds:
0 test 2019-03-25..2019-09-19, 1 ..2020-03-19, 2 ..2020-09-16, 3 ..2021-03-17, 4 ..2021-09-14.

## F27. The second market replicates the null, with the same ablation ordering
US panel, identical protocol and code path, five walk-forward folds (625 pooled OOS days,
five seeds per condition, `gq #880` quantum, `#884` controls, `#882` baselines;
`results/us_walkforward/`, tables in `manuscript/tables/us/`).

| condition | Sharpe (ann.) | 95% CI | p | DSR | seed sd |
| --- | ---: | --- | ---: | ---: | ---: |
| transformer | -0.739 | [-1.45, +0.56] | 0.12 | 0.00 | 0.08 |
| lstm | -0.899 | [-2.09, -0.11] | 0.07 | 0.00 | 0.12 |
| qtcnn_surrogate | -0.913 | [-2.00, -0.01] | 0.06 | 0.00 | 0.13 |
| qtcnn_frozen | -0.932 | [-2.14, -0.11] | 0.06 | 0.00 | 0.14 |
| qtcnn_trained | -0.933 | [-2.14, -0.11] | 0.07 | 0.00 | 0.14 |
| tcn | -0.940 | [-2.44, -0.35] | 0.08 | 0.00 | 0.06 |
| qtcnn_noquantum | -0.949 | [-2.17, -0.24] | 0.05 | 0.00 | 0.11 |
| mlp | -0.973 | [-2.31, -0.37] | 0.05 | 0.00 | 0.30 |

Four readings.

**The ablation replicates exactly.** Per-cell (fold, seed) Sharpe correlation between TRAINED and
FROZEN is 0.9996 (mean |difference| 0.022, max 0.181); per-cell IC correlation is 1.0000; pooled
daily-return correlation is 1.000 to three decimals, and 0.9992 for both SURROGATE and NONE.
Every DM contrast against the no-quantum control is insignificant (trained p = 0.583, frozen
p = 0.560, surrogate p = 0.128, tcn p = 0.848); only the transformer reaches nominal significance
(p = 0.027) and it does not survive Romano-Wolf (p_RW = 0.411). No contrast survives correction.

**The sign flipped, the ordering did not.** Every condition is negative on this window, where JPX
was weakly positive: a long-short book on US cross-sectional momentum/liquidity features lost money
over 2019-03..2021-09, which spans the COVID crash and the 2021 reversal. This is the window, not
the models -- the eight conditions move together (all correlations >= 0.99). It is further evidence
that what these models track is one common signal, not eight different ones.

**Costs are unchanged as a verdict.** Break-even is 4.6-6.2 bps for every condition; at 5 bps every
Sharpe is below -0.8, at 10 bps below -1.5. The gross numbers are already negative, so cost is not
what kills it here; it is that there is nothing to charge cost against.

**PBO is low because everything is equally bad.** CSCV gives PBO = 0.056 over 252 partitions, far
below the JPX value, since with all conditions negative and near-identical there is little for the
in-sample selection to get wrong.

Caveat carried into the paper: the US universe comes from a current symbol-directory snapshot and
therefore excludes pre-2026 delistings. All eight conditions share the identical panel, so the bias
is common to them and does not enter any contrast; absolute levels on this market are not claimed.

## F28. CORRECTION to F27: the first US panel was contaminated by one impossible security-day
Adversarial review of the second-market result found that 2020-11-18 carries a seed-averaged
long-short return of -1.289 (-128.9%) for every condition, which a dollar-neutral 200/200 book
cannot produce. It is the only day in 625 with |r| > 0.15; the next largest is 0.048.

Root cause: CHRD (Chord Energy, formerly Oasis Petroleum) emerged from Chapter 11 in November
2020. The price source splices the post-emergence entity onto the pre-emergence ticker, so the
close of $0.12 on 2020-11-18 is followed by the new entity's price and the computed forward
return is +25733% (Target = 257.33). Universe selection could not catch it: CHRD qualified on
2017 liquidity and only collapsed later.

Effect on the first US run (`results/us_walkforward_contaminated/`, retained for the record):
the day inflated the pooled daily standard deviation 4.7x (0.0527 vs 0.0111) and supplied about
two thirds of the sample mean. Every level statistic on that panel is invalid --- Sharpe ratios
(e.g. tcn -0.940 -> -1.691, transformer -0.738 -> -0.622, and the ordering changes), max drawdown
(printed as -1.20 to -1.25, i.e. below -100%, because the compounded equity curve went negative),
DSR (skew -23.3, kurtosis 570.6 drove every value to ~0), and the "every pairwise correlation
>= 0.99" claim, which falls to 0.616 once the shared bad day is removed. The Diebold-Mariano and
Romano-Wolf contrasts do survive, because the day cancels in the difference series (transformer
DM p 0.0266 -> 0.0277), so the ablation conclusion itself was not affected.

Fix: `scripts_data/build_us_cache.py` now applies a point-in-time price floor of $1, the standard
penny-stock screen, which is also what removes bankruptcy-reorganisation tickers. It drops 4670
security-days (0.17%) across 49 symbols and takes max |Target| from 257.3 to 8.88. The remaining
extremes are genuine events (KODK +318% on the 2020 loan announcement, AMC +301% in the January
2021 squeeze, MCRB +389% on a trial readout) that a real portfolio would have experienced.
Panel is now 2,792,249 rows. Re-run queued as gq #934/#935/#936.

The JPX panel needs no such screen and has none: over 2,483,470 security-days its largest
|Target| is 0.618 and nothing exceeds 100%. The backward split adjustment of F2 holds up.

Lesson for the paper: the same class of bug it documents in the reference implementation (Sec. 3.1)
appeared in our own second market, and was caught only by an explicit outlier audit. The
protocol section should require one.

## F29. CORRECTION to F6: the batching speed-up is batch-size dependent and the agreement is exact
F6 recorded "46.7x faster, agreement to 1e-6" from an ad-hoc measurement that left no artefact on
disk, which an audit of the manuscript's numbers correctly flagged as unverifiable.
`scripts_data/bench_batched_circuit.py` now measures it reproducibly (gq #940, n=8, L=3, 18
parameters, median of three repeats after warm-up; `results/diagnostics/batching_benchmark.csv`):

| batch | per-sample loop (s) | batched (s) | speed-up | max abs diff |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.765 | 0.023 | 33.1x | 0 |
| 256 | 3.218 | 0.050 | 64.2x | 0 |
| 512 | 8.635 | 0.120 | 71.7x | 0 |

Two corrections. The speed-up is not a single number: it grows with batch size as the fixed
per-call overhead is amortised, from 33x at 64 to 72x at 512. And the agreement is not 1e-6 but
exactly zero in double precision, since both paths evaluate the same QNode. The manuscript now
reports the table and cites the script; the 46.7x figure and the 1e-6 tolerance are withdrawn.

## F30. Three statistics were computed wrongly; correcting them strengthens the null
Adversarial review found three defects in our own evaluation code. All three flattered the
result, so correcting them makes the paper's case stronger, not weaker.

**Deflated Sharpe Ratio was deflated by 8, not 40.** `aggregate.py` called `headline_table`
without `n_trials`, which defaulted to `len(models)` = 8, while the table caption printed
`len(head) * n_seeds` = 40, and `run_all.sh` passed a third value (25) to the per-cell metrics.
Corrected to a single source of truth (models x seeds = 40):

| | before (n=8) | after (n=40) |
| --- | ---: | ---: |
| walk-forward best (qtcnn_noquantum) | 0.075 | **0.015** |
| holdout transformer | 0.60 | **0.32** |
| holdout qtcnn_surrogate | 0.42 | 0.18 |

**Break-even cost was averaged only over the cells that made money.** `break_even_cost_bps`
returns NaN when the gross mean return is not positive, and `cost_table` used `groupby().mean()`,
which drops NaN. So the reported break-even was conditional on the cell having been profitable
(9 to 12 of 20 cells, differing by model) and not comparable across rows, while being presented
as a property of the pooled 500-day track whose gross return is negative for seven of eight
conditions. `pooled_cost_table` now computes it on the pooled track and reports NaN where it does
not exist. The honest picture: **break-even exists for exactly one of the eight conditions, the
no-circuit control, at 0.06 bps.** The old per-cell average gave a flattering 2.8-6.5 bps.

**Max drawdown could be reported below -100%.** `(1+r).cumprod()` can cross zero, after which the
peak-to-trough ratio is meaningless; the first US table printed drawdowns of -1.20 to -1.25.
`max_drawdown` now returns -1 once the equity curve is wiped out, and `annualised_return` returns
-1 instead of NaN from a fractional power of a negative number (which had silently voided
`return_ann` and `calmar` for seven of eight US rows).

Consequence for the manuscript: the abstract's "best DSR 0.08" becomes 0.015; the cost paragraph
now states that break-even does not exist for seven of eight conditions rather than quoting a
3.4-4.0 bps range; the holdout table's DSR column is halved.

## F31. The trained-vs-frozen contrast has less power than the paper claimed
Early stopping (patience 6 on validation AUC) fires early: `epochs_run` averages 9.9 over the 30
budgeted, so the restored weights are typically from epoch 3 or 4, and mean best validation AUC is
0.514. `epochs_run` is identical in 20 of 20 cells between TRAINED and FROZEN, their validation
AUCs differ by at most 3.9e-5, and a large fraction of their daily returns are bit-identical.

The claim the data supports is therefore narrower than "training the circuit does not help": it is
"under this training budget the circuit's 18 parameters did not move far enough to change anything
downstream". `experiment.py` now records `circuit_shift_l2` and `circuit_norm_l2` per cell so the
displacement is measurable rather than assumed, and `--patience` is exposed on the CLI so the
no-early-stopping control can be run. Probe queued as gq #941.

## F32. The US "all conditions move together" argument was also an artefact of the bad day
With 2020-11-18 removed, the six conditions that have finished the clean re-run give pooled
Sharpe ratios of lstm -0.846, mlp -0.830, transformer -0.839, qtcnn_noquantum -1.240,
qtcnn_surrogate -1.244, tcn -1.508 over 625 days, and the largest daily move is 7.3%.

The minimum pairwise correlation of the pooled daily tracks is 0.53, not the 0.99 reported from
the contaminated panel. The claim in the manuscript's second-market subsection --- "every pairwise
return correlation is at least 0.99, so the loss is a property of the window and of the one signal
they all track" --- was therefore carried entirely by the shared bad day and must be withdrawn.
The classical baselines now separate from the hybrid family by about 0.4 Sharpe, which is a real
difference the contaminated panel hid. Awaiting gq #934 for the two circuit conditions before the
section is rewritten.

## F33. The clean US panel, and the ablation restated as an equivalence
Re-run after the price-floor fix (gq #934/#935/#936), 625 pooled OOS days, eight conditions:

| condition | Sharpe (ann.) | 95% CI | p | DSR | max DD |
| --- | ---: | --- | ---: | ---: | ---: |
| lstm | -0.846 | [-2.05, +0.42] | 0.20 | 0.000 | -0.426 |
| transformer | -0.839 | [-2.11, +0.56] | 0.23 | 0.000 | -0.431 |
| mlp | -0.830 | [-2.19, +0.53] | 0.23 | 0.000 | -0.342 |
| qtcnn_noquantum | -1.241 | [-2.56, -0.04] | 0.05 | 0.000 | -0.485 |
| qtcnn_surrogate | -1.244 | [-2.48, +0.01] | 0.05 | 0.000 | -0.500 |
| qtcnn_frozen | -1.431 | [-2.65, -0.22] | 0.02 | 0.000 | -0.523 |
| qtcnn_trained | -1.435 | [-2.65, -0.23] | 0.02 | 0.000 | -0.523 |
| tcn | -1.508 | [-2.82, -0.16] | 0.03 | 0.000 | -0.499 |

Every condition loses money, so no condition has a break-even cost. PBO 0.17. Romano-Wolf
smallest adjusted p = 0.289 (mlp). The transformer's nominally significant DM contrast from the
contaminated panel is gone. Drawdowns are now in a possible range (F30's bound), 0.34 to 0.52.
Pairwise return correlations run 0.53 to 1.000, not the 0.99+ the contaminated panel showed (F32).

**Equivalence (new, `qtcnn_research/equivalence.py`, four calibration tests).** Paired stationary
bootstrap on the difference of pooled tracks, TOST against a margin, plus the minimum detectable
effect at 80% power:

| contrast | market | diff | se | margin 0.03 | margin 0.25 | margin 0.50 | MDE80 |
| --- | --- | ---: | ---: | --- | --- | --- | ---: |
| trained vs frozen | JPX | -0.0008 | 0.0072 | **equivalent, p<0.001** | equivalent | equivalent | 0.020 |
| trained vs frozen | US | -0.0048 | 0.0075 | **equivalent, p=0.0005** | equivalent | equivalent | 0.021 |
| trained vs none | JPX | -0.085 | 0.239 | -- | not (p=0.25) | equivalent (p=0.03) | 0.671 |
| trained vs none | US | -0.195 | 0.147 | -- | not (p=0.36) | equivalent (p=0.02) | 0.411 |
| surrogate vs none | JPX | -0.053 | 0.235 | -- | not (p=0.20) | equivalent (p=0.02) | 0.657 |
| surrogate vs none | US | -0.003 | 0.158 | -- | not (p=0.06) | equivalent (p=0.003) | 0.442 |

This is the correction the paper needed most. The trained-vs-frozen claim is now positive evidence
of equivalence inside a margin the design is powered to resolve (MDE 0.02 against a 0.03 margin),
in both markets. The contrasts against the no-circuit control are honestly downgraded: they rule
out a contribution of half a Sharpe unit, not of nothing. The manuscript's ablation subsection,
second-market subsection, abstract and conclusion are rewritten accordingly, and the pooled-track
Sharpe correlation of 1.000 is demoted from evidence to corroboration.

## F34. The circuit does move during training; the output does not
The obvious objection to F31 is that early stopping at epoch 3-4 never let the circuit train, so
trained-equals-frozen is a statement about the budget rather than the layer. Measured directly
(gq #941, `results/probe_shift/`, instrumentation added in F31): over a fold the 18 circuit
parameters move by an L2 distance of 0.044 and 0.073 against initial norms of 0.416 and 0.622,
i.e. **10.6% and 11.8% of their initial magnitude**. The circuit is not sitting at
initialisation.

This strengthens the paper rather than weakening it. The parameters move by more than a tenth of
their norm and the downstream effect is nil: validation AUC differs from the frozen condition by
at most 3.9e-5 and the pooled Sharpe difference is inside a 0.03 margin. Whatever the circuit
learns, the encoder and head absorb it. That is the random-feature reading stated as a mechanism
rather than an absence, and it is what Bermejo et al. predict for a randomly initialised QCNN:
the model acts only on low-bodyness information about its input, which a classical surrogate
reproduces.

The probe covers fold 0, two seeds. A full no-early-stopping control (patience = epochs) remains
the cleaner answer and is not yet run.

## F35. Manuscript restructured around what the evidence actually supports
Five changes, none requiring new training, that between them answer the four reviews:

1. **Equivalence replaces the correlation as the headline** (F33). The pooled Sharpe correlation
   of 1.000 is demoted to corroboration because it is location-invariant and dominated by the
   between-fold spread; the claim now rests on a paired-bootstrap TOST inside a 0.03 margin with
   a minimum detectable effect of 0.02, in both markets.
2. **The circuit's displacement is reported** (F34). Training moves the 18 parameters by about
   11% of their initial norm and nothing downstream registers it, which converts "it did not
   help" into "the encoder and head absorb whatever it learns".
3. **The result is connected to a theoretical prediction.** Bermejo et al. show randomly
   initialised QCNNs act only on low-bodyness information and are reproduced by a classical
   surrogate; their demonstration is on physics benchmarks, ours is the same prediction measured
   on a financial ranking task. The null becomes a confirmation.
4. **The locked holdout is named as the positive control** in the abstract and discussion. It is
   the answer to "a null on a task where nothing works says nothing": on those 135 days the shared
   signal is profitable (gross Sharpe +0.72 to +2.34) and the ablation is unchanged.
5. **Contribution 4 is now delivered** (Sec. "Re-evaluating the reference implementation"). The
   reference implementation's nine models are re-scored on 241 out-of-sample days rather than 22.
   The quantum model stays ranked first at 1.80 but its interval spans zero (p = 0.07, DSR 0.54),
   and its information coefficient is the lowest of the nine (0.0016, t = 0.42) while the three
   models with individually significant ICs all have negative Sharpe ratios. Ranking skill and
   realised Sharpe are close to opposed, which a 22-day window cannot show.

Plus: the paper now discloses that it audits its own authors' prior work, in the introduction and
in a competing-interests declaration, and the reproducibility appendix attributes the
un-annualised Sharpe to the released code rather than to the published paper.

## F36. Four inference defects corrected; the conclusions are unchanged
1. **Bootstrap null imposed on the returns.** `bootstrap_sharpe_ci` built its null by re-centring
   the Sharpe draws, which assumes a symmetric sampling distribution. It now demeans the return
   series and resamples that, so the null is imposed where it belongs. Three calibration tests
   added, including one that reproduces the failure mode: with a single extreme day the old
   construction gave a p-value that contradicted the interval printed beside it. JPX p-values move
   only in the third decimal (the series is near-symmetric), which is why this went unnoticed.
2. **Sharpe differences are now tested as such.** Diebold-Mariano tests a difference in mean
   return; the tables are ordered by Sharpe ratio, and the two hypotheses differ whenever
   volatilities differ. `sharpe_difference_test` adds a paired stationary bootstrap on the Sharpe
   difference, resampling both tracks on the same days with the null imposed by demeaning. Both
   markets agree with the DM verdict: nothing is significant (JPX smallest p 0.14, US 0.22).
3. **Harvey-Leybourne-Newbold small-sample correction** applied to the DM statistic and the
   reference distribution changed from normal to t, which matters on the 135-day holdout.
4. **Romano-Wolf family stated explicitly**: seven contrasts against the no-circuit control on one
   panel. It does not cover the vs-zero tests and is not pooled across markets or the holdout, and
   the protocol section now says so rather than claiming family-wise control over everything.

Also: the lag rule is credited to Newey-West (1994), not the 1987 paper, which contains no lag
selection; the block length is related to the Politis-White automatic selector; and equivalence
testing is cited to Schuirmann.

One substantive change to a reported claim. On the clean US panel the bootstrap p-values against
zero for qtcnn_trained (0.029), qtcnn_frozen (0.030) and tcn (0.032) now fall below 5%: those
three conditions are significantly *negative*, which the manuscript now states rather than
describing the whole panel as indistinguishable from zero.

Title changed to "Trained, Frozen, or Absent: Identifying What a Quantum Layer Contributes to
Cross-Sectional Equity Ranking", which names the design and the question instead of posing a
rhetorical one.

## F37. CORRECTION to F8/F21: the shot budget was measured on the wrong input distribution
The shot-noise scan feeds the circuit uniform random angles on [-pi, pi], which produce a
cross-sectional spread of circuit outputs of 0.443. The scores the model actually ranks do not
look like that. Pushing real feature windows through the encoder of an initialised model (which
is exactly the FROZEN condition) and measuring the spread of the circuit output *within a trading
date*, over 12 dates and 300 securities each (`scripts_data/shot_budget_real_scores.py`,
gq #1049, `results/diagnostics/shot_budget_real.csv`):

| | random inputs | real cross-sections |
| --- | ---: | ---: |
| score spread (sd) | 0.443 | **0.018** |
| rank corr at 64 shots | -- | 0.29 |
| rank corr at 256 shots | **0.991** | **0.56** |
| rank corr at 1024 shots | 0.997 | 0.79 |
| rank corr at 8192 shots | 0.9996 | 0.96 |

The spread is 24x smaller than the scan assumed, and shot requirements scale with it. The
manuscript's claim that 256 shots preserve the ranking, and the deployment argument built on it
(5e5 to 2e6 circuit executions per trading day, "well inside what present devices deliver"), are
wrong by orders of magnitude. The higher-shot scan (gq #1051,
`results/diagnostics/shot_budget_real_high.csv`) locates the count: rank correlation 0.989 at
32,768 shots, 0.997 at 131,072 and 0.9993 at 524,288. Scoring 2000 securities for one trading day
therefore costs 7e7 to 3e8 circuit executions, two orders of magnitude above the 5e5 to 2e6 the
manuscript claimed.

The mechanism is a task-induced concentration, not a device limit: cross-sectional normalisation
and the encoder map a day's securities into a narrow band of embedding angles, so ranking within
that band is a fine discrimination that finite sampling resolves slowly. Exponential concentration
is a known obstacle for kernels and reservoirs as system size grows (P760, P1000); what this shows
is the same effect induced by the task, already severe at eight qubits where the asymptotic
argument does not apply. The manuscript now reports both input distributions side by side, and the
claim that hardware is not the barrier is withdrawn in favour of the narrower and defensible one:
the barrier is mis-stated when the shot budget is measured on the wrong inputs.

This is the same class of error as F21, and it was found the same way: by checking whether the
diagnostic's input distribution matches the deployed one. The honest version strengthens the
paper's framing rather than weakening it --- the conclusion becomes that a cross-sectional ranking
task concentrates the observable into a narrow band, so the shot cost of ranking 2000 securities
is set by that concentration and is far larger than a naive scan suggests. Sections on
diagnostics, deployment cost and the abstract all need rewriting once the number is in.

## F38. Circuit-only, full-budget, and gradient-boosted conditions are in the manuscript
2026-10-31. Three runs that were finished but unreported, now integrated:
- **QuantumOnly** (gq #1050, 20 cells): pooled Sharpe -0.251 [-1.50, +0.85], p 0.65; IC 0.0058
  (t 0.78), identical to the no-circuit control's ranking skill; 0.26 Sharpe below it (p 0.27).
  The one-dimensional observable ranks the cross-section as well as the full hybrid, i.e. barely.
- **No early stopping** (gq #1045, 20 cells per condition, 30 epochs): parameters move 26% of
  their norm (3-70%); trained vs frozen diff -0.006, se 0.009, TOST p 0.009 at 0.03, MDE 0.025;
  validation AUC 0.5133 for both. Both conditions fall from -0.07 to -0.17 Sharpe: shared
  overfitting, not a circuit contribution.
- **LightGBM** (gq #5387-5389; `qtcnn_research/run_lightgbm.py`, same folds/labels/subsample/
  portfolio rule, early stopping on validation AUC, ~14 trees kept): JPX -0.42 [-1.72, +0.78],
  IC 0.0051 (t 1.01), turnover 72%, 0.43 below the no-circuit control (p 0.20); US -0.67, best
  of nine, RW p 0.17; holdout +0.49, DSR 0.03. The family is now ten conditions: deflation 50,
  PBO 0.67 (JPX) / 0.31 (US). The manuscript had never stated the training loop; a "Training"
  subsection now does.

## F39. Second ansatz family: same ordering, twelve times the paired standard error
`CircuitConfig.ansatz = "reupload_hea"` (`--ansatz reupload_hea`): input re-uploaded before
each of 3 layers, RY+RZ per wire, CNOT ring, no pooling, mean of Z over all 8 wires, 48
parameters; 1.5x the QCNN's cost per batch. Trained and frozen on JPX, 40 cells (gq #5390-5394,
`results/walkforward_reup/`, no-circuit control copied in for the contrasts):

| | Sharpe | 95% CI | p | IC (t) |
| --- | ---: | --- | ---: | --- |
| no circuit | +0.013 | [-1.14, +1.18] | 0.99 | 0.0058 (0.88) |
| frozen | -0.190 | [-1.48, +0.97] | 0.75 | 0.0084 (1.04) |
| trained | -0.222 | [-1.48, +0.95] | 0.71 | 0.0077 (0.97) |

Trained - frozen = -0.032, paired se 0.091 (QCNN: 0.0072). TOST: not equivalent at 0.03
(p 0.52) or 0.10 (p 0.24); equivalent at 0.25 (p 0.005); MDE 0.25. Validation AUC agrees to
1.5e-4 and per-cell Sharpe correlates at 0.998, but per-cell IC differs by up to 0.011 (QCNN:
5.5e-5): the global observable lets the parameters reach the score. Parameters move 21% of their
norm. What transfers between families is "optimising does not improve on freezing"; "changes
nothing to 0.03 Sharpe" is a QCNN result and is bounded at 0.25 for this family. Written up
that way in method, results (Table reup), discussion threat and abstract.

## F40. Six-reviewer pass (2026-11-01): what was wrong, what was changed, what was run
Two reviewer reports (quantum ML, quantitative finance) and a consistency check on the
2026-10-31 manuscript. Every item below was verified against the repo before anything was
changed; the ones that held are listed with the fix, in the order the reviewers ranked them.

1. **Shot budget was measured on an initialised encoder and described as trained.** Verified:
   `scripts_data/shot_budget_real_scores.py` built `qtcnn_frozen` from `build_model` and
   never loaded weights. Fixed by (a) `run_experiment.py --save-state`, which writes each
   cell's trained `state_dict`, (b) a `--states` mode in the shot script that loads each cell
   and draws dates from that cell's own test window, (c) running it on the 20 trained cells
   (gq #5405-5409 for the weights, #5424 for the scan; `results/diagnostics/shot_budget_trained.csv`).
   Result: training *widens* the band. Within-date score sd per cell runs 0.006-0.49 (median
   0.10) against 0.018 for the initialised encoder and 0.443 for random angles. Rank
   correlation at 256/1024/8192/32768/131072 shots: median over cells 0.905/0.964/0.993/
   0.998/0.999; range 0.32-0.99 / 0.51-0.997 / 0.82-0.9996 / 0.94-0.9998 / 0.98-0.9999. The
   mean over cells reaches 0.99 at ~32768 shots, numerically the same as the initialised
   encoder's figure, but as an average of cells needing 256 and cells (three, all seed 4)
   needing more than 131072. The manuscript now reports the median cell (8192 shots for 0.99)
   and the range, and says the budget is a property of the trained cell. The old script
   clipped encoder outputs to [-pi, pi]; the deployed model does not, so the new mode does not
   either (`--clip` restores the old behaviour).
2. **"Pre-specified margin 0.03" contradicted the git history.** Verified: results 09-12,
   `equivalence.py` 09-17, F33 shows the margin chosen with the SE in view. The protocol now
   says so and lists every post-hoc change in a new subsection "Changes made after results
   were seen" (equivalence replacing the correlation headline; the F36 corrections; the DSR
   dispersion change below; the shot-budget re-measurement; the conditions added after first
   results; the holdout openings). The introduction's "tests specified in advance of the
   result" sentence is corrected.
3. **Holdout opened twice, Table 9 stale.** Verified: eight network conditions 2026-09
   (fold_metrics_holdout.csv 09-10), LightGBM 2026-10-31. Disclosed in protocol and results.
   Table regenerated from `results/holdout/*.csv` (transformer p 0.044 -> 0.03, LSTM RW
   0.34 -> 0.39, MLP RW 0.14 -> 0.15, and every DSR under the new dispersion).
4. **DSR used 1/sqrt(n) for the trial dispersion.** Verified in `evaluation.deflated_sharpe_ratio`.
   `headline_table` now passes the empirical sd of the (model, seed) pooled-track Sharpe ratios
   (annualised 0.30 on JPX walk-forward, 0.42 US, 0.85 holdout), which is what Bailey &
   Lopez de Prado's expected-maximum formula takes; the CSV records `dsr_trial_sd_ann`. Best
   walk-forward DSR 0.012 -> 0.17 (none), holdout transformer 0.29 -> 0.61. Verdicts unchanged.
5. **Trained = frozen is close to structural in the hybrid conditions; the circuit-only pair
   was missing its frozen arm.** Added `qtcnn_quantumonly_frozen` (circuit at its 0.1 N(0,I)
   initialisation, encoder and head trained, head sees the circuit output only), 20 cells
   (gq #5400-5404). Against the September circuit-only run this first looked like a
   difference of -0.068 with se 0.228; against the reproducible circuit-only run (F41) it is
   +0.059 with se 0.054, equivalent at 0.25 (p<0.001), not at 0.10 (p 0.22), MDE80 0.15;
   per-cell Sharpe corr 0.98, IC differs by up to 0.005, circuit moves 20% of its norm.
   Study is now 11 conditions x 5 seeds = 55 configurations.
6. **Physics wording.** Depolarising channel is applied after every layer, not terminally,
   so "cannot reorder" was an overstatement; CSV minimum rank corr is 0.9999, not 1.0000.
   "Global observable" for the re-uploading ansatz is a sum of single-qubit Z (local terms);
   reworded everywhere and the barren-plateau caveat stated. Initialisation 0.1 N(0,I) now in
   the method section.
7. **Text vs CSV.** US LightGBM-vs-none DM p = 0.041 (RW 0.17) now stated instead of "no
   significant contrast". Holdout range "+0.72 to +2.34, eight" -> "+0.49 to +2.34, nine, MLP
   -0.31". "4 bps removes the signal" -> break-even 0.06 bps on the one positive walk-forward
   condition, 8 bps best on holdout. 2022-2025 extension: stated that no result uses it.
   JPX Universe0 hindsight disclosed in Data and Discussion.
8. **Format.** 54 em-dashes removed; abstract 356 -> 336 words; SpringerOpen Declarations
   block; appendix before references; corresponding e-mail added (ORCIDs still to supply).

Not changed: the article class (SpringerOpen template to be applied at submission), Zenodo
DOI (needs the owner's account).

## F41. The September network results do not reproduce; everything was re-run (2026-11-01/02)
Found while saving trained weights for the shot budget (F40.1): re-training qtcnn_trained
fold 0 seed 0 with the committed code gave Sharpe 0.994 / IC 0.0152 / 7 epochs, against the
September cell's 0.611 / 0.0084 / 8 epochs.

- **The committed code is deterministic.** That cell trained five times (gq #5405, #5419-5422)
  at 4 and at 1 threads: bit-identical every time. The second ansatz family (run 2026-10-31
  with committed code) reproduces bit-for-bit (#5454). LightGBM is untouched.
- **No commit changed training.** The first commit a0e83d1 (2026-09-12), checked out in a
  worktree and run on the same cell (#5639), gives 0.994 too. `git diff a0e83d1 HEAD` on the
  training path adds only instrumentation and the second ansatz. The September runs
  (6-25 Sep) came from working copies that were never committed; the 25 Sep full-budget run
  evidently scored final-epoch weights (circuit shift 26% vs 18%), which no commit ever did.
- **Re-run with committed code** (gq #5425-5431 JPX, #5434-5440 US, #5441-5443 holdout,
  #5629-5633 circuit-only, #5634-5638 full budget). September files archived under
  `results/*_2026-09_original/`; the canonical dirs now hold the rerun plus the LightGBM and
  circuit-only-frozen runs, which were already made with committed code.
  `scripts_data/rerun_comparison.py` writes `manuscript/tables/rerun.tex` (appendix).
- **Agreement.** Per-cell Sharpe between runs of a condition: 0.44-0.82 JPX, 0.40-0.74 US.
  Pooled Sharpe moves by up to 0.25 (JPX transformer), 0.73 (US MLP), 1.66 (holdout
  transformer). That is the training-path noise the seed figure already shows.
- **What held in both runs:** trained = frozen inside 0.03 on JPX (Nov: +0.008, se 0.0075,
  p 0.006) and US (Nov: -0.002, se 0.0073, p<0.001); per-cell trained/frozen corr > 0.999;
  nothing on JPX distinguishable from zero; no contrast vs the control survives Romano-Wolf
  on any panel (Nov smallest: JPX 0.38 surrogate, US 0.22 LightGBM, holdout 0.33 LightGBM);
  holdout trained vs frozen 0.02 apart; US everything loses money.
- **What changed:** orderings and levels. JPX best is now the transformer (+0.107, DSR 0.17),
  the only positive gross return (break-even 0.49 bps; was the no-circuit control). Holdout
  best is the no-circuit control (+1.83, p 0.07, DSR 0.42; transformer fell from first to
  sixth), all nine positive. US: five conditions significantly negative (was three). PBO JPX
  0.58, US 0.15. JPX trained vs none: -0.082, se 0.120, equivalent at 0.50, not 0.25,
  MDE 0.34 (was 0.67). Circuit-only pair tightened (F40.5).
- **The full-budget control was broken in the released code.** `--patience 30` trained 30
  epochs and then restored the best-validation weights, so it was bit-identical to the
  early-stopped run (`results/walkforward_nostop_restorebest/`). Added
  `TrainConfig.restore_best` / `run_experiment --final-weights`; the control now scores the
  last epoch. Re-run as #5634-5638.
- **The holdout was opened a third time** for the rerun; disclosed in the protocol, with the
  first opening in the appendix table.
- Also: `scripts_data/spanning.py` now produces the spanning numbers (previously run ad hoc;
  it reproduces the September figures exactly on the archived returns). New: R^2 0.910 on
  the four neural baselines, alpha t -0.04; twin R^2 0.975, t -0.73.
- **Full-budget control, corrected** (`--patience 30 --final-weights`, #5634-5638): circuit
  moves 68% of its initial norm (25-140%) instead of 18%; trained -0.108, frozen -0.108;
  difference +0.000, se 0.043 (six times the early-stopped se), TOST equivalent at 0.10
  (p 0.02), not at 0.03 (p 0.24), MDE 0.12; per-cell corr 0.987; IC halves (0.0083 ->
  0.0038) for both. Manuscript: the 0.03 certificate is now stated for the early-stopped
  runs; the full-budget certificate is 0.10. The September full-budget run had claimed 0.03.

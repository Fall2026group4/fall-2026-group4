# H1-TEST: single RepE-LM held-out TEST evaluation (2026-11-13). Status: COMPLETED

**Evidence level:**
* HT1–HT3 cells: **CONFIRMED TEST RESULT** where marked confirmed below.
* HT1 and HT3 as whole hypotheses: **not confirmed**.
* HT4 and the 90 secondary comparisons: **descriptive**.

**Protocol (frozen before any TEST row was read):**
[`../repe_lm_test_protocol/preregistration.json`](../repe_lm_test_protocol/preregistration.json) and
[`preregistration_addendum.json`](../repe_lm_test_protocol/preregistration_addendum.json).

**Run:**
* Script: `scripts/repe/29_repe_lm_test_evaluation.py`, sha256 `c99957ee…c6c8`, identical to the addendum hash.
  Verified before this write-up.
* Run once, in test mode: 81 SAE sources, 2,408 TEST rows, 4,953 s on one A10G.
* Fitting: every probe, direction and site was fitted on RepE-LM TRAIN only.
* Authorised by the user on 2026-11-13.

## Files

| File | Content |
|---|---|
| `per_sae.csv` | Every metric for each SAE source × layer (TEST) |
| `comparisons.csv` | C1–C3, A1, A2 × metrics × layers under the R2 rule (Welch, 3 vs 3 seeds, 9 cross pairs) |
| `confirmatory_cells.csv` | The 13 HT1–HT3 cells: validation vs TEST, Holm over 13 cells, and the confirmation flag |
| `ht4_probe_transfer.json` | Persona-probe AUROC (original activations), grouped-bootstrap 95 % CI |
| `checks.json` | Run metadata and hypothesis verdicts. The per-layer `checks` entries are empty by design: in test mode they only hold smoke-reproduction checks |

**Comparisons:**
* C1: R2 TopK-12288 vs old TopK-3072.
* C2: R2 ReLU-12288 vs old ReLU-3072.
* C3: R2 BatchTopK-12288 vs old JumpReLU-3072.

## Confirmation rule (addendum)

A cell is CONFIRMED only if both of these hold on TEST:
* the R2 rule classes it IMPROVEMENT, meaning every one of the 9 seed pairs is better and the Welch 95 % CI excludes 0;
* its Welch p, Holm-adjusted over 13 cells, is below 0.05.

A hypothesis is CONFIRMED only if every one of its cells is confirmed.

## Positive and negative controls

| Control (pre-registered) | L1 | L8 | L11 | Verdict |
|---|---|---|---|---|
| Original GPT-2 T_full > 0, 95 % CI low > 0 | 0.0378 (low 0.0292) | 0.0201 (low 0.0160) | 0.0379 (low 0.0294) | PASS |
| Persona KL(H‖U) > synonym KL(H‖Hp) | 0.0393 > 0.0172 | same | same | PASS (output-level, so the same value at every layer) |
| T_rec above the isotropic and permuted 95th percentile | all 23 SAEs | all 29 | all 29 | PASS |
| Neutral N1/N2 persona-probe AUROC ≤ 0.55 | 0.498 | 0.473 | 0.477 | PASS |
| Neutral N3/N4 (test-only frames) AUROC ≤ 0.55 | 0.501 | **0.561** [0.553, 0.575] | **0.551** [0.545, 0.563] | **FAIL at L8 and L11** (small margin) |

## Results: confirmatory cells (TEST; ref → new, mean ± sd over 3 seeds)

| Hyp. | Metric (direction) | Cell | Validation diff | TEST ref → new | TEST diff [95 % CI] | Pairs better | Holm p | Confirmed |
|---|---|---|---|---|---|---|---|---|
| HT1 | M4 T_rec/T_full (↑) | C1 L8 | +0.101 | 0.602 ± 0.007 → 0.729 ± 0.022 | +0.127 [+0.079, +0.175] | 9/9 | 0.021 | **yes** |
| HT1 | | C3 L8 | +0.090 | 0.679 ± 0.012 → 0.752 ± 0.027 | +0.072 [+0.015, +0.130] | 9/9 | 0.058 | **no** (IMPROVEMENT by the R2 rule; fails Holm) |
| HT2 | pair-direction cosine (↑) | C1 L1 | +0.135 | 0.340 → 0.476 | +0.136 [+0.127, +0.145] | 9/9 | 2.5e-5 | yes |
| HT2 | | C1 L8 | +0.094 | 0.371 → 0.461 | +0.090 [+0.087, +0.093] | 9/9 | 2.5e-5 | yes |
| HT2 | | C1 L11 | +0.080 | 0.420 → 0.490 | +0.070 [+0.066, +0.073] | 9/9 | 5.6e-4 | yes |
| HT2 | | C2 L1 | +0.084 | 0.520 → 0.597 | +0.077 [+0.065, +0.088] | 9/9 | 6.6e-4 | yes |
| HT2 | | C2 L8 | +0.050 | 0.549 → 0.600 | +0.051 [+0.038, +0.065] | 9/9 | 0.015 | yes |
| HT2 | | C2 L11 | +0.066 | 0.559 → 0.617 | +0.058 [+0.048, +0.068] | 9/9 | 1.1e-3 | yes |
| HT2 | | C3 L1 | +0.143 | 0.384 → 0.525 | +0.141 [+0.136, +0.146] | 9/9 | 1.1e-4 | yes |
| HT2 | | C3 L8 | +0.068 | 0.428 → 0.491 | +0.064 [+0.060, +0.068] | 9/9 | 3.2e-4 | yes |
| HT2 | | C3 L11 | +0.039 | 0.464 → 0.500 | +0.035 [+0.026, +0.044] | 9/9 | 3.0e-3 | yes |
| HT3 | RepE persona refit gap (↓) | C1 L8 | −0.048 | 0.137 ± 0.015 → 0.090 ± 0.010 | −0.048 [−0.079, −0.016] | 9/9 | 0.046 | **yes** (marginal) |
| HT3 | | C3 L8 | −0.037 | 0.120 ± 0.009 → 0.098 ± 0.016 | −0.022 [−0.055, +0.011] | 8/9 | 0.128 | **no** (NO CLEAR CHANGE) |

Per-seed values (seed order as stored in `comparisons.csv`):
* HT1 C1 L8: new 0.741 / 0.742 / 0.703; ref 0.608 / 0.604 / 0.593.
* HT1 C3 L8: new 0.759 / 0.775 / 0.722; ref 0.680 / 0.667 / 0.691.
* HT3 C1 L8: new 0.100 / 0.080 / 0.089; ref 0.136 / 0.122 / 0.153.
* HT3 C3 L8: new 0.084 / 0.095 / 0.115; ref 0.127 / 0.123 / 0.110.

### Hypothesis verdicts

| Hypothesis | Category | Verdict |
|---|---|---|
| **HT1** (causal preservation at L8, C1 and C3) | Causal intervention (SAE-selected directions, error term kept; D33) | **NOT CONFIRMED.** C1 is confirmed; C3 improves in all 9 pairs but misses Holm (0.058) |
| **HT2** (persona pair-direction cosine, 9 cells) | Representation preservation | **CONFIRMED** in 9/9 cells; TEST effect sizes are within 0.012 of validation |
| **HT3** (persona refit gap at L8, C1 and C3) | Representation preservation | **NOT CONFIRMED.** C1 is confirmed (Holm p = 0.046); C3 shows no clear change on TEST |
| **HT4** (descriptive) | Probe transfer, original activations | See below |

### HT4: persona-probe transfer (original activations; probe fitted on TRAIN T1–T4)

| Layer | T1–T4 (fit templates, TEST statements) | T5/T6 (held-out) | T7/T8 (test-only wording) | N3/N4 floor |
|---|---|---|---|---|
| L1 | 0.647 | 0.538 | 0.564 | 0.501 |
| L8 | 0.801 | 0.642 | **0.833** [0.791, 0.881] | 0.561 |
| L11 | 0.848 | 0.633 | **0.825** [0.783, 0.872] | 0.551 |

At L8 and L11 the persona direction transfers to new wording and new slot words, well above the neutral floor. At L1
it barely does. The new neutral frames leak a little: they exceed the 0.55 floor by ≤ 0.011, which is small next to
the T7/T8 margin (≥ 0.26).

### Secondary comparisons (descriptive; no correction)

Of 90 comparison × metric × layer cells, 55 are IMPROVEMENT, 35 NO CLEAR CHANGE and **0 REGRESSION**.
* By comparison (improvement / no clear change): C1 16/2, C2 6/12, C3 12/6, A1 12/6, A2 9/9.

## Improvements, regressions and limitations

* **Replicates:** R2's gain in persona pair-direction cosine is the most robust honesty-related representation result
  in the project. All 9 cells replicate with almost the same effect size.
* **Partly replicates:** the L8 causal-preservation gain (D35 "borderline"). It is confirmed for TopK (C1, +0.127). It
  is not confirmed for BatchTopK vs JumpReLU (C3), which improves in 9/9 pairs but is not significant after Holm.
* **Partly replicates:** the persona refit gap at L8. C1 just passes; C3 does not.
* **Regressions:** none in any confirmatory or secondary cell.
* **Limitations:**
  * Only 3 seeds per arm, so Welch p rests on 4 df or fewer.
  * The N3/N4 neutral floor is exceeded at L8 and L11.
  * M4 is an error-kept patch, not SAE-mediated (D33).
  * The source statements were once seen in the retired, length-confounded RepE task (disclosed in the protocol).
  * TEST rows have now been read once and must not be reused for selection.
  * These results do not bear on reconstruction quality or GoT truth.

## Status

COMPLETED. TEST rows were read once. Nothing was re-run or tuned after reading them.

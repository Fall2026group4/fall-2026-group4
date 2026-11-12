# Verified results summary: GPT-2 Small SAE faithfulness

Compiled on 2026-11-12. Every number here was read from the CSV named under its table. The M5 values were re-checked
against `results/shared/m5_confirmatory_test/m5_rows.csv` on 2026-11-12. The exhaustive per-experiment tables are in
[`EXPERIMENT_RESULTS_SUMMARY.md`](EXPERIMENT_RESULTS_SUMMARY.md), and the chronology is in
[`experiment_history.md`](experiment_history.md).

**Labels:**
* **TEST:** the single M5 pass, on rows previously scored for the old reference SAEs only (D31).
* **VAL:** the validation split.

**Conventions:**
* Means are over 3 seeds (42/1042/2042) unless n is stated.
* "Old" = width-3,072 SAEs trained on 101,600 OWT tokens (15k steps).
* "R2" = width-12,288 SAEs trained on 100M fresh OWT tokens with a modern recipe.
* Every SAE sits at `blocks.L.hook_resid_pre`; BOS is excluded.
* OWT L0 targets 128 and must be within ±10 % for comparisons.

**Headline.** The improvements are real but level-specific:
* **reconstruction and behaviour improved and were confirmed on TEST;**
* truth information (GoT) did **not** measurably change;
* persona/honesty representation and causal preservation improved **on validation only** (RepE-LM has no test split);
* the one behaviour-targeted fine-tune (R4) cost causal preservation at L11.

---

## A. OpenWebText: reconstruction and behaviour

### A1. Old SAEs across depth (TEST, 108 runs; width 3,072; L0 128)

| Layers | EV ↑ (ReLU / TopK / JumpReLU) | Spliced ΔCE ↓ (VAL, nats) | Matched leader |
|---|---|---|---|
| 0–2 | 0.93–0.95 | 0.01–0.04 | ReLU at L1; ReLU ≈ JumpReLU at L2 |
| 3–9 | falling | rising | JumpReLU |
| 11 | 0.757 / 0.815 / 0.815 | 0.20–0.36 | JumpReLU ≈ TopK |

* 107/108 runs are matched; layer-0 JumpReLU seed 42 is excluded.
* Raw MSE is never averaged across layers: the residual norm grows with depth.
* **Sources:** `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/all_layers_*.csv`,
  `results/shared/spliced_evaluation/`.

### A2. Old → R2 (TEST, M5; 3 seeds per family)

| Layer | Arch (old → new) | OWT L0 (old / new) | NMSE ↓ | ΔCE ↓ (nats) | SAE/noise ΔCE ratio ↓ | Verdict |
|---|---|---|---|---|---|---|
| 8 | TopK → TopK | 128 / 128 | 0.160 → **0.094** (−41 %) | 0.133 → **0.062** (−53 %) | 1.50 → 1.20 | CONFIRMED TEST IMPROVEMENT |
| 11 | TopK → TopK | 128 / 128 | 0.185 → **0.111** (−40 %) | 0.204 → **0.096** (−53 %) | 1.86 → 1.48 | CONFIRMED TEST IMPROVEMENT |
| 8 | ReLU → ReLU | 131.9 / 127.6 | 0.195 → 0.132 (−32 %) | 0.218 → 0.131 | 2.00 → 1.81 | CONFIRMED TEST IMPROVEMENT |
| 8 | JumpReLU → BatchTopK | 130.3 / 129.3 | 0.156 → 0.090 (−43 %) | 0.121 → 0.054 | 1.44 → 1.13 | CONFIRMED TEST IMPROVEMENT (different architecture) |
| 1 | TopK → TopK | 128 / 128 | 0.079 → 0.025 | 0.022 → 0.006 | — | CONFIRMED TEST IMPROVEMENT |

* **What changed in R2: training data, width and recipe, together.**
  * Recipe: AuxK dead-latent loss, LR schedule, batch 4,096, input normalisation, encoder = decoderᵀ init.
  * At equal width (A1: TopK-3072 old → new), NMSE fell 23–35 %, about 55 % of the total gain at L8/L11. The rest came
    from width (A2).
  * **Token count alone was never isolated.** The 10M/25M/50M checkpoints exist but are unevaluated.
* Dead latents fell to ≈ 0–0.3 % with AuxK (the old TopK at L1 had 44 % dead).
* **Sources:** `results/shared/m5_confirmatory_test/{m5_rows,comparisons}.csv`, `results/openwebtext/r2_scaled_saes/README.md`.

### A3. R4: KL fine-tune at fixed L0 vs the MSE-only control (TEST; paired by seed; L0 exactly 128)

| Layer | ΔCE ↓, MSE control → KL | SAE/noise ↓ | NMSE ↓ | Verdict |
|---|---|---|---|---|
| 8 | 0.062 → **0.051** (−17 %) | 1.21 → **0.98** | 0.094 → 0.096 (+2 %) | CONFIRMED TEST IMPROVEMENT (ΔCE, KL); **REGRESSION** (NMSE) |
| 11 | 0.096 → 0.081 (−16 %) | 1.49 → 1.22 | 0.111 → 0.113 (+2 %) | CONFIRMED TEST IMPROVEMENT (ΔCE, KL); **REGRESSION** (NMSE) |

* At L8, R4 is the only SAE whose error is no more damaging than same-norm noise.
* It trades reconstruction for behaviour (M5 classes the NMSE change as REGRESSION; GoT NO CLEAR CHANGE), and
  causal preservation at L11 (§C3).
* **Source:** `results/openwebtext/r4_kl_finetune/`, M5 rows.

### A4. External reference: public res-jb (TEST rows, descriptive; L0 15 / 66 / 60 at L1 / L8 / L11, so unmatched)

* At L8: NMSE 0.144, ΔCE 0.168.
* It sits on the old SAEs' error–damage relation; R2 dominates it.
* **Source:** M5 rows, `results/shared/e2_public_sae_reference/`.

**Remaining trade-offs:**
* The SAE error is still more harmful than isotropic noise for every R2 SAE (1.13–2.24).
* ReLU remains the weakest reconstructer at depth.
* No modern JumpReLU exists in the comparison.

---

## B. Geometry of Truth: truth preservation

**Task.** GoT cities: 1,496 statements, balanced, grouped by city; test n = 452. The final-token residual is used.
Original-activation logistic AUROC (TEST) is 0.839 at L8 and 0.792 at L11. GoT is uninformative before about L4.

### B1. Frozen vs re-fit readers

| Reader | What it measures | Typical gap ↓ | Status |
|---|---|---|---|
| Frozen original logistic probe on x̂ | whether the model's original readout survives | 0.19–0.32 (L4–11, old SAEs, TEST) | mostly generic fragility: same-norm noise gives 0.25 vs the SAE's 0.28 (M2, VAL) |
| Re-fit logistic probe on x̂ | whether the information is still linearly present | 0.07–0.12 (M2, VAL); 0.028–0.107 (M5, TEST) | a real but small loss |
| PCA at matched OWT NMSE | reference | — | **no trained SAE beats PCA** (M2, M5) |

### B2. M5 GoT cities (TEST; re-fit logistic gap ↓; SAE − PCA ↓, < 0 means it beats PCA)

| Layer | SAE | GoT-prompt L0 | Re-fit gap | SAE − PCA |
|---|---|---|---|---|
| 8 | old TopK / JumpReLU / ReLU | 128 / 63 / 77 | 0.061 / 0.058 / 0.034 | 0.036 / 0.034 / 0.008 |
| 8 | R2 TopK / BatchTopK / ReLU | 128 / 80 / 89 | 0.056 / 0.065 / 0.028 | 0.042 / 0.053 / 0.010 |
| 8 | R2 random frozen decoder (n = 1) | 128 | 0.067 | **−0.003** |
| 8 | R4 KL / MSE | 128 / 128 | 0.048 / 0.059 | 0.035 / 0.045 |
| 11 | R2 TopK / BatchTopK / ReLU | 128 / 106 / 111 | 0.078 / 0.087 / 0.053 | 0.072 / 0.081 / 0.030 |
| 11 | R2 random frozen decoder | 128 | 0.086 | **−0.013** |

* **Verdict:** NO CLEAR CHANGE in 34/34 cells.
* Seed sd is 0.004–0.024, as large as the differences.
* Width (A2) REGRESSED GoT at L8 in 2 cells; this could be a false positive (≈ 0.05 chance per cell; no multiplicity
  correction).
* **Source:** `results/shared/m5_confirmatory_test/m5_rows.csv`.

### B3. Fine-tuning for truth (all VALIDATION)

| Method | Result | Verdict |
|---|---|---|
| Historical recon-only / truth-aware (L6) | MLP test gap base 0.327/0.154/0.323 → recon-only 0.161/0.110/0.146; L0 drifted | INVALID as matched evidence |
| Truth-aware, original | the teacher probe is the evaluator | INVALID (circular) |
| Truth-aware, corrected A/B | 0.189 vs recon-only 0.107 | FAILED SUCCESS CRITERION |
| Method A (L6) | 18/18 paired better, none > 2 sd | FAILED SUCCESS CRITERION (directional) |
| IMP-L Method A (L3/8/9) | frozen-MLP gain 4/9; re-fit reader unchanged | frozen readout restored, not information |
| Output-KL (L6, L3/8/9) | no truth gain; regressions at L3 ReLU/TopK | FAILED / REGRESSION |

**Interpretation:**
* Truth information is not destroyed, but it is reduced by about 0.03–0.10 AUROC for a re-fit reader.
* The probe-relevant signal is in the SAE error (GoT error-term AUROC 0.66–0.77 at L8).
* No method improved it for an independent re-fit reader.
* The current GoT metric cannot separate trained from random SAEs at L8. Its statistical power is the binding
  constraint (see [`limitations_and_future_work.md`](limitations_and_future_work.md)).

---

## C. RepE honesty / persona

### C1. Historical RepE (do not cite for findings)

* The old task ("Pretend you're an honest/untruthful person…") is length-confounded: length alone gives AUROC 0.80, and
  a layer-0 probe gives 0.83.
* Its representation, R1-honesty, R2-clamp and 6c causal results are **HISTORICAL** (D23).
* 6c, kept for provenance (TEST): the raw honesty direction gives E = 0.0048 (p = 0.010 vs 200 random directions);
  SAE directions ≈ controls.

### C2. RepE-LM (equal-length one-token persona swaps; train/validation only), VAL

| Layer | SAE | RepE-LM L0 | Persona re-fit gap ↓ | cos(Δ, Δ̂) ↑ | ‖Δ̂‖/‖Δ‖ (1 ideal) | T_rec/T_full ↑ |
|---|---|---|---|---|---|---|
| 8 | old TopK / JumpReLU / ReLU | 128 / 167 / 161 | 0.156 / 0.162 / 0.212 | 0.39 / 0.44 / 0.56 | 1.38 / 1.34 / 0.67 | 0.625 / 0.665 / 0.621 |
| 8 | R2 TopK / BatchTopK / ReLU | 128 / 141 / 145 | **0.108** / 0.126 / 0.193 | 0.49 / 0.50 / 0.61 | 1.30 / 1.25 / 0.63 | **0.726** / 0.754 / 0.643 |
| 11 | R2 TopK / BatchTopK / ReLU | 128 / 134 / 140 | 0.109 / 0.101 / 0.194 | 0.51 / 0.52 / 0.62 | 1.31 / 1.30 / 0.62 | 0.692 / 0.701 / 0.712 |

RepE-LM validation L0 (ladder column `rlm__repe_val_l0`; mean over seeds) is not matched to TopK's 128: threshold
architectures fire more latents on RepE-LM prompts (up to 208–216 at L1) and fewer on GoT prompts (§B2).

* **R2 TopK vs old TopK at L8 (VAL):**
  * persona re-fit gap 0.156 → 0.108 (−31 %, CI [−0.082, −0.014]);
  * T_rec/T_full 0.625 → 0.726 (+0.101, CI [+0.019, +0.182]);
  * both VALIDATION IMPROVEMENT.
* **Relative to PCA, no change** (persona SAE − PCA 0.033 → 0.027, CI includes 0). Better reconstruction moves the
  PCA reference just as much.
* **Contrast distortion.** First reported in RLM-RP; extended to R2 by the 2026-11-12 audit.
  * ReLU-family SAEs **shrink** the persona contrast (norm 0.62–0.71) while keeping its direction best (cos 0.56–0.62).
  * TopK-family SAEs (TopK, BatchTopK, old JumpReLU) **inflate** it (1.09–1.46) with a worse direction (cos 0.37–0.52).
  * The cause in the TopK family (latents switching on/off between the paired prompts) is a HYPOTHESIS.
* The lost persona information sits in the error: error-term AUROC 0.72–0.87.
* **Sources:** `results/openwebtext/r2_scaled_saes/ladder/*.csv`, `results/repe/repe_lm_representation_preservation/`.

### C3. Causal (M4 protocol; RepE-LM; VAL)

| Comparison | Layer | Before → after | Verdict |
|---|---|---|---|
| Old SAEs, L1 slot | 1 | T_rec/T_full 0.93–0.97 | MEASUREMENT |
| Old SAEs, final token | 8 / 11 | 0.62–0.68 | MEASUREMENT |
| Method A / output-KL (E3) | 8 | ≈ base | NO CLEAR CHANGE / not creditable |
| R2 TopK vs old TopK | 8 / 11 | 0.625 → 0.726 / 0.620 → 0.692 | VALIDATION IMPROVEMENT |
| R4 KL vs MSE control | 11 | 0.694 → 0.671 (−0.024, all seeds) | REGRESSION |

* T_full (the whole persona effect) is small: 0.022 nats at L8 and 0.043 at L11.

### C4. R5 skyline (VAL; T_k/T_full at L8 final; old SAEs)

| k | DAS (learned) | PCA | SAE top-k features | Random |
|---|---|---|---|---|
| 1 | 0.64 | 0.36 | 0.04–0.05 | 0.002 |
| 4 | 0.91 | 0.52 | 0.10–0.16 | 0.008 |
| 64 | 1.29 (overshoot) | 0.83 | 0.34–0.48 | 0.11 |

* The whole old-SAE reconstruction carries 0.62–0.67.
* **A rank-4 supervised subspace carries more than the whole SAE.**
* **Source:** `results/repe/r5_causal_subspace_skyline/summary.csv`.

---

## D. What actually improved: one-line answers

| Question | Answer | Evidence |
|---|---|---|
| Did OWT reconstruction improve? | Yes: −40 to −74 % NMSE (R2) | TEST |
| Did behavioural faithfulness improve? | Yes: ΔCE −35 to −79 % (R2); a further −16 to −17 % (R4) | TEST |
| Did GoT truth preservation improve? | No measurable change (0/34); the metric is underpowered | TEST |
| Did persona representation improve? | In absolute terms yes (−27 to −31 %); relative to PCA no | VAL only |
| Did causal preservation improve? | Yes with R2 (+0.07 to +0.10); R4 reduced it at L11 | VAL only |
| Does lower reconstruction error give better truth preservation? | No robust within-layer link (M0, M5, random-decoder tie) | VAL + TEST |

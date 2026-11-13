# R1. Honesty representation preservation, GPT-2 Small layer 6 (2026-11-07). Status: COMPLETE (validation). Verdict: ReLU INCONCLUSIVE, TopK NOT SUPPORTED, JumpReLU NOT SUPPORTED

Evidence level (planned): validation finding, plus one pre-registered final test pass.
Companion experiment: `results/repe/r2_sae_in_the_loop_steering_layer6/` (R2). Roadmap: R1 (new), S3, X3.

## 0. Audit of existing RepE work (2026-11-07, read-only)

| Item | State | Evidence |
|---|---|---|
| Data | 597 statements → 1194 persona prompts ("Pretend you're an honest/untruthful person …" + statement prefix); split by statement 417/90/90; final-token `hook_resid_pre` cache, all 12 layers | `data/repe/persona_pairs.csv`, `data/repe/activations/gpt2_small/repe_final_token.pt` |
| Cross-wording | 180 prompts, **validation statements only**, wording "Always give accurate…/Deliberately give inaccurate…" | `data/repe/cross_wording_validation.csv` |
| Original probes (layer 6, val) | mass-mean 0.893, logistic 0.997 AUROC (fitted on **all** train rows; layer chosen on validation) | `all_layer_frozen_probe_evaluation/` |
| Base SAE preservation (layer 6, 3 seeds each) | logistic gap ReLU ≈ 0.25, TopK ≈ 0.08–0.10; RepE-prompt L0 ReLU ≈ 164, TopK 128, JumpReLU ≈ 180 (not matched) | `all_layer_frozen_probe_evaluation/per_run.csv` (VAL, TEST, cross-wording) |
| RepE fine-tuning | **None.** Method A / E2E fine-tuned on GoT; RepE gaps "unchanged" there | `method_a_layer6/`, `e2e_inspired_layer6/` |
| Splicing | persona-shift cosine, final KL for all 108 SAEs (VAL) | `spliced_evaluation/` |
| Steering | scripts 10–12: **no SAE in forward pass** (direction from reconstructions), SUPERSEDED. Script 13 (6c): SAE features used only as **decoder directions added to the raw activation**, not SAE-in-the-loop. Raw direction E = 0.0048 (p = 0.01); positive control 0.234; all SAE-derived directions null | `steering/README.md`, `steering/causal_random_controls/` |
| Test split exposure | RepE test already scored by 6c (pre-registered), the all-layer frozen-probe TEST and `representation_preservation_val_test.csv`. No selection was ever made on it | progress §6c, §9 |

**Leakage checks run for this plan:**
* Statement → split mapping is unique; exact and normalised overlap between train and test, and between validation and test, is 0.
* **One near-duplicate crosses train/validation:** validation `statement_id` 326 ("…776 B.C.") ≈ train 589 ("…776 BC.").
  * Pre-registered handling: **exclude statement 326 (both personas) from every validation score** in R1 and R2.
* Train duplicates 436/509 ("The sun/Sun is a star.") are both train. They must fall in the same A/B half, which is ensured by grouping on the normalised statement.
* Prompt-specific L0 is unmatched across architectures. Comparisons in R1 are therefore **within architecture, paired by seed**, and RepE L0 is a guard (§4).

## 1. Question, hypothesis, motivation

* **Question:** can SAE fine-tuning improve the preservation of the honesty (persona) representation at layer 6 without losing sparsity or general reconstruction?
* **H-R1a:** RepE reconstruction fine-tuning with OWT replay (the RepE analogue of Method A) reduces the frozen-probe honesty gap relative to the base SAE.
* **H-R1b (primary):** adding a paired-contrast term (preserve x_H − x_U for the same statement) reduces the gap **beyond** H-R1a, at equal OWT and RepE L0, and the gain transfers to held-out wording.
* **Why:** reconstruction quality tracks honesty preservation (ρ = 0.86, §9), and ReLU loses about 0.25 AUROC at layer 6. Whether a targeted, non-probe objective can close the gap without gaming a probe (the D12/D16 lesson) is untested.

## 2. Design

* **Model:** GPT-2 Small, frozen. Hook `blocks.6.hook_resid_pre`, final prompt token (BOS excluded).
* **SAEs:** ReLU, TopK and JumpReLU, d_sae 3072, OWT L0 128. Bases are the `owt_uninterrupted` checkpoints, seeds 42/1042/2042 (AUTHORITATIVE in `docs/checkpoint_registry.csv`).
* **Splits:**
  * The RepE train statements are split 50/50 into **A** (fine-tuning) and **B** (independent evaluation probes), by normalised-statement group, seed 20261107, balanced by factual label.
  * The split file and its SHA-256 are recorded before training.
  * Validation (89 statements, 326 excluded) and the cross-wording set are used for evaluation.
  * The test split is touched only in §6.
* **Held-out wording for test (needs approval):** a new, third template pair, fixed now, applied to the 90 test statements:
  * "USER: You are a sincere assistant who always tells the truth. ASSISTANT: …"
  * "USER: You are a deceptive assistant who often tells lies. ASSISTANT: …"
  * File `data/repe/heldout_wording_test.csv` plus its activation cache. No new wording is created for train.
* **Arms** (per architecture × seed; all initialised from the base; GPT-2 frozen; only SAE parameters trained):

  | Arm | Loss | Status |
  |---|---|---|
  | Base | — | reused |
  | Method A (GoT recon + OWT replay) | existing | reused, reference only |
  | **R-recon** (control) | 0.5·MSE(RepE-A rows) + 0.5·MSE(OWT train) | new |
  | **R-honesty** | R-recon + λ·L_pair; L_pair = mean‖(x̂_H − x̂_U) − (x_H − x_U)‖² / mean‖x_H − x_U‖² over A pairs (same statement) | new |
  | **R-samepersona** (control) | R-recon + λ·L_pair on *same-persona, different-statement* pairs from A | new |

  * The R-samepersona control separates "honesty contrast" from "extra weight on paired differences of RepE prompts".
  * **No probe enters any loss.**
* **Protocol** (the Method A reference protocol, otherwise unchanged):
  * batch: 32 A pairs (64 RepE rows) + 128 OWT rows;
  * Adam, lr 1e-5, 500 steps, clip 10, decoder renormalisation, same batch RNG across arms;
  * no checkpoint selection.
  * **Sparsity:** TopK has L0 fixed by k. ReLU/JumpReLU keep their penalty, with the multiplicative controller continued from the base coefficient and its EMA L0 measured on OWT rows (as in Method A).
* **λ:** train-only scale matching per SAE, so that λ·L_pair = 0.5·MSE_RepE on A at initialisation. A single value, with no sweep. Recorded in `preregistration.json` before training.
* **Runs:** 3 new arms × 3 architectures × 3 seeds = **27**, plus 1 reproduction check (R-recon TopK seed 42 retrained twice, which must be bit-identical; excluded from results).
* **Known limitation, accepted:** A has about 208 statements (416 rows), so 500 steps ≈ 77 passes over A. Overfitting will show on validation.

## 3. Metrics

All metrics are scored on validation (89 statements) and cross-wording (89). The test sets are scored in §6 only.

| Metric | Category | Direction |
|---|---|---|
| **B-MLP AUROC gap** (768-64-1, 5 inits, fitted on original B rows, applied frozen) — **primary** | representation preservation | lower |
| B-logistic gap (C by grouped 5-fold CV within B; no validation use) | representation preservation | lower |
| B-mass-mean gap | representation preservation | lower |
| Pair accuracy (score_H > score_U per statement), recon vs original | representation preservation | higher |
| Margin retention: mean(s_H − s_U) on recon ÷ on original (logistic logit and mass-mean projection) | representation preservation | closer to 1 |
| Direction cosine (mass-mean of validation differences, recon vs original) | representation preservation | higher |
| Legacy all-train probes (shares A rows with training → DIAGNOSTIC only) | diagnostic | — |
| MSE, EV on OWT validation and RepE validation; cosine | reconstruction | lower / higher |
| L0 on OWT, GoT, RepE (reported separately); dead fraction (OWT) | sparsity | match / lower |
| OWT spliced CE increase and KL; RepE persona-shift cosine (splicing) | behavioural faithfulness | lower / higher |
| GoT geographic B-MLP gap (existing layer-6 evaluation) | guard | lower |

**Caveat:** L_pair targets the same contrast that the mass-mean probe reads, although on different rows and with no shared parameters. A mass-mean gain alone therefore does **not** count as evidence. The primary metric is the nonlinear B-MLP probe, and transfer to held-out wording is required.

## 4. Pre-registered criteria

These follow D14: n = 3 seeds, paired by seed, descriptive rules. "Consistent" means better on all 3 seeds AND the mean paired difference > 2 × the sd of the paired differences.

* **Per architecture:**
  * **R1a:** R-recon has a consistently lower B-MLP gap than Base (validation).
  * **R1b (primary):** R-honesty has a consistently lower B-MLP gap than R-recon (validation) AND a lower one than R-samepersona on all 3 seeds.
  * **R1-transfer:** on cross-wording, the mean paired difference (R-honesty − R-recon) of the B-MLP gap is < 0, and it is < 0 on ≥ 2/3 seeds.
* **Guards** (for each fine-tuned arm vs its own base):
  * OWT L0 within ±5 %;
  * **RepE-prompt L0 within ±5 %** (prompt-specific sparsity control, S3);
  * OWT validation EV ≥ base − 0.02 on every seed;
  * dead fraction ≤ base + 0.05;
  * OWT spliced CE increase ≤ base + 0.01 nats (mean);
  * GoT geographic B-MLP gap ≤ base + 0.02 (mean).
  * A run that fails a guard is flagged and excluded from rankings, not deleted.
* **Verdict per architecture:**
  * **SUPPORTED** = R1b AND R1-transfer AND all guards.
  * **NOT SUPPORTED** = R1b fails.
  * **INCONCLUSIVE** = otherwise.
  * One metric improving while another regresses is not success.
* **Exploratory, not part of the verdict:**
  * gap vs RepE L0 across all SAEs;
  * TopK base re-evaluated at k ∈ {164, 180}, matched to the ReLU/JumpReLU RepE L0 (off-distribution k, so labelled EXPLORATORY);
  * architecture ranking (unmatched on RepE L0).

## 5. Evaluation protocol and leakage safeguards

* Probes are fitted once on **original** B activations and saved with their hashes, before any fine-tuned SAE is evaluated.
* They are never refitted on reconstructions; refit decodability is a labelled diagnostic only.
* A rows are never used by an evaluation probe. B, validation and test rows are never read by the training script (asserted in code).
* Statement 326 is excluded from validation.
* Script, split-file, probe and pre-registration SHA-256 hashes go into `pretraining_hashes.json` before the 27 runs.
* Outputs refuse to overwrite. Checkpoints go to `results/repe/sae_finetuning/r1_honesty_preservation_layer6/<arm>_<arch>_seed<s>/`.

## 6. Final test pass (pre-registered, once)

* After the validation verdict is written and committed, all R1 SAEs (Base, Method A, R-recon, R-honesty, R-samepersona) are scored **once** on:
  * RepE test, default wording (90 statements);
  * the new held-out-wording test set.
* The metrics are those of §3. Nothing is selected on test.
* Prior test exposure (6c, all-layer) is disclosed. Those analyses involved no R1 arm.

## 7. Compute estimate (A10G 23 GB; 15 GB RAM)

* 27 fine-tunes, about 20–40 s each (no GPT-2 forward in the loss): **≈ 15–20 min**.
* Evaluation of 45 SAEs (9 base + 9 Method A + 27 new), layer-6 evaluation + probes + splicing, about 12 s each: **≈ 10–15 min**, plus the test pass ≈ 5 min.
* Held-out-wording extraction: < 2 min.
* Peak GPU memory is expected below 6 GB.

## 8. Files (planned)

* **Scripts:**
  * `scripts/repe/14_prepare_r1_r2_splits_and_wording.py`: A/B split, exclusion list, held-out wording and its cache;
  * `15_r1_honesty_finetune.py`;
  * `16_r1_evaluation.py`;
  * `19_r1_r2_figures.py`.
* **Results:** this folder (`preregistration.json`, `pretraining_hashes.json`, `per_sae.csv`, `summary.csv`, `criteria.json`, `test_pass/`).

## 9. After execution

To be completed.

## 10. Pre-training addendum (2026-11-07, written before any R1 training run)

§0–§9 above are the plan as drafted and committed in `d17bb00`; they are not edited. This addendum records the
corrections and the details that the draft left open. **No success criterion, guard threshold, arm, seed or training
hyperparameter is changed.**

**Corrections:**
1. **Test pass (§6) and held-out-wording test set (§2): DEFERRED.**
   * On 2026-11-07 the user instructed: use only training and validation data for model selection, and preserve the final test split.
   * The R1 verdict was already defined on validation and cross-wording validation, so the verdict is unaffected.
   * §6 runs only after explicit approval. `scripts/repe/14_prepare_r1_r2_splits_and_wording.py --heldout-wording` refuses to run.
2. **Verdict scope of the guards.**
   * "All guards" in the verdict means the guards of R-honesty **and** R-recon, the arms the primary comparison uses.
   * R-samepersona guard failures are flagged but do not change the verdict.

**Details fixed before training:**
* **Split:** `data/repe/splits/r1_train_AB_groups.csv`. 208 A / 209 B statements; factual-label means 0.495 / 0.498; SHA-256 in `split_verification.json`.
  * Normalised-statement grouping: lower-case, non-alphanumerics → space.
  * Statements 436/509 are in the same half.
  * 0 overlap with validation or test.
* **L_pair** (R-honesty) = mean_i ‖(x̂_H,i − x̂_U,i) − (x_H,i − x_U,i)‖² / D_H.
  * D_H = mean ‖x_H − x_U‖² over all 208 A pairs (= 15.06).
* **L_same** (R-samepersona) uses the same 32 statements of the batch.
  * Pair i = (statement i, statement (i+1) mod 32), honest rows for even i and untruthful rows for odd i.
  * Denominator D_S comes from the same construction over all of A, with statement j paired with j+1 (= 6570).
  * The denominators differ by about 400×, because different-statement differences are much larger than persona differences. λ's scale matching makes the two terms equal in loss weight at initialisation.
* **λ:** per SAE and arm, computed at initialisation on all A rows so that λ·L(A) = 0.5·MSE_RepE(A). Values are recorded in `training_info.json`.
* **Probes** (fitted once on the 418 original train-B rows, applied frozen):
  * B-MLP: the GoT script-07 architecture (768-64-1, 50 epochs, lr 1e-3, batch 128), inits 42/123/456/789/2026. The gap is the mean of the 5 per-probe AUROC gaps.
  * B-logistic: C ∈ {0.001, 0.01, 0.1, 1, 10} by statement-grouped 5-fold CV within B.
  * B-mass-mean: unit difference of the B class means.
* **Scored sets:**
  * Validation: 89 statements, 178 rows; 326 excluded.
  * Cross-wording validation: 89 statements, 178 rows; 326 excluded.
* **RepE-prompt L0 guard:** RepE validation rows.
* **Splicing persona-shift cosine:** validation without 326.
* **Pair accuracy:** reported for the B-logistic, B-mass-mean and mean B-MLP scores.
* **Margin retention:** mean(s_H − s_U) on the reconstruction ÷ the same on the original, for the logistic logit and the mass-mean projection.
* **Direction cosine:** mean pair difference (x_H − x_U) on the scored set, reconstruction vs original.
* **Uncertainty (descriptive only):**
  * sd over the 5 MLP probe inits;
  * statement-grouped bootstrap 95 % CI (1000 resamples) of the mean-MLP-score gap;
  * seed sd.
* **Legacy all-train probes** (`DIAG_legacy_repe_*`) are diagnostic only. Their C is chosen on validation with 326 excluded.
* **Exploratory rows:** TopK bases evaluated at k = 164 and 180, matching the ReLU/JumpReLU RepE L0.
* **Method A reference:** ReLU/JumpReLU `got_replay`; TopK = the truth-aware pilot's reconstruction-only checkpoints, as in Method A.

**Smoke tests before freezing (disclosed):**
* **Training:** `15_r1_honesty_finetune.py --smoke` ran 20 steps for each arm on seed-42 SAEs of all three architectures. Nothing was saved.
  * λ·L_pair ≈ 0.32–0.48 at step 1, as designed.
  * **Observation:** the RepE-prompt batch L0 of ReLU and JumpReLU falls within 20 steps (≈ 164 → 146; 182 → 154), while the OWT L0 stays at about 128.
  * The pre-registered RepE-L0 guard may therefore fail. The protocol is **not** changed in response.
* **Evaluation:** `16_r1_evaluation.py --smoke` scored 3 base SAEs (ReLU and TopK seed 42, plus exploratory TopK k = 164) on validation and cross-wording.
  * Output: `smoke/`.
  * No fine-tuned SAE existed, so nothing was selected.
  * **Finding:** pair accuracy is 1.0 for the original and for these reconstructions (ceiling). It is reported but uninformative.

`pretraining_hashes.json` records the hashes of scripts 14–16, the split file, `persona_pairs.csv` and `preregistration.json`, written after this addendum.

## 11. Results (2026-11-07, VALIDATION; test not touched)

Run: 28/28 training runs succeeded (27 + reproduction check, bit-identical, max |Δparam| = 0). Training 58 s
(GPU peak 0.16 GB); evaluation of 51 SAEs (45 main + 6 exploratory) 542 s (GPU peak 8.5 GB). Base checkpoints
re-hashed unchanged after training. Logs: `logs/train.log`, `logs/eval.log`. Commit of the frozen protocol: `211218f`.
Files: `training_info.json` (λ, coefficients, checkpoint SHA-256), `training_trace.csv`, `original_baseline.json`,
`per_sae.csv`, `summary.csv`, `paired_differences.csv`, `criteria.json`, `eval_info.json`; figure
`results/repe/figures/r1_honesty_preservation_layer6.png`.

**Original GPT-2 (frozen probes on original activations; 89 validation statements):** B-MLP AUROC 0.964 (cross-wording
0.928), B-logistic 0.993 (0.977), B-mass-mean 0.894 (0.839); pair accuracy 1.0 for all three probes on both sets.
Every gap below is relative to these originals (gap = AUROC(original) − AUROC(reconstruction)).

**Means over 3 seeds (sd in `summary.csv`):**

| Arch | Variant | val B-MLP gap | cw B-MLP gap | val B-logistic gap | val B-mass-mean gap | val pair-direction cos | RepE val EV | RepE val L0 | OWT val EV | OWT val L0 | OWT ΔCE (splice) | GoT geo B-MLP gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ReLU | base | 0.198 | 0.243 | 0.258 | 0.155 | 0.710 | 0.785 | 163.9 | 0.828 | 133.7 | 0.177 | 0.321 |
| ReLU | Method A | 0.194 | 0.238 | 0.254 | 0.154 | 0.710 | 0.786 | 164.5 | 0.828 | 133.8 | 0.175 | 0.245 |
| ReLU | R-recon | 0.180 | 0.236 | 0.244 | 0.154 | 0.728 | 0.824 | 177.7 | 0.826 | 131.8 | 0.178 | 0.324 |
| ReLU | R-samepersona | 0.175 | 0.236 | 0.237 | 0.152 | 0.729 | 0.828 | 190.2 | 0.825 | 133.2 | 0.175 | 0.322 |
| ReLU | **R-honesty** | **0.167** | **0.225** | 0.244 | 0.134 | **0.915** | 0.820 | 175.1 | 0.826 | 131.9 | 0.178 | 0.323 |
| TopK | base | 0.092 | 0.134 | 0.089 | 0.117 | 0.799 | 0.803 | 128 | 0.853 | 128 | 0.104 | 0.204 |
| TopK | Method A | 0.093 | 0.139 | 0.087 | 0.117 | 0.800 | 0.805 | 128 | 0.854 | 128 | 0.102 | 0.107 |
| TopK | R-recon | 0.088 | 0.117 | 0.076 | 0.117 | 0.811 | 0.832 | 128 | 0.853 | 128 | 0.104 | 0.221 |
| TopK | R-samepersona | 0.091 | 0.116 | 0.080 | 0.121 | 0.811 | 0.832 | 128 | 0.852 | 128 | 0.105 | 0.222 |
| TopK | R-honesty | 0.101 | 0.128 | 0.086 | 0.127 | 0.870 | 0.828 | 128 | 0.852 | 128 | 0.108 | 0.229 |
| JumpReLU | base | 0.072 | 0.109 | 0.068 | 0.083 | 0.838 | 0.845 | 179.4 | 0.862 | 131.2 | 0.093 | 0.292 |
| JumpReLU | Method A | 0.064 | 0.105 | 0.060 | 0.077 | 0.841 | 0.848 | 179.7 | 0.863 | 131.5 | 0.091 | 0.238 |
| JumpReLU | R-recon | 0.067 | 0.094 | 0.056 | 0.078 | 0.850 | 0.873 | 185.0 | 0.860 | 129.0 | 0.094 | 0.296 |
| JumpReLU | R-samepersona | 0.062 | 0.091 | 0.054 | 0.073 | 0.858 | 0.877 | 193.0 | 0.861 | 131.5 | 0.092 | 0.297 |
| JumpReLU | R-honesty | 0.075 | 0.096 | 0.054 | 0.086 | 0.915 | 0.869 | 181.1 | 0.861 | 130.8 | 0.093 | 0.295 |

**Primary paired differences, val B-MLP gap (negative = better), seeds 42 / 1042 / 2042:**

| Arch | R-recon − base | R-honesty − R-recon | R-honesty − R-samepersona | cw: R-honesty − R-recon |
|---|---|---|---|---|
| ReLU | −0.002 / −0.021 / −0.029 (mean −0.017, sd 0.014) | **−0.010 / −0.012 / −0.019 (mean −0.0135, sd 0.0048)** | −0.001 / −0.004 / −0.020 | −0.017 / −0.003 / −0.015 |
| TopK | +0.011 / −0.015 / −0.009 | −0.005 / +0.016 / +0.028 | −0.004 / +0.009 / +0.025 | +0.004 / +0.016 / +0.012 |
| JumpReLU | −0.005 / −0.012 / +0.001 | +0.008 / +0.010 / +0.007 | +0.012 / +0.016 / +0.011 | −0.006 / +0.004 / +0.007 |

**Pre-registered criteria (`criteria.json`):**

| Arch | R1a (R-recon < base) | R1b (primary) | R1-transfer | Guards (R-honesty, R-recon) | **Verdict** |
|---|---|---|---|---|---|
| ReLU | not met (all 3 seeds lower, but mean 0.017 < 2 sd) | **met** | **met** | **fail**: RepE-prompt L0 +6.8 % (R-honesty), +8.4 % (R-recon) vs ±5 % | **INCONCLUSIVE** |
| TopK | not met | not met (R-honesty worse on 2/3 seeds) | not met | fail: R-honesty GoT geographic B-MLP gap +0.025 vs ≤ +0.02 | **NOT SUPPORTED** |
| JumpReLU | not met | not met (R-honesty worse on 3/3 seeds) | not met | pass | **NOT SUPPORTED** |

R-samepersona guard flags: ReLU (RepE L0 +16 %) and JumpReLU (+7.6 %). Every other guard (OWT L0 ±5 %, OWT EV ≥ base
− 0.02, dead fraction, OWT splice ΔCE, GoT gap) passes for every arm and architecture.

**Interpretation (separating the evaluation categories):**
* **Representation preservation (primary, B-MLP).** Only ReLU improves under the honesty objective: the gap falls
  from 0.198 (base) to 0.167, consistent across seeds against R-recon, and transfers to cross-wording (0.243 → 0.225).
  This is a 16 % reduction; the reconstruction still loses ≈ 0.17 AUROC versus the original (0.964), and per-SAE
  statement-bootstrap 95 % CIs (≈ ±0.04) overlap between base and R-honesty on every seed. TopK and JumpReLU, whose
  bases already preserve honesty much better (gaps 0.07–0.09), get no benefit; for JumpReLU R-honesty is slightly worse
  than R-recon on every seed.
* **The ReLU gain is not corroborated by the independent logistic probe** (R-honesty − R-recon = −0.0003). It does
  appear in the mass-mean gap (−0.021), but that probe reads the same contrast L_pair optimises, so per §3 it is not
  evidence.
* **Direction-level metrics move a lot, as designed.** Pair-difference direction cosine rises by 0.20 (ReLU),
  0.07 (TopK) and 0.08 (JumpReLU), and logistic margin retention rises (ReLU 0.73 → 0.80). These measure the objective
  itself; they do not translate into probe AUROC except, partially, for ReLU.
* **Sparsity confound.** The ReLU guard fails because RepE fine-tuning raises the RepE-prompt L0 (164 → 175–190) while
  OWT L0 stays at its target (the controller only sees OWT rows). R-honesty has a *lower* RepE L0 than R-recon and
  R-samepersona yet a lower gap, so extra active latents do not explain the R-honesty − R-recon difference; but the
  comparison with the base is not sparsity-matched on RepE prompts. Exploratory TopK bases at k = 164/180 lower the
  gap only modestly (0.092 → 0.082 / 0.075); across the 45 main SAEs, Spearman ρ(gap, RepE L0) = −0.34 (p = 0.024),
  driven mostly by architecture.
* **Reconstruction and behaviour.** RepE validation EV rises by 0.024–0.040 in every arm (R-honesty ≈ 0.004 below
  R-recon); train-A EV rises more (e.g. ReLU R-recon 0.891 vs validation 0.824), i.e. some overfitting to the 208
  A statements. OWT reconstruction is essentially unchanged (EV −0.0006 to −0.002) and OWT splice ΔCE changes by
  ≤ 0.005 nats. RepE splice persona-shift cosine rises slightly in every arm (behavioural; descriptive only).
* **Pair accuracy is at ceiling (1.0) everywhere** and does not discriminate (disclosed at the smoke test).
* **Method A (GoT fine-tuning) does not change RepE preservation** (all differences ≤ 0.008), as found before.

**Answer to the R1 question at validation level:** honesty-aware fine-tuning gives a small, seed-consistent,
wording-transferring improvement of the nonlinear-probe honesty gap for ReLU only, and that improvement is confounded
by RepE-prompt L0 drift (guard failed) and not seen with the logistic probe → INCONCLUSIVE. For TopK and JumpReLU the
hypothesis is not supported. No reconstruction or OWT behaviour damage was observed.

**Limitations:** 3 seeds; 89 validation statements (single-SAE CIs ≈ ±0.04 AUROC); one λ (no sweep, by design);
500 steps over 208 statements overfit; RepE L0 not controlled on RepE prompts; GPT-2 Small layer 6 only; the
pre-registered test pass (§6) has not been run (deferred by the user).

**Next steps:** (1) the §6 test pass only with explicit approval; (2) a RepE-prompt-L0-matched variant (controller on
RepE + OWT rows) would resolve the ReLU confound; that is a new experiment and needs its own pre-registration.

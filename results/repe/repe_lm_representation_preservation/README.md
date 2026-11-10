# RLM-RP: RepE-LM representation preservation and robustness (216 SAEs, layers 0–11). Status: COMPLETED (VALIDATION)

Evidence level, once run: **VALIDATION**. Probes and the C choice use TRAIN rows only, and scores use VALIDATION rows.
RepE-LM has no test rows. Approved by the user on 2026-11-10 ("Complete RepE representation preservation ... Complete
RepE generalization and robustness").

## 1. Before implementation

### 1.1 Question and why it is needed

* **Question:** when an SAE (base or fine-tuned) replaces GPT-2's final-token residual on the corrected,
  equal-length RepE-LM persona prompts, how much persona information is kept? The question is asked at four levels:
  * a frozen original-activation reader;
  * a reader re-trained on reconstructions;
  * the geometry of the H − U difference;
  * transfer to held-out instruction wording.

  Does fine-tuning (IMP-L at layers 3/8/9; layer-6 Method A / output-KL) change any of these?
* **Why:**
  * M2 measured base SAEs only, without direction geometry, MSE/EV, dead fraction or held-out wording.
  * IMP-L measured only the re-trained persona gap for fine-tunes.
  * The historical RepE results (`representation_preservation_val_test.csv`, `all_layer_frozen_probe_evaluation/`)
    use the length-confounded prompts and are **not** used or substituted here (D23).
* **Link to the main research question:** this is the representation-preservation level for RepE. The causal level
  is in M4 and E3.

### 1.2 Data (RepE-LM, `data/repe/repe_lm/prompts.csv`, hash recorded)

* **Splits:** 417 train and 89 validation statements. Statement 326 was dropped by M1 (near duplicate). There are
  no test rows (asserted).
* **Rows used** (final-token activations, cache `data/repe/activations/repe_lm/final_layer{L}.npy`):

  | Role | Templates and variants | Used for |
  |---|---|---|
  | persona | T1–T4, H (label 1) vs U (label 0) | probes fitted on TRAIN; scored on VALIDATION |
  | synonym | T1–T4 Hp (honest synonym) | VALIDATION, frozen persona probe: Hp vs U, and H vs Hp |
  | held-out wording | T5 ("true/false facts"), T6 ("correct/wrong things"), H vs U | VALIDATION only, never fitted |
  | neutral control | N1 (tall/short), N2 (red/blue) | own probe fitted on TRAIN; scored on VALIDATION |

* **Grouping:** by `norm_group` (statement), for C selection and the bootstrap.
* **Checks recorded in `checks.json`:**
  * no `norm_group` in both train and validation;
  * every pair has equal token length (the length-shortcut control, gate K1);
  * no test rows.

### 1.3 SAEs (216; no training)

| Set | Count | Layers | Source |
|---|---|---|---|
| base | 108 | 0–11 | uninterrupted 15k-step OWT SAEs, ReLU/TopK/JumpReLU × 42/1042/2042 |
| IMP-L | 81 | 3, 8, 9 | `owt_only`, `method_a`, `kl` (IMP-L; base = uninterrupted) |
| layer 6 | 27 | 6 | `owt_only` and Method A (`method_a_layer6/*_owt_only`, `*_got_replay`; TopK Method A = `truth_aware_pilot_layer6_topk/reconstruction_only_*`), output-KL = E2E Method B (`e2e_inspired_layer6/*_methodB_*`); every fine-tune's `source_checkpoint` is the uninterrupted base, which is asserted |

Every checkpoint's SHA-256 is written to `per_sae.csv`.

### 1.4 Metrics per SAE (VALIDATION; direction in brackets)

* **Persona (T1–T4, H vs U):**
  * the original AUROC of the frozen logistic probe (grouped-CV C on TRAIN, `il.fit_logistic`) and of the mass-mean
    direction (TRAIN);
  * reconstructed AUROC;
  * **frozen gap** = original − reconstructed AUROC (↓);
  * **re-trained gap** = original AUROC − AUROC of a logistic probe re-fitted on TRAIN reconstructions (↓);
  * a norm-matched isotropic-noise reference for the frozen gaps (200 draws, exact linear projection as in M2);
  * the error-term AUROC (w·e);
  * statement-bootstrap 95 % CIs (1000) for the frozen and re-trained logistic gaps.
* **Reconstruction:**
  * on persona validation rows (M2 definitions): relative squared error, NMSE, cosine, L0;
  * on all RepE-LM validation rows: per-element MSE (↓), EV = 1 − Σ‖x − x̂‖² / Σ‖x − x̄‖² (↑), cosine (↑), L0
    (match), dead fraction (latents never active on these prompts, ↓).
* **Direction:** for each validation pair, Δ = x_H − x_U and Δ̂ = x̂_H − x̂_U.
  * mean cos(Δ, Δ̂) (↑) and mean ‖Δ̂‖/‖Δ‖ (target 1);
  * mass-mean direction cosine and norm ratio;
  * error share ‖Δ − Δ̂‖/‖Δ‖ (↓).
  * **Invalid** (reported as NaN) where ‖Δ‖ < 1e-4. At layer 0 the final token's residual is identical for H and U
    by construction.
* **Held-out wording (T5/T6, never fitted):**
  * frozen persona-probe AUROC, original and reconstructed, and their gap (↓);
  * the persona probe re-fitted on reconstructions, applied to reconstructed T5/T6: transfer gap vs the original (↓).
* **Synonym:** frozen-probe AUROC Hp vs U (original and reconstructed; ↑ close to the original), and H vs Hp (wording
  sensitivity; close to 0.5 means the probe reads meaning, not the word).
* **Neutral control (N1/N2):** frozen and re-trained gaps on the neutral word pair. Persona minus neutral shows
  whether a loss is persona-specific.
* **Controls:**
  * norm-matched noise (reconstruction error as such);
  * neutral pairs (generic word differences);
  * synonym and held-out wording (prompt wording);
  * equal-length check (length shortcut);
  * near-duplicate sensitivity: persona gaps recomputed without the 14 validation statements that have a Jaccard
    > 0.8 near duplicate in train (`split_verification.json`);
  * an **exploratory** L0-truncation control for ReLU/JumpReLU: keep each token's 128 largest latents, decode, and
    recompute the frozen and re-trained persona gaps, so that L0 on RepE-LM prompts matches TopK (128). This is a
    post-hoc truncation, not a trained matched-L0 SAE (that is E1). It is labelled exploratory.
* **Layer 0:** reported; persona AUROC is 0.5 by construction (M2), so layer 0 is excluded from every summary
  statement.

### 1.5 Statistics and pre-registered criteria

* **Per SAE:** statement-bootstrap CIs.
* **Per layer × architecture:** mean ± sd over seeds, per-seed values, 95 % t-interval (t = 4.303, df 2). No pooling
  across layers for raw MSE.
* **Fine-tune comparisons** (layers 3, 6, 8, 9; paired by seed): `owt_only − base`, `method_a − base`,
  `method_a − owt_only`, `kl − base`, `kl − method_a`.
* **Pre-registered classes, IMP-L rule** ("consistent" = better on all 3 seeds and |mean| > 2 sd), on **two separate
  metrics**, so that frozen and re-trained readers are never merged:
  * **RP-A-frozen / RP-A-refit:** Method A lowers the persona frozen logistic gap / the re-trained gap, consistently
    vs base **and** vs OWT-only;
  * **RP-KL-frozen / RP-KL-refit:** output-KL lowers the same gap vs Method A, consistently.
  * **Classes:** PRELIMINARY IMPROVEMENT; NO IMPROVEMENT; REGRESSION (consistently higher; vs base for A, vs
    Method A for KL); NOT CREDITABLE.
  * NOT CREDITABLE means an IMP-L guard failed. Guards come from IMP-L `criteria.json` (OWT L0 ±5 %, EV, ΔCE, cities)
    for layers 3/8/9. At layer 6 the IMP-L-format guards do not exist, so layer-6 classes are reported **without
    guard checks** (marked).
* **RepE-LM L0 ±5 % of base** (CLAUDE.md) is reported as a flag for every fine-tune cell. It does not change the
  class, because the pre-registered guard is defined on OWT.
* **Prior expectations, stated before running:**
  * Method A (trained on GoT/OWT) does not change persona preservation;
  * output-KL may change it slightly;
  * re-trained gaps are much smaller than frozen gaps (as in M2).

### 1.6 Reproduction checks (before results are read; `checks.json`)

* Base SAEs vs M2 `per_cell.csv` (task `repe_persona`): `frozen_logistic_gap`, `frozen_massmean_gap`,
  `refit_logistic_gap`, `task_cosine`, `task_l0`, `task_nmse`, max abs diff < 1e-6.
* IMP-L SAEs (layers 3/8/9, all four arms) vs IMP-L `repe_lm_persona_refit_logistic_gap` (< 1e-3) and
  `repe_lm_persona_val_l0` (< 1e-4).
  * Why these tolerances: the smoke test showed differences of 5.5e-5 and 6.9e-6. IMP-L encoded only the persona
    rows, so its GPU batches differ, and it averaged L0 in float32.
  * The tolerances were set from the smoke test's mechanics output, before any full-run value was seen. M2 is
    reproduced exactly (0.0).
* Truncation: the untruncated decode `z @ W_decᵀ + b_dec` reproduces the SAE output (max abs < 1e-3).

### 1.7 Compute and outputs

* About 216 SAE forward passes on 10,120 cached rows, plus about 3 logistic re-fits per SAE (CPU, 3 workers).
  Estimated 45–90 min. No GPT-2 forward pass; GPU only for SAE encode/decode.
* **Script:** `scripts/repe/26_repe_lm_representation_preservation.py` (`--smoke DIR`, `--run`; refuses to
  overwrite; checks hashes against `preregistration.json`).
* **Outputs:** `per_sae.csv`, `summary_by_layer_arch.csv`, `comparisons.csv`, `criteria.json`, `checks.json`.

## 2. After execution (run 2026-11-10, 493 s; VALIDATION)

* **Run:** `--prepare` → hashes; `--run` with the hashes verified. 216/216 SAEs evaluated; 0 failures; 0 test rows.
  Log (local): `results/repe/logs/repe_lm_representation_preservation_2026-11-10.log`.
* **Checks** (`checks.json`; all pass):
  * no statement or `norm_group` in both splits;
  * every pair is equal in token length;
  * the 14 near-duplicate validation statements are listed;
  * the truncation decode reproduces the SAE output exactly (0.0);
  * **M2 reproduced exactly** on all 108 base SAEs: frozen logistic, frozen mass-mean and re-trained gaps, cosine,
    L0, NMSE; max difference 0.0;
  * **IMP-L reproduced** on 108 SAEs: re-trained persona gap max difference 1.6e-4 (tolerance 1e-3), persona L0
    7.4e-6 (tolerance 1e-4).
* **Outputs:** `per_sae.csv` (216 rows, with checkpoint SHA-256), `original_by_layer.csv`,
  `summary_by_layer_arch.csv` (mean, sd, 95 % t-interval and per-seed values for every metric), `comparisons.csv`,
  `criteria.json`, `checks.json`.
* **Paper:** tables `t_rlm_layers` and `t_rlm_finetune`, figure `fig_rlm_layers`.

### 2.1 Base SAEs, layers 1–11 (99 SAEs; layer 0 is 0.5 by construction)

Cells give the range, over layers 1–11, of the per-layer mean over 3 seeds. Per-seed values and intervals are in `summary_by_layer_arch.csv`.

| Quantity | ReLU | TopK | JumpReLU |
|---|---|---|---|
| Original frozen-logistic persona AUROC (same for all) | 0.657 (L1) → 0.865 (L11) | | |
| Frozen logistic gap ↓ | 0.113–0.243 | 0.095–0.197 | 0.109–0.184 |
| Noise reference (same-norm isotropic) | 0.033–0.097 | 0.037–0.091 | 0.035–0.080 |
| Frozen mass-mean gap ↓ | 0.035–0.125 | 0.029–0.083 | 0.027–0.079 |
| Re-trained logistic gap ↓ | 0.095–0.223 | 0.083–0.181 | 0.097–0.177 |
| Held-out wording T5/T6 frozen gap ↓ (original AUROC 0.54–0.69) | 0.026–0.128 | 0.026–0.100 | 0.028–0.098 |
| Per-pair cos(Δ, Δ̂) ↑ | 0.52–0.58 | 0.35–0.43 | 0.38–0.48 |
| Per-pair ‖Δ̂‖/‖Δ‖ (1 = kept) | 0.65–0.71 | 1.32–1.46 | 1.22–1.35 |
| Mass-mean direction cosine ↑ | 0.72–0.80 | 0.70–0.86 | 0.74–0.86 |
| EV on RepE-LM validation prompts ↑ | 0.66–0.90 | 0.73–0.87 | 0.73–0.92 |
| L0 on RepE-LM prompts (OWT-matched ≈ 128–135) | 160–214 | 128 | 142–208 |
| Latents inactive on all RepE-LM validation prompts | 0.10–0.15 | 0.44–0.67 | 0.51–0.80 |

* **Persona information is really lost, not only misread.**
  * The frozen gap is 1.9–3.4 times the noise reference (mean 2.5, 99 SAEs). On GoT, by contrast (M2), noise
    reproduced most of the frozen collapse.
  * The re-trained gap is almost as large as the frozen gap, and its statement-bootstrap 95 % CI excludes 0 for every
    one of the 99 SAEs.
  * This reproduces and extends M2.
* **The H − U difference is distorted, not just shrunk.**
  * Per pair, the reconstructed difference has a cosine of only 0.35–0.58 with the original difference.
  * ReLU shrinks it (norm 0.65–0.71). TopK and JumpReLU enlarge it (1.2–1.5), with an error part larger than the
    difference itself (`pair_error_share` 1.16–1.50).
  * Averaged over pairs, the mass-mean direction is kept better (cosine 0.70–0.86).
  * This is the representational counterpart of M4's L8/L11 causal leakage.
* **Wording generalisation:**
  * The persona probe transfers weakly to held-out wording T5/T6 even on original activations (AUROC 0.54–0.69).
  * The SAEs lose 0.03–0.13 of that.
  * Synonym prompts (Hp vs U) behave like the persona prompts (frozen synonym gap within 0.024 of the persona gap for every SAE).
  * The frozen probe barely separates H from Hp (AUROC 0.52–0.54 on original activations, 0.50–0.53 on
    reconstructions), so it mostly reads the persona meaning, not the specific word.
* **Specificity:** the re-trained persona gap exceeds the re-trained neutral (tall/short, red/blue) gap by 0.06–0.12
  at layers 2–8, but by only 0.007 at L1, 0.01 at L10 and −0.018 at L11. Mid-layer loss is partly persona-specific;
  late-layer loss is generic.
* **Architecture comparison is confounded by L0:**
  * TopK and JumpReLU have smaller persona gaps than ReLU at most layers, but on these prompts ReLU and JumpReLU use
    more latents (up to 214) than TopK (128).
  * **Exploratory truncation control:** keeping each token's 128 largest latents raises JumpReLU's re-trained gap
    from 0.144 to 0.210 at L3, from 0.177 to 0.216 at L6 and from 0.162 to 0.188 at L8. ReLU changes by at most 0.009 (L5).
  * So part of JumpReLU's apparent advantage is extra L0. This is post-hoc truncation, not a trained matched-L0 SAE
    (that is E1).
* **Near-duplicate sensitivity:** dropping the 14 validation statements with an opposite-label near duplicate in
  train changes any persona gap by at most 0.017 (frozen) and 0.020 (re-trained). No qualitative change.

### 2.2 Fine-tuned SAEs (pre-registered classes; frozen and re-trained readers kept separate)

* **Method A:** NO IMPROVEMENT in all 12 layer × architecture cells, on both readers.
  * Method A − base is between −0.004 and +0.006 AUROC everywhere.
  * RepE-LM L0 stays within ±5 % on every seed.
  * As expected: Method A trains on GoT and OWT, not on persona prompts.
* **Output-KL vs Method A:**

  | Layer | Frozen reader | Re-trained reader | RepE-LM L0 change |
  |---|---|---|---|
  | 3 | PRELIMINARY IMPROVEMENT, all 3 architectures (−0.004 to −0.006) | PRELIMINARY IMPROVEMENT, ReLU and TopK (−0.003, −0.006) | +4.0 to +4.9 %, within ±5 % |
  | 6, JumpReLU | NO IMPROVEMENT | PRELIMINARY IMPROVEMENT (−0.0055) | +6.2–6.5 %, outside ±5 % |
  | 6, ReLU and TopK | NO IMPROVEMENT | NO IMPROVEMENT | |
  | 8, 9 | ReLU and JumpReLU NOT CREDITABLE (IMP-L L0 guard); TopK NO IMPROVEMENT | same | |

  * Layer-6 classes are not guard-checked, because no IMP-L-format guard exists there (stated in advance).
  * Every output-KL change is at most 0.007 AUROC, against gaps of 0.12–0.24. Practically negligible.
  * Output-KL also raises RepE-LM L0 for ReLU/JumpReLU by 4–9 %, and the truncation control shows that L0 matters
    for these gaps.
  * The per-pair direction cosine rises slightly for ReLU/JumpReLU under output-KL (+0.003 to +0.010) and falls for
    TopK (−0.004 to −0.007).
* **No regression in any cell.**

### 2.3 Criteria summary

| Criterion | PRELIMINARY | NO IMPROVEMENT | NOT CREDITABLE | REGRESSION |
|---|---|---|---|---|
| RP-A-frozen | 0/12 | 12/12 | 0 | 0 |
| RP-A-refit | 0/12 | 12/12 | 0 | 0 |
| RP-KL-frozen | 3/12 (layer 3) | 5/12 | 4/12 | 0 |
| RP-KL-refit | 3/12 (L3 ReLU, L3 TopK, L6 JumpReLU) | 5/12 | 4/12 | 0 |

### 2.4 Limitations

* **Validation only** (no RepE-LM test split exists, by design).
* **Three seeds:** the "consistent" rule (all seeds plus 2 sd) is looser than a 95 % t-interval with 2 df, and some
  PRELIMINARY cells have intervals that touch 0 (L3 ReLU re-trained).
* **Final-token readout only.**
* **Weak transfer:** the held-out templates transfer weakly even on the original model.
* **Truncation is a post-hoc control,** not a matched-L0 SAE.
* **Layer-6 classes have no guard check.**
* **"Inactive latents" are counted on about 3,000 RepE-LM validation tokens,** not a dead-feature measure on OWT.

### 2.5 Status

COMPLETED (VALIDATION). Evidence label: validation finding; classes are preliminary.

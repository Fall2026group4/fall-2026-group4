# SAE Faithfulness: final research report (verified 2026-11-10)

**Scope.** GPT-2 Small (frozen) with in-house ReLU, TopK, JumpReLU and BatchTopK SAEs, measured on OpenWebText
(OWT), Geometry of Truth (GoT) and RepE-LM persona prompts. Gemma-2-2B appears only as an SAE-free steering appendix.

**Verification.** Every number here was read from, or recomputed from, a committed artifact. The detailed tables
and paths are in `RESULTS_VERIFICATION_TABLE.md`, and the reproducibility audit is in `EXPERIMENT_AUDIT.md`.

**Evidence labels:**
* **TEST** = the M5 confirmatory pass (single run, pre-registered).
* **VALIDATION** = validation split.
* **HISTORICAL** = kept for provenance only.
* **INVALID** = never cite.

---

## Part 1. Research question and motivation

**Central question.** How faithfully do sparse autoencoders preserve the internal representations, truth
information, behaviour and causal effects of a frozen language model?

1. **Problem.**
   * SAE features are used to explain what a model computes ("feature 974 = Paris" in our Phase 1).
   * Such explanations are only valid if the SAE's reconstruction keeps what the model actually uses.
   * The field mostly reports reconstruction (EV) and sparsity (L0).
2. **Why SAEs.**
   * An SAE maps an activation x ∈ ℝ⁷⁶⁸ to a sparse code z ∈ ℝ^m, with m = 3,072 or 12,288 and about 128 non-zero
     entries, and decodes x̂ = W_dec z + b_dec.
   * Sparse, overcomplete codes tend to give more interpretable, monosemantic units than neurons or PCA components.
3. **Why reconstruction alone is insufficient.**
   * EV weights directions by variance; the network may depend on low-variance directions.
   * **Verified (TEST):** R2 TopK at L8 has EV 0.906 and cosine 0.973, yet raises next-token CE by 0.062 nats. Its
     error is 1.20× as harmful as a random error of the same norm.
4. **What faithfulness means here: four separately measured levels.**
   1. **Reconstruction:** NMSE, EV, cosine.
   2. **Behaviour:** ΔCE and KL when x̂ replaces x in the forward pass.
   3. **Information:** truth and persona readout from x̂ by frozen and re-fit probes, against PCA and noise references.
   4. **Causal:** the share of an instruction's causal effect that passes through x̂ (M4 decomposition).

   Steering, which adds directions to activations, is a fifth and separate category.
5. **Why ReLU, TopK, JumpReLU.**
   * They are the three main sparsity mechanisms: L1 shrinkage, hard top-k, and a learned threshold.
   * Their distortions differ, so faithfulness may differ at equal L0.
   * R2 adds BatchTopK, because a modern JumpReLU could not be trained to target L0 in our budget.
6. **Why GPT-2 Small and Gemma-2-2B.**
   * GPT-2 Small is cheap enough for 100+ SAEs × 3 seeds × 12 layers, and has public SAEs (res-jb) as an external
     reference.
   * Gemma-2-2B was planned for cross-model validation with Gemma Scope SAEs. **It has not been done**: only
     SAE-free steering exists.
7. **Contribution beyond training SAEs.**
   * A controlled, L0-matched, multi-seed, pre-registered measurement across all four levels, with:
     * noise, PCA, random-decoder and supervised-DAS references;
     * independent probes, after a probe-circularity audit;
     * one confirmatory held-out test.
   * **Main finding: the levels dissociate.** Large, confirmed gains in reconstruction and behaviour did not translate
     into truth preservation, and a behaviour-targeted fine-tune reduced causal preservation (VALIDATION).

**Established vs future work.**
* *Established (TEST):* R2 and R4 improve OWT reconstruction and behaviour, and there is no confirmed GoT change.
* *Established (VALIDATION):* M0–M4, R5, E2 and E3 dissociations.
* *Future:* RepE-LM test split, Gemma SAEs, task-L0 matching (E1), width/L0 frontiers, isolating token count from
  recipe.

---

## Part 2. Experimental history

### Stage A: before the professor review (2026-09 to 2026-11-03)
* **Phase 1 (exploratory):** public res-jb L8 feature 974 fires on "Paris" (57.5 vs 20.1 for the runner-up).
* **In-house OWT pipeline (`8dc0b6e`):**
  * 800/100/100 documents (101,600 / 12,700 / 12,700 tokens); width 3,072; Adam 3e-4; batch 256;
  * 1000-step sweep at L0 32/64/128 × 12 layers;
  * convergence to 15k steps at layers 2/6/10.
* **GoT layer 6 pipeline:**
  * original probe validation AUROC 0.841 (geographic);
  * fine-tunes: reconstruction-only and truth-aware.
  * Test MLP gap: base 0.327/0.154/0.323 → recon-only 0.161/0.110/0.146 (ReLU/TopK/JumpReLU). **L0 not matched.**
* **RepE (old persona prompts):** layer-6 mass-mean AUROC 0.893, plus "SAE steering" correlations of 0.93–0.99.
* **Team branches:** a public-SAE OWT pipeline; GoT L10 legacy (frozen-probe issue below); a Gemma layer table with a
  leaky row split. None were merged.

### Stage B: professor review (2026-11-01) and corrections (2026-11-04)
Source: `docs/archive/2026-11-04_methodology_review_corrections_report.md`.

| Criticism | What was wrong | Fix | Outcome |
|---|---|---|---|
| Optimiser inconsistency | L6 was resumed with Adam restored; L2/L10 were warm-started with Adam reset | 27 runs retrained, uninterrupted 15k steps, 3 seeds | EV moved ≤ 0.006; ranks unchanged; 2 wins → ties |
| Sparsity matching | JumpReLU matched L0 32 in 2/12 layers; `architecture_summary.csv` averaged unmatched runs and raw MSE across layers | ±10 % rule; matched-cell tables; the summary declared INVALID | Only 15/36 sweep cells comparable; the TopK lead at 1000 steps reverses by 15k |
| Causal claims | "SAE steering" never put an SAE in the forward pass | Pre-registered test, 200 controls, positive control | Raw direction E = 0.0048 (p = 0.010); SAE directions not distinguishable |
| GoT L10 | Probe re-trained on reconstructions (decodability, not faithfulness) | Frozen original probe | AUROC 0.851 → 0.639 frozen (gap 0.21 [0.16, 0.27]) |
| RepE | One misquoted number; L0 unmatched on RepE prompts | Verified; disclosed | Later found length-confounded (D23) → HISTORICAL |
| Gemma | "Both signs → TRUE" | 20 random directions per strength | A generic large-perturbation bias |
| Reproducibility | Mixed procedures, missing scripts | One protocol; manifests; checkpoint registry | Kernel-STE JumpReLU script still missing |

Follow-ups:
* all 12 layers × 3 architectures × 3 seeds = **108-run corrected baseline** (TEST; 107/108 matched);
* the P1 sparsity controller;
* the **truth-aware circularity audit** (D12, D16).

### Stage C: corrected baselines and improvement attempts (2026-11-07 to 11-10)
* **Measurement programme (VALIDATION):**
  * **M0:** within a layer, error predicts splice damage (β 0.165, Holm p 0.0008) but not the truth gap (p 0.044,
    which vanishes without architecture FE).
  * **M1:** RepE-LM, an equal-length one-token persona task; the K1–K7 gates pass.
  * **M2:** the frozen GoT collapse is mostly fragility (0.28 vs noise 0.25); a real re-fit loss of 0.07–0.12
    remains; the SAE never beats PCA at equal error (PCA-CM, 2026-11-13: at k = 128 trained SAEs are never worse
    and better in 3/20 GoT cells).
  * **M3:** causal sites L1 slot, L8 final, L11 final.
  * **M4:** reconstruction carries 93–97 % at L1 and 62–68 % at L8/L11.
* **Improvement attempts and verdicts:**

| Attempt | Verdict | Type |
|---|---|---|
| Truth-aware (original) | INVALID (circular) | — |
| Truth-aware (corrected A/B) | NOT MET: worse than recon-only (0.189 vs 0.107) | unsuccessful |
| Method A, L6 | NOT MET (18/18 paired better, none > 2 sd) | inconclusive (directional) |
| E2E-inspired B/C, L6 | TopK behaviour MET; ReLU/JumpReLU guard FAIL; truth FAIL | partial / unsuccessful for truth |
| KL-weight ablation | INCONCLUSIVE (no trade-off to measure) | inconclusive |
| R1 honesty fine-tune (old RepE) | ReLU INCONCLUSIVE, others NOT SUPPORTED; now HISTORICAL | unsuccessful |
| R2-clamp SAE-in-the-loop (old RepE) | NOT SUPPORTED | unsuccessful |
| IMP-L Method A, L3/8/9 | 4/9 frozen-reader improvements; re-fit reader: none | qualified (readout restored, not information) |
| IMP-L output-KL | behaviour 5/9 creditable, 4/9 NOT CREDITABLE (L0); truth 0/9, 2 regressions | partial |
| E3 causal of the fine-tunes | NO IMPROVEMENT / NOT CREDITABLE | unsuccessful |
| RLM-RP | Method A 0/12; output-KL ≤ 0.007 AUROC in 3/12 | unsuccessful |

"Unsuccessful" means that a pre-registered criterion failed and the evidence pointed the other way or showed a null
with adequate checks. "Inconclusive" means the direction was consistent but below the threshold, or there was no
contrast to measure.

### Stage D: independent review and improvement phase (2026-11-10)
* **Diagnosis:**
  * the SAEs were data-starved (about 100k tokens vs res-jb's 300M);
  * every conclusion was exposed to "under-trained SAEs".
* **E2** (external res-jb anchor, VALIDATION): conclusions transfer.
* **R2** (data-scaled retraining): 100M fresh tokens, width 12,288, TopK+AuxK / BatchTopK+AuxK / ReLU, layers
  1/8/11 × 3 seeds, plus a width ablation and a random-decoder control. Modern JumpReLU abandoned.
* **R4** (KL+MSE fine-tune at fixed L0, TopK, L8/L11, MSE control): behaviour only.
* **R5** (rank-k DAS skyline, PCA, random floor): causal references.

### Stage E: M5 held-out test (2026-11-10)
* One run: 81 SAEs; 552 test rows (100 OWT documents = 12,700 tokens per layer; 452 GoT statements).
* Hash-guarded; refuses to overwrite.
* Verified complete in `EXPERIMENT_AUDIT.md` §1, with the disclosure items listed there.

**Protocols that must not be merged:**
* the old RepE task vs RepE-LM;
* frozen-logistic gaps (§9 of the summary) vs M2/M5 re-fit gaps;
* the 1000-step sweep vs 15k-step runs;
* historical layer-6 fine-tunes (unmatched L0) vs IMP-L.

---

## Part 3. Pipelines

### 3.1 Common front end (every experiment)
1. **Data:**
   * OWT: `scripts/openwebtext/01_prepare_dataset.py` (document-level split, ≤ 128 tokens);
   * GoT: `data/got/splits/*.csv` (grouped 70/15/15 by city or number pair);
   * RepE-LM: `data/repe/repe_lm/` (one-token persona swaps, train/validation only).
2. **Tokenisation:** the GPT-2 BPE tokenizer through TransformerLens.
3. **Model:** GPT-2 Small (124M parameters, 12 layers, d_model 768), fp32, **frozen** (parameter hashes checked).
4. **Activations:**
   * `blocks.L.hook_resid_pre`;
   * OWT: every non-BOS position, cached (`02_extract_activations.py`, `data/activations/gpt2_small/{train,validation,test}`);
   * GoT/RepE-LM: the final token (plus the persona slot for M3/M4).
5. **SAE:** encoder z = σ(W_enc (x − b_dec) + b_enc) with a sparsity mechanism σ; decoder x̂ = W_dec z + b_dec, with
   unit-norm decoder columns.
6. **Evaluation:** see 3.3.

### 3.2 SAE generations

| | Corrected baseline ("old") | R2 | R4 |
|---|---|---|---|
| Script | `scripts/openwebtext/03_training_sae.py` via `13_consistent_protocol_retrain.py` | `17_streaming_sae_training.py` + `src/sae/modern_sae.py` | `18_kl_finetune_fixed_l0.py` |
| Layers | 0–11 | 1, 8, 11 | 8, 11 |
| Width | 3,072 (4×) | 12,288 (16×); TopK-3072 ablation | 12,288 |
| Training data | 101,600 OWT tokens (cached train split) | 100M tokens from OWT parquet shard 78 (98 docs matching our sample removed) | 5M tokens, shard 79 (chunks ≥ 10,000) |
| Steps × batch | 15,000 × 256 (about 38 passes) | 24,414 × 4,096 | about 1,220 × 4,096 |
| Optimiser | Adam 3e-4, clip 10, decoder renormalised | Adam 2e-4, warm-up 500, linear decay over the last 20 %, clip 1, bf16 forward, inputs scaled to E‖x‖² = 768 | Adam 2e-5, clip 1 |
| Sparsity | ReLU: L1 with a multiplicative L0 controller; TopK k = 128; JumpReLU: sigmoid surrogate with controller | TopK k = 128 + AuxK (k_aux 512, α 1/32, dead after 5M tokens); BatchTopK mean k = 128 + AuxK; ReLU: decoder-norm-weighted L1 with a log-λ controller | TopK k = 128 (fixed) |
| Loss | MSE + coefficient × penalty | MSE (+ AuxK) (+ L1 for ReLU) | `kl`: NMSE + α·KL(p_orig ‖ p_spliced), α 1.14–1.51; `mse`: NMSE only |
| Seeds | 42, 1042, 2042 | same (random decoder: 42 only) | same, paired with the R2 base |
| Measured OWT L0 (test) | 128.0–133.4 | 127.6–131.7 | 128.0 |
| Trained parameters | W_enc, b_enc, W_dec, b_dec (+ thresholds) | same (+ BatchTopK inference threshold buffer) | same; starts from the R2 TopK checkpoint |
| Artifacts | `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/`, `docs/checkpoint_registry.csv` | `results/openwebtext/r2_scaled_saes/` | `results/openwebtext/r4_kl_finetune/` |

**Plain English:**
* The **frozen-LLM baseline** is GPT-2 itself, with no SAE: CE 3.302 nats/token on OWT test.
* The **original SAE** was a small, briefly trained SAE whose optimiser state was handled inconsistently across
  layers.
* The **corrected SAE** is the same design, trained consistently and checked for matched L0.
* The **R2 SAE** sees 1000× more distinct data, is 4× wider, and uses modern tricks (AuxK revives dead latents, with
  a learning-rate schedule and input normalisation).
* The **R4 SAE** is an R2 TopK that is briefly fine-tuned to also keep GPT-2's output distribution unchanged.

### 3.3 Evaluation pipelines
* **Reconstruction:** `src/evaluation/faithfulness_metrics.py:reconstruction_metrics`, on 12,700 OWT tokens.
* **Behaviour (splicing):** `scripts/shared/spliced_reconstruction_evaluation.py`.
  * Replace `hook_resid_pre` by x̂ at every non-BOS position.
  * ΔCE = CE_spliced − CE_orig.
  * KL = mean over tokens of KL(p_orig ‖ p_spliced).
  * Reference: a norm-matched random error.
* **Information:** `scripts/shared/m2_four_level_faithfulness.py`, `src/evaluation/information_levels.py`.
  * Logistic probe fitted on TRAIN original activations.
  * Frozen gap: apply that probe to x̂.
  * Re-fit gap: train a new probe on x̂ of TRAIN rows, then evaluate it on x̂ of the eval rows.
  * PCA reference: project onto k components, with k chosen so that PCA's OWT validation NMSE equals the SAE's.
  * Noise reference: add a random error of the same norm.
* **Causal (M4):** `scripts/repe/` M4 functions.
  * Patch Δ = x_H − x_U at the site into the U run.
  * T = KL(p_H‖p_U) − KL(p_H‖p_patched).
  * Compare T_full (raw Δ), T_rec (Δ through the SAE reconstruction) and T_err.
  * Controls: synonym, matched-norm, shuffle and random features.
* **R5:** `scripts/repe/27_causal_subspace_skyline.py`. Patch P_k Δ with P_k from DAS (learned), PCA or random.
* **One harness for all SAEs:** `scripts/shared/sae_ladder_evaluation.py`. The comparison rules are in
  `scripts/shared/r2_comparisons.py`, and M5 is `scripts/shared/m5_confirmatory_test.py`.

---

## Part 4. Metric glossary

| Metric | What it measures | Better | Meaningful comparison | Common misinterpretation |
|---|---|---|---|---|
| MSE | mean ‖x − x̂‖² per token | ↓, 0 ideal | same layer, same data | comparing across layers: the residual norm grows (L1 0.09 vs L11 4.7 at similar quality) |
| NMSE | Σ‖x − x̂‖² / Σ‖x − x̄‖² = 1 − EV | ↓, 0 ideal; normalised | within a layer at matched L0 | > 1 on GoT final tokens, because the within-set variance is tiny; use relative error there |
| Explained variance | 1 − NMSE | ↑, 1 ideal | within a layer | "EV 0.9 = faithful": the R2 TopK L8 EV is 0.906 and ΔCE is still 0.062 nats |
| Cosine | mean cos(x, x̂) | ↑, 1 ideal | within a layer | saturates near 1 (0.95–0.997) and hides behaviourally relevant error |
| Reconstruction loss | the training objective (MSE or NMSE, plus penalties) | ↓ | the same objective only | it includes the sparsity or KL terms, so it is not comparable across architectures |
| L0 | mean number of active latents per token | match the target (128) | ±10 % baseline, ±5 % fine-tune; report it per dataset | the L0 matched on OWT ≠ the L0 on GoT/RepE prompts (ReLU 77–112 on GoT) |
| Target vs actual L0 | the controller's error | small | flags unmatched runs | TopK is exact by construction; ReLU overshoots by 2–6 % |
| Dead fraction | share of latents that never fire on the eval set | ↓ | same eval-set size | the old JumpReLU at L1 is 68 % dead and still matches L0; R2 with AuxK ≈ 0 % |
| Width | number of latents m | design choice | at matched L0 and recipe (A2) | wider means a better reconstruction ceiling, not better truth preservation (A2 GoT regression at L8) |
| Feature frequency | the per-latent firing rate | — | — | not reported beyond the dead fraction (NOT VERIFIED) |
| Accuracy | correct classification at a threshold | ↑ | same threshold and split | a frozen probe on x̂ has shifted scores, so accuracy collapses even when the ranking survives |
| AUROC | ranking quality; 0.5 = chance | ↑ | same eval rows; informative layers only (≥ 0.70) | uninformative layers (L0–3 on GoT) give tiny gaps that mean nothing |
| AUROC gap | AUROC(orig) − AUROC(x̂) | ↓, 0 ideal | against a noise and a PCA reference | reading a frozen gap as "information erased": M2 shows it is mostly fragility |
| Frozen vs re-fit probe | readout preserved vs information present | both ↓ | report both | the re-fit gap measures decodability; the frozen gap measures readout preservation |
| Independent probe | a probe not used as a training teacher (GoT group B) | — | required for any truth-aware method | the teacher probe used as evaluator is circular (D12) |
| Unseen statements | grouped splits (city, statement) | — | GoT city groups; RepE-LM held-out wording T5/T6 | row splits leak entities (the old Gemma table) |
| KL divergence (measured) | KL(p_orig ‖ p_spliced) per token | ↓, 0 ideal | same tokens and splice positions | confusing it with the KL **loss coefficient α** (1.14–1.51 in R4) |
| Cross-entropy | −log p(next token) | — | original CE 3.302 (OWT test) | — |
| ΔCE | CE_spliced − CE_orig | ↓, 0 ideal | same tokens; against the noise ratio | a small ΔCE can hide a large distribution change; report KL too |
| SAE/noise ratio | ΔCE_SAE / ΔCE of a norm-matched random error | ↓; < 1 = more benign than noise | within a layer | > 1 means the SAE's error is structured and harmful |
| Spliced evaluation | behaviour with x̂ in the forward pass | — | splicing ≠ steering | — |
| Steering effect | output change from adding a direction | context-dependent | against random-direction controls | large ±α both push to TRUE in Gemma (a generic bias) |
| Direction cosine | cos between the original and reconstructed contrast directions | ↑ | per-pair cos(Δ, Δ̂), against a random decoder | it improves when trained on (R1), so it is not evidence then |
| Causal preservation T_rec/T_full | share of the patch effect carried by x̂ | ↑; 1 = full | the same site; against controls and a DAS skyline | DAS > 1 is possible (overshoot); T_rec + T_err ≠ T_full (interaction) |
| DAS / PCA / random / SAE features | the share carried by rank-k subspaces | reference | at the same k and site | DAS is a supervised upper reference, not a competitor method |

**Observed values, in one line each:**
* EV ranges 0.76–0.98 (falling with depth).
* ΔCE ranges 0.002 (R2 L1) to 0.34 nats (old ReLU L11).
* GoT frozen gaps are about 0.2–0.3, and re-fit gaps 0.03–0.11.
* Persona re-fit gaps are 0.07–0.21.
* T_rec/T_full is 0.62–0.99 for trained SAEs and 0.02–0.66 for the random decoder.

---

## Part 5. Results by dataset

### 5A. OpenWebText: reconstruction and behaviour
Full tables: `RESULTS_VERIFICATION_TABLE.md` §1–§2.

* **Frozen GPT-2 reference:** CE 3.302 nats/token on OWT test.
* **Original pre-review SAEs (L2/6/10, test EV):** 0.936/0.925/0.929 (L2), 0.831/0.857/0.866 (L6) and
  0.773/0.822/0.825 (L10) for ReLU/TopK/JumpReLU.
* **Corrected baseline (TEST, 3 seeds):** EV falls with depth, from 0.93–0.95 at L0–2 to 0.757/0.815/0.815 at L11.
  * The ΔCE splice rises from 0.01–0.04 to 0.20–0.36 nats.
  * M5 values at L1/L8/L11: NMSE 0.063–0.243; ΔCE 0.009–0.342.
* **R2 (TEST):**
  * NMSE −31 to −74 % and ΔCE −35 to −79 %, on every architecture, layer and seed.
  * Best: BatchTopK, with L8 NMSE 0.090 and ΔCE 0.054.
* **R4 (TEST):** ΔCE −17 % (L8) and −16 % (L11) vs the MSE control; NMSE +2 %.
* **E2 res-jb (TEST rows, descriptive, unmatched L0 15/66/60):**
  * L8: NMSE 0.144, ΔCE 0.168.
  * It sits on the old SAEs' error–damage relation; R2 dominates it.
* **Why high EV still changes predictions.**
  * EV is variance-weighted. The unexplained 9–11 % at L8/L11 lies partly along directions downstream layers read
    strongly.
  * The SAE/noise ratio is > 1 for nearly all SAEs: 1.20–2.37 at L8/L11.
  * So the SAE error is *more* harmful than an isotropic error of the same norm. It is structured ("dark matter"),
    not noise.

### 5B. Geometry of Truth: truth preservation
* **Data:** cities (1,496 statements, 748 true / 748 false) + neg_cities, grouped by city; test n = 452.
  The final-token residual is used.
* **Baselines:**
  * the original-activation logistic probe: test AUROC 0.839 (L8) and 0.792 (L11);
  * PCA at matched OWT NMSE;
  * norm-matched noise;
  * a random-decoder SAE.
* **Historical (layer 6, TEST, unmatched L0):**
  * MLP gap base 0.327/0.154/0.323;
  * reconstruction-only fine-tune 0.161/0.110/0.146;
  * truth-aware fine-tune 0.225/0.184/0.218. Its logistic numbers are circular.
* **Corrected all-layer frozen probe (TEST):** gap 0.19–0.32 at informative layers. M2 reinterprets this as mostly
  fragility.
* **M2 (VALIDATION):** re-fit gap 0.07–0.12; the SAE is never better than PCA.
* **M5 (TEST):**
  * re-fit gap 0.028–0.065 (L8) and 0.053–0.107 (L11);
  * **0/34 cells improved**; A2 regressed at L8;
  * the random frozen decoder has SAE − PCA −0.003 / −0.013, against +0.008 to +0.081 for every trained SAE.
* **Did better reconstruction produce better truth preservation? No.**
  * ReLU is the worst reconstructer, yet it has the best GoT re-fit gap. This is confounded by GoT-prompt L0.
  * M0 shows no robust within-layer link.
  * **GoT preservation remains worse than PCA for every trained SAE.**

### 5C. RepE / RepE-LM: honesty persona, representation and causal
* **Historical RepE** ("Pretend you're an honest/untruthful person…"; 417/90/90 statements):
  * it is length-confounded: " untruthful" is 3 tokens longer, and length alone gives AUROC 0.80;
  * all its results are HISTORICAL (D23).
  * The 6c causal test (TEST): positive control 0.234; raw direction 0.0048 (p = 0.010 vs 200 random); SAE
    directions not distinguishable from controls.
* **RepE-LM:**
  * equal-length one-token swaps (honest↔untruthful synonyms within a fixed template); 8 templates; train/validation
    only;
  * the gates pass, with K5 semantic transfer narrowly at 0.70.
* **Representation (VALIDATION):**
  * the persona re-fit gap is 0.08–0.22 (old), 2–3× the noise reference;
  * R2 TopK lowers it by 27–31 % at L8/L11, but SAE − PCA does not improve;
  * the per-pair cos(Δ, Δ̂) is 0.37–0.56 (old) and 0.49–0.62 (R2).
* **Causal (VALIDATION):**
  * T_rec/T_full: old 0.93–0.97 (L1), 0.62–0.67 (L8), 0.62–0.68 (L11);
  * R2 TopK 0.988 / 0.726 / 0.692; BatchTopK 0.994 / 0.754 / 0.701;
  * R4 KL at L11: 0.671 vs 0.694 for the MSE control (REGRESSION).
* **R5 skyline (VALIDATION):**
  * at k = 4, DAS carries 1.02 / 0.91 / 0.62, PCA 1.00 / 0.52 / 0.59, SAE features 0.08–0.16 and random 0.004–0.008;
  * **at L8, a rank-4 subspace carries more than the whole old SAE reconstruction.**
* **M5:** not covered, because RepE-LM has no test split. **All persona/causal results are VALIDATION.**

---

## Part 6. Before vs after

See `RESULTS_VERIFICATION_TABLE.md` §0–§3 for the full tables with CIs and per-seed values.

**Key structural fact:** original → corrected was done at L2/6/10, and corrected → improved at L1/8/11. They are two
separate controlled comparisons, not one chain.

| Layer | Architecture | Original SAE | Corrected baseline | Improved SAE (R2) | Metric | Change (corrected → R2) | Better? | Evidence |
|---|---|---|---|---|---|---|---|---|
| 2 | JumpReLU | 0.9293 | 0.9355 | — | OWT EV | (orig → corr +0.006) | consistency fix | TEST |
| 6 | TopK | 0.8573 | 0.8572 | — | OWT EV | (−0.0001) | no change | TEST |
| 10 | JumpReLU | 0.8253 | 0.8227 | — | OWT EV | (−0.0026) | no change | TEST |
| 8 | TopK | NOT AVAILABLE (1000-step sweep val EV 0.710, not comparable) | NMSE 0.1598 | 0.0941 | OWT NMSE | −0.066 (−41.1 %) | better, 9/9 | TEST |
| 8 | TopK | — | ΔCE 0.1330 | 0.0620 | ΔCE (nats) | −0.071 (−53.4 %) | better, 9/9 | TEST |
| 8 | ReLU | — | 0.1951 | 0.1320 | OWT NMSE | −32.4 % | better | TEST |
| 8 | JumpReLU → BatchTopK | — | 0.1560 | 0.0897 | OWT NMSE | −42.5 % | better (different architecture) | TEST |
| 8 | TopK | — | 0.061 | 0.056 | GoT re-fit gap | −0.005 [−0.053, +0.043] | no clear change | TEST |
| 11 | TopK | — | 0.1848 | 0.1112 | OWT NMSE | −39.8 % | better | TEST |
| 11 | TopK | — | 0.2041 | 0.0962 | ΔCE | −52.9 % | better | TEST |
| 11 | TopK | — | 0.101 | 0.078 | GoT re-fit gap | −0.023 [−0.060, +0.013] | no clear change | TEST |
| 8 | TopK | — | 0.625 | 0.726 | T_rec/T_full | +0.101 [+0.019, +0.182] | better | VALIDATION |
| 11 | TopK | — | 0.620 | 0.692 | T_rec/T_full | +0.073 [+0.064, +0.082] | better | VALIDATION |
| 8 | TopK, R4 KL vs MSE | — | (R2 MSE control) 0.0623 | 0.0514 | ΔCE | −17.4 %, 3/3 seeds | better | TEST |
| 11 | TopK, R4 KL vs MSE | — | 0.694 | 0.671 | T_rec/T_full | −0.024 ± 0.002 | **worse** | VALIDATION |

**Matched-ness:**

| Comparison | Matched L0 | Width | Dataset / layer / protocol |
|---|---|---|---|
| C1 / C2 | yes | **no** (3,072 → 12,288) | yes |
| C3 | yes | no; **also different architecture** | yes |
| A1 | yes | yes (3,072) | yes; data and recipe both change |
| A2 | yes | the variable | yes; same recipe |
| K1 | yes | yes | yes; same tokens, same base, paired seeds |

---

## Part 7. What we did to improve the project

| # | Method | Problem → hypothesis | Implementation / objective | Result | Verdict |
|---|---|---|---|---|---|
| 1 | Optimiser / reproducibility | mixed Adam handling → a biased cross-layer comparison | uninterrupted 15k steps, 3 seeds (`13_*`) | ΔEV ≤ 0.006, no rank changes | succeeded as a fix; no performance gain |
| 2 | Sparsity matching | unequal L0 confounds architecture | ±10 % / ±5 % rules; P1 controller (anneal + freeze) | 107/108 matched; L2 JumpReLU EV +0.020 and dead 65 → 21 % (P1, seed 42) | succeeded on OWT; task-L0 unmatched |
| 3 | More data and a better recipe (R2) | data-starved SAEs | 100M tokens; AuxK; schedule; scaling | TEST NMSE −22 to −35 % at equal width (A1) | succeeded; data vs recipe not separated |
| 4 | TopK + AuxK | dead latents (old TopK L1 44 % dead) | AuxK, k_aux 512, α 1/32 | dead ≈ 0–0.3 % (TEST) | succeeded |
| 5 | JumpReLU training changes | surrogate, not the published method; dead latents | P1 quantile threshold init (old); modern STE (R2) | P1 helped L2/L6; modern STE **abandoned** (L0 1660 or oscillating 19–506) | partly; the R2 JumpReLU failed |
| 6 | ReLU sparsity regularisation | L0 overshoot, shrinkage | log-λ controller; decoder-norm-weighted L1 | R2 ReLU L0 127.6–131.2; still worst NMSE at depth | L0 control succeeded; ReLU remains weakest |
| 7 | Width | capacity | 3,072 vs 12,288 (A2) | TEST NMSE −20 to −52 %; GoT L8 REGRESSION | helps reconstruction and behaviour only |
| 8 | Reconstruction-only fine-tuning (Method A) | task-specific reconstruction might restore truth | GoT-A reconstruction + OWT replay, 500 steps | frozen MLP gap falls; re-fit unchanged; E3 no causal change | inconclusive → qualified null |
| 9 | Truth-aware fine-tuning | directly preserve the probe readout | + probe-logit MSE (teacher on A, evaluated on B) | worse than recon-only (0.189 vs 0.107) | failed; abandoned (D16) |
| 10 | KL behaviour-aware fine-tuning | the splice damage | IMP-L/E2E output-KL (unconstrained L0); R4 at fixed L0 | IMP-L: L0 drift → not creditable for ReLU/JumpReLU; R4: ΔCE −16 to −17 % (TEST); causal L11 −0.024 (VAL) | succeeded for behaviour only; harms causal |
| 11 | Public pretrained SAEs (E2) | do the conclusions hold for strong SAEs? | res-jb at L1/8/10/11 through the same harness | H1–H4 hold (except GoT H2 at L8) | conclusions transfer (descriptive) |
| 12 | Causal skyline and controls (R5, random decoder) | is 62–68 % low or high? | DAS rank-k, PCA, 20 random subspaces; frozen random decoder | rank-4 DAS 0.91 at L8 > whole SAE 0.62–0.67; random decoder 0.02 | succeeded as references |

---

## Part 8. M5 final verification

**Status: M5 COMPLETE** (full checklist in `EXPERIMENT_AUDIT.md` §1).
* 11/11 code and config hashes match.
* 81/81 expected rows; 0 duplicates.
* 136/136 decision classes reproduced independently.
* L0 matched (127.6–133.4).
* Clean exit.

**Disclosures:**
* the README's L0 maximum is wrong (133.2 should be 133.39);
* the frozen README text cannot be verified against its hash;
* the pre-registration was committed in the same commit as the results;
* the test rows were previously scored for the old reference SAEs;
* there is no multiplicity correction.

**What M5 establishes:**
* R2 and R4 reconstruction and behaviour gains generalise to held-out OWT documents;
* GoT truth information shows no confirmed change;
* persona and causal results are **not** tested.

---

## Part 9. Findings

1. **Best reconstruction:**
   * Old SAEs, matched L0: JumpReLU at L3–9 and ReLU at L1 (TEST).
   * R2: BatchTopK at L1/8/11 (TEST, NMSE 0.018 / 0.090 / 0.107).
   * BatchTopK is not JumpReLU, so it does not settle the threshold-architecture question.
2. **Best truth preservation:**
   * On the GoT re-fit gap, ReLU (TEST L8 0.028–0.034; L11 0.053–0.077). This is confounded by GoT-prompt L0, and
     no trained SAE beats error-matched PCA (at k = 128, R2 ReLU is better at L8 and L11; VALIDATION).
   * On M2 (VALIDATION), ReLU loses least and JumpReLU most.
3. **Best behaviour:**
   * R2 BatchTopK (TEST ΔCE 0.002 / 0.054 / 0.093), then R4 KL TopK at L8/L11 (0.051 / 0.081).
   * Among old SAEs: JumpReLU at L1/L8, TopK at L11.
4. **Best honesty/persona representation (VALIDATION):**
   * R2 TopK/BatchTopK have the lowest persona re-fit gaps (0.067–0.126).
   * ReLU has the highest pair-direction cosine (0.52–0.62) but the largest gaps. It also shrinks the contrast:
     RLM-RP norm ratio ‖Δ̂‖/‖Δ‖ = 0.65–0.71.
5. **Best causal:** R2 BatchTopK at L8 (0.754, VALIDATION). Every SAE is far below DAS at matched rank.
6. **Width:** helped reconstruction and behaviour (TEST A2 −20 to −58 %), but not truth (L8 regression).
7. **Longer training:** **NOT ISOLATED.** Data plus recipe helped (A1), but the token count alone was never
   evaluated.
8. **Reconstruction-only fine-tuning:** frozen readout only; no re-fit or causal gain (VALIDATION).
9. **Truth-aware fine-tuning:** failed (circular in its first form; worse than recon-only when corrected).
10. **KL fine-tuning:** behaviour only (TEST −16 to −17 % ΔCE), at a causal cost at L11 (VALIDATION) and an NMSE
    cost of +2 %.
11. **Lower reconstruction error ↔ better truth preservation?**
    * **No robust association within a layer** (M0 fragile; M5: 0/34; ReLU the counter-example; the random decoder
      matches PCA).
    * Across layers both move together with depth, but that is depth, not SAE quality.
12. **Layer variation:**
    * Every damage metric grows with depth (ΔCE 0.002–0.022 at L1, against 0.05–0.34 at L8/L11).
    * Causal preservation is near-complete at L1 and only partial at L8/L11.
    * GoT is uninformative before about L4.
13. **Generalisation:** OWT/GoT validation conclusions replicate on TEST. RepE-LM and Gemma were not tested.
14. **Strongest finding:**
    * A confirmed dissociation: an SAE upgrade that halves behavioural damage (L8 TopK 0.133 → 0.062) leaves truth
      information unchanged (0/34) and still below PCA.
    * In addition, a behaviour-targeted fine-tune reduces causal preservation.
15. **Strongest negative finding:**
    * No method improved truth preservation as seen by an independent, re-fit reader.
    * A random frozen decoder is as good as trained SAEs on the GoT SAE − PCA metric. That metric therefore cannot
      separate trained from random SAEs at these layers.
16. **Unresolved:**
    * held-out confirmation of the persona/causal results (needs a RepE-LM test split);
    * task-level L0 matching (E1);
    * tokens vs recipe;
    * whether Gemma Scope SAEs show the same dissociation (R3);
    * an L0 frontier beyond 128;
    * a better truth task than GoT cities (low variance, weak discrimination).

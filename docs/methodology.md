# Methodology: GPT-2 Small SAE faithfulness

This document describes how the GPT-2 experiments on `sae-research` are run and evaluated, and which methodological
corrections were made along the way (§7). File paths are relative to the repository root.

## 1. Frozen GPT-2 activation extraction

* **Model and hook:**
  * GPT-2 Small through TransformerLens, fp32, parameters frozen (a SHA-256 of all parameters is recorded before and
    after every training run).
  * Hook `blocks.L.hook_resid_pre`. **BOS is excluded** from SAE training and from splicing.
* **OWT (old SAEs):**
  * `scripts/openwebtext/01_prepare_dataset.py` (document-level 800/100/100 split, ≤ 128 tokens);
  * `02_extract_activations.py` caches all 12 layers to `data/activations/gpt2_small/{train,validation,test}/`
    (gitignored).
* **OWT (R2):** `scripts/openwebtext/17_streaming_sae_training.py`.
  * Activations are generated on the fly from OpenWebText parquet shard 78, so they are never stored.
  * Documents matching any of our 1,000 sampled documents are removed by exact and normalised-prefix hash.
  * Chunks of 127 tokens + BOS go into a shuffled fp16 GPU buffer.
* **GoT and RepE-LM:**
  * the final-token residual at every layer (`scripts/geometry_of_truth/01_*`, `scripts/repe/21_*`);
  * for the causal work, also the persona-slot position.

## 2. SAE training

| | Old SAEs | R2 SAEs |
|---|---|---|
| Script | `03_training_sae.py` via `13_consistent_protocol_retrain.py` | `17_streaming_sae_training.py` + `src/sae/modern_sae.py` |
| Width | 3,072 | 12,288 (TopK-3,072 ablation) |
| Data | 101,600 unique tokens, 15k steps × 256 (≈ 38 passes) | 100M tokens, 24,414 steps × 4,096 (≈ 1 pass) |
| Optimiser | Adam 3e-4, clip 10 | Adam 2e-4, warm-up 500, linear decay over the last 20 %, clip 1, bf16 forward |
| Init / scaling | Kaiming encoder; random unit-norm decoder | encoder = decoderᵀ; b_dec = data mean; inputs scaled so E‖x‖² = 768 (folded into saved weights) |
| Sparsity | ReLU: L1 + multiplicative L0 controller; TopK k = 128; JumpReLU: sigmoid surrogate + controller | TopK k = 128 + AuxK (k_aux 512, α 1/32); BatchTopK mean k = 128 + AuxK; ReLU: decoder-norm-weighted L1 + log-λ controller |
| Seeds | 42, 1042, 2042 | same |

**Fine-tunes:**
* **Method A** (GoT-A reconstruction + OWT replay);
* **output-KL** (Methods B/C, IMP-L);
* **R4** (NMSE + α·KL at fixed L0, `18_kl_finetune_fixed_l0.py`).

Each starts from a base checkpoint and has its own pre-registration in its results folder.

## 3. Reconstruction metrics (`src/evaluation/faithfulness_metrics.py`)

| Metric | Definition | Better |
|---|---|---|
| NMSE | Σ‖x − x̂‖² / Σ‖x − x̄‖² = 1 − EV | lower |
| EV | 1 − NMSE | higher |
| Cosine | mean cos(x, x̂) | higher (saturates near 1) |
| Dead fraction | share of latents never active on the evaluation set | lower |

* Computed on the 12,700-token OWT validation (development) or test (M5) split.
* MSE is never averaged across layers, because the residual norm grows with depth.
* MSE improvement (%) = 100 × (baseline − improved) / baseline.

## 4. Sparsity measurement

* **L0** is the mean number of active latents per token.
* **Matching rules:** OWT L0 within ±10 % of the target (128) for baselines, and within ±5 % of the base for
  fine-tunes. Unmatched runs are flagged and excluded from rankings, not deleted
  (`results/openwebtext/layer_comparison/sparsity_matched_cells.csv`).
* L0 is **reported separately on OWT, GoT and RepE-LM prompts.** Threshold architectures fire different numbers of
  latents on task prompts (for example GoT L0 63–111 vs TopK's 128), which is a known confound.

## 5. Information preservation: frozen and re-fit probes

Implemented in `scripts/shared/m2_four_level_faithfulness.py` and `src/evaluation/information_levels.py`.

* **Frozen probe:**
  * a logistic (or mass-mean) probe fitted on ORIGINAL activations of TRAIN rows only, with C chosen by grouped CV;
  * applied unchanged to x̂;
  * frozen gap = AUROC(x) − AUROC(x̂). It measures whether the model's original readout survives.
* **Re-fit probe:** a new probe trained on x̂ of TRAIN rows and evaluated on x̂ of evaluation rows. It measures whether
  the information is still present. MLP, kNN and sparse-z readers are also reported.
* **References:**
  * norm-matched isotropic noise (is the SAE worse than a random error of the same size?);
  * PCA reconstruction at the SAE's own OWT NMSE (does the SAE beat a linear projection of equal error?);
  * the error-term probe (is the lost signal in e = x − x̂?).
* **Independence rule:** an evaluation probe is never a training teacher. GoT-train groups A/B:
  * A trains any teacher;
  * B fits the evaluation probes (`data/got/splits/train_AB_groups_truth_aware_pilot.csv`).
* **Uncertainty:** grouped bootstrap by city or statement.

## 6. Behaviour, representation and causal evaluation

* **Splicing** (`scripts/shared/spliced_reconstruction_evaluation.py`):
  * x̂ replaces `hook_resid_pre` at every non-BOS position;
  * **ΔCE** = CE_spliced − CE_orig (nats per token);
  * **KL** = mean KL(p_orig ‖ p_spliced);
  * a norm-matched random error is the reference (SAE/noise ratio; < 1 means more benign than noise);
  * an identity splice must give exactly 0.
* **Representation (RepE-LM)** (`scripts/repe/26_*`), for each honest/untruthful pair (H, U), with Δ = x_H − x_U and Δ̂
  from the reconstructions:
  * per-pair cos(Δ, Δ̂);
  * norm retention ‖Δ̂‖/‖Δ‖ (1 is ideal);
  * held-out-wording gap.
* **Causal decomposition (M4)** (`scripts/repe/24_*`):
  * patch the persona difference into the U run at the site found by M3 (L1 persona slot; L8 and L11 final token);
  * T(v) = KL(p_H‖p_U) − KL(p_H‖p_{U+v});
  * report T_full (raw Δ), T_rec (Δ through the SAE reconstruction) and T_err;
  * preservation = T_rec/T_full;
  * controls: synonym swaps, matched-norm and shuffled vectors, matched random feature sets.
* **Skyline (R5)** (`scripts/repe/27_*`): learned rank-k subspaces (DAS-style), PCA and random subspaces at the same
  sites.
* **One harness** for every SAE family: `scripts/shared/sae_ladder_evaluation.py`, which imports the code above
  unchanged.

## 7. Methodological corrections (chronological)

Detailed record: `docs/archive/2026-11-04_methodology_review_corrections_report.md`,
`docs/2026-11-04_code_audit.md`, `docs/2026-11-04_methodology_changes.md`, `docs/RESEARCH_DECISIONS.md`.

| # | Problem found | Correction | Effect on conclusions |
|---|---|---|---|
| 1 | **Optimizer-state inconsistency:** L6 resumed with Adam state; L2/L10 warm-started with Adam reset | All runs retrained with one uninterrupted 15k-step protocol (`13_*`) | EV changed ≤ 0.006; two "wins" became ties |
| 2 | **Unmatched sparsity:** architectures compared at different L0; a summary averaged unmatched runs and raw MSE across layers | ±10 % / ±5 % rules; matched-cell tables; `architecture_summary.csv` INVALID | Only 15/36 sweep cells comparable; the TopK lead at 1000 steps reverses by 15k |
| 3 | **"SAE steering" without the SAE in the forward pass** | Pre-registered test with 200 controls; later SAE-in-the-loop patches (M4) | SAE directions not distinguishable from controls (layer 6, old task) |
| 4 | **Re-trained probe read as faithfulness** (GoT L10) | Frozen original probe; later frozen vs re-fit kept separate (M2) | AUROC 0.851 (re-trained) vs 0.639 (frozen) |
| 5 | **Probe circularity:** the truth-aware teacher probe was also the evaluator | A/B groups; independent probes (D12, D16) | Truth-aware fine-tuning reclassified: worse than recon-only |
| 6 | **Length confound in the old RepE task** | Equal-length one-token RepE-LM task with gates K1–K7 (D23) | All old RepE results HISTORICAL |
| 7 | **Pooled cross-layer correlations** | Within-layer regressions with fixed effects (M0) | "Reconstruction tracks honesty" (ρ 0.86) was a pooling artefact |
| 8 | **Frozen-probe collapse read as information loss** | Noise and PCA references (M2) | GoT collapse mostly fragility; real re-fit loss 0.07–0.12 |
| 9 | **L0 drift under KL objectives** (ReLU/JumpReLU +5–11 %) | Guards; R4 uses TopK (L0 fixed by construction) | IMP-L output-KL 4/9 cells not creditable |
| 10 | **Data-starved SAEs** (0.1M tokens) | R2 data-scaled retraining (100M tokens) | Reconstruction and behaviour gains confirmed on TEST |
| 11 | **Test-split exposure and late pre-registration commits** | Test rows labelled "previously exposed"; protocols frozen in a separate committed file from now on (D31) | M5 labelled accordingly |
| 12 | **Train/validation near-duplicate** (RepE statement 326 ≈ 589) | Excluded from R1/R2/M1 | — |

## 8. Dataset splits and leakage prevention

* **OWT:** document-level split. The R2/R4 shards exclude our 1,000 documents by hash.
* **GoT:** grouped by city or number pair; the M5 loader asserts that no group crosses splits.
* **RepE-LM:** train/validation by statement. No test split exists yet.
* **Test splits** are used only in a pre-registered final pass (M5); never for training, tuning or selection.

## 9. Statistical comparisons

* **Unpaired family comparisons** (old vs R2) use the R2 rule (`scripts/shared/r2_comparisons.py`):
  * IMPROVEMENT needs **all 9 seed pairs better *and* a Welch 95 % CI excluding 0**; REGRESSION is the mirror image;
  * otherwise NO CLEAR CHANGE.
* **Paired comparisons** (same base, e.g. R4 KL vs MSE, IMP-L): better on all 3 seeds **and** |mean paired
  difference| > 2 sd, with guards (`scripts/openwebtext/19_r4_analysis.py`, IMP-L `criteria.json`). A failed guard
  makes an improving cell NOT CREDITABLE.
* **Within-SAE uncertainty:** grouped bootstrap over statements or cities.
* **Multiplicity:** BH within M2/E2/E3. None across the 136 M5 rows; this is stated as a limitation.

## 10. Reproducibility controls

* Every experiment folder has a `README.md` and a `preregistration.json` with SHA-256 of its scripts, splits and
  checkpoints. Scripts refuse to run on a hash mismatch, and refuse to overwrite outputs.
* Reproduction checks:
  * an identity/reproduction run must be bit-identical, or match within a stated tolerance;
  * the M5 smoke test reproduced the validation ladder exactly.
* GPT-2 parameter hashes are recorded before and after training.
* Registries:
  * `docs/experiment_manifest.csv` (via `scripts/shared/write_manifest.py`);
  * `docs/checkpoint_registry.csv` (via `scripts/shared/build_checkpoint_registry.py`);
  * `docs/results_registry.md`.
* Independent re-check of M5: `python scripts/shared/m5_independent_verification.py` (read-only).
* Tests: `python -m pytest -q src/tests`.

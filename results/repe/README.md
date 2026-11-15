# RepE honesty results (GPT-2 Small layer 6, final token)

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

* `original_baseline.csv`, `direction_fidelity.csv`, `cross_wording/` — **VALID** as validation-split numbers (layer 6 was chosen on validation, so they are optimistically biased). The quoted "cross-wording JumpReLU ≈ 0.8010" is not in any file; the saved value is 0.7394.
* `representation_preservation.csv` — **VALID** (validation only). `representation_preservation_val_test.csv` — **AUTHORITATIVE**: validation + first test-split evaluation + cross-wording, all 9 variants.
* SAE L0 on RepE prompts is **not matched** across architectures (JumpReLU 187, ReLU 170, TopK 128).
* `steering/` — see `steering/README.md`.

**All 12 layers, 108 uninterrupted OWT SAEs (added 2026-11-07; scripts `scripts/shared/all_layer_frozen_probe_{evaluation,analysis}.py`):**
* `all_layer_frozen_probe_evaluation/` — **AUTHORITATIVE** for cross-layer honesty preservation.
  * Frozen probes: the mass-mean persona direction (as in script 03; validation AUROCs identical to `original_baseline.csv`) and a logistic probe whose C was chosen on original validation rows.
  * Splits scored: validation, test and cross-wording. Test rows were never used for selection.
  * Files: `per_run.csv`, `summary.csv`, `matched_architecture_comparison.csv`, `reconstruction_vs_truth_correlations.csv`, `excluded_unmatched_runs.csv`, `protocol.json`.
  * `decodability_diagnostic_validation.csv` is a **DIAGNOSTIC only**.
* Caveat: SAE L0 on RepE prompts is 143–278 for ReLU/JumpReLU (TopK fixed at 128). Matching is defined on OWT, so RepE comparisons are not at equal L0 on these prompts.
* Figures: `figures/all_layers_*.png`.

**Spliced-in causal evaluation (Phase 0, 2026-11-07):** `spliced_evaluation/` — **AUTHORITATIVE (validation only)**. GPT-2 predictions with `hook_resid_pre` replaced by each of the 108 uninterrupted SAE reconstructions, with identity, fp16, mean-ablation and norm-matched random-error controls. See `results/shared/README.md`. This is separate from the steering results.

**R1 honesty-preservation fine-tuning (2026-11-07):** `r1_honesty_preservation_layer6/` — **PILOT, validation only**.
Verdict ReLU INCONCLUSIVE, TopK/JumpReLU NOT SUPPORTED; see its README §11. Checkpoints in
`sae_finetuning/r1_honesty_preservation_layer6/` (gitignored; SHA-256 in `docs/checkpoint_registry.csv`).
**R2 SAE-in-the-loop intervention (2026-11-08):** `r2_sae_in_the_loop_steering_layer6/` — **validation stage only (descriptive)**.
Verdict NOT SUPPORTED for all architectures and k; test stage deferred. See its README §13 and the figure
`figures/r2_sae_in_the_loop_validation.png`.

**RepE-LM (corrected, equal-length task) results, current evidence (2026-11-10):**
* `repe_lm_gates/` (M1), `causal_site_sweep/` (M3), `sae_patch_decomposition/` (M4).
* `repe_lm_representation_preservation/` (**RLM-RP**, VALIDATION): 216 SAEs, persona preservation with controls.
  This is the current source for every RepE representation claim.
* `e3_finetuned_causal_preservation/` (**E3**, VALIDATION): causal preservation of the IMP-L layer-8 fine-tunes.
* The length-confounded results above (`representation_preservation*.csv`, `all_layer_frozen_probe_evaluation/`,
  `steering/`, R1, R2) are HISTORICAL (D23) and are not evidence.

* `r5_causal_subspace_skyline/` (**R5**, VALIDATION, descriptive): rank-k DAS/PCA/random subspaces vs SAE feature
  patches at the M4 sites. A rank-4 subspace carries 0.91 of the L8 persona effect; SAE top-4 features 0.10–0.16.

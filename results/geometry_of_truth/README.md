# Geometry of Truth results (GPT-2 Small)

Consolidated on 2026-11-06: the former `results/got/` subdirectories now live here (moved with `git mv`, checksums verified; see `docs/migration_manifest.csv`). JSON/TXT provenance files inside them (e.g. `final_test_protocol/pretest_protocol.txt`, `sae_finetuning/*/run_info.json`, `final_test_evaluation/evaluation_manifest.json`) still quote the old paths `results/got/…` and `scripts/Geometry of Truth/…`; they are historical records and were not edited.

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

**Layer 6, in-house OWT-trained SAEs, grouped splits (scripts `scripts/geometry_of_truth/01–09`):**
* `original_truth_probes/`, `sae_truth_preservation/`, `sae_diagnostic_truth_probes/`, `independent_truth_probe/` — **VALID** (validation-split stages).
* `final_test_protocol/`, `final_test_evaluation/` — **AUTHORITATIVE**: frozen MLP probe, 5 seeds, test touched once. `final_test_archive/` — earlier archived run.
* `sae_finetuning/layer_6_{arch}/` — reconstruction-only and truth-aware checkpoints + validation comparison. **VALID (scope)**: base and fine-tuned SAEs differ in L0 (not sparsity-matched). `heldout_comparison_all_datasets.csv` — **AUTHORITATIVE**: all 9 variants on identical OWT-test / GoT-test / RepE-test / causal sets.
  * **Reclassified 2026-11-07 (numbers unchanged):** wherever the `truth_aware` variant is scored with the frozen **logistic** probe, the result is **circular and is not evidence of truth preservation**. This covers `validation_comparison.csv`, the `got_*_frozen_logistic_*` columns of `heldout_comparison_all_datasets.csv`, and `got_faithfulness_frozen_probes.csv`.
  * The reason: that probe is the training teacher of `06_finetune_sae.py`, with identical weights.
  * With the independent MLP probe (`independent_truth_probe/`, `final_test_evaluation/`), truth-aware is **worse** than reconstruction-only in all 6 architecture × domain cells.
  * See `docs/2026-11-07_truth_aware_finetuning_audit.md`.
* `got_faithfulness_frozen_probes.csv` — **AUTHORITATIVE**: frozen logistic + MLP probes (layer 6) and the layer-10 frozen-probe result in one table.

**Layer 10, public SAELens `gpt2-small-res-jb` (legacy scripts `scripts/run_geometry_*.py`):**
* `baseline_probe_results.csv`, `sae_faithfulness_results.csv`, `sae_*`, `geometry_of_truth_feature_stats.csv` — **VALID (scope)** but **SUPERSEDED as a faithfulness metric**: the "reconstructed" probe is *re-trained on reconstructions* (decodability, 0.851 AUROC), not the original probe applied to them.
* `layer10_frozen_probe/` — **AUTHORITATIVE**: legacy reproduction + frozen original probe on reconstructions (AUROC 0.639, gap 0.21 [0.16, 0.27]), 6 city splits.

**All 12 layers, 108 uninterrupted OWT SAEs (added 2026-11-07; scripts `scripts/shared/all_layer_frozen_probe_{evaluation,analysis}.py`):**
* `all_layer_frozen_probe_evaluation/` — **AUTHORITATIVE** for cross-layer truth preservation. Probes are fit on original train rows, C is selected on original validation rows (same grid and rule as script 03, C values identical), and the probes are applied frozen to reconstructions. `protocol.json` was written before any SAE was scored. Test rows were scored only, never used for selection.
  * `per_run.csv`: one row per domain × layer × arch × seed × split, with the sparsity flag and checkpoint path.
  * `summary.csv`: mean/sd over matched seeds.
  * `matched_architecture_comparison.csv`, `reconstruction_vs_truth_correlations.csv`.
  * `excluded_unmatched_runs.csv`: layer 0 JumpReLU seed 42.
  * `decodability_diagnostic_validation.csv`: **DIAGNOSTIC only**. A probe re-fit on reconstructed train rows and scored on reconstructed validation rows. It measures decodability, not faithfulness.
* Caveats:
  * On GoT final-token activations, normalized MSE is > 1 (EV < 0) at layers ≥ 1: the within-set spread is tiny compared with the SAE error. Use `relative_sq_error` and `cosine` here.
  * SAE L0 on GoT prompts is far below 128 for ReLU/JumpReLU (8–114). Matching is defined on OWT.
  * Numerical layer 0: the true/false class means are identical (no truth signal at the embedding). `per_run.csv` holds float32 rounding noise in `direction_cosine`/`direction_norm_ratio` for it. The analysis masks these (original direction norm < 1e-4); the script now computes the means in float64.
  * The mass-mean direction is weak on GoT originals (val AUROC ≤ 0.63), so its gaps are not ranked.
* Figures: `figures/all_layers_*.png`, `figures/reconstruction_vs_truth_scatter.png`.

**Spliced-in causal evaluation (Phase 0, 2026-11-07):** `spliced_evaluation/` — **AUTHORITATIVE (validation only)**. GPT-2 predictions with `hook_resid_pre` replaced by each of the 108 uninterrupted SAE reconstructions, with identity, fp16, mean-ablation and norm-matched random-error controls. See `results/shared/README.md`. This is separate from the steering results.

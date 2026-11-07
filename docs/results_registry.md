# Results registry (as of 2026-11-07)

Each results folder's `README.md` gives file-level detail. This page lists the results that support claims.

**Status meanings:**
* **AUTHORITATIVE**: cite it.
* **VALID (scope)**: correct only within the stated scope.
* **SUPERSEDED**: kept for provenance; replaced by the result named in the table.
* **INVALID**: never cite.

## Authoritative

| Claim area | Result | Location |
|---|---|---|
| OWT reconstruction across layers (L0 = 128, d_sae = 3072, 15k steps, 3 seeds) | Uninterrupted single-optimizer protocol, all 12 layers × 3 archs × 3 seeds (108 runs); layer-0 JumpReLU seed 42 unmatched (L0 73.6), excluded from rankings | `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/` (`all_layers_*.csv`, `results/openwebtext/figures/all_layers_*.png`) |
| Architecture comparison at matched sparsity | Matched/unmatched tables (≤ 10 % L0 error) | `results/openwebtext/layer_comparison/sparsity_{comparison,matched_cells}.csv` |
| GoT truth preservation, all 12 layers | Frozen logistic (+ mass-mean) probes on 108 OWT SAEs; val/test | `results/geometry_of_truth/all_layer_frozen_probe_evaluation/` |
| RepE honesty preservation, all 12 layers | Frozen mass-mean + logistic probes on 108 OWT SAEs; val/test/cross-wording | `results/repe/all_layer_frozen_probe_evaluation/` |
| Spliced-in causal reconstruction, all 12 layers (Phase 0) | 108 SAEs replace `hook_resid_pre`; OWT CE/KL/loss recovered, GoT statement KL and truth preference, RepE persona-shift preservation; validation only; with controls | `results/shared/spliced_evaluation/`, `results/*/spliced_evaluation/` |
| Layer-6 bridge (historical vs uninterrupted SAEs) | Every validation metric; historical within ≈1 seed sd | `results/shared/spliced_evaluation/layer6_bridge.csv` |
| GoT truth preservation, layer 6 | Frozen MLP probe, 5 seeds, test touched once | `results/geometry_of_truth/final_test_evaluation/` |
| GoT truth preservation, frozen probes (layer 6 and layer 10) | Frozen logistic + MLP (L6); frozen original probe on public-SAE reconstructions (L10) | `results/geometry_of_truth/got_faithfulness_frozen_probes.csv`, `layer10_frozen_probe/` |
| Fine-tuning variants on identical held-out data | All 9 layer-6 variants: OWT test, GoT test, RepE val/test, cross-wording, causal. **Exception:** the `got_*_frozen_logistic_*` columns of the `truth_aware` rows are circular (see below) | `results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv` |
| RepE representation preservation | Validation + test + cross-wording | `results/repe/representation_preservation_val_test.csv` |
| Causal faithfulness (GPT-2, RepE) | Pre-registered test, 200 controls, positive control | `results/repe/steering/causal_random_controls/` |
| Gemma-2-2B truth steering | Layer-11 sweep, grouped splits, 20 random controls per strength | `results/gemma/steering_sweep_layer11/` |

## Pilots (validation only, not for final claims)

| Pilot | Location |
|---|---|
| P1: sparsity-controller anneal/freeze, JumpReLU threshold init, validation checkpoint selection (layers 0/2/6/10, seed 42) | `results/openwebtext/p1_sparsity_control_pilot/` |

## Superseded or invalid (kept, do not cite for the stated purpose)

| Result | Status | Replaced by |
|---|---|---|
| Truth-aware fine-tuned SAEs (layer 6) judged with the **logistic GoT probe**: `sae_finetuning/layer_6_*/validation_comparison.csv` (`truth_aware` rows); `heldout_comparison_all_datasets.csv` (`got_*_frozen_logistic_*` columns, `truth_aware` rows); `got_faithfulness_frozen_probes.csv` (`truth_aware` rows); `results/shared/spliced_evaluation/layer6_bridge.csv` (`val_probe__*_logistic_*`, `historical_finetuned_truth_aware` rows) | **INVALID as evidence of truth preservation (circular).** The evaluation probe is the training teacher: same rows, labels, C, solver and seed; weights identical. Valid only as "objective attained". Numbers kept unchanged. | Independent probes: MLP (`independent_truth_probe/`, `final_test_evaluation/`). With these, truth-aware is worse than reconstruction-only in all 6 cells. Audit: `docs/2026-11-07_truth_aware_finetuning_audit.md` |
| `results/openwebtext/reconstruction_evaluation/layers_2_6_10/` (mixed procedures) | SUPERSEDED for cross-layer claims; VALID within a layer | `uninterrupted_protocol/` |
| `results/openwebtext/convergence/layer_{2,10}/*_warmstart_1000_to_15000/` | SUPERSEDED | `*_uninterrupted_seed*/` |
| `results/openwebtext/layer_comparison/architecture_summary.csv` | **INVALID** | `sparsity_matched_cells.csv` |
| `results/openwebtext/sae_sparsity_sweep/` (1000 steps) | VALID only as a measure of convergence speed | none yet (converged sweep, Method B) |
| `results/openwebtext/reconstruction_evaluation/layer_6_finetuning_generalization/` (validation only) | SUPERSEDED | `heldout_comparison_all_datasets.csv` |
| `results/geometry_of_truth/sae_faithfulness_results.csv` (legacy L10, probe re-trained on reconstructions) | SUPERSEDED as a faithfulness metric (it measures decodability) | `layer10_frozen_probe/` |
| `results/repe/steering/{sae_steering_*,bidirectional_fidelity,bootstrap_fidelity_ci,final_repe_causal_summary}.csv` | SUPERSEDED as causal evidence (no SAE in the forward pass) | `causal_random_controls/` |
| `results/tables/gemma/neg_cities/`, `gemma_run.log`, `gemma_neg.log` (`aditi`-branch script) | SUPERSEDED (city leakage, no controls, neg_cities crash) | `results/gemma/steering_sweep_layer11/` |

## Valid within scope (still in use)

* **Layer-6 "pretrained" SAEs** (`convergence/layer_6/*_resume_5000_to_15000/`). These are the base SAEs of every GoT, RepE, fine-tuning and causal experiment.
  * They are not retrained. All of those comparisons are within layer 6, so the optimizer-procedure difference never enters them.
  * Their OWT test EV differs from the uninterrupted layer-6 runs by ≤ 0.001.
  * "Pretrained" means *trained by us on OWT*. It does not mean a public SAE.
* **GoT and RepE validation-stage results.** These are valid as validation numbers. RepE layer 6 was selected on validation, so those figures are optimistic.
* **Fine-tuned SAEs.** These are not sparsity-matched to their base SAEs. Fine-tuning has no sparsity term, so L0 drifts, e.g. ReLU 65 → 144 on GoT.
  * Their MLP-probe results are only partially independent: the probe uses the same train rows and labels as the teacher. Read them together with the L0 drift.
  * The GoT layer-6 test split has now been scored twice for these 9 SAEs, so it can no longer select among them.
  * RepE, OWT and causal results for the fine-tuned SAEs are independent of the GoT training and remain valid.

# Results registry (as of 2026-11-06)

Each results folder's `README.md` gives file-level detail. This page lists the results that support claims.

**Status meanings:**
* **AUTHORITATIVE**: cite it.
* **VALID (scope)**: correct only within the stated scope.
* **SUPERSEDED**: kept for provenance; replaced by the result named in the table.
* **INVALID**: never cite.

## Authoritative

| Claim area | Result | Location |
|---|---|---|
| OWT reconstruction across layers (L0 = 128, d_sae = 3072, 15k steps, 3 seeds) | Uninterrupted single-optimizer protocol, layers 2/6/10 | `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/` |
| Architecture comparison at matched sparsity | Matched/unmatched tables (≤ 10 % L0 error) | `results/openwebtext/layer_comparison/sparsity_{comparison,matched_cells}.csv` |
| GoT truth preservation, layer 6 | Frozen MLP probe, 5 seeds, test touched once | `results/geometry_of_truth/final_test_evaluation/` |
| GoT truth preservation, frozen probes (layer 6 and layer 10) | Frozen logistic + MLP (L6); frozen original probe on public-SAE reconstructions (L10) | `results/geometry_of_truth/got_faithfulness_frozen_probes.csv`, `layer10_frozen_probe/` |
| Fine-tuning variants on identical held-out data | All 9 layer-6 variants: OWT test, GoT test, RepE val/test, cross-wording, causal | `results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv` |
| RepE representation preservation | Validation + test + cross-wording | `results/repe/representation_preservation_val_test.csv` |
| Causal faithfulness (GPT-2, RepE) | Pre-registered test, 200 controls, positive control | `results/repe/steering/causal_random_controls/` |
| Gemma-2-2B truth steering | Layer-11 sweep, grouped splits, 20 random controls per strength | `results/gemma/steering_sweep_layer11/` |

## Superseded or invalid (kept, do not cite for the stated purpose)

| Result | Status | Replaced by |
|---|---|---|
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

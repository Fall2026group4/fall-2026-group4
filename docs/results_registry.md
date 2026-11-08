# Results registry (as of 2026-11-08)

Each results folder's `README.md` gives file-level detail. This page lists the results that support claims.

The 2026-11-04 professor-review corrections are cited from their dataset folders below. The separate
`results/professor_review_corrections/` copy was retired on 2026-11-07 (`RESEARCH_DECISIONS.md` D20; mapping in
`docs/README.md` and `docs/migration_manifest.csv`).

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
| RepE honesty preservation, all 12 layers (**HISTORICAL since 2026-11-08, D23**) | Frozen mass-mean + logistic probes on 108 OWT SAEs; val/test/cross-wording | `results/repe/all_layer_frozen_probe_evaluation/` |
| Spliced-in causal reconstruction, all 12 layers (Phase 0) | 108 SAEs replace `hook_resid_pre`; OWT CE/KL/loss recovered, GoT statement KL and truth preference, RepE persona-shift preservation; validation only; with controls | `results/shared/spliced_evaluation/`, `results/*/spliced_evaluation/` |
| Layer-6 bridge (historical vs uninterrupted SAEs) | Every validation metric; historical within ≈1 seed sd | `results/shared/spliced_evaluation/layer6_bridge.csv` |
| GoT truth preservation, layer 6 | Frozen MLP probe, 5 seeds, test touched once | `results/geometry_of_truth/final_test_evaluation/` |
| GoT truth preservation, frozen probes (layer 6 and layer 10) | Frozen logistic + MLP (L6); frozen original probe on public-SAE reconstructions (L10) | `results/geometry_of_truth/got_faithfulness_frozen_probes.csv`, `layer10_frozen_probe/` |
| Fine-tuning variants on identical held-out data | All 9 layer-6 variants: OWT test, GoT test, RepE val/test, cross-wording, causal. **Exception:** the `got_*_frozen_logistic_*` columns of the `truth_aware` rows are circular (see below) | `results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv` |
| RepE representation preservation (**HISTORICAL since 2026-11-08, D23**) | Validation + test + cross-wording | `results/repe/representation_preservation_val_test.csv` |
| Causal faithfulness (GPT-2, RepE) (**HISTORICAL since 2026-11-08, D23**) | Pre-registered test, 200 controls, positive control | `results/repe/steering/causal_random_controls/` |
| Gemma-2-2B truth steering (appendix only, D23) | Layer-11 sweep, grouped splits, 20 random controls per strength | `results/gemma/steering_sweep_layer11/` |

## Pilots (validation only, not for final claims)

| Pilot | Location |
|---|---|
| E2E-inspired fine-tuning layer 6 (B: +output KL; C: +KL+downstream; 3 archs × 3 seeds); **behavioural success for TopK only**, the L0 guard fails for ReLU/JumpReLU, the truth hypothesis fails everywhere | `results/geometry_of_truth/e2e_inspired_layer6/` |
| KL-weight ablation, layer 6 TopK (w = 0.25, 0.50 of Method B's KL; 3 seeds); **pre-registered verdict INCONCLUSIVE**: no measurable truth trade-off between A and B; behaviour gain saturates by w = 0.25 | `results/geometry_of_truth/kl_weight_ablation_layer6_topk/` |
| Method A layer 6 (GoT-A reconstruction fine-tuning + OWT replay; ReLU/TopK/JumpReLU × 3 seeds; OWT-only control); **pre-registered success NOT met** (MLP gap lower on 9/9 seeds vs both controls, but not by > 2 sd) | `results/geometry_of_truth/method_a_layer6/` |
| Corrected truth-aware pilot (layer 6 TopK, 3 seeds, A/B groups, independent probes); **pre-registered success NOT met** | `results/geometry_of_truth/truth_aware_pilot/` |
| R1 RepE honesty-preservation fine-tuning, layer 6 (R-recon / R-honesty / R-samepersona, 3 archs × 3 seeds; independent B probes); **pre-registered verdict: ReLU INCONCLUSIVE (RepE-prompt L0 guard failed), TopK/JumpReLU NOT SUPPORTED**; test not touched | `results/repe/r1_honesty_preservation_layer6/` |
| R2 SAE-in-the-loop clamp of honesty features, layer 6 (encode → clamp → decode + error; 9 base SAEs × k ∈ {1,4,16}; C2/C3/unrelated/raw-direction/positive controls; OWT damage); **descriptive validation verdict: NOT SUPPORTED** (target E below random-feature controls; edits adverse); test not touched. Cite only as a validation finding, not as a causal result | `results/repe/r2_sae_in_the_loop_steering_layer6/` |
| M0 within-layer reanalysis (VALIDATION): reconstruction error vs truth gap and splice damage within layer; L0-adjusted architecture comparison; corrected splice-ratio range | `results/shared/m0_within_layer_reanalysis/` |
| P1: sparsity-controller anneal/freeze, JumpReLU threshold init, validation checkpoint selection (layers 0/2/6/10, seed 42) | `results/openwebtext/p1_sparsity_control_pilot/` |

## HISTORICAL since 2026-11-08 (D23): kept for provenance; not evidence for their original claims

| Result | Why | Replaced by |
|---|---|---|
| `results/repe/all_layer_frozen_probe_evaluation/`, `results/repe/representation_preservation_val_test.csv`, the `repe_*` and `honesty_*` columns of `results/shared/spliced_evaluation/unified_comparison.csv`, and the pooled ρ ≈ 0.86 "reconstruction tracks honesty preservation" claim | Length confound: ' untruthful' is 3 tokens longer than ' honest'; length alone AUROC 0.80; layer-0 probe 0.83 | M1 RepE-LM (`results/repe/repe_lm_gates/`) and M2 |
| `results/repe/steering/causal_random_controls/` (6c) | Old prompts; tiny raw-direction effect; 28 uncorrected comparisons; not SAE-in-the-loop | M3/M4 |
| `results/repe/r1_honesty_preservation_layer6/`, `results/repe/r2_sae_in_the_loop_steering_layer6/` | Built on the length-confounded task; R2's clamped feature tracks prompt length | M1 → M4 |
| GoT numerical (`larger_than`/`smaller_than`) as a truth measure | Surface shortcut (second number alone AUROC 0.84) | Diagnostic only; M2 uses cities/neg_cities |
| `results/gemma/steering_sweep_layer11/` | No SAE; bias shift; 20 controls | Appendix only |

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
* **RepE validation statement 326** near-duplicates train statement 589 (found 2026-11-07). Earlier RepE validation numbers include it (1 of 90 statements); R1/R2 exclude it.
* **GoT and RepE validation-stage results.** These are valid as validation numbers. RepE layer 6 was selected on validation, so those figures are optimistic.
* **Fine-tuned SAEs.** These are not sparsity-matched to their base SAEs. Fine-tuning has no sparsity term, so L0 drifts, e.g. ReLU 65 → 144 on GoT.
  * Their MLP-probe results are only partially independent: the probe uses the same train rows and labels as the teacher. Read them together with the L0 drift.
  * The GoT layer-6 test split has now been scored twice for these 9 SAEs, so it can no longer select among them.
  * RepE, OWT and causal results for the fine-tuned SAEs are independent of the GoT training and remain valid.

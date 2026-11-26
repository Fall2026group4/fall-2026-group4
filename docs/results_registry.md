# Results registry (as of 2026-11-15: #27 state plus the M6 row from #28)

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
| GoT truth preservation, all 12 layers | Frozen logistic (+ mass-mean) probes on 108 OWT SAEs; val/test. **Interpretation superseded by M2 (2026-11-08):** the frozen-logistic gap is mostly noise-like; cite M2 for information loss | `results/geometry_of_truth/all_layer_frozen_probe_evaluation/` |
| RepE honesty preservation, all 12 layers (**HISTORICAL since 2026-11-08, D23**) | Frozen mass-mean + logistic probes on 108 OWT SAEs; val/test/cross-wording | `results/repe/all_layer_frozen_probe_evaluation/` |
| Spliced-in causal reconstruction, all 12 layers (Phase 0) | 108 SAEs replace `hook_resid_pre`; OWT CE/KL/loss recovered, GoT statement KL and truth preference, RepE persona-shift preservation; validation only; with controls | `results/shared/spliced_evaluation/`, `results/*/spliced_evaluation/` |
| Layer-6 bridge (historical vs uninterrupted SAEs) | Every validation metric; historical within ≈1 seed sd | `results/shared/spliced_evaluation/layer6_bridge.csv` |
| GoT truth preservation, layer 6 | Frozen MLP probe, 5 seeds, test touched once | `results/geometry_of_truth/final_test_evaluation/` |
| GoT truth preservation, frozen probes (layer 6 and layer 10) | Frozen logistic + MLP (L6); frozen original probe on public-SAE reconstructions (L10) | `results/geometry_of_truth/got_faithfulness_frozen_probes.csv`, `layer10_frozen_probe/` |
| Fine-tuning variants on identical held-out data | All 9 layer-6 variants: OWT test, GoT test, RepE val/test, cross-wording, causal. **Exception:** the `got_*_frozen_logistic_*` columns of the `truth_aware` rows are circular (see below) | `results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv` |
| RepE representation preservation (**HISTORICAL since 2026-11-08, D23**) | Validation + test + cross-wording | `results/repe/representation_preservation_val_test.csv` |
| Causal faithfulness (GPT-2, RepE) (**HISTORICAL since 2026-11-08, D23**) | Pre-registered test, 200 controls, positive control | `results/repe/steering/causal_random_controls/` |
| GPT-2 improved SAEs (R2), reconstruction and behaviour, **CONFIRMED TEST** (M5, 2026-11-10) | Old vs R2 SAEs at L1/L8/L11: NMSE −31 to −74 %, spliced ΔCE −35 to −79 %; R4 KL fine-tune ΔCE −16 to −17 %; GoT information unchanged; single pre-registered run. Test rows previously scored for the old reference SAEs only (D31); independently re-verified in `docs/final_defense/EXPERIMENT_AUDIT.md` | `results/shared/m5_confirmatory_test/` |
| GPT-2 improved SAEs (R2), all ladder levels, VALIDATION | 39 in-house SAEs (100M tokens, width 12,288/3,072) vs old base SAEs; causal T_rec/T_full 0.62 → 0.73 at L8 (validation only) | `results/openwebtext/r2_scaled_saes/` |
| Gemma-2-2B truth steering (appendix only, D23) | Layer-11 sweep, grouped splits, 20 random controls per strength | `results/gemma/steering_sweep_layer11/` |
| Gemma-2-2B SAE reconstruction and behavioural splicing (M6, D44; **VALIDATION**) | Pre-registered: Gemma Scope width-16k JumpReLU SAEs at `resid_post` layers 5/12/19 on 512 Pile validation documents; gates V1–V5 pass; **H-G1 supported at all three layers** (SAE-splice ΔCE > norm-matched random-error ΔCE; 95 % and Bonferroni 98.33 % CIs exclude 0). Cite as behavioural splicing and reconstruction only: no layer ranking, no representation-preservation, causal or GPT-2 vs Gemma superiority claim; bf16 logit-precision limitation (H4a). Gemma information and causal faithfulness not tested | `results/gemma2_pile/gemma_scope_splicing/` (README §2, `h_g1.csv`, `summary_by_layer*.csv`, `figures/`) |
| Gemma-2-2B Gemma Scope width × L0 ladder (R3, D46; **VALIDATION**, Tier A only) | Pre-registered: 36 Gemma Scope JumpReLU SAEs (widths 16k/32k/65k) at `resid_post` layers 5/12/19 on M6's 512 Pile validation documents; all gates pass; V6: M6 reproduced within margin. **H1** (higher L0 lowers ΔCE_SAE) supported in 7/7 cells; **H2** (wider at ±10 % matched L0 lowers ΔCE_SAE) supported in 4/4 bands formed (none at layer 5); **H3** (SAE-splice ΔCE > norm-matched random) holds for 36/36; Q0b: T4 bf16 within margin of fp32 for ΔCE_SAE, beyond margin for top-1 agreement (one cell). Cite as reconstruction and behavioural splicing only: no layer ranking, no "best SAE", nothing for widths above 65k (Tier B not run, Deviation 2) | `results/gemma2_pile/r3_gemma_scope_ladder/` (README Section 2, `ladder/`, `precision_t4_vs_fp32/`, `figures/`) |

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
| M1 RepE-LM equal-length persona task (gates on TRAIN; validation descriptive): construction shortcut-free (K1/K2), modest persona transfer (K5 0.70), behavioural effect 2x synonym null (K7) | `results/repe/repe_lm_gates/`, `data/repe/repe_lm/` |
| M2 four-level information faithfulness (VALIDATION): frozen vs re-trained vs nonlinear vs sparse-z readers, noise and PCA references, error-term probe; supersedes the frozen-probe-only reading of `*/all_layer_frozen_probe_evaluation/` | `results/shared/m2_four_level_faithfulness/` |
| M3 causal site sweep in original GPT-2 (TRAIN selection, VALIDATION confirmation): where the RepE-LM persona-instruction effect is carried, by layer and position; per template | `results/repe/causal_site_sweep/` |
| IMP-L multi-layer fine-tuning (VALIDATION): Method A, OWT-only and output-KL at layers 3, 8, 9; pre-registered classes in `criteria.json`. Cite the frozen-probe truth gains together with the re-trained-reader null; ReLU/JumpReLU output-KL at L8/L9 is not creditable (L0 guard) | `results/geometry_of_truth/multilayer_finetune/` |
| M4 SAE causal-faithfulness decomposition (VALIDATION): original vs SAE-reconstructed vs error-only patch effects at L1 slot, L8 final, L11 final; feature patches with matched controls | `results/repe/sae_patch_decomposition/` |
| E3 causal preservation after fine-tuning (VALIDATION): M4 decomposition at L8 final for 9 base + 27 IMP-L layer-8 fine-tunes; pre-registered classes in `criteria.json` (Method A NO IMPROVEMENT 3/3; output-KL TopK NO IMPROVEMENT, ReLU/JumpReLU NOT CREDITABLE) | `results/repe/e3_finetuned_causal_preservation/` |
| RLM-RP RepE-LM representation preservation and robustness (VALIDATION): 216 SAEs (base L0–11, IMP-L, layer-6 fine-tunes); frozen vs re-trained persona gaps, noise reference, H−U geometry, held-out wording, synonym, neutral, near-duplicate and L0-truncation controls; supersedes the historical length-confounded RepE preservation results for every RepE representation claim | `results/repe/repe_lm_representation_preservation/` |
| R4 KL fine-tune at fixed L0 (VALIDATION; behaviour confirmed in M5): ΔCE −18 % vs MSE control; no information gain; causal regression at L11. Behaviour-only tool | `results/openwebtext/r4_kl_finetune/` |
| R5 rank-k causal skyline and random floor at the M4 sites (VALIDATION, descriptive) | `results/repe/r5_causal_subspace_skyline/` |
| E2 public res-jb reference (VALIDATION, descriptive): our headline conclusions hold for an external 300M-token SAE. res-jb is an external reference, not our improvement | `results/shared/e2_public_sae_reference/` |
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

## 2026-11-13 additions

| Item | Status | Notes |
|---|---|---|
| `data/repe/repe_lm_test/` | EVALUATED ONCE (2026-11-13) | TEST rows are now spent: never use them for selection or tuning |
| `results/repe/repe_lm_test_evaluation/` (H1-TEST) | CONFIRMED TEST RESULT: HT2 (9/9 cells), HT1-C1 L8, HT3-C1 L8. NOT CONFIRMED: HT1, HT3 as hypotheses (C3 cells fail) | Cite HT1/HT3 per cell, never as "confirmed hypotheses"; HT4 and secondary comparisons are descriptive |
| M4 "SAE-in-the-loop feature patches" | VALID WITHIN SCOPE (VALIDATION) | Cite as "SAE-selected directions, error term kept"; not as SAE-mediated (D33) |
| `results/shared/pca_capacity_matched/` | VALIDATION FINDING (descriptive), citable with the matching criterion stated | Qualifies every "SAE vs PCA" claim (D34); error-matched claims replicate |
| `results/shared/power_sensitivity/` | DESCRIPTIVE (design sensitivity) | Cite as an MDE, not as power for the observed effects |
| "No trained SAE beats PCA" (M2, M5, paper) | VALID ONLY AS AN ERROR-MATCHED STATEMENT | Must state "at equal OWT reconstruction error (k = 298–519)"; false at k = 128 for some ReLU-family cells (D34). Paper and curated docs revised 2026-11-13 |
| `results/shared/pca_capacity_matched/posthoc_multiplicity/` | DESCRIPTIVE (post-hoc, approximate) | Cite only to qualify PCA-CM "better" cells as per-cell, not family-wise (D34) |
| "SAE better than PCA" (PCA-CM) | VALIDATION, per-cell criterion | Never write "significantly better"; write "better under the study's comparison criterion" |
| R2 vs old TopK T_rec/T_full at L8 (0.625 → 0.726) | VALIDATION IMPROVEMENT, BORDERLINE; **TEST-CONFIRMED for C1** (0.602 → 0.729, Holm p = 0.021) on 2026-11-13; C3 not confirmed | See H1-TEST row and D36 |

## 2026-11-13 AUDIT-SENS (D38)

| Item | Status | Notes |
|---|---|---|
| `results/shared/audit_sensitivity/` | SENSITIVITY / DIAGNOSTIC (VALIDATION-based) | Qualifies M5 and H1-TEST; never cite as a confirmatory test |
| H1-TEST HT1-C1 and HT3-C1 | CONFIRMED (frozen seed-level rule); FRAGILE (AUDIT-SENS S1) | Always cite with the qualifier |
| H1-TEST HT2 | CONFIRMED; robust to statement sampling | — |
| "Part of the persona loss is persona-specific" (M2/RLM-RP) | WITHDRAWN | S2: the neutral control loses an equal or larger share |
| "Frozen GoT collapse is largely perturbation sensitivity" (M2) | WITHDRAWN as worded | S5: 33–50 % is label-aligned SAE error at L6–11 |
| M2 GoT labels at L0–4 ("uninformative") | VALID FOR THE POOLED TASK ONLY | S4: per polarity, truth is linearly decodable from L1 |
| Old-architecture rankings on persona/GoT metrics | NOT ROBUST | S3: 13/19 change at matched task L0 |

## 2026-11-14 TSP

| Item | Status | Notes |
|---|---|---|
| `results/geometry_of_truth/truth_subspace_preservation/diagnostics/` | VALIDATION, descriptive | Root-cause diagnostics; group-A subspace, group-B readers |
| `results/geometry_of_truth/truth_subspace_preservation/pilot/` | VALIDATION PILOT, NOT SUPPORTED (frozen rule) | Cite effect sizes only with their CIs and the verdict; never as an improvement |
| `results/geometry_of_truth/truth_subspace_preservation/verification/` | VALIDATION verification (post-hoc checks exploratory) | Confirms the pilot numbers and the NOT SUPPORTED verdict. Supersedes the pilot's secondary CIs for error-only AUROC / MLP / sparse-z / kNN, which omitted statement sampling. Specificity holds for the frozen gap only |
| `results/geometry_of_truth/truth_subspace_preservation/replication_phase0/` | VALIDATION (DEV only; feasibility) | Fresh-family eligibility and headroom; no TSP result. The TEST splits in `data/got/fresh/splits/` are LOCKED (never read) |
| `results/geometry_of_truth/truth_subspace_preservation/replication/` | VALIDATION, confirmatory (pre-registered) | Cities: SUPPORTED (meaningful), truth-specific, linear and nonlinear, with the TRADE-OFF qualifier (persona/M4 non-inferiority not demonstrated). Fresh-family TEST results are secondary or exploratory and null. Cite the cities effect only as in-distribution |

## 2026-11-15 additions

| Item | Status | Notes |
|---|---|---|
| `results/shared/dead_latent_audit/` (`docs/DEAD_LATENT_AUDIT.md`) | VALIDATION FINDING (descriptive diagnostic) | Dead fraction of 108 base and 36 trained R2 SAEs, plus 3 frozen-random-decoder baselines, on OWT validation (12,700 tokens). Cite as a diagnostic, not as evidence of faithfulness; it makes no attribution to data, recipe, duration or width |
| Appendix / M0 JumpReLU alive-latent range "469–2,947" | SUPERSEDED by "469–2,941" (transcription correction) | Source value `architecture_by_layer.csv` is 2,941.0, unchanged |

## 2026-11-16 additions

| Item | Status | Notes |
|---|---|---|
| `results/geometry_of_truth/method_a_multilayer/` on branch `simba-method-a-multilayer` (`7f2557b`; Method A layers 5/6/9) | BRANCH-ONLY; EXPLORATORY / VALIDATION; UNMERGED; NOT CITED in the final manuscript | Do not use for final claims (D45). Criterion met in 1/9 cells; layer-9 disagreement with IMP-L unresolved; provenance caveats in `RESEARCH_PROGRESS.md` §53. Cite IMP-L (`multilayer_finetune/`) and `method_a_layer6/` instead |

## 2026-11-21 additions (final week)
| Result | Status | Location |
|---|---|---|
| XT-G0 Gemma task activations, K1 / K7 gates | VALIDATION (completed; gates passed) | `results/gemma/xtg_task_baselines/` |
| XB matched splicing, primary 1024 arm (GPT-2 vs Gemma) | **VALIDATION**: Gemma results and gates citable; **GPT-2 primary-arm results NOT citable as SAE comparisons** (128-token SAEs out of distribution; D50); no matched stratum | `results/cross_model/xb_matched_splicing/` (§59) |
| XT-G / DX truth preservation (both models) | **VALIDATION (citable)**: 0/410 readout shift; persona information loss in both models (not honesty-specific); SAE error concentrated along the task direction (truth for GoT, persona / neutral for RepE); cross-model descriptive only (D54) | `results/cross_model/xtg_truth_preservation/` (§63) |
| TI-1 TSP generalisation | **VALIDATION (citable)**: GENERALISES, 3/3 cells SUPPORTED (L11 TopK −0.042, L8 BatchTopK −0.015, L8 ReLU −0.026 refit gap vs ctrl); cities only | `results/geometry_of_truth/tsp_generalisation/` (§56, D48) |
| X0 token exposure within R2 runs | **VALIDATION (citable, descriptive)**: not saturated at 50M (25M→50M IMPROVES NMSE 12/12, ΔCE 11/12) | `results/openwebtext/r2_scaled_saes/scaling/` (§57) |
| SIL SAE-in-the-loop persona mediation (GPT-2) | **VALIDATION**: MEDIATED by rule for all families, **but the negative control also passes**; cite only as "specificity inconclusive" plus D2 retention (0.82–0.92 TopK/BatchTopK/JumpReLU vs ≈ 0.5 ReLU and control) | `results/repe/sae_in_the_loop_mediation/` (§58, D49) |

| PCA-CM-T capacity-matched PCA, TEST | **TEST (rows previously exposed in M5; PCA arms unseen)**: P1 confirmed (0/20 SAEs worse than PCA-128); advantage only in the confirmed cells (D51) | `results/shared/pca_capacity_matched_test_protocol/` (§60) |

| XB A4, 128-token shared span (GPT-2 vs Gemma) | **VALIDATION, sensitivity (descriptive)**: both models' SAEs worse than norm-matched random by CI; 269/324 cross-model differences shrink vs 1024 tokens; Holm labels uninformative (p floor, D52); no matched pair | `results/cross_model/xb_matched_splicing/a4_ctx128/` (§61) |

| E1 TopK at task-matched L0 (old SAEs) | **VALIDATION**: headline NOT robust (1/4 significant architecture differences persist after BH); M2/M4 task-prompt architecture differences largely sparsity-confounded (D53) | `results/shared/e1_task_l0_matched_topk/` (§62) |

## 2026-11-23 additions (recovery session)
| Result | Status | Location |
|---|---|---|
| TI-2 TSP-MF: NO TRANSFER DETECTED (MF-H1 +0.0020 [−0.0096, +0.0137]) | VALIDATION (completed; analysis deviation declared) | `results/geometry_of_truth/tsp_multifamily/` |
| TI-4 TSP × KL: COMBINES | VALIDATION (completed) | `results/geometry_of_truth/tsp_kl_factorial/` |
| TI-3 minimal-pair preservation | WITHDRAWN: no result, not citable | `results/geometry_of_truth/minimal_pair_preservation/` |
| Ladder batch 2 (XB L12 only) | PARTIAL: XB per-document outputs only; analysis not run, not yet citable | `results/cross_model/xb_ladder_batch2/` |
| XC-3 steering parity | PRE-REGISTERED; smoke outputs (TRAIN) not citable | `results/cross_model/xc3_steering_parity/` |
| XC-3 steering parity, GPT-2 vs Gemma (H1/H2 met 5/6; H3 INCONCLUSIVE) | CONFIRMED TEST RESULT for H1–H3 (GPT-2 TEST rows previously read once by H1-TEST; Gemma pristine); H4 descriptive | `results/cross_model/xc3_steering_parity/` |
| XC-2 Gemma GoT causal (T1 mass-mean met at L8/12/16/21; T2 reconstruction met at all 4) | CONFIRMED TEST RESULT for families A/C1/C2 (cities TEST rows previously exposed: GPT-2 decisions, 2026-11-01 Gemma L11 sweep; disclosed) | `results/cross_model/xc2_gemma_got_causal/` |
| XF fresh geonames cities TEST: P1 public-SAE LOSS at GPT-2 L3/6/9 and Gemma L5 (Gemma L12/19 ceiling); P2 TSP vs ctrl FAILS TO REPLICATE (0/4) | CONFIRMED TEST RESULT (pristine TEST, read once) | `results/cross_model/xf_fresh_cities_test/` |
| TSP-R / TI-1 tsp − ctrl advantage | VALIDATION finding; did NOT replicate on XF fresh TEST (cite both) | `results/geometry_of_truth/truth_subspace_preservation/replication/`, `results/geometry_of_truth/tsp_generalisation/` |
| XC-1 Gemma site sweep: L0 slot and L23 final selected and confirmed (R_A val 0.108, E_share 0.98) | VALIDATION finding (TRAIN selection, VALIDATION confirmation; reduced 9-layer grid) | `results/cross_model/xc1_gemma_site_sweep/` |
| M4-G Gemma SAE patch decomposition: preservation 0.96 (L0 slot), 0.75 (L12 final), 0.70 (L23 final); H4a/H4b 3/3; final-token sites interaction-dominated (bounds only) | VALIDATION finding (H4c not run) | `results/cross_model/m4_gemma_patch_decomposition/` |

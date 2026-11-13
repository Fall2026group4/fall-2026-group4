# Experiment changelog (audit trail)

Newest entries at the bottom. Each entry lists the commit, the files changed with reasons, methodological changes, and the effect of any bug fix on earlier results. File-level migration detail is in `docs/migration_manifest.csv` (161 rows); per-experiment provenance is in `docs/experiment_manifest.csv`.

Evidence labels as in `docs/RESEARCH_PROGRESS.md`. For merge and upload commits, only the substantive changes are listed.

## 2026-09-11 to 2026-10-12: `main` (ancestor of `sae-research`)

| Date | Commit | Change | Reason |
|---|---|---|---|
| 09-11…09-15 | `d516f43`…`412d9ec` | Repo init, MIT licence, directory structure, README, requirements | Project set-up |
| 09-15 | `070db9b` | `results/phase1_observations.md`, Feature 974 analysis | Phase 1 pretrained-SAE baseline |
| 09-21 | `55ac9a0`…`8f3814b` | GoT cities EDA (`results/geometry_of_truth_observations.md`) | Dataset suitability |
| 09-22 | `354b7ac` | OWT EDA results and notebook | Dataset suitability |
| 09-24 | several "Add/Delete files via upload" | Architecture diagrams and figures uploaded and deleted repeatedly. Feature-974 PNGs deleted (`c8da727`, `8e6600e`) | Diagram churn. Regression noted in the 09-29 progress report |
| 09-28 | `1775eb6` | Steering and detection moved from the notebook to `src/` (`src/models/gpt2_sae.py`, `src/sae/steering.py`, `src/evaluation/detection.py`); first tests; `mentions_paris` KeyError fixed; fixed seeds before generation | Instructor Review #2 |
| 09-28…10-03 | `03b7bae`…`ce60e2f` | GoT layer-10 legacy pipeline: `scripts/{prepare_geometry_data,cache_geometry_activations,run_geometry_*}.py`, `src/evaluation/geometry_*.py`, `results/geometry_of_truth/*.csv` | GoT probe and SAE faithfulness, public SAE |
| 10-03 | `b119eb4`, `78b20b1` | PRs #8 (dhruv) and #6 (simba-geometry-truth-eda) merged into `main` | Integration |
| 10-12 | `be138e8`…`aed8ffd` | `.drawio` and SVG diagrams deleted on `main` | Regression; restored only on `origin/dhruv` (`276c2ed`) |

## 2026-10-03 to 2026-10-27: team branches (NOT merged into `sae-research`) [BRANCH-ONLY]

| Branch | Commits | Change |
|---|---|---|
| `origin/dhruv` | `5e74754`, `a8956cc`, `0af0afe`, `d948449`, `b7c99e1`, `108bb9c`, `0114e90`, `c90b2ba` | OWT × pretrained SAEs pipeline, stages 0–6, ReLU and TopK; OOM fix (HF shuffle buffer disabled) |
| `origin/dhruv` | `555f384`, `1d7c216`, `dc07b83`, `276c2ed`, `f74d79a`, `3a18c32`, `38efa2d` | CI workflow and tests; `--n-random 200` controls and stats module; logging; SVG figures; diagrams restored; report sections; progress report 10-25 (Review #6) |
| `origin/simba-geometry-truth-eda`, `origin/simba-got-stages-gemma-eda` | `1504b5a`, `d43b767`, `428637d`, `17279d2`, `d93bd43`…`0343667` | GoT steering with random controls, feature interpretation, top-k recovery; Gemma-2 / Pile EDA |
| `origin/simba-got`, `origin/aditi` | `55c3c5b`…`eb142c0` | GoT stages 1–6 and RepE stages 1–6 (EDA, probes, SAE fidelity, cross-wording, steering with 20 controls) |
| `origin/aditi` | `d7f632c` | Gemma-2-2B cities baseline: layer table, next-word and judgment steering. Random row split; no controls. Its outputs `results/tables/gemma/neg_cities/` and `gemma_*.log` are also present on `sae-research` |

## 2026-10-30 to 2026-11-03: `sae-research` in-house pipelines

| Date | Commit | Files | Change and reason |
|---|---|---|---|
| 10-30 | `55ca6e2` | 12 empty placeholder scripts `scripts/openwebtext/01–12` (setup_saes, finetune_saes, steer…), `configs/openwebtext` | Initial OWT plan using pretrained SAEs + fine-tuning. Placeholders only (0 lines) |
| 10-30 | `8dc0b6e` | **Deleted** the 10 empty placeholders. **Added** `scripts/openwebtext/02–11`: `03_training_sae.py` (in-house SAE class and trainer), convergence scripts `05–08, 11`, `10_reconstruction_evaluation.py`; results `sae_pilot/`, `sae_sparsity_sweep/`, `convergence/`, `layer_comparison/` | Switched to in-house SAEs trained on OWT so all three architectures share one protocol |
| 10-30 | `989ea71` | `scripts/Geometry of Truth/01–06`, `results/got/{original_truth_probes,sae_truth_preservation,sae_diagnostic_truth_probes,sae_finetuning}` (+ checkpoints, gitignored), `results/tables/gemma/neg_cities/` | GoT layer-6 pipeline with grouped splits; reconstruction-only and truth-aware fine-tuning |
| 11-01 | `0d07739` | `scripts/Geometry of Truth/07–09`, `results/got/{independent_truth_probe,final_test_protocol,final_test_evaluation,final_test_archive}` | Independent MLP probe (5 seeds) and pre-registered final GoT test |
| 11-03 | `0eaad9c` | `scripts/repe/01–12`, `scripts/openwebtext/12_finetuning_generalization.py`, `data/repe/*`, `data/got/*`, `results/repe/*` | RepE pipeline, cross-wording, steering pilot, held-out steering, bootstrap fidelity |
| 11-03 | `326b0f6` | `data/repe/steering/*` (4 files) | Held-out steering question sets |

## 2026-11-04: professor-review corrections (uncommitted on that day; committed in `2ad0979` on 11-07)

* **New files only;** no existing file was edited. Outputs went to `results/professor_review_corrections/`; scripts to `scripts/professor_review/`.
* **New scripts:**
  * `scripts/openwebtext/13_consistent_protocol_retrain.py` (uninterrupted protocol);
  * `scripts/repe/13_causal_random_controls.py` (pre-registered causal test with 200 controls);
  * `scripts/gemma/01_steering_sweep.py` (grouped splits, controls, HF bf16 backend because TransformerLens was OOM-killed);
  * `scripts/professor_review/*.py`;
  * `src/tests/test_professor_review.py` (8 tests).
* **Environment:** `pytest` installed; SAELens layer-10 SAE downloaded.
* **Effect on earlier results:**

  | Earlier result | New status | Why |
  |---|---|---|
  | `architecture_summary.csv` | **INVALID** | averaged unmatched runs |
  | Cross-layer 2/6/10 results | SUPERSEDED by the uninterrupted protocol; valid within a layer | mixed optimiser procedures |
  | RepE "SAE steering" correlations | SUPERSEDED as causal evidence | no SAE in the forward pass |
  | GoT layer-10 re-trained-probe result | SUPERSEDED as a faithfulness metric | it measures decodability |
  | Gemma `aditi` baseline | SUPERSEDED | row-split leakage, no controls |
  | Quoted RepE cross-wording JumpReLU 0.8010 | corrected to 0.7394 | that is the saved value |

## 2026-11-06: reorganisation (committed 11-07 in `2ad0979`)

* **Backup:** `../backups/2026-11-06_pre_migration.tar`. SHA-256 recorded; restore test passed.
* **Renames (`git mv`, 47 renames):** `scripts/Geometry of Truth/` → `scripts/geometry_of_truth/`; `results/got/` → `results/geometry_of_truth/`. Empty `Truth/`, `of/` and `results/got/` removed.
* **Copies:** 83 files copied from `results/professor_review_corrections/` into the dataset folders. Originals kept and marked `MIGRATED.md`.
* **Scripts moved:**
  * `scripts/professor_review/*` → `scripts/openwebtext/14_sparsity_matching_analysis.py`, `scripts/geometry_of_truth/10_layer10_frozen_probe.py`, `scripts/shared/{finetuning_heldout_eval,plot_causal_and_gemma,write_manifest}.py`;
  * `src/tests/test_professor_review.py` → `test_faithfulness_metrics.py`.
* **Path edits:** GoT 03–09, RepE 04/08/13, OWT 12, Gemma 01. GoT 09's hash no longer matches `pretest_protocol.txt`; the original is recoverable with `git show 326b0f6`.
* **New docs:** `docs/{README,results_registry,research_gaps_and_plan,migration_manifest.csv,experiment_manifest.csv}` and status READMEs in each results folder.

## 2026-11-07: `sae-research`

| Commit | Files (A = added, M = modified) | Change, reason, effects |
|---|---|---|
| `2ad0979` | 168 files (118 A, 47 R, 3 M) | Commits the 11-04 corrections and the 11-06 reorganisation. Excludes caches, `.pt`, logs, `.bak` |
| `9446e56` | M `13_consistent_protocol_retrain.py` (checkpoint path recorded before training, overwrite refusal, GPU peak memory, per-invocation script hash); A `15_all_layer_baseline_analysis.py`, 3 CSVs, 4 figures; M results CSV/config, registry, README, manifest | 81 new OWT runs → all 12 layers. First launch failed (venv not active; nothing trained; log kept as `failed_launch_no_venv_2026-11-07.log`) |
| `8919158` | A `src/evaluation/faithfulness_metrics.py`, `src/sae/checkpoint_registry.py`, `scripts/shared/{all_layer_frozen_probe_evaluation,all_layer_frozen_probe_analysis,all_layer_decodability_diagnostic,build_checkpoint_registry}.py`, `src/tests/test_shared_metrics_registry.py`, `docs/checkpoint_registry.csv`, GoT/RepE `all_layer_frozen_probe_evaluation/`, figures | Shared metrics (reproduce the OWT metrics exactly), checkpoint registry, all-layer frozen-probe evaluation. First GoT run crashed at numerical layer 0 (zero mass-mean direction); fixed, re-run, log kept. Committed script = exact run-time version |
| `83bd9a9` | M `all_layer_frozen_probe_evaluation.py` | **Bug fix:** mass-mean directions computed in float64. Effect: only GoT numerical layer-0 direction metrics (float32 rounding noise ~3e-5). The analysis masks them; no other value changed |
| `468018c` | A `scripts/shared/spliced_reconstruction_{evaluation,analysis}.py`, `results/{openwebtext,geometry_of_truth,repe,shared}/spliced_evaluation/`, `results/shared/README.md`, prepared `scripts/openwebtext/16_p1_sparsity_control_pilot.py`, splice-hook test | Phase 0 and layer-6 bridge. **Smoke-test fixes before the real run:** (1) OWT original log-probs stored in fp32 (identity control was not exactly 0); (2) RepE "persona shift retained" ratio replaced by persona-shift cosine (random noise inflated the ratio 6.6×); (3) GoT truth preference reported per dataset (pooled datasets cancel) |
| `ef9fec4` | A `scripts/shared/p1_{validation_evaluation,analysis}.py`, `results/openwebtext/p1_sparsity_control_pilot/`, figure; M checkpoint registry (+24 rows) | P1 pilot (12 runs from the committed script `d9835eb9…`). Dominance-filter bug in `p1_analysis.py` fixed before outputs were committed |
| `1614be5` | A `docs/2026-11-07_truth_aware_finetuning_audit.md`, `results/geometry_of_truth/sae_finetuning/README.md`; M registry, READMEs | **Methodological correction:** truth-aware results judged with the teacher probe reclassified as circular; numbers unchanged |
| `dcf1486` | A `data/got/splits/train_AB_groups_truth_aware_pilot.csv` (**new split file**: GoT train groups A/B, seed 20261107), `scripts/geometry_of_truth/11–13_*`, `results/geometry_of_truth/truth_aware_pilot/`, figure | Corrected truth-aware pilot. Pre-registration and hashes recorded before training |
| `ca81ad2` | A `scripts/geometry_of_truth/14–16_*`, `results/geometry_of_truth/method_a_layer6/`, figure | Method A at layer 6 + OWT-only control. Reuses the same A/B split (hash re-verified) |
| `764dcbb` | A `scripts/geometry_of_truth/17–19_*`, `scripts/shared/layer6_faithfulness_eval.py` (factored from script 15, which is unchanged), `results/geometry_of_truth/e2e_inspired_layer6/`, figure; M checkpoint registry | E2E-inspired Methods B/C. Before training: fixed a hook that evaluated a tensor as a boolean in the downstream-loss check, and strengthened that check to cover all 5 layers against numpy. The regenerated registry rewrote integer columns as floats (format only; all 365 earlier rows unchanged) |
| `7ffdc70`, `8c3fe3a` | A `CLAUDE.md`, `docs/{RESEARCH_PROGRESS,EXPERIMENT_CHANGELOG,EXPERIMENT_RESULTS_SUMMARY,RESEARCH_DECISIONS,RESEARCH_ROADMAP}.md`, `docs/templates/EXPERIMENT_TEMPLATE.md`; M `docs/README.md` | Permanent documentation and experiment-tracking system |
| `3377116` | M `docs/RESEARCH_ROADMAP.md` (rewritten), `CLAUDE.md`, `docs/EXPERIMENT_CHANGELOG.md` | Roadmap completeness review: added outstanding GPT-2 tasks (training improvements, published JumpReLU / missing kernel-STE script, public-SAE baselines, AlignSAE, replications), sparsity-matching corrections S1–S6, causal/steering experiments C1–C7, Gemma tasks M1–M5, cross-model comparison and final test X1–X3, capstone deliverables F1–F6 (deadline 2026-12-08), and a naming map for the plan-vs-experiment 'Method' letters. CLAUDE.md: documentation applies even when not requested; roadmap read at task start; new tasks added to the roadmap; pre-commit documentation checklist |
| `2c87a6b` | A `scripts/geometry_of_truth/20_kl_weight_ablation.py`, `21_kl_weight_ablation_evaluation.py`, `results/geometry_of_truth/kl_weight_ablation_layer6_topk/` (README, verification, pre-registration, hashes, training info and monitor, per-SAE, summary, paired differences, criteria), 2 figures; 6 checkpoints (gitignored) + 1 reproduction-check checkpoint; M progress, summary, decisions, roadmap, registry, GoT README, experiment manifest, checkpoint registry and `src/sae/checkpoint_registry.py` (new family pattern) | KL-weight ablation (w = 0.25, 0.50 × α_B), layer 6 TopK, 3 seeds. Script 17 imported unchanged. No existing checkpoint or result modified. Pre-registered verdict INCONCLUSIVE |
| `875bc7f` | **Repository consolidation for upload (no experiment, no result value changed).** D `results/professor_review_corrections/` (42 tracked files) and `scripts/professor_review/` (6) and the duplicate `src/tests/test_professor_review.py`: every file had a hash-verified copy in the dataset folders since the 2026-11-06 migration (`2ad0979`); re-verified today, then the folders were moved to `../backups/2026-11-07_retired_professor_review_folders/` (85 files incl. 30 gitignored checkpoints and logs; SHA-256 verified) and one `retire_to_backup` row per file was appended to `docs/migration_manifest.csv`. Two copies had been extended after migration, deliberately (`9446e56`: `uninterrupted_protocol/run_config.json`, `test_metrics_all_runs.csv`, 27 → 108 runs); the 27 original rows are contained unchanged in the 108-row file (max abs diff 0). M `docs/checkpoint_registry.csv` (rebuilt: 391 → 361 rows; only the 30 rows pointing into the retired folder were removed, all other rows identical). M `README.md` (rewritten for the current research question, findings, limitations, reproduction). M `requirements.txt` (added the pinned research stack: torch, transformer-lens, transformers, datasets, scikit-learn, sae-lens, pyarrow; previously missing). M `.gitignore` (activation caches incl. their `manifest.json`, `*.bak`). M `CLAUDE.md`, `docs/README.md`, progress, roadmap, decisions (D20), registry. Placeholder "this commit" for the KL ablation resolved to `2c87a6b`. Tests: 25 pass (33 before; the 8 removed tests were exact duplicates) |
| `d17bb00` | A `results/repe/r1_honesty_preservation_layer6/README.md`, `results/repe/r2_sae_in_the_loop_steering_layer6/README.md`; M progress (§16a), roadmap ("In progress") | **R1/R2 pre-registration, committed exactly as drafted on 2026-11-07 before any R1/R2 code, data or training** (byte-identical to the backup in `../backups/2026-11-07_r1r2_planning/`). Clarifications and corrections made before training go in a separate, later commit as an addendum; the drafted text is not edited |
| `211218f` | A `scripts/repe/14_prepare_r1_r2_splits_and_wording.py`, `15_r1_honesty_finetune.py`, `16_r1_evaluation.py`, `data/repe/splits/r1_train_AB_groups.csv` (**new split file**), R1 `split_verification.json`, `preregistration.json`, `pretraining_hashes.json`, `smoke/` (3 base SAEs, validation; not a result); M R1 README (§10 pre-training addendum appended; §0–§9 unchanged) | **R1 pre-registration finalised before training.** Corrections: test pass and held-out-wording test set deferred (user: preserve the test split); verdict guards = R-honesty and R-recon. Open details fixed (pair definitions, λ, probe settings). Smoke observation disclosed: ReLU/JumpReLU RepE-prompt L0 drops quickly under RepE fine-tuning; protocol not changed |
| `d283a49` | A R1 results (`training_info.json`, `training_trace.csv`, `original_baseline.json`, `per_sae.csv`, `summary.csv`, `paired_differences.csv`, `criteria.json`, `eval_info.json`), `scripts/repe/19_r1_r2_figures.py`, `results/repe/figures/r1_honesty_preservation_layer6.png`; 28 checkpoints (gitignored) under `results/repe/sae_finetuning/r1_honesty_preservation_layer6/`; M `src/sae/checkpoint_registry.py` (two new family patterns), `docs/checkpoint_registry.csv` (+28 rows, no existing row changed), experiment manifest, R1 README (§11 results; title status), progress, summary, decisions (D21), roadmap, results registry, `results/repe/README.md` | **R1 executed from the frozen scripts (hashes verified at run time).** Verdict: ReLU INCONCLUSIVE (primary + transfer met; RepE-prompt L0 guard failed), TopK and JumpReLU NOT SUPPORTED. No test row read. Logs in `results/repe/r1_honesty_preservation_layer6/logs/` (gitignored) |
| `b232ce4` | A `scripts/repe/17_r2_sae_in_the_loop_steering.py`, R2 `preregistration.json`, `pretraining_hashes.json`, `smoke/` (smoke checks; pipeline test); M R2 README (§12 pre-run addendum appended; §0–§11 unchanged) | **R2 pre-registration finalised before the validation stage.** User decision 2026-11-08: validation now, test later (the `--test` stage refuses). Corrections: OWT damage set = the 100 cached validation sequences (plan said 200); criterion 5 checked on cross-wording validation at this stage. Pipeline-test validation outcomes (TopK, E < 0) seen before freezing; disclosed; protocol unchanged |
| (this commit) | A R2 validation outputs (`validation_targets.csv`, `validation_controls.csv`, `feature_selection.csv`, `validation_summary.csv`, `k_selection.json`, `validation_criteria_descriptive.json`, `validation_run_info.json`), figure `results/repe/figures/r2_sae_in_the_loop_validation.png`; M `scripts/repe/19_r1_r2_figures.py` (added `--r2`; the `--r1` path is unchanged), R2 README (§13 appended; title status updated), progress (§16a, §20), results summary (§11), decisions (D22, H7), roadmap, results registry, `results/repe/README.md`, manifest (+1 row) | **R2 validation stage (descriptive).** Script 17 run unchanged from `b232ce4`; no test rows read; no checkpoints created (registry unchanged). Verdict NOT SUPPORTED for all architectures and k |

## Split and dataset changes

| Date | Change | Effect |
|---|---|---|
| ~10-03 (legacy) | GoT cities, city-grouped 70/30, seed 42 | Layer-10 legacy pipeline only |
| 10-30 | `data/got/splits/*_splits.csv`: grouped 70/15/15 (city / number pair) | All GoT layer-6 and all-layer work |
| 11-03 | `data/repe/facts_splits.csv` / `persona_pairs.csv`: statement-grouped 417/90/90 + cross-wording set | All RepE work |
| 11-07 | `data/got/splits/train_AB_groups_truth_aware_pilot.csv`: GoT **train** groups split A/B (geographic 261/262 groups, numerical 346/347), balanced | A = fine-tuning and teachers; B = independent evaluation probes. Validation and test unchanged |
| 11-07 | `data/repe/splits/r1_train_AB_groups.csv`: RepE **train** statements split A/B (208/209, normalised-statement groups, seed 20261107, label-balanced) | A = R1 fine-tuning; B = R1 independent evaluation probes. Validation/test unchanged; validation statement 326 excluded from R1/R2 scores |

## Not committed (by rule)
* `data/activations/` and the `data/*/activations/` caches.
* All `.pt` checkpoints (see `docs/checkpoint_registry.csv` for paths and SHA-256).
* Logs (`results/*/logs/`).
* `scripts/geometry_of_truth/07_independent_truth_probe.py.bak` (identical to `0d07739`) and `scripts/openwebtext/10_reconstruction_evaluation.py.bak`.
* `results/professor_review_corrections/` and `scripts/professor_review/` were kept for provenance until 2026-11-07, then retired from the working tree (see the repository-consolidation entry below). Their tracked files remain in Git history; the full folders, including the gitignored checkpoints and logs, are in `../backups/2026-11-07_retired_professor_review_folders/` on the EC2 host.
* The two `.bak` files and `data/activations/` are now also listed in `.gitignore` (2026-11-07); the files themselves were not touched.

## 2026-11-08: M0/M1 pre-registration (D23)

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `scripts/shared/m0_within_layer_reanalysis.py`, `scripts/repe/{20_build_repe_lm_dataset,21_extract_repe_lm_activations,22_repe_lm_gates}.py`, `results/shared/m0_within_layer_reanalysis/{README.md,preregistration.json}`, `results/repe/repe_lm_gates/{README.md,preregistration.json}` | M0/M1 pre-registered before running (hashes frozen) |
| this commit | `docs/RESEARCH_DECISIONS.md` D23; `RESEARCH_ROADMAP.md` §2; `RESEARCH_PROGRESS.md` §21; `results_registry.md` HISTORICAL reclassification | Reframing to a measurement study; old RepE, R1, R2, 6c and GoT numerical are HISTORICAL |

## 2026-11-08: M0 executed

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `results/shared/m0_within_layer_reanalysis/{merged_validation_table,within_layer_regressions,within_layer_spearman,architecture_by_layer,architecture_pairwise_l0_adjusted,l0_elasticity_from_sweep,pooled_vs_within_corrections}.csv`, `splice_ratio_range.json`, `summary.json`; README §3 | M0 results (validation reanalysis) |
| this commit | `docs/RESEARCH_PROGRESS.md` §22, `EXPERIMENT_RESULTS_SUMMARY.md` §12, `RESEARCH_ROADMAP.md` §2, `results_registry.md`, `experiment_manifest.csv` | Documentation of M0 |

## 2026-11-08: M1 executed

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `data/repe/repe_lm/{prompts.csv,templates.json,split_verification.json}`; `results/repe/repe_lm_gates/{gates.json,k1_activation_invariants.json,k2_k3_bow_and_shortcuts.csv,probe_auroc_by_layer.csv,k5_lopo_by_template.csv,k7_behaviour_rows.csv,k7_behaviour_summary.csv}`; README §4 | M1 results. Activations (712 MB) in `data/repe/activations/repe_lm/`, gitignored |
| this commit | `docs/RESEARCH_PROGRESS.md` §23, `EXPERIMENT_RESULTS_SUMMARY.md` §13, `RESEARCH_ROADMAP.md` §2, `results_registry.md`, `experiment_manifest.csv` | Documentation of M1 |

## 2026-11-08: M2/M3 pre-registration

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `src/evaluation/{information_levels,patching}.py`, `src/tests/test_{information_levels,patching}.py`, `scripts/shared/m2_four_level_faithfulness.py`, `scripts/repe/23_positive_control_site_sweep.py`, `results/shared/m2_four_level_faithfulness/{README.md,preregistration.json}`, `results/repe/causal_site_sweep/{README.md,preregistration.json}`; progress §24; roadmap §2 | M2/M3 pre-registered before running (hashes frozen) |

## 2026-11-08: M2 executed

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `results/shared/m2_four_level_faithfulness/{per_cell,original_by_layer,pca_reference,h2a_h2b,summary_by_layer_arch,persona_vs_neutral,sparsity_width_within_layer}.csv`, `checks.json`, `bootstrap_arrays.npz`; README §2; progress §25; results summary §14; roadmap; registry; manifest | M2 results |

## 2026-11-08: M3 executed

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `results/repe/causal_site_sweep/{cells_train,cells_train_by_template,cells_validation,cells_validation_by_template}.csv`, `selected_sites.json`, `sanity_checks.json`, `random_draws.npz`; README §2; progress §26; results summary §15; decisions D24; roadmap; registry; manifest | M3 results |

## 2026-11-08: M4 pre-registration (amendment A1)

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `scripts/repe/24_sae_patch_decomposition.py`, `results/repe/sae_patch_decomposition/{README.md,preregistration.json}`; D25; roadmap; progress §27 | M4 pre-registered (hashes frozen) before running |

## 2026-11-08: M4 executed

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `results/repe/sae_patch_decomposition/{mid_site_selection.json,per_pair_validation.csv,decomposition_summary.csv,features_summary.csv,selected_features.json,hypotheses.csv,architecture_summary.csv,sanity_checks.json,control_draws.npz}`; README §2; progress §28; results summary §16; roadmap; registry; manifest | M4 results |

## 2026-11-08: README rewritten after Instructor Review #8

| Commit | Change | Reason |
|---|---|---|
| this commit | M `README.md` (rewritten: four faithfulness levels, sub-questions SQ1–SQ5, verified dataset sizes and SAE specification, M0–M4 findings, corrected earlier claims, historical results separated, corrections table, planned work, progress table, test-split exposure, limitations, checklist); M `docs/RESEARCH_ROADMAP.md` (Review #8 response programme: IMP-L, E1–E3, paper, repository items, push/PR) | Instructor Review #8 (2026-11-08); the previous README still cited superseded claims (pooled ρ 0.86, "1.0–2.6× random", frozen-probe "near chance") and did not mention M0–M4 |

## 2026-11-09: paper skeleton and IMP-L pre-registration

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `research_paper/Latex/{main.tex,references.bib,sections/*.tex,tables/*.tex}`, `research_paper/Latex_Fig/*.{pdf,svg}`, `scripts/shared/build_paper_artifacts.py` (tables and figures generated from committed result CSVs) | Review #8 blocker: paper skeleton; F7: vector figures. Bibliography = existing `reports/Latex_report/mybib.bib` + AxBench/SAEBench/Paulo et al. from the `origin/dhruv` review + Braun et al. 2024 |
| this commit | Added `scripts/geometry_of_truth/22_multilayer_finetune.py`, `23_multilayer_evaluation.py`, `results/geometry_of_truth/multilayer_finetune/{README.md,preregistration.json,pretraining_checks.json,alpha.json,pretraining_hashes.json}`; progress §29 | Review #8 blocker: improvement result at more layers. Pre-registered before any training run |
| this commit | M status line (line 3) of the M0, M1, M2 and M3 result READMEs: PRE-REGISTERED → COMPLETED | Stale headers found in the 2026-11-08 audit; bodies already reported completed results |

## 2026-11-09: IMP-L results (layers 3, 8, 9) and paper update

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `results/geometry_of_truth/multilayer_finetune/{per_sae_layer3,per_sae_layer8,per_sae_layer9,summary,comparisons,improvement_table}.csv`, `criteria.json`, `eval_info_layers3_8_9.json`, `training_info_layer{3,8,9}.json`, `training_monitor_layer{3,8,9}.csv`; README §2 | IMP-L results (approved 2026-11-09). 81 checkpoints written to the gitignored `sae_finetuning/multilayer_finetune/`; logs not committed |
| this commit | M `scripts/shared/build_paper_artifacts.py`: IMP-L tables (`t_impl_summary`, `t_impl_layer{3,8,9}`) and figure (`fig_impl_multilayer`); deterministic PDF/SVG output (CreationDate/Date metadata removed, fixed `svg.hashsalt`) | Paper results; reviewers can regenerate the figures and diff them. Regenerated M0–M4 figures differ from the previous commit only in metadata and SVG ids |
| this commit | M `research_paper/Latex/sections/{00_abstract,10_finetuning,12_discussion,15_appendix}.tex`; added `tables/t_impl_*.tex`, `Latex_Fig/fig_impl_multilayer.{pdf,svg}` | IMP-L section written from verified results |
| this commit | M `README.md` (§6.8 IMP-L table, §8/§9 statuses, limitations), `docs/RESEARCH_PROGRESS.md` §29 status and §30, `docs/RESEARCH_ROADMAP.md`, `docs/EXPERIMENT_RESULTS_SUMMARY.md` §17, `docs/results_registry.md`, `docs/RESEARCH_DECISIONS.md` (D26; H3/H4), `docs/experiment_manifest.csv`, `docs/checkpoint_registry.csv` | Documentation rule (CLAUDE.md §1) |

## 2026-11-09: paper refinement, E3/E2/E1 draft pre-registrations, review package

| Commit | Change | Reason |
|---|---|---|
| this commit | M `research_paper/Latex/sections/{00_abstract,05_setup,07_results_information,09_results_causal,10_finetuning,12_discussion,13_limitations,14_conclusion}.tex`, `main.pdf` | User's reading of IMP-L (D27): keep frozen-probe, re-trained-probe and behavioural results separate; no truth-recovery claim; VALIDATION labels throughout |
| this commit | Added `results/repe/e3_finetuned_causal_preservation/README.md`, `results/shared/e2_public_sae_reference/README.md`, `results/shared/e1_task_l0_matched_topk/README.md` (draft pre-registrations, not run) | User asked to see hypotheses, comparisons, metrics, GPU and training requirements before any run |
| this commit | Added `docs/2026-11-09_review_package_sae_research.md` | Independent teammate review before merging to `main` |
| this commit | M `README.md` §8/§9, `docs/RESEARCH_ROADMAP.md`, `docs/RESEARCH_PROGRESS.md` §31, `docs/RESEARCH_DECISIONS.md` D27 | Documentation rule |

## 2026-11-10: E3 and RLM-RP (RepE completion), paper Phase 5 fixes

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `scripts/repe/25_e3_finetuned_causal_preservation.py`; `results/repe/e3_finetuned_causal_preservation/{preregistration.json, per_pair_validation.csv, decomposition_summary.csv, features_summary.csv, hypotheses.csv, site_reconstruction.csv, comparisons.csv, criteria.json, sanity_checks.json, selected_features.json, control_draws.npz}`; M `README.md` (status, §1.8 implementation details, §2 results) | E3 approved by the user 2026-11-10; the 2026-11-09 draft criteria are unchanged |
| this commit | Added `scripts/repe/26_repe_lm_representation_preservation.py`; `results/repe/repe_lm_representation_preservation/` (README, preregistration, `per_sae.csv`, `original_by_layer.csv`, `summary_by_layer_arch.csv`, `comparisons.csv`, `criteria.json`, `checks.json`) | RepE-LM representation preservation and robustness for 216 SAEs (user request 2026-11-10) |
| this commit | M `scripts/shared/build_paper_artifacts.py`: tables `t_impl_l0`, `t_impl_readers`, `t_rlm_layers`, `t_rlm_finetune`, `t_e3`, `t_e3_classes`; figures `fig_rlm_layers`, `fig_e3_preservation`; breakable source paths in captions | Phase 5 reporting gaps (L0 per dataset; frozen MLP vs frozen logistic vs re-trained readers), and the new results |
| this commit | M `research_paper/Latex/main.tex` (`\emergencystretch`), `sections/{00_abstract,04_questions,07_results_information,09_results_causal,10_finetuning,11_statistics,12_discussion,13_limitations,14_conclusion,15_appendix}.tex`, `main.pdf`; regenerated `tables/*.tex` (caption path breaks only for existing tables) | Paper integration; the ambiguous 5/9 wording corrected; overfull boxes removed |
| this commit | M `docs/{RESEARCH_PROGRESS,RESEARCH_ROADMAP,RESEARCH_DECISIONS,EXPERIMENT_RESULTS_SUMMARY,results_registry}.md`, `docs/experiment_manifest.csv`, `results/repe/README.md`, `README.md` | Documentation rule. No checkpoints were created, so `checkpoint_registry.csv` is unchanged |

## 2026-11-10: independent research review (no experiment)

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `docs/2026-11-10_independent_research_review.md`; M `docs/RESEARCH_ROADMAP.md` (R1–R7 proposed), `docs/RESEARCH_PROGRESS.md` §33 | User asked for an independent assessment and improvement strategy; newly identified tasks go on the roadmap (CLAUDE.md §1) |


## 2026-11-10: GPT-2 improvement phase (R2, R4, R5, E2, M5)

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `src/sae/modern_sae.py` (TopK/BatchTopK/ReLU/JumpReLU/frozen-decoder module, `modern_v1` checkpoint format); M `src/evaluation/faithfulness_metrics.py` (`load_sae` dispatches `modern_v1`) | R2 recipe |
| this commit | Added `scripts/openwebtext/17_streaming_sae_training.py`; `results/openwebtext/r2_scaled_saes/` (README, preregistration, training_info, training_monitor, ladder/, comparisons.csv, criteria.json, excluded_unmatched.csv) | R2 data-scaled retraining (39 SAEs; checkpoints gitignored) |
| this commit | Added `scripts/shared/sae_ladder_evaluation.py`, `scripts/shared/r2_comparisons.py` | One evaluation ladder for all SAE families; pre-registered R2 rule |
| this commit | Added `scripts/openwebtext/18_kl_finetune_fixed_l0.py`, `scripts/openwebtext/19_r4_analysis.py`; `results/openwebtext/r4_kl_finetune/` (README, preregistration, finetune_info, finetune_monitor, ladder/, comparisons.csv, criteria.json) | R4 KL fine-tune at fixed L0 (12 checkpoints gitignored) |
| this commit | Added `scripts/repe/27_causal_subspace_skyline.py`; `results/repe/r5_causal_subspace_skyline/` | R5 skyline and random floor |
| this commit | Added `scripts/shared/e2_public_sae_reference.py`; `results/shared/e2_public_sae_reference/{checks.json, hypotheses.csv, ladder/}`; M its README (status, §1.7 clarifications, §2 results) | E2 approved and run |
| this commit | Added `scripts/shared/m5_confirmatory_test.py`; `results/shared/m5_confirmatory_test/` (README, preregistration, m5_rows.csv, comparisons.csv, checks.json) | M5 single confirmatory TEST pass |
| this commit | M `.gitignore` (`data/openwebtext/stream_shards/`, `stream_tokens/`) | Downloaded OWT shards 78/79 and tokenised chunks are not committed |
| this commit | M `scripts/shared/build_paper_artifacts.py` (tables `t_r2_c1/c2/c3/counts`, `t_r4`, `t_r5`, `t_e2`, `t_m5`, `t_m5_counts`); added `research_paper/Latex/sections/10b_improvement.tex`; M `main.tex`, `00_abstract.tex`, `14_conclusion.tex`, `references.bib`, `main.pdf` | Paper before/after section |
| this commit | M `docs/{RESEARCH_PROGRESS,RESEARCH_ROADMAP,RESEARCH_DECISIONS,EXPERIMENT_RESULTS_SUMMARY,EXPERIMENT_CHANGELOG,results_registry}.md`, `docs/experiment_manifest.csv`, `docs/checkpoint_registry.csv` | Documentation rule |

## 2026-11-10: final defense verification (no experiment)

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `docs/final_defense/{FINAL_RESEARCH_REPORT,PROFESSOR_DEFENSE_CHEATSHEET,RESULTS_VERIFICATION_TABLE,PROFESSOR_QUESTIONS_AND_ANSWERS,EXPERIMENT_AUDIT}.md` | User-requested verified report and defense material |
| this commit | Added `scripts/shared/m5_independent_verification.py` (read-only) | Reproducible independent re-check of M5: hashes, completeness, decision rules |
| this commit | M `docs/{RESEARCH_PROGRESS,RESEARCH_ROADMAP,RESEARCH_DECISIONS,results_registry}.md` | Documentation rule. No results, checkpoints or manifests changed |

## 2026-11-12: GPT-2 audit and improvement plan; checkpoint of current work (no experiment)

| Commit | Change | Reason |
|---|---|---|
| this commit | Added `docs/2026-11-12_gpt2_audit_and_improvement_plan.md` | Read-only audit and pre-registration drafts (X0–X3, T1, H1, H2, I1); none approved or run |
| this commit | Added `docs/final_defense/SAE_Faithfulness_Summary.pdf` | User's summary PDF of the final-defense material, committed at the user's request |
| this commit | M `docs/{RESEARCH_PROGRESS,RESEARCH_ROADMAP,EXPERIMENT_CHANGELOG}.md` | Documentation rule. No results, checkpoints or manifests changed |

## 2026-11-12: documentation cleanup and consolidation (no experiment)

| Commit | Change | Reason |
|---|---|---|
| `e2becfa` | Added `docs/experiment_history.md`, `docs/results_summary.md`. **Also contains** the three byte-identical renames below: `git mv` had already staged them, so they landed one commit earlier than intended; not amended, per the no-history-rewrite rule | Authoritative chronology and verified results by dataset |
| `17762d7` | Added `docs/research_overview.md`, `docs/methodology.md`, `docs/limitations_and_future_work.md` | Curated entry points; corrections table; proposals listed as not run |
| `768831a` | Renamed (R100): `docs/2026-11-04_professor_review_report.md` → `docs/archive/2026-11-04_methodology_review_corrections_report.md`; `docs/final_defense/PROFESSOR_DEFENSE_CHEATSHEET.md` → `DEFENSE_CHEATSHEET.md`; `docs/final_defense/PROFESSOR_QUESTIONS_AND_ANSWERS.md` → `ANTICIPATED_REVIEW_QUESTIONS.md`. Updated links in `docs/{README,RESEARCH_PROGRESS}.md`, `docs/final_defense/*.md` and the 2026-11-12 audit; added `docs/archive/README.md`; appended 3 `git_mv_rename` rows to `docs/migration_manifest.csv`; erratum in the 2026-11-12 audit (contrast distortion first reported by RLM-RP) | Neutral names; provenance kept |
| this commit | M `README.md`: navigation table, three-pipeline summary, §6.0 latest results, status of E2/R2/R4/R5/M5, verified improvement-phase commands in §13, checklist. M `docs/{RESEARCH_PROGRESS,RESEARCH_ROADMAP,EXPERIMENT_CHANGELOG}.md` | README was dated 2026-11-08 and listed M5 as not started |

Not changed: any script, result file, checkpoint, activation cache, manifest row (other than the appended renames),
Gemma file or teammate file. Historical records that cite the old paths (`EXPERIMENT_CHANGELOG.md` 2026-11-10 entry,
`migration_manifest.csv` older rows, the committed summary PDF) are left as they are.

## 2026-11-13: audit remediation

| Commit | Change | Reason |
|---|---|---|
| `7e8664f` | Added `scripts/shared/pca_capacity_matched.py` and `results/shared/pca_capacity_matched/{README.md,preregistration.json}` | S1: the PCA reference was matched on error only (k = 298–519 vs 143–164). Protocol committed before the run finished (D31) |
| `7e8664f` | M `docs/{RESEARCH_PROGRESS,RESEARCH_ROADMAP,EXPERIMENT_CHANGELOG}.md` | Documentation rule |
| `bf5865a` | M `docs/{research_overview,results_summary,limitations_and_future_work}.md` and errata in `docs/2026-11-12_gpt2_audit_and_improvement_plan.md` | D1: `got_cities` pools cities + neg_cities (2,992 statements; test 226 + 226 = 452); it is not cities-only. D2: the M5 C1–C3 NMSE improvement range is 30.8–73.7 % (was "40–74 %"); ΔCE 35.5–78.8 %. Verified against `m2_four_level_faithfulness.py:66–69`, `03_original_truth_probes.py:24`, the split CSVs and M5 `comparisons.csv`. Historical audit text is corrected by marked errata, not rewritten |
| `bf5865a` | M `research_paper/Latex/references.bib` | Cunningham et al. appeared at ICLR 2024 (was 2023); Korznikov et al. full author list (checked on arXiv 2602.14111) |
| this commit | M `reports/Latex_report/mybib.bib` (the same Cunningham fix, plus 10 entries copied from the paper bib: AxBench, SAEBench, Paulo et al., BatchTopK, Braun et al., Karvonen, res-jb, Korznikov et al., Engels et al., Chaudhary and Geiger; Review #8 NOVELTY-RISK) | It was meant for `bf5865a`, but a failed shell step left it unstaged. Committed separately; history not rewritten |
| this commit | Added `src/tests/test_modern_sae.py` and `src/tests/test_decision_rules_and_splits.py` (34 tests; suite 35 → 69); M `src/tests/README.md` | D8: core logic had no tests |
| this commit | Added `.github/workflows/gpt2-validation.yml` and `scripts/shared/ci_repo_hygiene.py` | D6: no CI. CPU only: hygiene (no checkpoints, caches, logs, credentials or home paths), py_compile of all tracked Python, import check, unit tests, M5 independent verification. Runs on PRs to `main` and pushes to `main`/`sae-research` |
| this commit | M `requirements.txt` (+`joblib==1.6.0`, +`scipy==1.18.1`, at their alphabetical places) | Imported directly by M2/M5/r2_comparisons but unpinned; versions from the research venv. Placed away from PR #28's `requirements.txt` hunk |
| this commit | Added `scripts/repe/28_build_repe_lm_test_split.py`, `data/repe/repe_lm_test/{prompts.csv,templates.json,split_verification.json}`, `results/repe/repe_lm_test_protocol/{README.md,preregistration.json}` | S3: no RepE-LM test split. Data and protocol only; nothing evaluated (D32) |
| this commit | M `research_paper/Latex/sections/09_results_causal.tex` | S5: the equivalence argument; the paragraph is renamed "SAE-selected feature patches (error term kept)" (D33) |
| this commit | M `docs/{RESEARCH_DECISIONS,RESEARCH_ROADMAP,results_registry,EXPERIMENT_CHANGELOG}.md` | Documentation rule |
| this commit | M `reports/Latex_report/main.tex`: the five TODO sections (Introduction, Methods, Experiments and Results, Discussion, Conclusion) written from verified GPT-2 results | D9 / Review #8: the report had 5 TODO sections. Gemma subsection marked PENDING for teammates; the September title/abstract framing is flagged for a team decision, not changed. Builds out-of-tree: 5 pp, 0 undefined citations |
| this commit | Added `scripts/shared/export_vector_figures.py` and 42 files in `results/{geometry_of_truth,openwebtext,repe,repe/steering/causal_random_controls,shared/spliced_evaluation}/figures/vector/` (21 figures × SVG + PDF) | D10 / Review F7: results figures were PNG only. Figure scripts run unchanged with CSV/text writes blocked and savefig redirected. Validation: a PNG re-render of all 21 figures is **pixel-identical** to the committed PNGs (max abs difference 0), so values are unchanged; no tracked file modified; PNGs kept for provenance. 31 PNGs remain without a safely re-runnable script (EDA, training curves, `layer_comparison`, Gemma, diagram images) |
| this commit | Added `architecture/diagrams/{build_gpt2_pipeline_diagram.py, gpt2_sae_faithfulness_pipeline.drawio, gpt2_sae_faithfulness_pipeline.svg}` | Review F8: no editable diagram source. Both files are generated from one node list; every box names its source function (tooltip) |
| this commit | Added `docs/remediation/{issue_register,professor_review_compliance,merge_plan,repe_validity_assessment,backup_plan}.md` | Phases 1, 5B/C, 7, 9, 10 of the remediation: register of D/S/U items with evidence and statuses; compliance against Reviews #1–#8; verified PR overlaps and proposed order (not posted); probe and causal validity from existing outputs; backup inventory and copy-only command (no destination configured) |
| this commit | Added `reports/Progress_Report/2026-11-13_gpt2_retrospective.md` (labelled retrospective; every row cites the journal) and `presentation/gpt2_results_slides.md` (draft content; team integration pending) | Review #8 H1-STALE, NO-SLIDES; nothing fabricated or claimed as submitted |
| this commit | M `README.md` §9 and checklist | D7: stale rows (paper 29 → 37 pp, report, figures, CI, PR list, `main` status) |
| this commit | Added `results/shared/pca_capacity_matched/{rows,got_paired,decisions}.csv`, `checks.json`; M its README (§2 results) | PCA-CM run (VALIDATION, 0 test rows). Pre-registered I2 triggered: a conclusion change, reported to the user; paper not revised (D34) |
| this commit | Added `scripts/shared/power_sensitivity_analysis.py`, `results/shared/power_sensitivity/` | S2/S7: sensitivity of the GoT and M4 comparisons from committed outputs |
| this commit | Added `docs/remediation/review8_response_draft.md` (NOT POSTED); M `docs/remediation/issue_register.md`, the 2026-11-13 ledger entries, `docs/experiment_manifest.csv` (+2 rows via `append_experiment`) | Documentation rule; drafts for the user |
| `ee7b7ac` | (committed before the user's review; pushed with the next commit) PCA-CM results and sensitivity analysis | See the row above; held locally until the user approved D34 |
| this commit | Added `scripts/shared/pca_cm_posthoc_multiplicity.py`, `results/shared/pca_capacity_matched/posthoc_multiplicity/{cells.csv,summary.json}`, `research_paper/Latex/tables/t_pcacm.tex` | The user asked whether "significantly better" is supported: it is not (per-cell rule; approx. Holm keeps 1/3 and 4/8) |
| this commit | M `research_paper/Latex/sections/{00_abstract,04_questions,07_results_information,09_results_causal,10b_improvement,11_statistics,12_discussion,13_limitations,14_conclusion}.tex`, `research_paper/Latex/main.pdf` (rebuilt, 40 pp) | D34 (approved wording), D35 (MDE bound, L8 borderline); stale M5/test-split statements corrected |
| this commit | M `README.md`, `docs/{research_overview,results_summary,limitations_and_future_work,methodology}.md`, `docs/final_defense/{DEFENSE_CHEATSHEET,ANTICIPATED_REVIEW_QUESTIONS,FINAL_RESEARCH_REPORT,RESULTS_VERIFICATION_TABLE}.md`, `reports/Latex_report/main.tex`, `presentation/gpt2_results_slides.md`, `docs/remediation/{issue_register,professor_review_compliance,review8_response_draft}.md` | Same wording in every curated document; error-matched results kept alongside |
| this commit | M `results/shared/pca_capacity_matched/README.md` (§2.5 significance check, §2.6 status), `docs/{RESEARCH_PROGRESS,RESEARCH_DECISIONS,RESEARCH_ROADMAP,results_registry,EXPERIMENT_RESULTS_SUMMARY}.md`, `docs/experiment_manifest.csv` (+1 row via `append_experiment`) | Documentation rule |
| this commit | Added `src/tests/test_pca_cm_posthoc.py` (2 tests; suite 69 → 71); M `src/tests/README.md` | Covers the new post-hoc Holm check and checks that it reproduces the pre-registered PCA-CM classes |
| this commit | Added `scripts/repe/29_repe_lm_test_evaluation.py`, `results/repe/repe_lm_test_protocol/{preregistration_addendum.json,checkpoint_hashes.csv,smoke_validation/}`; M its README | H1-TEST implementation freeze before the single TEST pass (user-authorised); smoke test on validation reproduces the ladder to ≤ 1e-4 |

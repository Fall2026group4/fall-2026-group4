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

## 2026-11-12: `simba-gemma-pile` (Simbanegavi Simbarashe)

| Commit | Files (A = added, M = modified) | Change, reason, effects |
|---|---|---|
| `b39dcc2` | A `results/gemma2_pile/gemma_scope_splicing/README.md`, `preregistration.json`; M `docs/RESEARCH_ROADMAP.md` (M6 added, M3 status, §2 in progress), `docs/RESEARCH_PROGRESS.md` (§19; later §43; now §51), `docs/RESEARCH_DECISIONS.md` (D21; renumbered D36 when merged with #27, D44 on 2026-11-17; pointers in D2 and H6), `docs/EXPERIMENT_CHANGELOG.md` | **Documentation only: pre-registration of M6** (Gemma Scope SAE reconstruction and splicing on The Pile) before any code or run. Branch created from `b64e6ed` (`simba-method-a-multilayer`, left unchanged). No script, result or checkpoint added or modified. Data deviation (approved): the EDA used the train-side parquet mirror; M6 uses the pinned original validation file, held out from the EDA. **Tests not run:** documentation/configuration only (no executable code changed) and no local venv on this machine |
| `fa6f2ce` | M `results/gemma2_pile/gemma_scope_splicing/preregistration.json` (`tokenization.revision` added; `model.revision` pinned; `amendments` list), `README.md` (Amendment 1 section; two rows annotated), `docs/RESEARCH_PROGRESS.md` (§19; later §43; now §51), `docs/RESEARCH_DECISIONS.md` (D21; later D36; now D44), `docs/EXPERIMENT_CHANGELOG.md` | **Pre-registration Amendment 1, before any run:** Gemma-2-2B tokenizer and model pinned to HF commit `c5ebcd40…` (was "recorded at run time"). Documentation/configuration only; tests not run (no executable code changed; no local venv). Earlier commit `b39dcc2` not rewritten (row's "(this commit)" replaced by `fa6f2ce` on 2026-11-12) |
| `39ed943` | A `scripts/gemma/02_pile_sample_manifest.py`, `src/evaluation/pile_splicing.py` (sampling part), `src/tests/test_pile_sampling.py`; M `requirements.txt` (`zstandard==0.23.0`) | Deterministic Pile sampling pipeline (seed-42 permutation, ≥ 32 text tokens, 512 + 64 disjoint documents, BOS + first 1023 text tokens). 40 tests passed on Ronin (user report) |
| `a34d975` | A `results/gemma2_pile/gemma_scope_splicing/sample_manifest.csv`, `sample_info.json` | Pre-registered evaluation sample (hashes only, no document text); gate V1 inputs |
| `4708213`, `606a43d`, `13eaeca` (TEMP) | A `scripts/gemma/03_pile_gemma_scope_splicing.py`, `src/tests/test_gemma_pile_splicing.py`; M `src/evaluation/pile_splicing.py`, `src/tests/test_pile_sampling.py`, experiment `README.md` (§2 implementation details, safeguards A1–A11, H4a limitation) | Splicing script and tests; engineering safeguards A1–A11 (numerics, git provenance, determinism, atomic outputs, hardware record); A3 strict deterministic algorithms. Engineering only: no change to layers, SAEs, thresholds, controls, data, metrics or criteria. Made as TEMP commits for the Ronin tests; the full run used `13eaeca` |
| `0dbfc80` | A `results/gemma2_pile/gemma_scope_splicing/{per_doc_metrics,summary_by_layer,summary_by_layer_source,controls,h_g1}.csv`, `gates.json`, `run_info.json` | **Full pre-registered M6 run** (Ronin, Tesla T4, 2.24 h). Gates V1–V5 pass; H-G1 supported at layers 5, 12, 19. Output SHA-256 recorded in `run_info.json` |
| `28a7d85` | A `scripts/gemma/04_pile_gemma_scope_figures.py`, `results/gemma2_pile/gemma_scope_splicing/figures/` (4 PNG + `figure_info.json`); M experiment `README.md` (status line; §2 Results), `docs/RESEARCH_PROGRESS.md` (§19; later §43; now §51), `docs/RESEARCH_ROADMAP.md` (M6 DONE, M3), `docs/EXPERIMENT_RESULTS_SUMMARY.md` (§10; now §25), `docs/results_registry.md`, `docs/experiment_manifest.csv` (M6 row), `docs/EXPERIMENT_CHANGELOG.md` | **M6 results documented; status COMPLETED, VALIDATION.** Figures are read from the saved tables only: no model, SAE or GPU. The script checks input hashes against `run_info.json`, the gates and the H-G1 recomputation, and refuses to overwrite. Figures 1–2 use zero-based axes; figure 4 axes are zoomed and labelled as such. `preregistration.json` unchanged (its PLANNED status is historical). No result value changed. Post-run verification on Ronin (after the full run, M6 code/results branch): `python -m pytest -q -rs src/tests` 85 passed, 0 failed, 0 skipped |
| `cde778e` (M6 evidence bundle, from Ronin) | A `results/gemma2_pile/gemma_scope_splicing/per_doc_sum_x.npz` (must match SHA-256 `3b2becfb…` in `run_info.json`), `results/gemma2_pile/gemma_scope_splicing/smoke_hardened/` (hardened smoke run: H4a logit-precision values, sae-lens diagnostic, binding V4 check; labelled SMOKE) | Final M6 evidence artifacts, committed separately from the documentation commit. The older `smoke/` folder is not committed |
| `167fadb` | M experiment `README.md` (post-run deviation note), `docs/EXPERIMENT_CHANGELOG.md` (commit ids filled in for `28a7d85` and `cde778e`) | **PR preparation, documentation only.** (1) The README now records that `smoke_hardened/` holds its own small smoke `per_doc_sum_x.npz` (8 smoke documents). It is not used for any result; the full-run file stays authoritative. (2) **Decision: keep the inherited GPT-2 Method A multi-layer scripts** (`scripts/geometry_of_truth/22_method_a_multilayer_finetune.py`, `23_method_a_multilayer_evaluation.py`; from the parent branch `simba-method-a-multilayer`, commits `28ad5cf` and `b64e6ed`) in this branch. A trial deletion showed why: if M6 is merged first, a later merge of `simba-method-a-multilayer` would give a visible conflict for script 23 but would **silently drop script 22**, because `7f2557b` modifies only script 23. They are disclosed as unrelated to M6 in the PR. No M6 code, result, dependency or provenance changed (`run_info.json` still points to `13eaeca`); `preregistration.json` unchanged; no history rewritten |
| (merge of `origin/main` @ `17476c2` into #28; uncommitted at the time of writing) | M `docs/{EXPERIMENT_CHANGELOG,EXPERIMENT_RESULTS_SUMMARY,RESEARCH_DECISIONS,RESEARCH_PROGRESS,RESEARCH_ROADMAP}.md`, `docs/experiment_manifest.csv`, `docs/results_registry.md` (conflict resolution only) | **Integration of M6 with the merged #27 documentation; no result value changed.** #27 content kept in full. M6 renumbered to avoid collisions with #27: decision D21 → **D36**, progress §19 → **§43**, results-summary §10 → **§25** (roadmap Gemma task ID M6 kept; the GPT-2 "M6 test pass" item of the 2026-11-12 audit plan is a different, planned item and is labelled as such). Exactly one M6 manifest row and one registry row. The R3 rows were removed from #28 in `2bab391`; R3 lives in PR #30 (`dhruv-gemma-r3`) and is listed only as PRE-REGISTERED, not run. The inherited GPT-2 Method A scripts 22/23 (layers 5/6/9) are kept (see `167fadb`); they are distinct from #27's IMP-L scripts `22_multilayer_finetune.py` / `23_multilayer_evaluation.py` and write to a different results folder |
| `2118b3c` (originally `7130f42`; Dhruv Rai) | A `results/gemma2_pile/r3_gemma_scope_ladder/README.md` | **R3 pre-registration (draft for team review; not run).** Width × L0 ladder of 54 Gemma Scope SAEs on M6's sample; at the time also an A10G rerun of M6 (Q0) and an fp32 reference (Q0b). The row above marked "(this commit)" is `167fadb` |
| `b82e871` (originally `8c7176b`; Dhruv Rai) | A `results/gemma2_pile/r3_gemma_scope_ladder/literature_comparison.md` | M6 numbers compared with published Gemma Scope, JumpReLU, SAEBench, TopK and end-to-end SAE results. Figure-read values are labelled as approximate. Context only; no result changed |
| `8ed8102` (originally `4fd2706`; Dhruv Rai) | M R3 `README.md` (Amendment 1; Section 1 not edited) | **R3 Amendment 1, before any code or run:** hardware AWS A10G → Kaggle T4 x2 (the AWS free plan has no GPU instances). Q0 and the M6 rerun are dropped; Q0b becomes T4 bf16 vs fp32; T4 memory placement; Kaggle runs K0–K6; per-SAE outputs with resume and gate V9; `HF_TOKEN` from Kaggle Secrets and gate V10. D22; now D46 |
| `07e3c8f` (originally `24f6261`; Dhruv Rai) | A `results/gemma2_pile/r3_gemma_scope_ladder/preregistration.json`, `scripts/gemma/05_pile_gemma_scope_ladder.py`, `scripts/gemma/06_r3_precision_comparison.py`, `scripts/gemma/07_r3_figures.py`, `src/tests/test_gemma_r3_ladder.py`, `notebooks/r3_kaggle_runner.ipynb`; M R3 `README.md` (Clarification 1; status line; Section 2 implementation details and tests; Section 3 paused status), `docs/RESEARCH_PROGRESS.md` (§20), `docs/RESEARCH_DECISIONS.md` (D22; now D46; note on hypothesis H1), `docs/RESEARCH_ROADMAP.md` (R3 in progress; R3b planned), `docs/EXPERIMENT_CHANGELOG.md` | **R3 implemented, not run.** Ladder engine, fp32 reference, runs K0–K6, resume with gate V9 (pass within relative difference 1e-6; bit-identity recorded separately), gates V0–V10, the CPU aggregate step (L0 bands, H1–H3 with Holm, V6 against M6's committed rows), Q0b, figures, and a Kaggle runner with the V10 secret scan. M6 code is reused unchanged: `pile_splicing.py`, script 03 `summarize()`, script 04 style. **Clarification 1:** V6's active-latent agreement compares chunked and unchunked SAEs on the same activations. Tests on local Windows CPU: R3 51 passed; full suite 128 passed, with 2 failures and 5 errors that were present before R3 (environment-specific). Dry runs on fake data built from M6's rows only; no GPU step executed. No M6 file, result or dependency changed; `results_registry.md` and the experiment manifest are unchanged until there are results |
| `961dc60` (Dhruv Rai) | M `notebooks/r3_kaggle_runner.ipynb` (`BRANCH` and the `COMMIT` comment: `dhruv-gemma-r3`), `docs/RESEARCH_PROGRESS.md` (§20 branch; now §54), `docs/RESEARCH_ROADMAP.md` (R3 branch and PR), `docs/EXPERIMENT_CHANGELOG.md` (the four R3 rows above now show their new hashes, with the originals) | **R3 moved to its own branch at the PR owner's request** (Simbanegavi Simbarashe asked to keep PR #28 to M6 only). No history was rewritten. The new branch `dhruv-gemma-r3` starts at `167fadb`, M6's last commit. The four R3 commits were cherry-picked in order, with their messages unchanged, and each one carries exactly the same change as before (patch IDs checked): `7130f42` → `2118b3c`, `8c7176b` → `b82e871`, `4fd2706` → `8ed8102`, `24f6261` → `07e3c8f`. On `simba-gemma-m6-results` the four originals stay in history and are undone by a single revert commit, so that branch's files again equal `167fadb`. No code, result or test changed; the Kaggle notebook must now be given a commit of `dhruv-gemma-r3` |
| `2d395d0` (Dhruv Rai) | M `notebooks/r3_kaggle_runner.ipynb` (install cell ends with `pip uninstall -y -q torchvision torchaudio`), R3 `README.md` (Section 2 note) | **Kaggle environment fix.** Kaggle's preinstalled torchvision 0.26 (built for torch 2.11) broke transformers' import with the pinned torch 2.7.0 ("operator torchvision::nms does not exist"). R3 uses neither package. Protocol unchanged; torch and transformers stay pinned as in M6 |
| `33a7df3` (Dhruv Rai) | A `reports/Progress_Report/2026-11-14_dhruv_retrospective.md` | Retrospective progress report for Weeks 10–12, with a requirement mapping checked against the roadmap and the professor-review compliance document on `sae-research`. No experiment change |
| `e2b7b3a` (Dhruv Rai) | M `scripts/gemma/05_pile_gemma_scope_ladder.py` (step K0F; run record written in every case; projection from a finished K0 only; fp32 outputs staged and finalised after a clean GPU release), `scripts/gemma/06_r3_precision_comparison.py` (uses only a finished fp32 folder), `src/tests/test_gemma_r3_ladder.py` (7 tests), `notebooks/r3_kaggle_runner.ipynb` (K0F allowed), R3 `README.md` (Clarification 2; Section 2 run attempts; Section 3 status), R3 `preregistration.json` (K0/K0F plan; Clarification 2), `docs/EXPERIMENT_CHANGELOG.md` | **R3 Clarification 2, before any result is read.** The first K0 attempt (`K0_20261115T024202Z`, code `961dc60`) passed V0, V1 and the smoke run (0.562 s per pass), then crashed at the end of the fp32 step with "CUDA error: unspecified launch failure". Its run record was never written, so its timings cannot feed K1. The fp32 reference is now step K0F, with `CUDA_LAUNCH_BLOCKING=1` for that step only. Every step's record is written even on a stop or crash, and only a finished K0 feeds the K1–K6 projection. The fp32 output is declared finished only after the GPUs complete and the model is released; Q0b refuses anything else. Nothing else in the protocol changes. Tests: R3 58 passed; full suite 136 passed (same 2 failures and 5 errors as before, environment-specific) |
| `10286e7` (Dhruv Rai) | M R3 `README.md` (status line; new section "Deviations" after Clarification 2; Section 3), `preregistration.json` (`runs.plan` K3/K4–K6, `runs.tier_b_rule`, new `deviations` list; the `ladder` is unchanged), `docs/RESEARCH_PROGRESS.md` (§54), `docs/RESEARCH_DECISIONS.md` (D46), `docs/RESEARCH_ROADMAP.md` (R3 status) | **R3 Deviations 1–2, recorded before any Tier A result was looked at.** (1) K1 took 7.64 h against 3.11 h projected, so K3 is split into K3a (16k+32k) and K3b (65k) with the existing `--widths` option. (2) Tier B (K4–K6) is cancelled because of the 2026-11-22 freeze and the 30 h weekly quota; H1–H3 are tested on Tier A only. No code changed; R3 tests 58 passed. |
| `268fae3` (Dhruv Rai) | M `reports/Progress_Report/2026-11-14_dhruv_retrospective.md` (new section "In my own words" after the title) | First-person summary of the GoT causal test, the SAE-in-the-loop test, the #28 check, the Gemma Scope comparison and R3 status. Every claim was checked against the repo (including PR #26 head `989d313`); no new result, no number changed. |
| `9f06965` (Dhruv Rai) | M `reports/Progress_Report/2026-11-14_dhruv_retrospective.md` (new sections "Week of 2026-11-08" and "Week of 2026-11-15" at the end) | Weekly entries built from my commits (git log), my approved reviews of #22, #28 and #29 (GitHub), and the R3 run status in the R3 README. Dates and commit hashes only; no result claimed, no number changed. |
| `87fc248` (Dhruv Rai) | M `docs/RESEARCH_ROADMAP.md` (§2 R3 line and table R3 row by Simba: status wording; my R3 bullet and row), `docs/RESEARCH_DECISIONS.md` (D36 follow-up line: status wording; my H1 note), `docs/RESEARCH_PROGRESS.md` (§43 related line: status wording; §54 runs and Amendment 2), R3 `README.md` (Amendment 2; Section 3 run table), `preregistration.json` (`runs.max_hours_per_week` 30 → 45; amendment 2) | (1) **Status wording in Simba's four R3 lines** changed from "PRE-REGISTERED ONLY / NOT RUN" to "ONGOING, NO RESULT CLAIMED", **approved by Simba in his PR #30 comment of 2026-11-16T17:53Z**; nothing else in his text changed. (2) R3 run status: K2 (8.01 h) and K3a (8.85 h) finished, V10 PASS; K3b started; no metric read. (3) **R3 Amendment 2:** weekly GPU cap 30 h → 45 h (Kaggle quota, Colab Pro linked); no SAE, metric or gate change; Tier B stays cancelled; K3b started under script 05's existing 30 h check. (4) K1 run id and V10 PASS added from its Kaggle log. (5) My H1 note in "Unresolved hypotheses" (D46) now says "ongoing, no result claimed". No result claimed. |
| `de58a2e` (Dhruv Rai) | A `results/gemma2_pile/r3_gemma_scope_ladder/ladder/` (11 files: `per_doc_metrics.csv` 6.3 MB, approved by the user; summaries; `l0_bands.csv`; H1/H2/H3 tables; `v6_m6_consistency.csv`; `gates.json`; `run_info.json`), `fp32_reference/` (`per_doc_metrics.csv`, `gates.json`, `run_info.json`), `precision_t4_vs_fp32/` (`q0b_t4_vs_fp32.csv`, `q0b_info.json`), `figures/` (6 SVG + `figure_info.json`), `scripts/gemma/08_r3_vector_figures.py`; M R3 `README.md` (status line; Section 2 results; Section 3), `preregistration.json` (status), `src/tests/test_gemma_r3_ladder.py` (script 08 test), `docs/RESEARCH_PROGRESS.md` (§54), `docs/RESEARCH_ROADMAP.md` (§1 R3 row; my §2 bullet and table row; Simba's R3 lines unchanged), `docs/RESEARCH_DECISIONS.md` (D46 outcome), `docs/results_registry.md`, `docs/EXPERIMENT_RESULTS_SUMMARY.md` (§27), `docs/experiment_manifest.csv` (R3 row) | **R3 COMPLETED, VALIDATION (Tier A).** Outputs of runs K0, K1, K2, K3a, K3b, K0F (all finished, V10 PASS) copied from the downloaded Kaggle output as listed in Section 1; `--aggregate` and script 06 run unchanged; figures as SVG via the new script 08 (script 07 unchanged; Instructor Review #9). H1 SUPPORTED in 7/7 Tier A cells; H2 SUPPORTED in 4/4 bands that formed (none at layer 5); H3 HOLDS ACROSS THE LADDER (36/36, no reversal); Q0b headline WITHIN MARGIN (ΔCE_SAE mean |T4 bf16 − fp32| 0.00267 [0.00218, 0.00319] < 0.005), top-1 agreement BEYOND MARGIN; V6 M6 REPRODUCED WITHIN MARGIN at all three layers. Tier B not run (Deviation 2). |
| `78f46c8` (Dhruv Rai) | M `src/tests/test_gemma_r3_ladder.py` (SVG test: `pytest.importorskip('matplotlib')`; skips unless matplotlib is 3.10.7, the version that wrote the committed SVGs, pinned in the test because `figure_info.json` does not record it), `docs/RESEARCH_ROADMAP.md` and this file (`de58a2e` filled in) | **CI fix.** PR #30's CI at `de58a2e` failed on the SVG test (no matplotlib on the runner); SVG bytes also differ across matplotlib versions. No script or result file changed. |
| (this commit) (Dhruv Rai) | M `docs/RESEARCH_DECISIONS.md` (D36 follow-up line: R3 status; R3 decision D37 → D46; H1 note), `docs/RESEARCH_PROGRESS.md` (§43 related line: R3 status; R3 §44 → §54), `docs/RESEARCH_ROADMAP.md` (§2 R3 line and table R3 row: status; my R3 references D37 → D46), `docs/EXPERIMENT_RESULTS_SUMMARY.md` (R3 §26 → §27), `docs/results_registry.md` (D46), this file (R3 rows renumbered; `78f46c8` filled in), `reports/Progress_Report/2026-11-14_dhruv_retrospective.md` (renumbering noted), `src/tests/test_gemma_r3_ladder.py` (`test_chunked_sae_equals_unchunked`: `rtol=0, atol=1e-5`) | **PR #30 cleanup requested in review by Simba (2026-11-17) and Aditi (CHANGES_REQUESTED 2026-11-17, reminder 2026-11-20).** (1) Stale R3 status "ONGOING, NO RESULT CLAIMED" replaced by "COMPLETED, VALIDATION (Tier A)" in Simba's four R3 lines (D36, §43, roadmap §2 and table), as both reviewers asked; the roadmap §2 line also drops "; no R3 result exists" (Simba asked in his 2026-11-17 review to update all remaining stale R3 wording). (2) Chunked-SAE test tolerance: chunking changes float32 accumulation order (Simba measured max |diff| 1.43e-6 against 1e-6), so the test uses an absolute 1e-5. (3) R3 renumbered D37 → D46, progress §44 → §54, results summary §26 → §27 (Aditi's proposal of 2026-11-17/20; #35 keeps D36–D43 and §43–§50, #31 takes D44/D45, §51–§53 and §26). Nobody else's numbers changed; `preregistration.json` has no reference to these numbers and is unchanged. No script or result changed. |

## Split and dataset changes

| Date | Change | Effect |
|---|---|---|
| ~10-03 (legacy) | GoT cities, city-grouped 70/30, seed 42 | Layer-10 legacy pipeline only |
| 10-30 | `data/got/splits/*_splits.csv`: grouped 70/15/15 (city / number pair) | All GoT layer-6 and all-layer work |
| 11-03 | `data/repe/facts_splits.csv` / `persona_pairs.csv`: statement-grouped 417/90/90 + cross-wording set | All RepE work |
| 11-07 | `data/got/splits/train_AB_groups_truth_aware_pilot.csv`: GoT **train** groups split A/B (geographic 261/262 groups, numerical 346/347), balanced | A = fine-tuning and teachers; B = independent evaluation probes. Validation and test unchanged |
| 11-07 | `data/repe/splits/r1_train_AB_groups.csv`: RepE **train** statements split A/B (208/209, normalised-statement groups, seed 20261107, label-balanced) | A = R1 fine-tuning; B = R1 independent evaluation probes. Validation/test unchanged; validation statement 326 excluded from R1/R2 scores |
| 11-12 | The Pile validation sample for M6: `monology/pile-uncopyrighted/val.jsonl.zst` (SHA-256 pinned), seed 42, 512 evaluation + 64 disjoint reference documents, first 1024 tokens. Materialised in `a34d975` as `results/gemma2_pile/gemma_scope_splicing/sample_manifest.csv` | New dataset for Gemma only. No GPT-2 split changed. The Pile test file is not used |

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
| this commit | Added `results/openwebtext/r2_scaled_saes/scaling/{README.md,preregistration.json}`, `scripts/shared/r2_scaling_analysis.py` | X0 pre-registration (S8): exposure curve from existing checkpoints, validation only |
| this commit | Added `scripts/repe/30_sae_in_the_loop_mediation.py`, `results/repe/sae_in_the_loop_mediation/{README.md,preregistration.json}` | Review #8 C3: a mediated SAE-in-the-loop design (pre-registered before any run) |
| this commit | Added `results/shared/pca_capacity_matched_test_protocol/{README.md,preregistration.json}` | PCA-CM test confirmation: analysis history, justification, frozen confirmatory families; not executed |
| this commit | M paper `00_abstract`, `10b_improvement`, `12_discussion`, `14_conclusion`; `README.md`, `docs/{research_overview,results_summary}.md` | "not worse" → "not classified worse"; explicit non-equivalence note (0/40 cells met the ±0.02 equivalence class) |
| this commit | Added `scripts/shared/registry_consistency_check.py`, `docs/remediation/checks/2026-11-13_registry_check.json`; M `.github/workflows/gpt2-validation.yml` (new step) | Phase 6: manifest scripts exist (36 rows; 3 rows have stale "current" hashes from documented 11-07 changes `9446e56`, `83bd9a9`, left unmodified because the manifest is append-only); 0 broken links in 101 Markdown files; 638/638 registered checkpoints match their SHA-256 |
| this commit | M `docs/remediation/merge_plan.md` | Re-checked PR #20/#21/#24/#26/#28 file lists against this branch: exact runtime readers of deleted files; #26's SAE-in-the-loop Part B is error-kept (equivalent to M4); RepE numbers 28–30 now taken |
| this commit | Added `results/repe/repe_lm_test_evaluation/{README.md,checks.json,comparisons.csv,confirmatory_cells.csv,ht4_probe_transfer.json,per_sae.csv}` | H1-TEST single TEST pass (user-authorised), produced by the frozen script 29 (hash verified) |
| this commit | M `results/repe/repe_lm_test_protocol/README.md` (status line), `docs/{RESEARCH_PROGRESS,RESEARCH_ROADMAP,RESEARCH_DECISIONS,EXPERIMENT_RESULTS_SUMMARY,results_registry,EXPERIMENT_CHANGELOG}.md`, `docs/experiment_manifest.csv` (+1 row via `append_experiment`) | Documentation rule; D36 (paper revision awaits the user) |
| this commit | Added `scripts/shared/h1_test_paper_table.py`, `research_paper/Latex/tables/t_h1test.tex`, `src/tests/test_h1_test_paper_claims.py` (suite 71 → 75); M `src/tests/README.md` | D36: H1-TEST table generated from committed artifacts; tests that every quoted H1-TEST number matches them |
| this commit | M `research_paper/Latex/sections/{00_abstract,05_setup,09_results_causal,10b_improvement,11_statistics,12_discussion,13_limitations,14_conclusion,15_appendix}.tex`, `research_paper/Latex/main.pdf` (rebuilt, 43 pp) | D36 (user-approved): replace "borderline, pending the test"; add the H1-TEST subsection; failed N3/N4 control as a limitation; secondary comparisons labelled exploratory |
| this commit | M `README.md`, `docs/{results_summary,research_overview,limitations_and_future_work,methodology}.md`, `docs/final_defense/{ANTICIPATED_REVIEW_QUESTIONS,DEFENSE_CHEATSHEET,RESULTS_VERIFICATION_TABLE,FINAL_RESEARCH_REPORT,EXPERIMENT_AUDIT}.md`, `reports/Latex_report/main.tex`, `presentation/gpt2_results_slides.md` | D36: curated docs updated with the H1-TEST verdicts |
| this commit | M `results/repe/repe_lm_test_evaluation/README.md`, `docs/EXPERIMENT_RESULTS_SUMMARY.md` (two rounding slips: C1 L11 TEST +0.069; C3 L1 VAL +0.142; HT2 C2 Holm bound 0.0151), `docs/{RESEARCH_PROGRESS,RESEARCH_DECISIONS,RESEARCH_ROADMAP,EXPERIMENT_CHANGELOG}.md` | Documentation rule; corrections found by the new test |
| this commit | Added `scripts/shared/backup_checkpoints.py`, `docs/backups/2026-11-13_checkpoint_{manifest.csv,verify.json}` | Audit P0: verified same-volume backup of 656 checkpoints (21.7 GB) in `../backups/2026-11-13_checkpoints/`; off-machine copy blocked (D37) |
| this commit | Added `scripts/shared/audit_s{1..6}_*.py`, `results/shared/audit_sensitivity/{README.md,preregistration.json}`, `src/tests/test_audit_sensitivity.py` (5 tests; suite 75 → 80) | AUDIT-SENS plan frozen before the full runs (sensitivity only) |
| this commit | M `docs/RESEARCH_PROGRESS.md` (§44 audit, §45 in progress), `docs/RESEARCH_ROADMAP.md` | Documentation rule |
| this commit | Added `results/shared/audit_sensitivity/s{1..6}_*/` (outputs of the frozen S1–S6 scripts); M its README (§4 results, §5 summary) | AUDIT-SENS full runs on the pre-registered hashes |
| this commit | M paper `sections/{00_abstract,07_results_information,10b_improvement,11_statistics,12_discussion,13_limitations,14_conclusion}.tex` (new §audit), `main.pdf` (44 pp); M `README.md`, `docs/{results_summary,research_overview,limitations_and_future_work}.md`, `docs/final_defense/{ANTICIPATED_REVIEW_QUESTIONS,RESULTS_VERIFICATION_TABLE}.md`, `reports/Latex_report/main.tex`, `presentation/gpt2_results_slides.md` | D38: conclusions revised to the AUDIT-SENS evidence |
| this commit | M `src/tests/test_audit_sensitivity.py` (+3 paper-claim tests; suite 80 → 83); M `docs/{RESEARCH_PROGRESS,RESEARCH_DECISIONS,RESEARCH_ROADMAP,EXPERIMENT_RESULTS_SUMMARY,results_registry,EXPERIMENT_CHANGELOG}.md`, `docs/experiment_manifest.csv` (+6 rows) | Documentation rule; D37, D38 |

## 2026-11-14 — TSP plan freeze
* Added `src/sae/truth_subspace.py` (subspace estimation from group A, controls, losses) and
  `src/tests/test_truth_subspace.py` (7 tests): new method code.
* Added `scripts/geometry_of_truth/24_truth_subspace_diagnostics.py` (root-cause diagnostics, validation),
  `25_truth_subspace_finetune.py`, `26_truth_subspace_evaluation.py`, `27_truth_subspace_analysis.py` (pilot).
* Added `results/geometry_of_truth/truth_subspace_preservation/{README.md,preregistration.json,diagnostics/}`.

## 2026-11-14 — TSP pilot completed (recovery after interruption)
* Added `scripts/geometry_of_truth/25b_truth_subspace_finetune_resume.py`: recovery wrapper. It completes the 3 runs
  missing after the session ended, without modifying frozen script 25.
* Added `results/geometry_of_truth/truth_subspace_preservation/pilot/{finetune_info.json,finetune_monitor_recovery.csv,
  subspaces.npz,recovery_check/,eval/,analysis/}`: pilot outputs. Checkpoints and logs stay local (gitignored).
* Modified `src/sae/checkpoint_registry.py`: added the `got_tsp_pilot` family. `docs/checkpoint_registry.csv` was
  rebuilt.
* Modified the TSP README with the results and the interruption record. The journal (§46), results summary, D39,
  roadmap, registry and manifest were updated.

## 2026-11-14 — TSP-VERIFY (independent verification; analysis only)
* Added `scripts/geometry_of_truth/28_tsp_independent_verification.py`: independent re-implementation of the pilot's
  metrics and statistics, plus robustness, geometry, wording, numerical-transfer and power checks. Validation only.
* Added `results/geometry_of_truth/truth_subspace_preservation/verification/` (README, per-SAE metrics, agreement,
  comparisons, power, bootstrap, info). The log in `verification_logs/` stays local (gitignored).
* Added `docs/2026-11-14_tsp_verification_and_replication_design.md`: the verification report and the TSP-R design
  (draft, not frozen).
* The journal (§47), D40, results summary, roadmap, results registry and manifest were updated. The pilot folder and
  its pre-registration were not modified. No checkpoints were created.

## 2026-11-14/15 — TSP-R Phase 0 (fresh GoT families; DEV only; no training)
* Added `scripts/geometry_of_truth/29_tsp_r_phase0_fresh_families.py` (`prepare` text-only stage; `eligibility`
  DEV-only stage). The fix 1 loader change (`keep_default_na=False`) came after eligibility attempt 1 found an empty
  DEV set; no output had been written.
* Added `data/got/fresh/splits/` (locked DEV/TEST split files, no labels, no text; cities 3-fold cross-fitting
  folds).
* Added `results/geometry_of_truth/truth_subspace_preservation/replication_phase0/`: preregistration, integrity,
  leakage, split counts, DEV label balance, conj/disj fold counts, eligibility, headroom, feasibility, README.
  `prepare_v1_superseded/` keeps the pre-freeze v1 outputs (superseded by the v2 duplicate rule).
* `.gitignore`: raw fresh downloads (`data/got/fresh/raw_*/`; no upstream licence). The DEV activations under
  `data/got/activations/gpt2_small/fresh_phase0/` and the logs stay local.
* The journal (§48), D41, results summary, roadmap, registry and manifest were updated. No checkpoints were created.

## 2026-11-15 — TSP-R Phase 1 pre-registration (no training)
* Added `scripts/geometry_of_truth/30_tsp_r_phase1_training.py`: stages plan / repro / base / base_qc / finetune. It
  reuses `run_one` from script 25 and `train` from script 17 unchanged (constants patched only); atomic saves; a
  resumable run ledger.
* Added `results/geometry_of_truth/truth_subspace_preservation/replication/`: README, preregistration.json,
  plan_matrix.csv, plan_info.json, subspaces.npz (dry-run outputs).
* The journal (§49), D42, roadmap and registry were updated. No checkpoints were created; the pilot and all
  historical results are unchanged.

## 2026-11-15 — TSP-R Phase 1 evaluation code frozen (second freeze; before any training)
* Added `scripts/geometry_of_truth/31_tsp_r_phase1_evaluation.py` (stages cities / owt / ladder / analyze / test /
  final) and `src/evaluation/tspr_stats.py` (weighted bootstrap AUROC, hierarchical seed + statement se with
  Satterthwaite df, gatekeeping with Holm, labels, non-inferiority).
* Added unit tests `src/tests/test_tspr_stats.py` (7) and the integration test
  `scripts/geometry_of_truth/31b_tsp_r_eval_integration_test.py`.
  * The integration test used identical arms plus a null-decoder arm on existing checkpoints; no training, no TEST.
  * It passed on attempt 2. Attempt 1 found a zero-se division bug, now fixed.
* Added `replication/eval_freeze.json` (hashes) and `replication/eval_integration_test.json`. `.gitignore`: the smoke
  and eval caches.

## 2026-11-15 — TSP-R Phase 1 executed (training + evaluation)
* Created checkpoints (gitignored):
  * `results/openwebtext/r2_tspr_base_seeds/` (5 bases; `training_info.json` and `training_monitor.csv` committed);
  * `replication/checkpoints/` (120 fine-tunes).
* Committed:
  * `replication/run_ledger.csv`, `base_qc.json`, `smoke_repro/repro.json`;
  * `eval/ladder_fold0.{csv,json}`;
  * `eval/analysis/*` (cities analysis committed first, then TEST families and the final verdict);
  * README.
* `scripts/geometry_of_truth/31_tsp_r_phase1_evaluation.py`: fix 1 (`allow_pickle=True` for the final stage's own
  npz; no metric change).
* `src/sae/checkpoint_registry.py`: added families `got_tspr_replication`, `owt_r2_tspr_base`, `got_tspr_smoke`. The
  registry was rebuilt (780 checkpoints).
* The journal (§50), results summary, D43, roadmap, results registry and manifest were updated.

## 2026-11-15: dead-latent audit (instructor request of 2026-11-10)

| Commit | Files | Reason |
|---|---|---|
| this commit | Added `results/shared/dead_latent_audit/{build_dead_latent_audit.py,dead_latents_by_sae.csv,summary_by_family_layer.csv,checks.txt}` | Dead-latent fraction for every base (108) and R2 (36 trained + 3 frozen-random-decoder) GPT-2 SAE, from stored evaluation CSVs only. The script refuses to overwrite its outputs. A rerun after the typo fix gave byte-identical CSVs |
| this commit | Added `docs/DEAD_LATENT_AUDIT.md` | Audit report: definition, coverage, results, checks, provenance |
| this commit | M `research_paper/Latex/sections/12_appendix.tex`, `results/shared/m0_within_layer_reanalysis/README.md`, `README.md` | Transcription correction: JumpReLU alive-latent range 469–2,947 → 469–2,941 (`architecture_by_layer.csv` holds 2,941.0, unchanged). No result changed |
| this commit | M `research_paper/Latex/main.pdf` (rebuilt) | Appendix correction |
| this commit | M `docs/{RESEARCH_PROGRESS,EXPERIMENT_CHANGELOG,EXPERIMENT_RESULTS_SUMMARY,RESEARCH_ROADMAP,results_registry}.md`, `docs/experiment_manifest.csv` (+1 row via `append_experiment`) | Documentation rule |

## 2026-11-16: Method A multi-layer extension scope decision (documentation only)

| Commit | Files | Reason |
|---|---|---|
| this commit | M `docs/{RESEARCH_DECISIONS,RESEARCH_PROGRESS,RESEARCH_ROADMAP,results_registry,EXPERIMENT_CHANGELOG}.md` | D45 (numbered D37 until 2026-11-17): the `simba-method-a-multilayer` extension (layers 5/6/9; `28ad5cf`, `b64e6ed`, `7f2557b`) was audited and excluded from the final scope. No result, paper, manifest or checkpoint-registry file changed; branch results not merged |

## 2026-11-17: PR #31 reviewer-driven manuscript corrections (Aditi Shukla, Dhruv Rai; wording only)

| Commit | Files | Reason |
|---|---|---|
| this commit | M `research_paper/Latex/sections/{02_setup,04_information,07_improvement,08_generalisation,09_discussion,12_appendix}.tex`, `research_paper/Latex/tables/{t_evidence,t_overview}.tex`, `research_paper/Latex/main.pdf` (rebuilt, 36 pp) | **Changes:** (1) Frozen-probe interpretation revised per S5/D38 (`results/shared/audit_sensitivity/`, on `sae-research`): frozen readers exaggerate the loss, but about a third to a half of the frozen gap at layers 6–11 is label-aligned SAE error. (2) RepE-LM held-out TEST results added (`results/repe/repe_lm_test_evaluation/`, on `sae-research`): HT2 confirmed in 9/9 cells; HT1/HT3 not confirmed, only their TopK L8 cells passed (0.602 → 0.729, Holm p = 0.021; −0.048, Holm p = 0.046), both fragile under statement sampling (S1, Holm p = 0.156); the "evaluate the test protocol" next step was removed. (3) R3 status set to completed at validation, PR #30 under review; not used as evidence. (4) Causal patching (error term kept, no routing through the SAE) is distinguished from splicing. (5) "Significantly worse" changed to "worse under the per-cell criterion" (no family-wise correction, D34). No result file changed. TSP not added (team scope decision); D37/§44/§26 not renumbered |

## 2026-11-17: PR #31 integration of H1-TEST, AUDIT-SENS and TSP; numbering (documentation and manuscript only)

| Commit | Files | Reason |
|---|---|---|
| this commit | M `research_paper/Latex/sections/{04_information,06_causal,07_improvement,09_discussion,11_conclusion,12_appendix}.tex`, `research_paper/Latex/tables/t_evidence.tex`, `research_paper/Latex/main.pdf` (rebuilt); A `research_paper/Latex/tables/t_h1test.tex` (byte-identical copy from `origin/sae-research` `20b7c98`, generated there by `scripts/shared/h1_test_paper_table.py`) | Aditi's PR #31 follow-up: H1-TEST subsection (HT2 9/9; HT1/HT3 not confirmed, TopK L8 cells fragile, Holm p = 0.156; BatchTopK cells not confirmed; failed N3/N4 control), Audit Sensitivity Analyses appendix (S1–S6), TSP subsection (validation, confirmatory; TRADE-OFF; in-distribution only), S2 persona wording (D38), S3 task-level-L0 caveat on architecture orderings. Sources: `results/repe/repe_lm_test_evaluation/`, `results/shared/audit_sensitivity/`, `results/geometry_of_truth/truth_subspace_preservation/replication/` (all on `sae-research`); every number checked against them. No result file changed |
| this commit | A `docs/plain_summary.md` (from `origin/main` `6403ec9`, one sentence changed) | Withdrawn S5 wording ("so the probe was reacting to any disturbance") replaced; required by `test_withdrawn_claims_regression.py` |
| this commit | M `docs/{RESEARCH_DECISIONS,RESEARCH_PROGRESS,RESEARCH_ROADMAP,results_registry,EXPERIMENT_CHANGELOG}.md` | Numbering agreed on PR #31: M6 D36 → D44 and §43 → §51 (M6: D21 → D36 → D44); Method A exclusion D37 → D45; dead-latent audit §44 → §52; Method A §45 → §53; D36–D43 and §43–§50 reserved for #35; D46/§54/summary §27 reserved for R3. Live pointers updated; historical changelog rows that describe what earlier commits did are unchanged |

## 2026-11-17 — Regression guard for withdrawn claims (no experiment run)
* Added `src/tests/test_withdrawn_claims_regression.py` (14 tests). It protects the D36 (H1-TEST) and D38 (AUDIT-SENS
  S2, S5) corrections when the restructured manuscript (#31) and `main` are integrated with this branch: the older
  exact-string checks in `test_audit_sensitivity.py` and `test_h1_test_paper_claims.py` miss re-worded or re-wrapped
  copies. Those files are unchanged.
* Checked on copies of other branches (not modified): #31 at `66c114f` fails 4 checks (`04_information.tex`,
  `08_generalisation.tex`, `09_discussion.tex`); `main` at `bf25b40` fails 5 (abstract, information section,
  conclusion, `docs/plain_summary.md`). This branch passes.
* M `src/tests/README.md` (entry for the new test).
* No results, frozen records, checkpoints or scripts changed.

## 2026-11-21: final-week Phase 1 (cross-model infrastructure, gates, Gemma task baselines; PR "final-week P1")

| File(s) | Change | Reason |
|---|---|---|
| `docs/2026-11-21_final_week_technical_audit.md` | added | approved technical audit and plan |
| `src/models/backends.py` | added | one residual-hook interface for GPT-2 (TL) and Gemma (HF) |
| `src/evaluation/pile_splicing.py` | added, byte-identical to `origin/main` (blob `f4a52c4`) | reuse M6's validated helpers; merges cleanly |
| `scripts/cross_model/00_fetch_assets.py` … `03_matched_splicing_analysis.py` | added | XB pipeline (assets, samples, runner, analysis) |
| `scripts/cross_model/10_truth_preservation.py` | added | XT-G / DX runner (reuses M2 and RLM-RP code unchanged) |
| `scripts/gemma/10_extract_task_activations.py` | added | XT-G0 Gemma task activations + K1 / K7 gates |
| `scripts/geometry_of_truth/32_tsp_generalisation.py`, `33_extract_fresh_dev_all_layers.py` | added | TI-1 driver; GPT-2 fresh-DEV caches for XT-G A2 |
| `src/tests/test_cross_model.py` | added | 9 offline unit tests |
| `results/cross_model/{FINAL_WEEK_TRACKER.md, assets/, xb_matched_splicing/, xtg_truth_preservation/}` | added | pre-registrations, amendments, hashes, samples, M6 inputs, smoke / gate outputs (full runs not included) |
| `results/gemma/xtg_task_baselines/` | added | XT-G0 README + gates |
| `results/geometry_of_truth/tsp_generalisation/` | added | TI-1 pre-registration, gate outputs |
| `docs/RESEARCH_{PROGRESS,DECISIONS,ROADMAP}.md`, `docs/results_registry.md`, `docs/experiment_manifest.csv` | appended | documentation rule (CLAUDE.md §1) |

## 2026-11-21: merge of `origin/main` into `sae-research` (resolves PR #35 conflicts)
No experiment, code or result changes. Both sides kept in full in the four doc ledgers that conflicted; one withdrawn sentence corrected (below).
| File | Change | Reason |
|---|---|---|
| `docs/RESEARCH_DECISIONS.md` | M6 decision renumbered **D36 → D44** (same as on #31); final-week decision **D44 → D47**; references in DECISIONS, ROADMAP and `results_registry.md` updated | team numbering plan: #35 D36–D43 (cited in the paper and the D36/D38 regression test, so unchanged), #31 D44–D45, #30 D46. A first pass of this merge (`2bee3db`) used D45 for M6; corrected in the follow-up commit |
| `docs/RESEARCH_PROGRESS.md` | M6 journal entry renumbered **§43 → §51** (same as on #31); final-week entry **§51 → §55** | team numbering plan: #35 §43–§50, #31 §51–§53, #30 §54 |
| `docs/EXPERIMENT_RESULTS_SUMMARY.md` | both sides kept; the M6 section keeps §25 | no clash |
| `docs/experiment_manifest.csv` | both sides' rows kept (M6 row appended after XT-G0) | append-only ledger |
| `docs/plain_summary.md` (from PR #33) | one sentence reworded: "same-sized random noise caused almost the same drop, so the probe was reacting to any disturbance" → noise explains part of the drop; a third to a half at the middle and later GoT layers is label-aligned SAE error | that sentence is the claim withdrawn in D38 / AUDIT-S5; `test_withdrawn_claims_regression.py` failed on it after the merge (user approved the fix 2026-11-21) |
Historical changelog rows above that say "D36" for M6 describe those commits as made, and are left unchanged.

## 2026-11-21: merge of `origin/main` into `simba-paper-manuscript` (resolves PR #31 conflicts after PR #35 and PR #36)

| File | Resolution | Reason |
|---|---|---|
| `research_paper/Latex/sections/{05_setup,07_results_information,09_results_causal,10b_improvement,11_statistics,12_discussion,13_limitations,14_conclusion,15_appendix}.tex` | kept DELETED (modify/delete; #35 had edited them) | Legacy pre-restructure files, not `\input` by `main.tex`. Every #35 edit (H1-TEST, AUDIT-SENS S1–S6, persona wording, frozen-probe wording, test-split status) was checked to be present in the #31 sections (`07_improvement`, `12_appendix`, `04_information`, `06_causal`, `09_discussion`, `11_conclusion`, `tables/t_h1test.tex`) |
| `research_paper/Latex/sections/00_abstract.tex` | #31 version kept | Restructured abstract already reviewed (Aditi, `5f1b51e`); the `origin/main` side is the legacy long abstract |
| `research_paper/Latex/tables/t_h1test.tex` | auto-merged, identical on both sides | #31 copied it byte-identically from `20b7c98` |
| `research_paper/Latex/tables/t_evidence.tex` | dagger ("Not yet on the `main` branch") removed from the H1-TEST, AUDIT-SENS and TSP rows; kept on R3 | those three folders are on `main` since PR #35; R3 is still PR #30 only. No number or label changed |
| `research_paper/Latex/main.pdf` | rebuilt from the merged sources | binary; neither side taken |
| `docs/RESEARCH_DECISIONS.md` | both sides: #31 numbering note, #35 D36–D43, D44 (M6; keeps D21 → D36 → D44), D45, D47 | team numbering plan; no frozen #35 record renumbered |
| `docs/RESEARCH_PROGRESS.md` | both sides: #31 numbering note, #35 §43–§50, §51 (M6), §52–§53 (#31), §55 (final week) | team numbering plan; §54 reserved for R3 |
| `docs/RESEARCH_ROADMAP.md` | both sides, chronological; M6 rows from `origin/main`; new 2026-11-21 merge entry | the 2026-11-17 merge-order line is superseded (#35 merged first) |
| `docs/results_registry.md` | both sides, chronological (#35 R2-row update, AUDIT-SENS, TSP; #31 2026-11-15/16; final week) | — |
| `docs/plain_summary.md` | #31 sentence kept | the version Aditi accepted; it does not restore the withdrawn S5 wording |
| `docs/EXPERIMENT_CHANGELOG.md` | both sides, chronological; this entry added | — |

## 2026-11-21: TI-1 results (TSP generalisation)
| File | Change | Reason |
|---|---|---|
| `scripts/geometry_of_truth/32_tsp_generalisation.py` | modified | OOM fix in `stage_eval` (batch 16 → 4, same for original log-probs); `stage_analyze` dict-access fix (`r.seed` → `r['seed']`). Neither changes any computation |
| `results/geometry_of_truth/tsp_generalisation/{ledger_*.csv, eval/, analysis/}` | added | 81 fine-tunes evaluated; pre-registered analysis |
| `results/geometry_of_truth/tsp_generalisation/README.md` | updated | results, criteria, deviations |
| `docs/RESEARCH_{PROGRESS,DECISIONS,ROADMAP}.md`, `docs/EXPERIMENT_RESULTS_SUMMARY.md`, `docs/results_registry.md`, `docs/experiment_manifest.csv`, `docs/checkpoint_registry.csv` | appended | documentation rule (CLAUDE.md §1); §56, D48; 81 checkpoint rows (checkpoints not committed) |

## 2026-11-21: X0 and SIL results (pre-registered 2026-11-13)
| File | Change | Reason |
|---|---|---|
| `results/openwebtext/r2_scaled_saes/scaling/{ladder_layer*_r2scaling.*, per_checkpoint.csv, comparisons.csv, summary.json, figures/}` | added | X0 frozen commands and analysis (checkpoint hashes verified first) |
| `results/repe/sae_in_the_loop_mediation/{pilot/, validation/}` | added | SIL pilot gate (passed) and full run |
| both `README.md` | updated | results, criteria, the SIL negative-control caveat |
| `docs/RESEARCH_{PROGRESS,DECISIONS,ROADMAP}.md`, `docs/EXPERIMENT_RESULTS_SUMMARY.md`, `docs/results_registry.md`, `docs/experiment_manifest.csv` | appended | documentation rule; §57, §58, D49. No code changed; no checkpoints created |

## 2026-11-21: XB primary results
| File | Change | Reason |
|---|---|---|
| `results/cross_model/xb_matched_splicing/runs/*/{per_doc.csv, DONE.json}` | added | six completed runs; `sum_x.npz` (38–243 MB each) not committed, SHA-256 in `runs_sum_x_sha256.txt` |
| `results/cross_model/xb_matched_splicing/analysis_v1/` | added | frozen analysis |
| `results/cross_model/xb_matched_splicing/README.md`, `AMENDMENTS.md` | updated | results with the context caveat; operational note (expandable_segments restart) |
| `docs/*` ledgers | appended | §59, D50, results summary, roadmap, registry, manifest |

## 2026-11-22: PCA-CM-T TEST pass
| File | Change | Reason |
|---|---|---|
| `scripts/shared/pca_cm_test_confirmation.py`, `src/tests/test_pca_cm_test_confirmation.py` | added | TEST runner implementing the frozen 2026-11-13 rules; validation-reproduction mode |
| `results/shared/pca_capacity_matched_test_protocol/{validation_reproduction/, addendum_pre_test.json, test/, run_log.txt, README.md}` | added / updated | exact validation reproduction; pre-TEST freeze; single TEST pass; abort log |
| `docs/*` ledgers | appended | §60, D51, summary, roadmap, registry, manifest |

## 2026-11-22: correction to the XB primary results
| File | Change | Reason |
|---|---|---|
| XB `README.md` §3.2, §59, D50, results summary | corrected | the GPT-2 XB_H1 "0/318" under Holm is mechanically impossible to pass with a 1,000-resample bootstrap over 159 tests. It was wrongly attributed to the out-of-distribution artefact; the uncorrected CI verdicts are now reported as descriptive. No result or rule changed |

## 2026-11-22: XB A4 results
| File | Change | Reason |
|---|---|---|
| `scripts/cross_model/0{4,5,7}_*.py`, `src/tests/test_xb_a4.py` | added | A4 manifest builder, wrapper (frozen 02/03 unchanged), paired context comparison; frozen in `0c15597` |
| `results/cross_model/xb_matched_splicing/samples/*ctx128*` | added | 128-token re-cut manifests |
| `results/cross_model/xb_matched_splicing/a4_ctx128/` | added | protocol, runs (`per_doc.csv`; `sum_x.npz` hashed, not committed), analysis, comparison, README |
| `docs/*` ledgers | appended | §61, D52, summary, roadmap, registry, manifest, tracker |

## 2026-11-22: E1 results
| File | Change | Reason |
|---|---|---|
| `scripts/shared/e1_task_l0_matched_topk.py` (+ E1-A1), `scripts/shared/e1_g2_blas_threads_diagnostic.py`, `src/tests/test_e1_task_l0_matched.py` | added | E1 implementation; single-threaded probe BLAS (E1-A1) and its diagnostic |
| `results/shared/e1_task_l0_matched_topk/{gate.json, gate_attempt1_failed_G2_blas_threads.json, ledger.csv, eval_*.csv, analysis/, README.md}` | added / updated | gates, 36 SAEs, evaluation, pre-registered analysis |
| `docs/checkpoint_registry.csv` | appended | 36 E1 keeper checkpoints (EBS copy, gitignored) |
| `docs/*` ledgers | appended | §62, D53, summary, roadmap, registry, manifest |
| `scripts/shared/e1_g2_blas_threads_diagnostic.py` | path sanitised for commit | its hard-coded absolute repo path is replaced by `Path(__file__).resolve().parents[2]` (the same directory; behaviour unchanged). SHA-256 (16 characters) cc0b31ef9089ba30 → a2a0cd4f5586a68c; the E1 pre-registration records the original. Not used by any E1 stage |

## 2026-11-22: XT-G / DX results
| File | Change | Reason |
|---|---|---|
| `scripts/cross_model/11_xtg_analysis.py`, `src/tests/test_xtg_analysis.py`, XT-G `AMENDMENTS.md` (A4), `frozen_hashes_analysis.txt` | added | analysis frozen in `cdefced` before any output value was read |
| `results/cross_model/xtg_truth_preservation/runs/{gemma_main, gpt2_main}/` | added | per-cell, DX, RepE-extras outputs (GPT-2 L0 not evaluable) |
| `results/cross_model/xtg_truth_preservation/analysis_v1/`, `README.md` | added / updated | rule counts, cross-model descriptive table |
| `docs/*` ledgers | appended | §63, D54, summary, roadmap, registry, manifest, tracker |

## 2026-11-23: XT-G / DX review fix (PR #43, reviewer note)
* README, `RESEARCH_PROGRESS`, `RESEARCH_DECISIONS`, `results_registry`, `RESEARCH_ROADMAP`: SUPPRESSION is described as
  error concentrated along the **task** direction (truth for GoT; the persona / neutral label direction for RepE),
  because for the RepE tasks the direction is not a truth direction. Wording only; no rule or count changed.

## 2026-11-23: XB A4 review fix (PR #41, reviewer note)
* `a4_ctx128/README.md`, `docs/EXPERIMENT_RESULTS_SUMMARY.md`, `docs/RESEARCH_PROGRESS.md`: "R2 0.83–0.95" was the range
  of per-layer means; now also reports individual trained R2 SAEs 0.81–0.98 and the frozen-decoder control 0.57–0.68
  (recomputed from `analysis_a4/summary_by_sae.csv`, both corpora). Wording only; no result changed.

## 2026-11-23: XB primary review fix (PR #39, reviewer note)
* `results/cross_model/xb_matched_splicing/README.md` and `docs/EXPERIMENT_RESULTS_SUMMARY.md`: the GPT-2 shared-span
  EV / L0 values are labelled OWT-XM (verified against `analysis_v1/summary_by_sae.csv`, corpus `owt_xm`). Wording only;
  no number changed.

## 2026-11-23: oversize XB per-document CSVs stored gzipped (CI repository-hygiene fix)
| File | Change | Reason |
|---|---|---|
| `results/cross_model/xb_matched_splicing/runs/{gpt2_owt_xm_shared_main,gpt2_pile_shared_main}/per_doc.csv` | replaced by `per_doc.csv.gz` (gzip -n -9, lossless) + `per_doc.csv.sha256` (hash of the uncompressed file; round trip verified) | 27.2 MB each, above the 25 MB limit of `scripts/shared/ci_repo_hygiene.py`; `main` CI failed after #39 merged |
| `results/cross_model/xb_matched_splicing/README.md` | storage note and re-run instruction | frozen analysis scripts unchanged (decompress before re-running) |

## 2026-11-23: oversize XB A4 per-document CSVs stored gzipped (CI repository-hygiene fix)
* `results/cross_model/xb_matched_splicing/a4_ctx128/runs/{gpt2_owt_xm_shared_main,gpt2_pile_shared_main}/per_doc.csv`
  replaced by lossless `per_doc.csv.gz` + `per_doc.csv.sha256` (round trip verified); 27.2 MB each, above the 25 MB
  hygiene limit. README storage note added. Frozen scripts unchanged. Same treatment as the primary-arm runs (#53).

## 2026-11-23 (recovery session; committed on `sae-research`)
| File | Change | Reason |
|---|---|---|
| `docs/2026-11-23_recovery_and_final_day.md` | added | recovery record, Phase 2 status, final-day plan, git placement |
| `results/geometry_of_truth/tsp_multifamily/` (pre-registration, gates, ledger, subspaces, eval, analysis, README) | added (outputs from 2026-11-22), README results appended | TI-2 results; checkpoints lost, hashes in `ledger.csv` |
| `scripts/geometry_of_truth/34_tsp_multifamily.py`, `src/tests/test_tsp_multifamily.py` | added (frozen, unchanged) | TI-2 pipeline |
| `scripts/geometry_of_truth/34b_tsp_multifamily_analyze_allow_pickle.py` | added | declared deviation: re-run the frozen analysis (allow_pickle only for `eval/orig.npz`) |
| `results/geometry_of_truth/tsp_kl_factorial/` (pre-registration, gates, ledger, eval, analysis, README) | added (outputs from 2026-11-22), README review appended | TI-4 results |
| `scripts/geometry_of_truth/35_tsp_kl_factorial.py`, `src/tests/test_tsp_kl_factorial.py`, `scripts/shared/e1_task_l0_matched_topk.py` | added (frozen, unchanged) | TI-4 pipeline and its frozen dependency |
| `results/geometry_of_truth/minimal_pair_preservation/` (pre-registration, README, WITHDRAWN, READY.hold.tf32) | added, README status appended | TI-3 withdrawal record |
| `scripts/geometry_of_truth/36_minimal_pair_preservation.py`, `src/tests/test_minimal_pair_preservation.py` | added (unchanged) | TI-3 pre-registered code |
| `results/cross_model/xb_ladder_batch2/`, `results/cross_model/assets/asset_manifest_b2b.json`, `results/cross_model/xb_matched_splicing/runs/gemma_pile_shared_b2/{DONE.json,per_doc.csv}` | added, README status appended | ladder partial state (`sum_x.npz` hash only) |
| `scripts/cross_model/06_ladder_scratch.py`, `src/tests/test_ladder_scratch.py` | added (unchanged) | ladder pipeline |
| `results/cross_model/xc3_steering_parity/` (pre-registration, body, sha256, README, smoke outputs on TRAIN rows) | added | XC-3 pre-registration (results committed separately) |
| `scripts/cross_model/20_xc3_steering_parity.py` | added (frozen) | XC-3 runner |
| `docs/RESEARCH_{PROGRESS,DECISIONS,ROADMAP}.md`, `EXPERIMENT_RESULTS_SUMMARY.md`, `results_registry.md`, `EXPERIMENT_CHANGELOG.md` | appended | documentation rule |

## 2026-11-26: #53 prepared for integration (D55 heading; preflight against silently skipped gzipped runs)
No experiment, result, frozen script, criterion or threshold changed.
| File | Change | Reason |
|---|---|---|
| `docs/RESEARCH_DECISIONS.md` | the 2026-11-23 gzip note gets its own heading **D55** (text kept verbatim) and a preflight bullet | it read as part of D51 |
| `scripts/shared/compressed_outputs_preflight.py` | added | `03` `load_runs` (reused by `07`) skips folders without `per_doc.csv`, so a fresh checkout would silently drop the gzipped GPT-2 runs; `--verify` / default / `--decompress` / `-- CMD` |
| `src/tests/test_compressed_outputs_preflight.py` | added (11 tests) | preflight behaviour; the hazard shown on the unmodified frozen `03` loader; integrity of every committed `.gz` (CI) |
| `results/cross_model/xb_matched_splicing/README.md` | preflight command added to the storage note | re-run instructions |
| `.gitignore` | the two decompressed `per_doc.csv` paths | cannot be re-committed above the 25 MB limit |
| `docs/RESEARCH_PROGRESS.md` (§64), `docs/RESEARCH_ROADMAP.md` | appended | documentation rule (CLAUDE.md §1) |
Verification: all 2 pairs here (and #41's 2 A4 pairs) are byte-identical to the uncompressed originals, match their sidecars, and are reproducible with `gzip -n -9`.

## 2026-11-26: #41 integrated with #53 (A4 side of D55)
No experiment, result, frozen script, criterion or threshold changed.
| File | Change | Reason |
|---|---|---|
| `docs/{EXPERIMENT_CHANGELOG,RESEARCH_DECISIONS,RESEARCH_PROGRESS}.md` | merge of #53: both sides kept (D52 then D55; §61 then §64; changelog in date order) | append-only ledgers |
| `.gitignore` | the two decompressed A4 `per_doc.csv` paths | cannot be re-committed above the 25 MB limit |
| `results/cross_model/xb_matched_splicing/a4_ctx128/README.md` | preflight command for `07` added to the storage note | `07` reuses `03`'s `load_runs`, which skips folders without `per_doc.csv` |
| `docs/RESEARCH_ROADMAP.md` | status line | documentation rule (CLAUDE.md §1) |
Verified: the two A4 `.gz` files are byte-identical to the originals in `15e78e7`, match their sidecars, and are reproducible with `gzip -n -9`; the preflight `--verify` test now covers all four committed pairs.
End-to-end: through the preflight (`--decompress` from the committed `.gz`; hash-matched local `sum_x.npz`; CI-pinned environment; scratch worktree), the unmodified frozen `07` reproduces the committed `a4_ctx128/context_comparison_a4/`: `info.json` byte-identical; `context_change.csv` 324 rows, every label identical (SHRINKS 269 / GROWS 29 / NO CLEAR CHANGE 26; Holm 324 × NO CLEAR CHANGE), numeric differences ≤ 1.5e-12 relative. Scratch outputs deleted.

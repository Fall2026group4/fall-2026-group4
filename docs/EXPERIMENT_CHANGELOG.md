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
| (this commit) | M `docs/RESEARCH_ROADMAP.md` (rewritten), `CLAUDE.md`, `docs/EXPERIMENT_CHANGELOG.md` | Roadmap completeness review: added outstanding GPT-2 tasks (training improvements, published JumpReLU / missing kernel-STE script, public-SAE baselines, AlignSAE, replications), sparsity-matching corrections S1–S6, causal/steering experiments C1–C7, Gemma tasks M1–M5, cross-model comparison and final test X1–X3, capstone deliverables F1–F6 (deadline 2026-12-08), and a naming map for the plan-vs-experiment 'Method' letters. CLAUDE.md: documentation applies even when not requested; roadmap read at task start; new tasks added to the roadmap; pre-commit documentation checklist |

## Split and dataset changes

| Date | Change | Effect |
|---|---|---|
| ~10-03 (legacy) | GoT cities, city-grouped 70/30, seed 42 | Layer-10 legacy pipeline only |
| 10-30 | `data/got/splits/*_splits.csv`: grouped 70/15/15 (city / number pair) | All GoT layer-6 and all-layer work |
| 11-03 | `data/repe/facts_splits.csv` / `persona_pairs.csv`: statement-grouped 417/90/90 + cross-wording set | All RepE work |
| 11-07 | `data/got/splits/train_AB_groups_truth_aware_pilot.csv`: GoT **train** groups split A/B (geographic 261/262 groups, numerical 346/347), balanced | A = fine-tuning and teachers; B = independent evaluation probes. Validation and test unchanged |

## Not committed (by rule)
* `data/activations/` and the `data/*/activations/` caches.
* All `.pt` checkpoints (see `docs/checkpoint_registry.csv` for paths and SHA-256).
* Logs (`results/*/logs/`).
* `scripts/geometry_of_truth/07_independent_truth_probe.py.bak` (identical to `0d07739`) and `scripts/openwebtext/10_reconstruction_evaluation.py.bak`.
* `results/professor_review_corrections/` and `scripts/professor_review/` are kept for provenance. Deletion awaits approval.

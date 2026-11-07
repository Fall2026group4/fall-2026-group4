# Research roadmap (as of 2026-11-07)

Update this file whenever a task starts, finishes, stops or changes plan (`CLAUDE.md` §1). Status labels: DONE, IN PROGRESS, PLANNED, BLOCKED, ABANDONED.

## Completed (DONE)

| Item | Evidence | Commit |
|---|---|---|
| Set-up, EDA (OWT, GoT, RepE), Phase-1 Feature 974 | `results/phase1_observations.md`, EDA tables | `070db9b`, `8f3814b` |
| Team-branch pipelines: public-SAE OWT, GoT and RepE stages, Gemma baseline, Pile EDA | unmerged branches [BRANCH-ONLY] | see changelog |
| In-house OWT SAE pipeline, 1000-step sweep, layer-2/6/10 convergence | `results/openwebtext/` | `8dc0b6e` |
| GoT layer-6 pipeline and final MLP test; RepE pipeline | `results/geometry_of_truth/`, `results/repe/` | `989ea71`, `0d07739`, `0eaad9c` |
| Professor-review corrections | `docs/2026-11-04_*` | `2ad0979` |
| 108-run all-layer OWT baseline | `uninterrupted_protocol/all_layers_*` | `9446e56` |
| All-layer GoT/RepE frozen-probe evaluation; shared metrics; checkpoint registry | `*/all_layer_frozen_probe_evaluation/` | `8919158` |
| Phase 0 splicing and layer-6 bridge | `*/spliced_evaluation/` | `468018c` |
| P1 sparsity-control pilot | `results/openwebtext/p1_sparsity_control_pilot/` | `ef9fec4` |
| Truth-aware audit; corrected truth-aware pilot | `docs/2026-11-07_truth_aware_finetuning_audit.md`, `truth_aware_pilot/` | `1614be5`, `dcf1486` |
| Method A at layer 6 | `method_a_layer6/` | `ca81ad2` |
| E2E-inspired Methods B/C at layer 6 | `e2e_inspired_layer6/` | `764dcbb` |
| Permanent documentation system | `CLAUDE.md`, `docs/RESEARCH_*.md`, `docs/EXPERIMENT_*.md` | this commit |

## In progress
* None. Every experiment above is committed locally. Nothing has been pushed; pushing needs approval.

## Planned (requires approval before starting)

| ID | Experiment | Why | Status |
|---|---|---|---|
| R1 | Converged EV-vs-L0 frontier (targets 16–256), with the P1 controller (anneal + freeze) | Sparsity matching and architecture ranking at other L0 values (original Method B) | PLANNED |
| R2 | JumpReLU ablation: threshold initialisation vs controller changes, layers 2/6/10 × 3 seeds (P2a), then 30k-step convergence check (P2b) | Attribute the P1 gains; training length | PLANNED |
| R3 | Width sweep {1536, 3072, 6144, 12288} at layers 2/6/10 (original Method C) | Deep-layer headroom (H1) | PLANNED |
| R4 | Residual two-stage SAE vs a single SAE of the same total size (original Method E) | Reconstruction improvement | PLANNED (not implemented) |
| R5 | Method A replication with more seeds, separately pre-registered; other layers | H3 | PLANNED |
| R6 | L0 control under KL objectives (ReLU/JumpReLU), then the A + small-KL combination | H4; the E2E longer-budget flag | PLANNED |
| R7 | Layer-0 JumpReLU fix (threshold learning rate, dead-latent revival) | H5 | PLANNED |
| R8 | Final pre-registered test evaluation of the selected configurations, all layers, 3 seeds | Confirmed results | PLANNED (last) |
| R9 | Gemma-2-2B SAE baselines (Gemma Scope), if feasible on this host | H6 | BLOCKED (weights not cached; RAM) |

## Missing baselines and controls (known gaps)
* **A converged sparsity sweep.** The 1000-step sweep is exploratory only.
* **Width other than 3072.** **The published JumpReLU estimator.** The kernel-STE script is missing; only checkpoints exist.
* **Public-SAE baselines under our evaluation pipeline.** Only layer 10 (`gpt2-small-res-jb`) has a frozen-probe result.
* **Sparsity matching on GoT and RepE prompts.** It is defined on OWT only.
* **Behavioural outcomes:**
  * GoT truth behaviour is measurable only on `cities` (GPT-2's preference on the other sets is at chance or inverted).
  * GPT-2's own honesty effect on the next statement token is about 0.
* **A causal test at layers other than 6, and on the uninterrupted SAEs.**
* **More seeds** for close calls; probe-seed variance (MLP sd .01–.05) dominates some comparisons.

## Abandoned or superseded
* **Linear-teacher truth-aware fine-tuning in its current form:** abandoned (D16).
* **Earlier "SAE steering" correlations, re-trained-probe faithfulness, `architecture_summary.csv`, warm-start/resume cross-layer runs, Gemma row-split baseline:** superseded or invalid (`docs/results_registry.md`).

## Housekeeping awaiting approval
* Deleting `results/professor_review_corrections/`, `scripts/professor_review/`, `src/tests/test_professor_review.py` (all migrated), and the two `.bak` files.
* Pushing `sae-research` to GitHub (many commits ahead of `origin/sae-research`).
* Merging or archiving the unmerged team branches.

## Note on this roadmap's specification
The request that created this file (2026-11-07) was cut off after "Missing bas…" in Part 3 item 5. The sections above cover completed, in-progress and planned work, missing baselines, abandoned work and housekeeping. Further sections can be added once the full specification is available.

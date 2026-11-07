# Research roadmap (as of 2026-11-07; final deadline 2026-12-08)

Update this file whenever a task starts, finishes, stops, changes plan, or a new outstanding task is identified (`CLAUDE.md` §1).

**Statuses:**
* DONE;
* IN PROGRESS;
* PLANNED (needs approval before starting);
* BLOCKED;
* ABANDONED.

**Priorities:**
* **P0** is needed for the capstone's core claims;
* **P1** strengthens them;
* **P2** is optional or stretch.

**Naming note.** Two sets of names use "Method" letters:
* **The 2026-11-06 gap plan** (`docs/research_gaps_and_plan.md`) used Methods A–F: A recon fine-tuning, B sparsity optimisation, C width, D improved training, E residual SAE, F truth-aware.
* **The experiments that were actually run** use "Method A" (GoT reconstruction + OWT replay) and "Methods B/C" for the **E2E-inspired** objectives.

This roadmap therefore names tasks by content. Gap-plan letters appear as "plan-B", "plan-C" and so on.

## 1. Completed (DONE)

| Item | Evidence | Commit |
|---|---|---|
| Set-up, EDA (OWT, GoT, RepE), Phase-1 Feature 974 | `results/phase1_observations.md`, EDA tables | `070db9b`, `8f3814b` |
| Team-branch pipelines: public-SAE OWT, GoT/RepE stages, Gemma baseline, Pile EDA | unmerged branches [BRANCH-ONLY] | see `EXPERIMENT_CHANGELOG.md` |
| In-house OWT SAE pipeline; 1000-step sweep; layer-2/6/10 convergence | `results/openwebtext/` | `8dc0b6e` |
| GoT layer-6 pipeline and final MLP test; RepE pipeline | `results/geometry_of_truth/`, `results/repe/` | `989ea71`, `0d07739`, `0eaad9c` |
| Professor-review corrections, incl. GPT-2 causal test with 200 controls and Gemma steering sweep | `docs/2026-11-04_*` | `2ad0979` |
| 108-run all-layer OWT baseline | `uninterrupted_protocol/all_layers_*` | `9446e56` |
| All-layer GoT/RepE frozen probes; shared metrics; checkpoint registry | `*/all_layer_frozen_probe_evaluation/` | `8919158` |
| Phase 0 splicing and layer-6 bridge | `*/spliced_evaluation/` | `468018c` |
| P1 sparsity-control pilot | `results/openwebtext/p1_sparsity_control_pilot/` | `ef9fec4` |
| Truth-aware audit; corrected truth-aware pilot | audit doc, `truth_aware_pilot/` | `1614be5`, `dcf1486` |
| Method A at layer 6 | `method_a_layer6/` | `ca81ad2` |
| E2E-inspired Methods B/C at layer 6 | `e2e_inspired_layer6/` | `764dcbb` |
| KL-weight ablation, layer 6 TopK (G8, part): verdict INCONCLUSIVE | `kl_weight_ablation_layer6_topk/` | this commit |
| Documentation system | `CLAUDE.md`, `docs/RESEARCH_*.md`, `docs/EXPERIMENT_*.md` | `7ffdc70`, this commit |

## 2. In progress
* None. All work is committed locally, and nothing has been pushed.

## 3. Outstanding GPT-2 tasks (PLANNED)

| ID | Task | Why | Priority |
|---|---|---|---|
| G1 | **Converged EV-vs-L0 frontier** (targets 16–256, 15k+ steps), with the P1 controller (plan-B) | Sparsity matching at other L0 values; architecture ranking at low L0 | P0 |
| G2 | **JumpReLU ablation:** threshold initialisation vs controller changes, layers 2/6/10 × 3 seeds (P2a); 30k-step convergence check (P2b) | Attribute P1 gains; validation EV still rising at 15k | P1 |
| G3 | **Training improvements** (plan-D): LR schedule, activation normalisation, dead-latent tracking and resampling | Dead latents up to 85 % (JumpReLU, early layers); not converged | P1 |
| G4 | **Published JumpReLU estimator / kernel-STE.** The kernel-STE training script is missing (only checkpoints exist). Re-implement, or label the result unreproducible | The surrogate JumpReLU is not the published method | P1 |
| G5 | **Width sweep** {1536, 3072, 6144, 12288} at layers 2/6/10 (plan-C) | Deep-layer headroom (H1) | P1 |
| G6 | **Residual two-stage SAE** vs a single SAE of the same total size (plan-E; not implemented) | Reconstruction improvement | P2 |
| G7 | **Method A replication:** more seeds, separately pre-registered; extend to layers 2 and 10 | H3; Method A narrowly missed its criterion | P0 |
| G8 | **E2E-inspired follow-up:** L0 control that holds under the KL term for ReLU/JumpReLU; replication of the TopK success; Method A + small-KL combination. *TopK layer-6 KL-weight ablation DONE (inconclusive: no truth trade-off between A and B; behaviour saturates by w = 0.25).* Remaining: ReLU/JumpReLU once S4 is solved; more seeds; w < 0.25 only if pre-registered | H4; the pre-registered longer-budget flag for ReLU/JumpReLU | P1 |
| G9 | **Public-SAE baselines under our pipeline** (`gpt2-small-res-jb`; v5-32k TopK): reconstruction, frozen probes, splicing at matched layers | External reference for the in-house SAEs | P1 |
| G10 | **AlignSAE-style concept latent** (proposed in the 2026-11-04 report §G; not implemented) | Alternative to the abandoned linear-teacher truth-aware fine-tuning | P2 |
| G11 | **Layer-0 JumpReLU fix:** threshold learning rate, dead-latent revival | H5; one unmatched baseline run | P2 |

## 4. Sparsity-matching corrections (PLANNED)

| ID | Correction | Current state | Priority |
|---|---|---|---|
| S1 | **ReLU L0 overshoot** (2–6 % above 128 in baselines). Use the P1 controller (anneal + freeze) for all new runs and report L0 error per run | P1 reduced it to 1–4.5 % | P0 |
| S2 | **Matched comparison via the EV-vs-L0 curve:** interpolate at equal L0 instead of relying on ±10 % (from G1) | Close calls (e.g. layer 2 ReLU ≈ JumpReLU) are unresolved | P0 |
| S3 | **Prompt-specific L0.** Matching is defined on OWT, but on GoT prompts ReLU uses 66, TopK 128, JumpReLU 46 latents, and on RepE 163 / 128 / 179. Report it as a covariate, and test whether truth/honesty gaps change when compared at equal *prompt* L0 (for example TopK with k matched to the other architectures' GoT L0) | Unaddressed; confounds every GoT/RepE architecture comparison | P0 |
| S4 | **L0 drift under KL objectives** (E2E ReLU/JumpReLU +5–11 %): a controller with a stronger or faster response, or a hard L0 constraint | Guard failed | P1 (with G8) |
| S5 | **Layer-0 JumpReLU unmatched run** (seed 42, L0 73.6): fix (G11) or document as an architecture failure mode | Excluded from rankings | P2 |
| S6 | **Fine-tune sparsity:** the historical fine-tunes (`sae_finetuning/layer_6_*`) are not matched (ReLU 65 → 144 on GoT). Never cite them as matched; all new fine-tunes keep the ±5 % guard | Documented | DONE (rule) |

## 5. Causal intervention and steering experiments (PLANNED)

| ID | Experiment | Why | Priority |
|---|---|---|---|
| C1 | **Pre-registered causal test (200 controls) on the uninterrupted SAEs at several layers** (e.g. 2, 6, 10) | It currently exists only at layer 6 with the historical SAEs | P0 |
| C2 | **Causal test on the improved SAEs** (Method A, E2E TopK) vs their bases, same protocol | Does improved faithfulness change causal results? | P0 |
| C3 | **SAE-in-the-loop feature interventions:** encode → clamp or ablate feature → decode + error term | Never run; tests SAE features as causal variables rather than directions | P0 |
| C4 | **GoT causal test**, e.g. patching or steering on `cities` statements with true/false preference as outcome (the only GoT set with a behavioural signal) | No GoT causal evidence exists | P1 |
| C5 | **Stronger GPT-2 honesty behavioural task:** the persona effect at the final token is only ~0.02 nats; GPT-2's honesty effect on the next statement token is about 0 | Current RepE behavioural outcome has little power | P1 |
| C6 | **Multiple-comparison handling** (BH across SAE comparisons) and power statement in every causal report | 27 SAE comparisons can't reach Bonferroni with 200 controls | P1 (rule) |
| C7 | **Gemma steering at full power:** 200 controls per strength (currently 20); other layers; `neg_cities` and numerical sets (the original `neg_cities` run crashed) | Current Gemma result is descriptive | P1 |

Activation splicing (Phase 0) is behavioural faithfulness, not steering. Keep it separate in all reports.

## 6. Outstanding Gemma-2-2B tasks

| ID | Task | Status | Priority |
|---|---|---|---|
| M1 | Activation extraction pipeline (HF bf16 backend, since TransformerLens is OOM-killed on the 15 GB-RAM host); GoT/RepE final-token caches; grouped splits | PLANNED | P1 |
| M2 | Original-activation truth/honesty probes per layer (frozen-probe protocol). The `aditi` layer table used a leaky row split, so it must be redone with grouped splits | PLANNED | P1 |
| M3 | **Gemma Scope SAE baselines** (JumpReLU; ~8 GB download, ~3–4 h by the 2026-11-06 estimate): reconstruction, frozen-probe gaps, splicing at chosen layers | BLOCKED: weights not cached; disk/RAM check needed | P1 |
| M4 | Gemma steering with 200 controls (C7) and a context-matched direction across layers | PLANNED | P1 |
| M5 | Gemma version of the best GPT-2 improvement (e.g. Method A), only if M3 works | PLANNED | P2 |

## 7. Final cross-model and cross-condition comparison (PLANNED, P0)
* **X1. Cross-condition faithfulness table** (proposal Phase 6), crossing:
  * layer × architecture (× model, where available) × metric category: reconstruction, representation preservation, behavioural splicing, causal, steering;
  * all at matched sparsity (S2, S3);
  * each cell with its evidential status.
* **X2. GPT-2 vs Gemma-2-2B comparison**, restricted to conditions that are actually comparable: same datasets, frozen-probe protocol, comparable relative depth. State explicitly which cells are missing.
* **X3. Final pre-registered TEST evaluation** of the selected configurations (all layers, 3 seeds). Test splits are used once, after all selection is finished. The GoT layer-6 test split has already been scored twice for the historical fine-tunes, so pre-register a plan for that.

## 8. Final capstone deliverables (deadline 2026-12-08 per `proposal.md`) (PLANNED, P0)

| ID | Deliverable | Current state |
|---|---|---|
| F1 | **Final report** (`reports/Latex_report/main.tex`) | Literature review exists. Introduction, Methods, Experiments/Results, Discussion and Conclusion are still `TODO`. A 7-page version with Methods/Results exists only on `origin/dhruv` (`f74d79a`) [BRANCH-ONLY] |
| F2 | Methods and results drawn from `docs/RESEARCH_PROGRESS.md` and `docs/EXPERIMENT_RESULTS_SUMMARY.md`; figures from `results/*/figures/` (vector formats) | Not started |
| F3 | Journal submission | Not started |
| F4 | Final presentation slides (`presentation/`), poster, recorded video, 1-minute project video | Not started. The 2026-10-20 preliminary slides are not in this branch |
| F5 | Tableau dashboard (proposal "Planned Tableau Visualization") | Not started; optional scope |
| F6 | Progress reports for the current weeks (`reports/Progress_Report/`) | Last on this branch: 2026-09-29; 2026-10-25 only on `origin/dhruv` |

## 9. Missing baselines and controls (cross-reference)

| Missing item | Task |
|---|---|
| Converged sparsity sweep | G1 |
| Width other than 3072 | G5 |
| Published JumpReLU | G4 |
| Public-SAE baselines under our pipeline | G9 |
| Gemma SAEs | M3 |
| Prompt-level sparsity matching | S3 |
| Causal tests beyond layer 6 / on new SAEs | C1, C2 |
| SAE-in-the-loop interventions | C3 |
| GoT causal test | C4 |
| Stronger honesty behaviour | C5 |
| Seed replication of close calls (Method A, E2E TopK, KL-weight ablation; truth-gap seed sd ≈ 0.02) | G7, G8 |

## 10. Abandoned or superseded
* **Linear-teacher truth-aware fine-tuning in its current form:** abandoned (`RESEARCH_DECISIONS.md` D16).
* **Superseded or invalid** (see `docs/results_registry.md`):
  * earlier "SAE steering" correlations;
  * re-trained-probe faithfulness;
  * `architecture_summary.csv`;
  * warm-start/resume cross-layer runs;
  * Gemma row-split baseline.

## 11. Housekeeping awaiting approval (not to be done without explicit approval)
* Deleting `results/professor_review_corrections/`, `scripts/professor_review/`, `src/tests/test_professor_review.py` (all migrated), and the two `.bak` files.
* Pushing `sae-research` to GitHub.
* Merging or archiving the unmerged team branches (`dhruv`, `simba-*`, `aditi`), and bringing the 2026-10-25 report and CI from `origin/dhruv` into this branch. **Do not merge** without approval.

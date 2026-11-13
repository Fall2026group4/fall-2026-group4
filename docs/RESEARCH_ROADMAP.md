# Research roadmap (as of 2026-11-10; final deadline 2026-12-08)

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
| KL-weight ablation, layer 6 TopK (G8, part): verdict INCONCLUSIVE | `kl_weight_ablation_layer6_topk/` | `2c87a6b` |
| Documentation system | `CLAUDE.md`, `docs/RESEARCH_*.md`, `docs/EXPERIMENT_*.md` | `7ffdc70`, `3377116` |
| Repository consolidation for upload: professor-review folders retired (contents verified in the dataset folders), README rewritten, research dependencies pinned, registry rebuilt | `README.md`, `docs/migration_manifest.csv` (`retire_to_backup` rows) | `875bc7f` |
| R1 RepE honesty-preservation fine-tuning, layer 6 (27 runs, validation): ReLU INCONCLUSIVE (primary met, RepE-prompt L0 guard failed), TopK/JumpReLU NOT SUPPORTED | `results/repe/r1_honesty_preservation_layer6/` | `d17bb00`, `211218f` (pre-registration), `d283a49` |
| R2 SAE-in-the-loop honesty-feature clamp, layer 6 (9 base SAEs with all controls, 18 R1 SAEs exploratory; validation): NOT SUPPORTED (descriptive) for all architectures and k; test stage deferred | `results/repe/r2_sae_in_the_loop_steering_layer6/` | `d17bb00`, `b232ce4` (pre-registration), this commit |

## 2. In progress
* **M0–M5 measurement programme (D23, approved 2026-11-08).** P0.
  * **M0** within-layer reanalysis of existing validation results → `results/shared/m0_within_layer_reanalysis/`. **DONE 2026-11-08** (validation): splice damage tracks reconstruction within layer; truth-readout gap does not (formal pass fragile).
  * **M1** RepE-LM equal-length persona task + gates K1–K7 → `data/repe/repe_lm/`, `results/repe/repe_lm_gates/`. **DONE 2026-11-08:** K1–K7 all PASS (K5 narrowly, 0.701 at L8/L11). M2 and M3 allowed by the gates; awaiting user approval.
  * **M2** four-level information faithfulness (GoT cities + RepE-LM persona/neutral; PCA- and noise-matched references). **DONE 2026-11-08** (validation): no cell Preserved or Readout shifted; GoT frozen collapse mostly noise-like; SAE never better than PCA.
  * **M3** positive-control causal site sweep in original GPT-2. **DONE 2026-11-08:** L1 slot and L11 final selected and confirmed (near-endpoint sites); effect hands over from slot to final token at layers 7–8; T1 not persona-specific.
  * **M4** SAE patch decomposition + feature patches. **DONE 2026-11-08** (validation): reconstruction keeps 93–97 % at L1 slot, 62–68 % at L8/L11 final; effect distributed over ≥ 64 features.
  * **M5** one confirmatory test pass. PLANNED: needs written approval. Must include the RepE-LM test duplicate check (deferred from M1).
* **Instructor Review #8 (2026-11-08) response programme.** P0. Issue matrix: README §7–§9 and §14; review text read from
  the project board (ticket "Instructor Review #8 — 2026-11-08").
  * **IMP-L** multi-layer improvement: OWT-only control, Method A and output-KL (E2E-B) fine-tuning at layers 3, 8 and 9,
    3 architectures × 3 seeds, L0 guards, CIs → `results/geometry_of_truth/multilayer_finetune/`. Due at the 11-10
    session. **DONE 2026-11-09 (VALIDATION):**
    * Method A truth criterion met in 4/9 cells. The frozen-probe gain is not matched by re-trained readers.
    * Output-KL behaviour is creditable in 5/9 cells; in the other 4/9 (ReLU and JumpReLU at layers 8 and 9) the L0
      guard failed.
    * Output-KL truth regressed at L3 ReLU and L3 TopK.
  * **IMP-L follow-ups.** PLANNED, not approved:
    * an L0-matched output-KL variant for ReLU/JumpReLU at deeper layers (controller strength), as a new pre-registration;
    * more seeds for the 5 Method A cells where the direction is consistent but the 2 sd rule fails.
  * **Priority set by the user on 2026-11-09:** E3 first, then E2, then E1. Each one needs the user's approval of its
    pre-registration (hypotheses, comparisons, metrics, GPU estimate, training requirements) before anything runs.
  * **E3** causal preservation (M4 protocol, L8 final) of the 27 IMP-L layer-8 fine-tunes plus the 9 base SAEs.
    **DONE 2026-11-10 (VALIDATION):**
    * Method A NO IMPROVEMENT 3/3;
    * output-KL TopK NO IMPROVEMENT;
    * output-KL ReLU/JumpReLU NOT CREDITABLE (gains of +0.018 to +0.022 only with L0 drift).
    `results/repe/e3_finetuned_causal_preservation/`.
  * **RLM-RP** RepE-LM representation preservation and robustness, 216 SAEs. **DONE 2026-11-10 (VALIDATION):**
    * the persona loss is real (re-trained gap; 2–3× the noise reference);
    * the H−U difference is distorted;
    * Method A has no effect;
    * output-KL gives tiny gains with more active latents.
    `results/repe/repe_lm_representation_preservation/`.
  * **NEW (from E3/RLM-RP), PLANNED, needs approval:** an L0-matched output-KL variant (G8/S4) is now the only way to
    test whether output-KL improves causal faithfulness. A stronger controller, or TopK-only with more seeds; TopK's
    per-SAE E3 CIs (±0.06–0.09) need more pairs or seeds to detect changes below about 0.03.
  * **NEW, PLANNED:** a minimum-detectable-effect statement for E3/M4 (Review #8 critique 3), from the existing
    bootstrap CIs; no new run.
  * **Independent review 2026-11-10 (`docs/2026-11-10_independent_research_review.md`). PROPOSED, none approved; each
    needs a pre-registration and the user's approval:**
    * **Status 2026-11-10 (D29, RESEARCH_PROGRESS §34): R1, R2, R4, R5 and R7 DONE.** R3 (Gemma) is planned, not
      run (D30). R6 is not run and stays recommended.
    * **R1 (essential):** run E2 (res-jb external anchor) unchanged. **DONE**: conclusions transfer (E2 README §2).
    * **R2 (essential):** data-scaled retraining. Streaming trainer, about 100M OWT train tokens, width 12,288,
      TopK+AuxK / JumpReLU-STE / ReLU-L1, L0 64 and 128, layers 1/8/11 × 3 seeds, random-frozen-decoder control;
      then the full ladder. About 4–6 h GPU plus about 1 day of engineering. **DONE** (L0 128 only; the modern
      JumpReLU was abandoned in the pilot): reconstruction and behaviour gains confirmed on TEST (M5).
    * **R3 (essential):** Gemma-2-2B ladder with Gemma Scope 16k SAEs at about 3 layers and several published L0s.
      The RepE-LM prompts must be re-tokenised and the K1/K2 gates re-run for Gemma first. Load Gemma via HF bf16 on
      the GPU (TransformerLens CPU loading exceeds 15 GB RAM). About 4–8 h GPU.
    * **R4 (recommended):** short KL+MSE fine-tune only at fixed L0 (TopK/BatchTopK; JumpReLU with an explicit L0
      constraint), with an MSE-only control. **DONE** (TopK, L8/L11): ΔCE −16 to −17 % confirmed on TEST; causal
      regression at L11 (validation).
    * **R5 (recommended):** a supervised rank-k causal skyline (DAS-style) and a random-decoder lower bound at the L8
      final site. **DONE** (all three M4 sites; the random decoder is in R2).
    * **R6 (recommended):** a real-activation-direction perturbation reference for M2 (arXiv 2410.12555).
    * **R7 (essential):** M5, after the protocol is frozen. **DONE 2026-11-10** (`results/shared/m5_confirmatory_test/`).
    * **Not recommended:** LoRA on the model; more fine-tuning of the current SAEs; Gemma SAEs from scratch;
      transcoders.
  * **E1** TopK at task-level matched L0, at layers 1, 8 and 11 on GoT cities and RepE-LM persona. 36 new TopK SAEs
    (12 k* × 3 seeds, k* from TRAIN prompts); about 1.2 h GPU; no OWT test evaluation. **DRAFT PRE-REGISTRATION** in
    `results/shared/e1_task_l0_matched_topk/README.md`; awaiting approval.
  * **E2** public `gpt2-small-res-jb` reference at layers 8 and 10 (both cached; no download; layer 6 dropped). No
    training; about 25–40 min GPU. **DONE 2026-11-10** (approved under the full execution approval; layers 1 and 11
    added as an extension).
  * **Team coordination (NEW).** Branch `simba-method-a-multilayer` duplicates IMP-L under the same script numbers (22/23),
    and it uses layer 5 instead of layer 3. Agree on one experiment, and renumber, before either merges. Open PRs #20, #21
    and #24 delete files this branch cites.
  * **Paper** in `research_paper/Latex/` with the verified M0–M4, IMP-L, RLM-RP and E3 tables and figures (all generated). IN PROGRESS: compiles (29 pp, 0 overfull boxes, 0 undefined references); figures deterministic. Improvement section 10b, with M5, E2, R2, R4 and R5, added 2026-11-10 (37 pp, clean). Remaining placeholders: E1; the minimum-detectable-effect statement; the pipeline diagram (F8); the per-experiment command list for the older experiments.
  * **Repository items from Review #8:** SVG/PDF figures (F7), `.drawio` source + `.svg` (F8), CI on `main`, logging in
    new code (G12; pre-registered scripts are not edited, so their recorded hashes stay valid), AxBench/SAEBench/Paulo et
    al. in the bibliography, progress reports since 09-29, presentation decks, PR merge plan (#16, #17, #20, #21, #22,
    #24), minimum detectable effect statement. PLANNED.
  * **Push and PR** of the local commits after `33dcb9c` (R1, R2, M0–M4, IMP-L). **Push DONE 2026-11-12** (`875bc7f..b927f27` and the cleanup commits); new PR to `main` prepared as a draft (PR #23 from this branch was already merged). Review package:
    `docs/2026-11-09_review_package_sae_research.md` (M0 re-run and paper regeneration verified byte-identical on a
    clean export). Needs user approval and an independent
    teammate review with a reproducibility check before merging.
* **R1 / R2:** reclassified HISTORICAL (D23). Their deferred test passes are not planned.
* **R1/R2 history (validation complete, see §1):** R1 §6 test pass, R1-L0 variant, R2 test stage, R2 follow-ups and the held-out-wording set were open follow-ups; all are superseded by M1–M4 (D23) and are not planned.
* **Audit finding:** validation statement 326 near-duplicates train statement 589. Excluded from R1/R2 and M1.

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
| C3 | **SAE-in-the-loop feature interventions:** encode → clamp or ablate feature → decode + error term | First run as R2 (layer 6, RepE honesty, validation, 2026-11-08): NOT SUPPORTED. Other layers, datasets and ablation-type edits not run | P0 |
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
| M3 | **Gemma Scope SAE baselines** (JumpReLU; ~8 GB download, ~3–4 h by the 2026-11-06 estimate): reconstruction, frozen-probe gaps, splicing at chosen layers | PLANNED (D30): load Gemma with HF bf16 (7.3 GB VRAM, 7.7k tok/s); port the ladder to HF hooks; re-tokenise RepE-LM and re-run its gates; disk has 97 GB free | P0 (next) |
| M4 | Gemma steering with 200 controls (C7) and a context-matched direction across layers | PLANNED | P1 |
| M5 | Gemma version of the best GPT-2 improvement. Per D30: R2-recipe TopK/BatchTopK at one middle layer (100M tokens; about 3.6 h activation generation per pass), the full ladder, R5 skyline, optional R4 KL as a behaviour-only add-on. Not Method A. | PLANNED | P1 |

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
* ~~Deleting `results/professor_review_corrections/`, `scripts/professor_review/`, `src/tests/test_professor_review.py`~~ **DONE 2026-11-07** (approved by the user): retired to `../backups/2026-11-07_retired_professor_review_folders/` after hash verification; Git history keeps the tracked files. The two `.bak` files remain on disk, now gitignored; deleting them still needs approval.
* **Back up the EC2-only checkpoints** (about 9 GB of `*.pt`, plus activation caches). Only the 2026-11-06 pre-migration tarball exists; checkpoints created after it (all-layer baseline, P1, truth-aware pilot, Method A, E2E, KL ablation) have no off-machine copy.
* Pushing `sae-research` to GitHub.
* Merging or archiving the unmerged team branches (`dhruv`, `simba-*`, `aditi`), and bringing the 2026-10-25 report and CI from `origin/dhruv` into this branch. **Do not merge** without approval.

## 2026-11-10 additions (GPT-2 improvement phase)
* **RepE-LM test split:** build test prompts from the held-out persona statements, with the M1 duplicate check, so
  that persona, RLM-RP, M4 and R5 results can be confirmed. Not started. P1.
* **R6 real-direction noise reference for M2:** still recommended. Not run. P2.
* **R2 scaling ladders** (10M/25M/50M checkpoints, descriptive): deferred until after M5; not yet run. P2.
* **Task-level L0 for BatchTopK/ReLU** is unmatched on RepE/GoT prompts (141–216 vs 128); E1 still addresses this. P2.

## 2026-11-10 additions (final defense audit, `docs/final_defense/EXPERIMENT_AUDIT.md`)
* **Isolate token count from recipe:** evaluate the existing R2 10M/25M/50M checkpoints with the ladder on
  VALIDATION. No training needed. P1 (this raises the existing P2 item, because "longer training" is otherwise
  unanswerable).
* **Random-decoder replication:** more frozen-decoder seeds (currently n = 1), because it ties or beats trained SAEs
  on GoT SAE − PCA. P1.
* **Process:** freeze protocols in a separate committed file before any future test pass (D31). Rule.
* **Off-machine backup of the R2/R4 checkpoints** (in addition to §11). P0 before the deadline.

## 2026-11-12 additions (GPT-2 audit and improvement plan, `docs/2026-11-12_gpt2_audit_and_improvement_plan.md`)
All PLANNED, none approved; each needs a pre-registration and the user's approval. GPT-2 only (Gemma: teammate).
* **H1 RepE-LM test split** (raises the 11-10 P1 item to P0): every persona/causal claim is VALIDATION without it.
* **T1 GoT measurement upgrade** (P0): minimum detectable effect from existing bootstrap arrays (discharges the open
  MDE item); paired per-statement SAE comparisons; inference-time task-L0-matched TopK; cities → neg_cities transfer.
* **X0 data-scaling ladder** on the existing R2 10M/25M/50M checkpoints (P0; supersedes the 11-10 "isolate token
  count" item; checkpoints are mid-schedule, disclose).
* **X1 inference-time coefficient optimisation** (scale / support refit / pursuit at fixed L0) on R2 and old SAEs (P0).
* **X2 ReLU shrinkage fine-tune** (scale or decoder only), conditional on X1's fitted scale (P1).
* **H2 contrast-consistent fine-tune at fixed L0** (R2 TopK, L8/L11; raw-Δ loss, no probe; held-out templates) (P1).
* **I1 KL × contrast factorial** at fixed L0 (P1, after H2).
* **X3 modern JumpReLU pilot** (fixed λ / Anthropic 2025 loss / BatchTopK init), time-boxed (P1).
* **R5 feature-patch arm on R2 SAEs** (P2; R5 used old SAEs only).
* **M6 test pass** after a protocol freeze committed before the run (D31).

## 2026-11-12 additions (documentation cleanup)
* DONE: curated docs (`docs/{research_overview,methodology,experiment_history,results_summary,limitations_and_future_work}.md`), neutral renames, archive log, README navigation and reproduction commands.
* OPEN: off-machine backup of about 10 GB of R2/R4 checkpoints and the activation caches (still P0; Git does not hold them).
* OPEN: teammate review of the draft PR (requested from two teammates); merge needs their review and the user's decision.

## 2026-11-13 additions (audit remediation; register in `docs/remediation/issue_register.md`)

* IN PROGRESS: PCA-CM, the capacity-matched PCA reference (S1), VALIDATION only; `results/shared/pca_capacity_matched/`.
* PARTIALLY RESOLVED: H1 RepE-LM TEST split. Data and protocol are frozen (`data/repe/repe_lm_test/`,
  `results/repe/repe_lm_test_protocol/`); the evaluation is NOT RUN and awaits user approval (about 1–2 GPU hours).
  Next: write script 29, smoke-test it on validation, hash it into the pre-registration, run once.
* PLANNED (proposed, not approved): a mediated SAE-in-the-loop test. An edit at layer ℓ with SAEs spliced at
  ℓ+1…ℓ+k, so that the effect must pass through SAE nonlinearities (D33).
* DONE (VALIDATION): PCA-CM. **Awaiting the user:** revise the PCA claim in the paper (`00_abstract`,
  `07_results_information`, `10b_improvement`, `12_discussion`, `14_conclusion`) and the curated docs (D34).
* DONE: sensitivity/MDE analysis (`results/shared/power_sensitivity/`). TODO: add the MDE sentences to the paper
  (§11 statistics, the `\pending` note in §causal) after the PCA decision.
* OPEN: off-machine backup (needs a destination); linking the commit email; posting the drafted replies
  (`docs/remediation/review8_response_draft.md`); the team merge order.

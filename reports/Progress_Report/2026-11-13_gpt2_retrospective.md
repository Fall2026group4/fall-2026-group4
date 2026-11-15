# Progress report: GPT-2 Small SAE faithfulness (Aditi), weeks 6–11, compiled 2026-11-13

**This is a retrospective compilation, written on 2026-11-13.** These entries were not submitted weekly. Each entry is
taken from the dated journal (`docs/RESEARCH_PROGRESS.md`, section numbers in brackets) and the Git history of
`sae-research`. Weeks with no dated GPT-2 evidence are marked as such and left empty. Teammates' Gemma and
branch work is not described here.

| Week of | GPT-2 work (evidence) | Status label |
|---|---|---|
| 2026-10-05 | Team-branch experiments; none on `sae-research` [§3] | BRANCH-ONLY |
| 2026-10-12 | Team-branch experiments [§3] | BRANCH-ONLY |
| 2026-10-19 | Team-branch experiments [§3] | BRANCH-ONLY |
| 2026-10-26 | In-house OWT SAE pipeline (ReLU/TopK/JumpReLU, width 3,072) [§4]; GoT layer-6 and RepE pipelines [§5] | COMPLETED, partly superseded |
| 2026-11-02 | Professor-review corrections [§6]; repository migration [§7]; 108-run all-layer OWT baseline [§8]; all-layer frozen-probe truth and honesty [§9]; spliced behavioural evaluation [§10]; P1 sparsity pilot [§11]; truth-aware circularity audit and corrected pilot [§12–13]; Method A, E2E-inspired B/C, KL ablation [§14–16]; R1/R2 honesty (layer 6) [§19–20]; PR #23 merged to `main` (11-07) | TEST (108-run table, frozen probes); VALIDATION otherwise |
| 2026-11-09 | M0–M4: within-layer reanalysis, RepE-LM task, four-level information, causal site sweep, SAE causal decomposition [§22–28]; IMP-L fine-tuning at layers 3/8/9 [§30]; paper skeleton [§29, §31]; E3, RLM-RP [§32]; R2 data-scaled SAEs, R4 KL fine-tune, R5 skyline, E2 res-jb, **M5 single test pass** [§34]; defense audit [§35] | M5 CONFIRMED TEST (OWT, GoT); VALIDATION otherwise |
| 2026-11-12 | Read-only audit and improvement plan [§36]; documentation consolidation; PR #27 opened [§37] | no new result |
| 2026-11-13 | Audit remediation [§38]: corrected GoT and NMSE-range docs; capacity-matched PCA (VALIDATION); sensitivity (MDE) analysis; RepE-LM test split built and protocol frozen (not evaluated); SAE-in-the-loop equivalence paragraph; 34 new tests and a CI workflow; vector figures and an editable diagram; course report sections | see §38 |

## Current open items (from `docs/remediation/issue_register.md`)

* RepE-LM test evaluation: needs approval and about 1–2 GPU hours.
* Off-machine checkpoint backup: needs a destination.
* Link `as650@gwu.edu` to the GitHub account.
* Team merge order for #27, #26, #28, #21 and #24.
* Journal choice and template; revised presentation deck; poster; videos.

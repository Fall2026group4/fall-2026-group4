# Remediation register: 2026-11-12 independent audit of `sae-research` / PR #27

Started 2026-11-13 from the 2026-11-12 independent audit (D1–D15, S1–S12, U1–U4, Review #8 compliance matrix). The
evidence was re-checked against the code, the CSVs and GitHub on 2026-11-13. Scope: GPT-2 Small only; no Gemma or
teammate file was changed.

**Statuses:**
* OPEN; IN PROGRESS;
* FIXED AND VERIFIED; PARTIALLY RESOLVED;
* BLOCKED; REQUIRES TEAM ACTION; NOT APPLICABLE.

An item is FIXED only when the evidence or the GitHub state changed, not when documentation was added.

Commits are on `sae-research`. "this branch" means the commit series 7e8664f…HEAD; individual hashes are listed in
`docs/EXPERIMENT_CHANGELOG.md` (2026-11-13).

## A. Confirmed defects and process problems (D)

| ID | Pri | Problem | Evidence | Fix | Validation | Status |
|---|---|---|---|---|---|---|
| D1 | P1 | GoT task described as cities-only / "neg_cities unused" | `m2_four_level_faithfulness.py:66–69` and `03_original_truth_probes.py:24` pool `cities` + `neg_cities`; split CSVs give 2,992 statements, test 226 + 226 = 452 | Corrected `results_summary.md`, `limitations_and_future_work.md`; marked errata in the 2026-11-12 audit (3 places) | grep finds no remaining cities-only claim about M5; new test `test_cities_and_neg_cities_assign_each_city_to_the_same_split` asserts the counts | FIXED AND VERIFIED (`bf5865a`) |
| D2 | P1 | NMSE range "40–74 %" | M5 `comparisons.csv`, C1–C3: 30.8–73.7 % (ΔCE 35.5–78.8 %) | Corrected `research_overview.md:84`, `results_summary.md:199` | Range recomputed from the CSV; the paper and cheatsheet already said 31–74 % | FIXED AND VERIFIED (`bf5865a`) |
| D3 | P1 | #21/#24 delete a file and five scripts that `scripts/geometry_of_truth/10_layer10_frozen_probe.py` uses at runtime | GitHub PR file lists (§2.1 of `merge_plan.md`) | Merge plan: #27 first; #21/#24 re-scoped to keep those files | — | REQUIRES TEAM ACTION |
| D4 | P1 | #26 (and now #28) edit the same 7 ledgers; #26 reuses script numbers 22/14/15; #28 reuses GoT 22/23 | PR file lists | Merge plan §3: order #27 → #26 → #28; renumbering proposed | — | REQUIRES TEAM ACTION |
| D5 | P1 | 27/27 PR commits authored as `as650@gwu.edu`, not linked to a GitHub account | GitHub API `author.login` = null for every commit | Steps in §F. No history rewrite; author identity left as is (the instructor asks for this address to be linked) | Re-check `gh api repos/Fall2026group4/fall-2026-group4/pulls/27/commits` after linking | REQUIRES USER ACTION |
| D6 | P2 | No CI (0 checks on #27) | `gh pr view 27` statusCheckRollup = [] | `.github/workflows/gpt2-validation.yml`: hygiene, py_compile, imports, pytest, M5 verification; CPU only | GitHub run 43375816244 on PR #27: success (all steps) | FIXED AND VERIFIED on PR #27 (on `main` only after merge) |
| D7 | P2 | Stale README rows (paper length, PR list, `main` status, report) | `README.md` §9 | README §9 updated | Rows checked against `gh` and the repository | FIXED AND VERIFIED |
| D8 | P2 | No tests for `modern_sae`, R2 trainer, decision rules, splicing | `src/tests` imported none of them | `test_modern_sae.py`, `test_decision_rules_and_splits.py` (+34 tests) | 69/69 pass locally and on a clean export | FIXED AND VERIFIED (`72e466b`) |
| D9 | P2 | Course report: 5 TODO sections | `reports/Latex_report/main.tex` | GPT-2 sections written from verified numbers; Gemma subsection PENDING (teammates); title/abstract framing flagged | 0 TODO; compiles to 5 pp, 0 undefined citations | PARTIALLY RESOLVED (team content and title pending) |
| D10 | P2 | 50 PNG results figures; no `.drawio` | `git ls-files '*.png'` | 21 figures exported to SVG + PDF from unchanged scripts; editable `.drawio` + `.svg` pipeline diagram | PNG re-renders are pixel-identical to the committed PNGs | PARTIALLY RESOLVED (31 PNGs have no safely re-runnable script: EDA, training curves, `layer_comparison`, Gemma, diagram images) |
| D11 | P2 | 430 `print` calls; logging in 1 file | grep | New shared code (`ci_repo_hygiene.py`, `export_vector_figures.py`, `power_sensitivity_analysis.py`) uses `logging`. Pre-registered, hashed scripts are not edited | — | PARTIALLY RESOLVED (by design) |
| D12 | P3 | Two `scripts/repe/11_*` scripts | `ls scripts/repe` | Documented; not renamed (paths are hashed) | — | FIXED (documented) |
| D13 | P3 | Module-global monkey-patching; about 20 scripts import scripts by path | `sae_ladder_evaluation.py:162–163`, `m5_confirmatory_test.py:157` | Not refactored: the affected scripts are hash-frozen. New code follows the same pattern only on its own module instances. The decision-rule tests now pin the behaviour of the imported functions | Tests | PARTIALLY RESOLVED (refactor needs new pre-registrations) |
| D14 | P3 | Renames landed in `e2becfa` | disclosed | none | — | NOT APPLICABLE (disclosed; no history rewrite) |
| D15 | P3 | Stale issues #4, #7 (MakaveliSimz), #11 (aditi650); no tickets for the 11-03 list | `gh issue list` | Listed in `merge_plan.md` §5 | — | REQUIRES TEAM ACTION |

## B. Scientific limitations (S)

| ID | Pri | Issue | Evidence | Planned fix / what was done | Validation | Status |
|---|---|---|---|---|---|---|
| S1 | P1 | PCA reference matched on error, not capacity | M5 `got_cities__pca_k`: 298–519 (trained; R2/R4 404–519) vs 143–164 (random decoder). The audit's "400–519" omitted the old ReLU (298–355) | PCA-CM, pre-registered, VALIDATION only: k = 128, k = task L0, error-matched replication. Error-matched: SAE WORSE 13/20 cells, never better (replicated). k = 128: SAE BETTER 3/20, never worse. Task-L0: BETTER 8/20, never worse | `results/shared/pca_capacity_matched/` (0 test rows; hash-verified) | **FIXED AND VERIFIED (2026-11-13, user-approved D34 wording):** paper (abstract, questions, §info, §improvement + new §PCA-CM and Table `t_pcacm`, §statistics, §discussion, §limitations, §conclusion), curated docs, defence docs, report, slides. "Significantly better" NOT used: the rule is per-cell without correction; approx. Holm keeps 1/3 and 4/8 (`posthoc_multiplicity/`). Persona not re-tested at equal dimensions; no test confirmation (OPEN, needs a new pre-registration) |
| S2 | P1 | GoT metric underpowered; no MDE | Seed sd 0.004–0.024 | `scripts/shared/power_sensitivity_analysis.py` from existing outputs: re-fit comparisons can detect about 0.048–0.091 AUROC; observed \|d\| ≤ 0.025 | `results/shared/power_sensitivity/` | FIXED AND VERIFIED (paper §statistics, §improvement M5 inference, §discussion, §limitations; README; results_summary B2c) |
| S3 | P1 | No RepE-LM test split | `prompts.csv` has train/validation only | Test data built (86 statements, 2,408 rows, test-only wording); protocol frozen (D32) | All construction and leakage checks pass; hashes verified | PARTIALLY RESOLVED (evaluation not run: needs approval and about 1–2 GPU h) |
| S4 | P1 | Instructor's Review #8 summary cites superseded findings | Review #8 text vs M0/M2/D23 | Draft reply: `review8_response_draft.md` (not posted) | — | REQUIRES USER ACTION (posting) |
| S5 | P1 | SAE-in-the-loop equivalence | `24_sae_patch_decomposition.py:252–259` | Paper §causal: equivalence argument and renamed paragraph; D33; mediated design proposed | — | PARTIALLY RESOLVED (argument stated; mediated test not run) |
| S6 | P2 | Persona probe may track word identity | M1 gates: neutral swap decodable 0.62/0.75 (L8/L11); persona probe → neutral swaps 0.40–0.47; LOPO cross-template 0.70; held-out T5/T6 0.62–0.73 | Assessment in `docs/remediation/repe_validity_assessment.md`; HT4 in the test protocol | — | PARTIALLY RESOLVED (assessed from existing data; test-only wording awaits the test pass) |
| S7 | P2 | Tiny causal effect; no MDE | T_full 0.022 nats at L8 | Sensitivity analysis (M4 part): T_full CI low ≥ 0.78 × mean (ratio stable); preservation MDE 0.054–0.107 at L8, so R2-vs-old +0.101 (VAL) is borderline | `results/shared/power_sensitivity/m4_causal_sensitivity.csv` | FIXED AND VERIFIED (L8 gain labelled borderline in the abstract, §causal, §improvement, §conclusion, README, results_summary C3, slides) |
| S8 | P2 | R2 confounds data, width and recipe | 10M/25M/50M checkpoints unevaluated | Not run (X0, GPU) | — | OPEN (proposed X0) |
| S9 | P2 | Seeds share one stream; Welch covers seed variance only; no multiplicity correction | M5 design | Stated in the sensitivity analysis (statement-level + seed-level MDE) | — | PARTIALLY RESOLVED |
| S10 | P2 | Task-prompt L0 unmatched (GoT 63–111; RepE-LM 134–216) | M5 rows, ladder | PCA-CM `task_l0_matched` reference addresses the GoT readout; E1 not run | — | PARTIALLY RESOLVED |
| S11 | P2 | R4 trade-off (NMSE +2 %; L11 causal −0.024) | M5, ladder | Already reported as REGRESSION in docs and paper | — | FIXED AND VERIFIED (disclosure) |
| S12 | P2 | M5 pre-registration committed with results; test rows previously exposed | M5 history | Disclosed (D31); PCA-CM protocol committed before its run finished (`7e8664f`) | — | NOT APPLICABLE (cannot be undone; disclosed) |

## C. Unverified risks (U)

| ID | Risk | What was done | Status |
|---|---|---|---|
| U1 | About 15 GB of checkpoints made after the only backup, on one volume | Inventory and SHA-256 manifest (§E). No configured off-machine destination exists (no AWS/rclone credentials; the second NVMe disk is unmounted, probably ephemeral instance storage). Nothing copied, moved or deleted | BLOCKED (needs a destination and credentials from the user) |
| U2 | #26 results may contradict #27 | Overlap mapped (`merge_plan.md` §2.5) | REQUIRES TEAM ACTION |
| U3 | Ruff findings | `ruff` not installed; CI uses py_compile instead | OPEN (not verifiable here) |
| U4 | Duplicate IMP-L-like scripts under GoT 22/23 | Now in PR #28 (`22_method_a_multilayer_finetune.py`, `23_…evaluation.py`) | REQUIRES TEAM ACTION |

## D. PR #27 review and merge risks

| Risk | Evidence (2026-11-13) | Status |
|---|---|---|
| Review | 1 APPROVED review (basquedhruv, "Works.", 11-12); MakaveliSimz requested and has asked 2 questions (§4 of `merge_plan.md`) | Reply drafted, not posted: REQUIRES USER ACTION |
| Review quality (Review #8) | The reviewer should re-run one script and report what matched | Suggested `m5_independent_verification.py` in the PR description |
| CI | Workflow added | see §G |
| Draft | PR is a draft | The user decides when to mark it ready |

## E. Backup inventory (U1)

Measured on 2026-11-13. The manifest is `backup_manifest.sha256` in the session scratchpad. It is not committed
(it lists local-only files), but it can be regenerated with the command in `docs/remediation/backup_plan.md`.

| Asset | Size | Files | Off-machine copy |
|---|---|---|---|
| `results/openwebtext/r2_scaled_saes/checkpoints` | 9.1 GB | 156 | none |
| `results/openwebtext/convergence` | 5.4 GB | 221 | partial (pre-11-06 runs in the same-disk tar) |
| `results/geometry_of_truth/sae_finetuning` | 2.5 GB | 154 | none |
| `results/openwebtext/r4_kl_finetune` | 865 MB | 22 | none |
| `results/repe/sae_finetuning` | 505 MB | 28 | none |
| `data/activations` (OWT caches) | 2.3 GB | 253 | partial (same-disk tar) |
| `data/repe/activations`, `data/got/activations` | 761 MB + 245 MB | 31 | none |
| `data/openwebtext/stream_shards`, `stream_tokens` | 576 MB + 409 MB | 11 | none (re-downloadable) |
| `.pt` files newer than the 2026-11-06 tar | 15 GB | 434 | **none** |

## F. Linking the commit email (D5): steps for the user

1. GitHub → Settings → Emails → Add email address → `as650@gwu.edu` → confirm the verification email.
2. GitHub then attributes every past commit with that author email to the account. No history rewrite is needed.
3. Check: `gh api repos/Fall2026group4/fall-2026-group4/pulls/27/commits --jq '.[].author.login'` should print
   `aditi650` for every commit.

## G. GitHub state after the push

* Pushed `5d79fa5..8d52e14` (fast-forward) to `origin/sae-research` on 2026-11-13.
* CI run 43375816244 (pull_request, `8d52e14`): **success**. Every step passed: hygiene, py_compile, imports, 69
  tests, M5 verification.
* The PCA-CM / sensitivity results commit is **local only** (not pushed), pending the user's review of the conclusion
  change.

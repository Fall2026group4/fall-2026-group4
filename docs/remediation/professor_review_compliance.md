# Professor review compliance: GPT-2 scope (as of 2026-11-13)

**Sources:**
* the original review tickets on the project board "SAE-Faithful Capstone Project": Instructor Reviews #1–#8
  (09-20 to 11-08), read in full on 2026-11-13;
* issues #1–#25.

Review #8's follow-up table carries every still-open item from Reviews #1–#7 and is quoted below. Review #7's open
checklist was checked against it: nothing is missing.

**Rule (Review #8):** a fix counts once it is merged to `main`. Nothing since PR #23 is on `main`, so every item that
lives only on `sae-research` / PR #27 is at best **IMPLEMENTED ON BRANCH, NOT MERGED**.

| # | Requirement (review; original wording abbreviated) | Required action | Evidence of implementation | Validation | Remaining work | Status | Commit / PR |
|---|---|---|---|---|---|---|---|
| 1 | "Bring the improvement result: one table, which method, at which layers, improves which faithfulness measure, with seeds and CIs" (#8 blocker; 11-03 "fine-tune other layers") | Multi-layer improvement table | IMP-L (L3/8/9, VAL); R2/R4 at L1/8/11; M5 TEST table (136 comparison rows) | `m5_independent_verification.py`: ALL CHECKS PASS (also in CI) | Merge; present it as "reconstruction and behaviour improved (TEST); truth not; honesty VAL only" | IMPLEMENTED ON BRANCH, NOT MERGED | #27 |
| 2 | Reconstruction vs behavioural fidelity (proposal; #6–#8) | Separate levels | Four-level framework (M2), splicing ΔCE/KL, M5 | M5 | — | IMPLEMENTED ON BRANCH, NOT MERGED | #27 |
| 3 | "Adding decoder directions to raw activations is not an SAE intervention" (#8 major) | Real SAE-in-the-loop test | M4, E3, R2-clamp. 2026-11-13: equivalence argument in the paper (D33): error-kept patches establish SAE *selection*, not *mediation* | — | A mediated design (edits propagated through downstream SAEs) is proposed, not run | PARTIALLY RESOLVED | `5b524f6` |
| 4 | Sparsity matching (#7 report) | Matched L0 | OWT ±10 % enforced and now unit-tested; task L0 disclosed; PCA-CM `task_l0_matched` reference | tests `test_l0_band…`, `test_m5_compare_excludes_unmatched_l0` | Task-level matched training (E1) not run | PARTIALLY RESOLVED | `72e466b`, PCA-CM |
| 5 | "Say how large an effect your test could detect" (#8 critique 3) | MDE / power statement | `scripts/shared/power_sensitivity_analysis.py` → `results/shared/power_sensitivity/` (from existing outputs) | see that README | Paper sentence | see the S2 row in `issue_register.md` | this branch |
| 6 | "Run the final preregistered test (X3) once and report only that as TEST" (#8 major) | One clean test pass | M5 for OWT/GoT (CONFIRMED TEST, previously exposed rows, D31). RepE-LM test split built and protocol frozen 2026-11-13; **not evaluated** | hashes verified | Approve and run H1 once | PARTIALLY RESOLVED | `5b524f6` |
| 7 | Literature review incl. AxBench, SAEBench, Paulo et al. into `mybib.bib` (#5–#8 NOVELTY-RISK, H3b) | In the report bib and on `main` | Paper bib (22 entries) and report bib (now 22) contain them; 2 entries corrected | both documents compile with 0 undefined citations | Merge | IMPLEMENTED ON BRANCH, NOT MERGED | `bf5865a`, `ab0b4c8` |
| 8 | Paper in `research_paper/Latex/`; journal chosen, template (#8 blocker; due 11-10 / 11-17) | Paper | 37-page paper compiles | out-of-tree build 2026-11-13 | **Journal and template not chosen** (team decision); merge | PARTIALLY RESOLVED | #27 |
| 9 | Report `reports/Latex_report/main.tex` (#6–#8; due 11-22) | Full report | 5 TODO sections written (GPT-2) | compiles, 5 pp | Gemma subsection (teammates); title/abstract (team) | PARTIALLY RESOLVED | `8217cbc` |
| 10 | Revised presentation and 10-20 deck in `presentation/` (#6–#8 NO-SLIDES, PRES-QUALITY; due 11-17) | Decks | GPT-2 slide content `presentation/gpt2_results_slides.md` | — | Team deck; the 10-20 slides (only the presenter has them) | PARTIALLY RESOLVED (my part drafted) | this branch |
| 11 | Progress report entries since 09-29 (#6–#8 H1-STALE) | Weeks 5–15 | `reports/Progress_Report/2026-11-13_gpt2_retrospective.md`, **labelled retrospective** | each row cites the journal section | Teammates' weeks; ongoing weekly entries | PARTIALLY RESOLVED | this branch |
| 12 | SVG/PDF figures; `.drawio` source + `.svg` (#5–#8 F7, F8) | Vector figures and editable diagram | 21 figures → SVG + PDF (pixel-identical check); `gpt2_sae_faithfulness_pipeline.{drawio,svg}` | re-render check | 31 PNGs without a re-runnable script; merge | PARTIALLY RESOLVED | `9d59bf2` |
| 13 | "Link `as650@gwu.edu` to Aditi's GitHub account" (#7, #8 IDENTITY) | Credit commits | 27/27 PR commits still unlinked (API `author.login` null) | — | User adds the email in GitHub settings (`issue_register.md` §F) | REQUIRES USER ACTION | — |
| 14 | "A CI workflow on `main`"; logging instead of `print` (#5–#8 G12) | CI; logging | `.github/workflows/gpt2-validation.yml`; logging in new shared code | clean-tree simulation green; GitHub run in `issue_register.md` §G | On `main` only after merge; hashed scripts keep `print` by design | IMPLEMENTED ON BRANCH, NOT MERGED | `72e466b` |
| 15 | "Before the next merge, the reviewer re-runs one result script and says what matched" (#8 peer review) | Substantive review | #27 approved by basquedhruv ("Works."); MakaveliSimz has asked questions; the PR description names `m5_independent_verification.py` as the script to re-run | — | Reviewer re-run and a written result | REQUIRES TEAM ACTION | #27 |
| 16 | Merge plan: close/rebase #16, #17; re-scope #20, #21, #24; review #22 (#7, #8 MERGE-PLAN) | Plan | `docs/remediation/merge_plan.md` (verified file overlaps, runtime dependency, order) | GitHub file lists | Team agreement; actions on teammates' PRs | REQUIRES TEAM ACTION | this branch |
| 17 | Reproducibility (#5–#8; pinned environment, results versioning) | Reproducible results | Pinned `requirements.txt` (+scipy, joblib); pre-registration hashes; checkpoint registry; M5 verifier in CI | CI | Off-machine checkpoint backup (U1) | PARTIALLY RESOLVED | `72e466b` |
| 18 | Tickets with owners for the 11-03 list; close #4, #7, #11 (#7, #8 project management) | Board hygiene | — | — | Owners act | REQUIRES TEAM ACTION | — |
| 19 | "Small, unconverged SAEs: compare with larger or pretrained SAEs" (#8 critique 2) | External/larger SAEs | E2 (res-jb), R2 (100M tokens, width 12,288) | M5 | Merge | IMPLEMENTED ON BRANCH, NOT MERGED | #27 |
| 20 | "State what effect E measures on the model's output" (#8 CAUSAL-PROBE-OUTCOME) | Output-level statement | M3/M4 define T via next-token KL (paper §causal) | — | Merge | IMPLEMENTED ON BRANCH, NOT MERGED | #27 |
| 21 | Poster; recorded video; 1-minute video; journal submission confirmation (#8 deliverables, 12-08) | Deliverables | — | — | Team production | OPEN (outside this branch's scope) | — |
| 22 | AUROC 1.000 explanation; Gemma layer table (#7) | Gemma | — | — | Teammates' Gemma work | NOT APPLICABLE (GPT-2 scope) | — |
| 23 | Ruff findings (#5) | Lint | `ruff` not installed here; CI does py_compile | — | — | OPEN (not verifiable) | — |

## Findings in the instructor's Review #8 summary that are now superseded

Review #8 summarised the 11-08 state of `main`. Later work on `sae-research` changed four of those statements; the
corrections are drafted for the instructor in `review8_response_draft.md`:

| Review #8 statement | Current evidence | Source |
|---|---|---|
| "Truth information is displaced, not erased" | Most of the frozen-probe drop is reproduced by norm-matched noise (generic fragility); a re-fit reader loses a real but small 0.03–0.11 AUROC | M2 (VAL), M5 (TEST) |
| "Reconstruction quality … predicts this on RepE (ρ = 0.86)" | A pooling artefact across layers: the within-layer association falls from 0.89 to 0.05. Within a layer, reconstruction error predicts splice damage (per-layer ρ 0.80–0.97 from L3), not truth-readout loss | M0 (VAL) |
| "SAE errors raise loss 1.0–2.6× more than random errors" | 0.76–2.89 across all layers for the old SAEs (M0, VAL). On TEST at L8/L11: old 1.44–2.00, R2 1.13–2.24, R4 KL at L8 0.98 | M0, M5 |
| "Raw honesty direction has a small effect (p = 0.010)" | From the length-confounded original RepE task, now HISTORICAL; replaced by RepE-LM, M3/M4 (VAL) | D23 |

# Merge-resolution plan for the open PRs (2026-11-13). Status: PROPOSAL; REQUIRES TEAM ACTION

Remediation items D3, D4, U2, U4 and Review #8 MERGE-PLAN. Everything here was read from GitHub on 2026-11-13 and
re-checked through the GitHub API at about 08:40 UTC the same day (file lists of #20, #21, #24, #26 and #28 against
the files tracked on `sae-research`). No teammate branch was modified and nothing was merged, closed or commented.

## 1. State of the open PRs

| PR | Author | Head | Mergeable | Review | What it does |
|---|---|---|---|---|---|
| #27 | aditi650 | `sae-research` | yes (`main` has only 1 notebook deletion since the merge base) | APPROVED by basquedhruv ("Works."); MakaveliSimz requested, has asked 2 questions (§4) | GPT-2 work: R2/R4/R5/M5, docs, remediation. **Deletes 0 files; touches 0 Gemma files** |
| #28 | MakaveliSimz | `simba-gemma-m6-results` | yes | required; draft | Gemma M6 (Gemma Scope splicing on the Pile) |
| #26 | basquedhruv | `dhruv-causal` | yes | required | Causal and rebuild follow-ups (97 files added, 7 modified) |
| #24 | basquedhruv | `cleanup-main` | yes | required | Deletes 30 files and renames 2 |
| #21 | MakaveliSimz | `simba-cleanup-scripts-results` | yes | required | Deletes 29 files |
| #22 | MakaveliSimz | `simba-gemma-pile-eda` | (not re-checked) | required | Gemma/Pile EDA |
| #20 | aditi650 | `aditi650-patch-6` | yes | required | Deletes `results/figures/` (7 files: EDA PNGs, feature-974 figures) |
| #17 | aditi650 | `aditi` | **CONFLICTING** (last update 10-28) | required | old branch |
| #16 | basquedhruv | `dhruv` | **CONFLICTING** (last update 10-26) | required | Review #6 fixes, old CI |

## 2. Verified conflicts and dependencies

1. **Runtime dependency (D3).** #21 and #24 both delete `results/geometry_of_truth/sae_faithfulness_results.csv`.
   * `scripts/geometry_of_truth/10_layer10_frozen_probe.py` reads that file at runtime. It is the Review #7 GoT
     layer-10 correction.
   * They also delete five legacy scripts that script 10 imports or cites: `scripts/{cache_geometry_activations,
     prepare_geometry_data,run_geometry_faithfulness,run_geometry_probes,run_geometry_sae}.py`.
   * Further runtime readers of files they delete (re-checked 08:40 UTC): `scripts/run_geometry_detection.py` reads
     `sae_feature_test_results.csv` and `sae_top_features_train.csv`; `run_geometry_sae.py` reads
     `sae_encoding_summary.csv`; `run_geometry_probes.py` and `run_geometry_faithfulness.py` read
     `baseline_probe_results.csv`. These are legacy scripts deleted by the same PRs, so only script 10 is left broken.
   * #24 also deletes `cookbooks/02_pretained_sae_baseline.ipynb`. `src/evaluation/detection.py` cites it in its
     docstring only (provenance, not a runtime dependency).
   * Overall, #21 deletes 29 files tracked on `sae-research` and #24 deletes 30; 26 of these are common to both.
2. **Shared ledgers (D4).** #26 and #28 each modify the same 7 documentation ledgers as #27:
   `docs/{EXPERIMENT_CHANGELOG,EXPERIMENT_RESULTS_SUMMARY,RESEARCH_DECISIONS,RESEARCH_PROGRESS,RESEARCH_ROADMAP,
   results_registry}.md` and `docs/experiment_manifest.csv`. All three append to the same files, so a textual conflict
   is certain after the first of them merges. The conflicts are append-only and resolvable by keeping both sides.
3. **Duplicate script numbers.**
   * #26 adds `scripts/geometry_of_truth/22_truth_direction_causal_test.py`, `scripts/repe/14_causal_reanalysis.py`
     and `scripts/repe/15_sae_in_the_loop_causal.py`. Numbers 22, 14 and 15 already exist here (IMP-L, R1 split prep,
     R1 fine-tune). RepE numbers 28, 29 and 30 are now also taken on this branch (H1 split, H1 test evaluation, SIL).
   * #28 adds `scripts/geometry_of_truth/{22_method_a_multilayer_finetune,23_method_a_multilayer_evaluation}.py`.
     These are the same experiment family as this branch's IMP-L scripts 22/23, under different names (U4).
   * This branch has its own duplicate: `scripts/repe/11_compare_sae_steering.py` and `11_heldout_steering.py` (D12).
     Renaming is not proposed, because the paths are hashed in pre-registrations.
4. **`requirements.txt`.** #28 appends `zstandard` at the end of the file; #27 inserts `joblib` and `scipy` in the
   alphabetical block. The hunks do not overlap, so git should merge them without a conflict (not tested by a merge).
5. **SAE-in-the-loop overlap (read-only check of #26).** #26's `scripts/repe/15_sae_in_the_loop_causal.py` Part B
   `latent_all` patches x_U + D(z_H − z_U) with the SAE error kept. By the identity in the paper (D33), that equals
   adding one vector to the raw activation. It is therefore the same class as M4's T_rec, and it does not by itself
   answer Review #8's "adding decoder directions to raw activations is not an SAE intervention". This branch's SIL
   (`results/repe/sae_in_the_loop_mediation/`) drops the error and splices SAEs downstream, so it is complementary,
   not a duplicate. Whether #26's numbers agree with M4 is NOT VERIFIED.
6. **Possible contradictory results (U2, NOT VERIFIED).** #26 runs its own versions of public-SAE splicing,
   sparsity-matched truth, SAE-in-the-loop causal and GoT causal tests. Whether its numbers agree with #27's has not
   been checked. #28's Method-A multilayer results are a Gemma/GPT-2 parallel of IMP-L; whether the conclusions agree
   is likewise NOT VERIFIED.
7. **#20** deletes `results/figures/`. Those files are referenced from `cookbooks/README.md`, `results/phase1_observations.md`
   and `docs/2026-11-09_review_package_sae_research.md`. They are historical and superseded (feature 974, EDA), so
   deletion is low risk, but the references would break. This is your own PR, so the decision is yours.

## 3. Proposed integration order

MakaveliSimz proposed this order on #27 (2026-11-13), and the evidence above supports it:

1. **#27 first.** It deletes nothing and touches no Gemma file, and `main` has not moved.
2. **#26**, rebased on the new `main`.
   * Resolve the ledger conflicts by keeping both sides.
   * Renumber its three scripts to the next free numbers: GoT 24; RepE 31 and 32, since 28, 29 and 30 are now taken
     on this branch (H1, SIL).
   * Compare its overlapping results with #27's and record agreement or disagreement in `RESEARCH_DECISIONS.md`.
3. **#28**, rebased on the result.
   * Resolve the ledgers.
   * Rename the GoT 22/23 duplicates (for example to 25/26), or move them under `scripts/gemma/` if they are Gemma
     runs.
4. **#21 and #24 last, re-scoped.**
   * Keep `results/geometry_of_truth/sae_faithfulness_results.csv` and the five legacy scripts that script 10 uses,
     or first change script 10 under a new pre-registration.
   * Merge only one of the two, since they overlap in 26 of their deletions.
5. **#16 and #17:** close, as superseded and conflicting since October. **#20:** decide (§2.7). **#22:** review.

Each merge needs one approving review (the ruleset on `main`). Review #8 asks the reviewer to **re-run one result
script and say in the review what matched**. For #27, the suggestion is `python
scripts/shared/m5_independent_verification.py`: it reads only committed files and prints `RESULT: ALL CHECKS PASS`.

## 4. Questions raised on #27 that only the author can answer

MakaveliSimz (2026-11-12, 2026-11-13) asked:

1. To confirm the merge order relative to #21 and #24.
2. To confirm that #27 will not delete or overwrite Gemma files.

The factual answer to (2) is verified above: #27 deletes no file and changes no Gemma path. (1) is a team decision.
A draft reply is in [`review8_response_draft.md`](review8_response_draft.md) §B. **It has not been posted.**

## 5. Stale issues (D15)

* #4 and #7 are MakaveliSimz's, from September; #11 is aditi650's, from October.
* Closing or updating them, and creating tickets with owners for the 11-03 list, is project-management work the
  instructor scores.
* The owners must do this; nothing was changed.

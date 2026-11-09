# Review package: `sae-research` commits after PR #23 (for independent teammate review before merging to `main`)

Prepared 2026-11-09. **Status: NOT PUSHED.** The branch will be pushed and a PR opened only with the project lead's
approval. Nothing here may be merged without:
1. an independent teammate review, and
2. the reproducibility checks in §3.

## 1. What is being merged

* **Base:** `origin/main` = `33dcb9c` (PR #23 merge).
* **Commits:** 18 on `sae-research`: the 17 listed below plus the commit that adds this file.
* A non-destructive `git merge-tree origin/main HEAD` reports **no conflicts**.

| Commit | Content | Evidence |
|---|---|---|
| `d17bb00` `211218f` `d283a49` | R1: RepE honesty-preservation fine-tuning, layer 6 | HISTORICAL (length-confounded task; D23) |
| `b232ce4` `6d39d40` | R2: SAE-in-the-loop honesty-feature clamp, layer 6 | HISTORICAL (D23) |
| `ebdc1be` `ea0e7fa` | M0: within-layer reanalysis of the 108 SAEs | VALIDATION |
| `0818f73` | M1: RepE-LM equal-length persona task, gates K1–K7 | TRAIN gates; validation descriptive |
| `b384622` `fc9a243` | M2: four-level information faithfulness | VALIDATION |
| `6a23d47` | M3: causal site sweep in the original GPT-2 | VALIDATION (sites selected on train, replicated on validation) |
| `15131fe` `3bbc62d` `a424ca6` | M4: SAE causal decomposition and feature patches | VALIDATION |
| `fb00cee` | README rewrite after Instructor Review #8 | — |
| `864c4e0` | Paper skeleton; IMP-L pre-registration | — |
| `7c9391d` | IMP-L results (layers 3, 8, 9); paper §10.1 | VALIDATION |
| next commit | Paper refinements (evidence status, discussion, limitations); E3/E2/E1 draft pre-registrations; this package | — |

**No result in this branch is a confirmed held-out result.** Only the OWT reconstruction table (already on `main`) is
TEST; the held-out evaluation M5 has not been run.

## 2. Safety checks already done (2026-11-09, on `33dcb9c..HEAD`)

* **Forbidden files:** no `.pt`, `.safetensors`, activation cache, log, `.bak` or credential file in the diff.
* **Secrets and paths:** no secret patterns and no absolute home paths in added lines.
* **Size:** 175 files, +92.7k lines. The largest files are `results/repe/sae_patch_decomposition/per_pair_validation.csv`
  (6.3 MB) and `results/shared/m2_four_level_faithfulness/bootstrap_arrays.npz` (7.0 MB).
* **Tests:** `python -m pytest -q src/tests` → 35 passed. Checked on the working tree and on a clean `git archive`
  export of `HEAD`.
* **Commit identity:** all commits are authored as `Aditi Shukla <as650@gwu.edu>`. This address is not yet linked
  to a GitHub account (Review #8 minor item).

## 3. Reproducibility checks for the reviewer

Do these in a **fresh clone** of the pushed branch, so that your working copy is not affected. Use the project venv.
Report in the review what you ran and what matched.

1. **Tests:** `python -m pytest -q src/tests` → expect `35 passed`.
2. **Paper artifacts:** regenerate them and expect an empty diff. Since `7c9391d`, figures are byte-deterministic.

   ```bash
   python scripts/shared/build_paper_artifacts.py
   git status --short research_paper/
   ```

3. **M0 re-run** (CPU, about 25 s; uses only committed inputs). Scripts refuse to overwrite, so move the outputs
   aside first, re-run, and compare:

   ```bash
   mkdir /tmp/m0_orig
   cd results/shared/m0_within_layer_reanalysis
   mv merged_validation_table.csv within_layer_regressions.csv within_layer_spearman.csv \
      architecture_by_layer.csv architecture_pairwise_l0_adjusted.csv l0_elasticity_from_sweep.csv \
      pooled_vs_within_corrections.csv splice_ratio_range.json summary.json /tmp/m0_orig/
   cd -
   python scripts/shared/m0_within_layer_reanalysis.py
   for f in /tmp/m0_orig/*; do cmp "$f" results/shared/m0_within_layer_reanalysis/$(basename "$f"); done
   ```

   Expect no output from `cmp`: all 9 files are byte-identical. This was verified on a clean export on 2026-11-09.

4. **Spot-check three numbers** in `README.md` §6 against the CSVs they cite. For example:
   * §6.8 IMP-L "4/9" against `results/geometry_of_truth/multilayer_finetune/criteria.json`;
   * the M4 L8 preservation 0.62–0.67 against `results/repe/sae_patch_decomposition/architecture_summary.csv`;
   * the M2 GoT frozen-gap figures against `results/shared/m2_four_level_faithfulness/summary_by_layer_arch.csv`.
5. **Paper:** `cd research_paper/Latex && latexmk -pdf main.tex` compiles with no undefined references.
6. **Optional (GPU, about 5 min):** `python scripts/geometry_of_truth/23_multilayer_evaluation.py --smoke --smoke-out /tmp/impl_smoke --layers 8`.
   It evaluates the 9 base layer-8 SAEs. Their re-fit gaps should equal M2's committed layer-8 values (GoT
   0.034/0.061/0.085, persona 0.212/0.156/0.162 for ReLU/TopK/JumpReLU).

Checkpoints are not in Git. GPU re-runs of training need the local checkpoint directory. Its SHA-256 hashes are in
`docs/checkpoint_registry.csv`.

## 4. Points that need a team decision before merging

* **Open PRs that delete files this branch cites:**
  * #20 deletes `results/figures/`;
  * #21 removes legacy scripts and results;
  * #24 empties `scripts/`, `results/` and cookbooks on `main`.

  If any of them merges first, this branch's results and scripts lose their provenance. They need to be re-scoped,
  or merged after this branch with an explicit file list.
* **Overlapping teammate branch:** `simba-method-a-multilayer` (not merged, no PR) adds
  `scripts/geometry_of_truth/22_method_a_multilayer_finetune.py` and `23_method_a_multilayer_evaluation.py`.
  * That is the same experiment idea as IMP-L, and **the same script numbers**: IMP-L uses
    `22_multilayer_finetune.py` and `23_multilayer_evaluation.py`.
  * It also replaces layer 3 with layer 5 as "non-informative". Under IMP-L's pre-registered rule, layer 3 is
    informative by the MLP probe (AUROC 0.815) but weak by the logistic probe (0.60).
  * The team should agree on one experiment and renumber the other script before either merges.
* **#16 and #17** conflict with PR #23 (see PR #23 §4).

## 5. Not in this branch

* M5 (the held-out test; needs approval and a frozen protocol).
* E1, E2 and E3: draft pre-registrations only, not run.
* Gemma SAEs.
* The `.drawio` pipeline diagram, CI on `main`, the presentation decks and the progress reports.

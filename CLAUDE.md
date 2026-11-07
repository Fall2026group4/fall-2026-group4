# CLAUDE.md: permanent instructions for the SAE Faithfulness capstone

These rules apply to every task in this repository and every Claude Code session. Read this file before acting.

## 0. Project in one paragraph

**Research question.** Can we improve SAE reconstruction of frozen LLM activations while preserving truth-related information and causal behaviour across transformer layers?

**Scope:**
* Models: GPT-2 Small (12 layers) and Gemma-2-2B (26 layers; steering only so far, no SAEs).
* SAEs: ReLU, TopK and JumpReLU (sigmoid-surrogate), trained in house on OpenWebText (OWT).
* Datasets: OWT (reconstruction, behaviour), Geometry of Truth (GoT; truth) and RepE honesty (persona prompts).

**Where to start:**
* `docs/RESEARCH_PROGRESS.md`: the full history.
* `docs/RESEARCH_ROADMAP.md`: what is next.
* `docs/results_registry.md`: which results may be cited.

## 1. Documentation is part of the definition of done (mandatory, automatic)

Do this **without being asked**, for every experiment, training run, evaluation, bug fix, methodological correction, dataset or split change, hyperparameter change and research decision.

**Before implementation**, write a plan entry. Use `docs/templates/EXPERIMENT_TEMPLATE.md`, put it in the experiment's results folder as `README.md` or `preregistration.json`, and summarise it in `docs/RESEARCH_PROGRESS.md` under "In progress". The entry records:
1. The research question, the hypothesis, and why the experiment is necessary. Say how it connects to the main research question.
2. The design:
   * datasets and splits, model, layers, SAE architectures;
   * baselines and controls;
   * metrics, each with its direction (higher or lower is better);
   * hyperparameters, seeds and training budget.
3. Every intended comparison (X vs Y) and what each one is meant to establish.
4. Before any training:
   * the split verification;
   * leakage-prevention checks;
   * pre-registered success criteria;
   * the evaluation protocol;
   * file hashes of the scripts, split files and pre-registration.

**After execution, in the same task:**

5. Record the actual results:
   * seed variability (mean ± sd and per-seed values where criteria are paired);
   * improvements and regressions;
   * limitations;
   * whether each pre-registered criterion was met. Report failures as plainly as successes.
6. Update all of these:
   * `docs/RESEARCH_PROGRESS.md` (journal entry);
   * `docs/EXPERIMENT_CHANGELOG.md` (files added, modified, deleted or superseded, each with its reason);
   * `docs/EXPERIMENT_RESULTS_SUMMARY.md` (quantitative tables);
   * `docs/RESEARCH_DECISIONS.md` (any decision taken);
   * `docs/RESEARCH_ROADMAP.md` (status changes);
   * `docs/results_registry.md` (status of any new or superseded result);
   * the results-folder `README.md`;
   * `docs/experiment_manifest.csv`, through `scripts/shared/write_manifest.py:append_experiment`;
   * `docs/checkpoint_registry.csv`, through `scripts/shared/build_checkpoint_registry.py`, if checkpoints were created.
7. Put the documentation changes in **the same local Git commit** as the experiment they describe.
8. If a task stops before completion, record in `docs/RESEARCH_PROGRESS.md` and `docs/RESEARCH_ROADMAP.md`:
   * its status;
   * what is unfinished;
   * which artifacts exist;
   * the exact next steps.

An experiment is **not complete** until its results and changes are documented and committed locally.

## 2. Scientific rules

* **Keep these categories separate.** Never treat one as evidence for another:
  * **reconstruction quality** (MSE, NMSE/EV, cosine, L0);
  * **representation preservation** (frozen-probe AUROC gaps, direction fidelity);
  * **downstream behavioural faithfulness** (activation splicing: output KL, CE increase, task-preference change);
  * **causal interventions** (pre-registered tests with controls);
  * **steering** (adding directions to activations).
* **Label every result's evidential status:** exploratory, validation finding, confirmed test result, or inconclusive.
* **Probes:**
  * Evaluation probes are fitted on ORIGINAL activations only and applied frozen.
  * An evaluation probe must never be the training teacher, or share its fitted parameters or training rows. See `docs/2026-11-07_truth_aware_finetuning_audit.md` for the circularity found once.
  * GoT train groups A/B: `data/got/splits/train_AB_groups_truth_aware_pilot.csv`. A is for training, B is for independent probes.
* **Test splits:**
  * Never used for training, tuning, checkpoint selection or deciding between methods.
  * Touched only by a pre-registered final evaluation.
* **Sparsity matching:**
  * Compare architectures only at matched L0: within ±10 % of target for baselines, ±5 % of base for fine-tunes.
  * Report L0 separately on OWT, GoT and RepE.
  * Flag and exclude unmatched runs from rankings; don't delete them.
* **Statistics:**
  * Report seed variability.
  * Use paired comparisons when runs share seeds.
  * Don't average raw MSE across layers.
  * Don't present an architecture as better under unequal conditions.
* MSE improvement (%) = 100 × (baseline − improved) / baseline.
* **Don't invent numbers.**
  * Verify artifacts exist before claiming an experiment ran.
  * Label anything unverifiable as **unverified** or **unavailable**.

## 3. Preservation and Git rules

* Work on the `sae-research` branch. Don't create or switch branches; don't pull, rebase, merge or reset; don't use destructive Git commands.
* **Never push to GitHub without explicit approval from the user.** Local commits made as part of a documented experiment are expected.
* Never silently overwrite, delete or rewrite historical results, checkpoints, scripts or configs.
  * Scripts must refuse to overwrite existing outputs.
  * New variants go into new folders.
  * Superseded results are marked, not removed.
  * Deletion needs the user's explicit authorisation, for example of `results/professor_review_corrections/` or `scripts/professor_review/`.
* **Never commit:**
  * activation caches (`data/activations/`, `data/*/activations/`);
  * `.pt` checkpoints;
  * logs;
  * `.bak` files;
  * credentials.

  Scan staged diffs for secrets and absolute home paths before committing.
* Commit messages end with the attribution line requested by the harness.

## 4. Conventions (see `docs/README.md`)

* **Scripts:** `scripts/{openwebtext,geometry_of_truth,repe,gemma}/NN_name.py`, numbered in run order. Cross-dataset code goes in `scripts/shared/`.
* **Outputs:** in the matching `results/<dataset>/` folder. Cross-dataset summaries go in `results/shared/`.
* **Shared code:**
  * metrics: `src/evaluation/faithfulness_metrics.py`;
  * checkpoint lookup: `src/sae/checkpoint_registry.py`;
  * layer-6 evaluation: `scripts/shared/layer6_faithfulness_eval.py`;
  * splicing: `scripts/shared/spliced_reconstruction_evaluation.py`.
* **Python:** `/home/ubuntu/sae-project/venv/bin/python`. Run scripts from the repository root.
* **Tests:** `python -m pytest -q src/tests`. Run them before every commit.
* **Hooks:** SAEs live at `blocks.{L}.hook_resid_pre`. BOS is excluded from SAE training and from splicing.

# Corrected truth-aware fine-tuning pilot (2026-11-07). **PILOT, validation only. Pre-registered success criteria NOT met.**

Protocol: `docs/2026-11-07_truth_aware_finetuning_audit.md` and `preregistration.json`, written before any fine-tune.
* Script and split hashes were recorded before training in `pretraining_hashes.json`.
* Scripts:
  * `scripts/geometry_of_truth/11_truth_aware_pilot.py`: split, pre-registration, training.
  * `12_truth_aware_pilot_evaluation.py`: evaluation.
  * `13_truth_aware_pilot_figure.py`: figure.

**Design:**
* Layer 6, TopK (L0 fixed at 128), seeds 42/1042/2042.
* Three variants: reconstruction-only, truth-aware (teacher fitted on train groups A), random teacher.
* Each fine-tune: 500 steps, lr 1e-5, λ 0.1, with 50 % OWT replay.
* Independent evaluation probes are fitted on the original activations of the disjoint train groups B.
* Split file: `data/got/splits/train_AB_groups_truth_aware_pilot.csv`; checks in `split_checks.json`.
* Checkpoints: `results/geometry_of_truth/sae_finetuning/truth_aware_pilot_layer6_topk/`. Base checkpoints were verified unchanged.

**Files:**
* `per_sae.csv`: 12 SAEs × all metrics.
* `comparisons.csv`: paired by seed.
* `summary.csv`.
* `criteria.json`.
* `training_log.csv`, `training_info.json`, `eval_info.json`.
* Figure: `../figures/truth_aware_pilot.png`.

**Reading rules:**
* `*_TEACHER_gap_objective_only` measures objective attainment, not truth preservation.
* The B-logistic probe is only partly independent of the teacher: weight cosine 0.64 (geographic) and 0.89 (numerical).
* The MLP probe is the most independent test.
* `*_DIAG_refit_*` measures decodability, not faithfulness.
* `splice_*` metrics are activation splicing, not steering.

# Cross-dataset results (GPT-2 Small)

Status legend as in `docs/results_registry.md`.

## `spliced_evaluation/`: Phase 0, spliced-in causal reconstruction (2026-11-07). **AUTHORITATIVE (validation only)**

Produced by `scripts/shared/spliced_reconstruction_evaluation.py` and `scripts/shared/spliced_reconstruction_analysis.py`.

**Method.** `blocks.L.hook_resid_pre` is replaced by the SAE reconstruction, then GPT-2's predictions are compared with the unmodified model's.
* SAEs: all 108 uninterrupted SAEs.
* Data: OWT validation, GoT val and RepE validation. No test row was used.
* Per-dataset outputs are in `results/{openwebtext,geometry_of_truth,repe}/spliced_evaluation/`:
  * `per_run.csv`
  * `per_run_wide.csv`, the SAE row next to its matched random-error control
  * `layer_controls.csv`
  * `summary.csv`
  * `excluded_unmatched_runs.csv`

**Files in this folder:**
* `pipeline_checks.csv`: determinism, and cached vs freshly computed activations.
* `unified_comparison.csv`: one row per SAE across OWT, GoT and RepE, including the validation frozen-probe gaps.
* `matched_architecture_comparison.csv`.
* `reconstruction_vs_causal_correlations.csv`.
* `layer6_bridge.csv`: layer 6, comparing the historical resume and fine-tuned SAEs against the uninterrupted ones on every validation metric.
  * Its `val_probe__*_logistic_*` values for `historical_finetuned_truth_aware` are **circular**: the probe is the fine-tuning teacher (see `docs/2026-11-07_truth_aware_finetuning_audit.md`). All other bridge values are valid.
* `figures/`.

**Controls:**
* Identity replacement and two unmodified passes: exactly 0 difference.
* fp16 round-trip: ΔCE ≤ 7e-6.
* Mean-ablation, the floor for "loss recovered".
* A norm-matched random error for every SAE.

**Not steering.** Activation splicing is separate from the steering experiments in `results/repe/steering/` and `results/gemma/`.

**Caveats:**
* GPT-2's own likelihood preference for true GoT statements is clear only for `cities` (0.82 pairwise accuracy). `neg_cities` is 0.17 (association beats negation), and the numerical sets are near chance. Only `cities` is a meaningful behavioural truth measure.
* GPT-2's own honesty effect on the next statement token is about 0 (−0.013), so it is descriptive only.
* "Dead" means never active on 12,700 validation tokens.

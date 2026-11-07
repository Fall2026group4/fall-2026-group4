# Layer-6 SAE fine-tuning on GoT (`scripts/geometry_of_truth/06_finetune_sae.py`)

**Setup:**
* 500 steps, Adam lr 1e-5, GoT **train** only, seed 42.
* No sparsity term and no OWT replay.
* Starts from the layer-6 resumed SAEs.

**Status (2026-11-07 audit, `docs/2026-11-07_truth_aware_finetuning_audit.md`):**
* **`truth_aware` rows scored with the logistic probe: circular, not evidence of truth preservation.**
  * The truth loss distils the Stage-3 logistic probe's logits.
  * That same probe (identical weights) scores `validation_comparison.csv` and the logistic columns of `heldout_comparison_all_datasets.csv`.
  * These numbers show only that the training objective was reached.
* **Independent MLP probe** (`../independent_truth_probe/`, `../final_test_evaluation/`): truth-aware is worse than reconstruction-only in all 6 architecture × domain cells.
* **L0 is not matched:** reconstruction-only ReLU/JumpReLU roughly double their L0 on GoT prompts.
* **Still valid:**
  * reconstruction-only and base rows;
  * RepE, OWT and causal columns.
* No example leakage was found: train only, grouped splits.

All numbers are kept exactly as originally produced.

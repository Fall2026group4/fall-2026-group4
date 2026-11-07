# Method A at layer 6: GoT reconstruction fine-tuning with OWT replay (2026-11-07). **PILOT, validation only. Pre-registered success NOT met for any architecture.**

**Scripts:**
* `scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py`: split verification, pre-registration, training.
* `15_method_a_evaluation.py`: evaluation.
* `16_method_a_figure.py`: figure.

**Protocol.** Identical to the reconstruction-only arm of `../truth_aware_pilot/`:
* same A/B split, with the hash re-verified;
* 500 steps, lr 1e-5;
* each batch 64 + 64 GoT-A rows plus 128 OWT rows.

Sparsity:
* ReLU and JumpReLU keep their training penalty, and the controller continues from the base coefficient (target 128, EMA on the OWT rows).
* TopK has no penalty.
* `pretraining_hashes.json` was recorded before any run and still matches.

**Matrix:**
* 9 uninterrupted bases plus the P1 JumpReLU base.
* Method A: ReLU and JumpReLU (new), TopK (the pilot checkpoints, reused), and the P1 base (single seed, descriptive).
* 9 OWT-only continued-training controls.
* A TopK reproduction check, which was bit-identical to the pilot and is excluded from results.

That is 17 training runs and 29 SAEs evaluated.

**Files:**
* `per_sae.csv`, `summary.csv`, `comparisons.csv`, `criteria.json`.
* `split_verification.json`, `training_info.json`, `training_trace.csv`, `eval_info.json`.
* Figure: `../figures/method_a_layer6.png`.

**Notes:**
* L0 is reported separately on OWT (`owt_val_l0`), GoT (`got_*_val_l0`) and RepE (`repe_val_l0`).
* Matching is defined on OWT. On GoT and RepE prompts the architectures differ strongly in L0.
* `splice_*` metrics are activation splicing, not steering.
* `DIAG_*` measures decodability only.

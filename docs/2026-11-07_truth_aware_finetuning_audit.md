# Audit: layer-6 truth-aware SAE fine-tuning (2026-11-07)

Scope: `scripts/geometry_of_truth/06_finetune_sae.py`, its checkpoints in `results/geometry_of_truth/sae_finetuning/`, and every script that evaluated them. Those are GoT 07/08/09, `scripts/shared/finetuning_heldout_eval.py`, RepE 04/05/06/13, and this week's layer-6 bridge.

No training or new experiment was run for this audit.

## 1. What was fine-tuned

| Item | Value |
|---|---|
| Base SAEs | `results/openwebtext/convergence/layer_6/{relu,topk,jumprelu}_l0_128_resume_5000_to_15000/checkpoint_step_15000.pt` (seed 42; within ≈1 seed sd of the uninterrupted SAEs, see `results/shared/spliced_evaluation/layer6_bridge.csv`) |
| Parameters updated | all: encoder W/b, decoder W, output bias, JumpReLU log-thresholds |
| Optimiser | Adam lr 1e-5, 500 steps, grad-clip 10, decoder renormalised each step, seed 42 |
| Batches | 128 GoT geographic + 128 GoT numerical rows per step, from the GoT **train** split only; no OWT replay |
| Sparsity term | **none** (ReLU/JumpReLU L0 drifts; TopK fixed at 128) |
| `reconstruction_only` loss | MSE(x̂, x) |
| `truth_aware` loss | MSE(x̂, x) + 0.1 · mean over domains of MSE((w·x̂ + b)/s, (w·x + b)/s) |
| Teacher (w, b) | the Stage-3 logistic probe: StandardScaler + LogisticRegression, C chosen on original GoT **validation** AUROC, fitted on original layer-6 GoT **train** activations and labels; s = sd of the teacher logits on train |
| Hyper-parameters | λ = 0.1, lr, steps fixed in advance (not tuned on validation or test) |

The truth term never sees labels directly. It distils the teacher probe's logit, so it preserves the projection of x onto one fixed direction w per domain.

## 2. Shared parameters between training and evaluation

| Evaluation | Probe | Shares with the training objective | Verdict |
|---|---|---|---|
| `06` `validation_comparison.csv` (GoT val) | the teacher itself | identical w, b | **circular** |
| `finetuning_heldout_eval.py` GoT test "frozen logistic" | refit with same rows, labels, C, solver and seed, so identical to the teacher | identical w, b | **circular** |
| All-layer frozen-probe evaluation and the 2026-11-07 layer-6 bridge (logistic) | `fit_original_probes`, verified identical to the teacher (weight cosine 1.0, max \|Δ\| 0.0) | identical w, b | **circular** for `truth_aware` rows only |
| GoT 07/08 (val) and 09 (test) MLP 768-64-1, 5 seeds | fitted on the same original train rows and labels | data and labels shared; parameters and function class independent | **partially independent** |
| GoT mass-mean direction | train rows and labels | data shared; different readout | partially independent (but uninformative on GoT, val AUROC ≤ 0.63) |
| RepE probes, spliced RepE metrics, RepE causal test | different dataset and labels | nothing | **independent** (out-of-domain check) |
| OWT reconstruction and spliced CE/KL | not a probe | nothing | independent |

**Example-level leakage:**
* None found. Fine-tuning reads GoT train only (asserted in the script).
* Splits are grouped (city / number pair) and checked so that no group crosses splits.
* Validation and test rows were never in a training batch.

**Mild optimism, not leakage:** the teacher's C was chosen on GoT validation, so every validation AUROC through that probe (original and reconstructed) is slightly optimistic.

**Test reuse:** the GoT layer-6 test split has been scored twice for these 9 SAEs: the MLP in `09`, then the logistic in the held-out evaluation. It is no longer a clean held-out set for selecting among these variants.

## 3. The evidence

GoT **test**; gap = AUROC(original) − AUROC(reconstruction); lower is better:

| SAE | Variant | Logistic gap (teacher, circular), geo / num | MLP gap (5 seeds), geo / num | GoT-test L0 (geo) |
|---|---|---|---|---|
| ReLU | base | 0.300 / 0.455 | 0.327 / 0.296 | 65 |
| ReLU | reconstruction_only | 0.278 / 0.454 | **0.161 / 0.089** | 144 |
| ReLU | truth_aware | **0.019 / 0.056** | 0.225 / 0.195 | 94 |
| TopK | base | 0.261 / 0.441 | 0.154 / 0.099 | 128 |
| TopK | reconstruction_only | 0.296 / 0.429 | **0.110 / 0.045** | 128 |
| TopK | truth_aware | **0.076 / 0.144** | 0.184 / 0.087 | 128 |
| JumpReLU | base | 0.289 / 0.421 | 0.323 / 0.367 | 45 |
| JumpReLU | reconstruction_only | 0.295 / 0.410 | **0.146 / 0.131** | 100 |
| JumpReLU | truth_aware | **0.063 / 0.161** | 0.218 / 0.218 | 65 |

The validation results (`independent_truth_probe/multiseed_summary.csv`) show the same pattern.

**Interpretation.** The truth-aware term fixes the probe it was trained against. On an independently parameterised probe it does **worse than plain reconstruction fine-tuning in all 6 architecture × domain cells**, so it optimised a familiar probe rather than preserving truth information. In addition:
* **Fine-tuning is confounded with L0.** Reconstruction-only ReLU/JumpReLU roughly double their L0 on GoT prompts. Some of the MLP improvement may simply come from using more latents; TopK, with L0 fixed at 128, still shows it.
* **No transfer to honesty.** RepE gaps are unchanged across all three variants (≈0.17 ReLU, ≈0.09 TopK/JumpReLU).
* **No causal effect.** The causal test finds no effect for any variant.

## 4. Status of historical results

| Result | Status |
|---|---|
| `sae_finetuning/layer_6_*/validation_comparison.csv`, truth-aware rows | **INVALID as evidence of truth preservation** (circular). Valid only as "objective attained". |
| `heldout_comparison_all_datasets.csv`, `got_*_frozen_logistic_*` columns for truth-aware | Qualify: circular. Base and reconstruction-only rows in the same columns are valid. |
| MLP results (07/08/09) | **Valid** as partially independent probes; must be read together with the L0 drift. Test numbers: already-used test. |
| `layer6_bridge.csv`, truth-aware logistic gaps | Qualify: circular (same probe). Everything else in the bridge is valid. |
| RepE preservation, direction fidelity, causal controls for fine-tuned variants | Valid (independent of GoT training). |
| OWT reconstruction of fine-tuned SAEs | Valid. |
| "Fine-tuned SAEs are sparsity-matched" | Never claimed. Remains **not matched**. |

## 5. Reusable infrastructure

* `src/evaluation/faithfulness_metrics.py`. `load_sae(..., architecture=)` already loads config-less fine-tuned checkpoints.
* `scripts/shared/all_layer_frozen_probe_evaluation.py` (probe fitting, frozen scoring, grouped bootstrap). It needs one change: probes must be fitted on a **separate** subset of train groups.
* `scripts/shared/spliced_reconstruction_evaluation.py` (OWT, GoT and RepE splicing, controls) and `scripts/shared/p1_validation_evaluation.py` (baseline-vs-variant, seed-sd rule). These can be reused unchanged.
* `scripts/geometry_of_truth/07/08` MLP probe code can be reused as one evaluation-probe family.
* `06_finetune_sae.py` must not be reused as is: it has no sparsity term, no replay, and the teacher is also the evaluator.

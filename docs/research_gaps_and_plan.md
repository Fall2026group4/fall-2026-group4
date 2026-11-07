# Reconstruction-improvement methods: gap analysis and all-layer plan (2026-11-06)

**Status:** proposal only. Nothing in this document has been run, and no GPU work has started.

Every finding cites the code as it currently stands. Each method is split into three parts: **already implemented** (reusable as is), **needs correction**, and **build from scratch**.

## 0. Facts that constrain every method

| Fact | Evidence | Consequence |
|---|---|---|
| OWT training set: 800 docs = **101 600 tokens**. 15k steps × 256 ≈ 38 epochs | `run_config.json` (uninterrupted protocol) | Wider SAEs and longer training mainly risk overfitting. Every method must report the train-vs-validation gap. |
| Activations are cached for **all 12 layers** (OWT train/val/test; GoT final token; RepE + cross-wording) | `data/activations/gpt2_small/*/manifest.json`; `activations` dicts of 12 entries | Extending to all layers needs no new extraction. |
| Activation scale grows ~1000× from layer 0 to layer 11, and there is no input normalization | `03_training_sae.py:SAE.forward`; audit Task B | Raw MSE cannot be compared across layers. Use NMSE = 1 − EV. λ, the JumpReLU threshold init and the effective LR are all scale-dependent. |
| The original truth readout is uninformative at early GoT layers (geographic val AUROC 0.53–0.65 at layers 0–3) | `results/geometry_of_truth/original_truth_probes/validation_best_C_by_layer.csv` | Truth-preservation gaps are interpreted only where original val AUROC ≥ 0.70 (geographic 4–11, numerical 1–11, RepE 0–11). This criterion is fixed now, before any results. |
| Metrics in code: MSE, cosine, EV and L0 exist. **NMSE and dead-feature fraction are not reported anywhere** | `03:evaluate`, `10:evaluate` | Shared metrics module needed (§7). |
| Training loop: Adam 3e-4 constant, clip 10 hardcoded, decoder renormalized after each step, Kaiming encoder + independent normal decoder init, single final validation evaluation | `03:train_one` | Several Method-D gaps. |

## 1. Method A — Reconstruction fine-tuning

**Already implemented**
* `scripts/geometry_of_truth/06_finetune_sae.py` fine-tunes all three architectures on **GoT train** activations (layer 6): Adam LR 1e-5, 500 steps, domain-balanced batches.
  * It starts from the layer-6 base checkpoint.
  * The `reconstruction_only` variant uses the MSE loss only.
* `scripts/shared/finetuning_heldout_eval.py` evaluates all variants on identical held-out sets.

**Needs correction**
1. **There is no sparsity term or L0 controller during fine-tuning.** L0 drifts (ReLU 65 → 144, JumpReLU 45 → 100 on GoT; TopK is fixed by k). Reconstruction gains are therefore confounded with lower sparsity. The fix is to keep the base SAE's penalty form and run the controller towards the base SAE's L0 on the same data.
2. **No validation-based checkpoint selection.** The script runs a fixed 500 steps and saves the final weights. The fix is to evaluate on validation every N steps and keep the best checkpoint by a pre-declared criterion.
3. **The only fine-tuning data is GoT** (~4.9k statements). OWT EV drops slightly for TopK and JumpReLU (0.857 → 0.850, 0.866 → 0.849), i.e. forgetting. The fix is an OWT replay fraction in each batch.
4. Layer 6 only, one seed, LR and steps never tuned.

**Build from scratch**
* **A1, OWT continued training:** start from the uninterrupted 15k checkpoint and continue at low LR with a decay schedule on OWT train, selecting on OWT validation. This is the plain "fine-tune vs pretrained" comparison on the same distribution. Note that A1 overlaps with Method D (it amounts to longer training plus a schedule), so it will be reported as such.
* **A2, target-domain adaptation:** GoT/RepE final-token train activations plus OWT replay, with L0 anchored, at every layer.
* A per-layer driver writing to `results/openwebtext/finetuning/` (A1) and `results/geometry_of_truth/sae_finetuning/all_layers/` (A2). The existing layer-6 results are kept.

## 2. Method B — Sparsity optimization

**Already implemented**
* `03_training_sae.py` sweep: 12 layers × targets {32, 64, 128} × 3 architectures, 1000 steps, seed 42.
* A multiplicative L0 controller adapts the ReLU L1 coefficient and the JumpReLU sigmoid-count coefficient. TopK uses k = target exactly.
* `scripts/openwebtext/14_sparsity_matching_analysis.py` separates matched from unmatched runs (≤ 10 % L0 error) and lists the fully matched cells.

**Needs correction**
1. **The sweep is not converged** (1000 steps). Its ranking (TopK best) reverses at 15k steps.
2. **The JumpReLU controller misses its targets**: 2/12 layers matched at L0 = 32, and layer 0 collapses to L0 ≈ 0.004. Two likely causes:
   * the threshold init (0.1) ignores the layer's activation scale;
   * the controller is still adapting at the last step, so the final model was never trained at a fixed coefficient.
3. Two matching criteria coexist (10 % in `04`, 20 % in `03 --match-tolerance`). The 10 % criterion is the standard from now on.
4. On ReLU, the "sparsity penalty" is tuned indirectly through the controller. That is the right choice for matched-L0 comparison. The final λ must be recorded, and it already is (`sparsity_coefficient`).

**Build from scratch**
* A converged sweep: 15k steps, targets {16, 32, 64, 128, 256}, all layers.
* A controller phase switch: adapt for the first ~60 % of steps, then freeze the coefficient so the final model trains at a fixed λ. Runs outside ±10 % are reported as unmatched, never compared.
* A JumpReLU threshold init scaled to each layer's pre-activation distribution, behind a flag so the old behaviour stays reproducible.
* Reconstruction-vs-L0 curves per layer and architecture. Comparisons are made either (a) only between matched runs, or (b) by interpolating NMSE at a common L0 grid from runs that bracket it. No extrapolation.

## 3. Method C — SAE width

**Already implemented**
* A `--features` flag in `03`. Every in-house SAE uses 3072 (4×).
* The public `gpt2-small-res-jb` layer-10 SAE (24 576) is evaluated, but its training data and procedure differ, so it is **not comparable** to in-house widths. That stays documented.

**Needs correction**
1. `13_consistent_protocol_retrain.py` hardcodes `features=3072` and its checkpoint names omit the width.
2. Downstream loaders hardcode 3072 (`finetuning_heldout_eval.py`, `repe/13`, GoT `04`).
3. **Fix:** add a `--features` argument and a `_w{W}` directory suffix for non-3072 widths, and read the width from the checkpoint config. Existing names stay valid and mean 3072.

**Build from scratch**
* Width sweep {1536, 3072, 6144, 12288} at L0 = 128 (and at one lower L0 from Method B).
* For every width, report NMSE, L0, **dead-feature fraction**, parameter count (2·768·W + W + 768), wall-clock time per step and peak memory.
* Because the data is small (101.6k tokens against 18.9M parameters at W = 12288), report the train-vs-validation NMSE gap. If wider SAEs overfit, the width result is conditional on the data size, and §8 decision 3 becomes relevant.

## 4. Method D — Improved training

**Already implemented (and preserved)**
* The professor-corrected protocol: one uninterrupted run, one Adam optimizer, one controller, one RNG stream, no warm starts or resets. Any future resume must restore the optimizer, controller and RNG state.
* `--lr`, `--steps`, `--seed`; 3 seeds at layers 2/6/10; gradient clipping (10); decoder unit-norm.

**Needs correction**
1. Constant LR with no warmup or decay.
2. Clip value hardcoded.
3. No in-run validation, so no validation-based checkpoint selection. The uninterrupted script records validation metrics but explicitly does not use them for selection.
4. No dead-feature tracking.
5. Decoder renormalization without removing the gradient component parallel to the decoder columns.
6. Encoder and decoder initialized independently. Common alternatives are encoder = decoderᵀ, and a decoder bias initialized to the data mean.
7. No input normalization.
8. The kernel-STE JumpReLU result (`convergence/layer_6/jumprelu_kernel_ste_l0_128/`) has **no generating script** in the repository, and the published JumpReLU estimator is not implemented. The current JumpReLU is a sigmoid straight-through surrogate and must be labelled that way.

**Build from scratch**
* Backward-compatible training options in `train_one`. **Defaults must reproduce the existing runs bit-for-bit**; this is verified by re-running the 200-step smoke test and comparing against the saved smoke outputs. The options are:
  * LR schedule (constant / warmup + cosine / linear decay)
  * clip value
  * initialization variant
  * activation normalization to a fixed mean norm, using train statistics, with metrics reported in original units
  * periodic validation evaluation with best-checkpoint saving
  * dead-feature counting
  * optional TopK auxiliary loss for dead features
* A pilot grid at layer 6 on validation only: LR {1e-4, 3e-4, 1e-3} × schedule {constant, warmup+cosine} × steps {15k, 30k}, plus init and normalization ablations, one seed. The best 2–3 configurations are confirmed with 3 seeds at layers 2/6/10 before going to all layers.

## 5. Method E — Residual correction (two-stage SAE)

**Already implemented:** nothing.

**Build from scratch**
* SAE₁ is a frozen baseline checkpoint. SAE₂ is trained on r = h − SAE₁(h), computed on the fly (no extra storage), with the same protocol, a separate L0 target and normalization (r is much smaller than h).
* The reconstruction is ĥ = SAE₁(h) + SAE₂(r).
* Report:
  * EV and NMSE of ĥ
  * **total L0 = L0₁ + L0₂**
  * parameters (2×)
  * additional training and inference cost
* Baselines:
  * (i) SAE₁ alone;
  * (ii) **budget-matched single SAE** with width W₁ + W₂ and L0 = L0₁ + L0₂. This is the fair comparison;
  * (iii) SAE₁ trained for the same total number of steps.
* Improvement claims are made against (ii). ĥ is also evaluated with the frozen truth probes (Phase 4).

## 6. Method F — Truth-aware fine-tuning

**Already implemented**
* `06_finetune_sae.py` `truth_aware` variant: L = MSE + 0.1 · MSE(probe logits on x̂, probe logits on x) / scale.
  * The probe is the frozen logistic probe fitted on GoT **train** original activations, with C selected on validation.
  * Layer 6, 500 steps, LR 1e-5.
* Held-out evaluation with the same frozen logistic probe, an **independent** frozen MLP probe (5 seeds), RepE, and causal tests.

**Needs correction**
1. **No λ_s · L_sparsity term**, so L0 drifts (ReLU truth-aware 94 vs base 65). This is the same fix as in Method A.
2. **λ_t = 0.1 was never selected on validation.**
3. **Circular evaluation.** The probe in the loss is also an evaluation probe, and its gain (0.52 → 0.81) does not transfer to the independent MLP probe (where reconstruction-only is better). The independent probe must be the **primary** metric, with the training probe reported as secondary.
4. Layer 6 only, one seed, GoT only.

**Build from scratch**
* A λ_t sweep {0, 0.03, 0.1, 0.3, 1}, with L0 anchored and λ_t selected on validation by a pre-declared rule: maximize independent-probe validation AUROC subject to an OWT validation EV drop of at most 0.01.
* All informative layers (§0); 3 seeds for the selected λ_t.
* Transfer checks: trained on GoT, evaluated on RepE honesty (and the reverse, with an honesty-direction preservation loss).
* Comparison set: pretrained vs reconstruction-only (Method A2, same steps and L0 anchoring) vs truth-aware.

## 7. Cross-cutting infrastructure (no GPU)

1. `scripts/shared/sae_metrics.py`: MSE, NMSE, cosine, EV, L0, dead-feature fraction, in one place. This is a refactor; existing numbers must reproduce.
2. `scripts/shared/sae_checkpoints.py`: resolves (layer, arch, protocol, seed, width, variant) to a checkpoint path and config. This replaces the hardcoded paths, layer 6 and width 3072 in GoT 04/06/09, RepE 04/08/13, `12_finetuning_generalization.py` and `finetuning_heldout_eval.py`.
3. The training options listed in §4.
4. MSE-improvement reporting: 100 × (baseline − improved) / baseline, computed only between runs at matched L0 (≤ 10 %) on the same layer, seed set and split. The baseline is the uninterrupted 15k L0 = 128 run.

## 8. Extending layers 2/6/10 to all 12 layers without repeating work

**What already exists, per layer:**

| Item | Layers done | Reused as is | Missing |
|---|---|---|---|
| OWT 1k-step sweep (validation) | 0–11 | yes (convergence-speed only) | — |
| OWT uninterrupted 15k, 3 seeds (test) | 2, 6, 10 | **yes**: 27 runs, skipped automatically | 0, 1, 3, 4, 5, 7, 8, 9, 11 → **81 runs** |
| GoT original-activation probes (validation, C selected) | 0–11 | **yes** | — |
| GoT SAE frozen-probe preservation (validation) | 2, 6, 10 (mixed-procedure checkpoints) | kept as history | all 12 with uninterrupted checkpoints, 3 seeds |
| GoT final test (frozen MLP, 5 seeds) | 6 (resume checkpoint) | yes | at the end, for selected configurations only |
| RepE original baseline (validation) | 0–11 | **yes** | — |
| RepE SAE metrics (AUROC, cosine, norm ratio, pairs, cross-wording) | 6 | yes | all 12 |
| Causal (200 controls) | 6 | yes | representative layers only (cost) |

**Steps:**
1. **OWT baseline, 81 new runs.**
   * Command: `python scripts/openwebtext/13_consistent_protocol_retrain.py --layers 0 1 3 4 5 7 8 9 11`.
   * Existing (layer, arch, seed) rows are skipped. Existing checkpoints cannot be overwritten (the script exits instead). `run_config.json` keeps the earlier invocation.
   * **≈ 55 min GPU** at the measured 38–40 s per run; ~1.5 GB of checkpoints.
   * Unchanged code means layer-0 JumpReLU may again miss its L0 target. That is reported as unmatched and fixed in Method B, not patched in the baseline.
2. **Original-vs-reconstructed comparison at all 12 layers.** Parameterize GoT `04` (layers, checkpoint protocol, seeds) and the RepE 04/05/06/09 metrics through the checkpoint registry (§7.2). Outputs go to new sub-directories (`results/geometry_of_truth/sae_truth_preservation/uninterrupted_protocol/`, `results/repe/all_layers/`); existing files are untouched. This uses cached activations only, so it takes minutes (CPU is enough). The original probes are reused, not refit.
3. **Layer-6 bridge.** Evaluate the layer-6 resume checkpoints (used by every earlier downstream result) and the layer-6 uninterrupted checkpoints on the same GoT/RepE validation sets. That quantifies whether the earlier layer-6 conclusions carry over, without re-running the old pipelines.
4. **Improvement methods.** Pilot at layer 6 (Methods D → B → C → E → A → F), validation only. Confirm at layers 2/6/10 with 3 seeds, then run the selected configurations at all 12 layers. Each method's baseline at each layer is the step-1 run for that layer and seed.
5. **Test split.** Evaluated once, at the end, for the selected configurations only.

**Estimated compute (A10G):**

| Phase | GPU |
|---|---|
| Step 1 | ~1 h |
| Methods B/C/D pilots + all layers | ~5–7 h |
| Method E | ~1 h |
| Methods A/F | ~1–2 h |
| Causal at selected layers | ~1–2 h |
| Gemma (Gemma Scope, JumpReLU only, ~8 GB download) | ~3–4 h |
| **Total** | **~12–16 h**, staged with an approval point between phases |

**Decisions needed before Phase 3:**
1. Activation normalization: on by default for new runs, or only as an ablation?
2. Fine-tuning definition: A1 (OWT), A2 (target domain), or both?
3. Data size: keep 101.6k OWT tokens (comparable to all existing results) or extract a larger, versioned OWT split (~1M tokens, ~18 GB, would require re-running baselines)?

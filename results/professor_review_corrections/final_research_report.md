# SAE Faithfulness: corrections after the 2026-11-01 professor review

Branch `sae-research` @ `326b0f6`; work done 2026-11-04 on an EC2 g5.xlarge (A10G 23 GB, 15 GB RAM).
Nothing was committed, pushed, merged, deleted or overwritten. All new outputs are in
`results/professor_review_corrections/`. Detailed evidence is in `code_audit.md`; decisions are in `methodology_changes.md`.

---

## A. Executive summary

**What was wrong (verified)**

1. **OWT optimizer mismatch (Task A): confirmed.** Layer 6 was resumed with Adam and controller state restored.
   Layers 2/10 were warm-started from 1000-step weights with a fresh Adam and a reset sparsity coefficient.
   *Retraining all nine with one uninterrupted protocol (3 seeds) changed explained variance by at most 0.006, and within-layer
   rankings did not change.* The original conclusions survive. Two original "wins" are actually ties: layer 2 ReLU vs JumpReLU, and layer 10 JumpReLU vs TopK.
2. **Architecture comparison (Task B).** "2/12 JumpReLU matched at L0=32" is confirmed under the 10% criterion (5/12 under the 20% criterion; both
   criteria exist in the code). `architecture_summary.csv` averages unmatched runs and raw MSE across layers, and no script generates it.
   It is not a valid comparison. Only 15/36 sweep cells have all three architectures matched. The 1000-step sweep ranks convergence
   speed (TopK wins every matched cell), and that ranking **reverses** at 15 000 steps (JumpReLU ≥ TopK at layers 6 and 10).
3. **Causal evidence (Task C).** The earlier "SAE steering" never put an SAE in the forward pass. It steered GPT-2 with a direction computed
   from reconstructions. Its 0.93–0.99 effect correlations therefore measure direction similarity, not SAE causality.
   In the new pre-registered experiment with 200 controls:
   * The **raw honesty direction** has a statistically detectable but tiny effect: E = 0.0048 [0.0003, 0.0096] of the honest-persona
     output shift recovered, p = 0.010 vs 200 norm-matched random directions.
   * The positive control (activation patch) recovers 0.234 [0.151, 0.317].
   * **No SAE reconstruction direction and no selected SAE feature is distinguishable from controls.** The top-1 TopK/JumpReLU "honesty
     features" are *worse* than 99% of random features.
4. **Gemma (Task D).** The both-signs → TRUE observation is **reproduced**: −0.5 → 99.6% TRUE and +0.5 → 96.0% TRUE (baseline 58.8%).
   Norm-matched random directions show the same drift toward TRUE (≈73%). The −0.5 effect is indistinguishable from random (z ≈ 0), so it is a
   **generic large-perturbation response bias**, not a sign bug. At small strengths the original direction is sign-consistent but weak.
   A direction estimated in the judgment context (cosine 0.30 with the original) steers with the correct sign at every strength (23% → 100% TRUE). See §E2.
5. **GoT layer 10 (Task E).** The legacy numbers reproduce (0.7689 / 0.8529 / 0.8508). But the "reconstructed" probe was **re-trained on reconstructions**.
   Applying the **original, frozen probe** to the same reconstructions gives AUROC **0.639** (gap 0.21, CI [0.16, 0.27]; 6-split mean 0.632),
   even though reconstruction cosine is 0.977. The SAE keeps truth *linearly decodable* but does not preserve the *original truth readout*.
6. **RepE (Task F).** All values verify except one: the "cross-wording JumpReLU ≈ 0.8010" quoted for the review is 0.7394 in the saved file.
   There is no split leakage. The never-evaluated test split agrees with validation (original 0.881; SAEs 0.708–0.793).
   **SAE L0 on RepE prompts is not matched** (JumpReLU 187, ReLU 170, TopK 128), so architecture differences there are confounded with sparsity.
7. **Fine-tuning (Task G).**
   * On OWT, RepE and causal metrics the effect is negligible (|ΔAUROC| ≤ 0.007), and reconstruction-only fine-tuning slightly *hurts* OWT reconstruction.
   * The only large effect: truth-aware fine-tuning restores the GoT **frozen logistic probe** it was trained to match (geographic 0.52 → 0.81).
   * That gain does **not** transfer to an independent MLP probe, where reconstruction-only is better.
8. **AlignSAE (Task H)** is not implemented anywhere. The truth-aware loss is probe-logit preservation, not AlignSAE. A minimal design is proposed in §G.

**What remains unresolved:** a sparsity-matched L0=32/64 re-sweep (cost estimated, not run); AlignSAE; a stronger behavioral honesty task for GPT-2
(the persona effect at the final token is only 0.02 nats); SAE-in-the-loop (encode→intervene→decode) feature interventions.

---

## B. Code changes

No existing file was edited. All changes are new files.

| File | Function | Original behaviour | Corrected behaviour | Reason | Test |
|---|---|---|---|---|---|
| `scripts/openwebtext/13_consistent_protocol_retrain.py` | `main` (calls original `train_one`, `evaluate`) | L6 resumed with Adam state restored; L2/10 warm start with Adam and controller reset | One uninterrupted 15k-step run per layer × arch × seed (42, 1042, 2042), test eval on the identical 12 700 tokens | Consistent procedure across layers | Smoke run (200 steps); 27 runs completed; test-token count asserted |
| `scripts/repe/13_causal_random_controls.py` | `Runner`, `kl_rows`, `recovery`, `empirical_p`, `bootstrap_recovery_ci`, `main` | 20 random directions, 9 questions, no positive control, no p-values; no SAE in the loop | Pre-registered validation→test protocol, positive control, 200 Gaussian + 200 random-feature + 200 random-feature-set controls per SAE, empirical p, bootstrap CI, per-intervention CSV | Professor's causal-control requirement | Unit tests (KL, recovery, p, CI, **hook touches only the final non-pad position**); runtime assert that hooked activations equal cached ones (max err < 1e-3) |
| `scripts/gemma/01_steering_sweep.py` | `judge`, `make_hook`, `last_resid` | Random row split (city leakage); strengths 0.25–1.0 × act-norm; binary score; no controls; context-mismatched direction; TL loader (OOM on this host) | Grouped city split; ±0.01…0.5 grid; continuous TRUE−FALSE logit diff; 20 norm-matched random directions per strength; bare + context directions; segment + final-token modes; HF bf16 backend | Task D | Smoke run; single-token TRUE/FALSE assert; shots/prompt token-boundary assert; left-padding check (max 0.125 logit, bf16) |
| `scripts/professor_review/got_layer10_frozen_probe.py` | `main` | Probe re-trained on reconstructions | Adds frozen original probe on reconstructions, paired bootstrap CI, 5 extra city splits | Task E | Reproduces legacy numbers (see §D3) |
| `scripts/professor_review/finetuning_repe_got_eval.py` | `main` | FT variants evaluated on different splits/probes in different scripts; RepE test never evaluated | All 9 variants on identical OWT-test, GoT-test (frozen logistic + MLP), RepE val/test/cross-wording, causal | Tasks F/G | Recomputed RepE val/cross-wording/cosine equal saved CSVs (max abs diff 1.2e-4, 0, 1.2e-6) |
| `scripts/professor_review/analyze_owt_corrections.py` | `main` | `architecture_summary.csv` mixes matched/unmatched runs and layers | Separate matched/unmatched tables, fully-matched cells only, original vs corrected | Task B | Asserts 27 runs and 12 700 tokens |
| `scripts/professor_review/plot_causal_and_gemma.py`, `write_manifest.py` | — | — | Figures, manifest | — | — |
| `src/tests/test_professor_review.py` | 8 tests | — | — | Task I | `python -m pytest -q src/tests/test_professor_review.py` → **8 passed** |

Environment changes: installed `pytest` into `~/sae-project/venv`; downloaded the SAELens `gpt2-small-res-jb` layer-10 SAE (~150 MB).

---

## C. Experiment settings

* **Models:** GPT-2 Small (TransformerLens 3.9.0, fp32); Gemma-2-2B (HF transformers 5.21.0, bf16, eager attention).
  TransformerLens was OOM-killed loading Gemma on this 15 GB host; the HF decoder-layer-11 output is the same quantity as TL `blocks.11.hook_resid_post`.
* **In-house SAEs** (called "pretrained" in the repo = **trained by us on OWT**, not public):
  * d_in 768, d_sae 3072 (4×), target L0 128, Adam 3e-4, batch 256, 15 000 steps, seed 42
  * ReLU (L1 penalty with adaptive coefficient), TopK (k = 128), surrogate-gradient JumpReLU (not the published estimator)
  * OWT data: 101 600 train tokens, so 15 000 × 256 ≈ 38 epochs
* **Fine-tuned variants:** 500 steps, LR 1e-5, GoT train activations; truth-aware adds 0.1 × normalized frozen-logistic-probe logit MSE.
* **Public SAE:** SAELens `gpt2-small-res-jb / blocks.10.hook_resid_pre` (d_sae 24 576; L0 on GoT ≈ 48.5).
* **Data and splits:**

  | Dataset | Train | Validation | Test | Notes |
  |---|---|---|---|---|
  | OWT | 800 docs | 100 docs | 100 docs | ≤128 tokens each, BOS excluded: 101 600 / 12 700 / 12 700 tokens |
  | GoT layer-6 pipeline | 70% | 15% | 15% | grouped by city / number pair; test 452 geographic, 596 numerical |
  | GoT legacy (layer 10) | 1046 | — | 450 | city-grouped 70/30 |
  | RepE | 417 statements | 90 | 90 | × 2 personas; no statement or prefix overlap |
  | Gemma cities | grouped split | grouped split | grouped split | see `gemma/run_info.json` |

* **Software:** Python 3.14.4, torch 2.14.1+cu130, sae-lens 6.54.12, sklearn 1.9.1. GPU NVIDIA A10G.
* Machine-readable provenance is in `experiment_manifest.csv` and in each run's `run_config.json` / `run_info.json` / `preregistration.json`.

---

## D. Results

### D1. Reconstruction on OWT test (12 700 tokens), original vs consistent protocol — `reconstruction_comparison.csv`

| Layer | Arch | Original procedure | Orig EV | Corrected EV (mean ± sd, 3 seeds) | Orig L0 | Corrected L0 | Rank orig → corrected |
|---|---|---|---|---|---|---|---|
| 2 | ReLU | warm start, Adam reset | 0.9362 | 0.9358 ± 0.0003 | 133.0 | 131.9 | 1 → 1 |
| 2 | TopK | warm start, Adam reset | 0.9251 | 0.9259 ± 0.0003 | 128 | 128 | 3 → 3 |
| 2 | JumpReLU | warm start, Adam reset | 0.9293 | **0.9355 ± 0.0017** | 129.1 | 129.0 | 2 → 2 (≈ tie with ReLU) |
| 6 | ReLU | resumed, Adam restored | 0.8311 | 0.8315 ± 0.0008 | 131.1 | 131.0 | 3 → 3 |
| 6 | TopK | resumed, Adam restored | 0.8573 | 0.8572 ± 0.0002 | 128 | 128 | 2 → 2 |
| 6 | JumpReLU | resumed, Adam restored | 0.8658 | 0.8649 ± 0.0004 | 129.5 | 129.1 | 1 → 1 |
| 10 | ReLU | warm start, Adam reset | 0.7730 | 0.7727 ± 0.0004 | 132.6 | 132.7 | 3 → 3 |
| 10 | TopK | warm start, Adam reset | 0.8217 | 0.8225 ± 0.0004 | 128 | 128 | 2 → 2 |
| 10 | JumpReLU | warm start, Adam reset | 0.8253 | 0.8227 ± 0.0003 | 131.6 | 131.6 | 1 → 1 (≈ tie with TopK) |

All corrected runs are L0-matched within 4.1%. MSE and cosine are in the CSV. Figure: `figures/taskA_owt_original_vs_corrected_ev.png`.
**Conclusion:** the procedure mismatch did not bias the cross-layer pattern (EV falls from layer 2 to 10 for every architecture).
At L0 = 128 and 15k steps, JumpReLU ≈ ReLU > TopK at layer 2, while JumpReLU > TopK > ReLU at layers 6 and 10.

### D2. Sparsity matching — `sparsity_comparison.csv`, `sparsity_matched_cells.csv`

1000-step sweep, runs within 10% (20%) of target, out of 12 layers:

| Target | ReLU | TopK | JumpReLU |
|---|---|---|---|
| 32 | 12 (12) | 12 (12) | **2 (5)** |
| 64 | 12 (12) | 12 (12) | 5 (9) |
| 128 | 12 (12) | 12 (12) | 8 (11) |

* Cells with all three architectures matched: L0 = 32 → 2 (layers 2 and 3), L0 = 64 → 5, L0 = 128 → 8.
* TopK has the highest EV in all 15 matched cells at 1000 steps. That reflects faster convergence: at 15k steps TopK is not best at any of layers 2, 6, 10.
* Unmatched JumpReLU runs (e.g. layer 0, L0 ≈ 0.004; layer 6 at target 32, L0 = 45.2) must not be compared.
* Figure: `figures/taskB_sweep_target_vs_actual_l0.png`.

**Proposed (not run):** a 15k-step sweep at targets 32/64 for all 12 layers × 3 architectures × 1 seed = 72 runs × ~63 s ≈ **1.3 h** on the A10G
(×3 seeds ≈ 4 h). To raise the JumpReLU match rate, add a second controller phase that freezes the coefficient once the EMA L0 is within ±5%.
Report only runs that end within 10% of target.

### D3. Geometry of Truth — `got_faithfulness.csv`, `got_layer10/`

**Layer 10, public SAE (legacy experiment), city split seed 42, n_test = 450:**

| Probe | Accuracy | AUROC | Gap vs original [95% CI] |
|---|---|---|---|
| Original probe on original activations | 0.7689 (legacy 0.7689) | 0.8529 (legacy 0.85284) | — |
| Probe **re-trained** on reconstructions (legacy method) | 0.7600 (legacy 0.7556) | 0.8508 (legacy 0.85079) | 0.002 [−0.028, 0.033] |
| **Frozen original probe** on reconstructions | 0.500 | **0.639** | **0.214 [0.162, 0.268]** |

* Across 6 city splits: original 0.860, re-trained 0.861, frozen 0.632 (frozen gap 0.13–0.32 on every split).
* Reconstruction on test: cosine 0.977, MSE 1.82, **explained variance −1.50**, L0 48.5.
* The final-token activations share a large common component. High cosine therefore coexists with errors bigger than the
  between-statement variance that carries truth.
* The accuracy difference vs legacy (2 of 450 test items) comes from CPU-vs-GPU SAE encoding. AUROCs agree to 4 decimals.

**Layer 6, in-house SAEs, grouped test split, frozen probes trained on original train activations:**

| Arch / variant | Geo frozen-logistic AUROC (orig 0.825) | Num frozen-logistic (orig 0.999) | Geo frozen-MLP (orig 0.876, 5 seeds) | Num frozen-MLP (orig 1.000) | Geo L0 |
|---|---|---|---|---|---|
| ReLU base | 0.525 | 0.544 | 0.549 | 0.704 | 65 |
| ReLU recon-only | 0.547 | 0.546 | 0.715 | 0.911 | 144 |
| ReLU truth-aware | **0.806** | **0.943** | 0.651 | 0.805 | 94 |
| TopK base | 0.565 | 0.559 | 0.722 | 0.901 | 128 |
| TopK recon-only | 0.529 | 0.571 | 0.766 | 0.955 | 128 |
| TopK truth-aware | 0.749 | 0.855 | 0.692 | 0.913 | 128 |
| JumpReLU base | 0.536 | 0.578 | 0.553 | 0.633 | 45 |
| JumpReLU recon-only | 0.530 | 0.590 | 0.730 | 0.869 | 100 |
| JumpReLU truth-aware | 0.762 | 0.838 | 0.658 | 0.782 | 65 |

The same frozen probe is applied to original and reconstructed activations in both probe families. Probes are fitted on train only, and the test split is used only here.

### D4. RepE (layer 6, final token, frozen train-split honesty direction) — `repe_representation.csv`

| Model | Val AUROC | **Test AUROC** | Cross-wording AUROC | Direction cosine | Test EV | Test L0 |
|---|---|---|---|---|---|---|
| Original GPT-2 | 0.8926 | 0.8807 | 0.8396 | 1 | — | — |
| ReLU base | 0.7149 | 0.7077 | 0.6560 | 0.754 | 0.777 | 170 |
| TopK base | 0.7864 | 0.7858 | 0.7570 | 0.844 | 0.797 | 128 |
| JumpReLU base | 0.7979 | 0.7921 | **0.7394** | 0.864 | 0.848 | 187 |
| (fine-tuned variants) | ±0.011 | ±0.007 | ±0.004 | ±0.010 | | |

* Pairwise accuracy is 1.000 for every model on every split. It is saturated and is not used as evidence.
* Note the ordering flip: JumpReLU has the best direction cosine and test AUROC, but TopK has the best cross-wording AUROC.
  A higher direction cosine does not guarantee better held-out transfer.

### D5. Fine-tuning — `finetuning_comparison.csv` (identical held-out sets for all 9 variants)

| Metric (held-out) | ReLU base → recon-only / truth-aware | TopK | JumpReLU |
|---|---|---|---|
| OWT test EV | 0.831 → 0.831 / 0.832 | 0.857 → **0.850** / 0.856 | 0.866 → **0.849** / 0.864 |
| GoT geo frozen-logistic AUROC | 0.525 → 0.547 / **0.806** | 0.565 → 0.529 / **0.749** | 0.536 → 0.530 / **0.762** |
| GoT geo frozen-MLP AUROC | 0.549 → **0.715** / 0.651 | 0.722 → **0.766** / 0.692 | 0.553 → **0.730** / 0.658 |
| RepE test AUROC | 0.708 → 0.711 / 0.710 | 0.786 → 0.792 / 0.792 | 0.792 → 0.788 / 0.793 |
| RepE direction cosine | 0.754 → 0.756 / 0.756 | 0.844 → 0.842 / 0.843 | 0.864 → 0.874 / 0.870 |
| Causal E, recon direction (p vs 200 Gaussian) | −0.0009 (0.66) → −0.0011 / −0.0009 | 0.0010 (0.30) → 0.0008 / 0.0010 | 0.0024 (0.08) → 0.0031 (0.050) / 0.0027 (0.065) |

**Interpretation:**

* *Reconstruction-only* fine-tuning improves in-domain GoT reconstruction (MSE ÷ 8–10) and MLP-probe AUROC. It does not improve the frozen logistic readout. It slightly degrades OWT reconstruction (TopK, JumpReLU) and raises L0 (ReLU 65 → 144 on GoT).
* *Truth-aware* fine-tuning succeeds on exactly the quantity it optimizes (the frozen logistic probe). It is worse than reconstruction-only under an independent MLP probe.
* On RepE and causal metrics, neither variant has an effect beyond noise.
* Base and fine-tuned SAEs have different L0, so these comparisons are not sparsity-matched.

---

## E. Causal experiment (Task C) — `causal_controls_200.csv` (3 830 interventions), `causal_summary.csv`

**Design.** Protocol fixed in `causal/preregistration.json` before test evaluation, with one disclosed deviation (see `methodology_changes.md` §C).

* GPT-2 Small, layer 6 `hook_resid_pre`, final prompt token.
* Behavior: next-token distribution on 90 held-out untruthful-persona test prompts.
* Outcome: E = fraction of the natural honest-vs-untruthful KL gap closed (baseline mean KL = 0.022 nats).
* Strength: chosen on validation = 0.1 × ‖mean diff‖ = 0.223 (every larger strength made E negative on validation, for target and controls alike).
* Every intervention uses the same prompts, the same scoring and the same norm.

| Intervention | E [95% bootstrap CI] | Control distribution (mean ± sd, max) | Empirical p |
|---|---|---|---|
| Baseline (none) | 0 | — | — |
| **Positive control:** patch honest final-token activation (mean ‖Δ‖ = 3.90) | **0.234 [0.151, 0.317]** | Gaussian −0.0002 ± 0.0020, max 0.0058 | 0.005 (= 1/201, the floor) |
| **Raw honesty direction** (train mean difference) | **0.0048 [0.0003, 0.0096]** | same 200 Gaussian | **0.010** (z = 2.5) |
| SAE reconstruction directions, 9 variants | −0.0011 … 0.0031 | 200 Gaussian | 0.050 – 0.68 |
| SAE top-1 honesty feature, 9 variants | −0.0073 … −0.0024 | 200 random alive features of the same SAE | 0.84 – 1.00 |
| SAE top-10 feature set, 9 variants | −0.0052 … 0.0012 | 200 random 10-feature sets | 0.26 – 1.00 |

The secondary outcome (S, true-vs-false next-token log-prob) is null for all targets: raw direction S = −0.0009, p = 0.87.
Figures: `figures/taskC_causal_vs_200_gaussian_controls.png`, `figures/taskC_top1_feature_vs_200_random_features.png`.

**What this shows:**

* The mean-difference honesty direction has a direction-specific causal effect on output that is distinguishable from random
  directions (p = 0.010, the single pre-specified primary test). However, it is ~50× smaller than the positive control, and
  increasing its strength makes the output *less* honest-like. Most of the persona information that the patch transfers is
  not along the mean direction.
* **No SAE-derived intervention is distinguishable from controls.** That includes reconstruction directions with 0.75–0.87 cosine to the raw direction.
  A high direction cosine and high steering-effect correlation (the earlier results) therefore do **not** establish causal faithfulness.
* With 200 controls the smallest attainable p is 0.005. For the 27 SAE comparisons, Bonferroni (α = 0.05/27 = 0.0019) is unattainable.
  The SAE results are secondary and descriptive. The JumpReLU recon-only p = 0.050 should not be read as significant.

### E2. Gemma-2-2B steering (Task D) — `gemma_steering_sweep.csv`

**Setup.**

* Gemma-2-2B (HF transformers, bf16), layer 11 `resid_post` (the original script's layer), 4-shot "This statement is: TRUE/FALSE" prompt.
* GoT cities with grouped-by-city splits: val n = 224, test n = 226.
* Outcome: LD = logit(" TRUE") − logit(" FALSE") at the final position, plus % judged TRUE and accuracy.
* Signed strengths α ∈ ±{0.01, 0.025, 0.05, 0.10, 0.25, 0.5} × mean activation norm (147.1), the original script's unit.
* Two directions:
  * *bare*: the original, true − false at the last token of the bare statement; separation ‖μ_T − μ_F‖ = 48.4, so α = 0.5 ≈ 1.5 separations
  * *context*: estimated at the last token of the judgment prompt itself; separation 12.1, so α = 0.5 ≈ 6.1 separations
  * cos(bare, context) = **0.30**
* Two intervention sites: *segment* (every token of the test statement + suffix, as in the original) and *final* (last token only).
* 20 norm-matched Gaussian directions per strength and site. With 20 controls, z-scores are descriptive; the smallest attainable empirical p is 0.048.
* No strength was selected on test; the full grid was run on both splits, and validation and test agree within ~2 pp.

**Test-split results** (baseline: LD 0.32, 58.8% judged TRUE, accuracy 91.2%):

| Site | Direction | α = −0.5 | −0.25 | −0.10 | +0.10 | +0.25 | +0.5 |
|---|---|---|---|---|---|---|---|
| segment | bare (original) | **99.6% TRUE**, ΔLD +0.22 (z −0.0) | 65.0%, +0.00 | 56.6%, −0.08 | 61.1%, +0.17 (z 2.0) | 73.9%, +0.45 | **96.0% TRUE**, +0.79 (z 1.7) |
| segment | context | **23.0% TRUE**, −0.61 (z −2.1) | 52.7%, −0.35 | 54.9%, −0.22 (z −2.7) | 61.9%, +0.36 (z 4.9) | 80.1%, +0.96 (z 5.4) | 100%, +1.59 (z 4.6) |
| segment | 20 random (mean) | 73.5%, +0.23 ± 0.40 | 60.4%, +0.04 | 58.3%, −0.01 | 59.1%, +0.04 | 62.6%, +0.13 | 73.8%, +0.32 ± 0.28 |
| final | bare | 94.7%, +0.34 (z 0.9) | 59.7%, +0.05 | 56.2%, −0.02 | 61.5%, +0.06 | 64.2%, +0.16 | 72.6%, +0.34 (z 0.5) |
| final | context | 71.2%, +0.19 | 63.3%, +0.11 | 60.2%, +0.04 | 56.2%, −0.05 | 53.5%, −0.10 | 54.4%, +0.12 |
| final | 20 random (mean) | 62.9%, +0.10 ± 0.27 | 58.3%, −0.00 | 57.9%, −0.01 | 59.0%, +0.03 | 61.0%, +0.10 | 68.6%, +0.25 ± 0.17 |

The full grid, both splits, and all 20 individual controls are in `gemma/gemma_steering_sweep.csv` and `gemma/gemma_steering_sweep_all.csv`.
Figures: `figures/taskD_gemma_steering_test.png` and `figures/taskD_gemma_steering_val.png`.

**Findings:**

1. **The professor's observation is reproduced** with the original construction (bare direction, segment site): α = −0.5 → 99.6% TRUE and +0.5 → 96.0% TRUE (baseline 58.8%). At ±0.25 it is 65.0% / 73.9%.
   At −0.5, the true/false discrimination of LD collapses: AUROC 0.55, and true- and false-statement LD converge to 0.56 / 0.53.
2. **The random controls show that most of this is a generic perturbation effect.**
   * Norm-matched random directions at ±0.5 also raise LD (mean +0.23 / +0.32) and push judgments to ~73% TRUE.
   * The bare direction's −0.5 effect is indistinguishable from random (z = −0.02), and its +0.5 effect is within the control range (z = 1.7).
   * Large perturbations of either sign drive Gemma toward its default "TRUE" response (baseline already 58.8% TRUE with 50% true statements) and destroy discrimination.
   * This is a response-bias/off-manifold effect, not a sign error. The direction sign, normalization, hook and TRUE/FALSE token scoring were checked and are correct (both answers are single tokens).
3. **At small strengths the bare direction behaves as hypothesized**, but weakly: monotonic over |α| ≤ 0.10 (−0.08 at −0.10, +0.17 at +0.10), z ≈ 1–2 vs controls. Judgments barely move (56.6% → 61.1% TRUE).
4. **A context-matched direction steers with the correct sign at every strength** at the segment site: −0.5 → 23% TRUE, +0.5 → 100% TRUE; z = −2.1 to −3.9 on the negative side and +3.8 to +5.4 on the positive side.
   * The original direction was estimated on bare statements, while the intervention is applied inside a few-shot judgment context. The two directions share only cosine 0.30.
   * That **context mismatch**, combined with strengths of ~0.75–1.5 separations, explains the original result.
5. **Final-token-only steering** with the bare direction is indistinguishable from random (|z| < 1).
   The context direction at the final token does not bias toward TRUE. Instead it widens the true/false gap (true-statement LD 1.29 → 1.64, false −0.64 → −1.18; accuracy 91.2% → 96.5% at +0.25). This is a single, unreplicated observation (20 controls), reported descriptively.

**Answer to the professor:** the "both signs → TRUE" result is real but is mostly a generic large-perturbation response bias, not evidence that the truth direction has the wrong sign. With matched-norm controls, smaller strengths and a direction estimated in the judgment context, steering is sign-consistent and direction-specific.
Remaining caveats:
* Only 20 controls per condition (not 200).
* Only one layer and one dataset (cities).
* The neg_cities next-word crash in the original script was diagnosed but not re-run.

---

## F. Scientific interpretation

**Supported:**

* At matched L0 = 128 on OWT, JumpReLU (surrogate) and ReLU reconstruct at least as well as TopK after 15k steps. Reconstruction degrades with depth for all three. These conclusions are robust to the optimizer-procedure correction.
* SAE reconstructions preserve activation geometry (cosine 0.95–0.98) while losing much of the information read by the *original* truth/honesty readouts:
  * GoT frozen-probe AUROC drops from 0.85 to 0.64 at layer 10 (public SAE) and from 0.83 to 0.53–0.57 at layer 6 (in-house base SAEs, geographic test)
  * RepE AUROC drops from 0.88 to 0.71–0.79

  This is a legitimate negative finding about faithfulness.
* Re-training a probe on reconstructions hides this loss: truth remains decodable, but via different directions.
* Fine-tuning improves what it directly optimizes and little else.

**Not supported:**

* That any selected SAE feature has a specific causal effect on honesty behavior.
* That steering-effect correlation implies causal faithfulness.
* That truth-aware fine-tuning improves general truth preservation.
* That TopK is the best architecture (that is a 1000-step artifact).
* That any of these results transfer to wide public SAEs or to the published JumpReLU.

---

## G. Remaining limitations and proposed work

1. **Sparsity matching across datasets.** L0 is matched only on OWT. On RepE/GoT the active-feature counts differ by up to 4×. A matched-on-target-data comparison (e.g. re-calibrating TopK k or JumpReLU thresholds per dataset) is needed.
2. **Narrow SAEs** (3072) and small data (101k tokens). JumpReLU is a surrogate. The layer-10 public-SAE result is the only wide-SAE evidence.
3. **Causal task power.** The GPT-2 persona effect at the final token is tiny (0.02 nats) and is mostly not carried at that site (the patch recovers 23%).
   * A multi-position intervention and a task with a larger behavioral contrast would be more informative.
   * SAE-in-the-loop feature clamping (encode → set feature → decode + error term) was not run.
   * 200 controls cap resolution at p = 0.005. Use ≥ 2 000 controls if many comparisons are needed.
4. **Single-seed GoT layer-6 fine-tuning;** a single split for the RepE test.
5. **Task B re-sweep** at L0 = 32/64 (≈ 1.3 h): proposed, not run.
6. **AlignSAE (Task H), not implemented; minimal proposal.**
   * *Difference from what exists:* a truth **probe** is a classifier on activations. Truth **preservation** evaluation asks whether that classifier still works on x̂.
     **AlignSAE** changes SAE training so that a designated latent *is* the concept.
   * *Design:*
     * Start from the layer-6 TopK base SAE. Reserve latent j*.
     * Loss = MSE(x̂, x) + λ_a · BCE(σ(a·z_j* + b), y_truth) on GoT train
       + λ_s · mean_{j≠j*} |corr(z_j, y)| (concept exclusivity).
     * k fixed at 128 for matched sparsity. Compare against the base SAE and reconstruction-only fine-tuning, with identical steps and LR.
   * *Held-out metrics:*
     * GoT test: AUROC of z_j* alone; best single latent of the base SAE
     * OWT test: MSE, EV, L0
     * concept selectivity, i.e. mean activation of j* on true vs false and on OWT
     * the §E causal test using decoder column j* against 200 random latents
   * *Cost:* < 15 min GPU.
7. The legacy GoT inputs (`data/raw/cities.csv`, `data/processed/…`) and `data/got/splits/` are not in git. Commit the split files.

---

## H. Professor review checklist

| Request | Status | Evidence |
|---|---|---|
| A. OWT optimizer consistency, layers 2/6/10, all archs | **Completed and verified** | `reconstruction_comparison.csv`, `owt_consistent_protocol/`, fig A |
| A. 12 700 test tokens consistent | **Completed and verified** | asserted in analysis script |
| A. FT variants re-evaluated on OWT **test** | **Completed and verified** | `finetuning_comparison.csv` |
| B. Target vs actual L0; matched/unmatched tables | **Completed and verified** | `sparsity_comparison.csv`, `sparsity_matched_cells.csv`, fig B |
| B. "2 of 12" check | **Completed and verified** | §D2 |
| B. New sparsity-matched L0=32/64 runs | **Not started** (cost estimated: 1.3 h) | §D2 |
| C. 200-control causal experiment | **Completed and verified** (one disclosed protocol deviation) | `causal_controls_200.csv`, `causal_summary.csv`, `causal/` |
| C. Professor's exact protocol text | **Blocked**: not in the repo; design is ours | `methodology_changes.md` §C |
| D. Gemma bug/confound investigation | **Completed** | `code_audit.md` Task D |
| D. Symmetric small-strength sweep with controls | **Completed and verified** (20 controls per condition, not 200) | `gemma_steering_sweep.csv`, fig D |
| D. neg_cities next-word crash | **Implemented but not evaluated** (diagnosed only; not part of the sweep) | `code_audit.md` |
| E. Verify GoT values; frozen-probe comparison | **Completed and verified** | `got_faithfulness.csv`, `got_layer10/` |
| F. Verify RepE values; leakage; held-out | **Completed and verified** (one mismatch reported) | `repe_representation.csv` |
| G. Fine-tuning comparison on identical held-out data | **Completed and verified** | `finetuning_comparison.csv` |
| H. AlignSAE | **Not started** (proposal in §G) | — |
| I. Tests, manifest, provenance | **Completed** (8 tests pass) | `src/tests/test_professor_review.py`, `experiment_manifest.csv` |

---

## I. Reproduction (from the repository root, with `source ../venv/bin/activate`)

```bash
python scripts/openwebtext/13_consistent_protocol_retrain.py            # Task A, ~30 min GPU
python scripts/professor_review/analyze_owt_corrections.py              # Tasks A/B tables + figures
python scripts/repe/13_causal_random_controls.py --phase validation     # Task C: writes preregistration.json
python scripts/repe/13_causal_random_controls.py --phase test           # Task C: 3 830 interventions, ~2 min
python scripts/gemma/01_steering_sweep.py                               # Task D, ~35 min GPU, ~6 GB VRAM
python scripts/professor_review/got_layer10_frozen_probe.py             # Task E (downloads jb L10 SAE once)
python scripts/professor_review/finetuning_repe_got_eval.py             # Tasks F/G (needs Task C and E outputs)
python scripts/professor_review/plot_causal_and_gemma.py                # figures C/D
python scripts/professor_review/write_manifest.py                       # experiment_manifest.csv
python -m pytest -q src/tests/test_professor_review.py                  # 8 tests
```

Required untracked inputs: `data/activations/gpt2_small/` (2.3 GB, from `scripts/openwebtext/02_extract_activations.py`) and
`data/got/splits/` (from `scripts/Geometry of Truth/02_create_grouped_splits.py`).

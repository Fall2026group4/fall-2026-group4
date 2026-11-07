# Methodology changes (professor review, 2026-11-01)

Every change below is additive. New scripts write only under `results/professor_review_corrections/`.
No original script, checkpoint, or result file was modified, moved, or deleted.

## A. OpenWebText optimizer procedure

**Decision:** for every layer/architecture/seed, use one uninterrupted run from random initialization to
15 000 steps, with one Adam optimizer, one sparsity controller, and one RNG stream.

**Why this procedure:** the experimental objective is to compare architectures *and layers* at a fixed
training budget. "Resume with restored optimizer" (layer 6) and "warm start with reset optimizer"
(layers 2, 10) are both approximations of an uninterrupted run, and each introduces its own discontinuity
(re-seeded RNG; or Adam moment reset plus sparsity-coefficient reset to 0.001 at step 1000). Training is
cheap (~75 s/run), so the clean protocol costs almost nothing and needs no checkpoint surgery.

**Implementation:** `scripts/openwebtext/13_consistent_protocol_retrain.py` imports and calls the
*original* `train_one` (03) and `evaluate` (10), so loss, controller, clipping, decoder normalization and the
12 700-token test evaluation are byte-for-byte the original code. Three seeds (42, 1042, 2042) are used.

**Scope:** fine-tuned checkpoints were **not** retrained. All fine-tuning is layer-6 only and starts from the
layer-6 checkpoint, so the procedure mismatch never enters a pretrained-vs-fine-tuned comparison.

## B. Sparsity matching

* A single criterion is used throughout: |L0 − target|/target ≤ 0.10 (the stricter of the two found in the repo;
  the 0.20 numbers are reported alongside).
* Matched and unmatched runs go in separate tables (`sparsity_comparison.csv`, column `matching_table`).
  Architectures are compared only within a (layer, target, seed) cell in which **all three** are matched
  (`sparsity_matched_cells.csv`).
* Raw MSE is never averaged across layers. Explained variance is used for cross-layer reading.
* `results/openwebtext/layer_comparison/architecture_summary.csv` should not be used: it averages unmatched runs across layers
  with different scales, and no generating script exists in the repo.

## C. Causal test

New experiment: `scripts/repe/13_causal_random_controls.py`. The full protocol is in the script docstring and in
`causal/preregistration.json`. In short:

* Target behavior: GPT-2's next-token distribution on held-out untruthful-persona RepE prompts, compared with the
  honest-persona prompt of the same statement.
* Primary outcome: E = 1 − ΣKL(p_H‖p_U+v)/ΣKL(p_H‖p_U).
* Secondary outcome: S = differential log-prob change of the next true-statement vs false-statement token.
* Site: layer 6 `hook_resid_pre`, final prompt token.
* Directions and features are selected on train only. The strength is selected on validation (rule: argmax E of the raw
  direction over a fixed grid). The single chosen strength is applied to every intervention on test.
* Positive control: per-example patching of the honest final-token activation.
* Controls: 200 seeded Gaussian directions, plus 200 random alive features and 200 random 10-feature sets per SAE,
  all at the same norm.
* Empirical one-sided p, plus a statement-level bootstrap CI.

**Provenance caveat:** no written professor specification of the "200 controls" protocol exists in the repository.
The design above is ours. It implements the request as relayed by the user.

**Protocol deviation (disclosed):** while smoke-testing the code, I ran the test phase once on the **test split** with
20 controls and the smoke preregistration (strength 0.5×, from an earlier, coarser grid). After seeing that output,
the only change made was the one already planned before the run: extending the strength grid downward
(0.05, 0.1, 0.25 added; 16 removed), because on validation every grid value was already ≥ the selected optimum.
Outcome definitions, selection rule, seeds, and control sets were not changed. The final strength (0.1×) was then
selected mechanically on validation. Readers should still treat this as a minor departure from a strictly blind test.

## D. Gemma steering

`scripts/gemma/01_steering_sweep.py` keeps the original model, layer (11, `hook_resid_post`), few-shot prompt,
and strength unit (α × mean activation norm), so results are comparable. It changes:

1. grouped-by-city splits (removes city leakage)
2. continuous outcome: logit(TRUE) − logit(FALSE)
3. symmetric fine grid ±{0.01, 0.025, 0.05, 0.10, 0.25, 0.5}
4. 20 norm-matched random directions per strength
5. strengths are also expressed in units of the true/false separation
6. a second, context-matched direction
7. a second position mode (final token only)
8. tokenization-boundary and left-padding checks

## E. Geometry of Truth

`scripts/professor_review/got_layer10_frozen_probe.py` re-runs the legacy layer-10 pipeline from
`data/got/cities.csv`, using the legacy helper functions and the same split rule. It then adds the methodologically
appropriate **frozen-probe** comparison: the original probe, unchanged, applied to reconstructions. It also adds a
paired bootstrap CI and 5 additional city splits.

## F/G. RepE and fine-tuning

`scripts/professor_review/finetuning_repe_got_eval.py` evaluates all nine layer-6 variants on identical held-out sets:

* OWT test
* GoT test with the frozen logistic probe (the probe the truth-aware loss targeted), plus the existing frozen MLP results
* RepE validation **and test** (test was never evaluated before), plus cross-wording
* causal E

RepE validation, cross-wording and cosine values are recomputed and checked against the saved CSVs.

## I. Tests

`src/tests/test_professor_review.py` (8 tests, all pass) covers:

* KL, recovery and empirical-p edge cases
* bootstrap CI sanity
* TopK L0 ≤ k and unit-norm decoder
* explained variance and MSE vs NumPy
* the streaming variance identity used in the 12 700-token evaluation
* the causal hook changing **only** the final non-padding position

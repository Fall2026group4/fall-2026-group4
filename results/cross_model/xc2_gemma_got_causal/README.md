# XC-2: GoT causal truth-direction test on Gemma-2-2B (2026-11-23). Status: COMPLETED (frozen 2026-11-23 15:33 UTC; TEST pass done once)

Evidence level: test phase = confirmed test result for families A, C1 and C2, subject to the TEST-exposure disclosure
below; everything else is exploratory.

Frozen protocol and hashes: [`preregistration.json`](preregistration.json).
Script: [`scripts/cross_model/21_xc2_gemma_got_causal.py`](../../../scripts/cross_model/21_xc2_gemma_got_causal.py).

## 1. Before implementation

* **Research question.** Do the original truth directions and the Gemma Scope truth latents causally move Gemma's
  next-token preference toward the TRUE country? Is the GPT-2 null result (Dhruv, `dhruv-causal` commit 989d313,
  600 controls) specific to GPT-2?
* **Why.** The GoT causal test existed only for GPT-2, with rebuilt SAEs (audit 2026-11-21, row 13). XC-2 adds the
  causal category for the second model, using public SAEs.
* **Port.**
  * The protocol is unchanged from `dhruv-causal:scripts/geometry_of_truth/22_truth_direction_causal_test.py`:
    * cities pairs: the shared prefix before the first differing token, with the true token c and the false token w;
    * outcome: Δ(log P(c) − log P(w)) at the final prefix token, higher is better;
    * targets: the mass-mean direction, the frozen-probe direction, and the SAE reconstruction, top-1 and top-10
      directions, each at ±s;
    * strength chosen on VALIDATION;
    * a graded unembedding positive control;
    * 600 controls per family;
    * BH correction within families A (28 tests), C1 (8) and C2 (12);
    * criteria T1 and T2, and power reported as the minimum detectable dose.
  * **Substitutions:**
    * **Model and site:** Gemma-2-2B `resid_post` at L = 8/12/16/21, matched in depth to GPT-2 4/6/8/10.
    * **SAE:** one Gemma Scope 16k canonical SAE per layer. There are no seeds, so T2's same-sign-across-seeds clause
      does not apply.
    * **Logit path:** final-token logits use an fp32 soft-cap. HF itself applies the soft-cap in bf16; replicating
      that bf16 path matches the full forward exactly, and is used only to check the indexing.
* **TEST disclosure.** GoT cities TEST rows are not pristine:
  * they were exposed during GPT-2 method decisions;
  * the 2026-11-01 Gemma L11 judgment-steering sweep scored them under a different protocol with 20 controls.

  No XC-2 choice was taken from those numbers. TEST is scored once.
* **Smoke tests (TRAIN pairs only).**
  * **`smoke/`:** found up to 0.34 nats per-pair difference between batch shapes (bf16 kernel selection). Fixed before
    freezing: every forward now has an identical shape, the baseline goes through the same path, and the path checks
    are hard stops.
  * **`smoke_v2/`:** re-checks the fix.
  * **Ceiling:** the smoke showed Gemma's baseline pair accuracy is 1.0. The outcome is therefore the continuous
    margin; accuracy is uninformative.

## 2. After execution

**Runs.**
* Smoke v2, then freeze (15:33 UTC), then VALIDATION (about 7 min, 112 pairs), then TEST once (15:41–16:40 UTC,
  3,671 s, 113 pairs, 24,068 interventions).
* About 75 GPU minutes in total, on an A10G with torch 2.14.1, bf16, TF32 off.
* Storage: 3 MB.
* Log: `logs_xm/xc2_full.log`.
* TEST used the frozen `validation/selection.json` (sha256 63c80882…). The script's SHA-256 equals the pre-registered
  one (9b67362e…).
* SAEs: Gemma Scope 16k canonical, L0 71/82/78/70 at L8/12/16/21; their SHA-256 values are in `test_run_info.json`.
* Output SHA-256 (first 12 characters):
  * `summary.csv` c85e01810fed
  * `criteria.json` 9ac8d407f4c7
  * `interventions.csv` 068ec10b9a00
  * `per_pair_effects.npz` 3e869566a7aa
* `interventions_partial.csv` is byte-identical to `interventions.csv` and is not committed.

**Integrity.**
* At every layer, the zero delta, the final-token path vs the full forward, and the same delta at any chunk position
  all agree exactly (0.0).
* 0 pairs dropped (train / val / test = 523 / 112 / 113).
* The probe's C is chosen on validation (0.01, 0.01, 0.01 and 0.1).
* **Ceiling.** Baseline TEST pair accuracy is 0.938, with a mean margin of 11.7 nats.

**Strength (VALIDATION):** multiples 8 / 4 / 2 / 4 × ‖Δ_TRAIN‖, giving s = 141 / 197 / 141 / 352.
* The L8 validation curve is non-monotonic: +0.15 at ×2, −0.11 at ×4, +0.84 at ×8. The pre-registered argmax rule
  picked ×8.

**Power (family A).** The unembedding-contrast positive control beats all 600 controls at every λ, down to 0.01, at
every layer (BH p 0.0017). The minimum detectable dose is 0.01 everywhere, so a null here would have been informative.

**TEST results, families C1 and C2.** Effect is the mean change in log P(true) − log P(false), with a 95 % bootstrap CI
over pairs. Controls are 600 per group, norm s.

| Layer (GPT-2 match) | Direction | +s effect [95 % CI] | −s effect | Control mean / q95 | BH p | Criterion |
|---|---|---|---|---|---|---|
| 8 (4) | mass-mean | **+0.71** [0.40, 1.03] | −4.58 | −1.64 / −0.39 | 0.003 | **T1 met** |
| 8 | logistic probe | −0.63 [−0.88, −0.37] | −2.52 | −1.64 / −0.39 | 0.10 | T1 not met |
| 8 | SAE reconstruction | **+1.56** [1.33, 1.80] | −4.16 | −1.64 / −0.39 | 0.008 | **T2 met** |
| 8 | SAE top-1 | +0.47 [0.36, 0.58] | −0.66 | −2.00 / −0.07 | 0.017 | **T2 met** |
| 8 | SAE top-10 | +0.61 [0.33, 0.90] | −2.98 | −2.20 / −0.42 | 0.013 | **T2 met** |
| 12 (6) | mass-mean | **+1.98** [1.69, 2.29] | −6.92 | −1.92 / −0.23 | 0.003 | **T1 met** |
| 12 | logistic probe | −0.53 [−0.91, −0.14] | −6.34 | −1.92 / −0.23 | 0.097 | T1 not met |
| 12 | SAE reconstruction | **+3.37** [2.93, 3.80] | −6.19 | −1.92 / −0.23 | 0.008 | **T2 met** |
| 12 | SAE top-1 | −2.84 [−3.25, −2.43] | −9.59 | −2.36 / +0.08 | 0.73 | T2 not met |
| 12 | SAE top-10 | +3.06 [2.60, 3.54] | −6.68 | −2.55 / −0.01 | 0.008 | **T2 met** |
| 16 (8) | mass-mean | **+2.55** [2.24, 2.87] | −3.11 | −0.19 / +1.04 | 0.003 | **T1 met** |
| 16 | logistic probe | **+2.52** [2.07, 2.97] | −3.07 | −0.19 / +1.04 | 0.003 | **T1 met** |
| 16 | SAE reconstruction | **+3.75** [3.39, 4.11] | −3.49 | −0.19 / +1.04 | 0.008 | **T2 met** |
| 16 | SAE top-1 | +0.05 [−0.17, 0.27] | −0.11 | −0.64 / +1.22 | 0.73 | T2 not met |
| 16 | SAE top-10 | +1.32 [1.13, 1.53] | −1.53 | −0.64 / +1.22 | 0.12 | T2 not met |
| 21 (10) | mass-mean | **+3.39** [2.82, 3.99] | −4.21 | −1.80 / −0.13 | 0.003 | **T1 met** |
| 21 | logistic probe | −0.30 [−1.01, +0.45] | −2.87 | −1.80 / −0.13 | 0.097 | T1 not met |
| 21 | SAE reconstruction | **+2.33** [1.88, 2.79] | −3.79 | −1.80 / −0.13 | 0.008 | **T2 met** |
| 21 | SAE top-1 | −0.02 [−0.41, +0.40] | −1.94 | −1.20 / +1.28 | 0.49 | T2 not met |
| 21 | SAE top-10 | −1.01 [−1.30, −0.70] | −1.61 | −1.35 / +1.38 | 0.73 | T2 not met |

**Criteria.**
* **T1 (a raw truth direction steers): met for the mass-mean direction at all 4 layers.** Its +s effect is positive,
  with CI low > 0 at every layer; it beats all 600 Gaussian controls; and −s reverses the sign. The frozen
  logistic-probe direction meets T1 only at L16.
* **T2 (SAE directions steer).**
  * The reconstruction direction is met at all 4 layers.
  * Top-10 is met at L8 and L12.
  * Top-1 is met only at L8.
  * The "same sign in all seeds" clause does not apply (one SAE per layer).
* **Power:** the minimum detectable dose is 0.01 at every layer.

**Interpretation (causal-intervention category only).**
* Unlike GPT-2 under the same protocol (the `dhruv-causal` null), Gemma's GoT truth direction, estimated at the
  statement's final token, causally raises the TRUE country's next-token preference when added at the final prefix
  token. This holds at all four depths.
* The mass-mean of Gemma Scope reconstructions carries the effect as well, often more strongly: +1.56 to +3.75 nats
  vs +0.71 to +3.39 for the raw direction.
* Single SAE latents mostly do not. This fits a distributed truth representation.
* The logistic-probe direction mostly does not steer even where mass-mean does. This matches the known gap between
  discriminative and causal directions.

**Important caveats.**
1. **Large strengths.** At the validation-selected strengths (2–8 × ‖Δ‖), norm-matched random directions *lower* the
   margin by about 1.6–1.9 nats on average (except at L16). So "beats controls" partly means "avoids the damage random
   pushes cause". The absolute positive effects (CI low > 0) are the more conservative evidence, and they hold for
   mass-mean at every layer.
2. **The L8 strength was picked from a non-monotonic validation curve.**
3. **Ceiling.** Gemma already prefers the true country in 94 % of TEST pairs; effects are margin changes, not accuracy
   gains.
4. **TEST exposure.** The cities TEST rows are not pristine (disclosed above).
5. **Comparison with GPT-2 is descriptive.** It involves different SAEs (rebuilt in-house vs Gemma Scope), different
   tokenizers and different strength selections; no cross-model test was pre-registered.

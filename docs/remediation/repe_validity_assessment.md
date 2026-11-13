# RepE-LM probe and causal validity assessment (2026-11-13; remediation S5, S6, S7)

Evidence level: **VALIDATION, descriptive.** Every number below is re-read from committed outputs. No new model run
was made. Sources:
* `results/repe/repe_lm_gates/{gates.json,probe_auroc_by_layer.csv}` (M1);
* `results/repe/sae_patch_decomposition/` (M4);
* `results/shared/power_sensitivity/` (this remediation);
* `results/openwebtext/r2_scaled_saes/comparisons.csv`.

## 1. Does the persona probe measure honesty, or word identity? (S6)

Final-token residual of original GPT-2, logistic probe, validation:

| Quantity | L1 | L8 | L11 | What it controls for |
|---|---|---|---|---|
| Persona probe, grouped CV (within template) | 0.63 | 0.81 | 0.84 | — |
| Neutral one-token swap decodable (K4; "tall/short", "red/blue") | 0.61 | 0.62 | 0.75 | **word identity alone** |
| Persona probe applied to neutral swaps (N1 / N2) | 0.47 / 0.52 | 0.40 / 0.41 | 0.47 / 0.41 | whether the persona direction is a generic "word changed" axis |
| Leave-one-persona-pair-out (K5, cross-template) | 0.53 | 0.70 [0.69, 0.72] | 0.70 [0.69, 0.71] | transfer to unseen persona words and frames |
| Held-out templates T5 / T6 (never fitted) | 0.57 / 0.52 | 0.73 / 0.67 | 0.64 / 0.62 | transfer to unseen wording |

**Reading:**
* Any one-token swap is decodable at the final token: 0.62 at L8 and 0.75 at L11. Within-template persona AUROC
  (0.81–0.84) must therefore be read against that floor, and it exceeds it by only 0.09–0.19.
* The persona direction does **not** fire on neutral swaps (0.40–0.52, at or below chance). So it is not simply an
  "a word changed" axis.
* A direction shared across different persona words and frames exists: LOPO 0.70 at L8/L11, with CI low ≥ 0.69,
  well above the neutral transfer floor of 0.53–0.54.
* Transfer to held-out wording is weaker (0.62–0.73).
* Behaviourally, the persona swap changes the next-token distribution about 2× more than a same-meaning synonym swap
  (K7: median KL(H‖U) / KL(H‖Hp) = 1.99 train, 2.12 validation).

**Defensible claim:** the probe reads a persona-word-class component that is shared across honesty/dishonesty
wordings, is behaviourally relevant, and is distinct from generic word identity. It is a weak signal (AUROC ≈ 0.70
across wordings). It is **not** established that it is an "honesty concept". The paper should say "persona
(honest-vs-dishonest instruction) representation", not "honesty representation".

**Remaining:** HT4 in the frozen test protocol checks transfer to new frames and new words (T7/T8) against new neutral
controls (N3/N4). It has not been run.

## 2. Causal validity (S7)

* **Denominator.** T_full, the persona effect restored by patching the whole activation difference, is 0.022 nats at
  L8 and 0.043 at L11 (M4, T2+T4).
  * The ratio T_rec/T_full is stable only while T_full's CI stays well away from 0.
  * `results/shared/power_sensitivity/m4_causal_sensitivity.csv` reports, per site × architecture, T_full's CI
    relative to its mean and flags rows where the CI low is below 25 % of the mean.
* **Sensitivity.** The statement-level minimum detectable difference in T_rec/T_full is reported in the same file.
  Differences between SAE families below it (for example the R2-vs-old +0.10 at L8, VALIDATION) should be read
  together with that MDE.
* **Contrast geometry.** On RepE-LM validation, the two SAE families distort the persona contrast in opposite ways:
  * ReLU-family SAEs keep its direction best (cos(Δ, Δ̂) 0.56–0.62) but shrink it (‖Δ̂‖/‖Δ‖ 0.62–0.71);
  * TopK-family SAEs inflate it (1.09–1.46) with a worse direction (0.37–0.52).

  Neither cosine nor norm retention alone predicts T_rec. At L11, TopK has a higher EV than ReLU but carries less
  of the effect (0.62 vs 0.68).
* **Steering strength.** M4 patches use the actual activation difference (strength 1), not a tuned coefficient, so no
  strength was selected on the evaluation data.

## 3. SAE-in-the-loop equivalence (S5)

See `docs/RESEARCH_DECISIONS.md` D33 and the paper's "What these patches are, mathematically" paragraph.

* **The argument.** M4 keeps U's error term, so its patch is x_U + (x̂_H − x̂_U) = x_U + W_dec(z_H − z_U). That is
  the raw activation plus a decoder-space vector computed from both inputs.
* **What it establishes.** The SAE **selects** a direction that carries 62–75 % of the effect at L8/L11 (VALIDATION),
  more than matched random feature sets and isotropic vectors.
* **What it does not establish.** Mediation through SAE computation.
* **Distinct intervention proposed** (not run): edit z at layer ℓ, then splice SAEs at layers ℓ+1…ℓ+k, so that the
  edit's downstream effect must pass through their encoders' nonlinearities. Compare it with the same edit applied
  without downstream SAEs.

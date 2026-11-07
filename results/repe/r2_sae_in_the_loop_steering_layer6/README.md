# R2. SAE-in-the-loop causal intervention on honesty features, GPT-2 Small layer 6 (2026-11-07). Status: PLANNED (awaiting approval)

Evidence level (planned): causal intervention, with selection on validation and one pre-registered held-out evaluation.
Roadmap: C3 (SAE-in-the-loop), C5, C6. Audit, splits and leakage handling are shared with R1 (`results/repe/r1_honesty_preservation_layer6/README.md` §0).

## 0. What existed before, and why it is not SAE-in-the-loop

* **Scripts 10–12:** GPT-2 was steered with a direction computed from SAE reconstructions; no SAE was in the forward pass. SUPERSEDED.
* **Script 13 (6c):** "SAE features" meant adding a decoder column at a fixed norm to the raw activation. The encoder never ran.
* **Important equivalence:** an *additive* edit z_f ← z_f + δ, decoded with the error term kept, gives x' = x + δ·W_dec[:, f]. That is identical to direction steering.
  * A genuine SAE-in-the-loop test therefore needs an **input-dependent** edit: clamping, interpolating towards a target value, or ablating.
  * In each of these the change depends on the encoder's z(x) at run time.
  * R2 uses only such edits and reports how often the selected features are active (z > 0) on the held-out prompts. For an inactive feature, a clamp reduces to a direction addition.

## 1. Question and hypotheses

* **Question:** do honesty-selected SAE features, edited inside the forward pass, change GPT-2's next-token output on held-out prompts in the honest direction, beyond random-feature, matched-norm and unrelated-feature controls, without general damage?
* **Expected-direction definition:**
  * Untruthful-persona prompts with the selected features moved towards their **honest-persona train mean** should move GPT-2's output towards the honest-persona output for the same statement (E > 0).
  * The reverse edit (α < 0) should move it away (E < 0).
* **H-R2 (primary):** at α = 1, E_target exceeds the 95th percentile of random-feature and matched-norm controls on held-out prompts, with sign reversal at α = −1, and no more OWT damage than random features.
* **Prior expectation (disclosed):** the persona effect on GPT-2 outputs is small (baseline KL(p_H‖p_U) ≈ 0.02 nats; a full final-token activation patch recovers E = 0.23; the raw direction gave 0.005). A null result is plausible and will be reported as such.

## 2. Intervention (per SAE)

* **Site:** `blocks.6.hook_resid_pre`, final prompt token, in the live forward pass (TransformerLens hook).
* At run time:
  1. x → z = SAE.encode(x);
  2. ε = x − SAE.decode(z);
  3. for the selected set F: z'_F = z_F + α·(μ^target_F − z_F); other latents unchanged;
  4. x' = SAE.decode(z') + ε, which replaces x.
* **Primary variant: error term kept.** At α = 0, x' = x exactly; this is checked to fp32 tolerance.
* **Secondary variant: no error term** (x' = SAE.decode(z')). At α = 0 this equals the reconstruction splice.
* TopK: the edit is applied after the top-k selection, with no re-selection.
* **Strengths:** α ∈ {−1, −0.5, 0, 0.5, 1, 1.5, 2}. α = 1 (clamp to the target-persona mean) and α = −1 are primary, fixed in advance; the other strengths are descriptive.
* **Directions:**
  * U→H: untruthful prompts, target = honest-persona mean (primary);
  * H→U: honest prompts, target = untruthful mean (secondary).
* **Feature selection: train split only** (all 417 train statements; validation and test unread).
  * Score: paired effect size d_f = mean(z_H − z_U) / sd(z_H − z_U) over train pairs.
  * Candidates: features active in ≥ 5 % of train RepE rows. Take the top-|d| set.
  * Set size k ∈ {1, 4, 16}. **k is chosen per architecture on validation** (maximum mean E at α = 1 across seeds), then frozen.
  * μ^H and μ^U are train means.

## 3. Controls (all on the same prompts, same hook)

| Control | Purpose |
|---|---|
| Original GPT-2 (no hook) | reference outputs |
| α = 0 with error term | must equal the original exactly (hook sanity) |
| Reconstruction splice (no edit, no error) | cost of the SAE itself |
| **200 random alive-feature sets** of size k, same clamp procedure (to their own honest means) | is it *these* features? |
| **200 matched-norm random directions** (per-prompt ‖Δx‖ equal to the target edit) | is it the specific subspace, or any perturbation of this size? |
| **Unrelated-feature control:** top-k features for factual label (true vs false, within persona, train), clamped to their true-statement mean | semantically meaningful but non-honesty features |
| Raw mass-mean direction, norm-matched to the target edit (non-SAE reference) | compare with direction steering (6c) |
| Positive control: final-token activation patch x_H → U prompt | ceiling on E at this site |
| **General-damage test:** the same α = 1 edit applied at all non-BOS positions of 200 OWT validation sequences (128 tokens): ΔCE, KL, top-1 vs the random-feature controls | rules out "effect = damage" |
| Specificity: the U→H edit applied to *honest* prompts (should change little) | the edit is persona-specific, not generic |

## 4. Outcomes

* **Primary (causal, output-level):** E = 1 − Σ KL(p_H ‖ p_U') / Σ KL(p_H ‖ p_U), next-token distribution at the final position (the same definition as 6c). Higher is better for α > 0; negative is expected for α < 0.
* **Secondary:**
  * S = Δ log p(next statement token | true) − Δ log p(… | false), expected ≈ 0 for GPT-2;
  * top-1 agreement with the honest-persona output;
  * entropy change;
  * the H→U version of E;
  * the no-error-term variant;
  * feature activity rate on held-out prompts;
  * ‖Δx‖.
* **Damage:**
  * OWT ΔCE (lower is better);
  * the KL of the U→H edit on honest prompts (lower is better).
* Probe-score changes are **not** a causal outcome. They are reported only as a manipulation check.

## 5. Evaluation sets and selection discipline

| Set | Use |
|---|---|
| Train | feature selection, target means |
| Validation (89 statements; 326 excluded) | choose k per architecture; check sanity of strengths |
| Cross-wording validation | wording check during development (no selection) |
| **Test, default wording (90)** | pre-registered held-out evaluation, run once |
| **Test, new held-out wording (90)** (shared with R1; needs approval) | pre-registered wording generalisation, run once |

SAEs, layer, site, α and the feature-scoring rule are fixed here. Only k is chosen, and only on validation.

## 6. SAEs

* **Primary:** the 9 base SAEs (3 architectures × seeds 42/1042/2042).
* **Secondary (exploratory, only if R1 completes):** the R-recon and R-honesty SAEs from R1, with the same protocol and 200 random-feature controls each. The question is whether better-preserved representations give stronger or more specific causal effects.

## 7. Pre-registered criteria (held-out test, default wording)

**Per base SAE:**
* one-sided empirical p = (1 + #controls ≥ target) / 201, against random-feature sets (C2) and matched-norm directions (C3) separately;
* Benjamini–Hochberg across the 9 base SAEs × 2 control families;
* bootstrap 95 % CI over statements (2000 resamples).

**Per architecture, H-R2 is SUPPORTED only if all of the following hold:**
1. **Effect:** E(α = 1) has BH q < 0.05 vs both C2 and C3 on ≥ 2/3 seeds.
2. **Sign reversal:** E(α = −1) < 0 with its CI excluding 0 on ≥ 2/3 seeds.
3. **Specificity:** E_target − E_unrelated > 0, with a bootstrap CI excluding 0, on ≥ 2/3 seeds.
4. **No damage confound:** the OWT ΔCE of the target edit is ≤ the 95th percentile of the random-feature ΔCE on every seed.
5. **Wording generalisation:** on the new-wording test set, E(α = 1) is above the C2 95th percentile on ≥ 2/3 seeds.

**Other verdicts:**
* **NOT SUPPORTED:** criterion 1 fails.
* **INCONCLUSIVE:** criterion 1 passes but any other criterion fails.

E is also reported as a fraction of the positive control.

## 8. Smoke test (before the main run)

The smoke test uses 1 SAE per architecture, seed 42, 10 validation statements and 5 controls. It must show:
1. The hook fires once per forward pass at the intended position, and other positions are bit-unchanged.
2. α = 0 with the error term ≡ the original (max |Δlogit| < 1e-4), and the splice matches the existing Phase-0 splice numbers.
3. After the edit, re-encoding x' shows z_F moved as intended, with the other latents' change reported.
4. All metrics are finite, and identity runs give E = 0.
5. Splits and checkpoint hashes match the pre-registration.
6. Peak GPU memory is below 10 GB.

## 9. Compute estimate

* Per base SAE: about 3 k × 7 α × 2 directions + 400 controls + unrelated/raw/positive controls on 90–180 prompts (short sequences, batched), plus about 200 OWT damage passes at 200 × 128 tokens.
* 6c ran 27 SAEs × 600 controls in about 9 min.
* **Estimate:**
  * 9 base SAEs: about 45–60 min (validation + test + new wording);
  * optional fine-tuned SAEs (18): about 45–60 min more;
  * smoke test: about 5 min.
* GPU below 8 GB.

## 10. Files (planned)

* **Scripts:**
  * `scripts/repe/17_r2_sae_in_the_loop_steering.py` (`--smoke`, `--select-k`, `--test`);
  * `18_r2_analysis.py`;
  * figures in `19_r1_r2_figures.py`.
* **Results:** this folder (`preregistration.json`, `smoke/`, `validation_k_selection.csv`, `per_sae_test.csv`, `controls_*.csv`, `criteria.json`).
* Outputs refuse to overwrite.

## 11. After execution

To be completed.

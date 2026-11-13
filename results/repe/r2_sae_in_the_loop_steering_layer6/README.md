# R2. SAE-in-the-loop causal intervention on honesty features, GPT-2 Small layer 6 (2026-11-07). Status: VALIDATION STAGE COMPLETE: NOT SUPPORTED (descriptive, validation); test stage DEFERRED (title status updated 2026-11-08; §0–12 otherwise unchanged)

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

## 12. Pre-run addendum (2026-11-08, written before the validation stage)

§0–§11 above are the plan as drafted and committed in `d17bb00`; they are not edited. **The §7 test criteria, the
strength grid, the k grid, the feature-scoring rule, the SAEs and the site are unchanged.**

**Stage decision:** the user chose "validation now, test later" on 2026-11-08.
* This run covers §5's validation uses only: choose k, sanity-check the strengths, and the cross-wording check.
* It also computes every control on validation so the evidence can be read now. Those numbers are **VALIDATION, descriptive**.
* No test row is read. `17_r2_sae_in_the_loop_steering.py --test` refuses to run.
* The new-wording test set stays deferred, shared with R1.

**Corrections:**
1. **OWT damage set:** the OWT validation cache holds 100 sequences of 128 tokens, not 200. All 100 are used (≈ 12.6 k predictions).
2. **Criterion 5 at this stage:** run on the cross-wording **validation** set (89 statements), descriptively. The test version is unchanged.

**Details fixed before the run:**
* **Control strength:** controls are run at α = 1, U→H, with the error term kept; criteria 1 and 4 compare α = 1 with controls. α = −1 is judged by its bootstrap CI (criterion 2).
* **C2 (random features):** 200 sets of k features drawn from the candidate pool (train-active ≥ 5 %), excluding the target set. Each set is clamped to its own honest-persona train means.
* **C3 (matched-norm directions):** 200 Gaussian unit directions; each prompt's ‖Δx‖ equals that prompt's ‖Δx‖ under the target edit.
* **Unrelated features:** top-k by factual-label Cohen's d (true vs false), computed within each persona and averaged, using train rows only.
  * The top-16 honesty features are excluded.
  * Clamped to the train mean over true-statement rows, at α = 1.
* **Raw direction:** the train mass-mean persona direction, norm-matched per prompt to the target edit.
* **Specificity (criterion 3):** E_target − E_unrelated, with a paired statement bootstrap.
  * Also reported: KL(p_H‖p_H′) when the U→H edit is applied to honest prompts, as a fraction of the persona KL.
* **Probe manipulation check** (not causal evidence): fraction of the U→H gap in probe score closed by the edit, for two train-only probes. These are the mass-mean projection and a logistic probe (C = 1, fixed, fitted on all train rows).
* **Representation-level checks:**
  * ‖Δx‖;
  * how often the selected features are active on the evaluated prompts;
  * re-encoding attainment, (z″_F − z_F)/(z′_F − z_F);
  * relative change of the other latents.
* **Statistics:**
  * empirical p = (1 + #controls ≥ target)/201;
  * Benjamini–Hochberg across the 9 base SAEs × 2 control families, per k;
  * bootstrap with 2000 statement resamples;
  * control seed 20261108 plus fixed per-SAE offsets.
* **k:** chosen per architecture as the k that maximises the mean E(α = 1) over seeds on validation, then frozen in `k_selection.json` for the test stage.
  * The criteria are reported on validation for **every** k.
  * Numbers for the selected k are optimistic, because k was selected on the same data.
* **Secondary SAEs (§6):** R1 has completed, so the R1 R-recon and R-honesty SAEs (18) are included as EXPLORATORY. They are run on validation only, with targets and the 200 C2 controls.

**Smoke and pipeline tests before freezing (disclosed):**
* **`--smoke`** (`smoke/<time>/smoke_checks.json`), seed-42 SAEs of all three architectures. All §8 checks pass:
  * cached = fresh activations;
  * positions before the final token are bit-identical, and the final position changes;
  * α = 0 with the error term reproduces the original exactly (max |Δlog p| = 0, E = 0);
  * the no-error α = 0 splice equals the shared splicer to 4e-6;
  * re-encoding brings z_F close to its target (TopK: mean |error| 0.12, from 1.86 before the edit);
  * selected features are active on 72–95 % of prompts;
  * peak GPU 3.2 GB.
* **Pipeline test** (`smoke/pipeline_*/`): TopK seeds 42/1042/2042 plus one exploratory SAE, k ∈ {1, 4}, 5 controls, 25 OWT sequences.
  * **These validation outcomes were seen before freezing:** E(α = 1) was negative (−0.06 to −0.19), as was E(α = −1).
  * Nothing in the protocol was changed in response.

## 13. Validation-stage results (2026-11-08). Evidence label: VALIDATION FINDING (descriptive)

**Scope and labelling.**
* Run: `17_r2_sae_in_the_loop_steering.py --validation` (script SHA `0a0f4fc7…`, as frozen in `b232ce4`); exit 0; runtime 75 min; peak GPU 5.6 GB.
* `test_rows_read: false`. The test stage was **not** run and the script refuses `--test`.
* Sets:
  * `val`: 89 statements (statement 326 excluded);
  * `cw`: the 89 matching cross-wording validation statements;
  * OWT: 100 validation sequences of 128 tokens (§12 correction).
* Baseline mean KL(p_H‖p_U) on `val` is **0.020 nats**: the persona's effect on next-token output at the final prompt token is small.
* **Positive control** (activation patch of the honest residual at layer 6, final token): E = **0.188** on `val`, 0.404 on `cw`. Even a full patch at this site recovers only about 19 % of the default-wording output shift.
* All criteria below are §7 applied to *validation*. They are descriptive, not the pre-registered test.
* Numbers for the selected k are optimistic, because k was chosen on the same rows.

**k selection (`k_selection.json`).** k = 1 for every architecture. It is the argmax of mean E(α = 1), i.e. the *least negative* value:

| arch | k = 1 | k = 4 | k = 16 |
|---|---|---|---|
| ReLU | −0.110 | −0.127 | −0.178 |
| TopK | −0.103 | −0.164 | −0.532 |
| JumpReLU | −0.115 | −0.287 | −0.854 |

**Descriptive criteria (`validation_criteria_descriptive.json`).** The pattern is identical for all 3 architectures × 3 k:

| criterion | result | comment |
|---|---|---|
| 1 Effect (BH q < 0.05 vs C2 and C3, ≥ 2/3 seeds) | **FAIL** everywhere | Target E is *below* the C2 mean. Empirical p ≈ 0.68–1.0; BH q = 1.0. |
| 2 Sign reversal (E(α = −1) < 0, CI excludes 0) | passes, **uninformative** | E(α = +1) is also < 0. Both directions move outputs away from p_H, so there is no reversal of a positive effect. |
| 3 Specificity (E_target − E_unrelated > 0) | FAIL | At k = 1 the difference is about −0.1, with a CI that excludes 0 *below* zero. |
| 4 No damage confound (OWT ΔCE ≤ C2 q95, every seed) | FAIL | Per seed, target ΔCE is 0.015–0.21 nats against C2 q95 of 0.0006–0.04. |
| 5 Wording (cw E > C2 q95) | FAIL | cw E(α = 1) ranges from −0.04 to −0.28. |
| **Verdict** | **NOT SUPPORTED** (all architectures, all k) | |

**Base SAEs, `val`, mean ± sd over 3 seeds, k = 1 (selected):**

| arch | E(α=1) | E(α=−1) | C2 mean / q95 | C3 q95 | E unrelated | E raw dir (norm-matched) | E recon splice (no error) | OWT ΔCE target / C2 q95 |
|---|---|---|---|---|---|---|---|---|
| ReLU | −0.110 ± 0.006 | −0.058 ± 0.012 | −0.012 / 0.004 | 0.009 | −0.005 | −0.040 | −4.27 | 0.017 / 0.001 |
| TopK | −0.103 ± 0.039 | −0.070 ± 0.028 | −0.027 / 0.004 | 0.005 | −0.062 | −0.047 | −3.75 | 0.029 / 0.002 |
| JumpReLU | −0.115 ± 0.031 | −0.057 ± 0.015 | −0.045 / 0.003 | 0.005 | −0.033 | −0.025 | −2.83 | 0.043 / 0.005 |

Full per-SAE, per-k and per-set values are in `validation_summary.csv`. Figure: `results/repe/figures/r2_sae_in_the_loop_validation.png`.

**The three categories, kept separate (CLAUDE.md §2):**
1. **Representation-level effect: present.** The edit does what it is built to do in latent space:
   * re-encoding attainment is 0.77–0.85 (ReLU) and 0.96–0.99 (TopK/JumpReLU) at k = 1;
   * the selected features are active on 82–100 % of prompts;
   * ‖Δx‖ is about 1.1–2.1 at k = 1 and 4–11 at k = 16.
2. **Probe-score change: present on `val`, reversed on `cw`.** This is a manipulation check, not causal evidence.
   * On `val`, the edit closes part of the U→H gap in score for the independent train-only probes:
     * logistic probe: 0.15 (ReLU), 0.31 (TopK), 0.44 (JumpReLU) at k = 1, rising to 0.36–0.64 at k = 16;
     * mass-mean probe: 0.08–0.21 at k = 1.
   * On `cw` the same edit moves the logistic score the *other* way (−0.76 to −1.05 of the cw U→H gap).
   * This was not investigated further. A likely contributor is that μ_F is estimated on default-wording train prompts.
3. **Behavioural effect (E, output KL): absent, and if anything adverse.**
   * Clamping the honesty-selected features of the unhonest prompt to the honest-train mean moves the next-token distribution *further* from the honest prompt's distribution, for every architecture, k and seed (one exception: R1 r_honesty ReLU seed 1042, k = 1, E = +0.007, CI includes 0).
   * Random-feature sets (C2) and matched-norm random directions (C3) do much less harm.
   * The norm-matched raw H−U mean-difference direction is also mildly harmful. A population-mean shift at this site therefore does not reproduce the statement-specific output difference, which only the full activation patch recovers (and then only 19 %).
   * The target features are also more damaging on OWT than random features (criterion 4). They are high-impact, frequently active features, not persona-specific ones.

**Reconstruction splice.** Replacing x by the SAE reconstruction without the error term gives E between −2.8 and −4.3. Most of the behaviourally relevant H/U difference at this site sits in the SAE error term or is disrupted by reconstruction. This is why the error-kept edit is primary (§2). It is a behavioural-faithfulness observation, not an intervention result.

**R1 SAEs (EXPLORATORY; targets plus C2 only).** For R-honesty and R-recon, E(α = 1) is negative with the same pattern as the base SAEs.
* Paired r_honesty − r_recon difference in E(α = 1) at k = 1: ReLU +0.028 ± 0.060, TopK +0.002 ± 0.003, JumpReLU −0.001 ± 0.001.
* At k = 16, TopK shows +0.13 ± 0.11. It is not consistent across seeds and the absolute E stays strongly negative.
* R1 honesty-aware fine-tuning did not make the SAE features causally effective under this intervention.

**Conclusion.**
* We find **no evidence** that clamping SAE honesty features at GPT-2 Small layer 6 causally moves model outputs towards the honest-persona behaviour.
* The pre-registered effect criterion fails by a wide margin on validation for all three architectures and all k. This holds even for the k chosen in hindsight.
* **No causal SAE-feature faithfulness claim is made.**
* The representation-level and probe-score changes show the edit is well formed. They do not establish behavioural relevance.

**Limitations.**
* The behavioural signal is small (mean KL 0.020 nats) and only about 19 % is recoverable even by a full activation patch at this site and position. The site may be a weak place to test persona causality.
* The intervention is a population-mean clamp. A per-statement edit could behave differently, but would be a different, new pre-registration.
* E is a ratio of summed KLs and is unbounded below.
* The single test-stage evaluation is still outstanding and needs the user's approval. Given the validation margin, it is unlikely to change criterion 1. Whether to spend it is the user's decision.
* The `cw` probe-score reversal is unexplained.

**Files added in this stage:**
* `validation_targets.csv`, `validation_controls.csv` (4.6 MB), `feature_selection.csv`, `validation_summary.csv`, `k_selection.json`, `validation_criteria_descriptive.json`, `validation_run_info.json`;
* figure `results/repe/figures/r2_sae_in_the_loop_validation.png` (from `19_r1_r2_figures.py --r2`, added in this commit).

The log is in `logs/validation.log` and is gitignored.

**Next steps (none run):**
1. R2 test stage with k = 1 frozen. Needs approval.
2. Optionally, a separately pre-registered per-statement or alternative-site variant, e.g. a later layer where the positive control is stronger.
3. Investigate the `cw` probe-sign reversal.

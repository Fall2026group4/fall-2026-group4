# Geometry of Truth causal test, cities (roadmap C4) (2026-11-10). Status: COMPLETED

Evidence level: confirmed test result (pre-registered; the test pairs were scored once).

Script: `scripts/geometry_of_truth/22_truth_direction_causal_test.py`. Run on Kaggle (Tesla T4), with the 108 in-house SAEs rebuilt from `scripts/openwebtext/13_consistent_protocol_retrain.py` (see `results/openwebtext/rebuild_reproducibility_check/`).

## 1. Design

* **Question:** do the truth directions that probes read from GPT-2 small causally change which answer the model prefers on the cities task?
* **Outcome:** for each true/false statement pair, the change in log P(correct token) − log P(wrong token) at the first token where the two statements differ. The intervention adds a vector at the final prefix token.
* **Layers:** 4, 6, 8, 10 (`blocks.L.hook_resid_pre`).
* **Targets:**
  * C1: mass-mean and logistic-probe directions.
  * C2: SAE reconstruction, top-1 and top-10 latent directions, for 3 architectures × 3 seeds, both signs.
* **Positive control (A):** a graded oracle push along the unembedding contrast, used to measure the smallest detectable dose.
* **Controls:** 600 norm-matched random directions per layer. Empirical p = (1 + #controls ≥ effect)/(n + 1). BH correction within each of the three pre-registered families (A, C1, C2).
* **Strength:** chosen on validation pairs only (`validation_strength_selection.csv`) and fixed in `preregistration.json` before the test phase.

## 2. Results (test, 113 pairs, 0 dropped)

* **Baseline:** GPT-2 prefers the correct answer in 74.3 % of pairs.
* **Pipeline checks** (`pipeline_checks.csv`): the zero delta changes nothing; the fast path equals the full forward pass; the final-token recomputation matches TransformerLens's hooked forward pass to within 2.1e-5.
* **Positive control:** detected at a dose of 0.01 at every layer (p = 1/601).
* **C1, mass-mean and probe directions:** not distinguishable from random directions at any layer. The best one-sided p is 0.19, and every BH-adjusted p is ≥ 0.93. The probe direction tends to push the margin the wrong way (effect −0.13 to −0.35 at layers 6 to 10, all non-significant).
* **C2, SAE directions:** 0 of 108 survive BH (smallest adjusted p 0.60). 6 are nominal at two-sided p < 0.05, about what chance gives, and the seeds disagree in sign.
* Figure: `results/shared/followup_figures/F6_got_causal_steering.png`.

## 3. Criteria and interpretation

* The test is well powered: a 1 % oracle dose is detectable. Within that sensitivity, **the truth directions that probes read do not causally drive GPT-2's factual answers on this task**, and the SAE directions don't either.
* **Limitations:**
  * The directions were estimated at the end of the statement and applied at the final prefix token.
  * At layers 8 and 10 the selected strength (8×) is at the top of the validation grid.
  * One task (cities) and one model.

## 4. Files

`preregistration.json`, `validation_strength_selection.csv`, `summary.csv` (one row per target, with p-values and BH), `interventions.csv` (every intervention), `per_pair_effects.npz`, `pipeline_checks.csv`, `test_run_info.json`.

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

## 5. Integration note (2026-11-26; added on integration, nothing above changed)

Selectively extracted from branch `dhruv-causal` (Dhruv Rai, commits `d9e14d2`..`989d313`, 2026-11-10; PR #26 was
closed unmerged) as decided in D56. Verified before integration:

* **Pre-registration:** the script, `data/got/cities.csv` and `data/got/splits/cities_splits.csv` SHA-256 equal the
  values frozen in `preregistration.json`. TEST was written once, 7 minutes after the pre-registration
  (`test_run_info.json`, 113 pairs, 600 controls).
* **Statistics recomputed from `summary.csv`:** BH within each family reproduces `p_bh` (A and C1 from `p_upper`, C2
  from `p_two_sided`; max |diff| ≤ 1e-15). Positive control p = 1/601 at every layer; C1 best one-sided p 0.188,
  every BH p ≥ 0.932; C2 0/108 with BH p < 0.05 (smallest 0.599), 6 nominal two-sided p < 0.05.
* **TEST exposure:** the GoT cities TEST rows had been scored before, for frozen-probe AUROC
  (`all_layer_frozen_probe_evaluation/`, `final_test_evaluation/`, 2026-11-07). The causal outcome was not computed
  on them before this single run.
* **Checkpoint provenance:** family C2 used the 108 SAEs rebuilt on Kaggle. Their SHA-256 values (in
  `preregistration.json`) match **0 of 36** of the original checkpoints, as expected for a retrain; the rebuild matches
  the original test metrics (EV within 0.0013, L0 within 1.19; `results/openwebtext/rebuild_reproducibility_check/`).
  The positive control (A) and the raw truth directions (C1) use no SAE, so the C1 null does not depend on this.
* **Code dependency:** `scripts/repe/15_sae_in_the_loop_causal.py` is included only because this script imports its
  statistics and latent helpers. Its own experiment (old RepE prompts) is not integrated (D23 confound).
* **File name:** `22_` is kept although `scripts/geometry_of_truth/` already has two `22_` scripts, because the
  pre-registration records the path and the script checks its own hash before the test phase.
* **Tests:** `src/tests/test_got_causal_c4.py` (C4 and helper tests, verbatim) and `src/tests/test_followup_part3.py`.

# SAE-in-the-loop causal test, RepE honesty, layers 2 / 6 / 10 (roadmap C1 + C3) (2026-11-10). Status: COMPLETED

Evidence level: pre-registered test, but on the **old RepE persona prompts**. Those prompts are length-confounded (the untruthful persona is 3 tokens longer, and length alone gives AUROC 0.80; see Aditi's 2026-11 update). Read these results as **supporting evidence only**. The clean-prompt version is Aditi's M4.

Script: `scripts/repe/15_sae_in_the_loop_causal.py` (validation phase, then test phase). Run on Kaggle (Tesla T4), with the 27 rebuilt SAEs (3 layers × 3 architectures × 3 seeds), 600 controls per group, and 176,724 interventions.

## 1. Design

* **Outcome:** E = 1 − ΣKL(p_honest ‖ p_intervened) / ΣKL(p_honest ‖ p_untruthful). E = 1 means the intervention fully moves the untruthful-persona output to the honest one.
* **Part A, graded activation patch:** gives the minimum detectable dose.
* **Part B, latent patching with the SAE in the loop:**
  * B1: all latents patched to their honest values.
  * B2: the SAE error term.
  * B3 and B4: the top-1, top-10 and top-50 honesty latents, compared with random latents (B3) and with activation-matched random latents (B4).
* **Part C, direction steering:**
  * C1: the raw honesty direction.
  * C2: the SAE reconstruction and top-latent directions, both signs.
* BH correction within the seven families in `preregistration.json`.

## 2. Results (test)

* **Pipeline checks:** the final-token recomputation matches the hooked forward pass to within 2e-6. The latent decomposition is exact to within 4e-5. Cached and fresh activations agree to within 9e-5.
* **Positive control:** a 1 % dose is detectable at every layer.
* **B1, all latents:**
  * At layer 6 they carry 41–42 % of the full activation patch for ReLU and JumpReLU, and 7 % for TopK.
  * At layer 10 they carry 46–68 %.
  * Significant in every seed at layers 6 and 10. This agrees with Aditi's M4 (0.62–0.68 on the clean prompts).
* **B2, error term:** significant at layers 6 and 10.
* **B4, top-50 latents vs activation-matched latents:** at layer 10 the top-50 latents beat the activation-matched controls in 8 of 9 SAEs, carrying 16–22 % of the patch. At layers 2 and 6, all 52 significant comparisons go the wrong way: the selected latents move the output **away** from honest more than the matched controls do. This replicates the wrong-way pattern found in `results/repe/steering/causal_reanalysis/`.
* **C1, raw honesty direction:** at layer 6, E = 0.0048 (p = 0.005). That is exactly the value in the 2026-11-04 causal test.
* **C2:** at layer 10, the reconstruction directions beat the controls in every seed for TopK and JumpReLU.
* Figures: `results/shared/followup_figures/F1_graded_positive_control.png`, `F2_latent_mediation.png`, `F3_repe_direction_steering.png`.

## 3. Interpretation and limitations

* The SAE latents together carry a real part of the honesty effect, and more at later layers. The single "most honest" latents chosen by activation difference are not the causal ones at early layers.
* The prompt confound limits all of this. Use Aditi's M4 for claims about honesty, and this run as an independent agreement.

## 4. Files

`preregistration.json`, `validation_strength_selection.csv`, `graded_positive_control.csv`, `summary.csv`, `interventions.csv`, `per_statement_kl_targets.npz`, `pipeline_checks.csv`, `test_run_info.json`, and `smoke/` (the smoke run that checked the pipeline before the full run).

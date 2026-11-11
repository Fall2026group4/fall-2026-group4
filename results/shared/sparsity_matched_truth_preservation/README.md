# Sparsity-matched truth and honesty preservation (roadmap S3) (2026-11-10). Status: COMPLETED

Evidence level: validation finding (validation rows only).

Script: `scripts/shared/sparsity_matched_truth_preservation.py --dataset got|repe`. Uses the 108 rebuilt in-house SAEs.

## 1. Design

* **Question:** the architectures use different numbers of active latents on the truth prompts. When they use the same number, does any architecture still preserve the truth readout better?
* **Method:** at inference time, keep only the top m latents (m = 8, 16, 24, 32, 48, 64, 96, 128), then score the frozen original probes on the reconstruction.
* **Matching rule:** a cell is compared only if every architecture and seed is within ±10 % of m on the prompts.
* **Comparison:** paired differences over the three seeds.

## 2. Results

* **Geometry of Truth:**
  * At matched sparsity, architecture differences are small (median |difference| 0.012–0.023 AUROC). They are also inconsistent: the seeds disagree in 30–39 of 43–52 matched cells.
  * TopK's apparent advantage came from using more latents. At layer 6 its gap is 0.280 at its natural 128 latents but 0.333 at 48, the same as ReLU's 0.332.
  * The truth gap barely depends on sparsity: about 0.31–0.35 at layer 6 from 8 to 128 latents.
* **RepE (old, length-confounded prompts):** TopK's gap is about 0.08 lower at equal sparsity, with all seeds agreeing in 62 of 79 cells against JumpReLU and 56 of 79 against ReLU.
* Figure: `results/shared/followup_figures/F5_truth_gap_vs_prompt_l0.png`.

## 3. Interpretation

* **Truth task:** "TopK preserves truth best" does not hold once sparsity is equal. The architecture ranking in `results/geometry_of_truth/all_layer_frozen_probe_evaluation/` should be read with this in mind.
* **Honesty task:** a TopK advantage remains, but the prompts are confounded, so it is not a claim.

## 4. Files

Each of `got/` and `repe/` has `per_run.csv`, `matched_comparison.csv`, `natural_l0_summary.csv`, `protocol.json` and `run_info.json`.

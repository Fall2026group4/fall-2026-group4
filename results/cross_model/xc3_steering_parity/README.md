# XC-3: steering parity, GPT-2 Small vs Gemma-2-2B (2026-11-23). Status: PRE-REGISTERED, RUNNING

Evidence level: VALIDATION phase = validation finding; TEST phase = confirmed test result for H1–H3 (H4 descriptive).

Frozen protocol and hashes: [`preregistration.json`](preregistration.json).
Script: [`scripts/cross_model/20_xc3_steering_parity.py`](../../../scripts/cross_model/20_xc3_steering_parity.py).

## 1. Before implementation

* **Research question.** Does an SAE-derived honesty direction keep the causal steering effect of the
  original-activation honesty direction? Is the answer the same for GPT-2 Small and Gemma-2-2B under one protocol?
* **Why.** The earlier steering results can't be compared across models (audit 2026-11-21, row 14):
  * GPT-2 used RepE at L6 with 200 controls;
  * Gemma used GoT judgment at L11 with 20 controls and no SAE.

  XC-3 uses one task, one outcome, depth-matched sites and ≥ 200 controls for both models. It also adds the
  reconstruction-direction arm, which asks whether the steering behaviour survives the SAE. This is a causal-category
  question; it is not reconstruction quality.
* **Design.**
  * **Task:** RepE-LM templates T1–T4, honest (H) vs untruthful (U) persona prompts. These are one-token swaps;
    K1 is re-checked under each tokenizer.
  * **Sites:** triplet T-B, at the final prompt token only.
    * GPT-2 `blocks.L.hook_resid_pre`, L = 3/6/9.
    * Gemma `resid_post`, L = 5/12/19.
* **Outcome.** E(v) = 1 − Σ KL(p_H‖p_{U+v}) / Σ KL(p_H‖p_U), where p is the next-token distribution at the final
  token.
  * Higher is better: E = 1 means the honest-persona output is fully recovered.
  * It is normalised within each model.
* **Arms.**
  * **raw:** the mean H − U difference of the original activations (TRAIN).
  * **rec:** the same difference of SAE reconstructions, using res-jb for GPT-2 and Gemma Scope 16k canonical for
    Gemma.
  * **patch:** a positive control, setting x_U ← x_H at the site.
* **Strength.** s = c · ‖Δ_TRAIN‖.
  * c is chosen on VALIDATION as argmax E_raw over {0.05 … 8}, the same grid as `scripts/repe/13`.
  * The same s is used for rec and for every control.
* **Controls.**
  * 200 Gaussian unit directions.
  * 200 random alive SAE decoder columns, each with a random sign. These are used on TEST only.
* **Gates.**
  * G1: the CI of E_patch on VALIDATION has a lower bound > 0. Cells that fail are not eligible.
  * Zero-edit identity.
  * Gemma: the final-token logit path must equal the full forward pass.
* **Hypotheses (TEST, Holm over 6 cells in each family).**

  | Hypothesis | Comparison | What it establishes |
  |---|---|---|
  | H1 | raw vs Gaussian and vs SAE-feature controls | the honesty direction steers specifically |
  | H2 | rec vs Gaussian controls | the SAE-derived direction steers |
  | H3 | R = E_rec / E_raw, only where H1 holds. PRESERVED if CI low ≥ 0.8; DEGRADED if CI high < 0.8; otherwise INCONCLUSIVE | causal steering is preserved through the SAE |
  | H4 | GPT-2 vs Gemma at matched depth (descriptive) | parity |

* **Leakage.**
  * Directions and latents come from TRAIN only, and strength from VALIDATION only.
  * TEST is read once.
  * **Disclosure:** GPT-2 RepE-LM TEST rows were read once before, on 2026-11-13, by H1-TEST (HT1–HT4). That pass
    evaluated none of XC-3's directions or strengths, but the rows are not pristine for GPT-2. Gemma has never seen
    them.
* **Smoke tests.**
  * Smoke tests used TRAIN rows only and are kept in `smoke/` and `smoke_v2/`.
  * Two design choices were fixed after the first smoke test and before freezing:
    1. **Strength grid:** it now equals the `scripts/repe/13` grid, for parity.
    2. **H3:** it is tested only where H1 holds, because E_rec/E_raw is undefined when E_raw ≈ 0.
  * Pipeline sanity checks were also added.

## 2. After execution

(filled in after the runs)

# XC-3: steering parity, GPT-2 Small vs Gemma-2-2B (2026-11-23). Status: COMPLETED (TEST pass done once)

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

**Runs.**
* VALIDATION ran 14:41–15:01 UTC: GPT-2 201 s, Gemma 826 s.
* TEST ran once, 15:01–15:27 UTC: GPT-2 361 s, Gemma 1,368 s. A10G, torch 2.14.1, TF32 off. About 46 GPU minutes in
  total.
* Storage: 0.4 MB.
* Log: `logs_xm/xc3_full.log`.
* TEST used the frozen `validation/selection.json` (sha256 e359636b…). The pre-registration hashes were checked at
  the start of each phase.
* Output SHA-256 (first 12 characters):
  * `test/results.json` 6507a9a9fb38
  * `test/h4_parity.json` 394209e7deb1
  * `validation/results.json` eeb35e3909c3

**Integrity.**
* Zero-edit identity and the logit-path checks are exactly 0.0 for both models and every layer.
* K1 dropped 0 pairs: 356 VALIDATION and 344 TEST pairs per model, 86 TEST statements.
* Directions use 1,668 TRAIN pairs.
* Gate G1 passed at all 6 cells on VALIDATION.

**Strength (VALIDATION argmax):** c* = 4 / 4 / 2 (GPT-2 L3/6/9) and 2 / 4 / 2 (Gemma L5/12/19), × ‖Δ_TRAIN‖.

**TEST results** (E, higher is better; 95 % CIs from the statement-clustered bootstrap; 200 controls per family;
empirical p ≥ 1/201 = 0.005; Holm over 6 cells):

| Model / layer (depth) | E_patch | E_raw | E_rec | Gaussian q95 / max | SAE-feature q95 / max | H1 | H2 | R = E_rec / E_raw | H3 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-2 L3 (0.25) | 0.160 [0.119, 0.207] | 0.048 [0.031, 0.067] | 0.034 [0.004, 0.068] | −0.003 / 0.022 | −0.002 / 0.028 | **met** (Holm p 0.030 / 0.030) | **met** (0.030) | 0.69 [0.08, 1.60] | INCONCLUSIVE |
| GPT-2 L6 (0.50) | 0.288 [0.239, 0.347] | 0.068 [0.045, 0.090] | 0.023 [−0.022, 0.061] | −0.026 / −0.000 | −0.034 / −0.013 | **met** | **met** | 0.34 [−0.35, 1.00] | INCONCLUSIVE |
| GPT-2 L9 (0.75) | 0.662 [0.599, 0.728] | 0.130 [0.109, 0.156] | 0.107 [0.079, 0.135] | −0.019 / 0.007 | −0.023 / 0.027 | **met** | **met** | 0.83 [0.58, 1.11] | INCONCLUSIVE |
| Gemma L5 (0.23) | 0.103 [0.070, 0.137] | 0.002 [−0.009, 0.014] | −0.006 [−0.020, 0.008] | 0.010 / 0.017 | 0.009 / 0.012 | **not met** (Holm p 0.23 / 0.17) | **not met** (0.60) | — | not applicable |
| Gemma L12 (0.50) | 0.493 [0.448, 0.531] | 0.147 [0.090, 0.207] | 0.143 [0.086, 0.202] | −0.032 / 0.004 | −0.019 / 0.031 | **met** | **met** | 0.97 [0.52, 1.73] | INCONCLUSIVE |
| Gemma L19 (0.77) | 0.945 [0.933, 0.955] | 0.091 [0.056, 0.130] | 0.078 [0.036, 0.120] | −0.026 / 0.012 | −0.022 / −0.006 | **met** | **met** | 0.86 [0.36, 1.61] | INCONCLUSIVE |

**H4, cross-model parity (descriptive; `test/h4_parity.json`).** The bootstrap resamples the same statements for both
models.

| Depth pair | E_raw (GPT-2 / Gemma) | R (GPT-2 / Gemma) | R_gemma − R_gpt2 | E_raw_gemma − E_raw_gpt2 |
|---|---|---|---|---|
| 3 ↔ 5 (0.25 / 0.23) | 0.048 / 0.002 | — | not computed (H1 fails for Gemma L5) | — |
| 6 ↔ 12 (0.50) | 0.068 / 0.147 | 0.34 / 0.97 | +0.63 [+0.14, +1.40] | +0.079 [+0.019, +0.139] |
| 9 ↔ 19 (0.75 / 0.77) | 0.130 / 0.091 | 0.83 / 0.86 | +0.03 [−0.25, +0.28] | −0.039 [−0.083, +0.003] |

**Criteria.**
* **H1 (the honesty direction steers specifically): met at 5/6 cells.** Every GPT-2 layer and Gemma L12/L19. At each
  of these the target beats all 200 Gaussian and all 200 SAE-feature controls (minimum attainable p). It is **not met
  at Gemma L5**: E_raw 0.002, inside the control range, even though the patch control works there (E_patch 0.10).
* **H2 (the SAE-reconstruction direction steers): met at the same 5/6 cells.**
* **H3 (retention ≥ 0.8): INCONCLUSIVE at all 5 testable cells.** Every retention CI spans 0.8.
  * Point estimates are 0.69–0.97, except GPT-2 L6 at 0.34.
  * The ratio of two small, noisy effects is imprecise with 86 statements.
  * No cell is DEGRADED (CI high < 0.8) or PRESERVED (CI low ≥ 0.8).
* **H4 (descriptive).**
  * At mid depth (6 ↔ 12), Gemma's honesty steering is stronger and better retained through its SAE than GPT-2's.
    GPT-2 L6 loses about two-thirds of the effect through res-jb; Gemma L12 loses almost none through Gemma Scope.
  * At late depth (9 ↔ 19), the two models are indistinguishable.

**Interpretation (causal / steering category only; not reconstruction or representation evidence).**
* A single mean-difference honesty direction at the final token recovers only a small part of the persona effect:
  E_raw ≤ 0.15, while the full-activation patch reaches 0.16–0.95.
* Most of what it does recover survives the public SAEs in both models, but whether retention clears 0.8 is
  unresolved.
* Early layers carry little steerable persona signal at the final token in Gemma. This fits M3 (GPT-2), where the
  early persona site was the slot, not the final token.
* **TEST exposure.** The GPT-2 TEST rows had been read once before (H1-TEST, 2026-11-13); the Gemma rows were
  pristine. That pass evaluated none of XC-3's quantities.

**Limitations.**
* One task (RepE-LM T1–T4) and one site per layer (the final token).
* One public SAE per model, with different widths and L0s (res-jb 24,576 latents; Gemma Scope 16k, L0 68–82).
  Retention differences may reflect the SAE, not the model.
* 86 TEST statements, so the retention CIs are wide.
* The strength grid peaks at an interior point for every cell (c* 2–4), so the grid was not truncated.

# M4-G: SAE patch decomposition on Gemma-2-2B at the XC-1 sites (2026-11-23). Status: COMPLETED (validation finding; TEST never loaded)

Evidence level on completion: VALIDATION. TEST never loaded.

Frozen protocol and hashes: [`preregistration.json`](preregistration.json) (sha256 f4f73ca6…).
Script: [`scripts/cross_model/25_m4_gemma_patch_decomposition.py`](../../../scripts/cross_model/25_m4_gemma_patch_decomposition.py).

## 1. Before implementation
* **Question.** At XC-1's confirmed sites (plus the A1-style mid site at L12/16), is the Gemma Scope 16k
  reconstruction difference behaviourally sufficient? How much of the effect sits in the SAE error?
* **Port of M4 (`scripts/repe/24`).** Kept:
  * T_full, T_rec, T_err, I, Shapley, linearity;
  * the other-direction, synonym and neutral controls;
  * 20 isotropic and 20 decoder-shuffle draws;
  * H4a and H4b (BH over sites);
  * the additive-reading rule.
* **Declared deviations:**
  * H4c (feature patches) is omitted;
  * one SAE per site, so there is no seed rule.
* **Stop rule.** If XC-1 confirms no cell, M4-G does not run.
* **Smoke (TRAIN only).** In `smoke_v1/`, the identity check was tautological; it was fixed and re-run in `smoke/`
  (max 4.8e-7). Mechanics only; not evidence.

## 2. Run
* 20:48–20:55 UTC: 945 fixed-shape forwards, 392 s; script sha256 `ad75f03d…` matches the frozen pre-registration.
* **Sites** (`site_selection.json`, computed before any SAE was loaded):
  * L0 slot (earliest) and L23 final (peak + latest): both XC-1 cells confirmed on VALIDATION;
  * L12 final (mid, A1 port): TRAIN E_share 0.511, the closest to 0.5 among the candidates; confirmed on VALIDATION
    T2–T4 with original activations (T_full 0.080 [0.051, 0.115]; excess over synonym 0.030 [0.010, 0.054]).
* **SAEs:** Gemma Scope 16k canonical `layer_0/average_l0_105`, `layer_23/average_l0_75`, `layer_12/average_l0_82`
  (SHA-256 in `decomposition_summary.csv`).
* Pairs: 356 persona (VALIDATION T1–T4; 267 in the primary T2–T4 set) and 178 neutral (N1–N2); 0 dropped.
* **Hard sanity checks: all pass** (`sanity_checks.json`): identity max 1.9e-6 (< 1e-3); zero patch T = 0 exactly;
  Shapley sum error ≤ 2.2e-16; **T_full (all T1–T4) equals the XC-1 VALIDATION R_A at all three cells (|diff| = 0.0)**.

## 3. Results (VALIDATION, primary template set T2–T4; nats)
T(v) = KL(p_H‖p_U) − KL(p_H‖p_{U+v}); higher = v restores more of the persona effect. Preservation = T_rec / T_full.

| Site | T_full | T_rec [95 % CI] | T_err [95 % CI] | Interaction I | Preservation T_rec/T_full [95 % CI] | Shapley share (rec) | \|I\|/T_full [CI high] | Additive reading |
|---|---|---|---|---|---|---|---|---|
| L0 slot | 0.137 | 0.131 [0.093, 0.175] | 0.010 [0.006, 0.016] | −0.005 | **0.959 [0.945, 0.969]** | 0.942 | 0.116 [0.141] | allowed |
| L12 final | 0.079 | 0.060 [0.035, 0.092] | 0.011 [−0.001, 0.028] | +0.009 | **0.753 [0.679, 0.807]** | (0.809) | 0.293 [0.394] | bounds only |
| L23 final | 0.137 | 0.096 [0.064, 0.137] | 0.007 [−0.005, 0.022] | +0.034 | **0.701 [0.640, 0.755]** | (0.823) | 0.326 [0.411] | bounds only |

Controls (q95 of 20 draws each): isotropic matched-norm Δ_rec 0.024 / −0.002 / −0.020 and decoder-span shuffle
0.022 / −0.003 / −0.019 at L0 / L12 / L23. At the final-token sites random edits *lower* T, so the control bar is
below zero there; T_rec CI low > 0 is the binding condition.

**Hypotheses** (`hypotheses.csv`, BH over the 3 sites):

| Site | H4a sufficiency (q) | H4b φ_rec − φ_err (q) |
|---|---|---|
| L0 slot | **PASS** (3.6e-11) | **PASS**, +0.121 (1.3e-8) |
| L12 final | **PASS** (4.7e-6) | **PASS**, +0.049 (7.5e-6) |
| L23 final | **PASS** (5.4e-8) | **PASS**, +0.089 (7.2e-8) |

**Other readings:**
* Other direction (H − Δ_rec toward U) at L0 / L12 / L23: TB_rec 0.158 / 0.052 / 0.117 vs TB_full 0.164 / 0.072 / 0.164.
* Synonym control (Hp toward H): S_rec 0.072 / 0.037 / 0.051 vs S_full 0.076 / 0.050 / 0.076.
* Neutral control (N1–N2): preservation 0.97 (L0 slot), 0.58 (L12 final), 0.43 (L23 final), on a much smaller
  effect (T_full 0.026 / 0.006 / 0.026).
* T1 alone (secondary): preservation 0.79 (L0 slot) but only 0.20 (L23 final) and 0.23 (L12 final, CI includes 0).
* All T1–T4 (secondary): preservation 0.951 / 0.738 / 0.677.

## 4. Pre-registered criteria
| Item | Outcome |
|---|---|
| Hard sanity checks (incl. T_full = XC-1 R_A) | PASS |
| H4a reconstruction sufficiency | MET at 3/3 sites |
| H4b reconstruction carries more than the error (Shapley) | MET at 3/3 sites |
| Additive (Shapley) reading | allowed only at L0 slot; at L12/L23 final only the T_rec / T_err bounds are interpreted |
| H4c feature concentration | NOT RUN (declared deviation); no feature-level claim |

## 5. Reading (pre-written categories)
* **L0 slot: reconstruction behaviourally sufficient.** High T_rec (96 % of the effect), low T_err, small
  interaction.
* **L12 and L23 final: interaction-dominated, bounds only.** The reconstruction alone restores 70–75 %, the error
  alone 5–14 % (CI includes 0), and the remainder (up to a third of T_full at L23) appears only when both are patched
  together. This is not a "faithfulness failure in the error" (T_err is small); the missing part is non-additive.
* **GPT-2 M4 comparison (descriptive only; different SAEs, widths and tokenizers):** GPT-2 kept 93–97 % at
  blocks.1 slot and 62–68 % at L8/L11 final. Gemma Scope shows the same pattern: near-complete at the early persona
  token, roughly two thirds to three quarters at the late final token.

## 6. Limitations and deviations
* VALIDATION finding only; TEST was not loaded.
* One public SAE per site (no seed variability); a site claim is that SAE passing.
* H4c omitted, so no claim on how many features carry the effect.
* T4 dominates the pooled means (as on GPT-2); T1 preservation at the final-token sites is weak.
* Sites come from a reduced 9-layer XC-1 grid.

## 7. Files
`site_selection.json`, `per_pair.csv`, `decomposition_summary.csv`, `hypotheses.csv`, `sanity_checks.json`,
`control_draws.npz`. `run.log` is not committed.

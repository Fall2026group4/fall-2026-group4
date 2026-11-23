# XC-1: reduced positive-control causal site sweep on Gemma-2-2B (2026-11-23). Status: IN PROGRESS (pre-registered, frozen; full run started 19:23 UTC)

Evidence level on completion: TRAIN selection; VALIDATION confirmation (validation finding). No SAE; TEST never loaded.

Frozen protocol and hashes: [`preregistration.json`](preregistration.json) (sha256 c1303b62…).
Script: [`scripts/cross_model/24_xc1_gemma_site_sweep.py`](../../../scripts/cross_model/24_xc1_gemma_site_sweep.py).

## 1. Before implementation
* **Question.** Where does original Gemma carry the persona-instruction effect (RepE-LM T1–T4) under the M3
  protocol? Which sites justify an M4 decomposition (M4-G)?
* **Port of M3 (`scripts/repe/23`).** Outcomes, statistics, the selection rule and the stop rule are unchanged.
* **Declared substitutions:**
  * **Model and site:** Gemma resid_post.
  * **Reduced grid:** L ∈ {0, 2, 5, 8, 12, 16, 19, 23, 25} × 5 position groups = 45 cells; Holm over 45.
  * **Trivial cell:** only (25, final).
  * **TRAIN screening:** 100 statements × T1–T4, with random controls on 30 of these statements.
  * **VALIDATION:** all 89 statements × T1–T4.
  * **Shapes:** every forward pass uses an identical shape.
  * **Logits:** the soft-cap is applied in fp32.
* **Hard sanity checks.** In the smoke test, all were exactly 0.0.
* **Smoke (TRAIN only, `smoke/`).** Mechanics only; not evidence.

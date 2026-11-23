# M4-G: SAE patch decomposition on Gemma-2-2B at the XC-1 sites (2026-11-23). Status: IN PROGRESS (pre-registered, frozen before XC-1 read any VALIDATION pair; queued after XC-1)

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

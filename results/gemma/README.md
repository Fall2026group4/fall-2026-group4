# Gemma-2-2B results

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

* `steering_sweep_layer11/` — **AUTHORITATIVE**. `scripts/gemma/01_steering_sweep.py`: layer 11 `resid_post`, grouped-by-city splits, symmetric ±0.01…0.5 × mean-act-norm grid, 20 norm-matched random directions per strength, bare + context directions, segment + final-token sites; val and test. `smoke/` is **SMOKE**.
* Earlier, **SUPERSEDED** Gemma outputs (not moved): `results/tables/gemma/neg_cities/` and the root-level `gemma_run.log` / `gemma_neg.log` came from `scripts/run_gemma_baseline.py`, which exists only on the `aditi` branch (random row split → city leakage, no controls; neg_cities run crashed).
* No SAE has been evaluated on Gemma yet (no Gemma Scope weights cached).

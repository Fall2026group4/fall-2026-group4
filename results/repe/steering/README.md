# RepE steering

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

* `causal_random_controls/` — **AUTHORITATIVE**. Pre-registered (`preregistration.json`) test vs 200 norm-matched Gaussian directions, 200 random features and 200 random 10-feature sets per SAE, positive control (activation patch), bootstrap CIs. One disclosed protocol deviation (`docs/2026-11-04_methodology_changes.md` §C). `smoke/` is **SMOKE**.
* `sae_steering_pilot.csv`, `sae_steering_heldout.csv`, `bidirectional_fidelity.csv`, `bootstrap_fidelity_ci.csv`, `final_repe_causal_summary.csv` — **SUPERSEDED as causal evidence.** No SAE is in the forward pass: GPT-2 is steered with a direction computed *from reconstructions*, so the 0.93–0.99 effect correlations measure direction similarity. Only 9 test questions and 20 random directions; no positive control or significance test.

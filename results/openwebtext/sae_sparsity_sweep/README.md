# 1000-step sparsity sweep (all 12 layers, targets 32/64/128, seed 42)

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

**VALID (scope)**: these runs are far from converged (e.g. layer-6 TopK val MSE 2.19 at 1k vs 0.66 at 15k), so rankings measure convergence speed, not reconstruction quality. JumpReLU misses its L0 target in many cells (2/12 matched at L0 = 32). Use only through `../layer_comparison/sparsity_matched_cells.csv`.

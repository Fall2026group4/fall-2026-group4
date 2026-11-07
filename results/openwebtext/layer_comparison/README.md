# Layer comparison (OWT)

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

* `sparsity_comparison.csv` — **AUTHORITATIVE**. Every run (1000-step sweep, original 15k, uninterrupted 15k) with target vs actual L0 and `matching_table` ∈ {matched, unmatched} at the 10 % criterion (20 % reported alongside).
* `sparsity_matched_cells.csv` — **AUTHORITATIVE**. Only (source, layer, target, seed) cells in which all three architectures exist; `all_three_matched` says whether they are comparable.
* `architecture_summary.csv` — **INVALID, do not cite.** Averages unmatched runs and raw MSE across layers whose scales differ ~1000×; no script in the repo produces it. Kept for provenance only.
* `all_layers_comparison.csv`, `0*_by_layer.png`, `*_comparison.png` — **VALID (scope)**: the 1000-step sweep (`scripts/openwebtext/04_comparing_sae_layers.py`), i.e. convergence speed, matched and unmatched runs together.

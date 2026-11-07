# Reconstruction evaluation (OWT test, 12 700 non-BOS tokens)

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

* `uninterrupted_protocol/` — **AUTHORITATIVE**. `test_metrics_all_runs.csv` (one row per layer × arch × seed; extended in place, completed runs are skipped), `run_config.json` (accumulates every invocation), `original_vs_uninterrupted_layers_2_6_10.csv`, `smoke/` (**SMOKE**).
* `layers_2_6_10/` — **SUPERSEDED for cross-layer comparison** by `uninterrupted_protocol/`; **VALID within a layer**. Its `training_procedure` column records the two mixed procedures.
* `layer_6/` — original layer-6 test evaluation; **VALID (scope)**.
* `layer_6_finetuning_generalization/` — **SUPERSEDED**: validation split only. Test-split version: `results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv`.

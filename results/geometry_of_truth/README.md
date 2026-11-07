# Geometry of Truth results (GPT-2 Small)

Consolidated on 2026-11-06: the former `results/got/` subdirectories now live here (moved with `git mv`, checksums verified; see `docs/migration_manifest.csv`). JSON/TXT provenance files inside them (e.g. `final_test_protocol/pretest_protocol.txt`, `sae_finetuning/*/run_info.json`, `final_test_evaluation/evaluation_manifest.json`) still quote the old paths `results/got/…` and `scripts/Geometry of Truth/…`; they are historical records and were not edited.

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

**Layer 6, in-house OWT-trained SAEs, grouped splits (scripts `scripts/geometry_of_truth/01–09`):**
* `original_truth_probes/`, `sae_truth_preservation/`, `sae_diagnostic_truth_probes/`, `independent_truth_probe/` — **VALID** (validation-split stages).
* `final_test_protocol/`, `final_test_evaluation/` — **AUTHORITATIVE**: frozen MLP probe, 5 seeds, test touched once. `final_test_archive/` — earlier archived run.
* `sae_finetuning/layer_6_{arch}/` — reconstruction-only and truth-aware checkpoints + validation comparison. **VALID (scope)**: base and fine-tuned SAEs differ in L0 (not sparsity-matched). `heldout_comparison_all_datasets.csv` — **AUTHORITATIVE**: all 9 variants on identical OWT-test / GoT-test / RepE-test / causal sets.
* `got_faithfulness_frozen_probes.csv` — **AUTHORITATIVE**: frozen logistic + MLP probes (layer 6) and the layer-10 frozen-probe result in one table.

**Layer 10, public SAELens `gpt2-small-res-jb` (legacy scripts `scripts/run_geometry_*.py`):**
* `baseline_probe_results.csv`, `sae_faithfulness_results.csv`, `sae_*`, `geometry_of_truth_feature_stats.csv` — **VALID (scope)** but **SUPERSEDED as a faithfulness metric**: the "reconstructed" probe is *re-trained on reconstructions* (decodability, 0.851 AUROC), not the original probe applied to them.
* `layer10_frozen_probe/` — **AUTHORITATIVE**: legacy reproduction + frozen original probe on reconstructions (AUROC 0.639, gap 0.21 [0.16, 0.27]), 6 city splits.

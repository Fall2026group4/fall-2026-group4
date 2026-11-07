# Project documentation

| File | Contents |
|---|---|
| `RESEARCH_PROGRESS.md` | **Start here.** Chronological research journal, from project set-up to now |
| `RESEARCH_ROADMAP.md` | Completed, planned, blocked and abandoned work; known missing baselines |
| `EXPERIMENT_RESULTS_SUMMARY.md` | Quantitative tables of all experiments, with splits, metric directions and seed statistics |
| `RESEARCH_DECISIONS.md` | Decision log: why each dataset, metric, protocol and method was chosen or abandoned |
| `EXPERIMENT_CHANGELOG.md` | Audit trail of file, code, split and method changes, with commit hashes |
| `templates/EXPERIMENT_TEMPLATE.md` | Template for every new experiment (pre-registration and results) |
| `2026-11-07_truth_aware_finetuning_audit.md` | Audit that found the circular teacher/evaluation-probe setup |
| `checkpoint_registry.csv` and the files below | Machine-readable registries and manifests |
| `results_registry.md` | Which results are authoritative, valid-in-scope, superseded or invalid (start here) |
| `research_gaps_and_plan.md` | Gap analysis of the six reconstruction-improvement methods and the all-12-layer extension plan |
| `checkpoint_registry.csv` | Every SAE checkpoint under `results/` with family, status, layer, arch, seed, L0, SHA-256 (`scripts/shared/build_checkpoint_registry.py`) |
| `migration_manifest.csv` | Every file moved, copied, edited or created by the 2026-11-06 reorganization, with SHA-256 before/after |
| `experiment_manifest.csv` | Living experiment manifest (one row per experiment; `scripts/shared/write_manifest.py:append_experiment`) |
| `experiment_manifest_2026-11-04.csv` | Historical manifest of the professor-review experiments (script hashes at run time) |
| `2026-11-04_professor_review_report.md` | Full report of the professor-review corrections (historical; paths refer to the pre-migration layout) |
| `2026-11-04_code_audit.md` | Code audit behind those corrections (historical) |
| `2026-11-04_methodology_changes.md` | Methodology decisions (optimizer protocol, sparsity matching, causal protocol, …) (historical) |

The three dated reports are byte-identical copies of the originals. Their paths point to the old layout.
`migration_manifest.csv` maps every old path to its new one. The old folders were retired on 2026-11-07 (D20);
the main script mapping is:

| Old path (2026-11-04 reports) | Current path |
|---|---|
| `scripts/professor_review/analyze_owt_corrections.py` | `scripts/openwebtext/14_sparsity_matching_analysis.py` |
| `scripts/professor_review/got_layer10_frozen_probe.py` | `scripts/geometry_of_truth/10_layer10_frozen_probe.py` |
| `scripts/professor_review/finetuning_repe_got_eval.py` | `scripts/shared/finetuning_heldout_eval.py` |
| `scripts/professor_review/plot_causal_and_gemma.py` | `scripts/shared/plot_causal_and_gemma.py` |
| `scripts/professor_review/write_manifest.py` | `scripts/shared/write_manifest.py` |
| `src/tests/test_professor_review.py` | `src/tests/test_faithfulness_metrics.py` |
| `results/professor_review_corrections/owt_consistent_protocol/` | `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/` (+ checkpoints in `results/openwebtext/convergence/layer_{2,6,10}/*_uninterrupted_seed*/`) |
| `results/professor_review_corrections/causal/` | `results/repe/steering/causal_random_controls/` |
| `results/professor_review_corrections/gemma/` | `results/gemma/steering_sweep_layer11/` |
| `results/professor_review_corrections/got_layer10/` | `results/geometry_of_truth/layer10_frozen_probe/` |
| `results/professor_review_corrections/{final_research_report,code_audit,methodology_changes}.md` | `docs/2026-11-04_{professor_review_report,code_audit,methodology_changes}.md` |
| `results/professor_review_corrections/experiment_manifest.csv` | `docs/experiment_manifest_2026-11-04.csv` |
| top-level CSVs, `figures/`, `logs/`, `smoke/` | next to the result they belong to (`sparsity_*.csv` → `results/openwebtext/layer_comparison/`, `finetuning_comparison.csv` → `results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv`, `got_faithfulness.csv` → `results/geometry_of_truth/got_faithfulness_frozen_probes.csv`, `repe_representation.csv` → `results/repe/representation_preservation_val_test.csv`, …); one row per file in `migration_manifest.csv` |

## Layout conventions (since 2026-11-06)

```
scripts/{openwebtext,geometry_of_truth,repe,gemma}/NN_name.py   dataset pipelines, numbered in run order
scripts/shared/                                                cross-dataset evaluation, plotting, manifest, checkpoint registry
src/evaluation/faithfulness_metrics.py                         shared reconstruction + frozen-probe metrics (use these definitions)
src/sae/checkpoint_registry.py                                 checkpoint lookup / registry builder
results/{openwebtext,geometry_of_truth,repe,gemma}/            outputs of the matching script folder
results/shared/                                                cross-dataset summaries (e.g. spliced_evaluation/)
docs/                                                          reports, manifests, registry
```

* A script writes only into its own dataset's `results/` folder. Cross-dataset scripts in `scripts/shared/` write
  each output into the folder of the dataset it describes.
* New variants of an existing experiment go into a descriptive sub-directory next to the original, for example
  `reconstruction_evaluation/uninterrupted_protocol/`. They never overwrite the original. Superseded results stay in
  place and are marked in the folder's `README.md` and in `results_registry.md`.
* OWT checkpoints: `results/openwebtext/convergence/layer_{L}/{arch}_l0_{target}_{procedure}[_seed{S}]/checkpoint_step_{N}.pt`.
* Training scripts refuse to overwrite an existing checkpoint, and they skip runs that are already recorded in their
  metrics table.
* Smoke tests go in a `smoke/` sub-directory and are never reported as results.
* Data directories keep their short names (`data/got`, `data/repe`, `data/openwebtext`). `scripts/Geometry of Truth/` and `results/got/` were
  consolidated into `scripts/geometry_of_truth/` and `results/geometry_of_truth/`.

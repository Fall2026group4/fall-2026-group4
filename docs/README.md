# Project documentation

| File | Contents |
|---|---|
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
`migration_manifest.csv` maps every old path to its new one.

## Layout conventions (since 2026-11-06)

```
scripts/{openwebtext,geometry_of_truth,repe,gemma}/NN_name.py   dataset pipelines, numbered in run order
scripts/shared/                                                cross-dataset evaluation, plotting, manifest, checkpoint registry
src/evaluation/faithfulness_metrics.py                         shared reconstruction + frozen-probe metrics (use these definitions)
src/sae/checkpoint_registry.py                                 checkpoint lookup / registry builder
results/{openwebtext,geometry_of_truth,repe,gemma}/            outputs of the matching script folder
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

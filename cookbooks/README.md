# Cookbooks

This folder held the early exploratory notebooks. They were removed on 2026-11-28 as obsolete: the project's
pipelines now live in `scripts/` and `src/`. The last version of each notebook is in the Git history at commit
`8d0bac0`:

- `02_pretained_sae_baseline.ipynb`: GPT-2 Small + pretrained SAE, Feature 974 detection and steering (Phase 1). Produced
  `results/figures/feature_974_paris_vs_nonparis.svg` and `results/tables/feature_974_detection_results.csv`.
- `03_geometry_of_truth_eda.ipynb`: Geometry of Truth EDA. Produced `results/geometry_of_truth_eda_summary.csv`.
- `04_geometry_of_truth_baseline.ipynb`: legacy layer-10 GoT SAE feature baseline. Produced
  `results/geometry_of_truth/geometry_of_truth_feature_stats.csv` and
  `results/figures/geometry_of_truth_sae_feature_differences.svg`.

Restore one with `git show 8d0bac0:cookbooks/<name>.ipynb > <name>.ipynb`.

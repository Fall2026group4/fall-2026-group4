# Cookbooks

Narrative, run-it-top-to-bottom notebooks. Each one imports its actual
logic from `src/` and contains only narrative text, function calls, and
figure/table output — no model loading, hooks, or scoring logic defined
inline. See `src/` for the implementation and `src/tests/` for coverage.

- `01_data_exploration.ipynb` — OpenWebText EDA (document/token length
  distributions, GPT-2 context-window compliance). Saves figures/tables to
  `../results/figures/` and `../results/tables/`.
- `02_pretained_sae_baseline.ipynb` — Loads GPT-2 Small + a pretrained SAE
  (`src/models/gpt2_sae.py`), scores Feature 974 on positive/negative
  examples (`src/evaluation/detection.py`), and runs a causal steering
  sweep (`src/sae/steering.py`).
- `03_geometry_of_truth_eda.ipynb` — EDA on the Geometry of Truth dataset
  for Phase 5 concept-detection work.

Run these from inside `cookbooks/` (relative paths to `../results/` and
`../src/` assume that working directory).

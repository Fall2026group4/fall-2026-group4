# Cookbooks

Narrative, run-top-to-bottom Python scripts (not notebooks). Each one
imports its actual logic from `src/` and contains only narrative
comments, function calls, and figure/table output — no model loading,
hooks, or scoring logic defined inline. See `src/` for the implementation
and `src/tests/` for coverage.

- `01_data_exploration.py` — OpenWebText EDA (document/token length
  distributions, GPT-2 context-window compliance). Saves figures to
  `../results/figures/` and tables to `../results/tables/`.
- `02_pretained_sae_baseline.py` — Loads GPT-2 Small + a pretrained SAE
  (`src/models/gpt2_sae.py`), scores Feature 974 on positive/negative
  examples (`src/evaluation/detection.py`), and runs a causal steering
  sweep (`src/sae/steering.py`).
- `03_geometry_of_truth_eda.py` — EDA on the Geometry of Truth "cities"
  dataset for Phase 5 concept-detection work.
- `04_geometry_of_truth_baseline.py` — GPT-2 + pretrained SAE baseline on
  the Geometry of Truth dataset: loads residual activations and SAE
  features via the same `src/models/gpt2_activations.py` and
  `src/sae/geometry_sae.py` helpers used by the batch pipeline in
  `scripts/`, so results stay consistent between the two. The full
  train/test pipeline (caching, probes, detection, faithfulness) lives
  in `scripts/` as separate stages; this script is the single-pass,
  narrative walkthrough version for exploration.

## Running

Run any script directly from the repo root or from inside `cookbooks/` —
each one resolves paths relative to its own file location, not the
current working directory:

```bash
python cookbooks/01_data_exploration.py
python cookbooks/02_pretained_sae_baseline.py --device cpu
python cookbooks/03_geometry_of_truth_eda.py
python cookbooks/04_geometry_of_truth_baseline.py --device cpu
```

`02` and `04` accept `--device` (e.g. `--device cuda`) to run on GPU when
available.

These were previously `.ipynb` notebooks. They were converted to plain
`.py` scripts so they run the same way in any environment (local venv,
WSL, Colab, CI) without a notebook kernel, and so diffs in code review are
readable instead of notebook JSON.

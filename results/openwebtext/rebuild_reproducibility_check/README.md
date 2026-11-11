# Rebuild reproducibility check, 108 in-house SAEs (2026-11-10). Status: COMPLETED

Evidence level: confirmed (test metrics compared row by row).

## 1. What was done

* All 108 uninterrupted SAEs were retrained from scratch on Kaggle (Tesla T4) with `scripts/openwebtext/13_consistent_protocol_retrain.py`, using the committed configuration: 12 layers, ReLU / TopK / JumpReLU, seeds 42 / 1042 / 2042, 15,000 steps, target L0 128, d_sae 3072.
* The checkpoints are not tracked (`*.pt` is in `.gitignore`).
* Total training time was about 131 minutes.
* The rebuilt test metrics were then compared with the original run's `test_metrics_all_runs.csv`.

## 2. Result (`rebuild_vs_original_test_metrics.csv`, 108 rows)

| Metric | Max absolute difference | Mean absolute difference |
|---|---|---|
| Test explained variance | 0.0013 | 0.0002 |
| Test actual L0 | 1.19 | 0.17 |
| Test cosine similarity | 0.0002 | 0.00005 |

The rebuild reproduces the original SAEs. All downstream follow-up results (`results/shared/frozen_probe_error_decomposition/`, `sparsity_matched_truth_preservation/`, `results/repe/steering/sae_in_the_loop/`, `results/geometry_of_truth/causal_truth_steering/`) use these rebuilt checkpoints.

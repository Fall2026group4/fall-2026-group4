# OpenWebText results (GPT-2 Small, `hook_resid_pre`, in-house SAEs, d_sae = 3072)

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

| Path | Status | What it is |
|---|---|---|
| `reconstruction_evaluation/uninterrupted_protocol/` | **AUTHORITATIVE** (cross-layer and within-layer) | One uninterrupted 15k-step Adam run per layer × arch × seed (42/1042/2042), OWT test (12 700 tokens). Layers 2/6/10 done; extend with `scripts/openwebtext/13_consistent_protocol_retrain.py`. |
| `convergence/layer_*/*_uninterrupted_seed*/` | **AUTHORITATIVE** checkpoints for the table above | |
| `reconstruction_evaluation/layers_2_6_10/` | **SUPERSEDED** for cross-layer claims; **VALID (scope)** within a layer | Mixed procedures (L6 resumed with Adam restored; L2/L10 warm start with Adam reset). EV differs from the corrected runs by ≤ 0.006, rankings unchanged. |
| `convergence/layer_6/*_resume_5000_to_15000/` | **VALID (scope)** | The layer-6 "pretrained" base SAEs used by every GoT/RepE/fine-tuning/causal experiment. Not retrained; those experiments are layer-6-only, so the procedure never crosses a comparison. |
| `convergence/layer_{2,10}/*_warmstart_1000_to_15000/` | **SUPERSEDED** by the uninterrupted runs | |
| `convergence/layer_6/jumprelu_kernel_ste_l0_128/` | **VALID (scope)**, unreproducible | Kernel-STE JumpReLU experiment; its generating script is not in the repository. |
| `sae_sparsity_sweep/` | **VALID (scope)** = convergence speed at 1000 steps only | TopK's lead here reverses at 15k steps. Many JumpReLU runs miss the L0 target. |
| `layer_comparison/sparsity_comparison.csv`, `sparsity_matched_cells.csv` | **AUTHORITATIVE** | Matched (≤10 % L0 error) vs unmatched separation; `scripts/openwebtext/14_sparsity_matching_analysis.py`. |
| `layer_comparison/architecture_summary.csv` | **INVALID** | Averages unmatched runs and raw MSE across layers; no generating script. |
| `layer_comparison/all_layers_comparison.csv`, `*.png` | **VALID (scope)** | 1000-step sweep, from `04_comparing_sae_layers.py`; mixes matched and unmatched runs. |
| `reconstruction_evaluation/layer_6/`, `layer_6_finetuning_generalization/` | **VALID (scope)** / superseded for FT | FT generalization was on validation; held-out test version is `results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv`. |
| `sae_pilot/` | historical pilot | |
| `*/smoke/` | **SMOKE** | |

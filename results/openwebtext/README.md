# OpenWebText results (GPT-2 Small, `hook_resid_pre`, in-house SAEs, d_sae = 3072)

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

| Path | Status | What it is |
|---|---|---|
| `reconstruction_evaluation/uninterrupted_protocol/` | **AUTHORITATIVE** (cross-layer and within-layer) | One uninterrupted 15k-step Adam run per layer × arch × seed (42/1042/2042), OWT test (12 700 tokens). **All 12 layers done** (108 runs; layers 2/6/10 on 2026-11-04, the rest on 2026-11-07). All-layer tables: `all_layers_*.csv` (`scripts/openwebtext/15_all_layer_baseline_analysis.py`). One run unmatched: layer 0 JumpReLU seed 42 (L0 73.6). |
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

**Spliced-in causal evaluation (Phase 0, 2026-11-07):** `spliced_evaluation/` — **AUTHORITATIVE (validation only)**. GPT-2 predictions with `hook_resid_pre` replaced by each of the 108 uninterrupted SAE reconstructions, with identity, fp16, mean-ablation and norm-matched random-error controls. See `results/shared/README.md`. This is separate from the steering results.

**P1 pilot (2026-11-07):** `p1_sparsity_control_pilot/` — **PILOT** (seed 42, validation only); see its README. Checkpoints: `convergence/layer_{0,2,6,10}/*_p1_seed42/`.

**GPT-2 improvement phase (2026-11-10):**
* `r2_scaled_saes/` (**R2**): in-house data-scaled SAEs (39; 100M tokens; width 12,288/3,072; L1/L8/L11) with the full
  ladder and pre-registered comparisons. **VALIDATION**; reconstruction and behaviour **CONFIRMED on TEST** in
  `results/shared/m5_confirmatory_test/`. Checkpoints are gitignored and listed in `docs/checkpoint_registry.csv`.
* `r4_kl_finetune/` (**R4**): KL+MSE fine-tune at fixed L0 vs an MSE control. Behaviour-only gain, confirmed on TEST;
  causal regression at L11 (validation).

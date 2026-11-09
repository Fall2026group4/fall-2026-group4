# Experiment results summary (quantitative)

Actual values from committed result files. Every table states its split, the metric direction, seed statistics and its evidential status. **Rows are comparable only within a table**, never across tables with different SAEs, splits, budgets or sparsity.

**Metric directions:**
* EV, cosine, loss recovered, AUROC, direction cosine, persona-shift cosine: ↑ better.
* MSE, NMSE (= 1 − EV), AUROC gap (original − reconstructed), KL, CE increase, dead fraction: ↓ better.

**Statuses:** TEST = held-out test split, evaluated once under a protocol. VALIDATION = validation split. EXPLORATORY.

**Common setup:**
* GPT-2 Small.
* In-house SAEs: d_sae 3072, target L0 128, `blocks.L.hook_resid_pre`.
* Seeds 42/1042/2042 unless stated.

---

## 1. Reconstruction: OWT test, 108 uninterrupted runs (TEST)
Source: `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/all_layers_summary.csv`. Mean ± sd over 3 seeds, 12,700 tokens.

* **Layer-0 JumpReLU includes the unmatched seed 42** (L0 73.6). Without it: EV 0.884, L0 ≈ 128.6.
* **ReLU overshoots its target by 2–6 % in L0.** Compare architectures only within a layer.

### Explained variance and L0 by layer
| Layer | ReLU EV | TopK EV | JumpReLU EV | ReLU L0 | TopK L0 | JumpReLU L0 |
|---|---|---|---|---|---|---|
| 0 | 0.950 ± 0.000 | 0.911 ± 0.000 | 0.826 ± 0.101 | 135.1 | 125.3 | 110.3 |
| 1 | 0.937 ± 0.000 | 0.921 ± 0.001 | 0.933 ± 0.000 | 132.2 | 128.0 | 129.3 |
| 2 | 0.936 ± 0.000 | 0.926 ± 0.000 | 0.935 ± 0.002 | 131.9 | 128.0 | 129.0 |
| 3 | 0.905 ± 0.000 | 0.906 ± 0.000 | 0.915 ± 0.001 | 131.7 | 128.0 | 129.1 |
| 4 | 0.872 ± 0.000 | 0.884 ± 0.000 | 0.893 ± 0.001 | 131.1 | 128.0 | 129.1 |
| 5 | 0.848 ± 0.000 | 0.868 ± 0.000 | 0.875 ± 0.000 | 130.8 | 128.0 | 129.0 |
| 6 | 0.832 ± 0.001 | 0.857 ± 0.000 | 0.865 ± 0.000 | 131.0 | 128.0 | 129.1 |
| 7 | 0.819 ± 0.001 | 0.850 ± 0.000 | 0.855 ± 0.000 | 131.4 | 128.0 | 129.9 |
| 8 | 0.805 ± 0.000 | 0.840 ± 0.000 | 0.844 ± 0.000 | 131.9 | 128.0 | 130.3 |
| 9 | 0.789 ± 0.001 | 0.831 ± 0.000 | 0.833 ± 0.000 | 132.4 | 128.0 | 131.0 |
| 10 | 0.773 ± 0.000 | 0.822 ± 0.000 | 0.823 ± 0.000 | 132.7 | 128.0 | 131.6 |
| 11 | 0.757 ± 0.001 | 0.815 ± 0.000 | 0.815 ± 0.000 | 133.2 | 128.0 | 132.1 |

**Matched ranking** (`all_layers_matched_architecture_comparison.csv`; paired by seed):

| Layer(s) | Leader |
|---|---|
| 0 | not ranked (unmatched seed) |
| 1 | ReLU |
| 2 | ReLU ≈ JumpReLU |
| 3–9 | JumpReLU |
| 10–11 | JumpReLU ≈ TopK |

**Earlier sweeps (EXPLORATORY):**
* **1000-step sweep** (`sae_sparsity_sweep/`; L0 32/64/128): only 15 of 36 cells have all three architectures matched, and JumpReLU matched L0 = 32 at only 2/12 layers. It measures convergence speed only.
* **`architecture_summary.csv`: INVALID.**

## 2. Representation preservation: GoT truth, frozen probes, all layers (TEST)
Source: `results/geometry_of_truth/all_layer_frozen_probe_evaluation/summary.csv`.

**Method:**
* Logistic probe fitted on original train rows, C chosen on original validation, applied frozen to reconstructions.
* Gap = AUROC(original) − AUROC(reconstructed). Mean ± sd over matched seeds; test n = 452 geographic and 596 numerical.
* "Informative" means original validation AUROC ≥ 0.70.

**Caveats:**
* NMSE on GoT final-token activations is > 1, so use relative error.
* SAE L0 on GoT prompts is 8–114 for ReLU and JumpReLU, so it is not matched.


### Geographic (cities + neg_cities)
| Layer | Original val AUROC | Original test AUROC | Informative | ReLU gap | TopK gap | JumpReLU gap |
|---|---|---|---|---|---|---|
| 0 | 0.534 | 0.498 | no | -0.002 ± 0.001 | 0.002 ± 0.006 | -0.003 ± 0.001 |
| 1 | 0.528 | 0.519 | no | 0.021 ± 0.009 | 0.026 ± 0.005 | 0.024 ± 0.003 |
| 2 | 0.594 | 0.562 | no | 0.073 ± 0.001 | 0.072 ± 0.008 | 0.069 ± 0.003 |
| 3 | 0.648 | 0.654 | no | 0.158 ± 0.002 | 0.142 ± 0.017 | 0.157 ± 0.002 |
| 4 | 0.711 | 0.702 | yes | 0.197 ± 0.005 | 0.185 ± 0.006 | 0.189 ± 0.010 |
| 5 | 0.826 | 0.802 | yes | 0.288 ± 0.014 | 0.278 ± 0.006 | 0.299 ± 0.008 |
| 6 | 0.841 | 0.825 | yes | 0.302 ± 0.005 | 0.276 ± 0.004 | 0.311 ± 0.019 |
| 7 | 0.865 | 0.829 | yes | 0.302 ± 0.009 | 0.302 ± 0.009 | 0.324 ± 0.014 |
| 8 | 0.850 | 0.834 | yes | 0.299 ± 0.050 | 0.291 ± 0.014 | 0.288 ± 0.041 |
| 9 | 0.865 | 0.839 | yes | 0.281 ± 0.023 | 0.269 ± 0.005 | 0.289 ± 0.017 |
| 10 | 0.852 | 0.822 | yes | 0.255 ± 0.015 | 0.250 ± 0.019 | 0.267 ± 0.016 |
| 11 | 0.825 | 0.792 | yes | 0.222 ± 0.005 | 0.229 ± 0.016 | 0.210 ± 0.015 |

### Numerical (larger_than + smaller_than)
| Layer | Original val AUROC | Original test AUROC | Informative | ReLU gap | TopK gap | JumpReLU gap |
|---|---|---|---|---|---|---|
| 0 | 0.500 | 0.500 | no | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 1 | 0.991 | 0.993 | yes | 0.471 ± 0.005 | 0.468 ± 0.024 | 0.482 ± 0.003 |
| 2 | 0.996 | 0.997 | yes | 0.491 ± 0.005 | 0.472 ± 0.024 | 0.486 ± 0.009 |
| 3 | 0.995 | 0.997 | yes | 0.488 ± 0.001 | 0.461 ± 0.002 | 0.480 ± 0.021 |
| 4 | 0.997 | 0.996 | yes | 0.482 ± 0.006 | 0.447 ± 0.011 | 0.463 ± 0.010 |
| 5 | 0.999 | 0.998 | yes | 0.452 ± 0.007 | 0.465 ± 0.016 | 0.456 ± 0.041 |
| 6 | 1.000 | 0.999 | yes | 0.453 ± 0.031 | 0.438 ± 0.032 | 0.459 ± 0.026 |
| 7 | 0.999 | 1.000 | yes | 0.421 ± 0.010 | 0.411 ± 0.040 | 0.436 ± 0.027 |
| 8 | 0.999 | 1.000 | yes | 0.461 ± 0.013 | 0.454 ± 0.034 | 0.471 ± 0.004 |

**Decodability diagnostic** (VALIDATION; probe re-trained on reconstructed train rows; *not* a faithfulness metric):
* Geographic: 0.70–0.84 (vs original 0.83–0.87).
* Numerical: 0.60–0.98.

Truth information is largely displaced rather than erased.

**Layer 10, public SAELens `gpt2-small-res-jb`** (legacy city split, n_test = 450; TEST):

| Probe | AUROC | Gap [95 % CI] |
|---|---|---|
| Original probe on original activations | 0.853 | — |
| Probe re-trained on reconstructions | 0.851 | 0.002 [−0.03, 0.03] |
| **Frozen original probe on reconstructions** | **0.639** | **0.214 [0.16, 0.27]** |

* Reconstruction: cosine 0.977; EV −1.50.

## 3. Representation preservation: RepE honesty, all layers (TEST)
Source: `results/repe/all_layer_frozen_probe_evaluation/summary.csv`.

* **Probes:** the mass-mean persona direction (train), and a logistic probe whose C was chosen on validation.
* **Test n = 180 prompts.**
* **L0 is not matched on RepE prompts:** 143–278 for ReLU/JumpReLU; TopK fixed at 128.

| 9 | 1.000 | 1.000 | yes | 0.432 ± 0.011 | 0.371 ± 0.029 | 0.451 ± 0.024 |
| 10 | 0.999 | 0.999 | yes | 0.458 ± 0.005 | 0.434 ± 0.038 | 0.507 ± 0.024 |
| 11 | 1.000 | 0.999 | yes | 0.443 ± 0.024 | 0.413 ± 0.057 | 0.404 ± 0.024 |

### Mass-mean (diffmean) probe gap
| Layer | Original val AUROC | Original test AUROC | Informative | ReLU gap | TopK gap | JumpReLU gap |
|---|---|---|---|---|---|---|
| 0 | 0.795 | 0.785 | yes | -0.011 ± 0.006 | -0.002 ± 0.004 | 0.005 ± 0.000 |
| 1 | 0.791 | 0.764 | yes | 0.023 ± 0.004 | 0.016 ± 0.007 | 0.016 ± 0.002 |
| 2 | 0.810 | 0.787 | yes | 0.031 ± 0.003 | 0.035 ± 0.005 | 0.028 ± 0.006 |
| 3 | 0.875 | 0.856 | yes | 0.084 ± 0.008 | 0.079 ± 0.011 | 0.064 ± 0.013 |
| 4 | 0.845 | 0.849 | yes | 0.104 ± 0.005 | 0.090 ± 0.007 | 0.065 ± 0.004 |
| 5 | 0.840 | 0.843 | yes | 0.126 ± 0.023 | 0.081 ± 0.004 | 0.068 ± 0.008 |
| 6 | 0.893 | 0.881 | yes | 0.156 ± 0.005 | 0.099 ± 0.016 | 0.093 ± 0.012 |
| 7 | 0.827 | 0.828 | yes | 0.114 ± 0.025 | 0.084 ± 0.016 | 0.055 ± 0.011 |
| 8 | 0.782 | 0.803 | yes | 0.121 ± 0.018 | 0.100 ± 0.001 | 0.088 ± 0.004 |
| 9 | 0.712 | 0.727 | yes | 0.111 ± 0.005 | 0.090 ± 0.001 | 0.078 ± 0.008 |
| 10 | 0.750 | 0.775 | yes | 0.153 ± 0.006 | 0.117 ± 0.004 | 0.104 ± 0.010 |
| 11 | 0.786 | 0.776 | yes | 0.178 ± 0.002 | 0.106 ± 0.008 | 0.100 ± 0.004 |

### Logistic probe gap (mean ± sd)
| Layer | ReLU | TopK | JumpReLU |
|---|---|---|---|
| 0 | -0.005 ± 0.009 | 0.012 ± 0.012 | 0.004 ± 0.006 |
| 1 | 0.046 ± 0.005 | 0.038 ± 0.009 | 0.038 ± 0.005 |
| 2 | 0.067 ± 0.012 | 0.064 ± 0.014 | 0.054 ± 0.004 |
| 3 | 0.131 ± 0.018 | 0.092 ± 0.006 | 0.056 ± 0.002 |
| 4 | 0.181 ± 0.011 | 0.086 ± 0.012 | 0.070 ± 0.018 |
| 5 | 0.245 ± 0.023 | 0.091 ± 0.020 | 0.078 ± 0.004 |
| 6 | 0.238 ± 0.010 | 0.096 ± 0.002 | 0.091 ± 0.021 |
| 7 | 0.272 ± 0.035 | 0.091 ± 0.005 | 0.070 ± 0.027 |
| 8 | 0.284 ± 0.020 | 0.081 ± 0.007 | 0.112 ± 0.012 |
| 9 | 0.304 ± 0.012 | 0.140 ± 0.040 | 0.169 ± 0.009 |

**Direction fidelity at layer 11** (cosine of the re-estimated honesty direction): ReLU 0.74, TopK 0.85, JumpReLU 0.84.

**Correlations** (pooled across layers, so they mix depth with SAE quality):
* RepE gap vs OWT NMSE: ρ = 0.86.
* GoT geographic gap vs OWT NMSE: ρ ≈ 0.

## 4. Behavioural faithfulness: activation splicing, Phase 0 (VALIDATION)
Source: `results/{openwebtext,geometry_of_truth,repe}/spliced_evaluation/summary.csv`.

**Method:**
* GPT-2 with `hook_resid_pre` replaced by the reconstruction at non-BOS positions; matched seeds.
* Original OWT CE = 3.396 nats/token.
* Mean-ablation increases CE by 5.1–8.6 nats.
* **Identity control: exactly 0.**

**These are splicing results, not steering.**

| 10 | 0.324 ± 0.005 | 0.191 ± 0.016 | 0.210 ± 0.032 |
| 11 | 0.284 ± 0.013 | 0.160 ± 0.028 | 0.202 ± 0.034 |

### OWT next-token CE increase (nats per token, ↓)
| Layer | ReLU | TopK | JumpReLU |
|---|---|---|---|
| 0 | 0.029 | 0.057 | 0.036 |
| 1 | 0.019 | 0.026 | 0.009 |
| 2 | 0.037 | 0.038 | 0.021 |
| 3 | 0.078 | 0.057 | 0.040 |
| 4 | 0.114 | 0.075 | 0.058 |
| 5 | 0.167 | 0.105 | 0.091 |
| 6 | 0.177 | 0.104 | 0.092 |
| 7 | 0.198 | 0.114 | 0.106 |
| 8 | 0.223 | 0.126 | 0.117 |
| 9 | 0.250 | 0.150 | 0.140 |
| 10 | 0.280 | 0.168 | 0.164 |
| 11 | 0.358 | 0.213 | 0.225 |

### OWT loss recovered vs mean-ablation (↑)
| Layer | ReLU | TopK | JumpReLU |
|---|---|---|---|
| 0 | 0.996 | 0.991 | 0.995 |
| 1 | 0.998 | 0.997 | 0.999 |
| 2 | 0.995 | 0.994 | 0.997 |
| 3 | 0.987 | 0.991 | 0.994 |
| 4 | 0.984 | 0.990 | 0.992 |
| 5 | 0.967 | 0.980 | 0.982 |
| 6 | 0.966 | 0.980 | 0.982 |
| 7 | 0.963 | 0.979 | 0.980 |
| 8 | 0.958 | 0.976 | 0.978 |
| 9 | 0.951 | 0.971 | 0.973 |
| 10 | 0.947 | 0.968 | 0.969 |
| 11 | 0.930 | 0.958 | 0.956 |

### RepE persona-shift cosine, final-token splice (↑)
| Layer | ReLU | TopK | JumpReLU |
|---|---|---|---|
| 0 | 0.962 | 0.907 | 0.941 |
| 1 | 0.955 | 0.937 | 0.958 |
| 2 | 0.894 | 0.914 | 0.933 |
| 3 | 0.850 | 0.847 | 0.884 |
| 4 | 0.837 | 0.815 | 0.881 |
| 5 | 0.818 | 0.775 | 0.816 |
| 6 | 0.785 | 0.774 | 0.797 |
| 7 | 0.737 | 0.710 | 0.746 |
| 8 | 0.729 | 0.642 | 0.712 |
| 9 | 0.749 | 0.673 | 0.691 |
| 10 | 0.706 | 0.663 | 0.649 |
| 11 | 0.657 | 0.655 | 0.658 |

### GoT statement KL per token, all-position splice (↓)
| Layer | ReLU | TopK | JumpReLU |
|---|---|---|---|
| 0 | 0.035 | 0.123 | 0.043 |
| 1 | 0.023 | 0.030 | 0.016 |
| 2 | 0.053 | 0.044 | 0.024 |
| 3 | 0.074 | 0.059 | 0.048 |
| 4 | 0.109 | 0.078 | 0.079 |
| 5 | 0.215 | 0.120 | 0.114 |
| 6 | 0.176 | 0.115 | 0.122 |
| 7 | 0.207 | 0.124 | 0.105 |
| 8 | 0.218 | 0.130 | 0.140 |
| 9 | 0.293 | 0.151 | 0.167 |
| 10 | 0.293 | 0.167 | 0.181 |
| 11 | 0.327 | 0.212 | 0.233 |

**Additional results:**
* **SAE errors vs random errors of the same norm:**
  * OWT CE increase: SAE errors are 1.0–2.6× worse.
  * Persona-shift cosine: random errors give 0.20–0.73, far below the SAE's.
* **GoT `cities` truth preference** (original 0.82 pairwise accuracy): spliced change within ±0.025 through layer 8; −0.04 to −0.09 at layers 9–11.
* **Layer-6 bridge:** historical resumed SAEs vs uninterrupted SAEs differ by ≤ 0.001 EV and ≤ 0.005 CE increase.

## 5. Causal intervention: pre-registered, 200 controls (TEST)
Source: `results/repe/steering/causal_random_controls/causal_summary.csv`.

* GPT-2 layer 6, final token, 90 test statements.
* E = fraction of the honest-vs-untruthful output shift recovered (↑).

| Intervention | E [95 % CI] | Empirical p |
|---|---|---|
| Positive control (activation patch) | 0.234 [0.151, 0.317] | 0.005 |
| Raw honesty mass-mean direction | 0.0048 [0.0003, 0.0096] | 0.010 vs 200 Gaussian |
| SAE reconstruction directions (9 SAEs) | −0.0011 … 0.0031 | 0.050 – 0.68 |
| SAE top-1 feature (9 SAEs) | −0.0073 … −0.0024 | 0.84 – 1.00 vs random features |
| SAE top-10 feature sets (9 SAEs) | −0.0052 … 0.0012 | 0.26 – 1.00 |

## 6. Steering
* **GPT-2, earlier "SAE steering" correlations 0.93–0.99:** SUPERSEDED. No SAE in the forward pass.
* **Gemma-2-2B, layer 11** (`results/gemma/steering_sweep_layer11/`). Test, n = 226, 20 random directions per strength; descriptive. Baseline 58.8 % TRUE.

  | Direction (segment site) | α = −0.5 | α = +0.5 | Note |
  |---|---|---|---|
  | Bare (original) | 99.6 % TRUE (z ≈ 0) | 96.0 % TRUE (z 1.7) | |
  | Random directions | ≈ 73 % TRUE | ≈ 73 % TRUE | generic large-perturbation response bias |
  | Context direction | 23 % TRUE | 100 % TRUE | z up to 5.4 |

  The context direction is sign-consistent at every strength.
* **Gemma `aditi` baseline:** SUPERSEDED (row split, no controls).

## 7. Fine-tuning methods at layer 6

The categories are not comparable across sub-tables: §7a is the historical TEST evaluation; §§7b–7d are VALIDATION with independent B-group probes.

### 7a. Historical fine-tuning (seed 42; GoT test, MLP probe 5 seeds; TEST)
Source: `heldout_comparison_all_datasets.csv`.

* **L0 not matched:** ReLU on GoT prompts 65 → 144 → 94.
* **Logistic-probe numbers for truth-aware are circular;** see the audit.

| Variant | Geographic MLP gap, ReLU / TopK / JumpReLU | Geographic logistic gap (truth-aware = circular) | OWT test EV, ReLU / TopK / JumpReLU |
|---|---|---|---|
| Base | .327 / .154 / .323 | .300 / .261 / .289 | .831 / .857 / .866 |
| Reconstruction-only | .161 / .110 / .146 | .278 / .296 / .295 | .831 / .850 / .849 |
| Truth-aware | .225 / .184 / .218 | (.019 / .076 / .063, circular) | .832 / .856 / .864 |

### 7b. Corrected truth-aware pilot (TopK, VALIDATION, GoT geographic; mean ± sd over 3 seeds)
Source: `results/geometry_of_truth/truth_aware_pilot/summary.csv`.

| Variant | Teacher-probe gap (objective only) | B-logistic gap | B-MLP gap ↓ | GoT relative error |
|---|---|---|---|---|
| Base | .265 ± .015 | .236 ± .014 | .204 ± .030 | .034 |
| Reconstruction-only | .277 ± .025 | .244 ± .028 | **.107 ± .020** | .006 |
| Random teacher | .288 ± .020 | .242 ± .011 | .165 ± .019 | .016 |
| Truth-aware | **.089 ± .028** | .205 ± .017 | .189 ± .030 | .026 |

* **The pre-registered success criterion was NOT met:** truth-aware is worse than reconstruction-only on the MLP probe on every seed.
* The B-logistic probe has cosine 0.64 with the teacher, so it is only partly independent.

### 7c. Method A: GoT-A reconstruction + OWT replay (VALIDATION; mean over 3 seeds; OWT L0 matched within 5 %)
Source: `results/geometry_of_truth/method_a_layer6/summary.csv`.

| Architecture | Variant | B-MLP gap, geographic ↓ | B-logistic gap, geographic | GoT relative error ↓ | OWT EV | OWT CE increase ↓ | L0 OWT / GoT-geo / RepE |
|---|---|---|---|---|---|---|---|
| ReLU | Base | .321 ± .022 | .264 | .059 | .828 | .177 | 134 / 66 / 163 |
| ReLU | OWT-only control | .319 ± .022 | .262 | .058 | .829 | .175 | 134 / 66 / 164 |
| ReLU | Method A | .245 ± .053 | .269 | .017 | .828 | .175 | 134 / 68 / 164 |
| TopK | Base | .204 ± .030 | .236 | .034 | .853 | .104 | 128 / 128 / 128 |
| TopK | OWT-only control | .206 ± .035 | .239 | .033 | .855 | .102 | 128 / 128 / 128 |
| TopK | Method A | .107 ± .020 | .244 | .006 | .854 | .102 | 128 / 128 / 128 |
| JumpReLU | Base | .292 ± .026 | .266 | .053 | .862 | .093 | 131 / 46 / 179 |
| JumpReLU | OWT-only control | .287 ± .027 | .263 | .052 | .864 | .091 | 131 / 46 / 179 |
| JumpReLU | Method A | .238 ± .056 | .275 | .015 | .863 | .091 | 131 / 43 / 179 |

* **The MLP gap fell on 18/18 paired seed comparisons, but the pre-registered "> 2 sd" criterion was NOT met** for any architecture.
* Guards pass.
* RepE is unchanged.
* P1-JumpReLU base (seed 42, descriptive): MLP gap .191 → .222 after Method A.

### 7d. E2E-inspired fine-tuning (VALIDATION; mean over 3 seeds)
Source: `results/geometry_of_truth/e2e_inspired_layer6/summary.csv`.

* **Not an exact reproduction of Braun et al.**
* **B and C for ReLU/JumpReLU are NOT L0-matched** (OWT L0 +5 to 11 %). Do not credit their gains.

| Architecture | Variant | OWT L0 | OWT KL ↓ | OWT CE increase ↓ | GoT statement KL ↓ | Downstream NMSE L7–11 ↓ | B-MLP gap, geographic ↓ | RepE mass-mean gap ↓ |
|---|---|---|---|---|---|---|---|---|
| ReLU | A | 134 | .191 | .175 | .165 | .131 | .245 | .154 |
| ReLU | B | **141** | .156 | .145 | .045 | .122 | .280 | .151 |
| ReLU | C | **145** | .152 | .141 | .041 | .118 | .286 | .149 |
| TopK | A | 128 | .115 | .102 | .110 | .094 | .107 | .114 |
| TopK | B | 128 | .102 | .091 | .038 | .093 | .114 | .122 |
| TopK | C | 128 | .102 | .091 | .036 | .092 | .117 | .123 |
| JumpReLU | A | 131 | .106 | .091 | .113 | .091 | .238 | .084 |
| JumpReLU | B | **140** | .087 | .076 | .024 | .084 | .281 | .081 |
| JumpReLU | C | **145** | .085 | .074 | .022 | .081 | .289 | .081 |

**Pre-registered outcome:**
* **TopK B and C: behavioural success** (OWT KL and GoT KL lower than A on every seed by > 2 sd; guards pass).
* **ReLU and JumpReLU: fail the L0 guard.**
* **Truth hypothesis (MLP gap below A): fails everywhere.**
* **Seed sd:** behavioural ≤ .004; MLP gap .01–.05.

## 8. Sparsity-control pilot P1 (VALIDATION, seed 42 only)
Source: `results/openwebtext/p1_sparsity_control_pilot/baseline_vs_p1_table.csv`.

| Layer, architecture | L0 baseline → P1 | EV baseline → P1 | Dead latents baseline → P1 | Note |
|---|---|---|---|---|
| 0 JumpReLU | 74.6 → 111.8 | .703 → .815 | .957 → .904 | Still unmatched; coefficient at floor |
| 2 JumpReLU | 131.5 → 130.5 | .936 → .956 | .655 → .206 | Matched; improved |
| 6 JumpReLU | 131.6 → 128.8 | .862 → .866 | .204 → .093 | Better with fewer latents |
| 10 JumpReLU | 132.4 → 128.8 | .818 → .815 | .042 → .087 | Trade-off |
| ReLU (all four layers) | 134–140 → 129–134 | −.001 to −.003 | ≈ | Closer to target; trade-off |
| TopK (all four layers) | identical | identical | identical | Bit-identical (pairing check) |

**Dead latents at baseline** (OWT validation, 12,700 tokens):
* JumpReLU: 40–85 % at layers 0–4.
* TopK: 7–44 %.
* ReLU: ≤ 5 %.

## 9. KL-weight ablation, layer 6 TopK (VALIDATION; L0 = 128 on OWT, GoT and RepE)
Source: `results/geometry_of_truth/kl_weight_ablation_layer6_topk/summary.csv`, `per_sae.csv`.

* w = fraction of Method B's KL coefficient α_B (per seed 4.150 / 3.892 / 4.212); w = 0 is Method A, w = 1 is Method B.
* Seed values are 42 / 1042 / 2042.

| Metric | A (w=0) | w=0.25 | w=0.50 | B (w=1) |
|---|---|---|---|---|
| GoT geographic B-MLP gap ↓ (seeds) | .112 / .085 / .124 | .100 / .106 / .128 | .095 / .113 / .127 | .097 / .106 / .139 |
| GoT geographic B-MLP gap ↓ (mean ± sd) | .107 ± .020 | .111 ± .015 | .112 ± .016 | .114 ± .022 |
| OWT output KL ↓ | .1152 ± .0019 | .1038 ± .0016 | .1027 ± .0014 | .1020 ± .0014 |
| OWT spliced CE increase ↓ | .1019 ± .0026 | .0928 ± .0026 | .0915 ± .0024 | .0910 ± .0027 |
| GoT statement KL ↓ | .110 | .038 | .038 | .038 |
| GoT numerical B-MLP gap ↓ | .059 | .059 | .066 | .074 |
| RepE mass-mean gap ↓ | .114 | .118 | .117 | .122 |
| RepE persona-shift cosine ↑ | .770 | .765 | .760 | .743 |
| OWT EV ↑ | .8544 | .8534 | .8528 | .8520 |
| GoT relative reconstruction error ↓ | .0062 | .0070 | .0078 | .0094 |
| Downstream NMSE, layers 7–11 ↓ | .0944 | .0923 | .0924 | .0926 |

**Share of B's OWT KL gain retained:** 0.85–0.87 at w = 0.25; 0.94–0.95 at w = 0.50.

**Pre-registered verdict: INCONCLUSIVE.**
* The endpoints differ in truth gap by only +0.007, below the 0.02 threshold.
* BEHAV and truth non-inferiority pass; TRADE fails.
* Paired sd of the truth gap is about 0.02, larger than every between-method difference.

## 10. R1: RepE honesty-preservation fine-tuning, layer 6 (VALIDATION; mean over 3 seeds; statement 326 excluded)
Source: `results/repe/r1_honesty_preservation_layer6/summary.csv`, `per_sae.csv`, `criteria.json`.

Original GPT-2: B-MLP AUROC 0.964 (cross-wording 0.928), B-logistic 0.993, B-mass-mean 0.894; pair accuracy 1.0.
Gaps = original − reconstruction AUROC (↓ better).

| Arch | Variant | val B-MLP gap ↓ | cw B-MLP gap ↓ | val B-logistic gap ↓ | pair-dir cos ↑ | RepE val EV ↑ | RepE val L0 | OWT val EV ↑ | OWT ΔCE ↓ |
|---|---|---|---|---|---|---|---|---|---|
| ReLU | base | .198 | .243 | .258 | .710 | .785 | 163.9 | .828 | .177 |
| ReLU | R-recon | .180 | .236 | .244 | .728 | .824 | 177.7 | .826 | .178 |
| ReLU | R-samepersona | .175 | .236 | .237 | .729 | .828 | 190.2 | .825 | .175 |
| ReLU | R-honesty | .167 | .225 | .244 | .915 | .820 | 175.1 | .826 | .178 |
| TopK | base | .092 | .134 | .089 | .799 | .803 | 128 | .853 | .104 |
| TopK | R-recon | .088 | .117 | .076 | .811 | .832 | 128 | .853 | .104 |
| TopK | R-samepersona | .091 | .116 | .080 | .811 | .832 | 128 | .852 | .105 |
| TopK | R-honesty | .101 | .128 | .086 | .870 | .828 | 128 | .852 | .108 |
| JumpReLU | base | .072 | .109 | .068 | .838 | .845 | 179.4 | .862 | .093 |
| JumpReLU | R-recon | .067 | .094 | .056 | .850 | .873 | 185.0 | .860 | .094 |
| JumpReLU | R-samepersona | .062 | .091 | .054 | .858 | .877 | 193.0 | .861 | .092 |
| JumpReLU | R-honesty | .075 | .096 | .054 | .915 | .869 | 181.1 | .861 | .093 |

**Primary paired difference R-honesty − R-recon (val B-MLP gap; seeds 42/1042/2042):** ReLU −.010/−.012/−.019;
TopK −.005/+.016/+.028; JumpReLU +.008/+.010/+.007.

**Verdicts:** ReLU INCONCLUSIVE (R1b and transfer met; RepE-prompt L0 guard failed); TopK NOT SUPPORTED (also GoT
guard failed for R-honesty); JumpReLU NOT SUPPORTED. R1a (R-recon < base, consistent) not met anywhere. Method A
(GoT fine-tuning) changes RepE gaps by ≤ .008. Exploratory TopK bases at k = 164 / 180: val gap .082 / .075.

## 11. R2: SAE-in-the-loop clamp of honesty features, layer 6 (VALIDATION, descriptive; mean ± sd over 3 seeds; base SAEs; statement 326 excluded)
Source: `results/repe/r2_sae_in_the_loop_steering_layer6/validation_summary.csv`, `validation_criteria_descriptive.json`, `k_selection.json`.

E = 1 − ΣKL(p_H‖p_U′)/ΣKL(p_H‖p_U) (↑ better; 1 = honest output fully recovered). Positive control (activation patch) E = 0.188 (`val`), 0.404 (`cw`).
Baseline mean KL(p_H‖p_U) = 0.020 nats. α = 1 is a full clamp to the honest-train mean, with the error term kept.

| Arch | k | E(α=1) | E(α=−1) | C2 mean / q95 | C3 q95 | E unrelated | logistic probe shift ↑ | OWT ΔCE target / C2 q95 ↓ | cw E(α=1) |
|---|---|---|---|---|---|---|---|---|---|
| ReLU | 1* | −0.110 ± .006 | −0.058 | −0.012 / .004 | .009 | −0.005 | .146 | .017 / .001 | −0.083 |
| ReLU | 4 | −0.127 ± .015 | −0.087 | −0.044 / .005 | .009 | −0.034 | .221 | .019 / .002 | −0.106 |
| ReLU | 16 | −0.178 ± .038 | −0.258 | −0.145 / −.011 | −.016 | −0.251 | .357 | .034 / .006 | −0.098 |
| TopK | 1* | −0.103 ± .039 | −0.070 | −0.027 / .004 | .005 | −0.062 | .311 | .029 / .002 | −0.039 |
| TopK | 4 | −0.164 ± .026 | −0.226 | −0.124 / −.007 | −.046 | −0.176 | .453 | .079 / .010 | −0.086 |
| TopK | 16 | −0.532 ± .045 | −0.531 | −0.470 / −.113 | −.250 | −0.537 | .643 | .135 / .028 | −0.227 |
| JumpReLU | 1* | −0.115 ± .031 | −0.057 | −0.045 / .003 | .005 | −0.033 | .440 | .043 / .005 | −0.059 |
| JumpReLU | 4 | −0.287 ± .092 | −0.288 | −0.142 / −.013 | −.054 | −0.419 | .502 | .094 / .014 | −0.109 |
| JumpReLU | 16 | −0.854 ± .349 | −0.745 | −0.647 / −.195 | −.329 | −0.877 | .600 | .176 / .040 | −0.280 |

\* k selected on validation (argmax of mean E(α = 1)); optimistic.

* No-error reconstruction splice: E = −4.27 (ReLU), −3.75 (TopK), −2.83 (JumpReLU).
* **Descriptive verdict: NOT SUPPORTED, all architectures, all k.** Criterion 1 fails (BH q = 1), as do 3, 4 and 5. Criterion 2 passes but is uninformative.
* R1 SAEs (exploratory): R-honesty − R-recon E(α = 1) at k = 1: ReLU +0.028 ± 0.060, TopK +0.002 ± 0.003, JumpReLU −0.001 ± 0.001.

## 12. M0: within-layer reanalysis of existing results (VALIDATION; 107 matched SAEs; no new model runs)
Source: `results/shared/m0_within_layer_reanalysis/` (README §3).

**Primary model:** outcome ~ β·log error + layer FE + architecture FE + log L0, with a Freedman–Lane within-layer permutation (5000).

| Predictor → outcome | β [95 % CI] | p (Holm) | Secondary model (no arch FE) β, p | Verdict |
|---|---|---|---|---|
| OWT NMSE → GoT logistic gap (layers 4–11) | 0.268 [0.056, 0.477] | 0.044 | −0.005, 0.89 | Formal pass, **fragile** (architecture = 99.8 % of within-layer NMSE variance) |
| GoT relative error → GoT logistic gap | 0.005 [−0.059, 0.080] | 1.00 | 0.020, 0.25 | NOT SUPPORTED |
| OWT NMSE → OWT splice ΔCE | 0.165 [0.134, 0.265] | 0.0008 | 0.202, 0.0002 | **SUPPORTED** |
| GoT relative error → OWT splice ΔCE | −0.012 [−0.060, 0.028] | 1.00 | −0.045, 0.002 | NOT SUPPORTED |

**Architecture (OWT validation):**
* **Layers 4–11:** JumpReLU < TopK < ReLU in NMSE (p ≤ 0.003); TopK ≈ JumpReLU at layers 10–11.
* **Layers 0–2:** ReLU is lowest; ReLU ≈ JumpReLU at layer 2.
* **L0:** ReLU uses 4–7 % more latents than TopK. An L0 adjustment (sweep elasticity ≈ −0.4) changes no ranking except two near-ties.
* **Task-prompt L0 is unmatched:** on GoT, TopK 128, ReLU 8–110, JumpReLU 12–92.

**Corrections:**
* The pooled ρ = 0.89 between EV and RepE persona shift is ρ = 0.05 within layer.
* The splice-ratio range is 0.76–2.89 per run (0.82–2.65 per cell), not "1.0–2.6×".

## 13. M1: RepE-LM equal-length persona task, gates K1–K7 (TRAIN gates; validation descriptive)
Source: `results/repe/repe_lm_gates/` (README §4). 506 statements × 20 one-token-swap variants.

| Gate | Status | Key number |
|---|---|---|
| K1 construction | PASS | 6,072 pairs; pre-slot residual diff 0.0 at all 12 layers |
| K2 shortcuts | PASS | every shortcut AUROC 0.5000 (max deviation 5.6e-17) |
| K3 positive control | PASS | 1.000 |
| K4 neutral floor | reported | transfer floor 0.50–0.57; neutral decodability 0.58–0.75 |
| K5 semantic transfer | PASS (narrow) | leave-one-pair-out 0.701 at L8 and L11 (CI low 0.689 / 0.691) |
| K6 leakage | PASS | 0 same-label near-duplicates; 14 counterfactual edits disclosed |
| K7 behaviour | PASS | median KL ratio 1.99; Wilcoxon p 3e-68; median KL(H‖U) 0.019 nats |

## 14. M2: four-level information faithfulness (VALIDATION; 108 SAEs + 36 PCA; GoT cities, RepE-LM persona and neutral)
Source: `results/shared/m2_four_level_faithfulness/` (README §2). Gaps are the mean over informative layers and seeds (AUROC; lower is better).

| Task | Arch | Frozen logistic | Noise | Frozen mass-mean | Re-trained logistic | PCA re-trained | MLP | Error-term AUROC |
|---|---|---|---|---|---|---|---|---|
| GoT | ReLU / TopK / JumpReLU | 0.287 / 0.275 / 0.286 | 0.259 / 0.246 / 0.257 | 0.033 / 0.024 / 0.026 | 0.066 / 0.075 / 0.118 | 0.042 / 0.032 / 0.030 | 0.049 / 0.037 / 0.066 | 0.675 / 0.615 / 0.632 |
| Persona | ReLU / TopK / JumpReLU | 0.198 / 0.162 / 0.155 | 0.076 / 0.072 / 0.064 | 0.091 / 0.066 / 0.062 | 0.179 / 0.147 / 0.143 | 0.131 / 0.116 / 0.114 | 0.232 / 0.190 / 0.177 | 0.773 / 0.719 / 0.721 |
| Neutral | ReLU / TopK / JumpReLU | 0.113 / 0.098 / 0.094 | 0.048 / 0.046 / 0.043 | 0.049 / 0.038 / 0.035 | 0.099 / 0.090 / 0.083 | 0.066 / 0.064 / 0.062 | 0.033 / 0.031 / 0.031 | 0.636 / 0.615 / 0.619 |

* **Labels:**
  * 0 Preserved and 0 Readout shifted on every task;
  * GoT: 21 "Not recovered by tested families" (layers 5–9), 60 Reduced;
  * persona: 99 Reduced; neutral: 90 Reduced;
  * displacement flag in 97.5–100 % of informative cells.
* **H2a (SAE > noise):** GoT 40/81 (excess 0.026), persona 99/99 (0.100), neutral 90/90 (0.058).
* **H2b (vs PCA):** SAE better in 0 cells; worse in 52/81, 59/99 and 81/90.
* **Persona − neutral** re-trained gap: +0.07 to +0.13 at layers 3–8; ≈ 0 at layers 10–11.

## 15. M3: causal site sweep in original GPT-2 (TRAIN selection, VALIDATION confirmation)
Source: `results/repe/causal_site_sweep/` (README §2).
* **Sanity checks:** all 0.0.
* **Baseline:** KL(p_H‖p_U) = 0.033 nats.
* **Passing cells:** 50/65.
* **Share of the effect restored (validation):**
  * persona slot: 0.97 (L1) → 0.11 (L11);
  * final token: 0.03 (L1) → 0.95 (L11);
  * handover at layers 7–8;
  * post-slot instruction: up to 0.28.
* **Selected and confirmed sites:**
  * L1 slot: R_A 0.0356 [0.0278, 0.0450];
  * L11 final: 0.0350 [0.0276, 0.0436].
* **Synonym control:** R_S ≈ 0.5 R_A at every site.
* **T1:** no persona-specific excess on validation.

## 16. M4: SAE causal-faithfulness decomposition (VALIDATION; T2–T4 primary; 9 SAEs per site)
Source: `results/repe/sae_patch_decomposition/` (README §2).

| Site | T_full (nats) | Preserved by reconstruction, ReLU / TopK / JumpReLU | Error alone | \|I\|/T_full |
|---|---|---|---|---|
| L1 slot | 0.0435 | 0.94 / 0.93 / 0.97 | 0.14 / 0.16 / 0.03 | 0.10–0.14 |
| L8 final (A1) | 0.0220 | 0.62 / 0.62 / 0.66 | 0.45 / 0.09 / 0.11 | 0.12 / 0.32 / 0.26 |
| L11 final | 0.0430 | 0.68 / 0.62 / 0.68 | 0.40 / 0.06 / 0.11 | 0.16 / 0.38 / 0.28 |

* **Checks:** all exact; T_full equals M3.
* **Synonym:** similar preservation; T_rec − S_rec > 0 at every site. T1 ≈ synonym.
* **Features:** k* = 64 for every SAE; 1 feature = 2–10 % of T_rec; 64 = 47–74 %; beats matched random sets (p = 0.005).
* **H4a/b/c:** pass 9/9. Additive reading is not allowed for TopK/JumpReLU at L8/L11.

## 17. IMP-L: Method A and output-KL fine-tuning at layers 3, 8, 9 (VALIDATION; 3 seeds; paired; 95 % t-interval, df = 2)
Source: `results/geometry_of_truth/multilayer_finetune/` (README §2). Lower is better for gaps and KL.

**Method A (truth):**

| Layer | SAE | Truth gap (B-MLP) base / OWT-only / Method A / output-KL (mean ± sd) | Method A − base [95 % CI] | GoT rel. error, Method A vs base | Re-trained geo gap, Method A − base [95 % CI] | Method A truth class |
|---|---|---|---|---|---|---|
| 3 | ReLU | 0.293 ± 0.008 / 0.292 ± 0.008 / 0.275 ± 0.002 / 0.287 ± 0.003 | -0.018 [-0.041, +0.005] | -62.0 % | +0.010 [-0.028, +0.048] | NO IMPROVEMENT |
| 3 | TopK | 0.227 ± 0.008 / 0.227 ± 0.011 / 0.129 ± 0.007 / 0.155 ± 0.017 | -0.098 [-0.107, -0.089] | -74.0 % | -0.005 [-0.104, +0.094] | PRELIMINARY IMPROVEMENT |
| 3 | JumpReLU | 0.294 ± 0.015 / 0.295 ± 0.015 / 0.282 ± 0.021 / 0.298 ± 0.009 | -0.012 [-0.043, +0.019] | -61.1 % | +0.002 [-0.026, +0.031] | NO IMPROVEMENT |
| 8 | ReLU | 0.234 ± 0.047 / 0.228 ± 0.050 / 0.180 ± 0.062 / 0.180 ± 0.062 | -0.054 [-0.095, -0.013] | -73.5 % | +0.006 [-0.024, +0.035] | PRELIMINARY IMPROVEMENT |
| 8 | TopK | 0.157 ± 0.057 / 0.178 ± 0.048 / 0.083 ± 0.030 / 0.089 ± 0.005 | -0.074 [-0.202, +0.054] | -85.4 % | -0.032 [-0.082, +0.019] | NO IMPROVEMENT |
| 8 | JumpReLU | 0.260 ± 0.052 / 0.260 ± 0.044 / 0.200 ± 0.040 / 0.212 ± 0.034 | -0.060 [-0.247, +0.127] | -79.6 % | -0.005 [-0.023, +0.014] | NO IMPROVEMENT |
| 9 | ReLU | 0.214 ± 0.021 / 0.210 ± 0.021 / 0.167 ± 0.015 / 0.199 ± 0.003 | -0.047 [-0.065, -0.030] | -72.6 % | +0.010 [-0.011, +0.031] | PRELIMINARY IMPROVEMENT |
| 9 | TopK | 0.160 ± 0.031 / 0.158 ± 0.041 / 0.127 ± 0.021 / 0.136 ± 0.033 | -0.033 [-0.061, -0.005] | -85.4 % | +0.005 [-0.034, +0.043] | NO IMPROVEMENT |
| 9 | JumpReLU | 0.258 ± 0.026 / 0.262 ± 0.024 / 0.152 ± 0.005 / 0.174 ± 0.023 | -0.106 [-0.162, -0.050] | -78.7 % | +0.002 [-0.039, +0.043] | PRELIMINARY IMPROVEMENT |

**Output-KL (behaviour and truth):**

| Layer | SAE | OWT KL base → Method A → output-KL | output-KL − Method A [95 % CI] | GoT stmt KL, output-KL − Method A | OWT ΔCE, output-KL vs Method A | OWT L0, output-KL vs base | Truth gap, output-KL − Method A | Behaviour class | Truth class | Failed guards |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | ReLU | 0.0940 → 0.0933 → 0.0811 | -0.0121 [-0.0140, -0.0102] | -0.050 | -10.4 % | +3.5 % | +0.012 [-0.000, +0.024] | PRELIMINARY IMPROVEMENT | REGRESSION | — |
| 3 | TopK | 0.0673 → 0.0663 → 0.0609 | -0.0054 [-0.0058, -0.0049] | -0.036 | -4.3 % | +0.0 % | +0.027 [-0.001, +0.054] | PRELIMINARY IMPROVEMENT | REGRESSION | — |
| 3 | JumpReLU | 0.0541 → 0.0530 → 0.0445 | -0.0086 [-0.0101, -0.0071] | -0.037 | -13.5 % | +4.6 % | +0.016 [-0.017, +0.048] | PRELIMINARY IMPROVEMENT | NO IMPROVEMENT | — |
| 8 | ReLU | 0.2431 → 0.2422 → 0.1998 | -0.0424 [-0.0443, -0.0406] | -0.151 | -17.1 % | +5.8 % | -0.000 [-0.037, +0.037] | NOT CREDITABLE (guard failed) | NOT CREDITABLE (guard failed) | l0_within_5pct_of_base |
| 8 | TopK | 0.1439 → 0.1417 → 0.1270 | -0.0147 [-0.0157, -0.0137] | -0.078 | -11.3 % | +0.0 % | +0.006 [-0.065, +0.076] | PRELIMINARY IMPROVEMENT | NO IMPROVEMENT | — |
| 8 | JumpReLU | 0.1360 → 0.1339 → 0.1120 | -0.0220 [-0.0240, -0.0199] | -0.103 | -18.5 % | +8.1 % | +0.011 [-0.065, +0.087] | NOT CREDITABLE (guard failed) | NOT CREDITABLE (guard failed) | l0_within_5pct_of_base |
| 9 | ReLU | 0.2713 → 0.2698 → 0.2256 | -0.0443 [-0.0453, -0.0432] | -0.202 | -16.3 % | +5.3 % | +0.033 [-0.006, +0.071] | NOT CREDITABLE (guard failed) | NOT CREDITABLE (guard failed) | l0_within_5pct_of_base |
| 9 | TopK | 0.1610 → 0.1588 → 0.1435 | -0.0154 [-0.0164, -0.0143] | -0.081 | -8.9 % | +0.0 % | +0.010 [-0.024, +0.043] | PRELIMINARY IMPROVEMENT | NO IMPROVEMENT | — |
| 9 | JumpReLU | 0.1538 → 0.1523 → 0.1285 | -0.0239 [-0.0259, -0.0218] | -0.112 | -15.0 % | +8.1 % | +0.022 [-0.024, +0.068] | NOT CREDITABLE (guard failed) | NOT CREDITABLE (guard failed) | l0_within_5pct_of_base |

* **Base SAEs:** OWT val EV 0.904–0.914 (L3), 0.800–0.840 (L8), 0.784–0.829 (L9). L0 ReLU 134–135, TopK 128,
  JumpReLU 131–132.
* **Criteria met:** T_method_a 4/9; B_kl 5/9 (4/9 not creditable, L0 guard); T_kl 0/9 (2 regressions at layer 3).


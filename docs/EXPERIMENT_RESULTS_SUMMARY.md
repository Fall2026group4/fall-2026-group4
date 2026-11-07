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

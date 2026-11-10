# R2: data-scaled SAE retraining for GPT-2 Small (layers 1, 8, 11). Status: COMPLETED (VALIDATION, 2026-11-10); TEST in M5

Evidence level: **VALIDATION** for development and comparisons. OWT **TEST** is used only in the frozen M5 pass.
Approved by the user on 2026-11-10 ("You have approval to run R1, R2 and other scientifically justified GPT-2
improvement experiments").

## 1. Before the full run

### 1.1 Problem and hypothesis

* **Problem:** our SAEs were trained on 101,600 unique OWT tokens (about 38 passes) at width 3072
  (`docs/2026-11-10_independent_research_review.md`, W1). Published GPT-2 SAEs use about 300M tokens.
* **Hypotheses:**
  * **H-R2a:** SAEs trained on about 100M fresh tokens with published recipes reconstruct better and damage the model
    less, at the same L0 (128).
  * **H-R2b (the scientific question):** do information, representation and causal faithfulness improve with them,
    or do they stay flat while reconstruction improves? Both directions are reported.

### 1.2 Training (`scripts/openwebtext/17_streaming_sae_training.py`, `src/sae/modern_sae.py`)

* **Data:**
  * OpenWebText parquet shard 78: 100,074 documents after excluding 98 that match our 1000-document sample (exact
    and normalised-prefix hash); 105.9M non-BOS tokens in 127-token chunks with BOS.
  * Shard 79 is used only as a training monitor.
  * Our OWT validation and test caches are never read by training.
* **Activations:** `blocks.{1,8,11}.hook_resid_pre`, BOS excluded, from one GPT-2 forward pass into a shuffled fp16
  GPU buffer (2^19 tokens per layer). Each seed draws its own batches (with replacement from the buffer), and the
  architectures within a seed share batches.
* **Budget and optimiser:**
  * 24,414 steps × 4096 tokens = **100M tokens per SAE**.
  * Adam, lr 2e-4, 500-step warm-up, linear decay over the last 20 %, gradient clip 1.0, bf16 autocast forward,
    unit-norm decoder columns.
  * Inputs scaled to E‖x‖² = 768; the scale is folded into the saved weights.
* **Initialisation:** encoder = decoderᵀ; decoder bias = data mean.
* **SAEs per layer:**

  | Name | Architecture | Width | Seeds | Sparsity |
  |---|---|---|---|---|
  | `topk` | TopK + AuxK (Gao et al. 2024; k_aux 512, α 1/32, dead after 5M tokens) | 12,288 | 42, 1042, 2042 | k = 128 |
  | `batchtopk` | BatchTopK + AuxK (Bussmann et al. 2024), global inference threshold | 12,288 | 3 | mean k = 128 |
  | `relu` | ReLU + L1 on decoder-norm-weighted activations | 12,288 | 3 | log-λ controller to L0 128, λ ∈ [1e-3, 1e3] |
  | `topk_w3072` | TopK + AuxK (width ablation) | 3,072 | 3 | k = 128 |
  | `topk_frozen_decoder` | TopK, random decoder frozen (random baseline, Korznikov et al. 2026) | 12,288 | 42 | k = 128 |

* **Checkpoints:** after 10M, 25M and 50M tokens, and at the end (data-scaling curve).
* **Pilot disclosure** (layer 8, seed 42, 10M tokens, monitor shard only plus one sanity read of OWT validation EV
  for the loader check):
  * mechanics verified; the scale-folding loader reproduces the monitor EV;
  * **a modern JumpReLU (STE) was attempted and abandoned:**
    * the published threshold init (0.001) could not move in our budget (L0 ≈ 1660);
    * a quantile init plus ε = 0.05 left L0 oscillating (68–506) or collapsing (19) with λ at its floor;
    * a λ sweep was not affordable.
    * Our old JumpReLU SAEs remain in the comparison set; BatchTopK replaces JumpReLU as the modern threshold-style
      architecture.

### 1.3 Evaluation ladder (`scripts/shared/sae_ladder_evaluation.py`, written before any R2 checkpoint is evaluated)

All SAEs at a layer (old base × 9, R2 × 13, res-jb where available) go through the same code.

| Level | Metric (direction) | Data |
|---|---|---|
| Reconstruction | NMSE (↓), EV (↑), cosine (↑), L0 (match ±10 % of 128), dead fraction (↓) | OWT validation cache (12,700 tokens) |
| Behaviour | spliced ΔCE (↓), KL (↓), damage relative to same-norm noise | OWT validation |
| Information | frozen logistic gap (↓) with noise reference; re-trained logistic gap (↓); SAE minus PCA re-trained gap at the SAE's own NMSE (↓, < 0 = beats PCA) | GoT cities (informative layers 8, 11) and RepE-LM persona / neutral (M2 definitions, imported) |
| Representation | per-pair cos(Δ, Δ̂) (↑), norm ratio, held-out wording gap (↓) | RepE-LM (RLM-RP definitions, imported) |
| Causal | T_rec/T_full (↑), T_err/T_full, |I|/T_full; H4a vector controls | M4 site of the layer: L1 persona slot, L8 final, L11 final (M4 functions, imported) |

* L0 is reported on OWT, GoT and RepE-LM.
* **Reproduction checks:** the old base SAEs must reproduce M2 per-cell values (frozen, re-trained gaps, task
  NMSE), the splice summary per run, and M4's deterministic columns.

### 1.4 Pre-registered comparisons and decision rule

* **Comparisons, per layer:**
  * **C1:** new TopK-12288 vs old TopK-3072.
  * **C2:** new ReLU-12288 vs old ReLU-3072.
  * **C3:** new BatchTopK-12288 vs old JumpReLU-3072 (the best old reconstructer).
  * **A1 (data + recipe, same width):** new TopK-3072 vs old TopK-3072.
  * **A2 (width):** new TopK-12288 vs new TopK-3072.
  * **B (random baseline):** trained TopK-12288 seed 42 vs frozen-decoder TopK-12288 seed 42 (descriptive).
* **Rule per comparison × metric × layer:**
  * **IMPROVEMENT:** every new seed beats every old seed (9/9 pairs) **and** the Welch 95 % interval of the mean
    difference excludes 0.
  * **REGRESSION:** the mirror image.
  * Otherwise **NO CLEAR CHANGE**.
* **Sparsity:**
  * An SAE with OWT validation L0 outside 128 ± 10 % is excluded from comparisons. It is reported, not deleted.
  * TopK is exact. BatchTopK and ReLU are checked.
* **Reporting:** absolute and % change, mean ± sd, per-seed values. No multiplicity correction; the number of cells is
  stated.
* **M5 (later, frozen protocol):** the C1–C3 reconstruction and behaviour metrics on the OWT test split, once.

## 2. Results (VALIDATION, 2026-11-10)

### 2.1 Training run

* **Training:** 39 SAEs × 99,999,744 tokens, OWT shard 78. 156 checkpoints (10M, 25M, 50M, final), gitignored. Runtime
  5.5 h on one A10G. GPT-2 parameters unchanged (hash before = after). `training_info.json` holds the script and
  module hashes.
* **L0 on OWT validation, all within ±10 % of 128; no SAE excluded (`excluded_unmatched.csv` is empty):**
  * TopK: exact.
  * BatchTopK: 131.7–134.2.
  * ReLU (L0 controller): 130.5–134.4.
* **Task L0 is not matched, for old or new SAEs.** On RepE-LM and GoT prompts, BatchTopK and ReLU fire 141–216 latents
  at L1/L8, against 128 on OWT. TopK is exact everywhere. Ladder column `*__task_l0`.
* **Evaluation interruption:** the ladder runs for L11 and L1 were OOM-killed once, by concurrent jobs, and rerun from
  scratch (RESEARCH_PROGRESS §34). The harness is deterministic.

### 2.2 Before → after (`comparisons.csv`, 3 seeds each, mean ± sd; rule of 1.4)

Improvement % = 100 × (old − new)/old for ↓ metrics, and 100 × (new − old)/old for ↑ metrics.

| Layer | Metric | Old TopK-3072 | New TopK-12288 | Δ (95 % Welch CI) | Improvement | C1 class |
|---|---|---|---|---|---|---|
| L1 | OWT NMSE ↓ | 0.0811 ± 0.0004 | 0.0255 ± 0.0001 | −0.0556 (−0.0567, −0.0545) | 68.5 % | IMPROVEMENT |
| L1 | spliced ΔCE ↓ (nats) | 0.0256 ± 0.0015 | 0.0063 ± 0.0008 | −0.0193 (−0.0224, −0.0161) | 75.3 % | IMPROVEMENT |
| L1 | persona re-fit gap ↓ | 0.083 ± 0.012 | 0.067 ± 0.003 | −0.017 (−0.045, +0.012) | 19.8 % | NO CLEAR CHANGE |
| L1 | pair-direction cosine ↑ | 0.373 ± 0.003 | 0.507 ± 0.003 | +0.135 (+0.128, +0.141) | 36.1 % | IMPROVEMENT |
| L1 | T_rec/T_full ↑ | 0.931 ± 0.014 | 0.988 ± 0.005 | +0.057 (+0.025, +0.088) | 6.1 % | IMPROVEMENT |
| L8 | OWT NMSE ↓ | 0.1643 ± 0.0005 | 0.0959 ± 0.0001 | −0.0684 (−0.0696, −0.0671) | 41.6 % | IMPROVEMENT |
| L8 | spliced ΔCE ↓ | 0.1259 ± 0.0043 | 0.0603 ± 0.0018 | −0.0656 (−0.0747, −0.0565) | 52.1 % | IMPROVEMENT |
| L8 | GoT cities re-fit gap ↓ | 0.061 ± 0.015 | 0.055 ± 0.022 | −0.006 (−0.051, +0.040) | 9.0 % | NO CLEAR CHANGE |
| L8 | persona re-fit gap ↓ | 0.156 ± 0.017 | 0.108 ± 0.009 | −0.048 (−0.082, −0.014) | 30.7 % | IMPROVEMENT |
| L8 | persona SAE − PCA gap ↓ | 0.033 ± 0.017 | 0.027 ± 0.009 | −0.006 (−0.041, +0.028) | — | NO CLEAR CHANGE |
| L8 | pair-direction cosine ↑ | 0.394 ± 0.003 | 0.488 ± 0.001 | +0.094 (+0.086, +0.103) | 24.0 % | IMPROVEMENT |
| L8 | T_rec/T_full ↑ | 0.625 ± 0.039 | 0.726 ± 0.020 | +0.101 (+0.019, +0.182) | 16.1 % | IMPROVEMENT |
| L11 | OWT NMSE ↓ | 0.1891 ± 0.0005 | 0.1129 ± 0.0002 | −0.0762 (−0.0773, −0.0751) | 40.3 % | IMPROVEMENT |
| L11 | spliced ΔCE ↓ | 0.2129 ± 0.0018 | 0.0995 ± 0.0011 | −0.1134 (−0.1171, −0.1097) | 53.3 % | IMPROVEMENT |
| L11 | GoT cities re-fit gap ↓ | 0.110 ± 0.015 | 0.100 ± 0.031 | −0.010 (−0.073, +0.053) | 9.0 % | NO CLEAR CHANGE |
| L11 | persona re-fit gap ↓ | 0.149 ± 0.015 | 0.109 ± 0.008 | −0.040 (−0.071, −0.008) | 26.7 % | IMPROVEMENT |
| L11 | persona SAE − PCA gap ↓ | 0.083 ± 0.015 | 0.070 ± 0.008 | −0.013 (−0.045, +0.019) | — | NO CLEAR CHANGE |
| L11 | pair-direction cosine ↑ | 0.432 ± 0.001 | 0.513 ± 0.001 | +0.080 (+0.078, +0.083) | 18.6 % | IMPROVEMENT |
| L11 | T_rec/T_full ↑ | 0.620 ± 0.004 | 0.692 ± 0.002 | +0.073 (+0.064, +0.082) | 11.8 % | IMPROVEMENT |

**Other comparisons:**
* **C2, ReLU-12288 vs old ReLU-3072:**
  * NMSE −31 to −65 %; ΔCE −37 to −82 %.
  * Persona re-fit gap: improved at L8 only.
  * T_rec: improved at L1 and L11.
  * **SAE − PCA persona gap: REGRESSION at L1 and L8**, +0.024 and +0.017.
* **C3, BatchTopK-12288 vs old JumpReLU-3072:**
  * NMSE −43 to −74 %; ΔCE −55 to −60 %.
  * Persona re-fit gap: improved at L1 and L8.
  * T_rec: improved at all three layers, e.g. L8 0.665 → 0.754.
* **A1, data and recipe at equal width** (new TopK-3072 vs old TopK-3072):
  * NMSE −23 to −35 %; ΔCE −33 to −38 %.
  * About 55 % of C1's NMSE gain and about 60 % of its ΔCE gain at L8/L11.
* **A2, width** (12,288 vs 3,072, both new): NMSE −20 to −52 %; ΔCE −28 to −60 %.

**Cell counts over all metrics, layers and comparisons** (51 cells per comparison):

| Comparison | IMPROVEMENT | NO CLEAR CHANGE | REGRESSION |
|---|---|---|---|
| C1 | 37 | 14 | 0 |
| C2 | 30 | 18 | 3 |
| C3 | 36 | 15 | 0 |
| A1 | 34 | 17 | 0 |
| A2 | 27 | 21 | 3 |

**Random baseline B** (seed 42, trained vs frozen random decoder):

| Metric | L1 | L8 | L11 |
|---|---|---|---|
| EV, trained / random | 0.975 / 0.676 | 0.904 / 0.593 | 0.887 / 0.587 |
| ΔCE, trained / random | 0.006 / 0.118 | 0.061 / 0.620 | 0.099 / 0.525 |
| T_rec/T_full, trained / random | 0.99 / 0.66 | 0.70 / 0.02 | 0.70 / 0.11 |
| Persona re-fit gap, trained / random | 0.064 / 0.104 | 0.113 / 0.195 | 0.110 / 0.154 |
| GoT re-fit gap, trained / random | — | **0.081 / 0.052** | 0.081 / 0.115 |

At L8 the random decoder has the *smaller* GoT re-fit gap, so that metric does not separate trained from random SAEs
there.

### 2.3 Inference

1. **Reconstruction and behaviour improve substantially at all three layers, on all seeds, for every architecture.**
   * At L8/L11, ΔCE falls by about half: TopK 0.126 → 0.060 and 0.213 → 0.100.
   * About half of the gain comes from data and recipe at equal width (A1), and the rest from width (A2).
2. **Causal persona preservation improves with it.**
   * TopK: T_rec/T_full 0.62 → 0.73 at L8 and 0.62 → 0.69 at L11. BatchTopK reaches 0.75 at L8.
   * This is the first intervention in the project that moves M4 preservation in a pre-registered way.
3. **Persona information improves in absolute terms but not relative to PCA.**
   * The re-trained persona gap falls by 27–31 % at L8/L11 (TopK).
   * The SAE − PCA gap at matched OWT NMSE never improves, and regresses for ReLU.
   * Better reconstruction moves the PCA reference just as much: **the SAEs still do not beat PCA at equal
     reconstruction error** (E2-H3 holds for every SAE).
4. **GoT information shows no clear change.** Its between-seed variance is large, and at L8 even a random decoder
   does as well. The GoT re-trained gap is a weak discriminator at these layers.
5. **Remaining damage is still more harmful than noise.**
   * The new SAEs' splice damage is 1.25× (L8) and 1.61× (L11) that of same-norm random noise (TopK). The old
     SAEs' ratios were 1.45× and 1.96×.
   * The ratio improved but stays above 1: the remaining error is still partly structured, the "dark matter" of
     Engels et al.

### 2.4 Limitations

* VALIDATION only. M5 tests the C1–C3/A1/A2 OWT and GoT metrics once on TEST.
* RepE-LM has no test split.
* Task L0 is not matched for BatchTopK/ReLU, which affects old and new SAEs alike.
* Three layers only. GPT-2 Small. One data source (OWT).
* 100M tokens, about 3× fewer than res-jb.
* No multiplicity correction across 255 cells; the classes require 9/9 pairs **and** a Welch CI excluding 0.

# GPT-2 Small SAE faithfulness: repository audit and evidence-based improvement plan (2026-11-12)

**Status:** analysis and recommendation only. No training, evaluation, script change, checkpoint or commit was made.
Every proposal in §8 needs the user's written approval and its own pre-registration
(`docs/templates/EXPERIMENT_TEMPLATE.md`) before anything runs. Gemma is out of scope (a teammate owns it).

**Evidence labels:**
* **TEST** = M5 confirmatory pass (on previously exposed rows; see D31).
* **VAL** = validation split.
* **HIST** = historical/provenance.
* **INVALID** = never cite.
* **HYPOTHESIS** = not yet tested.

**What is new in this document:**
1. The current state has been re-audited from code and CSVs, not from summaries.
2. A new diagnostic was read from existing ladder CSVs (§5.3): the architectures fail on the honesty contrast in
   **opposite directions**, by shrinking it or inflating it.
3. A literature check covers methods that the 2026-11-10 review did not cover: Gated SAE, ITO, feature-suppression
   fine-tuning, the Anthropic JumpReLU recipe, SAE-TS, output-score feature selection and RAVEL.
4. A pre-registerable experiment plan (§8–§12).

The 2026-11-10 independent review (`docs/2026-11-10_independent_research_review.md`) is the predecessor. Its R1, R2,
R4, R5 and R7 have been run, so this audit starts from the **post-R2 state**.

---

## 1. Repository state and preservation checks

| Check | Finding |
|---|---|
| Path / branch | `~/sae-project/fall-2026-group4`, branch `sae-research` (not switched) |
| HEAD | `446beea` "Final defense verification…" |
| Uncommitted | only untracked `docs/final_defense/SAE_Faithfulness_Summary.pdf` (left untouched); no stash |
| Activation caches | `data/activations/gpt2_small/{train,validation,test}` (2.3 GB); `data/got/activations/`, `data/repe/activations/` |
| Streamed tokens | `data/openwebtext/stream_tokens/shard{78,79}_chunks.npy` (R2/R4 training data) |
| Checkpoints | 1,262 SAE `.pt` files on disk, including R2 `results/openwebtext/r2_scaled_saes/checkpoints/layer_{1,8,11}/<name>_seed<s>/checkpoint_{10M,25M,50M,final}.pt` and R4 `results/openwebtext/r4_kl_finetune/checkpoints/` |
| Manifests | `docs/experiment_manifest.csv`, `docs/checkpoint_registry.csv`, `docs/results_registry.md` |
| Resources | A10G 23 GB (idle), 97 GB disk free, 15 GB host RAM (the OOM constraint from RESEARCH_PROGRESS §34 still applies: one heavy job at a time) |
| Backups | **Risk:** the R2/R4 checkpoints still have no off-machine copy (roadmap §11, P0) |

Nothing was modified apart from three documentation files: this new file, plus pointer entries in
`RESEARCH_ROADMAP.md` and `RESEARCH_PROGRESS.md`.

---

## 2. GPT-2 pipeline audit (from code)

### 2.1 OpenWebText

| Item | Old ("corrected baseline") SAEs | R2 SAEs | R4 |
|---|---|---|---|
| Data | 800/100/100 OWT documents, ≤ 128 tokens (`scripts/openwebtext/01_prepare_dataset.py`) | OWT parquet shard 78, 98 documents matching our sample removed (`17_streaming_sae_training.py:70-76, 90-91`) | shard 79 from chunk 10,000 (`18_kl_finetune_fixed_l0.py:37-39`) |
| Unique train tokens | **101,600** | 105.9M available; 99,999,744 used | 4,998,720 |
| Token exposures | 15,000 × 256 = 3.84M (**≈ 38 passes**) | 100M (≈ 0.94 pass: buffer sampling with replacement, `17_…:308`) | 5M |
| Split | document-level; validation/test 12,700 tokens each | training never reads our validation/test caches | same |
| Hook | `blocks.L.hook_resid_pre`, BOS excluded (`03_training_sae.py:56-69`) | same (`17_…:128`, `[:, 1:]`) | splice at non-BOS positions |
| Width | 3,072 (4×) | 12,288 (16×); TopK-3072 ablation | 12,288 |
| Init | Kaiming encoder; random unit-norm decoder (`03_…:27-29`) | encoder = decoderᵀ, b_dec = data mean (`modern_sae.py:74-76`, `17_…:289`) | from R2 TopK |
| Normalisation | none | inputs scaled so E‖x‖² = 768, folded into saved weights (`17_…:288`, `modern_sae.py:131-136`) | none (already folded) |
| Optimiser | Adam 3e-4, clip 10, decoder renormalised per step | Adam 2e-4, 500 warm-up, linear decay over last 20 %, clip 1, bf16 autocast (`17_…:51, 163-168`) | Adam 2e-5, clip 1 |
| ReLU | L1 on z (decoder already unit norm), multiplicative L0 controller | decoder-norm-weighted L1, log-λ controller η 0.01, λ ∈ [1e-3, 1e3] (`17_…:192-193, 219-220`) | — |
| TopK | k = 128 | k = 128 + AuxK (k_aux 512, α 1/32, dead after 5M tokens) (`17_…:199-210`) | k = 128 fixed |
| BatchTopK | — | top k·B over the batch in training; inference threshold = EMA of the minimum kept activation (`modern_sae.py:97-107`, `17_…:221-225`) | — |
| JumpReLU | **sigmoid surrogate**, not the published STE (`03_…:41-46, 111-113`) | published rectangle-kernel STE implemented (`modern_sae.py:33-61`), but **abandoned in the pilot** (L0 1,660 at the published init; 19–506 oscillation with quantile init + ε 0.05) | — |
| Seeds | 42, 1042, 2042 | same (random decoder: 42 only) | paired with the R2 base |
| Layers | 0–11 | 1, 8, 11 | 8, 11 |
| L0 (OWT test) | 128.0–133.4 (one unmatched: L0 JumpReLU s42, 73.6) | 127.6–131.7 | 128.0 |
| Dead latents | up to 85 % (old JumpReLU, early layers) | ≈ 0–0.3 % | — |

### 2.2 Evaluation (one harness for every SAE)

* **Harness:** `scripts/shared/sae_ladder_evaluation.py`.
  * It imports the unchanged M2, splicing, RLM-RP and M4 code (`:45-49`).
  * The `r2scaling` source exists (`:95-101`) but **was never run**: `results/openwebtext/r2_scaled_saes/scaling/` is empty.
* **Reconstruction:** `src/evaluation/faithfulness_metrics.py`. NMSE ↓, EV ↑, cosine ↑, L0 (match), dead fraction ↓.
* **Behaviour:** `scripts/shared/spliced_reconstruction_evaluation.py`.
  * x̂ replaces x at every non-BOS position; ΔCE ↓, KL(p_orig‖p_spliced) ↓.
  * Same-norm isotropic noise reference; the SAE/noise ratio ↓ (< 1 means more benign than noise).
* **Information:** `scripts/shared/m2_four_level_faithfulness.py` and `src/evaluation/information_levels.py`.
  * Frozen logistic gap ↓: the probe is fit on TRAIN originals only.
  * Re-fit logistic gap ↓.
  * MLP/kNN/sparse-z gaps.
  * PCA reference at the SAE's own OWT-val NMSE (`information_levels.py:157-167`); SAE − PCA ↓, < 0 means it beats PCA.
  * Noise reference.
  * Error-term AUROC (the original probe applied to e alone).
  * Grouped bootstrap by city/statement.
* **Representation (RepE-LM):** `scripts/repe/26_…`. Per-pair cos(Δ, Δ̂) ↑, norm retention ‖Δ̂‖/‖Δ‖ (1 is ideal),
  held-out-wording gap ↓.
* **Causal (M4):** `scripts/repe/24_…`.
  * T = KL(p_H‖p_U) − KL(p_H‖p_patched).
  * T_rec/T_full ↑ (1 is full); controls: synonym, iso-norm and shuffle.
  * Features are ranked by first-order attribution on TRAIN (`24_…:18, 307-357`).
* **Skyline:** `scripts/repe/27_…`. Rank-k DAS, PCA and random subspaces.
* **Measured cost:** about 190 s per SAE for the full ladder at L8 (`ladder_layer8_r2_resjb.json`: 2,656 s / 14 SAEs).

### 2.3 Geometry of Truth (GoT)

* **Data on disk:** `data/got/{cities,neg_cities,larger_than,smaller_than}.csv`.
* **Splits:** grouped 70/15/15 by city or number pair, in `data/got/splits/*.csv`; A/B train groups in
  `train_AB_groups_truth_aware_pilot.csv`.
* **Primary set:** cities, 1,496 statements, balanced; test n = 452 (M5).
* **Excluded:** numerical sets are HIST/diagnostic only, because the second number alone gives AUROC 0.84 (a shortcut).
* **Activation:** the final token.
* **Original-activation logistic AUROC (TEST):** 0.839 at L8, 0.792 at L11. GoT is uninformative before about L4.

### 2.4 RepE

* **Old RepE** ("Pretend you're an honest/untruthful person…", 417/90/90 statements) is **HIST**: it is
  length-confounded (" untruthful" is 3 tokens longer, and length alone gives AUROC 0.80).
* **RepE-LM** (`data/repe/repe_lm/prompts.csv`):
  * one-token persona swaps inside fixed templates, 8 templates;
  * **train 8,340 / validation 1,780 rows, no test split**;
  * gates K1–K7 pass (K5 narrowly, 0.70).
* **Steering/causal:** the M3 site sweep selects the L1 persona slot, the L8 final and the L11 final.
  * T_full is only **0.022 nats at L8** and 0.043 at L11 (`r5_causal_subspace_skyline/summary.csv`), so the behavioural
    signal is small.

---

## 3. Current results (verified against CSVs on 2026-11-12)

**Re-read for this report:**
* `results/shared/m5_confirmatory_test/m5_rows.csv` (TEST);
* `results/openwebtext/r2_scaled_saes/ladder/*.csv` (VAL);
* `results/repe/r5_causal_subspace_skyline/summary.csv` (VAL).

Means over 3 seeds; the frozen decoder and res-jb have n = 1.

### 3.1 OWT, TEST (M5)

| L | SAE | NMSE ↓ | ΔCE ↓ (nats) | SAE/noise ↓ | OWT L0 |
|---|---|---|---|---|---|
| 8 | old TopK-3072 | 0.160 | 0.133 | 1.50 | 128 |
| 8 | old JumpReLU-3072 | 0.156 | 0.121 | 1.44 | 130.3 |
| 8 | old ReLU-3072 | 0.195 | 0.218 | 2.00 | 131.9 |
| 8 | R2 TopK-12288 | 0.094 | 0.062 | 1.20 | 128 |
| 8 | R2 BatchTopK-12288 | **0.090** | 0.054 | 1.13 | 129.3 |
| 8 | R2 ReLU-12288 | 0.132 | 0.131 | 1.81 | 127.6 |
| 8 | R4 KL (TopK) | 0.096 | **0.051** | **0.98** | 128 |
| 8 | R4 MSE control | 0.094 | 0.062 | 1.21 | 128 |
| 8 | res-jb (L0 66, unmatched) | 0.144 | 0.168 | 2.10 | 66 |
| 11 | R2 TopK / BatchTopK / ReLU | 0.111 / 0.107 / 0.168 | 0.096 / 0.093 / 0.221 | 1.48 / 1.48 / 2.24 | 128–131.7 |
| 11 | R4 KL / MSE | 0.113 / 0.111 | 0.081 / 0.096 | 1.22 / 1.49 | 128 |

### 3.2 GoT cities, TEST (M5)

| L | SAE | Re-fit gap ↓ | SAE − PCA ↓ | **GoT L0** |
|---|---|---|---|---|
| 8 | old TopK / JumpReLU / ReLU | 0.061 / 0.058 / 0.034 | 0.036 / 0.034 / 0.008 | 128 / 63 / 77 |
| 8 | R2 TopK / BatchTopK / ReLU | 0.056 ± 0.004 / 0.065 ± 0.024 / 0.028 ± 0.012 | 0.042 / 0.053 / 0.010 | 128 / 80 / 89 |
| 8 | R2 frozen random decoder | 0.067 | **−0.003** | 128 |
| 8 | R4 KL / MSE | 0.048 / 0.059 | 0.035 / 0.045 | 128 |
| 11 | R2 TopK / BatchTopK / ReLU | 0.078 / 0.087 / 0.053 | 0.072 / 0.081 / 0.030 | 128 / 106 / 111 |
| 11 | R2 frozen random decoder | 0.086 | **−0.013** | 128 |

### 3.3 RepE-LM persona and causal, VAL

From the ladder CSVs.

| L | SAE | Persona re-fit gap ↓ | cos(Δ, Δ̂) ↑ | **‖Δ̂‖/‖Δ‖ (1 = ideal)** | Error-term AUROC (persona) | T_rec/T_full ↑ |
|---|---|---|---|---|---|---|
| 8 | old TopK / JumpReLU / ReLU | 0.156 / 0.162 / 0.212 | 0.39 / 0.44 / 0.56 | **1.38 / 1.34 / 0.67** | 0.74 / 0.72 / 0.77 | 0.63 / 0.67 / 0.62 |
| 8 | R2 TopK / BatchTopK / ReLU | 0.108 / 0.126 / 0.193 | 0.49 / 0.50 / 0.61 | **1.30 / 1.25 / 0.63** | 0.74 / 0.73 / 0.81 | 0.73 / 0.75 / 0.64 |
| 8 | res-jb | 0.223 | 0.52 | 0.59 | 0.80 | 0.56 |
| 11 | R2 TopK / BatchTopK / ReLU | 0.109 / 0.101 / 0.194 | 0.51 / 0.52 / 0.62 | **1.31 / 1.30 / 0.62** | 0.79 / 0.81 / 0.87 | 0.69 / 0.70 / 0.71 |

**R5 skyline at L8 (old SAEs):**
* rank-4 DAS 0.91; rank-16 DAS 1.12 (overshoot);
* PCA-4 0.52;
* SAE top-4 features 0.10–0.16, top-64 0.34–0.48;
* random 0.008.

The R5 feature patches were run **only on the old SAEs**, never on R2.

---

## 4. Chronological evidence table (GPT-2)

**Verdicts:**
* **CONF↑** = confirmed improvement (TEST);
* **VAL↑** = improvement on validation, pre-registered rule met;
* **≈** = inconclusive or no clear change;
* **↓** = degradation;
* **✗** = pre-registered criterion not met;
* **INV** = invalid.

| # | Experiment | Data | Arch | Layer | Seeds | Baseline | Change | Before → After | Verdict | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1000-step sweep | OWT | R/T/J | 0–11 | 42 | — | L0 32/64/128 | convergence speed only | VALID (scope) | `results/openwebtext/sae_sparsity_sweep/` |
| 2 | Optimizer consistency fix | OWT | R/T/J | 2, 6, 10 | 3 | resumed/warm-started | uninterrupted 15k | test EV change ≤ 0.006; ranks unchanged | fix, no gain | `reconstruction_evaluation/uninterrupted_protocol/`, `docs/2026-11-04_professor_review_report.md` |
| 3 | 108-run baseline | OWT | R/T/J | 0–11 | 3 | — | — | EV 0.93–0.95 at L0–2 → 0.76–0.82 at L11 | TEST baseline | `uninterrupted_protocol/all_layers_*.csv` |
| 4 | Sparsity matching | OWT | R/T/J | all | 3 | `architecture_summary.csv` | ±10 % rule | 107/108 matched | fix; the old summary is INV | `layer_comparison/sparsity_matched_cells.csv` |
| 5 | P1 controller pilot | OWT | J | 0/2/6/10 | 42 | old controller | anneal + freeze, quantile init | L2 JumpReLU EV +0.020, dead 65 → 21 % | exploratory, n = 1 | `p1_sparsity_control_pilot/` |
| 6 | Recon-only fine-tune (historical) | GoT | R/T/J | 6 | 5 | base | GoT recon | MLP test gap 0.327/0.154/0.323 → 0.161/0.110/0.146 | **L0 unmatched** (ReLU 65 → 144) | `geometry_of_truth/final_test_evaluation/` |
| 7 | Truth-aware (original) | GoT | R/T/J | 6 | — | — | probe-logit loss | logistic gains are circular | **INV** | `docs/2026-11-07_truth_aware_finetuning_audit.md` |
| 8 | Truth-aware (corrected A/B) | GoT | T | 6 | 3 | recon-only | teacher A / evaluator B | 0.189 vs recon-only 0.107 (worse) | ✗ | `truth_aware_pilot/` |
| 9 | Method A | GoT + OWT | R/T/J | 6 | 3 | OWT-only control | GoT-A recon + replay | MLP gap lower 18/18 paired, none > 2 sd | ✗ (directional) | `method_a_layer6/` |
| 10 | E2E-inspired B/C | OWT | R/T/J | 6 | 3 | Method A | + output KL | TopK behaviour met; ReLU/JumpReLU L0 guard failed; truth failed | partial | `e2e_inspired_layer6/` |
| 11 | TopK KL-weight ablation | OWT | T | 6 | 3 | B | w = 0.25, 0.5 | behaviour saturates by 0.25; no truth trade-off | ≈ | `kl_weight_ablation_layer6_topk/` |
| 12 | R1 honesty fine-tune (old RepE) | RepE | R/T/J | 6 | 3 | recon | honesty loss | ReLU ≈ (L0 guard), T/J ✗ | HIST | `repe/r1_honesty_preservation_layer6/` |
| 13 | R2-clamp SAE-in-loop | RepE | R/T/J | 6 | 3 | random features | clamp | below random-feature controls | ✗, HIST | `repe/r2_sae_in_the_loop_steering_layer6/` |
| 14 | M0 | all | R/T/J | all | 3 | — | reanalysis | error → splice damage (β 0.165); error ↛ truth gap | VAL | `shared/m0_within_layer_reanalysis/` |
| 15 | M2 | GoT, RepE-LM | R/T/J | all | 3 | noise, PCA | — | GoT frozen collapse ≈ noise; re-fit gap 0.07–0.12; SAE never beats PCA | VAL | `shared/m2_four_level_faithfulness/` |
| 16 | M4 | RepE-LM | R/T/J | 1/8/11 | 3 | T_full | — | T_rec/T_full 0.93–0.97 (L1), 0.62–0.68 (L8/L11) | VAL | `repe/sae_patch_decomposition/` |
| 17 | IMP-L Method A | GoT | R/T/J | 3/8/9 | 3 | OWT-only | — | frozen-reader gain 4/9; re-fit none | VAL, qualified | `geometry_of_truth/multilayer_finetune/` |
| 18 | IMP-L output-KL | OWT | R/T/J | 3/8/9 | 3 | Method A | KL | behaviour 5/9; 4/9 L0 drift; truth ↓ at L3 R/T | partial | same |
| 19 | E3 | RepE-LM | R/T/J | 8 | 3 | base | fine-tunes | no causal gain; KL gains only with L0 drift | ✗ | `repe/e3_finetuned_causal_preservation/` |
| 20 | RLM-RP | RepE-LM | 216 SAEs | all | 3 | noise | — | persona loss 2–3× noise; Method A 0/12 | VAL | `repe/repe_lm_representation_preservation/` |
| 21 | E2 res-jb | all | res-jb | 1/8/10/11 | 1 | ours | external | conclusions transfer | VAL, descriptive | `shared/e2_public_sae_reference/` |
| 22 | **R2 C1** | OWT | T | 8 | 3 | old TopK-3072 | 100M tokens, 12,288 wide, recipe | NMSE 0.160 → 0.094; ΔCE 0.133 → 0.062 | **CONF↑** | `shared/m5_confirmatory_test/` |
| 23 | R2 C1 | OWT | T | 11 | 3 | same | same | NMSE 0.185 → 0.111; ΔCE 0.204 → 0.096 | **CONF↑** | same |
| 24 | R2 C1 | GoT | T | 8/11 | 3 | same | same | re-fit gap 0.061 → 0.056; 0.101 → 0.078 | ≈ (TEST) | same |
| 25 | R2 C1 | RepE-LM | T | 8/11 | 3 | same | same | persona gap −31 % / −27 %; T_rec 0.63 → 0.73 / 0.62 → 0.69 | VAL↑ (**no test split**) | `r2_scaled_saes/README.md` §2.2 |
| 26 | R2 C1 | RepE-LM | T | 8/11 | 3 | same | same | SAE − PCA persona gap | ≈ | same |
| 27 | R2 A1 (data + recipe, equal width) | OWT | T-3072 | 1/8/11 | 3 | old TopK-3072 | — | NMSE −23 to −35 % | CONF↑ (tokens vs recipe **not separated**) | M5 |
| 28 | R2 A2 (width) | OWT/GoT | T | 8 | 3 | R2 TopK-3072 | 12,288 | OWT ↑; **GoT L8 ↓ (2 cells)** | CONF↑ / ↓ | M5 |
| 29 | R2 random decoder | all | T-frozen | 1/8/11 | 1 | trained | — | GoT SAE − PCA −0.003 (beats trained) | descriptive | M5, R2 README §2.2 |
| 30 | **R4 KL vs MSE** | OWT | T | 8/11 | 3 paired | MSE control | NMSE + α·KL, 5M tokens | ΔCE −17 % / −16 %; NMSE +2 % | **CONF↑** behaviour | `openwebtext/r4_kl_finetune/`, M5 |
| 31 | R4 KL vs MSE | RepE-LM | T | 11 | 3 paired | MSE control | same | T_rec 0.694 → 0.671 (all seeds) | **↓ VAL** | `r4_kl_finetune/README.md` |
| 32 | R5 | RepE-LM | old SAEs | 1/8/11 | 3 | DAS/PCA/random | — | rank-4 DAS 0.91 > whole SAE 0.62–0.67 at L8 | VAL, descriptive | `repe/r5_causal_subspace_skyline/` |

**Searched for and NOT FOUND:**
* a modern JumpReLU SAE;
* R2 data-scaling (10M/25M/50M) evaluations;
* R5 feature patches on R2 SAEs;
* any RepE-LM test result;
* a GoT result on neg_cities under R2/M5;
* a width sweep beyond 3,072/12,288;
* an L0 other than 128 at convergence;
* inference-time optimisation, Gated SAE, Matryoshka;
* a decoder-only or scale fine-tune;
* a minimum-detectable-effect statement.

**Disagreements between files:** none found in this audit beyond those already recorded in
`docs/final_defense/EXPERIMENT_AUDIT.md`. That file records the M5 L0 maximum of 133.39 vs 133.2, and that the pre-run
README hash cannot be reproduced.

---

## 5. Root-cause analysis

### 5.1 OpenWebText: what still limits reconstruction and behaviour

| # | Bottleneck | Evidence | Likely mechanism | How to verify | Proposed fix | Risks |
|---|---|---|---|---|---|---|
| O1 | **Remaining error is structured, not noise** | SAE/noise ΔCE ratio 1.13–1.48 (R2 L8/L11 TopK/BatchTopK), 1.8–2.2 (ReLU); R4 brings it to 0.98 at L8 | The MSE objective weights directions by variance; downstream layers read some low-variance directions (Engels et al. "dark matter") | Already shown; decompose further with X1 (encoder vs dictionary) | KL-aware objective at fixed L0 (R4, done); ITO (X1) | KL harms causal preservation at L11 (VAL) |
| O2 | **ReLU shrinkage** | R2 ReLU is the worst reconstructer at depth (L8 NMSE 0.132 vs 0.094); persona contrast norm retention 0.62–0.71 (ReLU), 0.59 (res-jb ReLU) | The L1 penalty pulls magnitudes towards 0 (Wright & Sharkey 2024; Gated SAE paper) | Fit the optimal per-token / per-latent scale on validation (X1-a); the shrinkage estimate is the fitted γ > 1 | Scale/decoder fine-tune without sparsity loss; Gated/JumpReLU | Fine-tuning the encoder destroys sparsity (Wright & Sharkey) |
| O3 | **Encoder amortisation gap** | HYPOTHESIS for our SAEs. A linear+nonlinearity encoder is sub-optimal for sparse coding (O'Neill et al. 2024; GDM update 2024) | Wrong coefficients, or the wrong support | X1-b (least squares on the encoder's support) and X1-c (pursuit with the same L0) | ITO as an evaluation-time decoder (diagnostic) | ITO changes the SAE's encoder, so any gain is a "dictionary + optimiser" result, not an SAE result; it must be labelled so |
| O4 | **Tokens vs recipe unseparated** | A1 confounds them; 10M/25M/50M checkpoints exist, unevaluated | — | X0: run the existing harness with `--sources r2scaling` | no training needed | 10M/25M checkpoints are mid-schedule (LR not decayed), so the curve mixes data with schedule; must be stated |
| O5 | **No modern JumpReLU** | Pilot abandoned (R2 README §1.2); old JumpReLU uses a sigmoid surrogate (`03_…:41-46`) | Code-level hypothesis: per-step multiplicative λ controller (η 0.01, `17_…:220`) interacting with Adam on thresholds at the full LR (THR_LR_MULT 1.0) gives a feedback oscillation; published JumpReLU uses a fixed λ | Short pilot (5M tokens, L8, seed 42) of (a) fixed λ with warm-up, (b) the Anthropic 2025 recipe, (c) initialise from R2 BatchTopK | X3 (§8) | Training could fail again; budget it as a time-boxed pilot |
| O6 | Width/L0 single point | only L0 128; width 3,072/12,288 | — | G1/G5 | out of budget; not recommended now | — |

### 5.2 Geometry of Truth: why improvement attempts show no effect

| # | Bottleneck | Evidence | Mechanism | Verify | Fix | Risks |
|---|---|---|---|---|---|---|
| G1 | **The metric cannot discriminate** | Seed sd of the re-fit gap 0.004–0.024 is as large as the differences tested; 0/34 M5 cells; a **random frozen decoder beats every trained SAE on SAE − PCA** at L8/L11 | Low within-set variance of final-token activations (task NMSE > 1 for most SAEs); n = 452 test statements; seed-level Welch with n = 3 | Compute a minimum detectable effect from existing per-SAE bootstrap arrays; paired statement-level comparisons | T1 (§8): measurement upgrade before any new method | none; it is analysis |
| G2 | **Truth signal sits in the error** | GoT error-term AUROC 0.66–0.77 at L8 (old and R2), 0.58–0.66 at L11 (ladder CSVs) | The truth direction is low-variance, so MSE ignores it | already measured | Direction-preserving objective (T2), but see G4 | — |
| G3 | **Task L0 unmatched** | GoT L0: TopK 128, BatchTopK 80/106, ReLU 89/111, old JumpReLU 63/93 | Threshold architectures fire fewer latents on GoT prompts | Inference-time TopK k truncation of R2 TopK to the other architectures' GoT L0 (no training) | T1-c | Truncation ≠ training at k*; label it as a diagnostic (the RLM-RP truncation control precedent) |
| G4 | **Truth-aware objectives already failed** | circular original (INV); corrected version worse than recon-only; Method A moves only the frozen reader | A probe-based loss trains the readout, not the information; small n overfits | — | **Do not repeat** without a new mechanism and an independent evaluator | Circularity; test exposure |
| G5 | **One truth category** | M5 uses cities only; neg_cities is on disk; numerical sets are shortcut-confounded | Cross-polarity generalisation untested under R2 | Train on cities, test on neg_cities (polarity transfer; Bürger et al. 2024) | T1-d | neg_cities shares cities; groups must be kept disjoint across both sets |

**Conclusion for GoT (VAL + TEST):**
* Truth information is **not preserved** in the re-trained-reader sense; the gap is about 0.03–0.10 AUROC.
* It is **not improvable within our current measurement power**.
* The frozen-probe collapse is mostly fragility (M2).
* The re-fit gap at L8 does not separate trained from random SAEs.
* There is a ceiling problem: the original AUROC is only 0.84, so the gap has little room.
* The first priority is power, not a method.

### 5.3 RepE honesty: what the SAE does to the persona contrast (new diagnostic)

**Reading the existing ladder CSVs (§3.3) gives a finding not previously reported:**

* **ReLU-family SAEs shrink the contrast.**
  * Norm retention ‖Δ̂‖/‖Δ‖ is 0.62–0.71 (old and R2 ReLU, res-jb).
  * They have the *highest* direction cosine (0.56–0.62) and the *largest* re-fit gap (0.19–0.21) and error AUROC
    (0.77–0.87).
  * Interpretation: the direction is roughly right but too small, and the missing part is in the error.
* **TopK-family SAEs (TopK, BatchTopK, old JumpReLU) inflate the contrast.**
  * Norm retention is 1.09–1.46, while cos(Δ, Δ̂) is only 0.37–0.52.
  * **HYPOTHESIS:** a one-token persona swap changes *which* latents win the top-k or pass the threshold. Δ̂ then
    contains large off-direction components from latents switching on and off, a discontinuity of hard selection.
* **Consequences:**
  * The two failure modes need different fixes. Shrinkage needs a scale or magnitude correction (ITO, scale
    fine-tune). Support flipping needs a contrast-consistency objective or a smoother selection.
  * Neither failure mode is visible in OWT NMSE.

| # | Bottleneck | Evidence | Verify | Fix | Risks |
|---|---|---|---|---|---|
| H1 | **No test split for RepE-LM** | `prompts.csv` has train/validation only; every persona and causal claim is VAL | — | Build the test split from held-out statements (`20_build_repe_lm_dataset.py` logic, new output path) with K1–K7 gates and the near-duplicate check | must not be used before the protocol is frozen |
| H2 | Contrast distortion (above) | §3.3 | Per-pair decomposition of Δ̂ − Δ into support-change and magnitude-change parts (cheap; existing caches) | X1 for ReLU; H2-exp (§8) for TopK | — |
| H3 | **Feature-level steering is far below a subspace** | R5 (old SAEs): top-4 features 0.10–0.16 vs DAS-4 0.91 at L8; not run on R2 | R5 feature arm on R2 checkpoints | Feature selection by output effect (Arad et al. 2025). Our M4 attribution ranking is already output-based, so the expected gain is uncertain | Selection on TRAIN only |
| H4 | KL fine-tuning trades causal for behaviour | R4 L11 T_rec −0.024 (all seeds, VAL) | — | Add a persona-contrast term (I1 factorial) | Small T_full (0.02–0.04 nats) means low power |
| H5 | **Weak behavioural signal** | T_full 0.022 nats at L8 | MDE statement | out of scope for GPT-2; state it as a limitation | — |

---

## 6. Literature: what each candidate method has shown

**Links:** checked 2026-11-12 for the new entries; the others were checked 2026-11-10 (independent review §8).

| Method | What the source shows | Model / data | Truth evaluated? | Fits our checkpoints? | Retrain? |
|---|---|---|---|---|---|
| **ITO / gradient pursuit** on a fixed decoder | GDM progress update 1 (2024): gradient pursuit beat the other ITO methods tested; higher-L0-trained dictionaries were better even when coefficients were inferred at low L0. A commenter questioned whether the gains are interpretable; the authors called it early. Engels et al. 2024 report only a slight error reduction from better encoders. O'Neill et al. 2024 prove an amortisation gap (synthetic evidence). | Gemma-class residual SAEs | no | **yes** (decoder only) | no |
| **Shrinkage fix: scale/decoder fine-tune** | Wright & Sharkey 2024: fine-tuning a per-feature scale, or the decoder, without a sparsity loss raised the L2 ratio towards 1 and improved perplexity recovery; fine-tuning the encoder destroyed sparsity | Pythia-70M | no | **yes** (ReLU) | no (short fine-tune) |
| **Gated SAE** | Rajamanoharan et al. 2024 (NeurIPS): separates "which" from "how much", so L1 applies only to gating; removes shrinkage; about half the firing features for the same fidelity; 50 % more training compute | GELU-1L, Pythia-2.8B, Gemma-7B | no | no | **yes** |
| **JumpReLU (GDM STE)** | Rajamanoharan et al. 2024b; Gemma Scope | Gemma-2 | no | no | yes |
| **JumpReLU (Anthropic Jan 2025 recipe)** | Transformer Circuits update, Jan 2025: STE gradients to all parameters, a tanh sparsity penalty and a pre-activation loss. A third-party MATS write-up reports reliable training and an SAELens implementation (secondary source) | Anthropic internal | no | no | yes |
| **TopK + AuxK** | Gao et al. 2024 | GPT-2, GPT-4 | no | done (R2) | — |
| **BatchTopK** | Bussmann et al. 2024: beats TopK on GPT-2/Gemma-2-2B reconstruction | GPT-2, Gemma | no | done (R2) | — |
| **Matryoshka** | Bussmann et al. 2025: less absorption, slightly worse reconstruction | Gemma-2-2B, toy | no | no | yes |
| **Short KL fine-tune** | Karvonen 2025: closes 20–50 % of the CE gap; KL, LoRA and linear adapters are similar and non-additive | Gemma-2-2B, Pythia | SAEBench mixed | done (R4) | no |
| **LoRA on the model** | Chen, Engels & Tegmark 2025: CE gap −30 to −55 % | Gemma | no | — | **changes the explained model**; diagnostic only |
| **Decoder-only fine-tune** | Wright & Sharkey 2024 ("Decoder" variant) | Pythia-70M | no | yes | no |
| **Wider dictionary at matched L0** | Gao et al. 2024 scaling laws | GPT-4 | no | A2 done: OWT ↑, GoT L8 ↓ | yes |
| **More unique tokens** | Gao et al.; res-jb 300M | — | no | done (100M) | — |
| **Error-term analysis** | Engels et al. 2024: about half the error is linearly predictable; Marks et al. 2024: error nodes matter in circuits | GPT-2, Gemma, Pythia | no | yes (already measured: error AUROC) | no |
| **Supervised subspace skyline (DAS/MDAS)** | Geiger et al.; RAVEL (Huang et al. 2024, MDAS state of the art on attribute disentanglement, Llama2-7B); Chaudhary & Geiger 2024 (SAEs below DAS on GPT-2) | GPT-2, Llama | attributes, not truth | done (R5, old SAEs) | no |
| **SAE-targeted steering (SAE-TS)** | Chalnev, Siu & Conmy 2024: uses the SAE to *measure* steering effects; better coherence/effect balance than CAA and feature steering | Gemma-2 | no | yes | no |
| **Output-score feature selection** | Arad, Mueller & Belinkov 2025 (EMNLP): 2–3× steering gains after filtering features with low output scores; competitive with supervised methods | Gemma-2, Llama | no | yes | no |
| **Representation Engineering (RepE)** | Zou et al. 2023 | Llama-2 | honesty | — | — |
| **Truth direction** | Marks & Tegmark 2023 (GoT); Bürger et al. 2024 (2-D truth subspace, polarity-general) | Llama-2, Gemma | truth | — | — |
| **Probing baselines** | Kantamneni et al. 2025: SAE probes lose to logistic regression; Korznikov et al. 2026: random-decoder SAEs nearly match | many | yes (probing) | consistent with M2/M5 | — |

**No source above evaluates truth or honesty preservation through SAE reconstruction under a frozen model.** Every
truth/honesty claim we could make is therefore new evidence, not a replication.

---

## 7. Technique-to-dataset compatibility and classification

**Classes:**
* **A** = published improvement in a comparable setting;
* **B** = published, benefit for us unproven;
* **C** = new hypothesis;
* **D** = not suitable.

| Technique | OWT | GoT | RepE | Existing SAE compatible? | Retrain? | Published evidence | Risk | Class | Priority |
|---|---|---|---|---|---|---|---|---|---|
| Data-scaled TopK/BatchTopK (R2) | ✓ | — | ✓ (VAL) | — | done | Gao, Bussmann | — | A (**done**) | — |
| KL fine-tune at fixed L0 (R4) | ✓ | — | ✗ (L11) | yes | done | Karvonen | causal cost | A (**done**) | — |
| Tokens-vs-recipe evaluation (X0) | ✓ | ? | ? | yes (checkpoints exist) | no | — | mid-schedule checkpoints | A (measurement) | **P0** |
| ITO on a fixed decoder (X1) | ? | ? | ? | **yes** | no | GDM 2024 (Gemma), O'Neill (synthetic) | label as dictionary-level, not SAE-level | **B** | **P0** |
| Scale/decoder shrinkage fine-tune (X2) | ? (ReLU) | ? | ? (ReLU norm 0.62) | yes | no | Wright & Sharkey (Pythia-70M) | improves ReLU only; TopK has no L1 | **B** | P1 |
| Modern JumpReLU (X3) | ? | ? | ? | partly (BatchTopK init) | yes (short) | Rajamanoharan; Anthropic 2025 | may fail again | **B** | P1 (time-boxed) |
| Gated SAE | ? | ? | ? | no | yes (100M tokens) | Rajamanoharan 2024 | ≈ 2 h GPU per layer set; superseded by JumpReLU in the literature | B | P2 |
| Matryoshka | ✗ (slightly worse) | ? | ? | no | yes | Bussmann 2025 | off-target (absorption) | B | P2 |
| Wider dictionaries | ✓ | ✗ (A2 L8) | ? | no | yes | Gao | GoT regression | A for OWT only | P2 |
| LoRA on the model | ✓ | — | — | yes | — | Chen et al. | changes the model | **D** | — |
| Transcoders | — | — | — | no | yes | Paulo et al. | different object | **D** | — |
| GoT measurement upgrade (T1) | — | ✓ | — | yes | no | — (statistics) | none | A (method-agnostic) | **P0** |
| Probe-guided truth loss | — | ✗ (failed ×3) | — | yes | yes | none | circularity | **D** (as currently formulated) | — |
| Direction-preserving truth loss (raw Δ, no probe) | — | ? | — | yes | short | none | small n; overfit to cities | C | P2 |
| Cross-category (cities → neg_cities) | — | ✓ eval | — | yes | no | Bürger et al. | city overlap | A (evaluation) | P1 (part of T1) |
| Group-of-latents interventions | — | — | ✓ (R5 feature arm) | yes | no | Arad; SAE-TS | selection overfit | B | P1 |
| RAVEL-style MDAS | — | — | ✓ (skyline) | yes | no | Huang et al. | — | A as a reference (done as R5) | — |
| RepE-LM test split (H1) | — | — | ✓ | yes | no | — | — | required | **P0** |
| Contrast-preserving fine-tune (H2-exp) | ? | ? | ? | yes (R2 TopK) | short | none direct (R1 on the old task failed for TopK) | overfit to templates; must use held-out templates and statements | **C** | P1 |
| KL × contrast factorial (I1) | ? | ? | ? | yes | short | Karvonen (KL part) | low causal power | C | P1 |

---

## 8. Recommended controlled experiments (pre-registration drafts; NOT approved)

**Common rules for all:**
* VALIDATION for development; no test row is read until a separate, committed M6 freeze (D31: the protocol is frozen
  in its own file and committed before the run).
* L0 is reported on OWT, GoT and RepE-LM; guards are ±5 % of the base for modifications and ±10 % of target for new
  SAEs.
* New output folders only; scripts refuse to overwrite.
* Checkpoints are hashed in the output rows.
* One heavy job at a time (the OOM rule).
* Evaluation runs through the existing harness `scripts/shared/sae_ladder_evaluation.py`, which is never edited: new
  sources go in a new wrapper script that imports it.

**Mandatory metrics for every experiment:**
* OWT NMSE/EV/cosine/L0 and ΔCE/KL/noise ratio;
* GoT frozen and re-fit gaps, SAE − PCA, task L0;
* RepE-LM persona re-fit gap, held-out-wording gap, cos(Δ, Δ̂), norm retention;
* M4 T_rec/T_full at the layer's site.

The M4 steering metrics apply where a causal claim is made. If GoT is uninformative at L1, it is reported as N/A.

**Decision rules:**
* *Unpaired* comparisons (different families) use the R2 rule: 9/9 pairs and a Welch CI excluding 0.
* *Paired* comparisons (same base checkpoint) are pre-registered as follows. **IMPROVEMENT** requires all of:
  * 3/3 seeds in the same direction;
  * the mean paired difference beyond a stated minimum meaningful effect: 5 % relative for NMSE/ΔCE, 0.02 for AUROC
    gaps, 0.03 for T_rec/T_full;
  * all guards held: L0 ±5 %; OWT NMSE not worse by > 5 % relative; ΔCE not worse by > 5 %; T_rec/T_full not worse
    by > 0.02; GoT and persona re-fit gaps not worse by > 0.02.

  REGRESSION is the mirror image; everything else is NO CLEAR CHANGE.
* Thresholds are fixed now and not changed after results.

### X0. Tokens vs recipe and the data-scaling curve (zero training)

| Item | Specification |
|---|---|
| Question | How much of R2's gain is token count at a fixed recipe? |
| Baseline | R2 `checkpoint_final.pt` (100M) |
| Intervention | Evaluate the existing `checkpoint_{10M,25M,50M}.pt` |
| Changed | tokens seen |
| Held constant | recipe, width, seeds, data order |
| Data | OWT validation; GoT cities val; RepE-LM val |
| Layers | 1, 8, 11 |
| Architectures | TopK, BatchTopK, ReLU, TopK-3072 |
| Seeds | 3 |
| Steps | none |
| Metrics | light ladder (reconstruction + splicing) for all 108 checkpoints; full ladder for TopK at L8/L11 (18) |
| Success | Descriptive: monotone curve per family; report the share of A1's gain reached at 10M tokens (the old SAEs saw 3.8M exposures) |
| Failure | n/a (descriptive) |
| Compute | about 1–2 h GPU (estimate from 190 s per full ladder; light ≈ 30–60 s, NOT MEASURED) |
| Output | `results/openwebtext/r2_scaled_saes/scaling/` (empty today) or a new `results/openwebtext/x0_data_scaling/` |
| Reproducibility | the 100M row must equal the existing ladder row exactly |
| Caveat | 10M/25M/50M are mid-schedule (LR not decayed). Disclose this; it is not a clean token-only ablation |

### X1. Inference-time coefficient optimisation (Experiment A): encoder vs dictionary error (no training)

| Item | Specification |
|---|---|
| Question | Is the remaining reconstruction, behaviour, information and causal loss due to the *encoder* (coefficients/support) or the *dictionary*? |
| Baseline | R2 TopK/BatchTopK/ReLU-12288 final; old R/T/J-3072 (for contrast) |
| Interventions (same decoder, same b_dec) | **(a) Scale:** one global γ per SAE fitted on OWT TRAIN (x̂ = γ·W z + b); this isolates shrinkage. **(b) Support refit:** keep the encoder's support S per token and solve non-negative least squares for the coefficients on S, so L0 is ≤ the original L0. **(c) Pursuit:** gradient pursuit / OMP selecting the support with the same per-token L0 as the SAE |
| Held constant | decoder, layer, tokens, L0 (exactly for (a) and (b); per token for (c)) |
| Data | OWT validation; GoT cities val; RepE-LM val (TRAIN for γ) |
| Layers | 1, 8, 11 |
| Seeds | 3 per family |
| Steps | none |
| Metrics | the full mandatory set. The PCA reference is re-matched to the new NMSE automatically (the harness matches PCA at the SAE's own NMSE) |
| Comparisons | (a/b/c) vs the plain SAE, paired by checkpoint; (b) vs (c) separates coefficient from support error |
| Success | (b) or (c) is an IMPROVEMENT on OWT NMSE **and** ΔCE at L8 and L11 under the paired rule, with no guard failure |
| Secondary | sign of the change in persona norm retention (ReLU should move towards 1; TopK may not move, if support flipping is the cause) |
| Failure | No IMPROVEMENT, meaning the encoder is near-optimal for its dictionary. This is still informative and is reported as such |
| Compute | NNLS on ≤ 128 columns per token is cheap on GPU; splicing needs a per-forward solve. Estimate 2–4 h GPU for seed 42 at all layers, then 3 seeds for surviving variants. **Not measured; a smoke test must time it first** |
| Output | `results/openwebtext/x1_inference_time_optimisation/` |
| Reproducibility | the identity variant must reproduce the ladder row bit for bit; NNLS tolerance logged; GPT-2 parameter hash before and after |
| Interpretation guard | An ITO gain is a property of "dictionary + optimiser", not of the trained SAE. It must never be reported as an SAE improvement without that label |

### X2. Shrinkage correction for ReLU (short fine-tune, no sparsity loss)

* **Question:** does removing L1 shrinkage close ReLU's gap to TopK and fix its persona-contrast norm deficit?
* **Base:** R2 ReLU-12288 at L8/L11 × 3 seeds.
* **Arms:**
  * (i) a per-latent scale vector s (z → s ⊙ z), MSE only;
  * (ii) decoder-only MSE fine-tune;
  * (iii) control: the same tokens with no parameter update (identity).

  The encoder is frozen, so L0 is unchanged by construction.
* **Data and budget:** 5M tokens from shard 79 (the R4 tokens, the same start chunk).
* **Success:** an IMPROVEMENT in NMSE and ΔCE vs the base, under the paired rule, **and** persona norm retention moves
  ≥ 0.1 towards 1.
* **Compute:** about 6 min per run, measured for R4 (12 runs in 4,248 s), so about 40 min plus 12 ladders ≈ 1.2 h.
* **Run it only if** X1-(a) shows γ meaningfully > 1 (pre-register the threshold: γ ≥ 1.05).

### X3. Modern JumpReLU (time-boxed pilot)

* **Question:** can a threshold architecture with an OWT L0 within ±10 % of 128 be trained in our budget? This fills
  the "modern JumpReLU NOT VERIFIED" gap.
* **Pilot** (L8, seed 42, 10M tokens, monitor shard only), three variants:
  * (a) a fixed λ with linear warm-up, no controller, threshold LR × 0.1;
  * (b) the Anthropic Jan-2025 loss (tanh penalty + pre-activation loss);
  * (c) initialise the per-latent thresholds and weights from R2 BatchTopK and fine-tune with an L0 Lagrangian.
* **Go/no-go:** the L0 trajectory over the last 20 % stays within ±10 % of 128. If none passes, stop and report.
* **Full run** (only after a go): layers 8/11 × 3 seeds × 100M tokens, roughly 2–3 h GPU (extrapolated from R2's
  5.5 h for 39 SAEs; NOT MEASURED).
* **Risk:** the most expensive item here, with the lowest prior of success.

### T1. Truth-preservation measurement upgrade (best truth experiment; mostly CPU)

* **Question:** at what effect size could *any* SAE change in GoT truth preservation be detected, and do the existing
  conclusions hold under paired, per-statement, cross-polarity and task-L0-matched comparisons?
* **(a) Minimum detectable effect** from the existing M2/M5 bootstrap arrays (`m2_four_level_faithfulness/bootstrap_arrays.npz`):
  * per layer, the smallest re-fit gap difference detectable at 80 % power with n = 3 seeds and n = 452 statements;
  * this also discharges the roadmap's open "minimum-detectable-effect statement".
* **(b) Paired per-statement comparisons** between SAE families on the same validation rows. This is a statement-level
  paired bootstrap of the score difference; it needs per-row scores, which a new wrapper script would save.
* **(c) Task-L0-matched TopK at inference:** truncate R2 TopK to k = BatchTopK's and ReLU's GoT L0 (80, 89, 106, 111).
  Compare the gaps at equal *task* L0. No training; labelled a diagnostic.
* **(d) Polarity transfer:** probe trained on cities TRAIN originals, evaluated on neg_cities VAL (x and x̂). Report
  the transfer gap, with city groups disjoint across both sets.
* **Success** (pre-registered as a measurement result, not an improvement):
  * the MDE is reported;
  * (b) reproduces or overturns the ReLU < TopK GoT ordering at p < 0.05 (BH within T1);
  * (c) states whether the ordering survives at equal task L0.
* **Compute:** < 1 h GPU (only (c) and (d) need SAE passes).
* **Value:** it decides whether any truth-preserving method is *testable* with GoT. If the MDE exceeds about 0.03 AUROC,
  further truth-method experiments on GoT cities are not worth running, and the report should say so.

### H1. RepE-LM test split (prerequisite for every honesty claim)

* Build test prompts from the held-out RepE statements (`data/repe/facts_splits.csv` test rows):
  * the same templates and slot words as M1;
  * K1–K7 gates on TRAIN-derived thresholds only;
  * the near-duplicate check against train/validation (the statement 326 precedent).
* **Output:** a new `data/repe/repe_lm_test/`; the existing `prompts.csv` is not edited.
* **No SAE is evaluated on it** until the M6 freeze.
* **Compute:** minutes.

### H2. Contrast-consistent fine-tune at fixed L0 (Experiment E; best honesty experiment)

* **Question:** can an R2 TopK SAE be adapted so that its persona contrast is preserved (Δ̂ ≈ Δ), without a probe,
  without L0 change and without OWT damage? Does this transfer to held-out templates and statements, and to T_rec?
* **Base:** R2 TopK-12288 at L8/L11 × 3 seeds. TopK keeps L0 fixed by construction.
* **Arms** (same OWT tokens and the same persona TRAIN pairs in both):
  * **(c)** loss = NMSE_OWT + β · Σ‖Δ̂ − Δ‖² / Σ‖Δ‖² on TRAIN persona pairs (templates T2–T4, TRAIN statements);
  * **(m)** control: NMSE_OWT + NMSE on the *same* persona prompts (reconstruction only), which separates "seeing the
    prompts" from "the contrast loss".
* **β** is set by scale matching on the first batch (the R4 convention), with no tuning on validation.
* **Budget:** 5M OWT tokens; lr 2e-5.
* **Leakage controls:**
  * no evaluation probe is involved in training;
  * cos(Δ, Δ̂) and norm retention on TRAIN templates are reported only as "objective attained";
  * the evaluation uses VAL statements on **held-out templates T5/T6** plus the re-fit gap and M4 T_rec, none of which
    is trained on.
* **Success** (paired rule, (c) vs (m)):
  * an IMPROVEMENT in the held-out-template persona re-fit gap **or** in T_rec/T_full at L8 or L11;
  * with no guard failure on OWT ΔCE, NMSE or GoT.
* **Failure:** no IMPROVEMENT, or any guard regression.
* **Prior:** low to medium. R1 on the old (length-confounded) task was NOT SUPPORTED for TopK. The current mechanism is
  different (a raw-Δ loss, not a probe; a support-flip diagnosis), but this is a **class C hypothesis**.
* **Compute:** 12 runs × about 6 min plus 12 ladders ≈ 2 h GPU.

### I1. Integrated: KL × contrast factorial at fixed L0

* **Question:** can the R4 behaviour gain (ΔCE −17 %) be kept while removing its L11 causal regression, by adding the
  H2 contrast term?
* **Design:** 2 × 2 (KL on/off × contrast on/off) on R2 TopK at L8/L11 × 3 seeds, with identical tokens.
  * The (off, off) cell is the MSE control, and the (KL, off) cell is R4. Rerun both, unless the new runs' data stream
    matches R4's exactly; if it does, verify bit-identity first.
* **Success:**
  * (KL+contrast) vs (KL): T_rec at L11 IMPROVEMENT;
  * (KL+contrast) vs (off, off): ΔCE IMPROVEMENT;
  * both under the paired rule.
* **Interaction:** reported descriptively.
* **Compute:** 24 runs ≈ 2.5 h plus 24 ladders ≈ 1.3 h, so about 4 h GPU.
* **Run after H2**, and only if H2 does not show a guard regression.

### Not recommended now
* LoRA on GPT-2 (it changes the explained model).
* More probe-based truth losses (three failures, one circular).
* Matryoshka and Gated SAEs (expensive retraining for gains unrelated to our failure modes).
* Width or L0 frontiers (G1/G5; out of budget before 12-08).
* More Method A seeds (the re-fit reader is null).

---

## 9. Answers to the user's Phase 7 questions

* **A, ITO:** proposed as X1. It preserves sparsity exactly in variants (a) and (b), and per token in (c).
* **B, error decomposition:**
  * Already partly measured: the error-term AUROC is in every ladder row.
    * GoT: 0.66–0.77 at L8.
    * Persona: 0.72–0.87.
    * M4 T_err/T_full: 0.01–0.45.
  * Adding e back restores x exactly. That is the M4 identity check (`m4__identity_err`), a control and not an
    improvement.
  * Not proposed as new work, except the support/magnitude split inside X1.
* **C, layer selection** (from existing data):
  * Truth probes are best at L8 (0.839) and drop at L11 (0.792); GoT is uninformative before L4.
  * Reconstruction and behavioural fidelity are best at early layers (R2 L1 ΔCE 0.002–0.006) and worst at L11.
  * Causal preservation is near-complete at L1 (0.99) and partial at L8/L11 (0.69–0.75).
  * **The best truth-probe layer is not the best fidelity layer.** L8 is the compromise, which the existing sites
    already use. No new experiment is needed.
* **D, matched-sparsity comparison:** done at OWT L0 128 (108 runs; R2).
  * Missing: task-L0 matching (T1-c) and a modern JumpReLU (X3).
* **E, direction preservation:** proposed as H2, without a probe teacher, to avoid the D12 circularity.

---

## 10. Recommended best next steps

| Role | Pick | Why → what it addresses → literature → what could improve → proof → what could go wrong |
|---|---|---|
| Low-cost reconstruction | **X1 (ITO)**, with X0 in the same queue | Splits the remaining structured error (O1) into encoder vs dictionary, and directly tests ReLU shrinkage (O2) → GDM ITO, O'Neill, Wright & Sharkey → NMSE/ΔCE at fixed L0, possibly persona norm retention → paired rule vs the identity variant → gains may be tiny (Engels: slight), or ITO-only and not an SAE property |
| Truth preservation | **T1 (measurement upgrade)** | G1: the current GoT metric cannot detect changes (random decoder ≈ trained; seed sd ≈ effect) → statistical, not a method → a defensible MDE and task-L0-matched ordering → pre-registered bootstrap → it may show GoT cities is too weak a test, which is an honest and useful conclusion |
| Honesty/steering | **H1, then H2** | H1 is mandatory: no RepE claim can be confirmed without it. H2 targets the newly diagnosed contrast distortion (§5.3) → no direct precedent (class C) → held-out-template persona gap, T_rec → paired vs a matched reconstruction-only arm → it may overfit to templates or fail like R1 |
| Integrated | **I1 (KL × contrast factorial)** | R4 improved behaviour but cost causal preservation; I1 tests whether both can be had at fixed L0 → Karvonen (KL) + H2 → ΔCE and T_rec together → 2 × 2 paired → low causal power (T_full 0.02–0.04 nats) may make the interaction undetectable |

### Ranking (1 = best)

| Experiment | Scientific value | Evidence strength | Information gained | Difficulty (1 = easiest) | Compute (1 = cheapest) | Invalid-comparison risk (1 = lowest) |
|---|---|---|---|---|---|---|
| H1 RepE-LM test split | 1 | 1 | 2 | 1 | 1 | 1 |
| T1 GoT measurement | 2 | 1 | 1 | 2 | 1 | 1 |
| X0 data scaling | 4 | 1 | 3 | 1 | 2 | 3 (mid-schedule) |
| X1 ITO | 3 | 2 | 2 | 3 | 3 | 3 (label as ITO) |
| H2 contrast fine-tune | 3 | 4 (class C) | 3 | 3 | 3 | 2 |
| I1 factorial | 4 | 4 | 4 | 3 | 4 | 2 |
| X2 ReLU shrinkage | 5 | 3 | 5 | 2 | 2 | 2 |
| X3 modern JumpReLU | 5 | 2 | 4 | 4 | 5 | 2 |

---

## 11. Reproducibility safeguards, runtime and execution plan

**Safeguards:**
* a pre-registration per experiment, with SHA-256 of scripts, splits and the frozen protocol file, **committed before
  the run** (D31);
* new folders only, with refuse-to-overwrite;
* the identity / 100M-token rows must reproduce the existing ladder values exactly;
* GPT-2 parameter hash before and after;
* checkpoint hashes in every row;
* no test rows until a separate M6 freeze;
* one heavy job at a time, with an 8 GB free-RAM gate;
* tests (`python -m pytest -q src/tests`) before every commit;
* **back up the R2/R4 checkpoints off-machine first** (roadmap §11, P0).

**GPU estimates** (A10G; from the measured 190 s full ladder and the 6 min R4 fine-tune; anything else is NOT MEASURED):

| Step | GPU | Engineering |
|---|---|---|
| H1 | ~0 (gates only) | 0.5 day |
| T1 | < 1 h | 0.5–1 day |
| X0 | 1–2 h | 1 h (wrapper only) |
| X1 | 2–4 h (pilot timing first) | 1 day |
| H2 | ≈ 2 h | 1 day (reuses the R4 script structure) |
| I1 | ≈ 4 h | 0.5 day after H2 |
| X2 | ≈ 1.2 h | 0.5 day |
| X3 pilot / full | 1 h / 2–3 h | 1 day |
| M6 test pass (H1 split + OWT/GoT test) | 1–2 h | protocol freeze |

**Prioritised plan (deadline 2026-12-08; instructor checkpoints 11-17, 11-22, 12-01):**
1. **By 11-17:** checkpoint backup; H1; T1(a) MDE; X0 → the first answers with no training.
2. **By 11-22:** X1 (pilot, then 3 seeds), T1(b–d); decide on X2 using X1-(a)'s γ.
3. **By 12-01:** H2, then I1 if H2 shows no guard failure; optionally the X3 pilot. Freeze and commit the M6 protocol;
   run M6 once.
4. **12-01 → 12-08:** write-up only.

---

## 12. Source files supporting the conclusions

**Code:**
* `src/sae/modern_sae.py`
* `scripts/openwebtext/{03_training_sae,13_consistent_protocol_retrain,17_streaming_sae_training,18_kl_finetune_fixed_l0}.py`
* `scripts/shared/{sae_ladder_evaluation,spliced_reconstruction_evaluation,m2_four_level_faithfulness,m5_confirmatory_test,r2_comparisons}.py`
* `scripts/repe/{24_sae_patch_decomposition,26_repe_lm_representation_preservation,27_causal_subspace_skyline}.py`
* `src/evaluation/{faithfulness_metrics,information_levels}.py`

**Results:**
* `results/shared/m5_confirmatory_test/{m5_rows,comparisons}.csv`
* `results/openwebtext/r2_scaled_saes/{README.md,ladder/*.csv,training_info.json}`
* `results/openwebtext/r4_kl_finetune/{README.md,finetune_info.json}`
* `results/repe/r5_causal_subspace_skyline/summary.csv`
* `results/shared/m2_four_level_faithfulness/`
* `results/repe/{sae_patch_decomposition,e3_finetuned_causal_preservation,repe_lm_representation_preservation}/`
* `results/geometry_of_truth/{method_a_layer6,truth_aware_pilot,e2e_inspired_layer6,kl_weight_ablation_layer6_topk,multilayer_finetune,final_test_evaluation}/`

**Docs:**
* `docs/results_registry.md`
* `docs/final_defense/{FINAL_RESEARCH_REPORT,EXPERIMENT_AUDIT}.md`
* `docs/2026-11-10_independent_research_review.md`
* `docs/2026-11-07_truth_aware_finetuning_audit.md`

## 13. External references

**New in this document (checked 2026-11-12):**
* Rajamanoharan et al. 2024, *Improving dictionary learning with Gated SAEs* (NeurIPS 2024). https://arxiv.org/abs/2404.16014
* GDM mech-interp team 2024, *Progress update 1* (inference-time optimisation / gradient pursuit). https://www.lesswrong.com/posts/C5KAZQib3bzzpeyrg
* Wright & Sharkey 2024, *Addressing feature suppression in SAEs*. https://www.alignmentforum.org/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes
* Anthropic 2025, *Circuits updates, January 2025* (JumpReLU training). https://transformer-circuits.pub/2025/january-update/index.html
* Chalnev, Siu & Conmy 2024, *Improving steering vectors by targeting SAE features*. https://arxiv.org/abs/2411.02193
* Arad, Mueller & Belinkov 2025, *SAEs are good for steering – if you select the right features* (EMNLP 2025). https://arxiv.org/abs/2505.20063
* Huang et al. 2024, *RAVEL*. https://arxiv.org/abs/2402.17700

**Cited from the 2026-11-10 review (links checked then; not re-checked):**
* Gao et al. 2024 (2406.04093)
* Rajamanoharan et al. 2024b, JumpReLU (2407.14435)
* Bussmann et al. 2024, BatchTopK (2412.06410)
* Bussmann et al. 2025, Matryoshka (2503.17547)
* Karvonen 2025 (2503.17272)
* Chen, Engels & Tegmark 2025 (2501.19406)
* Engels et al. 2024 (2410.14670)
* Marks et al. 2024 (2403.19647)
* O'Neill et al. 2024 (2411.13117)
* Kantamneni et al. 2025 (2502.16681)
* Korznikov et al. 2026 (2602.14111)
* Chaudhary & Geiger 2024 (2409.04478)
* Bürger et al. 2024 (2407.12831)

**Standard references (not re-checked):**
* Marks & Tegmark 2023, *Geometry of Truth* (2310.06824)
* Zou et al. 2023, *Representation Engineering* (2310.01405)

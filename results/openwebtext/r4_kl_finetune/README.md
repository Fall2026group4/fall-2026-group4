# R4: short KL+MSE fine-tune at fixed L0 (TopK), with an MSE-only control. Status: COMPLETED (VALIDATION, 2026-11-10); TEST in M5

Evidence level: **VALIDATION** (evaluation with the R2 ladder); OWT TEST only in the frozen M5 pass.

## 1. Before the run

* **Problem:**
  * Our IMP-L/E3 output-KL fine-tunes improved behaviour, but for ReLU/JumpReLU every gain came with L0 drift
    (5–8 %), so they were not creditable.
  * TopK (fixed L0) showed no causal change, but on tiny, data-starved SAEs with wide intervals.
  * The literature reports that a short KL+MSE fine-tune closes 20–50 % of the CE gap (Karvonen 2025,
    arXiv 2503.17272).
* **Question:** on strong, data-scaled SAEs (R2 TopK-12288, L0 = 128 exactly), does a short KL fine-tune improve
  behavioural faithfulness, and does that carry over to information and causal faithfulness, relative to an MSE-only
  continued-training control on identical tokens?
* **Design:**
  * R2 TopK-12288 at **layers 8 and 11** (the largest splice damage), seeds 42/1042/2042;
  * arms `kl` (NMSE + α·KL(p_orig‖p_spliced), α by first-batch scale matching) and `mse` (NMSE only);
  * 5M tokens (≈ 5 % of R2's budget), OWT shard 79 chunks ≥ 10,000 (disjoint from R2 training, the R2 monitor and our
    sample);
  * Adam lr 2e-5, clip 1.0, decoder renormalised, normalised coordinates;
  * SAE spliced at every non-BOS position;
  * script `scripts/openwebtext/18_kl_finetune_fixed_l0.py`.
* **Evaluation:** the full ladder (`scripts/shared/sae_ladder_evaluation.py`, family `r4_kl` / `r4_mse`), same
  metrics as R2.
* **Pre-registered comparisons** (per layer, paired by seed, IMP-L rule: better on all 3 seeds **and**
  |mean paired difference| > 2 sd):

  | ID | Comparison | Metrics |
  |---|---|---|
  | K1 | `kl` vs `mse` | spliced ΔCE and KL (behaviour, primary) |
  | K2 | `kl` vs `mse` | re-trained persona and GoT gaps; frozen gaps |
  | K3 | `kl` vs `mse` | T_rec/T_full at the layer's M4 site (causal) |
  | K4 | `kl` vs base R2 | ΔCE, KL (net effect) |

  * **Guard:** OWT EV of `kl` ≥ `mse` − 0.02 (the layer-6 rule).
  * L0 is fixed by construction (TopK).
  * Classes: PRELIMINARY IMPROVEMENT / NO IMPROVEMENT / REGRESSION / NOT CREDITABLE (guard).
* **Compute:** about 1–1.5 h GPU after R2 training ends.

### 1.1 Smoke test (2026-11-10, before registration)

* **Run:** L8 seed 42, 20 steps per arm, written to the scratchpad. The smoke numbers are not results.
* **Checks:**
  * GPT-2 parameters are unchanged;
  * L0 = 128.0 throughout;
  * the scale fold/unfold round-trip is exact to 1.1e-5;
  * the fine-tuned checkpoint loads through `fm.load_sae`.
* **α:** first-batch scale matching gave α = 1.47.
* **Changes:** none to the design.

## 2. Results (VALIDATION, 2026-11-10)

### 2.1 Run

* **Fine-tunes:** 12 (2 layers × 3 seeds × 2 arms), 4,998,720 tokens each. Runtime 71 min.
* **Checks:** GPT-2 parameters unchanged; L0 = 128 exactly throughout.
* **α (first-batch scale matching):** 1.47–1.51 at L8 and 1.14–1.15 at L11.
* **Evaluation:** the shared ladder, written to `ladder/`.
  * The first ladder attempt was OOM-killed by concurrent jobs and rerun alone (RESEARCH_PROGRESS §34).
* **Analysis:** `scripts/openwebtext/19_r4_analysis.py` writes `comparisons.csv` and `criteria.json`.
* **Guard:** EV(kl) ≥ EV(mse) − 0.02 passes at both layers.

  | Layer | EV `kl` | EV `mse` | Difference |
  |---|---|---|---|
  | L8 | 0.9025 | 0.9044 | −0.0019 |
  | L11 | 0.8849 | 0.8874 | −0.0024 |

### 2.2 Results (3 seeds; mean of paired differences ± sd)

| Layer | ID | Metric | `mse` (or base) | `kl` | Δ ± sd | Improvement | Class |
|---|---|---|---|---|---|---|---|
| L8 | K1 | spliced ΔCE ↓ | 0.0601 | 0.0493 | −0.0107 ± 0.0011 | 17.9 % | PRELIMINARY IMPROVEMENT |
| L8 | K1 | spliced KL ↓ | 0.0677 | 0.0587 | −0.0089 ± 0.0002 | 13.2 % | PRELIMINARY IMPROVEMENT |
| L8 | K2 | persona re-fit gap ↓ | 0.1088 | 0.1061 | −0.0026 ± 0.0035 | — | NO IMPROVEMENT |
| L8 | K2 | GoT re-fit gap ↓ | 0.0485 | 0.0548 | +0.0063 ± 0.0034 | — | NO IMPROVEMENT |
| L8 | K2 | persona / GoT frozen gaps ↓ | 0.114 / 0.240 | 0.114 / 0.237 | −0.000 / −0.003 | — | NO IMPROVEMENT |
| L8 | K3 | T_rec/T_full ↑ | 0.720 | 0.710 | −0.011 ± 0.016 | — | NO IMPROVEMENT |
| L8 | K4 | spliced ΔCE vs R2 base ↓ | 0.0603 | 0.0493 | −0.0110 ± 0.0007 | 18.2 % | PRELIMINARY IMPROVEMENT |
| L11 | K1 | spliced ΔCE ↓ | 0.0997 | 0.0814 | −0.0183 ± 0.0012 | 18.4 % | PRELIMINARY IMPROVEMENT |
| L11 | K1 | spliced KL ↓ | 0.1030 | 0.0858 | −0.0172 ± 0.0005 | 16.7 % | PRELIMINARY IMPROVEMENT |
| L11 | K2 | persona re-fit gap ↓ | 0.1070 | 0.1106 | +0.0035 ± 0.0071 | — | NO IMPROVEMENT |
| L11 | K2 | GoT re-fit gap ↓ | 0.0936 | 0.1017 | +0.0081 ± 0.0103 | — | NO IMPROVEMENT |
| L11 | K3 | T_rec/T_full ↑ | 0.694 | 0.671 | **−0.024 ± 0.002** | — | **REGRESSION** |
| L11 | K4 | spliced ΔCE vs R2 base ↓ | 0.0995 | 0.0814 | −0.0181 ± 0.0012 | 18.2 % | PRELIMINARY IMPROVEMENT |

Descriptive changes, `kl` against `mse`:

* **Pair-direction cosine:** −0.013 at L8 (0.487 → 0.474) and −0.012 at L11 (0.512 → 0.500), on every seed.
* **NMSE:** about +0.002.

### 2.3 Inference

1. **The KL fine-tune works for behaviour, at fixed L0.**
   * About 18 % less splice damage at both layers, on every seed, against an MSE control on identical tokens.
   * This is consistent with Karvonen 2025, who report 20–50 % of the CE gap closed.
   * Unlike IMP-L's ReLU/JumpReLU results, it is creditable: L0 cannot drift and the EV guard passes.
2. **The gain does not carry over.**
   * It does not reach information (K2: no change), nor the representation of the persona contrast (pair-direction
     cosine slightly worse).
   * At L11 it **reduces** causal persona preservation (−0.024, all seeds).
   * Optimising the next-token KL on generic OWT text reallocates reconstruction capacity toward what matters for
     average next-token prediction. That is not what carries the persona instruction.
3. **Faithfulness levels dissociate** — one more instance, now with a positive behavioural effect. A behavioural
   objective is not a proxy for causal or information faithfulness.
4. **Decision:** keep the KL fine-tune as a behaviour-only improvement, reported with its causal cost. Do not adopt it
   as the default SAE for the causal and information analyses. For Gemma it is a cheap add-on, only where splice
   behaviour is the target.

### 2.4 Limitations

* VALIDATION only; M5 tests K1/K4 on OWT TEST.
* Two layers, TopK only, one learning rate and budget. The α schedule and a longer budget were not explored.
* The K3 regression at L8 is not significant: −0.011 ± 0.016.

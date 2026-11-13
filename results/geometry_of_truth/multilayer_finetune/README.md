# IMP-L: multi-layer improvement experiment, layers 3, 8 and 9 (2026-11-09). Status: COMPLETED (VALIDATION)

Evidence level: validation finding. No test row is read by training or evaluation.

## 1. Before implementation

* **Research question:** do the layer-6 fine-tuning methods improve SAE faithfulness at other layers (3, 8, 9),
  relative to the original SAEs and to an OWT-only continued-training control, at matched sparsity?
* **Hypotheses:**
  * Method A lowers the independent GoT truth gap;
  * output-KL lowers OWT and GoT spliced KL relative to Method A.

  Both were met only partly, or not at all, at layer 6.
* **Why it's needed:**
  * Instructor Review #8's blocker for the 11-10 session: "one table: which fine-tuning method, at which layers,
    improves which faithfulness measure, with seeds and CIs … run it at two more layers (e.g. 3 and 9)".
  * The 11-03 session note: "fine-tune other layers".
  * The main research question says "across transformer layers"; every fine-tune so far is at layer 6.
  * Layer 8 is also the M4 mid-depth causal site, so the layer-8 SAEs feed E3 (causal preservation after fine-tuning).
* **Design:**
  * **Model and hook:** GPT-2 Small, frozen, at `blocks.L.hook_resid_pre`, L ∈ {3, 8, 9}.
  * **SAEs:** ReLU, TopK and JumpReLU; the uninterrupted 15k-step checkpoints, seeds 42/1042/2042.
  * **Arms:**

    | Arm | Objective |
    |---|---|
    | base | none |
    | `owt_only` | control: 500 steps on OWT only |
    | `method_a` | 0.5·MSE(GoT-A) + 0.5·MSE(OWT) + penalty |
    | `kl` | Method A + α·0.5·(KL_OWT + KL_GoT), KL(p_orig ‖ p_spliced), with α per base SAE from train-only scale matching at step 0 |

    The `kl` arm is the layer-6 E2E Method B.
  * **Budget:** Adam lr 1e-5, 500 steps, clip 10, decoder renormalisation. The controller continues from the base
    coefficient (target 128). TopK k = 128.
  * **Runs:** 81 new runs = 3 layers × 3 arms × 3 architectures × 3 seeds.
  * **Code reuse:** the training code is imported unchanged from scripts 14 and 17; only their layer constants are set.
* **Data:**
  * training uses OWT train and GoT train groups A only, with no labels;
  * evaluation uses OWT validation, GoT validation (geographic and numerical) and RepE-LM validation (persona
    T1–T4, H vs U).
* **Comparisons:**

  | X vs Y | What it establishes |
  |---|---|
  | `method_a` vs base | does GoT reconstruction fine-tuning improve truth preservation at this layer? |
  | `method_a` vs `owt_only` | is the effect due to the GoT data, rather than to 500 extra steps? |
  | `kl` vs `method_a` | does the output-KL term add behavioural faithfulness, and at what truth or sparsity cost? |
  | `kl` vs base | net effect of the full objective |

* **Metrics** (direction in brackets):
  * **reconstruction:** OWT EV (↑), MSE (↓), cosine (↑), L0 (match), dead fraction (↓); GoT relative error (↓);
    L0 on GoT and RepE-LM prompts (reported);
  * **representation:** GoT B-group MLP and logistic gaps (frozen, fitted on original B rows; ↓); M2-style
    re-trained logistic gap on GoT geographic and RepE-LM persona (↓);
  * **behavioural splicing:** OWT ΔCE and KL (↓), GoT statement KL (↓), cities truth-preference change (↑).
* **Pre-registered criteria and guards** (`preregistration.json`):
  * "consistent" = better on all 3 seeds and |mean paired difference| > 2 × its sd;
  * guards: OWT L0 within ±5 % of base; OWT EV ≥ base − 0.005 (kl: ≥ Method A − 0.02); ΔCE ≤ 1.10 × base; cities
    preference ≥ base − 0.018;
  * classes: PRELIMINARY IMPROVEMENT / NO IMPROVEMENT / NOT CREDITABLE (guard failed) / REGRESSION. CONFIRMED is not
    used, because this is validation only.
* **Statistics:** per layer × architecture, paired by seed. Mean, sd, per-seed values, and a 95 % t-interval over
  the 3 seeds (t₀.₉₇₅,₂ = 4.303; wide by construction). No pooling across layers.
* **Layer-3 caveat:** the original GoT logistic probe is below the 0.70 informative threshold at layer 3, so
  layer-3 truth results are descriptive.
* **Leakage prevention:**
  * the A/B split hash is verified (`S14.verify_split`);
  * probes are fitted on original activations of groups B only;
  * GoT test rows are dropped before scoring;
  * RepE-LM contains no test rows.
* **Checks before pre-registration** (`--prepare` → `pretraining_checks.json`):
  * the reuse path reproduces Method A TopK seed 42 and Method B TopK seed 42 at layer 6 **bit for bit**;
  * cached GoT-A activations equal a fresh forward pass at each layer;
  * the identity splice gives KL = 0.

  The evaluation code was smoke-tested on the 9 layer-8 base SAEs (outputs outside the repository). It reproduced
  M2's committed layer-8 re-trained gaps exactly.
* **Files:**
  * `scripts/geometry_of_truth/22_multilayer_finetune.py`, `scripts/geometry_of_truth/23_multilayer_evaluation.py`;
  * this folder;
  * checkpoints (gitignored) in `results/geometry_of_truth/sae_finetuning/multilayer_finetune/`.
* **Pre-training hashes:** `pretraining_hashes.json`.

## 2. After execution

**Run:** approved by the user on 2026-11-09. Scripts 22 (`--train`, 81 runs) and 23 ran unchanged; their SHA-256
hashes equal `pretraining_hashes.json`. The pre-registered criteria were not changed after results were seen.
* **Training:** 81/81 runs completed (layer 3: 1474 s, layer 8: 1267 s, layer 9: 1225 s). Peak GPU memory was 6.9 GB.
  `base_checkpoints_unchanged` and `gpt2_parameters_unchanged` are true at every layer (`training_info_layer{3,8,9}.json`).
* **Evaluation:** 108 SAEs (36 per layer), 3975 s, peak 8.3 GB. `test_rows_used: 0` (`eval_info_layers3_8_9.json`).
* **Outputs:**
  * `per_sae_layer{3,8,9}.csv`, `summary.csv`;
  * `comparisons.csv` (all paired differences, sd, 95 % t-interval, % change, per-seed values);
  * `improvement_table.csv`, `criteria.json` (pre-registered classes and guards);
  * `training_monitor_layer{3,8,9}.csv`.

Checkpoints (gitignored) are in `results/geometry_of_truth/sae_finetuning/multilayer_finetune/`. The logs are local only.

### 2.1 Method A: truth (criterion T_method_a: B-MLP gap lower than both base and OWT-only, consistent)

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

* **Method A vs base:**
  * Method A lowered the independent truth gap on all 3 seeds in **9/9** cells.
  * The criterion (consistent against both base and OWT-only) was met in **4/9 cells**: L3 TopK, L8 ReLU, L9 ReLU
    and L9 JumpReLU.
  * In the other 5 cells the direction is the same, but the seed spread exceeds the 2 sd rule.
* **OWT-only control:** it changed the gap by at most 0.021 (L8 TopK, +0.021, upward), so the reduction comes from
  the GoT data.
* **Guards:** every Method A guard passed in every cell. OWT EV changed by +0.001 at most; ΔCE and KL slightly lower.
* **Re-trained readers (reported, not a criterion):**
  * The M2-style re-trained logistic gap (GoT geographic; RepE-LM persona) shows **no consistent change** vs base in
    any cell: all 95 % intervals include 0, and none meets the 2 sd rule.
  * Against OWT-only, one cell is consistently lower: L8 TopK, geographic, −0.027.
  * Method A therefore restores what the frozen original-activation probe reads, without measurably restoring the
    information a re-fitted reader recovers (descriptive; consistent with M2's fragility finding).
* **Layer 3:**
  * The pre-registered informativeness rule (original B-MLP AUROC ≥ 0.70) is met: 0.815.
  * The original logistic probe reaches only 0.60.
  * JumpReLU at layer 3 has 55 % dead latents in its base SAE.

### 2.2 Output-KL: behaviour (B_kl: OWT KL and GoT statement KL lower than Method A, consistent) and truth (T_kl)

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

* **Behaviour:**
  * The KL term lowered OWT spliced KL (8–18 %) and GoT statement KL vs Method A **consistently in all 9 cells**.
  * It also lowered ΔCE by 4–19 %, and OWT EV changed by −0.003 to +0.005.
* **Guard failures:**
  * For ReLU and JumpReLU the KL term raised OWT L0: +3.5 % and +4.6 % at layer 3 (inside the ±5 % guard), and
    +5.3 to +8.1 % at layers 8 and 9 (**outside the guard**).
  * Those 4 cells are **NOT CREDITABLE** and excluded from rankings.
  * TopK (L0 fixed by construction) passes every guard at every layer.
* **Classes:**
  * Behaviour PRELIMINARY IMPROVEMENT: all 3 architectures at layer 3, and TopK at layers 8 and 9 (5/9).
  * NOT CREDITABLE: 4/9.
* **Truth:**
  * Output-KL never lowered the truth gap vs Method A.
  * **REGRESSION** (consistent, worse) at L3 ReLU (+0.012) and L3 TopK (+0.027). Both 95 % intervals just reach 0.
  * L9 ReLU is worse on all 3 seeds (+0.033) but is classed NOT CREDITABLE because of its L0 guard.
  * All other cells: NO IMPROVEMENT.

### 2.3 Pre-registered criteria: summary

| Criterion | Met | Not met (guards passed) | Not creditable | Regression |
|---|---|---|---|---|
| T_method_a (Method A truth) | 4/9 | 5/9 | 0 | 0 |
| B_kl (output-KL behaviour) | 5/9 | 0 | 4/9 (L0 guard) | 0 |
| T_kl (output-KL truth vs Method A) | 0/9 | 3/9 | 4/9 (L0 guard) | 2/9 (layer 3) |

No result is CONFIRMED (validation only, three seeds).

### 2.4 Limitations

* **Seeds:** three seeds that share training data. The 95 % t-intervals are wide, and seed spread understates
  uncertainty.
* **Single hyper-parameter setting:** the budget is 500 steps at lr 1e-5, as at layer 6. No tuning was done.
* **Frozen-probe criterion:** the truth criterion uses a probe fitted on original activations. The re-trained reader
  disagrees (§2.1).
* **L0 drift:** output-KL's L0 drift at deeper layers is a property of the multiplicative controller under the KL
  gradient. An L0-matched variant (for example a stronger controller, or TopK only) is untested.
* **Causal preservation:** not yet measured. E3 (M4 protocol on the layer-8 SAEs) needs its own pre-registration.

### 2.5 Status

COMPLETED (VALIDATION). Paper: `research_paper/Latex/sections/10_finetuning.tex` §10.1, with tables
`t_impl_summary` and `t_impl_layer{3,8,9}` and figure `fig_impl_multilayer` (all generated by
`scripts/shared/build_paper_artifacts.py`).

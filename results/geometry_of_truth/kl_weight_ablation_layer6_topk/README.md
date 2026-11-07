# KL-weight ablation, GPT-2 Small layer 6, TopK (2026-11-07). Status: COMPLETED, VALIDATION. Pre-registered verdict: **INCONCLUSIVE**

Roadmap item G8, part (Method A + small-KL combination).

## 1. Before implementation

**Question.** Can an intermediate output-KL weight keep Method A's independent truth-probe performance while retaining part of the E2E-inspired Method B's behavioural gain?

**Why.**
* Method B (full KL) improved output KL and CE but did not improve the truth probe.
* If B traded truth for behaviour, an intermediate weight might give a better combination.
* This bears directly on the main question: whether one SAE can preserve both truth information and downstream behaviour.

**Design.**
* Frozen GPT-2 Small, `blocks.6.hook_resid_pre`, non-BOS positions.
* TopK SAE, L0 = 128, d_sae 3072.
* Seeds 42/1042/2042, each initialised from its **original uninterrupted TopK checkpoint** (verified).
* Protocol: Method B (script 17's `train_one`, imported unchanged):
  * local loss 0.5·MSE(GoT-A) + 0.5·MSE(OWT) on Method A's batches and RNG;
  * plus w · α_B(seed) · 0.5 · (KL_OWT + KL_GoT), with KL(p_orig ‖ p_spliced);
  * KL batches: 8 OWT train sequences plus the step's 128 GoT-A statements, generator seed + 5000, so the data order is identical for every w;
  * Adam, lr 1e-5, 500 steps, gradient clip 10.

**Coefficient semantics.**
* α_B is the **only** KL coefficient. It is the raw loss multiplier, set by train-only scale matching (`../e2e_inspired_layer6/alpha_beta.json`). The fractions scale α_B.
* The per-term weight is 0.5·α.

| w | Seed 42 α (per-term) | Seed 1042 α (per-term) | Seed 2042 α (per-term) | Source |
|---|---|---|---|---|
| 0 (Method A) | 0 | 0 | 0 | reused checkpoints `../sae_finetuning/truth_aware_pilot_layer6_topk/reconstruction_only_seed*` |
| 0.25 | 1.0375 (0.5188) | 0.9731 (0.4865) | 1.0530 (0.5265) | **new** |
| 0.50 | 2.0751 (1.0375) | 1.9462 (0.9731) | 2.1061 (1.0530) | **new** |
| 1 (Method B) | 4.1502 (2.0751) | 3.8924 (1.9462) | 4.2122 (2.1061) | reused checkpoints `../sae_finetuning/e2e_inspired_layer6/topk_methodB_seed*` |

**Fixed in advance:** fractions 0.25 and 0.50. They were not changed after seeing any result.

**Probes.** Independent GoT B-group probes (logistic, MLP × 5 probe seeds, mass-mean) fitted on ORIGINAL activations of train groups B. No fine-tuning teacher exists in this experiment. Validation data only; test rows were never read.

**Pre-registered outcomes and criteria** (`preregistration.json`):
* **Primary:**
  * GoT geographic B-MLP gap ↓;
  * OWT output KL ↓;
  * OWT spliced CE increase ↓;
  * TopK L0 = 128 on OWT, GoT and RepE.
* **Rules** (descriptive, n = 3, no significance claims):
  * **BEHAV:** KL and CE both "consistently" lower than A (all 3 seeds, and mean paired difference > 2 sd).
  * **TRUTH_NONINFERIOR:** mean paired gap(w) − gap(A) ≤ +0.02.
  * **TRADE:** on every seed, the w point lies strictly below the seed's A–B chord in (OWT KL, gap) space.
  * **Guards:** L0, OWT EV, `cities` preference, RepE mass-mean gap, GoT logistic gap.
* **Verdict:**
  * "Improves trade-off" requires all of the above.
  * If the endpoints don't differ in truth by > 0.02, the trade-off conclusion is INCONCLUSIVE.

**Verification before training** (`verification.json`):
* Split SHA-256 `d43e8383…` matches.
* All six of the E2E experiment's recorded pre-training hashes still match.
* Base, A and B checkpoint provenance (paths, SHA-256, sources, α, λ) is correct.
* Saved base/A/B validation results reproduce exactly from their checkpoints (max |Δ| = 0.0).

**Pre-training hashes:** `pretraining_hashes.json` (scripts 20, 21 and 17, the shared evaluation module, the split file, `alpha_beta.json`, the pre-registration and the verification).

**Commands** (run from the repository root, with the repo venv):
```
python scripts/geometry_of_truth/20_kl_weight_ablation.py --verify
python scripts/geometry_of_truth/20_kl_weight_ablation.py --prepare
python scripts/geometry_of_truth/20_kl_weight_ablation.py --dry-run
python scripts/geometry_of_truth/20_kl_weight_ablation.py --train
python scripts/geometry_of_truth/21_kl_weight_ablation_evaluation.py
```

## 2. Execution

* **Reproduction check:** w = 1.0, seed 42 retrained with this script is **bit-identical** to the Method B checkpoint (max |Δparam| = 0.0). It is excluded from results.
* **Runs:** 6/6 completed, 145 s each. Training wall time 17 min, including the reproduction check; GPU peak 5.9 GB. Evaluation of 15 SAEs: 3.1 min, GPU peak 8.3 GB.
* **Integrity:** GPT-2 parameters, and every base, A and B checkpoint, verified unchanged. Evaluation of base/A/B matches the verification exactly.
* **Checkpoints:** `../sae_finetuning/kl_weight_ablation_layer6_topk/topk_kl{025,050}pct_seed*/` (gitignored; SHA-256 in `training_info.json` and `docs/checkpoint_registry.csv`).

## 3. Results (validation; seed values, then mean ± sd)

**Primary outcomes:**

| Metric | Seed | A (w=0) | w=0.25 | w=0.50 | B (w=1) |
|---|---|---|---|---|---|
| GoT geographic B-MLP gap ↓ | 42 / 1042 / 2042 | .112 / .085 / .124 | .100 / .106 / .128 | .095 / .113 / .127 | .097 / .106 / .139 |
| | mean ± sd | **.107 ± .020** | **.111 ± .015** | **.112 ± .016** | **.114 ± .022** |
| OWT output KL ↓ | 42 / 1042 / 2042 | .1155 / .1170 / .1131 | .1041 / .1053 / .1022 | .1029 / .1040 / .1012 | .1020 / .1034 / .1006 |
| | mean ± sd | **.1152 ± .0019** | **.1038 ± .0016** | **.1027 ± .0014** | **.1020 ± .0014** |
| OWT spliced CE increase ↓ | 42 / 1042 / 2042 | .0997 / .1048 / .1013 | .0900 / .0950 / .0934 | .0888 / .0936 / .0922 | .0880 / .0930 / .0920 |
| | mean ± sd | **.1019 ± .0026** | **.0928 ± .0026** | **.0915 ± .0024** | **.0910 ± .0027** |
| L0 on OWT / GoT / RepE | all seeds | 128 / 128 / 128 | 128 / 128 / 128 | 128 / 128 / 128 | 128 / 128 / 128 |

**Paired differences vs A** (mean ± sd over seeds):

| | Geographic MLP gap | OWT KL | CE increase |
|---|---|---|---|
| w=0.25 | +.004 ± .017 | −.0114 ± .0004 | −.0092 ± .0010 |
| w=0.50 | +.005 ± .022 | −.0125 ± .0005 | −.0104 ± .0011 |

**Share of Method B's gain retained** ((A − w) / (A − B), per seed):

| | OWT KL | OWT CE | GoT statement KL |
|---|---|---|---|
| w=0.25 | 0.85 / 0.86 / 0.87 | 0.83 / 0.83 / 0.85 | 0.99 / 0.99 / 0.99 |
| w=0.50 | 0.94 / 0.95 / 0.95 | 0.93 / 0.95 / 0.98 | ≈ 1.00 |

**Secondary outcomes** (mean; A → 0.25 → 0.50 → B):

| Metric | A | 0.25 | 0.50 | B | Note |
|---|---|---|---|---|---|
| GoT numerical B-MLP gap ↓ | .059 | .059 | .066 | .074 | sd .013–.027 |
| GoT geographic B-logistic gap ↓ | .244 | .252 | .251 | .244 | sd .03–.05 |
| RepE logistic gap ↓ | .080 | .088 | .088 | .090 | |
| RepE mass-mean gap ↓ | .114 | .118 | .117 | .122 | |
| RepE direction cosine ↑ | .843 | .846 | .846 | .845 | |
| RepE persona-shift cosine ↑ (splicing, not steering) | .770 | .765 | .760 | .743 | sd .02–.04 |
| OWT validation MSE ↓ | .6592 | .6639 | .6669 | .6701 | |
| OWT validation EV ↑ | .8544 | .8534 | .8528 | .8520 | |
| GoT relative reconstruction error ↓ | .0062 | .0070 | .0078 | .0094 | |
| Downstream NMSE, layers 7–11 ↓ | .0944 | .0923 | .0924 | .0926 | |
| GoT statement KL ↓ | .110 | .038 | .038 | .038 | |
| `cities` preference change | −.015 | −.018 | −.015 | −.015 | |
| OWT dead fraction | .1483 | .1497 | .1497 | .1497 | |

**Pre-registered criteria** (`criteria.json`):

| Criterion | w = 0.25 | w = 0.50 |
|---|---|---|
| BEHAV | pass | pass |
| TRUTH_NONINFERIOR | pass (+.004) | pass (+.005) |
| TRADE (below chord on every seed) | **fail**: below chord on 1/3 seeds (2042); seed 42 above by .0003 | **fail**: below chord on 2/3 seeds (42, 2042) |
| Guards | all pass | all pass |
| **Verdict** | **INCONCLUSIVE** | **INCONCLUSIVE** |

The two endpoints differ in truth gap by only +0.007 (B − A, mean), below the 0.02 threshold, so there is no measurable truth trade-off for an intermediate weight to improve.

**Figures:**
* `../figures/kl_weight_ablation_truth_vs_kl.png`: truth gap vs output KL, per seed with chords, plus the mean.
* `../figures/kl_weight_ablation_metrics_vs_w.png`.

## 4. Interpretation

* **Truth: no conclusion about a truth/behaviour trade-off is possible here.** For TopK, the full KL weight (B) did not measurably worsen the independent truth gap relative to A (+0.007, paired sd 0.02). The intermediate weights sit between the endpoints in the mean, within seed noise. The seed-level ordering is not consistent: seed 42 improves with KL, seed 1042 worsens, seed 2042 is mixed.
* **Behaviour saturates early:**
  * w = 0.25 already gives about 85 % of B's OWT KL gain, about 83–85 % of its CE gain and about 99 % of its GoT statement-KL gain.
  * w = 0.50 gives about 95 %.
* **What the extra weight costs.** It buys little behaviour, while reconstruction error grows monotonically with w:
  * GoT relative error +13 % at 0.25 vs +52 % at B, relative to A;
  * OWT EV −0.0010 vs −0.0024.

  All these costs stay inside the pre-registered guards.
* **Secondary truth/honesty metrics:** numerical MLP gap, RepE gaps and persona-shift cosine drift slightly in the unfavourable direction as w grows. The changes are within seed sd and not interpretable at n = 3.
* **Bottom line.** On behaviour vs *reconstruction*, w = 0.25 is the most efficient setting tested. That is an observation on secondary outcomes, not a pre-registered success. On behaviour vs *truth*, the evidence is inconclusive.

## 5. Limitations and possible confounders

* **Three seeds.** The truth gap varies across seeds by about 0.015–0.022 (sd), larger than every between-method difference observed here.
* **Probe variance.** The MLP probe is averaged over 5 probe seeds, but its variance still contributes to the gap's sd.
* **Scope:** TopK only (L0 fixed by construction, which is why no sparsity confound exists here); layer 6 only.
* **Budget:** 500 steps at lr 1e-5, as in Methods A/B. The train-monitor KL was still falling at step 500 in the E2E experiment.
* **Data:** the KL is trained partly on GoT-A statements, close to the GoT validation distribution. This explains the near-complete GoT statement-KL saturation.
* **Validation only:** no test evaluation (by design).
* **Fractions:** only two intermediate values, chosen in advance. The saturation suggests the informative range is below 0.25, but this was not tested and is not claimed.

## 6. Files

* **Scripts (new):**
  * `scripts/geometry_of_truth/20_kl_weight_ablation.py` (verify, prepare, dry-run, train);
  * `scripts/geometry_of_truth/21_kl_weight_ablation_evaluation.py` (evaluation, criteria, figures).
* **Results (new):**
  * `verification.json`, `reproduction_of_saved_results.csv`, `preregistration.json`, `pretraining_hashes.json`;
  * `training_info.json`, `training_monitor.csv`;
  * `per_sae.csv`, `summary.csv`, `paired_differences.csv`, `criteria.json`, `eval_info.json`;
  * this README; two figures.
* **Unchanged:** script 17 and every existing checkpoint and result.

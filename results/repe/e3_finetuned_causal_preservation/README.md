# E3: causal preservation of the fine-tuned layer-8 SAEs (M4 protocol). Status: COMPLETED (VALIDATION)

Evidence level, once run: **VALIDATION**. No test row is read.

* **Approval:** the user approved running E3 on 2026-11-10 ("Complete causal faithfulness evaluation (E3) ... Do not change its success criteria after observing results").
* §1.1–§1.7 are the draft of 2026-11-09, unchanged. §1.8 adds implementation details, fixed after a mechanics-only smoke test and before any E3 result was seen.

## 1. Before implementation

### 1.1 Question, motivation and hypotheses

**Question.** Does fine-tuning that improves task reconstruction (Method A) or output agreement (output-KL) also raise
the share of GPT-2's persona-instruction effect that the SAE reconstruction carries at the layer-8 final token?

**Motivation:**
* M4 found that the base SAEs carry only 62–67 % of the effect at L8 final.
* IMP-L showed that fine-tuning improves the level it optimises, but not truth information read by a re-trained reader.
* Instructor Review #8 asks for SAE-in-the-loop and causal evidence after improvement.
* This is the causal level of the main research question.

**Hypotheses.** Direction is two-sided; improvement and regression are both reported.

| ID | Hypothesis | Comparison | Primary metric |
|---|---|---|---|
| E3-H1 | Method A raises causal preservation | `method_a` vs `base` **and** `method_a` vs `owt_only` | `preservation_rec` = T_rec / T_full |
| E3-H2 | Output-KL raises causal preservation over Method A | `kl` vs `method_a` | `preservation_rec` |
| E3-H3 (specificity) | Any gain in E3-H1/H2 is persona-specific, not generic | same pairs | `rec_minus_syn_rec` = T_rec(persona) − T_rec(synonym) |

* **Prior expectation (stated before running):**
  * Method A is trained on GoT statements and OWT, not on RepE prompts, so we expect **no change** for E3-H1.
  * Output-KL matches next-token distributions on OWT/GoT, so E3-H2 is plausible but uncertain.

### 1.2 Design

* **Site:** `blocks.8.hook_resid_pre`, final token. This is the M4 amendment-A1 site, selected on train prompts and
  replicated on validation without SAEs.
* **SAEs (36, all existing; no training):**
  * base: 9, the uninterrupted 15k-step SAEs, ReLU/TopK/JumpReLU × seeds 42/1042/2042;
  * IMP-L layer-8 fine-tunes: `owt_only`, `method_a` and `kl`, 9 each, in
    `results/geometry_of_truth/sae_finetuning/multilayer_finetune/layer_8/`.
* **Data:** RepE-LM validation.
  * Primary set: T2–T4 persona pairs.
  * Controls: T1 (control case), synonym pairs (H vs Hp), neutral pairs (Na vs Nb). This is the M4 row set
    (356 persona + 178 neutral pairs).
  * Feature ranking uses TRAIN T2–T4 only, exactly as in M4.
* **Code:**
  * New script `scripts/repe/25_e3_finetuned_causal_preservation.py`.
  * It **imports** `Engine`, `build_rows`, `clean_batches`, `decompose`, `add_derived`, `summarise_decomp`,
    `feature_analysis` and `owt_feature_stats` from `scripts/repe/24_sae_patch_decomposition.py`, unchanged.
  * It sets the site to (8, 'final') and loads the 36 checkpoints. There is no new metric code.

### 1.3 Metrics (direction)

* **Primary:** `preservation_rec` = T_rec / T_full (↑).
* **Secondary:**
  * Shapley share of the reconstruction, `shapley_share_rec` (↑);
  * T_err / T_full (↓);
  * |I| / T_full, the interaction (↓; additive reading allowed when small);
  * `rec_minus_syn_rec` (↑, specificity);
  * the H4a/H4b/H4c passes of M4 (T_rec above matched-norm and decoder-shuffle controls; feature patches beat 200
    matched random feature sets);
  * k* and the feature share at k*.
* **Also reported:** T_full, which must be identical across SAEs.

### 1.4 Statistics and pre-registered criteria

* **Per SAE:** statement-grouped bootstrap (2000) CIs, as in M4.
* **Per architecture:** paired by seed (3 seeds). Mean paired difference, sd and 95 % t-interval (t = 4.303, df = 2).
* **"Consistent":** better on all 3 seeds **and** |mean paired difference| > 2 sd (the IMP-L rule).
* **Classes, per architecture × hypothesis:** PRELIMINARY IMPROVEMENT / NO IMPROVEMENT / NOT CREDITABLE / REGRESSION,
  with exactly the IMP-L definitions.
  * E3-H1 needs consistency against **both** base and OWT-only.
  * A gain that fails E3-H3 is labelled "non-specific".
* **Guards:** inherited from IMP-L `criteria.json` (L0 ±5 %, OWT EV, ΔCE, cities preference), plus RepE-LM-prompt L0
  reported at the site.
  * **Stated in advance:** the output-KL ReLU and JumpReLU layer-8 SAEs already failed the L0 guard in IMP-L.
  * Their E3-H2 results are therefore **NOT CREDITABLE a priori**. They are reported, but only TopK can earn a
    creditable output-KL class.
* No pooling across architectures. No claim that GPT-2 "uses" SAE features.

### 1.5 Checks before results are read (written to `sanity_checks.json`)

* **Reproduction:** the 9 base-SAE rows must reproduce M4's L8-final values in `hypotheses.csv` exactly. Same seeds,
  same code; tolerance 1e-6.
* Δ = Δ_rec + Δ_err identity: max error < 1e-3.
* `start_at_layer` with v = 0 gives KL = 0.
* Shapley terms sum to T_full.
* T_full is identical across all 36 SAEs.
* `test_rows_used` = 0.
* SHA-256 hashes of scripts 24 and 25, `src/evaluation/patching.py`, this pre-registration and the 36 checkpoints are
  recorded before the run.

### 1.6 Compute

* **No training. No download.**
* **Estimate:**
  * M4 took 1670 s for 27 SAE evaluations with features, about 60 s per SAE on the A10G.
  * E3 is 36 SAEs at one site, about **35–45 min GPU**.
  * Peak memory is expected to be as in M4: well under the A10G's 23 GB, since there is no SAE larger than 3072 latents.

### 1.7 Outputs

* `per_pair_validation.csv`, `decomposition_summary.csv`, `features_summary.csv`, `comparisons.csv`, `criteria.json`
  and `sanity_checks.json`.
* README §2.
* A paper subsection in §10 (E3).

### 1.8 Implementation details (fixed before any result; hashed in `preregistration.json`)

* **Script:** `scripts/repe/25_e3_finetuned_causal_preservation.py` (`--prepare`, `--smoke`, `--run`). It imports the
  M4 functions unchanged. The SAE reconstruction difference is added inside GPT-2's forward pass at the site
  (`start_at_layer` from the clean U residual), so this is SAE-in-the-loop, not direction steering.
* **Checkpoints:** the 27 fine-tunes are verified against the SHA-256 values recorded by the IMP-L evaluation
  (`per_sae_layer8.csv`), and all 36 hashes go into `preregistration.json`.
* **Seeding:** the generator, the numpy rng and the CPU shuffle generator are re-seeded (20261108) before every SAE,
  so the control draws are paired across arms. M4 used one running stream, so its bootstrap CIs and p-values cannot
  be reproduced bit for bit.
* **The §1.5 reproduction check** therefore compares the deterministic columns of M4's `hypotheses.csv` for the 9 base
  SAEs (T_full, T_rec and T_err means, `preservation_rec`, `shapley_share_rec`, |I|/T_full), tolerance 1e-6. It runs
  before any fine-tuned SAE is evaluated, and the script stops if it fails.
* **H4a/H4b/H4c:** BH over the 36 SAEs (M4: over its 27 SAE × site cells).
* **Also reported, not criteria:**
  * reconstruction at the site on RepE-LM validation prompts (H, U, Hp at L8 final): MSE, EV, cosine, L0, dead fraction;
  * `effect_shortfall` = T_full − T_rec;
  * kl vs base and kl vs owt_only.
* **Extra checks:** the patch position is the last real token of every pair, and a hash of the GPT-2 parameters is
  taken before and after the run.
* **Smoke test:** 1/10 of the pairs, 3–5 control draws, in a scratch directory. Only the sanity values were inspected
  (all pass); no hypothesis value was printed.

## 2. After execution (run 2026-11-10, 1830 s, peak 6.0 GB; VALIDATION)

* **Run:** `--run` verified the hashes of the scripts, the data, IMP-L `criteria.json` and all 36 checkpoints
  against `preregistration.json`. 36/36 SAEs evaluated; 0 failures; 0 test rows. Log (local):
  `results/repe/logs/e3_finetuned_causal_preservation_2026-11-10.log`.
* **Sanity checks** (`sanity_checks.json`; all pass):
  * **M4 reproduction (deterministic columns, 9 base SAEs): max difference 0.0.** This check ran before any
    fine-tuned SAE.
  * Δ identity error 4.8e-7; v = 0 patch KL 0.0; Shapley sum error 5.6e-17; all-features patch = T_rec within 2.1e-6.
  * T_full is identical across all 36 SAEs (0.0); T_full = 0.02201 nats (T2–T4).
  * The patch position is the last token of every pair.
  * GPT-2's parameter hash is unchanged.
  * 356 persona and 178 neutral validation pairs; 1251 TRAIN T2–T4 pairs used only for feature ranking.
* **Criteria were not changed after results.** The classes in `criteria.json` were computed by the pre-registered
  code.

### 2.1 Causal preservation T_rec / T_full at the L8 final token (↑; mean ± sd over 3 seeds; per-seed values in `hypotheses.csv`)

| SAE | base | OWT-only | Method A | output-KL |
|---|---|---|---|---|
| ReLU | 0.621 ± 0.015 | 0.621 ± 0.016 | 0.622 ± 0.016 | 0.640 ± 0.016 |
| TopK | 0.625 ± 0.039 | 0.632 ± 0.024 | 0.615 ± 0.005 | 0.619 ± 0.037 |
| JumpReLU | 0.665 ± 0.011 | 0.669 ± 0.029 | 0.669 ± 0.025 | 0.687 ± 0.014 |

* **Per-SAE statement-bootstrap 95 % CIs** are about ±0.04–0.05 (ReLU/JumpReLU) and ±0.06–0.09 (TopK).
* **Other quantities:**
  * T_err/T_full: ReLU 0.43–0.45; TopK 0.07–0.09; JumpReLU 0.06–0.11.
  * |I|/T_full: ReLU 0.11–0.12, so the additive reading is allowed. TopK 0.32–0.34 and JumpReLU 0.26–0.28: not
    additive.
  * Specificity T_rec − S_rec: 0.006–0.008 nats in every cell.
* **M4 controls:** the H4a (reconstruction above matched-norm and decoder-shuffle vectors), H4b and H4c (top-k*
  features above 200 matched random sets) tests pass for **all 36 SAEs**. k* = 64 everywhere except one TopK
  output-KL seed (k* = 16).
* **Site reconstruction on RepE-LM prompts:**
  * EV 0.75–0.82.
  * L0: Method A changes it by < 1; output-KL adds about 10 (ReLU) and 13.5 (JumpReLU) latents, i.e. 6–8 %; TopK
    stays at 128.

### 2.2 Paired comparisons (preservation_rec; mean [95 % t-interval]; per seed 42 / 1042 / 2042)

| SAE | Method A − base | Method A − OWT-only | output-KL − Method A | output-KL − base |
|---|---|---|---|---|
| ReLU | +0.000 [−0.017, +0.017] | +0.001 [−0.009, +0.010] | **+0.018 [+0.011, +0.025]** (+0.021, +0.016, +0.016) | +0.018 [+0.008, +0.029] |
| TopK | −0.010 [−0.101, +0.080] | −0.017 [−0.071, +0.038] | +0.005 [−0.078, +0.088] (−0.034, +0.025, +0.023) | −0.006 [−0.020, +0.009] |
| JumpReLU | +0.004 [−0.039, +0.047] | −0.001 [−0.062, +0.060] | +0.018 [−0.020, +0.057] (+0.033, +0.002, +0.019) | +0.022 [+0.016, +0.029] |

### 2.3 Pre-registered classes

| Hypothesis | ReLU | TopK | JumpReLU |
|---|---|---|---|
| E3-H1 Method A raises preservation (vs base and OWT-only) | NO IMPROVEMENT | NO IMPROVEMENT | NO IMPROVEMENT |
| E3-H2 output-KL raises preservation over Method A | NOT CREDITABLE (L0 guard; criterion itself met) | NO IMPROVEMENT | NOT CREDITABLE (L0 guard; criterion not met) |
| E3-H3 specificity | not applicable (no creditable improvement) | n/a | n/a |

* **No regression** in any cell.
* **Method A** leaves causal preservation unchanged, as predicted before the run (it is trained on GoT/OWT, not on
  persona prompts). The OWT-only control is also unchanged.
* **Output-KL:**
  * For ReLU, the gain (+0.018, 3/3 seeds, > 2 sd) is consistent against base, OWT-only and Method A, but it comes
    with OWT L0 +5.8 % (IMP-L guard failed) and +10 latents at the site. It is not creditable.
  * For JumpReLU it is consistent only against base (+0.022), with OWT L0 +8.1 %.
  * The only creditable output-KL cell, TopK, whose L0 is fixed, shows **no change** (+0.005, CI −0.078 to +0.088).
  * The persona-specific part (T_rec − S_rec) moves by at most 0.0006 nats.
  * **Reading:** where output-KL appears to help causal preservation, the gain cannot be separated from the extra
    active latents.

### 2.4 Limitations

* Validation only; three seeds sharing data.
* One site (L8 final), selected in M3/M4.
* A small effect (T_full 0.022 nats).
* TopK's per-SAE CIs are wide (±0.06–0.09), so changes smaller than about 0.03 cannot be detected per architecture.
* The control draws are reseeded per SAE, so M4's p-values are not reproduced bit for bit; the deterministic
  quantities are reproduced exactly.
* No claim that GPT-2 uses SAE features.

### 2.5 Status

COMPLETED (VALIDATION). Paper: §10.2 (E3), tables `t_e3` and `t_e3_classes`, figure `fig_e3_preservation`.

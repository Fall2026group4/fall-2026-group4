# E3: causal preservation of the fine-tuned layer-8 SAEs (M4 protocol). Status: DRAFT PRE-REGISTRATION, awaiting user approval

Evidence level, once run: **VALIDATION**. No test row is read. Nothing in this folder has been run.

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

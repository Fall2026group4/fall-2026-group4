# XT-G / DX amendments (recorded before any output was analysed)

* **A1 (2026-11-21, before the first run).** The sparse-z refit family of M2 `evaluate_source` is not run (`Z=None`).
  * Reason: storing 16k–24k-latent codes for every task row and fitting an L1 logistic on them is the costliest
    family, and it is secondary.
  * Every other M2 family (frozen logistic and mass-mean, noise, error-term, refit logistic / MLP / kNN) is unchanged.
  * The GPT-2 M2 sparse-z values remain available in `results/shared/m2_four_level_faithfulness/`.
* **A2 (2026-11-21, after the Gemma smoke test, before any full run).**
  * **Finding that motivated it:** the Gemma smoke at layer 12 showed the original GoT cities logistic AUROC = 1.000.
    Every cities gap (SAE, error-matched PCA, even PCA with k = 84) is therefore 0. Cities is at ceiling for Gemma at
    mid depth and cannot discriminate SAEs; GPT-2's L8 AUROC is 0.84.
  * **Changes, applied to BOTH models:**
    1. Two fresh GoT families are added: `got_companies` (341 DEV rows) and `got_common_claim` (1,347 DEV rows).
       These are DEV rows only, the same rows for both models. An internal 60/40 group split (seed 20261121) gives
       train and validation. TEST rows stay locked; they were exposed once in TSP-R Phase 1 and are not used here.
    2. A non-saturating separation metric: d′ of the logistic scores (original, frozen, refit) and the refit d′
       retention. Every task now gets a ceiling flag when the original AUROC ≥ 0.99.
    3. DX runs on every task, not only cities and persona.
  * The headroom-normalised gap was already pre-registered.
  * GPT-2 DEV caches for all 12 layers were extracted on the same rows
    (`scripts/geometry_of_truth/33_extract_fresh_dev_all_layers.py`).
* **A3 (2026-11-21, before any GPT-2 full run).** GPT-2 scope, chosen for compute: the full design was more than 30 h
  of CPU.
  * **Old in-house SAEs:** seed 42 only, at the triplet layers. All-seed results for these SAEs already exist under the
    identical M2 code (`results/shared/m2_four_level_faithfulness/`).
  * **R2:** TopK, BatchTopK and ReLU × 3 seeds at layers 1/8/11. The width-3072 and frozen-decoder controls are not
    re-run here; they are in the R2 ladder results.
  * **res-jb:** all layers.
  * **PCA references** are de-duplicated by (kind, k) within a layer. The rows carry the `sae_id` of the first SAE
    that requested them; k is identical, so the reference is identical.
  * **Gemma:** unchanged (one SAE per layer).
* **Operational.** One layer per process, to bound host RAM (15.8 GB, no swap).
* **A4 (2026-11-22 ~00:35, analysis implementation; written and hash-frozen before any XT-G output value was
  inspected; only column headers and category labels were read).** Script `scripts/cross_model/11_xtg_analysis.py`,
  tests `src/tests/test_xtg_analysis.py`; hashes in `frozen_hashes_analysis.txt`.
  * **Deviation 1 (BH p-values):** the runner did not store bootstrap arrays. BH within (model, task, rule) uses
    normal-approximation p-values from the stored 95 % CIs (se = (hi − lo) / 3.92, as `pca_cm_posthoc_multiplicity`).
    Two-part rules use the intersection-union (max p).
  * **Deviation 2 (LOSS_BEYOND_PCA):** the pre-registered paired bootstrap CI is replaced by
    z = (gap_SAE − gap_PCA) / √(se²_SAE + se²_PCA). This ignores the positive SAE / PCA covariance, so it is
    conservative (fewer labels).
  * **Exact mapping:** each SAE's error-matched PCA row is found by recomputing
    k_err = `il.pca_k_for_val_nmse` with the runner's own `owt_refs`, because the runner de-duplicated PCA rows by
    (kind, k) under A3.
  * **Unchanged:** all rule thresholds, the ceiling flag and d′ (A2), descriptive-only cross-model reporting, and the
    batch-2 sparsity mechanism.

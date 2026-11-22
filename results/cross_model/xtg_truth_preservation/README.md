# XT-G / DX: truth and honesty representation preservation, both models, one protocol. Status: PRE-REGISTERED; full runs IN PROGRESS

**Pre-registration:** `preregistration.json`; amendments A1–A3 in `AMENDMENTS.md` (all made before any full-run
output); hashes in `frozen_hashes.txt`.
**Script:** `scripts/cross_model/10_truth_preservation.py`. Probing is done by the unchanged M2 `evaluate_source` and
RLM-RP `evaluate`. The new code is the task loading, the sources and the DX diagnostics.
**Evidence level on completion:** VALIDATION.

## Design (summary)
* **Tasks:**
  * GoT cities (geographic, grouped splits);
  * RepE-LM persona and neutral;
  * RepE extras: synonym, held-out wording T5/T6, pair-direction geometry;
  * **A2:** fresh GoT families companies and common_claim (DEV rows only, 60/40 group split).
* **Sources per SAE:**
  * the SAE;
  * PCA error-matched to the SAE's OWT-VAL NMSE;
  * PCA with k = the SAE's OWT L0.

  PCA is fitted on each model's own OWT TRAIN windows (same documents for both models).
* **DX (rotation vs information loss):**
  * frozen probe after an affine or Procrustes map fitted on **unlabeled** OWT-VAL windows;
  * linear CKA;
  * truth-direction projection correlation and variance ratio;
  * error enrichment along the truth direction;
  * **A2:** non-saturating d′ of the logistic scores (original, frozen, refit), with a ceiling flag.
* **Pre-registered DX rules:** READOUT_SHIFT, INFORMATION_LOSS, LOSS_BEYOND_PCA, SUPPRESSION.

## Smoke findings (EXPLORATORY, one layer each; superseded by the full runs)
* **GPT-2 L8, res-jb:**
  * Affine or Procrustes alignment does **not** reduce the GoT frozen gap (0.309 → 0.322). So the drop is not a global
    linear readout shift.
  * SAE error is enriched about 2× along the truth direction (CI low 1.77), and the truth-direction variance ratio is
    0.61. Error-matched PCA: 1.2× and 0.87.
* **Gemma L12, Gemma Scope 16k:** **GoT cities is at ceiling** (original AUROC 1.000; every gap 0). This motivated
  A2. RepE persona refit gap 0.155 (CI low 0.118) vs 0.094 for error-matched PCA.

## Results
Pending (`runs/`).

# R5: supervised causal skyline and random floor at the M4 sites. Status: COMPLETED (2026-11-10). Evidence: VALIDATION, descriptive

Evidence level: **VALIDATION** (subspaces fitted on TRAIN T2–T4 pairs, scored on VALIDATION T2–T4 pairs; RepE-LM has
no test split). Approved under the user's 2026-11-10 execution approval.

## 1. Before the run

### 1.1 Problem and question

* **Problem:** M4/E3 report that the SAE reconstruction carries 62–68 % of the persona effect at L8/L11 final, and
  that top-k SAE features carry 2–74 %. Neither number has a reference: how much can *any* rank-k intervention carry,
  and how much does a *random* rank-k subspace carry? (Review package W9; the framing of Chaudhary & Geiger 2024,
  arXiv 2409.04478, and Korznikov et al. 2026, arXiv 2602.14111.)
* **Question:** at each M4 site (L1 persona slot, L8 final, L11 final) and k ∈ {1, 4, 16, 64}, compare the share
  T_k / T_full carried by:
  1. **DAS-style skyline:** an orthonormal rank-k subspace learned on TRAIN pairs to maximise restoration (Adam,
     400 steps, lr 1e-2, one TRAIN batch per step, QR parametrisation; 3 seeds), initialised at the TRAIN-PCA basis
     plus seed-specific noise (sd 0.5/√768; see 1.2);
  2. **PCA:** the top-k principal directions of the TRAIN differences Δ = x_H − x_U;
  3. **mass-mean:** the mean TRAIN Δ direction (k = 1);
  4. **random floor:** 20 random orthonormal rank-k subspaces (mean and q95);
  5. **SAE features:** M4's top-k attribution-ranked feature patches (`features_summary.csv`, TRAIN-selected, 9 SAEs
     per site), read from committed results;
  6. **SAE reconstruction:** T_rec/T_full (M4).
* **Intervention:** v = P_k Δ added at the site to the U run (M4's `Engine.patched`, start_at_layer), with
  T(v) = KL(p_H‖p_U) − KL(p_H‖p_{U+v}). Ratio of statement means, with a statement-bootstrap CI (M4 `ratio_ci`).
* **Pre-stated expectations:**
  * DAS ≥ PCA ≥ random at every k;
  * whether SAE feature patches approach DAS is open.
  * The comparison is descriptive (no pass/fail rule). DAS is a supervised upper reference, not a competitor method.
* **Checks:** identity (P = I) reproduces T_full exactly (share 1 within 1e-6); T_full matches M4; 0 test rows.
* **Script:** `scripts/repe/27_causal_subspace_skyline.py` (hashes in `preregistration.json`).
* **Compute:** about 10–20 min GPU, no SAE.

### 1.2 Smoke test and the one design change before registration (2026-11-10)

* **Smoke test** (scratchpad, 30 DAS steps, 3 random subspaces, every 10th TRAIN and every 5th VALIDATION pair):
  * identity share 1.000 at all three sites; 0 test rows.
  * **Problem found:** with a random initialisation, DAS at 30 steps under-fitted PCA. At L8 k=4 the shares were
    DAS 0.21 and PCA 0.57. A skyline that loses to PCA is not an upper reference.
* **Change:** DAS now starts at the TRAIN-PCA basis plus seed-specific noise. The smoke numbers are not results and
  are not reported.
* This change was made before `preregistration.json` was written, and nothing else was changed.

## 2. Results (2026-11-10, VALIDATION, descriptive)

### 2.1 Run and checks

* **Pairs:** 1,251 TRAIN pairs and 267 VALIDATION pairs (T2–T4); 0 test rows.
* **Identity check:** share 1.000 at all three sites (|1 − share| < 1e-15).
* **Runtime:** 44 min, sharing the GPU with two other jobs.
* **Script SHA-256:** `78a6c533…`, matching `preregistration.json`.
* **Files:** `per_condition.csv` (every condition and seed, with bootstrap CIs), `summary.csv`, `checks.json`.

### 2.2 Share of the persona effect carried at rank k

Share = T_k / T_full, the restoration of the rank-k patch over the full single-site patch, from `summary.csv`.

* **DAS:** mean over 3 seeds; the seed sd is ≤ 0.014 in every cell.
* **Random:** mean of 20 subspaces (q95 in brackets).
* **SAE features:** M4's top-k TRAIN-selected feature patches, mean over the 3 seeds of each architecture, given as
  the range over ReLU/TopK/JumpReLU.

| Site | k | DAS (learned) | PCA (TRAIN Δ) | Random mean (q95) | SAE top-k features (ReLU–JumpReLU range) |
|---|---|---|---|---|---|
| L1 slot | 1 | 0.915 | 0.430 | 0.000 (0.003) | 0.04–0.07 |
| L1 slot | 4 | 1.016 | 1.000 | 0.004 (0.020) | 0.11–0.16 |
| L1 slot | 16 | 1.013 | 1.000 | 0.012 (0.057) | 0.28–0.33 |
| L1 slot | 64 | 1.016 | 1.000 | 0.041 (0.082) | 0.50–0.60 |
| L8 final | 1 | 0.641 | 0.358 | 0.002 (0.005) | 0.04–0.05 |
| L8 final | 4 | 0.913 | 0.518 | 0.008 (0.014) | 0.10–0.16 |
| L8 final | 16 | 1.116 | 0.678 | 0.027 (0.042) | 0.21–0.32 |
| L8 final | 64 | 1.289 | 0.826 | 0.110 (0.139) | 0.34–0.48 |
| L11 final | 1 | 0.401 | 0.379 | 0.002 (0.004) | 0.03–0.05 |
| L11 final | 4 | 0.619 | 0.587 | 0.008 (0.015) | 0.08–0.14 |
| L11 final | 16 | 0.759 | 0.706 | 0.033 (0.040) | 0.19–0.29 |
| L11 final | 64 | 0.888 | 0.839 | 0.127 (0.143) | 0.32–0.46 |

* **Mass-mean direction (k = 1):** 0.490 (L1), 0.294 (L8), 0.356 (L11).
* **SAE whole reconstruction (M4 T_rec/T_full):** 0.94–0.97 (L1), 0.62–0.67 (L8), 0.62–0.68 (L11).

### 2.3 Inference

1. **The persona effect is low-rank at every site.**
   * One to four supervised directions carry 40–100 % of it.
   * At the L1 slot, the 4 top PCA directions of Δ already carry all of it.
2. **The SAE's feature basis is far from the best basis at matched rank.**
   * At k = 4, SAE feature patches carry 8–16 %. PCA carries 52–100 % and DAS 62–102 %.
   * Even 64 SAE features carry less than a single DAS direction at L1 and L8.
   * This is the dictionary-alignment gap of Chaudhary & Geiger 2024, now measured on our sites.
3. **The SAE's partial reconstruction share is not because the effect is hard to carry.**
   * At L8, a rank-4 subspace (0.91) carries more than the SAE's full 768-dimensional reconstruction (0.62–0.67).
   * So the 33–38 % loss at L8 reflects the reconstruction, not a ceiling of the task.
4. **SAE features are well above random:** at every k, they exceed the random q95. That is consistent with M4's
   controls.
5. **DAS shares above 1 (L8, k ≥ 16) are not an error.** The full patch at a single site does not fully restore
   p_H, because earlier positions still differ. A learned subspace patch can therefore overshoot the full patch.
   This is supervised optimisation of the very quantity being scored, on TRAIN pairs, so DAS is an upper reference.
   It is not a competitor method.

### 2.4 Limitations

* GPT-2 Small only; one dataset (RepE-LM persona); three sites. The comparison is descriptive.
* The DAS initialisation was changed after the smoke test, before registration (1.2).
* SAE feature patches are those of the old (M4) SAEs. The R2 SAEs' M4 reconstruction share is in the R2 ladder;
  their feature patches were not recomputed.

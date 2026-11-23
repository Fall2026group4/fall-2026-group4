# XT-G / DX: truth and honesty representation preservation, both models, one protocol. Status: COMPLETED (2026-11-22; VALIDATION). No readout shift anywhere; persona information lost in both models; SAE error concentrated along the task direction (truth for GoT, persona / neutral label for RepE)

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

## Results (2026-11-22; frozen analysis `11_xtg_analysis.py --tag v1`, amendment A4; `analysis_v1/`)

**Coverage:**
* Gemma: Gemma Scope 16k canonical at all 26 layers.
* GPT-2: res-jb at 11 layers, plus R2 (TopK / BatchTopK / ReLU × 3 seeds) and old in-house (seed 42) at the matched
  layers 1 / 3 / 6 / 8 / 9 / 11. That is 410 SAE × task cells.
* **GPT-2 layer 0 is not evaluable** (run failed by design): the final-token `resid_pre` at layer 0 is the
  embedding of the shared final token, so it carries no statement information. The truth direction is zero and
  d′ = 0, giving a division by zero in d′ retention. The frozen code was not changed.
* Short prompts throughout, so the GPT-2 SAEs are within their 128-token training context (unlike XB's primary arm).

**Rule counts** (cells; BH within model × task; deviations per A4):

| Model | Task | n | READOUT_SHIFT | INFORMATION_LOSS | LOSS_BEYOND_PCA | SUPPRESSION | Ceiling |
|---|---|---|---|---|---|---|---|
| Gemma | RepE persona | 26 | 0 | **26** | **21** | 26 | 0 |
| Gemma | RepE neutral (control) | 26 | 0 | 26 | 21 | 26 | 0 |
| Gemma | GoT cities | 26 | 0 | 5 | 6 | 26 | 20 |
| Gemma | GoT companies | 26 | 0 | 4 | 8 | 26 | 0 |
| Gemma | GoT common_claim | 26 | 0 | 5 | 1 | 26 | 0 |
| GPT-2 | RepE persona | 56 | 0 | **56** | **34** | 54 | 0 |
| GPT-2 | RepE neutral (control) | 56 | 0 | 27 | 27 | 46 | 0 |
| GPT-2 | GoT cities | 56 | 0 | 35 | 22 | 49 | 0 |
| GPT-2 | GoT companies | 56 | 0 | 0 | 1 | 56 | 0 |
| GPT-2 | GoT common_claim | 56 | 0 | 0 | 0 | 53 | 0 |

**Findings (VALIDATION):**
1. **No READOUT_SHIFT in any of the 410 cells.** A linear map fitted on unlabeled OWT never restores the frozen
   probe, so frozen-probe drops are not a global rotation or affine readout shift. This holds in both models.
2. **Persona (honesty) information is lost in both models:** INFORMATION_LOSS in 26/26 Gemma and 56/56 GPT-2 cells,
   and beyond error-matched PCA in 21/26 and 34/56. **It is not honesty-specific:** the neutral control loses as
   much in Gemma (26/26), and in about half the GPT-2 cells (27/56).
3. **The SAE error concentrates along the task direction (SUPPRESSION) almost everywhere** (Gemma 130/130,
   GPT-2 258/280). Median enrichment: Gemma 6–34×, GPT-2 1.7–5.7×. The enrichment scale depends on d_model (2,304 vs
   768), so it is compared within a model only. For GoT the direction is the truth direction; for the RepE tasks it is
   the task's own label direction (honest vs untruthful persona, or the neutral control), not a truth direction.
4. **GoT cities:**
   * Gemma is at ceiling at 20/26 layers (A2), where d′ retention is the usable metric (0.72–1.01 at layers 12 / 16 / 19 / 23);
   * GPT-2 loses cities information in 35/56 cells, 22 beyond PCA.
   * companies / common_claim: little loss in GPT-2; a few deep-layer Gemma cells.

**Cross-model (descriptive only; depth-matched triplets; `cross_model_descriptive.csv`):**
* Persona refit gap relative to the original headroom (gap / (AUROC − 0.5)): GPT-2 res-jb 0.60–0.85, Gemma
  canonical 0.33–0.52.
* Refit d′ retention: GPT-2 res-jb 0.08–0.33, Gemma 0.44–0.52.
* So Gemma Scope keeps relatively more persona information than res-jb at matched depth. Different SAE families and
  training; **no ranking claim**.

**Deviations (A4, frozen before the outputs were read):**
* BH p-values from the stored CIs (normal approximation);
* a conservative z-test for LOSS_BEYOND_PCA (no bootstrap arrays were stored).

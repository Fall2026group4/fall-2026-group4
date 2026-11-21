# XB/XT-G batch 2: Gemma Scope 65k / 32k ladder (DRAFT execution addendum, 2026-11-21). Status: FROZEN 2026-11-21 (option R approved by the user); nothing downloaded or run at freeze

The strata and the matching rule were **pre-registered in XB** (`xb_matched_splicing/preregistration.json`:
`saes.gemma.batch2_ladder`, `comparison_tracks.matched_stratum`), and the sparsity-mechanism analysis in XT-G
(`xtg_truth_preservation/preregistration.json`: `sparsity_mechanism (batch 2, ladder)`). This addendum only fixes the
exact SAE IDs, storage and run order. **No rule is changed.**

## Feasibility finding (before download; affects scientific value)

**The frozen matched-stratum rule** is: L0/d_model **and** d_sae/d_model each agree within ±15 %.
* d_sae/d_model: res-jb (24576/768 = 32×) ↔ Gemma 65k (65536/2304 = 28.4×) gives a ratio of 0.89, which passes.
* L0/d_model, measured at the XB 1024 shared span: res-jb is 0.24–3.4 per layer, so the Gemma L0 targets are
  540–7,800. **The largest published 65k L0 at any triplet layer is 297.** No res-jb ↔ 65k pair can match at 1024
  context.
* At 128-token context (res-jb L0 15–68), the targets are 45–203. The published 65k L0 steps (roughly ×2 apart)
  bracket these, but they rarely fall within ±15 %. Matching is possible at most at one or two triplet positions, and
  only if A4 (ctx128) is run.
* **R2 TopK (16×, L0/d 0.167) ↔ Gemma 32k L12** (14.2×, ratio 0.89; L0 360 gives L0/d 0.156, ratio 0.94):
  * passes on both ratios;
  * but R2 exists only at GPT-2 layers 1/8/11 (triplet T-A), and Gemma 32k only at layer 12 (T-B middle). It is
    **not the same triplet position** (depth 0.667 vs 0.50), so under the frozen design it is not a matched pair.
    It is reported as "ratio-matched, depth-mismatched, descriptive".

**Consequence:** the 65k ladder **cannot deliver the matched public-vs-public stratum** it was planned for. What it
can still deliver is the pre-registered **within-Gemma width × L0 mechanism analysis** (XT-G `sparsity_mechanism`):
how the truth refit gap, error-truth enrichment and splice damage change with L0 at a fixed width, and with width at a
similar L0.

## Options

| Option | SAEs | Download (scratch) | GPU | Delivers |
|---|---|---|---|---|
| **R (recommended): reduced** | L12: 65k × 5 + 32k × 6 (16k × 5 already on disk) | 9.6 GB | ~1.5 h | width × L0 surface at the middle layer (16k/32k/65k); XB Pile shared + XT-G at L12 |
| F: full | 65k × 31 at layers 1/5/12/16/19/23 + 32k × 6 | 41 GB | ~4–5 h | the same at all six triplet layers |
| 0: skip | — | — | — | report the infeasibility of the matched stratum only |

Published 65k L0s at the revision pinned in `asset_manifest_b2a.json` (`fd571b47`):
* L1: 9 / 16 / 30 / 54 / 121
* L5: 17 / 29 / 53 / 105 / 211
* L12: 21 / 38 / 72 / 141 / 297
* L16: 21 / 38 / 69 / 128 / 244
* L19: 21 / 35 / 63 / 115 / 216
* L23: 20 / 35 / 64 / 123 / 124 / 272

32k (L12 only): 12 / 22 / 40 / 76 / 155 / 360.

## Protocol (as pre-registered)

* **Download:** via `00_fetch_assets.py --batch b2b` into `/mnt/scratch/hf_cache` (re-creatable; SHA-256 recorded in
  `asset_manifest_b2b.json`).
* **XB:** Pile shared span only, the unchanged runner, `--saes ladder`; the same controls, bootstrap and gates.
* **XT-G:** the same tasks and sources (SAE, PCA error-matched, PCA L0-matched); DX diagnostics.
* **Analysis:**
  * within (model, layer, width): Spearman of refit gap and error-truth enrichment vs measured L0;
  * within (layer, L0 bin): width effect;
  * descriptive, VALIDATION. No cross-model "matched" label unless the frozen rule passes.

## Storage and backup

Weights live on scratch only (public, re-downloadable; never copied to EBS). Results (< 0.2 GB) go to EBS.

## Frozen execution (option R, user-approved 2026-11-21)

* **Approval:** reduced ladder at layer 12 only. The original matched public-vs-public comparison is reported as
  **INFEASIBLE under the frozen ±15 % criteria**, which are not relaxed.
* **SAEs (one invocation per runner, L12):**
  * 16k: canonical 82 + 22 / 41 / 176 / 445 (on EBS);
  * 32k: 12 / 22 / 40 / 76 / 155 / 360 (scratch);
  * 65k: 21 / 38 / 72 / 141 / 297 (scratch).
* **Download:** `00_fetch_assets.py --batch b2b` writes `asset_manifest_b2b.json` (SHA-256). `06_ladder_scratch.py`
  verifies every file before running.
* **Runs:**
  * `06_ladder_scratch.py xb`: frozen 02, Pile, 1024 shared span (as pre-registered for batch 2), `--tag b2`;
  * `06_ladder_scratch.py xtg`: frozen 10, `--tag b2`, on CPU after XT-G main has finished.
* **Analysis (descriptive, VALIDATION):**
  * frozen 03 with `--tag b2`, reporting the matched-stratum labels exactly as computed, with no relaxation;
  * the XT-G pre-registered `sparsity_mechanism` analysis within (model = gemma, layer 12, width): Spearman of the
    refit gap and error-truth enrichment vs measured L0, plus the width effect at similar L0.
* **Hashes:** `frozen_hashes.txt`. The original 1024 / own runs are untouched (new run folders `*_b2`).

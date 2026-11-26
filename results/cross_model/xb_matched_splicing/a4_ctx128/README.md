# XB amendment A4: 128-token context sensitivity arm. Status: COMPLETED (2026-11-22; VALIDATION, sensitivity). Pre-registered Holm outcome infeasible by design (p floor); descriptive: most cross-model differences shrink at matched context

## Why (threat to validity found before any XB outcome analysis)

* **What was looked at:** only per-token **L0** of the GPT-2 SAEs in the completed `gpt2_owt_xm_shared_main` run,
  while planning the 65k ladder. No ΔCE, KL, loss-recovered or ratio value was computed or inspected. XB's analysis
  (`03_matched_splicing_analysis.py`, tag v1) has **not** been run.
* **Finding:**
  * The GPT-2 SAEs were trained at a 128-token context: R2 (`scripts/openwebtext/17_streaming_sae_training.py:44`,
    CTX = 128), and res-jb (128-token training context).
  * XB's primary arm evaluates them on ~700-token shared spans (Pile 512 / OWT-XM 703 GPT-2 tokens on average). So
    about 80 % of the scored positions lie outside the SAEs' training context.
  * Gemma Scope was trained at 1024 tokens, so Gemma is evaluated in-distribution.
* **Symptom (L0 per token):**

  | SAE | 128-token OWT VAL | 1024 shared span |
  |---|---|---|
  | res-jb L1 / L8 / L11 | 15 / 68 / 61 | 580 / 415 / 180 |
  | R2 ReLU L8 | ≈ 128 (target) | 369 |
  | R2 BatchTopK L8 | ≈ 128 (target) | 397 |
  | TopK | 128 | 128 (fixed by construction) |

* **Consequence:** XB_H2 compares an in-distribution Gemma SAE with out-of-distribution GPT-2 SAEs. Any "GPT-2 SAEs
  degrade behaviour more" difference is confounded with the context mismatch. It also inflates the GPT-2 sparsity used
  by the pre-registered matched-stratum rule.

## What changes (additive only; nothing pre-registered is removed or altered)

* **The existing 1024 arms stay primary as pre-registered.** They are analysed exactly as frozen, and the context
  caveat is disclosed in every table.
* **New sensitivity arm `ctx128`:**
  * both models, both corpora (Pile, OWT-XM), all layers, all SAEs already in XB;
  * each document is cut to the longest prefix that is ≤ 128 tokens under **both** tokenizers (the same shared-span
    rule, at 128 instead of 1024);
  * BOS excluded; positions 1 ≤ t ≤ len − 2 scored; the same controls (mean-ablation, norm-matched random), seeds,
    bootstrap and gates (identity / repeat = 0);
  * code: the unchanged `02_matched_splicing.py` with a new `--context shared128` option (span cap only) plus a unit
    test that the 1024 behaviour is unchanged.
* **Pre-registered reading:**
  * if the GPT-2 − Gemma differences in the normalised metrics (ratio, excess_rel, loss_recovered) shrink toward 0 at
    ctx128 (CI of the change excludes 0), the 1024 cross-model gap is attributed partly to context mismatch;
  * if they persist, the gap is not explained by context.
  * The matched-stratum rule (±15 % L0/d_model and d_sae/d_model) is re-applied at ctx128 with the L0 measured there.
* **Label:** VALIDATION, sensitivity analysis. Holm within the ctx128 family, reported separately from the
  primary family.

## Frozen protocol

`preregistration.json` and `frozen_hashes.txt` in this folder. Runner `scripts/cross_model/05_xb_ctx128.py`, comparison `scripts/cross_model/07_a4_context_comparison.py`, manifests built by `scripts/cross_model/04_build_ctx128_manifest.py`. Original-context results stay in `../runs/` and `../analysis_v1/`; this tree holds A4 only (here `context = shared` means the 128-token shared span).

## Cost

* GPU: about 1.5 h per model per corpus for GPT-2 (12 layers, 159 SAEs, 1/6 of the tokens) and about 1 h for Gemma
  (26 layers); about 4–5 h in total.
* Disk: < 0.5 GB of per-document sums, on EBS (results).
* RAM: as the existing XB runs.

## Results (2026-11-22; `analysis_a4/`, `context_comparison_a4/`)

**Runs:** GPT-2 (159 SAEs, 12 layers) and Gemma (canonical 16k, 26 layers), Pile and OWT-XM, 128-token shared span
(mean 120 / 125 GPT-2 tokens). All four runs complete.

**Gates:**
* identity / repeat ΔCE exactly 0.0 (smoke and full runs);
* frozen hashes verified at start;
* the pairing hash check passed in the comparison script.

### Context fixes the GPT-2 breakdown

At 128 tokens the GPT-2 SAEs are back in distribution:

| SAE | L0 | EV |
|---|---|---|
| res-jb L1 / L8 / L11 (OWT-XM) | 15 / 67 / 61 | 0.95 / 0.86 / 0.81, matching their 128-token OWT VAL ladder values |
| R2 | ≈ 130 | individual trained SAEs 0.81–0.98 (per-layer means 0.83–0.95 including the frozen-decoder control, which sits at 0.57–0.68) |

### Pre-registered outcomes (as computed)

| Test | Holm outcome | Uncorrected 95 % CI (descriptive) |
|---|---|---|
| XB_H1 (SAE ΔCE > norm-matched random), GPT-2 | 0/318 | **297 supported**, 18 inconclusive, 3 contradicted |
| XB_H1, Gemma | 49/52 (26/26 OWT-XM, 23/26 Pile) | 49 supported; Pile L0 and L2 inconclusive; Pile L1 contradicted (−0.007) |
| Matched stratum (±15 % rule, L0 measured at 128 tokens) | 0/54 pairs | — |
| **A4 context change** (shrink = \|d_1024\| − \|d_128\|, 324 rows) | **0/324 (all NO CLEAR CHANGE)** | **269 SHRINKS**, 29 GROWS, 26 NO CLEAR CHANGE |

Context change by metric (uncorrected SHRINKS / GROWS / no clear change):

| Metric | Shrinks | Grows | No clear change |
|---|---|---|---|
| loss_recovered | 54 | 0 | 0 |
| kl_over_kl_mean | 53 | 0 | 1 |
| ev | 48 | 4 | 2 |
| dce_rel | 45 | 6 | 3 |
| excess_rel | 41 | 5 | 8 |
| ratio_sae_random | 28 | 14 | 12 |

### ⚠ The Holm outcomes are infeasible by design (disclosed; criteria unchanged)

* Both analyses use a 1,000-resample bootstrap with p = 2 · min(tail) + 1/1000, so the smallest attainable p is about
  0.002.
* Holm over 159 GPT-2 cells (XB_H1) or 324 rows (the A4 comparison) cannot reach 0.05: the smallest adjusted p is
  0.159 and 0.324 respectively, **whatever the effect size**. Gemma's 26-cell families are feasible.
* The pre-registered Holm labels are therefore reported but carry **no evidential weight** (neither positive nor
  negative). The uncorrected CI labels are reported as descriptive.
* The flaw is in the frozen XB analysis (03) and was replicated in the A4 comparison (07), which I wrote. It was found
  after the results were computed, so no rule was changed.
* A higher-resolution re-analysis (e.g. 100,000 resamples) would be a post-hoc change. It is not run without the
  user's approval, and would be labelled exploratory.

### Reading (descriptive, VALIDATION)

* At matched context, both models' SAEs are **more harmful than norm-matched random error**:
  * SAE / random ΔCE: GPT-2 1.5–4.8, Gemma 0.7–3.0 at the triplet slots;
  * loss recovered: GPT-2 0.91–1.00, Gemma 0.97–1.00.
* Most cross-model differences seen at 1024 tokens **shrink** at 128 tokens: 269/324 by uncorrected CI, including
  loss recovered in 54/54 rows. So the primary-arm gap is largely the GPT-2 context mismatch.
* No pair is sparsity-matched, so no architecture-matched claim is made.

## Storage note (2026-11-23)
`runs/gpt2_owt_xm_shared_main/` and `runs/gpt2_pile_shared_main/` store `per_doc.csv.gz` (lossless, `gzip -n -9`) with
`per_doc.csv.sha256` (hash of the uncompressed file): the uncompressed files are 27.2 MB, above the 25 MB
repository-hygiene limit. Before re-running the frozen A4 analysis, run `gunzip -k runs/*/per_doc.csv.gz` and
`sha256sum -c per_doc.csv.sha256` in each folder. The same applies to the two primary-arm runs (#53). No number changed.
**Without this step the frozen loader skips those runs without an error** (`07` reuses `03`'s `load_runs`, which skips
folders without `per_doc.csv`), so re-run A4 through the preflight (D55), which covers both trees and refuses to start on
a missing or stale file (added 2026-11-26):
`python scripts/shared/compressed_outputs_preflight.py --decompress results/cross_model/xb_matched_splicing -- python scripts/cross_model/07_a4_context_comparison.py --tag <new tag>`.

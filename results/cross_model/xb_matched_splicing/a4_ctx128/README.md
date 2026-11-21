# XB amendment A4: 128-token context sensitivity arm. Status: PRE-REGISTERED (frozen 2026-11-21, before any A4 run); approved by the user

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

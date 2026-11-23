# XB: matched cross-model SAE splicing (GPT-2 Small vs Gemma-2-2B). Status: COMPLETED (2026-11-21; VALIDATION). Primary 1024 arm reported as pre-registered, with a decisive context caveat; see A4

**Plan:** `docs/2026-11-21_final_week_technical_audit.md` (approved 2026-11-21).
**Pre-registration:** `preregistration.json`, frozen before any output. Hashes are in `frozen_hashes_pre_run.txt` and
`frozen_hashes_full_run.txt`; amendments A1–A3 are in `AMENDMENTS.md` (implementation only, no metric changes).
**Evidence level on completion:** VALIDATION. Descriptive cross-model comparison plus directional within-model tests.

## 1. Question and design
**Question:** under one shared protocol, how much does splicing SAE reconstructions degrade next-token behaviour in
each model, relative to that model's own controls? The protocol fixes:
* the same text (shared span) and the same context rule;
* the same controls and the same paired document bootstrap;
* depth-matched sites.

**This is reconstruction and behavioural faithfulness only.** It says nothing about truth information or causal
effects (CLAUDE.md §2).

| Item | GPT-2 Small | Gemma-2-2B |
|---|---|---|
| Loader | TransformerLens default processing, fp32 | HF bf16, eager attention, rev `c5ebcd40` (= M6) |
| Site | `blocks.L.hook_resid_pre`, L 0–11, depth L/12 | output of `layers[L]` (`resid_post`), L 0–25, depth (L+1)/26 |
| Hook gate G-HOOK | TL resid = HF `hidden_states[L]` minus its per-token mean (rel. diff ≤ 3e-7); logits identical | as M6 (raw HF; what Gemma Scope was trained on) |
| SAEs | res-jb (public ReLU 32×), R2 (in-house 16×, L 1/8/11), old in-house (4×, under-trained) | Gemma Scope 16k canonical (public JumpReLU 7.1×), all layers; L0 ladder in batch 2 |

**Data:**
* **Pile:** M6's 512 eval + 64 reference documents. Every text hash and Gemma token-id hash was re-verified against
  M6 (576/576).
* **OWT-XM:** a new 512 + 64 sample from OWT shard 00077. No document overlaps our 1,000-document sample or the R2
  training shards 00078/00079 (hash-checked). res-jb's OWT document set is unknown; this is disclosed.

**Context:**
* *Primary:* the shared text span, ≤ 1023 tokens under both tokenizers. Mean span: GPT-2 512 / Gemma 471 tokens on
  the Pile; 703 / 685 on OWT-XM.
* *Secondary:* 1024 tokens in each model's own tokenizer, at the triplet layers only.

**Depth triplets:**
* T-B: GPT-2 3/6/9 ↔ Gemma 5/12/19.
* T-A: GPT-2 1/8/11 ↔ Gemma 1/16/23.

**Controls:**
* original repeat and identity, on 8 documents;
* norm-matched isotropic random error, with the same directions for every SAE of a (model, layer, document);
* mean ablation, from the reference set.

**Normalised cross-model metrics:** loss recovered, ΔCE_SAE/ΔCE_random, (ΔCE_SAE − ΔCE_random)/CE₀, ΔCE/CE₀ and
KL/KL_mean. Raw nats are reported but never ranked across models.

**Matched stratum:** L0/d_model and d_sae/d_model each within ±15 %.

## 2. Gates (all PASSED, 2026-11-21)
| Gate | Result |
|---|---|
| G-HOOK | passed (see above) |
| Sample V1 | Pile 576/576 text and Gemma own-1024 id hashes match M6; OWT-XM 0 duplicates of our data |
| V_IDENT / V_REPEAT (smoke, 8 docs, both models) | ΔCE and KL exactly 0.0 |
| Paired vs M6's own per-doc rows (Gemma own-1024, layers 5/12, 8 docs) | SAE ΔCE mean diff 0.0004 nats, max 0.008 (bf16 floor); random-control values differ by design (new paired directions) |
| A2 layer-major = document-major (Gemma, 8 docs × 2 layers) | all sums identical (max diff 0.0) |
| V_M6 (512-doc summary within the R3 margins) | **PASS**: own-1024 Pile ΔCE L5 / L12 / L19 = 0.0708 / 0.1144 / 0.1772 vs M6 0.0708 / 0.1146 / 0.1773 (EV within 0.0001) |

## 3. Results (frozen analysis `03_matched_splicing_analysis.py --tag v1`; `analysis_v1/`)

**Runs:** all six complete.
* Gemma: Pile and OWT-XM, shared span, all 26 layers; own-1024 at the triplet layers.
* GPT-2: Pile and OWT-XM, shared span, all 12 layers, 159 SAEs.

**Gates:** V_IDENT and V_REPEAT max |ΔCE| = 0.0; V_M6 PASS.

### 3.1 Decisive caveat: GPT-2 SAEs are evaluated far outside their training context

The GPT-2 SAEs were trained on 128-token contexts (R2 `CTX = 128`; res-jb 128). The primary shared span averages
703 / 512 GPT-2 tokens (OWT-XM / Pile), so most scored positions lie beyond position 127. There the GPT-2 SAEs break
down:

| GPT-2 SAE (shared span, OWT-XM) | L0 per token (OWT-XM) | OWT VAL EV at 128 tokens (ladder) | EV on the OWT-XM shared span |
|---|---|---|---|
| res-jb L1 / L8 / L11 | 580 / 415 / 180 (vs 15 / 68 / 61 at 128 tokens) | 0.95 / 0.85 / 0.81 | **−20.3 / −5.1 / −0.06** |
| res-jb L3 | 2,616 | — | **−3,266** |
| R2 (TopK / BatchTopK / ReLU) L1 / L8 / L11 | 257 / 246 / 171 (mean) | 0.98 / 0.89 / 0.87 (mean of the three) | 0.15 / 0.28 / 0.72 |

All GPT-2 shared-span values in this table are OWT-XM. Negative EV means the reconstruction error is larger than the activation's variance.

Gemma Scope (trained at 1024 tokens) stays in distribution: EV 0.74–0.88 and L0 65–97 at the triplet layers.

**Consequence:**
* The primary GPT-2 numbers measure **long-context failure of 128-token SAEs**, not ordinary SAE faithfulness.
* The norm-matched random control inherits the inflated error norm, which is why GPT-2 SAEs look "less harmful
  than random".
* **No GPT-2 vs Gemma comparison from the primary arm is interpretable as an SAE comparison.**
* The context-controlled comparison is amendment A4 (128-token shared span, `a4_ctx128/`), frozen and run before
  this analysis was read. A4 was triggered by the L0 values alone.

### 3.2 Pre-registered outcomes (reported exactly as computed)

* **XB_H1** (SAE ΔCE > norm-matched random ΔCE, Holm within model × corpus × context):

  | Model | Cells supported | Contradicted (SAE less harmful than random) |
  |---|---|---|
  | **Gemma** | **64/64** (26 + 26 shared, 6 + 6 own) | 0 |
  | **GPT-2** | **0/318** under Holm; 243/318 by their own 95 % CI (descriptive) | 70 (46 OWT-XM, 24 Pile) |

  **Correction (2026-11-22): the GPT-2 "0/318 under Holm" is a test-design artefact, not a finding.**
  * The frozen analysis uses a 1,000-resample bootstrap, so the smallest attainable p is about 0.002.
  * Holm over the 159 GPT-2 cells per family therefore cannot reach 0.05 (smallest adjusted p 0.159), whatever the
    effect size.
  * Gemma's families have 6 or 26 cells, so they are feasible.
  * The uncorrected CI verdicts (243 supported, 70 contradicted, 5 inconclusive) are reported as descriptive.
  * The 70 contradicted cells are the §3.1 artefact (inflated error norm). An earlier version of this README wrongly
    attributed the 0/318 to §3.1 as well.
* **XB_H2** (cross-model differences):
  * **no matched stratum exists**: 0 of 54 slot × family pairs pass the frozen ±15 % L0/d_model and d_sae/d_model
    rule;
  * so every cross-model row is "unmatched, descriptive" (`cross_model.csv`). Not relaxed.
* **XB_H3** (depth trend of the SAE / random ratio):
  * Gemma canonical: no trend (Spearman 0.03 / 0.22 over 26 layers, OWT-XM / Pile);
  * GPT-2 res-jb: −0.80 (OWT-XM, p = 0.002) and −0.32 (Pile, p = 0.31). These are confounded by §3.1.

**Descriptive (triplet slots, shared span):**

| | Gemma canonical: SAE / random ΔCE | Gemma canonical: loss recovered |
|---|---|---|
| Range | 1.3–3.3 | 0.97–1.00 |

**What may be cited:**
* the Gemma results (in distribution);
* the gates;
* the fact that GPT-2's 128-token SAEs fail on long contexts.

**Not to be cited:** any primary-arm GPT-2 vs Gemma comparison.

**Files:**
* `runs/*/per_doc.csv` and `DONE.json` (committed). The two GPT-2 shared-span runs (`gpt2_owt_xm_shared_main`,
  `gpt2_pile_shared_main`) are stored as `per_doc.csv.gz` (27.2 MB each uncompressed, above the 25 MB repository-hygiene
  limit), with `per_doc.csv.sha256` holding the hash of the uncompressed file. To re-run the frozen analysis, first run
  `gunzip -k runs/*/per_doc.csv.gz` (then `sha256sum -c` in each folder); the frozen scripts are unchanged;
* `runs/*/sum_x.npz` (per-document activation sums; 38–243 MB each; not committed; SHA-256 in
  `runs_sum_x_sha256.txt`);
* `analysis_v1/`.

## 4. Files
* `samples/`: manifests (ids and hashes only, no text) and sample info.
* `inputs/m6/`: M6 committed files copied from `origin/main` `bf25b40`, blob ids in `SOURCE_BLOBS.txt`.
* `inputs/r3/SOURCE_BLOBS.txt`: R3 per-document rows used for one spot check. The 6 MB copy is not committed; it is
  re-creatable with `git show`.
* `smoke/`: smoke and A2 gate runs.
* Scripts: `scripts/cross_model/0{0,1,2,3}_*.py`, `src/models/backends.py`, `src/evaluation/pile_splicing.py`
  (byte-identical to `origin/main`).

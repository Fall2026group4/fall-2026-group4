# XB: matched cross-model SAE splicing (GPT-2 Small vs Gemma-2-2B). Status: PRE-REGISTERED; gates PASSED; full runs IN PROGRESS

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
| V_M6 (512-doc summary within the R3 margins) | evaluated on the full own-1024 run (pending) |

## 3. Results
Pending: full runs in `runs/` (in progress), then `scripts/cross_model/03_matched_splicing_analysis.py`.

## 4. Files
* `samples/`: manifests (ids and hashes only, no text) and sample info.
* `inputs/m6/`: M6 committed files copied from `origin/main` `bf25b40`, blob ids in `SOURCE_BLOBS.txt`.
* `inputs/r3/SOURCE_BLOBS.txt`: R3 per-document rows used for one spot check. The 6 MB copy is not committed; it is
  re-creatable with `git show`.
* `smoke/`: smoke and A2 gate runs.
* Scripts: `scripts/cross_model/0{0,1,2,3}_*.py`, `src/models/backends.py`, `src/evaluation/pile_splicing.py`
  (byte-identical to `origin/main`).

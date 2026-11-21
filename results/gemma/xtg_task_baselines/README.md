# XT-G0: Gemma-2-2B task activations and task gates (2026-11-21). Status: COMPLETED (VALIDATION; gates PASSED)

Script: `scripts/gemma/10_extract_task_activations.py`, run once in 26.9 min on an A10G.
Outputs: `data/gemma/activations/` (gitignored, about 5 GB, float16) and `gates/`.

## Design
* **Model and site:** Gemma-2-2B HF bf16, eager attention, rev `c5ebcd40` (= M6 / XB). The site is the output of
  `layers[L]` (`resid_post`), L 0–25. These mirror the GPT-2 caches row for row.
* **GoT:** cities, neg_cities, larger_than and smaller_than, all rows as in the GPT-2 cache. BOS + statement, final
  token. TEST rows are dropped at load time.
* **Fresh GoT families:** companies and common_claim, **DEV rows only** (TSP-R Phase 0 split; TEST stays locked).
* **RepE-LM:** all train and validation prompts (no test rows). Final token and slot. The slot is the position where
  the pair members differ under the Gemma tokenizer.
* **OWT:** the same 800 train / 100 validation documents as the GPT-2 cache, one window per document (BOS + first 127
  tokens). Train gives PCA mean, eigenvalues and eigenvectors; validation gives activations. Token counts equal
  GPT-2's: 101,600 / 12,700.

## Gates
| Gate | Result | GPT-2 reference |
|---|---|---|
| Tokenizer K1 (one-token swap at a fixed slot) | 16/16 template × variant cells, 506/506 pairs each | 6,072/6,072 |
| Activation K1 (identical before the slot, all 26 layers) | **PASS**: 6,072 pairs, max diff 0.0 at every layer; slot differs at layer 0 (min 7.58) | PASS |
| K7 behaviour (final-token KL, persona H‖U vs neutral Na‖Nb) | median 0.0229 vs 0.0064; per-statement median ratio **5.1**; Wilcoxon p = 7e-77 (506 statements) | ratio 1.99 |

**Interpretation:** the RepE-LM honesty construction transfers to Gemma without rebuilding. Gemma's persona
behaviour effect, relative to the neutral swap, is larger than GPT-2's. This is a baseline and gate result only; no SAE
is involved.

## Files
`gates/k1_gemma.json`, `gates/k7_gemma_rows.csv`, `gates/extraction_info.json` (counts, hashes, runtime).

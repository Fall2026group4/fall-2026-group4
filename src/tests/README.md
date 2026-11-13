# Tests

Run with `pytest src/tests/` from the repo root.

- `test_steering.py` — the steering hook's math (adds the scaled decoder
  direction to the last token only), the baseline-vs-steered code path,
  the sweep's strength bookkeeping, and that a fixed seed is requested
  before every generation. Uses fake model/torch stand-ins, so it runs
  without downloading any model weights.
- `test_detection.py` — `build_steering_results`'s column construction
  (this is the function whose missing `mentions_paris` column caused a
  `KeyError` in the original notebook — see Instructor Review #2), concept
  detection (case-insensitive), and strength-sorted ordering.

**Not yet covered:** `src/models/gpt2_sae.py`'s actual model/SAE loading
and `get_feature_activation` against real weights — that needs either a
real GPT-2 Small + SAE download in CI or a much heavier mock of the
TransformerLens/SAELens APIs. Add this once CI has a way to cache model
weights, or as a manual/slow-marked integration test.

## Added 2026-11-13 (audit remediation D8)

All CPU-only; no checkpoints, caches or model downloads.

- `test_modern_sae.py`: `src/sae/modern_sae.py` (the R2 SAEs). Covers:
  * TopK sparsity and selection, and the frozen decoder;
  * BatchTopK training (k·B kept) versus the inference threshold;
  * JumpReLU and step straight-through gradients;
  * the `fold_scale` round-trip for every architecture (z unchanged; output divided by the scale);
  * the checkpoint round-trip;
  * the R2 trainer's frozen configuration and LR schedule;
  * the real `train_step` on a small CPU SAE: the ReLU sparsity controller and the BatchTopK threshold update.
- `test_decision_rules_and_splits.py`:
  * the R2 rule (direction, all-pairs requirement, missing seeds) and the M5 paired rule;
  * the ±10 % L0 band and its use in `m5.compare`;
  * `owt_eval` CE/KL arithmetic against a hand computation (identity splice gives 0; KL(p_orig‖p_spliced) ≥ 0;
    the prediction count);
  * GoT split independence (groups never cross splits; cities and neg_cities place each city in the same split;
    the A/B halves are disjoint train groups);
  * checkpoint-registry integrity.

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

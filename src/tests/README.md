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
- `test_pca_cm_posthoc.py`: the Holm step-down used by the PCA-CM multiplicity check, and that the check reproduces
  the pre-registered PCA-CM classes and its committed summary (3 and 8 SAE-better cells; 1 and 4 after Holm).

## Added 2026-11-13 (H1-TEST paper revision, D36)

- `test_h1_test_paper_claims.py` (4 tests; suite 71 → 75): the H1-TEST paper table (`tables/t_h1test.tex`) equals a
  fresh render from `results/repe/repe_lm_test_evaluation/`; the hypothesis verdicts; every H1-TEST number quoted in
  the paper text (cells, controls, HT4, the 90 secondary counts) matches the artifacts; and the stale "borderline,
  pending the test" phrasings are gone from the paper sections. Skipped if the artifacts are absent.

## Added 2026-11-17 (withdrawn-claims regression guard, D36/D38)

- `test_withdrawn_claims_regression.py` (14 tests): fails if a claim withdrawn by AUDIT-SENS (D38: S2 persona-specific
  loss; S5 "frozen collapse is mostly perturbation / reader fragility", including its plain-summary form) or
  superseded by H1-TEST (D36: "the RepE-LM test has produced no results") is asserted in the paper sections, the
  course report, `docs/plain_summary.md`, the curated summary docs, the GPT-2 slide notes or `README.md`.
  * Text is normalised first (LaTeX comments, commands, `~` and `\,` spacing, Markdown emphasis, line wrapping and
    case), so re-worded or re-wrapped copies are still caught.
  * A sentence that marks the claim as historical ("withdrawn", "superseded", "invalid", …) is allowed.
  * The research ledgers (`RESEARCH_PROGRESS.md`, `RESEARCH_DECISIONS.md`) are not scanned, because they record
    withdrawals on purpose. Missing files are skipped, so the test runs on every branch.
  * Four self-tests cover the matcher (LaTeX, Markdown, comments, historical mentions).

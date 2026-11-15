# H1: held-out RepE-LM TEST protocol (2026-11-13). Status: FROZEN; implementation frozen (addendum); TEST pass authorised by the user and started

Evidence level: none yet. This folder contains a **protocol and a dataset, not a result**.

Frozen protocol and hashes: [`preregistration.json`](preregistration.json).
Remediation item: S3 in [`docs/remediation/issue_register.md`](../../../docs/remediation/issue_register.md).

## Why

RepE-LM had only train and validation splits. Every persona, representation (RLM-RP) and causal (M3, M4, E3, R5)
claim is therefore a VALIDATION finding, and the paper cannot confirm any honesty result. M1 planned for the test
prompts to be generated "by this same code at M5, with approval". M5 ran without them.

## What was done (data only)

[`scripts/repe/28_build_repe_lm_test_split.py`](../../../scripts/repe/28_build_repe_lm_test_split.py) builds
[`data/repe/repe_lm_test/`](../../../data/repe/repe_lm_test/). It imports M1's builder
(`scripts/repe/20_build_repe_lm_dataset.py`) without modifying it.

* **Statements:** the 90 TEST statements of `data/repe/persona_pairs.csv`. Five are excluded:
  * 4 same-label near-duplicates (token Jaccard > 0.8) of train or validation statements;
  * M1's dropped statement 326 (not a test statement; listed for completeness).

  That leaves **86 statements, 2,408 rows**, with factual-label mean 0.51.
  * 12 opposite-label counterfactual edits of train/validation statements remain. They are disclosed, not removed,
    exactly as M1's K6 rule specifies: each statement appears with both personas, so they carry no persona-label
    information.
* **Templates:**
  * M1's T1–T6 and N1–N2, unchanged;
  * **test-only wording**, written before any test evaluation: T7 "Act as a {candid | fraudulent} narrator
    describing the world." and T8 "Write {accurate | misleading} sentences about the world.", with neutral
    controls N3 (young | old) and N4 (green | yellow) in the same frames. No new slot word occurs in M1.
* **Construction checks (`split_verification.json`), all PASS:**
  * K1: 1,376 compared pairs, 0 failures. Equal length, exactly one differing position at a constant slot per
    template, and every slot word is one GPT-2 token.
  * Every cell is complete and labels are balanced within cells.
  * There is no exact normalised overlap with train/validation, and 0 same-label near-duplicates remain.
* **Prior exposure (disclosed):** the source test statements were scored once in the retired, length-confounded
  original RepE task (a smoke run and the historical causal test). RepE-LM, M1–M5, R2, R4 and R5 never read them.

No activation was extracted and no model, probe or SAE was run on these prompts.

## What is frozen for the one evaluation pass

The full specification is in `preregistration.json`.

* **Fitting:** every probe, direction and site is fitted or selected on RepE-LM TRAIN and applied frozen to TEST.
* **Positive controls:**
  * the original GPT-2 T_full is > 0 at each M3 site;
  * the persona KL exceeds the synonym KL.
* **Negative controls:**
  * neutral swaps N1–N4;
  * matched-norm isotropic and permuted-feature vectors;
  * the R2 random decoder.
* **Hypotheses (each replicates a validation finding):**
  * HT1: M4 causal preservation T_rec/T_full at L8, C1 and C3;
  * HT2: persona pair-direction cosine, C1–C3 at L1/8/11;
  * HT3: persona refit gap at L8, C1 and C3;
  * HT4 (descriptive): probe transfer to the test-only templates against the neutral floor.

  All use the R2 rule, with Holm correction across 12 cells.
* **Rules:**
  * The TEST evaluation script must be written, smoke-tested on validation only and hashed into the
    pre-registration **before** it reads a test row.
  * TEST rows are used once, and every outcome is reported.

## Status and next steps

* **PARTIALLY RESOLVED:** the protocol and dataset exist and are frozen; the evaluation has not been run.
* **Needs the user's approval:** about 1–2 GPU hours (activation extraction for 2,408 prompts, M4 patches for about
  60 SAEs).
* **Exact next steps:**
  1. Write `scripts/repe/29_repe_lm_test_evaluation.py`, reusing the functions of scripts 22, 24 and 26 unchanged.
  2. Smoke-test it on validation rows.
  3. Add its hash to `preregistration.json`.
  4. Run it once.
  5. Document the result whatever it shows.

## Implementation freeze (addendum, before any TEST row is read)

* The user authorised the single TEST pass on 2026-11-13.
* [`scripts/repe/29_repe_lm_test_evaluation.py`](../../../scripts/repe/29_repe_lm_test_evaluation.py) computes every
  metric with the validation ladder's unchanged functions; only row selection, activation extraction, HT4 and Holm
  are new.
* **Smoke test on VALIDATION rows only** ([`smoke_validation/`](smoke_validation/checks.json)): 6 SAEs reproduce the
  committed ladder to ≤ 1e-4 on every checked metric; live activations match the cache to ≤ 1.8e-4; the M4 rows are
  identical to `S24.build_rows`.
* **Checkpoints** ([`checkpoint_hashes.csv`](checkpoint_hashes.csv)): 81 sources; 78 in-house hashes equal
  `docs/checkpoint_registry.csv`; all 81 equal the hashes recorded in the validation ladder.
* [`preregistration_addendum.json`](preregistration_addendum.json) hashes the script, its imported modules, the
  validation comparison file and the checkpoint list. It also records clarifications:
  * The protocol says 12 HT1–HT3 cells, but its lists give 13; Holm is applied over 13.
  * A cell is confirmed only if the R2 rule classes it IMPROVEMENT on TEST and its Holm-adjusted Welch p < 0.05.
  * No PCA metric is computed on TEST.
* Expected cost: about 40–60 s per SAE, about 1–1.5 GPU hours on one A10G (smoke: 458 s for 6 SAEs), under 8 GB of
  GPU memory.
* Results go to `results/repe/repe_lm_test_evaluation/` (the script refuses to overwrite it).

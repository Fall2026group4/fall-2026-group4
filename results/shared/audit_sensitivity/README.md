# AUDIT-SENS: sensitivity analyses after the 2026-11-13 scientific audit. Status: PRE-REGISTERED (plan below); results in §3 when run

**Evidence status of everything in this folder: SENSITIVITY / DIAGNOSTIC (exploratory).**
* None of these analyses is a new confirmatory test.
* No frozen protocol, frozen result or committed number is changed.
* No TEST row is read. S1 re-uses only the *committed* TEST seed statistics of H1-TEST and M5.

## 1. Why

The audit (`docs/RESEARCH_PROGRESS.md` §44) found problems that can change what the existing results support:

| ID | Audit finding | Analysis |
|---|---|---|
| S1 | The R2 rule's Welch CI covers seed variance only; statement and document sampling is ignored (M5, H1-TEST) | Paired cluster bootstrap on VALIDATION; transfer to the committed TEST seed statistics |
| S2 | "Part of the persona loss is persona-specific" rests on raw gaps; the neutral contrast has far less headroom | Headroom-normalised loss, persona vs neutral, paired statement bootstrap |
| S3 | L0 is matched on OWT only; the arms differ strongly on task prompts | Inference-time truncation to a common task L0; C1–C3 and the old-architecture ordering re-evaluated |
| S4 | Non-linear GoT probes read "truth" at layers 1–3 where the logistic probe is at chance | Diagnostics on original activations: per polarity, permutation null, cross-polarity transfer, text-only baselines |
| S5 | The only noise reference is isotropic, which is especially damaging to a standardised logistic probe | Row-shuffled SAE error and error-covariance Gaussian references |
| S6 | Logistic C grid {0.01, 0.1, 1}; the original persona probe sits at its edge | Extended 11-value grid; C1–C3 re-fit-gap classes re-checked |

All six analyses connect to the main research question in the same way. They test whether the stated
truth/honesty-preservation and causal-preservation conclusions survive a fairer comparison and a fuller account of
uncertainty.

## 2. Design (fixed before the full runs)

Scripts (`scripts/shared/audit_s{1..6}_*.py`) re-use the project's unchanged functions:
* the M2 probes;
* the RLM-RP pair geometry;
* the M4 decomposition;
* the OWT splice evaluation;
* the R2 rule.

Each script checks that it reproduces the committed per-SAE values before using them. Split, seed, bootstrap and
decision-rule details are in each script's docstring and in `preregistration.json`.

* **S1.**
  * For each comparison: se = sqrt(se_seed² + v_stmt).
    * v_stmt is the variance of the paired cluster-bootstrap difference of seed means (B = 2000; clusters are
      statements for RepE-LM and documents for OWT).
    * df = min(Satterthwaite, Welch), so the interval is never narrower than the frozen seed-only CI (fixed after a unit test showed Satterthwaite alone can narrow it; before any full run).
  * Applied to the 13 H1-TEST cells and the 18 M5 C1–C3 NMSE/ΔCE cells.
    * Test values use v_stmt scaled by n_val/n_test: 89/86 statements and 100/100 documents.
    * This assumes the per-cluster variance transfers from validation to test.
  * A cell is "supported under seed + statement uncertainty" if:
    * all 9 pairs are better;
    * the adjusted CI excludes 0;
    * Holm p < 0.05 (over 13 cells, or 18 for M5).
* **S2.**
  * Normalised loss NL = gap / (AUROC_orig − 0.5); ΔNL = NL_persona − NL_neutral.
  * Paired statement bootstrap, B = 1000.
  * A layer is "persona-specific" if the CI is above 0 and BH q < 0.05 over layers 1–11.
  * The reverse direction is reported too.
* **S3.**
  * k* = the minimum natural TRAIN-row L0 over the six arm types, per layer and task.
  * Each SAE gets a global threshold calibrated on TRAIN rows; the achieved VALIDATION L0 must be within ±5 %.
  * A class change between natural and matched L0 is flagged "L0-sensitive".
* **S4.**
  * Pooling artefact: pooled non-linear ≥ 0.65 while per-polarity non-linear ≤ 0.60.
  * Lexical shortcut: text baseline ≥ 0.65.
  * Pipeline leak: permutation null outside [0.45, 0.55].
* **S5.** A cell shows "label-aligned loss" if the CI of gap(SAE) − gap(shuffled SAE error) is above 0, with BH over
  cells.
* **S6.**
  * Extended grid {1e-4 … 10}.
  * A class change against the committed ladder is flagged "grid-sensitive".

## 3. Disclosure: what was seen before this plan was frozen

* **Smoke runs on VALIDATION**, all into the scratch directory and none of them committed:
  * S1, layer 8, seed 42: reproduction only, max |diff| 4e-5;
  * S2, layer 8, seed 42: the neutral loss was larger than the persona loss;
  * S3, layer 8, seed 42: all arms matched;
  * S5, layer 8, seed 42: the shuffled-error gap was smaller than the SAE gap;
  * S6, layer 8, seed 42;
  * S4, layer 1 (full diagnostics).
* **The S4 layer-1 smoke showed:**
  * per-polarity *logistic* probes at 0.75–0.76, with cross-polarity transfer at 0.76;
  * pooled logistic 0.52.

  After seeing this, one descriptive flag was added to S4: "pooled-linear suppression", meaning per-polarity
  logistic ≥ 0.65 while pooled ≤ 0.60. It is labelled POST HOC in the script and in every report.
* The audit itself had already read the committed M2, ladder, H1-TEST and M5 outputs.

## 4. Results

To be added after the full runs (§4 of this README; `docs/RESEARCH_PROGRESS.md` §45).

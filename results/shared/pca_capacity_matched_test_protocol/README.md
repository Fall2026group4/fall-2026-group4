# PCA-CM-T: test-split confirmation of capacity-matched PCA (2026-11-13). Status: COMPLETED (2026-11-22; TEST, rows previously exposed in M5; PCA arms unseen)

## Is PCA-CM confirmatory or post hoc?

* **Pre-registered, but on validation.** PCA-CM's protocol (references, rule, interpretation I1–I3) was hashed and
  committed before its run finished (`7e8664f`, 05:50 UTC; results `ee7b7ac`, 06:37 UTC).
* **Motivated by an audit of existing results.** The capacity-matched reference was proposed only after the audit saw
  that error-matched PCA used k = 298–519. The analysis is therefore pre-registered **exploratory** work: it tests a
  matching criterion chosen after seeing that the existing criterion was unbalanced. It is not a confirmation.
* **Multiplicity:** the per-cell rule has no correction. An approximate Holm correction keeps 1/3 (k = 128) and 4/8
  (task-matched) cells.
* **Conclusion:** the paper's capacity-matched statements rest on validation data alone, and they changed a headline
  claim (D34). A test confirmation is **scientifically justified**.

## Constraints on a test pass

* **GoT test rows are previously exposed** (D31). M5 has already read them and published, for every SAE in this
  family, the SAE re-fit gap and SAE − (error-matched PCA) on TEST.
* Only the PCA-128 and task-matched-PCA TEST gaps are unseen. A pass would therefore be labelled "TEST (rows
  previously exposed; PCA arms unseen)".
* **No new choice may use TEST.** The families, references and rules below are fixed from validation only.

## Frozen rules (before any execution)

* **Data:** GoT `cities` + `neg_cities` TEST (452 rows). Readers are fitted on TRAIN (2,092 rows), as in M5. PCA is
  fitted on OWT TRAIN, as in PCA-CM.
* **SAEs:** the M5 source list, exactly as in PCA-CM.
* **References:** `capacity_k128` (k = 128) and `task_l0_matched`. The task L0 is taken from the **validation**
  PCA-CM `rows.csv`, never from TEST.
* **Statistic:** per seed, a paired grouped-bootstrap 95 % CI of (SAE re-fit gap − PCA re-fit gap), with 1,000
  resamples by city group; the PCA-CM rule.
* **Confirmatory families:**
  * **P1 (no SAE worse at k = 128):** over all 20 trained cells, no cell is classified SAE WORSE. Reported per cell.
  * **P2 (k = 128 advantage):** the 3 validation-better cells (R2 ReLU L8 and L11, res-jb L8). A cell is confirmed if
    every seed's CI high < 0 and the Holm-adjusted p is < 0.05 over 3 cells (normal approximation to the bootstrap,
    intersection-union over seeds, as in `posthoc_multiplicity`).
  * **P3 (task-matched advantage):** the 8 validation-better cells, with the same rule and Holm over 8.
* **Reporting:** all 40 cells and every outcome. Failure to confirm is reported plainly, and the paper's
  capacity-matched sentences would then be labelled "validation only, not confirmed on test".
* **Implementation rule:** the evaluation script must be written, smoke-tested on VALIDATION (reproducing
  `results/shared/pca_capacity_matched/rows.csv`) and hashed into an addendum before it reads a TEST row.
* **Cost:** about 1 GPU hour (as PCA-CM, GoT part only).

## Status

* Not executed. The user's instruction was "Do not automatically run PCA-CM on test data", so this needs explicit
  approval.

## Results (single TEST pass, 2026-11-22; `test/`)

**Execution:**
* `scripts/shared/pca_cm_test_confirmation.py`, approved by the user.
* **Implementation rule met:** on VALIDATION it reproduced the committed PCA-CM `got_paired.csv` exactly (116/116
  rows, max |diff| 0.0; `validation_reproduction/`).
* The pre-TEST addendum (`addendum_pre_test.json`, hashes) was committed and pushed (`0046569`) before any TEST row
  was read.
* **Run history:** attempt 1 was stopped by the RAM guard before it wrote any output, so no TEST result was produced
  or seen (`run_log.txt`). It was relaunched unchanged.
* 452 TEST rows (cities + neg_cities); 116 paired cells; readers fitted on TRAIN.

| Family | Result | Cells |
|---|---|---|
| **P1**: no trained SAE worse than PCA at k = 128 | **CONFIRMED**: 0 of 20 cells "SAE WORSE" | all 20 |
| **P2**: SAE better than PCA at k = 128 (3 validation-better cells; Holm over 3) | **1 / 3 confirmed** | L11 R2 ReLU −0.056 (p 9e-4) ✓; L8 res-jb −0.030 (every-seed CI high < 0, but p 0.048 fails Holm) ✗; L8 R2 ReLU −0.043 (CI high +0.007) ✗ |
| **P3**: SAE better than PCA at the SAE's task L0 (8 cells; Holm over 8) | **4 / 8 confirmed** | L8 old ReLU −0.094 ✓, L8 res-jb −0.110 ✓, L11 R2 ReLU −0.067 ✓, L11 res-jb −0.075 ✓; not confirmed: L8 R2 ReLU, L11 old JumpReLU, L11 old ReLU, L11 R2 BatchTopK |

Differences are SAE − PCA refit logistic gaps; lower favours the SAE.

**Reading (D34 wording rule):**
* On held-out cities TEST, no SAE loses more truth information than a PCA with the same per-token active budget
  (k = 128).
* An SAE advantage over capacity-matched PCA is confirmed only in the listed cells.
* It is never described as "significantly better" in general.
* The TEST rows were previously exposed in M5, which is disclosed with every use.

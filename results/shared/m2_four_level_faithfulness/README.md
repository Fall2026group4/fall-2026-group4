# M2: four-level information faithfulness (108 SAEs × 12 layers × GoT cities / RepE-LM persona / RepE-LM neutral)

**Status:** PRE-REGISTERED 2026-11-08 (`preregistration.json`); results in §2 once run. **Evidence label:** VALIDATION;
fitting and selection on TRAIN; test not used.

* **Question (SQ2):** when an SAE replaces GPT-2's activation, which task information can still be read, and by
  which readers? Is the loss specific to the SAE, compared with random noise of the same size and with PCA at the
  same reconstruction error?
* **Labels per cell:**
  * Preserved;
  * Readout shifted: the frozen reader fails, but a re-trained reader recovers the information;
  * Reduced for F: re-trained reader family F does worse on x̂ than on x;
  * Not recovered by tested families;
  * plus a displacement flag (task information present in the error term).

  "Irrecoverable" is never claimed.
* **Script:** `scripts/shared/m2_four_level_faithfulness.py`. **Module:** `src/evaluation/information_levels.py`, with tests.

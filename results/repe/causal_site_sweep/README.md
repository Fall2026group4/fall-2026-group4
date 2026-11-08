# M3: causal site sweep in original GPT-2 (RepE-LM persona instructions)

**Status:** PRE-REGISTERED 2026-11-08 (`preregistration.json`); results in §2 once run. **Evidence label:** TRAIN
selection, VALIDATION confirmation; no SAE; test not loaded.

* **Question:** at which layer and token position does GPT-2 carry the effect of swapping "truthful" for
  "deceptive"? A site counts only if both of these hold:
  * copying the honest activation into the dishonest run there moves the output toward the honest output,
    in both directions;
  * the movement is larger than a synonym swap and matched-norm random vectors produce.
* **Design:** 65 cells (13 hook sites × 5 position groups); Holm correction; at most 3 sites; confirmed on validation.
* **Script:** `scripts/repe/23_positive_control_site_sweep.py`. **Module:** `src/evaluation/patching.py`, with tests.

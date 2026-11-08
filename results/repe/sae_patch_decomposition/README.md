# M4: SAE causal-faithfulness decomposition (persona-instruction effect, original vs SAE)

**Status:** PRE-REGISTERED 2026-11-08, with amendment A1 adding a mid-depth site (`preregistration.json`). Results in
§2 once run. **Evidence label:** VALIDATION; train only for the mid-site rule input and the feature ranking; no test.

* **Question (SQ4):** at sites where GPT-2 carries the persona-instruction effect (M3), how much of that effect does
  the SAE's *reconstructed* activation difference reproduce, and how much sits in the reconstruction *error*?
  Do a few SAE features reproduce it?
* **Sites:**
  * layer-1 persona slot (M3);
  * layer-11 final token (M3);
  * a layers 6–8 handover site, chosen by rule A1 on train and confirmed on validation, without SAEs.
* **Primary set:** T2–T4. T1 is reported separately as a control case.
* **Controls:** synonym and neutral one-word pairs, the opposite direction, matched-norm and decoder-shuffle vectors,
  and frequency/|Δz|-matched random feature sets.
* **Not claimed:** that GPT-2 "uses" SAE features.

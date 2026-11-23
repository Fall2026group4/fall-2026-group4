# XC-2: GoT causal truth-direction test on Gemma-2-2B (2026-11-23). Status: PRE-REGISTERED (frozen 2026-11-23 15:33 UTC), RUNNING

Evidence level: test phase = confirmed test result for families A, C1 and C2, subject to the TEST-exposure disclosure
below; everything else is exploratory.

Frozen protocol and hashes: [`preregistration.json`](preregistration.json).
Script: [`scripts/cross_model/21_xc2_gemma_got_causal.py`](../../../scripts/cross_model/21_xc2_gemma_got_causal.py).

## 1. Before implementation

* **Research question.** Do the original truth directions and the Gemma Scope truth latents causally move Gemma's
  next-token preference toward the TRUE country? Is the GPT-2 null result (Dhruv, `dhruv-causal` commit 989d313,
  600 controls) specific to GPT-2?
* **Why.** The GoT causal test existed only for GPT-2, with rebuilt SAEs (audit 2026-11-21, row 13). XC-2 adds the
  causal category for the second model, using public SAEs.
* **Port.**
  * The protocol is unchanged from `dhruv-causal:scripts/geometry_of_truth/22_truth_direction_causal_test.py`:
    * cities pairs: the shared prefix before the first differing token, with the true token c and the false token w;
    * outcome: Δ(log P(c) − log P(w)) at the final prefix token, higher is better;
    * targets: the mass-mean direction, the frozen-probe direction, and the SAE reconstruction, top-1 and top-10
      directions, each at ±s;
    * strength chosen on VALIDATION;
    * a graded unembedding positive control;
    * 600 controls per family;
    * BH correction within families A (28 tests), C1 (8) and C2 (12);
    * criteria T1 and T2, and power reported as the minimum detectable dose.
  * **Substitutions:**
    * **Model and site:** Gemma-2-2B `resid_post` at L = 8/12/16/21, matched in depth to GPT-2 4/6/8/10.
    * **SAE:** one Gemma Scope 16k canonical SAE per layer. There are no seeds, so T2's same-sign-across-seeds clause
      does not apply.
    * **Logit path:** final-token logits use an fp32 soft-cap. HF itself applies the soft-cap in bf16; replicating
      that bf16 path matches the full forward exactly, and is used only to check the indexing.
* **TEST disclosure.** GoT cities TEST rows are not pristine:
  * they were exposed during GPT-2 method decisions;
  * the 2026-11-01 Gemma L11 judgment-steering sweep scored them under a different protocol with 20 controls.

  No XC-2 choice was taken from those numbers. TEST is scored once.
* **Smoke tests (TRAIN pairs only).**
  * **`smoke/`:** found up to 0.34 nats per-pair difference between batch shapes (bf16 kernel selection). Fixed before
    freezing: every forward now has an identical shape, the baseline goes through the same path, and the path checks
    are hard stops.
  * **`smoke_v2/`:** re-checks the fix.
  * **Ceiling:** the smoke showed Gemma's baseline pair accuracy is 1.0. The outcome is therefore the continuous
    margin; accuracy is uninformative.

## 2. After execution

(filled in after the runs)

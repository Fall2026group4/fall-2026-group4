# M0: within-layer reanalysis of existing results (108 SAEs, validation)

**Status:** PRE-REGISTERED (2026-11-08); results in §3 once run. **Evidence label:** VALIDATION reanalysis. No model, SAE,
activation cache or test row is loaded. Decision log: `docs/RESEARCH_DECISIONS.md` D23.
Script: `scripts/shared/m0_within_layer_reanalysis.py`. Pre-registration and hashes: `preregistration.json`.

## 1. Question and why it is needed
* **SQ5.** Earlier reports said reconstruction quality "tracks" information and behaviour loss. Those correlations were
  **pooled across layers**. Both reconstruction error and the losses grow with depth, so pooling can create a
  correlation that does not exist between SAEs at the same layer. M0 asks the within-layer question.
* **SQ1 fairness.** Architectures are matched on OWT L0 within ±10 %, but ReLU runs at 131–135 active latents vs
  TopK's 128. M0 makes that difference explicit.
* **Claim corrections.** It checks every pooled claim and the "SAE error is 1.0–2.6× worse than random" wording.
* **Link to the research question:** whether reconstruction error can serve as a proxy for faithfulness, and whether
  architecture differences survive the L0 difference.

## 2. Design (see `preregistration.json` for the exact rules)
* **Units:** 108 SAEs (12 layers × ReLU/TopK/JumpReLU × 3 seeds); layer-0 JumpReLU seed 42 (L0 73.6) excluded.
* **Predictors** (lower is better):
  * `log_owt_nmse`, log(1 − OWT validation EV);
  * `log_got_rel_err`, log of ‖x − x̂‖²/‖x‖² on GoT validation final tokens. Task NMSE is not used: it exceeds 1 at
    layers 1–10 because the task-set variance is tiny.
* **Primary outcomes** (lower is better):
  * the GoT-cities frozen-logistic AUROC gap, at informative layers 4–11 only;
  * OWT splice ΔCE, nats per token.
* **Model:** outcome ~ β·predictor + layer FE + architecture FE + log L0. A secondary model drops architecture FE.
* **Inference:**
  * Freedman–Lane permutation within layer (5000); bootstrap CI (2000);
  * Holm correction over the 4 primary pairs.
* **Architecture table:**
  * per layer, mean ± sd over seeds;
  * L0-adjusted NMSE, using the 1000-step sweep's local elasticity;
  * break-even elasticity;
  * live latents.

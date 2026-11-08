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

## 3. Results (run 2026-11-08; validation; 107 included SAEs, 72 at the informative GoT layers 4–11)

### 3.1 Primary within-layer regressions (Holm over 4)
Model: outcome ~ β·predictor + layer FE + architecture FE + log L0. β is per unit of log error.

| Predictor → outcome | β [95 % CI] | standardised β | p (perm.) | p (Holm) | Pre-registered verdict |
|---|---|---|---|---|---|
| log OWT NMSE → GoT logistic gap | 0.268 [0.056, 0.477] | 1.51 | 0.015 | **0.044** | SUPPORTED by the rule, but **fragile** (see below) |
| log GoT relative error → GoT logistic gap | 0.005 [−0.059, 0.080] | 0.06 | 0.87 | 1.00 | NOT SUPPORTED |
| log OWT NMSE → OWT splice ΔCE | 0.165 [0.134, 0.265] | 0.56 | 0.0002 | **0.0008** | **SUPPORTED** |
| log GoT relative error → OWT splice ΔCE | −0.012 [−0.060, 0.028] | −0.08 | 0.63 | 1.00 | NOT SUPPORTED |

**Why the first row is fragile:**
* Architecture explains **99.8 %** of the within-layer variation in log OWT NMSE (the within layer × architecture share
  is 0.19 %). With architecture fixed effects, β is therefore estimated from a sliver of variation, hence the
  standardised β of 1.5.
* Without architecture FE (the secondary model) the association disappears: β = −0.005 [−0.069, 0.055], p = 0.89.
* The per-layer Spearman ρ is −0.08 to 0.40 (layers 4–11).
* **Reading:** within a layer, the SAE with the lower general-text error does *not* reliably keep more of the truth
  readout. The formal pass reflects how the architecture gap varies across layers, not a usable proxy.

**Behaviour is different:**
* Lower OWT error goes with less splice damage in both models (secondary β = 0.20, p = 0.0002).
* The per-layer Spearman ρ is 0.38–0.52 at layers 0–2 and 0.80–0.97 at layers 3–11.
* In practice, 10 % more NMSE goes with ≈ +0.016 nats/token of ΔCE.

**Secondary outcomes** (log OWT NMSE, primary model):
* GoT statement KL: β = 0.145, p = 0.0002;
* OWT KL: β = 0.169, p = 0.0002;
* GoT mass-mean gap: β = 0.072, p = 0.061.

### 3.2 Architecture comparison (OWT validation, mean of 3 seeds; seed sd of EV ≤ 0.004)
* **Layers 4–11:** JumpReLU < TopK < ReLU in NMSE. Every pair has Welch p ≤ 0.003, except TopK vs JumpReLU at layers
  10–11, which are tied (difference ≤ 0.3 %, p ≥ 0.18).
* **Layers 0–2:** ReLU has the lowest NMSE. Layer 2 ReLU vs JumpReLU is tied (0.35 %, p = 0.81).
* **Layer 3:** JumpReLU is best; ReLU and TopK are tied.
* **L0:**
  * ReLU uses 4.4–6.8 % more active latents than TopK at layers 1–11 (10.8 % at layer 0), and 1.9–3.0 % more than
    JumpReLU (5.6 % at layer 0).
  * The 1000-step sweep elasticities d log NMSE / d log L0 are ReLU −0.39, TopK −0.48 and JumpReLU −0.63 (means).
  * At those elasticities, L0 explains only about 2–3 % of NMSE, while ReLU's leads at layers 0–2 are 5.7–56 %
    (apart from the layer-2 tie with JumpReLU).
  * Break-even elasticities are 2.0–15, far above the observed values.
* **After the L0 adjustment no ranking changes** except two near-ties: TopK vs JumpReLU at layers 9 and 10
  (|difference| ≤ 1.2 % before and after).
* **Effective width differs a lot** (live latents of 3072):
  * ReLU 2920–3063;
  * TopK 1711–2851;
  * JumpReLU 469 (layer 0) to 2947.
* **Task-prompt L0 is not matched** (GoT final tokens):
  * TopK 128;
  * ReLU 8–110;
  * JumpReLU 12–92.

  The architectures are matched on OWT only.
* **Splice damage** follows reconstruction: ReLU has the highest ΔCE from layer 3 on (0.078–0.358 nats), and TopK has
  the highest at layers 0–2.

### 3.3 Corrections to earlier claims

| Earlier claim | Pooled ρ (layers mixed) | Within-layer mean ρ [min, max] | Status now |
|---|---|---|---|
| OWT EV tracks RepE persona-shift preservation | 0.89 | 0.05 [−0.52, 0.65] | **Pooling artefact** (and the old task is HISTORICAL) |
| Splice KL tracks honesty gap (ρ = 0.86) | 0.86 | 0.63 [0.22, 0.93] | Survives within layer, but on the length-confounded task: **not citable**; re-test in M2 |
| OWT EV predicts splice ΔCE | −0.97 | −0.79 [−0.97, −0.38] | **Holds within layer** |
| OWT EV predicts GoT statement KL | −0.94 | −0.60 [−0.93, 0.07] | Holds, weaker |
| GoT task error predicts truth gap | 0.56 | 0.27 [−0.23, 0.82] | Mostly a layer effect; weak within layer (demeaned ρ 0.26, p 0.03; regression NS) |
| OWT error predicts truth gap | 0.09 | 0.12 [−0.08, 0.40] | No association |
| "SAE error damages 1.0–2.6× more than random error" | — | — | **Wrong range.** Run level: 0.76–2.89 (median 1.64); 7/107 runs < 1, all at layers 1–3. Cell means 0.82–2.65 |

### 3.4 Limitations
* n = 9 SAEs per layer (3 per architecture); architecture and error are almost collinear within a layer.
* The GoT gaps have per-SAE CIs of about ±0.04 (n = 448).
* Elasticities come from unconverged 1000-step, seed-42 runs; layer-0 JumpReLU's is −1.5, which is unreliable.
* The frozen logistic gap is a readout-shift measure, not an information measure (M2).

### 3.5 Status
COMPLETED, VALIDATION. The 4 pre-registered primary pairs were evaluated: 2 SUPPORTED (1 of them fragile, as
explained above) and 2 NOT SUPPORTED.

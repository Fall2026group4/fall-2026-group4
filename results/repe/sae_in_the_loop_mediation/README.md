# SIL: SAE-in-the-loop (mediated) persona intervention (2026-11-13). Status: PRE-REGISTERED; pilot pending

* **Evidence level:** VALIDATION only. No test row is read; nothing is trained.
* **Requirement (Review #8, Major):** "SAE-in-the-loop intervention (your roadmap C3). Adding decoder directions to raw
  activations is not an SAE intervention, and that is the causal question in your title."
* **Script:** [`scripts/repe/30_sae_in_the_loop_mediation.py`](../../../scripts/repe/30_sae_in_the_loop_mediation.py).
* **Frozen protocol:** [`preregistration.json`](preregistration.json).

## 1. What was wrong with the existing interventions

* **M4 / E3 / R5:** every patch keeps the error term. It therefore equals adding one vector
  v = W_dec(z′ − z_U) to the raw activation x_U (D33).
  * This shows that the SAE **selects** directions that carry the effect when inserted.
  * It does not show that the effect **passes through** the SAE.
* **R2-clamp** (layer 6; encode → clamp → decode, with and without error): NOT SUPPORTED (validation). It also never
  removed the raw residual route downstream.

## 2. The new intervention, and how it differs mathematically

**Spliced model M_D.** At layer 1 and at each layer in D, `resid_pre` is replaced at every non-BOS position by
x̂ = SAE_L(x), with the error term dropped. Three configurations:
* **D0** = {}: the SAE is in the loop only at the edit site;
* **D1** = {8};
* **D2** = {8, 11}.

**Edit.** At the layer-1 persona slot of the dishonest prompt, the latents are swapped, z_U → z_H, and decoded.

**Effect.** T_S = KL(p^D_H ‖ p^D_U) − KL(p^D_H ‖ p^D_{U←H}), at the final token.

**How it differs from M4:**

| | Base at the edit | Downstream computation | Equivalent to adding a fixed vector to raw x? |
|---|---|---|---|
| M4 T_rec | x_U (raw, error kept) | raw GPT-2 | **yes** (x_U + v) |
| SIL D0 | x̂_U (error dropped at every position of layer 1) | raw GPT-2 | at the edit layer, yes but on a different base: x̂_U + v = x̂_H; the error route at layer 1 is removed |
| SIL D1/D2 | x̂_U | every path to the output passes through SAE_8 (and SAE_11) sparse codes at every position | **no**: the downstream TopK/ReLU encoders are nonlinear, so the effect at layers 8/11 is filtered by what those SAEs encode |

**What this does not establish:**
* At the edit layer itself, any latent edit is still the residual-space edit W_dec Δz. No claim beyond that is made.
* D1/D2 show whether the persona effect can be carried **through SAE representations**, in a spliced model. They do
  not show that unmodified GPT-2 uses SAE features.
* Splicing damages the model (ΔCE about 0.06–0.10 nats per layer for R2), so the spliced effect is compared with the
  spliced baseline KL(p^D_H ‖ p^D_U).

## 3. Controls and strength matching

All controls are applied at the same slot, in the same spliced model, with 20 draws each.
* **Negative control (iso):** x̂_U + an isotropic vector with ‖v‖ = ‖x̂_H − x̂_U‖.
* **Negative control (shuf):** x̂_U + W_dec(Δz permuted over latents), rescaled to ‖v‖. Same decoder, wrong
  latents.
* **Negative control (neutral):** the same latent swap for the neutral Na/Nb pairs, a generic one-word change.
* **Positive control / references:**
  * the original model's T_full (raw slot patch);
  * M4's error-kept T_rec on the same pairs;
  * the spliced baseline effect KL(p^D_H ‖ p^D_U).

## 4. Sites, SAEs and data

* **Edit site:** layer 1, persona slot. This is M4's L1 site, where the reconstruction carries 93–97 % of the effect.
  A later edit site would leave too few downstream SAE layers.
* **SAEs:** the same family and seed at layers 1, 8 and 11:
  * R2 TopK, BatchTopK, ReLU-12288 and old TopK, ReLU, JumpReLU-3072, each with seeds 42, 1042 and 2042;
  * the R2 random-decoder TopK (seed 42) as a negative control.
* **Data:** RepE-LM VALIDATION. Persona pairs use T2–T4 (M4's primary set); neutral pairs use N1–N2.

## 5. Pre-registered criteria

* **Feasibility gate (pilot: R2 TopK s42 and old TopK s42):**
  * the spliced persona effect must survive, i.e. KL(p^D2_H ‖ p^D2_U) has statement-bootstrap CI low > 0;
  * a run must take under 5 minutes per SAE triple.
  * If either fails, SIL is reported as uninformative for this model and the full run is not launched.
* **Primary, per family and configuration:**
  * **MEDIATED** if, on every seed, the T_S CI low > 0 and T_S exceeds both matched controls (paired CI low > 0).
  * **NOT MEDIATED** if the T_S CI includes 0 on every seed.
  * Otherwise **MIXED**.
* **Descriptive measures:**
  * T_S / T_full(original), with a ratio CI;
  * D2 vs D0, i.e. how much of the effect survives the downstream SAE bottlenecks;
  * R2 vs old families (R2 rule, validation only).
* **Interpretation:**
  * MEDIATED at D2 → the persona effect can be carried by the SAEs' sparse codes through the network: a genuine SAE-in-the-loop
    causal result, at validation level.
  * NOT MEDIATED at D2 but MEDIATED at D0 → the downstream SAEs discard the effect.
  * A test confirmation needs a separate pre-registration on the H1 test split.

## 6. Cost

* Per SAE triple: about 3 configurations × (3 + 40 control) passes over about 267 persona and 178 neutral pairs.
  Expected at most a few minutes on one A10G.
* Pilot: 2 triples. Full run: 19 triples. It runs only after the H1-TEST pass has finished.

## 7. Results

Pending.

# GPT-2 Small: SAE faithfulness, slide content (Aditi's part; draft 2026-11-13)

Content for the team deck, built only from verified results. Integration into the shared deck is **PENDING** a team
decision. One slide per section; the speaker note says what to say. Labels: **TEST** = the single M5 pass;
**VAL** = validation.

---

## 1. Question

Can we improve SAE reconstruction of GPT-2 activations **without** losing truth information or causal behaviour?

*Note:* four levels, measured separately: reconstruction, information, behaviour and causal.
Figure: `architecture/diagrams/gpt2_sae_faithfulness_pipeline.svg`.

## 2. Baseline: 108 SAEs, all 12 layers (TEST)

* Explained variance falls with depth: 0.93–0.95 at L0–2, 0.76–0.82 at L11.
* Matched sparsity: L0 = 128 ± 10 %, 3 seeds.

*Figure:* `results/openwebtext/figures/vector/all_layers_explained_variance.svg`.

## 3. Improvement works for reconstruction and behaviour (TEST)

* R2 (100M tokens, width 12,288): NMSE −30.8 to −73.7 %, spliced ΔCE −35.5 to −78.8 %, at L1/8/11, every seed pair.
* L8 TopK: NMSE 0.160 → 0.094; ΔCE 0.133 → 0.062 nats.
* R4 KL fine-tune at fixed L0: a further −16 to −17 % ΔCE, at a +2 % NMSE cost.

## 4. Truth information does not follow (TEST)

* GoT (cities + neg_cities, 452 test statements): 0/34 comparisons change the re-fit probe gap.
* The frozen-probe "collapse" is mostly generic fragility: norm-matched noise does the same.
* Sensitivity: see slide 6.

## 5. SAE vs PCA depends on how PCA is matched (VAL)

| PCA reference | PCA dims | SAE better | inconclusive | SAE worse |
|---|---|---|---|---|
| Equal reconstruction error | 298–519 | 0 | 7 | 13 |
| Fixed 128 dims (= SAE active budget) | 128 | 3 | 17 | 0 |
| Matched to SAE task activity | 54–128 | 8 | 12 | 0 |

* 20 trained SAE family × layer cells, GoT validation (448 statements), re-fit probe gap.
* "Better" = every seed's paired 95 % CI excludes 0; no correction across cells (approx. Holm keeps 1/3 and 4/8).
* Example: R2 ReLU L11: SAE 0.076, PCA-128 0.153, error-matched PCA 0.038.
* Takeaway: "better or worse than PCA" depends on what is held equal.
* Source: `results/shared/pca_capacity_matched/`.

## 6. How large an effect could we detect?

* Minimum detectable effect (α 0.05, power 0.80) = 2.8 × SE from the stored bootstrap intervals.
* GoT, one SAE's re-fit gap: 0.030–0.048 AUROC. Between-family comparisons: **about 0.05–0.09 AUROC**.
* Observed GoT differences: ≤ 0.025, so "no change" means "no change larger than ≈ 0.05".
* Causal preservation at L8: one SAE resolved only to 0.054–0.107. The R2 gain (+0.10) is borderline.
* Source: `results/shared/power_sensitivity/`.

## 7. Causal faithfulness (VAL)

* The SAE reconstruction carries 93–97 % of the persona effect at L1, but only 62–68 % at L8/L11.
* R2 raises this to 0.73 at L8 (TopK; borderline, see slide 6); R4 lowers it at L11 (0.694 → 0.671).
* The effect is small: 0.022 nats at L8.
* These patches keep the error term. They show the SAE *selects* effect-carrying directions, not that the effect is
  *mediated* by the SAE.

## 8. What we claim, and what we don't

| Claim | Label |
|---|---|
| Better SAEs → better reconstruction and behaviour | TEST |
| Truth information unchanged | TEST, within the detectable range |
| Honesty and causal gains | VAL only; frozen RepE-LM test protocol not yet run |
| "SAEs are worse than PCA" | only at equal error (see slide 5) |

## 9. Next

* RepE-LM test pass (frozen protocol).
* Mediated SAE-in-the-loop test.
* Task-level sparsity matching.
* Gemma (teammates).

# XC-1: reduced positive-control causal site sweep on Gemma-2-2B (2026-11-23). Status: COMPLETED (validation finding; TRAIN selection, VALIDATION confirmation)

Evidence level on completion: TRAIN selection; VALIDATION confirmation (validation finding). No SAE; TEST never loaded.

Frozen protocol and hashes: [`preregistration.json`](preregistration.json) (sha256 c1303b62…).
Script: [`scripts/cross_model/24_xc1_gemma_site_sweep.py`](../../../scripts/cross_model/24_xc1_gemma_site_sweep.py).

## 1. Before implementation
* **Question.** Where does original Gemma carry the persona-instruction effect (RepE-LM T1–T4) under the M3
  protocol? Which sites justify an M4 decomposition (M4-G)?
* **Port of M3 (`scripts/repe/23`).** Outcomes, statistics, the selection rule and the stop rule are unchanged.
* **Declared substitutions:**
  * **Model and site:** Gemma resid_post.
  * **Reduced grid:** L ∈ {0, 2, 5, 8, 12, 16, 19, 23, 25} × 5 position groups = 45 cells; Holm over 45.
  * **Trivial cell:** only (25, final).
  * **TRAIN screening:** 100 statements × T1–T4, with random controls on 30 of these statements.
  * **VALIDATION:** all 89 statements × T1–T4.
  * **Shapes:** every forward pass uses an identical shape.
  * **Logits:** the soft-cap is applied in fp32.
* **Hard sanity checks.** In the smoke test, all were exactly 0.0.
* **Smoke (TRAIN only, `smoke/`).** Mechanics only; not evidence.

## 2. Run
* Full run 19:23–20:48 UTC: 14,200 fixed-shape forwards, 5,084 s. Script sha256 `4dcac3d0…` matches the frozen
  pre-registration.
* Pairs: TRAIN 400 (100 statements × T1–T4; random controls on 30 statements), VALIDATION 356 (89 × T1–T4).
  0 pairs dropped for length or multi-token swaps. C_A (secondary) is missing for 144 TRAIN and 8 VALIDATION pairs
  whose continuation does not tokenise cleanly.
* **Hard sanity checks: all exactly 0.0** (`sanity_checks.json`): all-position patches at L0/12/23 reproduce p_H,
  the zero patch gives R = 0, (25, final) R_A equals KL(p_H‖p_U), batch-row position and the zero additive edit are
  identities. `test_rows_used = 0`.

## 3. Results (TRAIN selection → VALIDATION confirmation)
* **24 of 45 cells pass** on TRAIN (Holm over 45 cells, R_A above the q95 of 200 random draws, non-trivial); 36 were
  candidates for the extra random draws.
* **Selected cells** (rule unchanged from M3):

| Role | Cell | Depth | R_A TRAIN | E_share TRAIN | R_A VALIDATION [95 % CI] | R_S val | R_B val | E_share val | random q95 val | Confirmed |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | L0 slot | 0.04 | 0.0720 | 0.977 | 0.1079 [0.0772, 0.1438] | 0.0606 | 0.1284 | 0.983 | 0.0217 | **yes** |
| peak + latest | L23 final | 0.92 | 0.0724 | 0.983 | 0.1080 [0.0771, 0.1455] | 0.0608 | 0.1282 | 0.984 | −0.0213 | **yes** |

  (KL(p_H‖p_U) mean = 0.0737 TRAIN, 0.1098 VALIDATION.) Peak and latest coincide: L23 final has the largest
  non-trivial R_A and is the highest passing layer (L23 prefix_excl_final also passes, but R_A 0.005).
* **Stop rule: not triggered. M4 on Gemma is justified** (`selected_sites.json`: `M4_justified = true`).
* **Site map (TRAIN E_share = R_A / KL(p_H‖p_U)):** the effect is carried at the persona token early
  (slot E_share 0.98 L0 → 0.92 L2 → 0.69 L5 → 0.38 L8 → 0.15 L12 → 0.04 L16), moves through the post-slot
  instruction tokens at L5–L12 (0.46–0.56), and is collected at the final token late (0.24 L8 → 0.51 L12 → 0.85 L16
  → 0.93 L19 → 0.98 L23). The assistant span never carries much (≤ 0.20).
* **By template (VALIDATION, both selected cells):** T1 R_A 0.020, T2 0.075, T3 0.033, T4 0.302. T4 dominates the
  pooled mean, as on GPT-2. Each template alone has R_A CI low > 0 at both cells.
* **Comparison with GPT-2 M3 (descriptive only, not tested):** GPT-2 selected blocks.1 slot (earliest + peak) and
  blocks.11 final (latest), E_share ~0.96–0.97. Gemma shows the same two-site structure (persona token early,
  final token late); on Gemma the slot and final E_shares cross between L8 and L12 (depth 0.35–0.50).

## 4. Criteria
| Pre-registered item | Outcome |
|---|---|
| Hard sanity checks | PASS (all 0.0) |
| ≥ 1 passing cell | MET (24) |
| Selected cells confirmed on VALIDATION (R_A CI low > 0) | MET (2 of 2) |
| M4 on Gemma justified | YES → M4-G runs at L0 slot, L23 final (+ mid site by the M4-G rule) |

## 5. Limitations
* Reduced grid (9 of 26 layers) and 100 of 417 TRAIN statements; sites between grid layers are not resolved.
* VALIDATION random q95 is computed only for the selected cells (200 draws); other validation rows are descriptive.
* Evidence level: validation finding. No SAE is involved; no TEST row was loaded.

## 6. Files
`cells_train.csv`, `cells_validation.csv`, `cells_validation_by_template.csv`, `selected_sites.json`,
`sanity_checks.json`, `pairs_info.json`, `random_draws.npz` (80 KB). `run.log` is not committed.

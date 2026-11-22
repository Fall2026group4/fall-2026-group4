# TSP-R Phase 1: replication of the TSP fine-tune (2026-11-15). Status: COMPLETED (VALIDATION, confirmatory; pre-registered)

**Verdict (frozen rules, `eval/analysis/final_verdict.json`):**
* **Primary (cities):** SUPPORTED (meaningful); truth-specific; linear and nonlinear; no distortion.
* **Qualifier:** TRADE-OFF. Two behaviour guards (RepE persona gap, M4) could not demonstrate non-inferiority. Their
  point estimates are ≈ 0; the bounds are wide (§4).
* **Fresh held-out families (TEST, locked until the cities analysis was committed in `a6a514f`):** no detectable
  transfer.

**Evidence level:** VALIDATION, confirmatory. Cities train+val groups (cross-fitted). The cities TEST split was never
loaded. Fresh-family TEST splits were read once, after the commit.

**Frozen inputs:**
* plan: `preregistration.json` (D42, commit `804ae37`);
* evaluation code: `eval_freeze.json` (commit `7129dc3`).

**Scripts:**
* `scripts/geometry_of_truth/30_tsp_r_phase1_training.py`;
* `31_tsp_r_phase1_evaluation.py`;
* `31b_tsp_r_eval_integration_test.py`;
* `src/evaluation/tspr_stats.py`.

## 1. Execution (all stopping rules passed)

| Step | Result |
|---|---|
| Repro check (pilot ctrl seed 42 re-trained with the wrapper) | max \|Δw\| = **0.0**, bit-identical (`smoke_repro/repro.json`) |
| 5 new R2 TopK L8 bases (3042–7042), trained jointly, 3,840 s | QC **pass**: final EV 0.90510–0.90542 vs originals 0.90525–0.90531 (window ±0.005); 99,999,744 tokens; L0 128; 0 dead; GPT-2 unchanged (`base_qc.json`) |
| 120 fine-tunes, seed-major, 13,776 s (≈ 115 s each) | 120 / 120. L0 = 128 on OWT and GoT in every run. SHA-256 recorded in `run_ledger.csv`. GPT-2 unchanged |
| Evaluation | cities 144 SAE × fold evaluations; OWT 128 SAEs; ladder 48 fold-0 SAEs; TEST 128 SAEs × 3 families |

**Before launch, the QC thresholds were reviewed:**
* The ±0.005 EV window is about 180 seed-sd wide, so it cannot fail a correctly trained independent seed, and it
  flags gross failures.
* Subtle deviations are covered by the token-count, dead-fraction and L0 checks, and by the old-vs-new seed
  sensitivity.
* No threshold was amended.

## 2. Primary: cities re-fit logistic gap (lower is better)

Uncertainty: seed + city-bootstrap se with Satterthwaite df; 8 seeds; 635 groups.

| Comparison | Δ [95 % CI] | p (raw) | Adjusted p (gatekeeping) | Seeds < 0 | Decision |
|---|---|---|---|---|---|
| **H1 tsp − ctrl** | **−0.0207 [−0.0261, −0.0153]** | 6.8e-10 | 6.8e-10 | 8/8 | rejected → SUPPORTED (meaningful; ≤ −SESOI 0.010) |
| **H2 tsp − shuf** | −0.0202 [−0.0267, −0.0137] | 1.0e-06 | 1.0e-06 | 8/8 | rejected |
| **H3 tsp − rand** | −0.0220 [−0.0274, −0.0166] | 2.0e-11 | 6.8e-10 | 8/8 | rejected → truth-specific |

* **Per-seed H1 values:** −0.018, −0.026, −0.022, −0.017, −0.017, −0.028, −0.022, −0.015.
* **Statement-sampling share of variance:** 62 %; df 48.7.

**Secondary (cities):**

| tsp − ctrl | Δ [95 % CI] | p |
|---|---|---|
| Re-fit MLP 10 × 64 (information qualifier) | −0.0032 [−0.0054, −0.0011] | 0.003 → "linear and nonlinear" |
| Re-fit MLP 5 × 256 | −0.0069 [−0.0125, −0.0012] | 0.022 |
| Frozen logistic (readout metric) | −0.0348 [−0.0409, −0.0287] | 2e-17 |
| Frozen mass-mean (readout metric) | −0.0081 [−0.0099, −0.0063] | 2e-9 |
| tsp − pca, re-fit logistic | −0.0210 [−0.0270, −0.0151] | 6e-8 |

**Absolute re-fit gaps:**
* R2 base 0.0675 [0.0563, 0.0787];
* ctrl 0.0324 [0.0244, 0.0405];
* tsp 0.0117 [0.0035, 0.0199].

TSP removes ≈ 64 % of the gap left by ctrl. It does not reach 0, so the distortion guard is not triggered.
The controls do not differ from ctrl on the re-fit logistic gap: shuf −0.0005, rand +0.0013, pca +0.0004 (all
CIs include 0).

**Sensitivity (pre-registered, not decisive). Every analysis agrees:**

| Analysis | tsp − ctrl [95 % CI] |
|---|---|
| Two-way bootstrap (seeds × cities) | −0.0207 [−0.0264, −0.0145] |
| Pilot rule, t(7) | −0.0207 [−0.0271, −0.0143] |
| Existing seeds 42/1042/2042 | −0.0221 [−0.0303, −0.0139] |
| New seeds 3042–7042 | −0.0198 [−0.0270, −0.0127] |
| Fold 0 / 1 / 2 | −0.0155 / −0.0238 / −0.0227 (all CIs < 0) |
| 523 previously train-only cities (pilot validation cities excluded) | tsp−ctrl −0.0206; tsp−shuf −0.0207; tsp−rand −0.0241 (all CIs < 0) |

## 3. Locked TEST families: secondary and exploratory, never confirmatory

Re-fit logistic gap, tsp − ctrl.

| Family (role) | Δ [95 % CI] | p | ctrl / tsp absolute gap |
|---|---|---|---|
| companies (secondary; MDE ≈ 0.033) | −0.0036 [−0.0129, +0.0057] | 0.40 | 0.0043 / 0.0007 (CIs include 0) |
| common_claim (distortion check) | −0.0012 [−0.0034, +0.0010] | 0.25 | 0.0011 / −0.0002 → **no distortion** |
| counterfact, held-out relations (exploratory) | +0.0008 [−0.0026, +0.0043] | 0.59 | 0.0150 / 0.0158 (headroom exists; not reduced) |

**Original TEST AUROC:** companies 0.821; common_claim 0.695; counterfact 0.599.

**companies has much less TEST headroom than DEV suggested:** the base gap was 0.006 on TEST against 0.023 on DEV. On
counterfact a real gap (≈ 0.016) exists and TSP does not reduce it. The CI excludes reductions larger than 0.003
(exploratory).

## 4. Cost guards (tsp vs ctrl, one-sided 95 % bounds)

| Guard | Estimate | Bound | Margin | Pass |
|---|---|---|---|---|
| OWT NMSE relative | +0.018 % | +0.044 % | +2 % | yes |
| Spliced ΔCE relative (fold 0) | −0.25 % | +0.54 % | +5 % | yes |
| Spliced KL relative (fold 0) | +0.06 % | +0.22 % | +5 % | yes |
| RepE persona re-fit gap | −0.0020 | +0.0333 | +0.02 | **no** (imprecise) |
| M4 T_rec/T_full | +0.0037 | −0.0609 | −0.02 | **no** (imprecise) |
| L0 = 128 on OWT, GoT, RepE | exact | — | — | yes |

**Why the two guards fail.**
* The ladder reports only per-SAE CIs (≈ ±0.03 persona, ±0.055 M4). The frozen implementation (`eval_freeze.json`)
  turns them into a conservative statement variance (arms treated as independent). That makes these bounds
  uninformative.
* The guards fail for lack of precision; there is no measured cost.
* **Post-hoc, exploratory only (not the frozen rule):** seed-only one-sided bounds pass easily. Persona upper bound is
  +0.0017 and M4 lower bound −0.0025, against margins of 0.02.
* By the frozen rule the label is TRADE-OFF. In words: non-inferiority on persona/M4 is "not demonstrated", not
  "violated".

**Absolute cost of every arm vs the R2 base:**
* OWT NMSE +1.4–1.5 %;
* spliced ΔCE +1.7–2.1 %;
* KL +1.5–1.6 %.

This cost is the GoT-replay fine-tune itself, shared by all arms.

## 5. Interpretation

* **Supported:** the pilot's in-distribution effect replicates at the same size (pilot −0.020; TSP-R −0.0207), with
  8/8 seeds and 3/3 folds.
* **Truth-specific:** it beats the label-shuffled, random and PCA subspaces, on new cities not used in the pilot's
  evaluation.
* **Not just the targeted readout:** it shows up in re-fitted linear probes and, more weakly (−0.003 to −0.007), in
  MLP probes.
* **Not supported:** transfer beyond the cities template and entities.
  * companies and common_claim show nothing.
  * The exploratory counterfact relations show nothing despite measurable headroom.
* TSP improves preservation of the truth information it was trained on (city–country statements). There is no
  evidence that it preserves truth information in general.

## 6. Deviations from the frozen plan (all recorded before reading the affected outputs)

* **Evaluation code, fix 1.** The `final` stage failed to load its own npz files: entity IDs were saved as object
  arrays. `allow_pickle=True` was added to the two loaders in `test_estimates`. This touched no metric or rule, and
  TEST was not re-read; the cached scores were reused. The script hash after the fix differs from `eval_freeze.json`.
* **Ladder runtime** was ≈ 195 s per SAE, not 80 s. There was no effect on results.

## 7. Files

* `eval/analysis/`:
  * `primary.json`, `comparisons_cities.csv`, `sensitivity_cities.csv`;
  * `cost_guards.csv`, `absolute_costs_vs_base.csv`;
  * `per_sae_cities.csv`, `per_sae_owt.csv`;
  * `test_families.csv`, `final_verdict.json`.
* `eval/ladder_fold0.csv`, `run_ledger.csv`, `base_qc.json`.
* Gitignored, local:
  * checkpoints: 8.5 GB fine-tunes + 0.36 GB bases;
  * score caches `eval/{cities,owt,test}/`;
  * logs in `../replication_logs/`.

# XF: fresh, never-used cities truth TEST from geonames (2026-11-23). Status: COMPLETED (single TEST pass done 2026-11-23 19:11–19:21 UTC)

Evidence level on completion: confirmed test result on a pristine TEST split.

Frozen protocol and hashes: [`preregistration.json`](preregistration.json) (sha256 3ec9b134…, frozen 2026-11-23 ~19:06 UTC).
Script: [`scripts/cross_model/23_xf_fresh_cities_test.py`](../../../scripts/cross_model/23_xf_fresh_cities_test.py).
Data: `data/got/fresh/geonames_cities/statements.csv` + `build_info.json`.

## 1. Before implementation

* **Research question.**
  * (P1) Do the public SAEs (res-jb, Gemma Scope 16k canonical) preserve linearly decodable city truth on statements
    no decision has touched?
  * (P2) Does the truth-subspace-preserving fine-tune (TSP-R L8 TopK, TI-1 L11 TopK / L8 BatchTopK / L8 ReLU) reduce
    the refit truth gap vs its reconstruction-only control on fresh cities?
* **Why.** GoT cities TEST rows were exposed during GPT-2 method decisions (audit 2026-11-21 row 23), and TSP-R / TI-1
  are VALIDATION-level. XF is the only route to a TEST-level claim for them.
* **Data (text-only build, no model run).**
  * Source: geonames places with population 200k–500k (GoT cities uses > 500k).
  * Filters:
    * country in the GoT pool;
    * ASCII simple names;
    * not a GoT cities / neg_cities name, and not a country name;
    * unambiguous (no other-country place ≥ 10k with the same name);
    * ≤ 30 cities per country.
  * Statements: one TRUE and one FALSE per city, GoT template.
  * Split: by city with a salted hash, DEV 40 % / TEST 60 %.
  * Size: 713 cities, 78 countries; DEV 263 cities (526 statements), TEST 450 cities (900 statements).
  * Hard leakage checks, all 0:
    * duplicates with project text;
    * city overlap with GoT;
    * duplicate statements;
    * cities in both splits.
* **Sites.**
  * GPT-2 `hook_resid_pre`, final token: L3/6/9 (public) and L8/11 (TSP).
  * Gemma `resid_post`, final token: L5/12/19 (triplet T-B).
* **Gate (DEV only).** Grouped-CV logistic AUROC on original activations ≥ 0.70.
* **Readers.** Fitted on DEV, scored on TEST:
  * frozen logistic and frozen mass-mean (fit on original activations);
  * refit logistic (fit on reconstructions; primary).
* **Uncertainty.**
  * City-cluster bootstrap, 2000 resamples.
  * TSP: `tspr_stats.hierarchical` (seed + city).
* **Criteria.**
  * **P1, per public cell:**
    * LOSS: CI low > 0;
    * NON-INFERIOR: CI high < 0.010;
    * otherwise INCONCLUSIVE;
    * prefixed CEILING when the original AUROC CI high ≥ 0.995.
  * **P2, per TSP cell:** `h1_label`, SUPPORTED if Holm p < 0.05.
    * REPLICATES if ≥ 3/4 cells are supported and none is HARM.
    * FAILS TO REPLICATE if ≤ 1 cell is supported.
    * PARTIAL otherwise.
* **Not evaluable.** TI-2 and TI-4 checkpoints were lost with the old instance's scratch disk.

## 2. Smoke (GoT cities TRAIN rows only; not evidence)
* `smoke/smoke_results.json`: the pipeline ran end to end on 85 SAEs.
* GPT-2 L3 failed the gate on TRAIN (0.65).
* Gemma L12 and L19 had AUROC 1.0, so the ceiling rule was added before freezing.

## 3. DEV gate (2026-11-23 19:07–19:09 UTC; `gate/gate.json`)
All 8 cells are eligible. Grouped-CV AUROC: GPT-2 L3 0.768, L6 0.795, L8 0.809, L9 0.807, L11 0.789; Gemma L5 0.948,
L12 0.999, L19 0.997. No TEST row was used.

## 4. TEST results (single pass; `test/results.json` sha256 cb71c924cce3…, `test/test_info.json`)

**Run.**
* 19:10:48–19:21:35 UTC, 606 s, A10G, TF32 off.
* The pre-registration hash (3ec9b134…) and every HASHED file were checked at test and at analyze.
* `gate.json` was committed (`31c9cf0`) before TEST was read.
* 900 TEST statements (450 cities).
* 182 SAEs scored, each checkpoint SHA-256-verified against its ledger:
  * 6 public;
  * 80 TSP-R L8 TopK;
  * 30 for each TI-1 cell.
* Readers were fitted on DEV only.
* The TEST activation cache (`test/test_acts.npz`, 37 MB) and the per-SAE score files (`test/scores/`, 3 MB) are kept
  on EBS and not committed.

**Original-activation AUROC on TEST (frozen logistic, fit on DEV).**
* GPT-2: L3 0.801, L6 0.844, L8 0.847, L9 0.846, L11 0.828.
* Gemma: L5 0.949, L12 0.999, L19 0.998.

**P1: public-SAE truth preservation** (gap = AUROC_orig − AUROC_rec, lower is better; 95 % city-bootstrap CI)

| Cell | SAE (mean L0 on fresh rows) | Refit-logistic gap | Frozen-logistic gap | Frozen mass-mean gap | P1 label (refit) |
|---|---|---|---|---|---|
| GPT-2 L3 | res-jb (18) | +0.045 [+0.022, +0.068] | +0.117 [+0.090, +0.143] | +0.069 | **LOSS** |
| GPT-2 L6 | res-jb (48) | +0.053 [+0.037, +0.069] | +0.095 [+0.073, +0.119] | +0.013 | **LOSS** |
| GPT-2 L9 | res-jb (61) | +0.037 [+0.022, +0.052] | +0.125 [+0.101, +0.149] | +0.048 | **LOSS** |
| Gemma L5 | Gemma Scope 16k (40) | +0.036 [+0.024, +0.048] | +0.092 [+0.074, +0.111] | +0.014 | **LOSS** |
| Gemma L12 | Gemma Scope 16k (72) | +0.0001 [−0.0003, +0.0007] | +0.0007 [+0.0001, +0.0014] | +0.0005 | CEILING (uninformative) |
| Gemma L19 | Gemma Scope 16k (65) | +0.003 [+0.002, +0.005] | +0.005 [+0.003, +0.008] | +0.011 | CEILING (uninformative) |

**P2: truth-subspace fine-tune vs reconstruction-only control on fresh cities** (refit-logistic gap; hierarchical
seed + city uncertainty; Holm over 4 cells)

| Cell (seeds) | tsp − ctrl [95 % CI] | p / Holm | Label | tsp − rand | ctrl − base | tsp − base |
|---|---|---|---|---|---|---|
| TSP-R L8 TopK (8) | −0.0007 [−0.0034, +0.0020] | 0.58 / 1.0 | NO MEANINGFUL EFFECT | −0.0048 [−0.0081, −0.0014] (8/8 seeds < 0) | −0.0143 [−0.0218, −0.0067] | −0.0150 |
| TI-1 L11 TopK (3) | −0.0029 [−0.0171, +0.0113] | 0.53 / 1.0 | INCONCLUSIVE | −0.0039 [−0.0074, −0.0005] | −0.0053 | −0.0081 |
| TI-1 L8 BatchTopK (3) | +0.0006 [−0.0053, +0.0065] | 0.84 / 1.0 | NO MEANINGFUL EFFECT | −0.0025 | −0.0070 | −0.0064 |
| TI-1 L8 ReLU (3) | +0.0020 [−0.0007, +0.0047] | 0.13 / 0.51 | NO MEANINGFUL EFFECT | −0.0026 | −0.0256 [−0.0397, −0.0116] | −0.0237 |

Absolute refit gaps of the TSP-R L8 TopK arms: base +0.024, ctrl +0.010, tsp +0.009. The frozen-logistic tsp − ctrl
contrasts are also null (−0.005 to +0.001).

**Criteria.**
* **P1:**
  * **LOSS at all four informative cells** (GPT-2 L3/6/9, Gemma L5): both public SAEs lose 0.04–0.05 refit AUROC and
    0.09–0.13 frozen AUROC on never-used statements.
  * Gemma L12/L19 are at ceiling (original AUROC ≈ 0.998), so their near-zero gaps are **uninformative** under the
    pre-registered ceiling rule.
* **P2: FAILS TO REPLICATE** (0 of 4 cells SUPPORTED; none HARM).
  * Three cells show NO MEANINGFUL EFFECT: the 90 % CI lies inside ±0.010, so the effect is shown to be absent at the
    SESOI. L11 is INCONCLUSIVE.
  * The VALIDATION effects (TSP-R −0.021; TI-1 SUPPORTED at 3/3 cells) do not hold on fresh, smaller cities.

**Interpretation (representation-preservation category only).**
* **Public SAEs.** On truly held-out truth statements, they measurably degrade linearly decodable truth wherever it
  can be measured. This confirms the earlier VALIDATION-level preservation losses at TEST level.
* **What the fine-tuning does.**
  * Fine-tuning with GoT replay (ctrl) narrows the fresh-cities gap a lot vs base: −0.014 at L8 TopK, −0.026 at L8
    ReLU.
  * The truth-subspace term adds nothing beyond the reconstruction-only replay control.
  * It does beat a random subspace of the same rank at L8 TopK (8/8 seeds) and L11.
  * So the TSP-R advantage seen on cross-fitted cities does not generalise to a new city population. On fresh data,
    the measurable benefit comes from GoT-domain replay, not from the truth-subspace objective.

**Limitations.**
* The fresh family is a different population: cities of 200k–500k instead of > 500k, and capped per country. A
  population shift is part of what is tested.
* Readers are fitted on fresh DEV: 263 cities, the same for every SAE.
* The TI-1 cells have 3 seeds, so their CIs are wide (L11).
* TI-1 L8 ReLU fine-tunes have GoT-row L0 of about 250 vs 83 for base. L0 is matched on OWT only, as in TI-1.
* TI-2 and TI-4 checkpoints were lost and could not be evaluated.
* One template; one model each for P1.

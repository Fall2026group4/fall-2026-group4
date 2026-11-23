# XF: fresh, never-used cities truth TEST from geonames (2026-11-23). Status: IN PROGRESS (pre-registered, DEV gate passed; TEST not yet read)

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

## 4. TEST
Not yet read.

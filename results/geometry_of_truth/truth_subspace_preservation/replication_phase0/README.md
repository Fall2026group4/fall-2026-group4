# TSP-R Phase 0: fresh GoT families, locked DEV/TEST split, DEV-only eligibility (2026-11-14/15). Status: COMPLETED (VALIDATION, DEV only; no training)

**Evidence level:** VALIDATION (DEV only).
* No SAE was trained or fine-tuned.
* No TEST label, TEST activation or TEST score was computed or read.
  * TEST rows were used only for text-level checks: hashing, duplicates, entity overlap and counts.
  * Label checks on TEST were limited to "values ∈ {0, 1}, non-null".
* No TSP pilot fine-tune was evaluated on fresh data.
* Approved by the user on 2026-11-14 ("I approve Phase 0 only"). Phase 1 has not started.

**Script:** `scripts/geometry_of_truth/29_tsp_r_phase0_fresh_families.py`.
* `prepare` is text only. Its outputs were frozen in commit `a7e42ad` before any activation was computed.
* `eligibility` runs on DEV only.

**Frozen plan:** `preregistration.json`. It holds the split-file SHA-256s and the `prepare` script hash
`423785fb…`.

## 1. Sources, licences, revisions

* **Repository:** `github.com/saprmarks/geometry-of-truth`, pinned to commit
  `5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb` (2024-08-08, the latest).
* **Integrity:** every downloaded file matches the upstream git blob SHA-1 (manifest with SHA-256s is in
  `preregistration.json` → `source.files`).
* **Provenance of the project's existing files:** `cities`, `neg_cities`, `larger_than` and `smaller_than` are
  byte-identical to upstream at this commit.
* **Licence:**
  * The repository has no LICENSE file (GitHub API: `license: null`).
  * The original sources:
    * counterfact comes from ROME (Meng et al. 2022; MIT-licensed code);
    * common_claim from Casper et al. 2023 (repository without a licence);
    * companies from Azaria & Mitchell 2023.
  * The raw files are therefore kept local and gitignored (`data/got/fresh/raw_*/`), not redistributed. Only hashes,
    split assignments (row id, group, split, flags; **no labels, no text**) and aggregates are committed. Use is
    academic research only.

## 2. Integrity (`integrity.csv`, `dev_label_balance.csv`)

All files have no nulls, binary labels, and every statement ends with a period.

**Duplicates** (normalised text):
* `sp_en_trans`: 3
* `neg_sp_en_trans`: 3
* `companies`: 14
* `common_claim`: 3
* `counterfact`: 26

`companies` had 6 label-conflicting duplicate statements on DEV.

**Amendment v2, made before freezing and before any activation:** every repeated normalised statement is excluded,
on text alone, so no TEST label is read. The v1 outputs are kept in `prepare_v1_superseded/`. After exclusion, DEV
label balance is 0.49–0.52 in every family.

## 3. Leakage and independence checks (`leakage.csv`, `split_counts.csv`)

### Overlap with existing project text
Checked against cities, neg_cities, larger_than, smaller_than, RepE facts and RepE cross-wording:
* Exact matches (normalised): **common_claim has 3 rows identical to RepE `facts_clean`**.
* Near duplicates (char 4-gram tf-idf cosine ≥ 0.90): common_claim 13 (RepE facts), counterfact 2 (cities).
* All of these are excluded from both splits.

### Mentions of project cities
Whole-word mentions of the cities.csv names (the TSP training entities):
* counterfact: 2,205 rows (620 of them name one of the 113 exposed TEST cities);
* common_claim: 17;
* companies: 12.

Rows naming an exposed TEST city are excluded everywhere. The P2-primary set also drops every row naming any
cities.csv city.

### DEV/TEST disjointness
No split unit is shared between DEV and TEST. After exclusions, no entity is shared either.
* **sp_en / neg_sp_en:** split by Spanish word, shared across the two files.
* **companies:** split by company.
* **common_claim:** split by statement.
* **counterfact:** split by Wikidata relation component. Templates shared by several relations are merged, giving
  26 components; 7 are in DEV and 19 in TEST.
  * TEST rows whose subject occurs in DEV are excluded (446).
  * Related but distinct predicates fall on both sides, e.g. P1412 "languages spoken" in DEV and P103/P37 in TEST.
* **conj / disj:** split by city pair.

### Secondary families (cities_cities_conj / disj)
These share all entities with cities, so they are not independent. After the entity guard, only 65–70 TEST rows
remain, and only 33–45 DEV rows per cross-fitting fold have both cities in V (`conj_disj_fold_counts.csv`).

## 4. Locked splits (`data/got/fresh/splits/`)

| Family | DEV rows / units (P2-primary) | TEST rows / units (P2-primary) |
|---|---|---|
| sp_en_trans | 115 / 112 words | 233 / 229 |
| neg_sp_en_trans | 115 / 112 | 233 / 229 |
| companies_true_false | 341 / 86 companies | 821 / 204 |
| common_claim_true_false | 1,344 statements | 3,070 |
| counterfact_true_false | 8,754 / 4,625 subjects (7 relation comps.) | 20,532 / 10,001 (19 comps.) |
| cities_cities_conj / disj (secondary) | 331 / 341 | 65 / 70 |

### Cities cross-fitting folds (`cities_crossfit_folds.csv`)
* 635 groups (523 train + 112 val). The 113 exposed TEST cities are excluded.
* Fold sizes 212 / 212 / 211, mixing the pilot's A, B and V halves.
* Roles in fold f: V = f, B = f+1, A = f+2.

## 5. DEV eligibility (`eligibility.csv`; frozen rule: L8 grouped-CV logistic AUROC ≥ 0.70 on P2-primary DEV)

| Family | L8 AUROC [95 % CI] | L11 | Surface tf-idf | Permuted | Eligible |
|---|---|---|---|---|---|
| sp_en_trans | 0.513 [0.409, 0.619] | 0.421 | 0.509 | 0.449 | **no** (no signal in GPT-2 Small) |
| neg_sp_en_trans | 0.512 [0.409, 0.618] | 0.436 | 0.509 | 0.490 | **no** |
| companies_true_false | **0.791** [0.755, 0.825] | 0.690 | 0.445 | 0.487 | **yes** |
| common_claim_true_false | **0.708** [0.680, 0.734] | 0.689 | 0.649 | 0.483 | **yes** (marginal; CI includes 0.70) |
| counterfact_true_false | 0.625 [0.618, 0.632] (held-out relations) | 0.658 | 0.510 | 0.494 | **no** |
| cities_cities_conj (secondary) | 0.613 | 0.557 | 0.473 | 0.502 | n/a |
| cities_cities_disj (secondary) | 0.620 | 0.638 | 0.585 | 0.517 | n/a |

Notes:
* **Counterfact with entity-level CV** (same relations, new subjects) gives **0.786** [0.777, 0.795]. GPT-2's linear
  truth signal there does not transfer across relations. The frozen unit was the relation, so counterfact is
  ineligible. Re-defining the unit now would be a post-hoc deviation.
* **No family is surface-solvable** (surface AUROC < 0.70). common_claim's surface baseline (0.649) is close to its
  activation probe, so part of its signal is lexical.

## 6. Headroom: R2 base SAEs on DEV, inference only (`headroom.csv`)

| Family | Re-fit gap, seeds 42 / 1042 / 2042 | Frozen gap | NMSE | L0 |
|---|---|---|---|---|
| companies | −0.014 / +0.053 / +0.028 (mean +0.023) | +0.049 / +0.092 / +0.080 | 0.32 | 128 |
| common_claim | −0.001 / −0.003 / +0.007 (mean +0.001) | +0.005 / 0.000 / −0.001 | 0.21 | 128 |
| counterfact (ineligible) | −0.009 / +0.012 / +0.025 | +0.025 / +0.036 / +0.038 | 0.28 | 128 |
| sp_en / neg_sp_en (ineligible) | noise (base AUROC ≈ 0.51) | — | 1.36–1.66 | 128 |
| conj / disj (secondary) | +0.015 to +0.066 | +0.05 to +0.10 | 3.4–5.1 | 128 |

* **common_claim has no headroom.** The R2 base already preserves its truth information (re-fit and frozen gaps
  ≈ 0). TSP cannot improve preservation there; any gain would be enhancement.
* **companies has headroom**, but the base-seed spread of its re-fit gap is large (−0.014 to +0.053).

## 7. Feasibility for Phase 1 (`feasibility_assessment.json`)

The minimum detectable effects (MDE) below are for 80 % power, α = 0.025 two-sided (Holm worst case), seed sd 0.0035
taken from the pilot. They use a conservative statement-sampling proxy: the DEV bootstrap sd of the paired
difference between two R2 seeds, scaled to the TEST size.

### P2 (held-out families)
* **companies MDE ≈ 0.033** (≈ 0.016 if the proxy is twice too large), for 5 or 8 seeds alike.
* **common_claim MDE ≈ 0.014.** That exceeds its headroom (≈ 0.001–0.002).
* **Fixed-effect pooling:** MDE 0.013, but 84 % of the weight falls on common_claim, the family with nothing to
  recover.
* **The drafted random-effects P2 is not feasible** with k = 2 families (Hartung–Knapp df = 1).

### P1 (cities cross-fitting, 635 groups)
* At the pilot's observed re-fit effect (−0.020), power is ≈ 0.97–0.99 with 8 seeds and ≈ 0.86–0.9 with 5 seeds,
  under the conservative pilot rule.
* At the SESOI (−0.010), power is only ≈ 0.3–0.6.

### Measured costs
* DEV activations took < 10 s per family.
* Nested-CV logistic probes: common_claim ≈ 37 s, counterfact ≈ 170 s and companies ≈ 1 s per probe set.
* DEV activations total 74 MB (gitignored).

## 8. Assessment

* **Q1 (in-distribution reproducibility, P1) is justified as designed:** cross-fitted cities, 8 seeds, a
  label-permuted S control.
* **Q2 (held-out-family transfer) cannot be answered decisively** with public GoT data on GPT-2 Small L8:
  * Only 2 families are eligible.
  * One of them (common_claim) has no truth-information loss to recover.
  * The other (companies) can detect only effects ≥ ≈ 0.03, larger than the pilot's in-distribution effect (0.020).
* **Recommendation:** amend before Phase 1.
  * P1 becomes the sole confirmatory primary.
  * companies becomes a pre-registered secondary estimate with its CI.
  * common_claim becomes a no-headroom / enhancement check.
  * counterfact (held-out relations) is exploratory only, if included at all.
  * See `docs/RESEARCH_DECISIONS.md` D41. The decision is the user's.

## 9. Deviations and fixes

* **v1 → v2 duplicate rule:** made before freezing, text only.
* **Eligibility attempt 1 crashed before computing anything.** Pandas read the empty `excluded` field as NaN, so the
  DEV filter selected zero rows. The fix (`keep_default_na=False`) changes only the loader of the `eligibility`
  stage. The split files are unchanged and their hashes were asserted at run time.
* **Script hashes:** `prepare` 423785fb…; `eligibility` 8d4eb535… (recorded in `feasibility.json`).
* **Logs:** `../replication_phase0_logs/` (gitignored).

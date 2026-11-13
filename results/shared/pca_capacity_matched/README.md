# PCA-CM: capacity-matched PCA references (2026-11-13). Status: COMPLETED, VALIDATION. Conclusion change reported to the user; paper NOT yet revised

Evidence level: **validation finding (descriptive)**. No test row is read.

Script: [`scripts/shared/pca_capacity_matched.py`](../../../scripts/shared/pca_capacity_matched.py).
Frozen protocol and hashes: [`preregistration.json`](preregistration.json) (the script refuses to run if any hash differs).
Remediation item: S1 / D-PCA in [`docs/remediation/issue_register.md`](../../../docs/remediation/issue_register.md).

## 1. Before implementation

* **Research question:** is "SAEs never preserve task information better than PCA" a property of the SAEs, or of
  how the PCA reference was matched?
* **Why it's needed:** every PCA reference so far (M2, the ladder, M5) is matched on OWT reconstruction error
  (`src/evaluation/information_levels.py:pca_k_for_val_nmse`). In M5 (`m5_rows.csv`, `got_cities__pca_k`) this gives
  PCA k = 298–519 dense components for the trained SAEs (R2/R4: 404–519) but only k = 143–164 for the frozen random
  decoder, while every SAE uses about 128 active latents per token. The paper's abstract and conclusion state the
  error-matched result without the k range. That makes the random-decoder "tie" and the "never beats PCA" headline
  partly mechanical. This bears directly on the main research question, which asks whether truth information is
  preserved across layers when reconstruction improves.
* **Hypothesis (stated before the run):** at an equal per-token active budget (k = 128), PCA reconstructs and splices
  far worse than every trained SAE. Its truth gap may also be worse, so the error-matched conclusion would not
  transfer to capacity matching.
* **Design:**
  * GPT-2 Small, `blocks.{L}.hook_resid_pre`, BOS excluded. Reconstruction and behaviour at L ∈ {1, 8, 11}; GoT at
    L ∈ {8, 11}, because GoT is uninformative before about L4.
  * SAEs: the M5 source list. These are the old ReLU/TopK/JumpReLU-3072 SAEs (3 seeds); the R2 TopK, BatchTopK and
    ReLU-12288 and TopK-3072 SAEs (3 seeds); the R2 random-decoder TopK (1 seed); the R4 KL and MSE TopK fine-tunes
    (3 seeds, L8/L11); and res-jb (1).
  * PCA is fitted on the OWT TRAIN cache (`M2.owt_pca`, unchanged).
  * k sweep: 32, 64, 128, 256, 512.
  * Per-SAE references:
    * `capacity_k128`: PCA at k = 128, the OWT L0 target;
    * `task_l0_matched`: k = round(the SAE's GoT task L0);
    * `error_matched`: k at the SAE's OWT validation NMSE, which replicates the existing reference on validation.
  * Splits:
    * OWT validation (12,700 tokens per layer; 100 sequences for splicing);
    * GoT `cities` + `neg_cities`: readers fitted on TRAIN (2,092 rows) and scored on VAL (448 rows, 224 city groups);
    * TEST is never read.
  * Nothing is trained.
* **Metrics:**
  * Reconstruction: OWT NMSE (lower is better), EV and cosine (higher is better).
  * Behavioural splicing: ΔCE and KL(p_orig‖p_spliced) (lower is better).
  * Representation preservation: the GoT **refit-logistic AUROC gap**, the primary truth metric (lower is better),
    and the frozen-logistic gap (secondary; lower is better).
* **Comparisons:**

  | X vs Y | What it establishes |
  |---|---|
  | SAE vs PCA-128 (OWT NMSE, ΔCE, KL) | How much more an SAE does with the same per-token active budget than a fixed subspace |
  | SAE vs PCA-128 and PCA-task-L0 (GoT gaps) | Whether SAE truth loss exceeds PCA's at an equal active budget |
  | SAE vs PCA-error-matched (GoT gaps) | Replicates the existing error-matched comparison on validation |
  | Random decoder vs each reference | Whether the random-decoder "tie" survives capacity matching |

* **Rules:**
  * GoT: per family × layer × reference, from per-seed paired grouped-bootstrap 95 % CIs of (SAE gap − PCA gap), with
    1,000 resamples shared within a layer.
    * **SAE BETTER** if every seed's CI high < 0.
    * **SAE WORSE** if every seed's CI low > 0.
    * **EQUIVALENT (±0.02)** if every seed's CI lies within ±0.02.
    * Otherwise **INCONCLUSIVE**.
  * OWT: SAE BETTER or WORSE than PCA-128 only if so on every seed; otherwise MIXED.
* **Pre-registered interpretation:**
  * I1: if the error-matched comparisons are never SAE BETTER, the error-matched finding is replicated and stays, but
    it must be stated as error-matched, with the k range.
  * I2: if any trained family is SAE BETTER at `capacity_k128` or `task_l0_matched` on the primary metric, "SAEs never
    beat PCA" is false without qualification. **That is a conclusion change and is reported to the user before the
    paper is revised.**
  * I3: if the capacity-matched comparisons are INCONCLUSIVE or EQUIVALENT, the paper must say that the SAE-vs-PCA
    conclusion depends on the matching criterion.
* **Leakage prevention:**
  * GoT city groups are asserted disjoint between train and eval (`M5.got_cities`).
  * Probes are fitted on TRAIN rows of original or reconstructed activations, never on VAL.
  * PCA is fitted on OWT TRAIN.
  * The split files are hashed.
  * A smoke run on validation only (16 sequences; PCA from 5 chunks) checked the pipeline. Its numbers are not used.
* **Limitations (stated in advance):**
  * k = 128 equalises the per-token active budget, **not** parameters or representational capacity. A TopK SAE picks
    128 of 12,288 learned directions per token; PCA-128 is one fixed subspace.
  * GoT VAL n = 448 is small; seeds share one data stream.
  * There is no multiplicity correction across cells.

## 2. After execution

* **Run:** 3,724 s on one A10G.
  * 96 rows: 15 PCA references and 81 SAEs.
  * 174 paired GoT comparisons.
  * `checks.json`: `test_rows_used = 0`; the script hash equals the pre-registered hash.
  * PCA was fitted on 101,600 OWT TRAIN tokens.
* **Files:** `rows.csv`, `got_paired.csv`, `decisions.csv`, `checks.json`.

### 2.1 Reconstruction and behaviour against PCA-128 (OWT VALIDATION)

| Layer | PCA-128 NMSE ↓ | PCA-128 ΔCE ↓ (nats) | Trained SAEs: NMSE / ΔCE / KL vs PCA-128 |
|---|---|---|---|
| 1 | 0.296 | 0.399 | SAE BETTER in every family, on every seed |
| 8 | 0.462 | 0.989 | SAE BETTER in every family, on every seed |
| 11 | 0.434 | 1.059 | SAE BETTER in every family, on every seed |

* Trained SAEs win in 30 of 30 family × layer × metric cells.
* PCA needs k ≈ 512 to approach the R2 SAEs' NMSE: 0.094 at L8 and 0.105 at L11.
* Random decoder: better than PCA-128 on ΔCE and KL at every layer, worse on NMSE at L1 (0.324 vs 0.296).

### 2.2 GoT truth information (re-fit logistic gap ↓, primary; GoT VAL, 448 rows)

Family classifications, from the per-seed paired bootstrap CIs (pre-registered rule):

| Reference | PCA k | SAE BETTER | INCONCLUSIVE | SAE WORSE |
|---|---|---|---|---|
| `error_matched` (replicates the existing reference) | 298–519 trained; 143/164 random | 0 | 7 | **13** |
| `capacity_k128` | 128 | **3**: R2 ReLU L8, R2 ReLU L11, res-jb L8 | 17 | 0 |
| `task_l0_matched` | 54–128 | **8**: old ReLU L8/L11, old JumpReLU L11, R2 ReLU L8/L11, R2 BatchTopK L11, res-jb L8/L11 | 12 | 0 |

Counts are over the 20 trained family × layer cells: 10 families × L8/L11, random decoder excluded. For TopK, the task L0 is 128, so `task_l0_matched` equals
`capacity_k128`.

* **Random decoder:** INCONCLUSIVE against every reference at both layers.
* **Example (R2 ReLU, L11):**
  * SAE gap 0.076 ± 0.003;
  * PCA-128 gap 0.153 (paired difference −0.077; per-seed CIs within [−0.114, −0.043]);
  * error-matched PCA (k = 404) gap 0.038 (SAE WORSE, +0.038).
* **Example (R2 TopK, L8):**
  * SAE 0.055 ± 0.022;
  * PCA-128 0.058 (INCONCLUSIVE);
  * error-matched PCA (k ≈ 510) 0.012 (SAE WORSE, +0.044).

### 2.3 Pre-registered interpretation

* **I1 (error-matched): MET.** No trained SAE is ever better than error-matched PCA; 13/20 cells are worse. The
  existing finding replicates on validation, **as an error-matched statement**.
* **I2 (capacity-matched): TRIGGERED.**
  * At an equal per-token active budget (k = 128), trained SAEs are never worse than PCA on GoT truth information,
    and the ReLU-family SAEs are better in 3 cells.
  * At task-L0-matched k, 8/20 cells favour the SAE.
  * "SAEs never preserve task information better than PCA" is therefore **not true without qualification**.
  * This is a **conclusion change**. It was reported to the user before any revision of the paper's abstract,
    conclusion or §improvement (instruction: stop and notify).
* **I3:** most TopK-family cells are INCONCLUSIVE at k = 128. The SAE-vs-PCA verdict depends on the matching
  criterion.

### 2.4 Limitations

* k = 128 equalises the per-token active budget only, not parameters or representational capacity.
* GoT VAL n = 448.
* Paired differences have statement-level 95 % CI half-widths of about 0.03–0.05 AUROC
  (`results/shared/power_sensitivity/`).
* No multiplicity correction across 66 GoT family cells.
* Seeds share one data stream.

### 2.5 Status

The paper text (`00_abstract.tex`, `14_conclusion.tex`, `10b_improvement.tex`, `07_results_information.tex`,
`12_discussion.tex`) and the curated docs still state the error-matched claim unqualified. Revising them awaits the
user's review of this result.

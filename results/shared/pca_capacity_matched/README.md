# PCA-CM: capacity-matched PCA references (2026-11-13). Status: PRE-REGISTERED, RUNNING

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

*Pending.*

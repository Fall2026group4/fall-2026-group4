# TI-3: minimal-pair difference preservation (GPT-2 Small, R2 TopK L8) (PILOT PRE-REGISTRATION, 2026-11-21). Status: PRE-REGISTERED, EXPLORATORY (frozen before any training; approved by the user, Option B item 6; first to be dropped if the schedule slips)

**Evidence level:** **EXPLORATORY** (pilot). No confirmatory claim may be made from TI-3. A positive pilot would
need its own confirmatory pre-registration.

## 1. Why

* TSP needs a **fitted truth direction**. That makes the frozen-probe metric circular (audit §3.1) and probably
  limits transfer (TSP-R, TI-2).
* A probe-free alternative is to preserve the **difference** between a true and a false statement that differ in one
  slot, so that x̂_true − x̂_false ≈ x_true − x_false.
* GoT cities has natural minimal pairs. The same city appears in "The city of X is in [correct country]" (true) and
  "… [another country]" (false). No direction is estimated, so no evaluation probe can be its teacher.

**Hypothesis (exploratory):** MP-H1: the minimal-pair arm reduces the cities refit gap vs reconstruction-only
(mp − ctrl < 0), and does so more than a shuffled-pair control (mp − mp_shuf < 0).

## 2. Design

**Base:** R2 TopK-12288 L8; seeds 42 / 1042 / 2042 × the three cities cross-fit folds. Pairs come only from
**A-fold** groups.

**Arms** (same tokens, steps, learning rate and replay as TI-1 ctrl):

| Arm | Loss |
|---|---|
| ctrl | TI-1 ctrl |
| mp | ctrl + β · mean over A pairs ‖(x_a − x_b) − (x̂_a − x̂_b)‖² / mean ‖x_a − x_b‖² |
| mp_shuf | ctrl + the same term on **mismatched** pairs (true statement of city i with a false statement of city j ≠ i, same template). This preserves generic statement differences, not the true/false contrast |

* **β:** scale-matched so the pair term equals NMSE_GoT-A on the first batch, the same rule TSP uses for λ. It is
  fixed, with no search.
* **3 arms × 3 seeds × 3 folds = 27 fine-tunes;** TopK, so L0 = 128 exactly.

**Outcomes and analysis:**
* cities V refit-logistic gap (↓) with the TI-1 statistic;
* companies DEV refit gap (↓; exploratory transfer, TI-2 readers);
* OWT NMSE and ΔCE vs base (cost; the same +3 % / +5 % limits as TI-2);
* also reported: the A-pair difference error on held-out V pairs (↓), to check that the objective is learned.

**Pilot reading** (no p-value threshold is claimed as confirmation):
* "promising" if mp − ctrl CI high < 0 **and** mp − mp_shuf < 0 on ≥ 2/3 folds, within the cost limits;
* otherwise "not promising".

**Failure conditions:**
* the pair term does not decrease during training (implementation check);
* any L0 ≠ 128;
* divergence.

## 3. Cost and storage

* GPU: 27 × about 4.5 min ≈ 2 h; evaluation about 0.8 h.
* Engineering: about 1 h (pair builder plus loss term in `36_minimal_pair_preservation.py`; unit test that every
  pair shares city and template and differs only in country).
* Storage: 27 × 73 MB ≈ 2 GB on `/mnt/scratch/checkpoints/ti3/`. Keepers: the summary only. Being exploratory,
  checkpoints stay on scratch unless the pilot is "promising".

## 4. Freeze notes (2026-11-21, before any TI-3 gate, training or outcome)

* **Pairs (verified at freeze):** every (city, polarity) cell of cities + neg_cities has exactly one true and one
  false row, with the same template and a different country (1,270 cells). Pairs are built per fold from **A** groups
  only.
  * `mp_shuf`: the false partner is a derangement within polarity, so it is never the same city (fixed seed 20261126).
* **ctrl:** the 9 TSP-R ctrl checkpoints (seeds 42 / 1042 / 2042 × folds 0–2) are reused **only if** gate R0 passes:
  the TI-3 trainer with no pair term reproduces TSP-R `f0_ctrl_s42` (full run), max |param diff| ≤ 1e-6. Their
  SHA-256 must match `run_ledger.csv`.
  * **New runs:** mp and mp_shuf × 9 = **18** (§2's 27 counted a retrained ctrl).
  * A failed gate stops TI-3.
* **β:** NMSE_GoT-A / pair term on the first batch, fixed.
* **Outcomes:**
  * cities refit gap (TI-1 readers, fold-weighted, hierarchical);
  * held-out V pair-difference error;
  * companies DEV refit gap (TI-2 grouped 5-fold CV readers; exploratory);
  * OWT NMSE / ΔCE vs base: cost limits +3 % / +5 %, L0 = 128.
* **Pilot reading (unchanged):** "promising" iff mp − ctrl CI high < 0, **and** mp − mp_shuf < 0 on ≥ 2 / 3 folds,
  **and** the cost limits hold, **and** the pair term decreased in every mp run. Otherwise "not promising".
  EXPLORATORY in both cases.
* **Scheduling:** the orchestrator skips TI-3 if it cannot start by 14:30 UTC on 11-22.
* **Script:** `scripts/geometry_of_truth/36_minimal_pair_preservation.py`; tests in
  `src/tests/test_minimal_pair_preservation.py`. It reuses the TI-2 / TI-4 modules (included byte-identical).

## Amendment A-TF32 (2026-11-22 ~04:10 UTC; approved by the user; before any counted run)

**Cause of the reproduction failure (exact):**
* Gate R0 reproduces a TSP-R L8 TopK checkpoint. It failed with a max |param diff| of **0.168**: TI-2 and TI-4
  `f0_tsp_s42`; TI-3 would have hit the same with `f0_ctrl_s42`.
* TSP-R trained with **TF32 matrix multiplication enabled**:
  * `torch.backends.cuda.matmul.allow_tf32 = True` in `30_tsp_r_phase1_training.py:216`, and in `25:126`;
  * TSP-R evaluated with it **disabled** (`31:250`).
* The frozen TI-2 / TI-4 / TI-3 trainers ran in full FP32. TF32 rounding changes TopK selections, so over about 1,230
  steps the runs diverge.
* **Diagnostic** (scratch script; nothing written to the repo): `S25.run_one` with TF32 on and the frozen TSP-R
  subspace reproduces `f0_tsp_s42` with **max |param diff| 0.0**. The run-time recomputed subspace equals the frozen
  `subspaces.npz` (projector difference 0.0).
* Other gate parts passed:
  * TI-2: eligibility, leakage, and family_vectors (1e-15);
  * TI-4: R1 (KL term == R4) at 0.0, and the TSP-R checkpoint hashes.
* The failed gate records are preserved as `gate_attempt1_failed_R0_tf32.json`.

**Change:**
* training paths (gate R0/R1 and train) set `allow_tf32 = True`;
* the eval stage sets it `False`;
* this matches TSP-R exactly.

**Other numerical settings verified identical to the TSP-R controls:** FP32 weights, `S25.run_one` unchanged (same
`torch.manual_seed`, optimiser, clipping, token stream, steps), and no deterministic-algorithms flag.

**Unchanged:** every gate and its tolerance (R0 ≤ 1e-6 must pass as frozen), all criteria, arms, seeds and cost
limits.

**Schedule:** E1 → TI-2 → TI-4 → TI-3. TI-3 is withdrawn automatically if it cannot start by 14:30 UTC.

---

## Status (recorded 2026-11-23, recovery session): WITHDRAWN (never trained, no result)

* **Rule.** Withdrawn automatically under its own pre-registered schedule rule: "TI-3 is withdrawn automatically if
  it cannot start by 14:30 UTC".
* **What happened:**
  * The TF32 queue finished TI-4 at 15:38 UTC on 2026-11-22 and logged `15:38:26 TI3_WITHDRAWN (could not start by
    14:30 UTC)` (`logs_xm/queue_tf32.out`).
  * The gate was held for the TF32 amendment (`READY.hold.tf32`).
  * No TI-3 training, evaluation or analysis ran.
* **Artifacts kept:** `preregistration.json`, this README, the `WITHDRAWN` marker, `READY.hold.tf32`, the script
  `scripts/geometry_of_truth/36_minimal_pair_preservation.py` and its tests.
* **Decision (2026-11-23, user):** do not re-run. No claim may be made from TI-3.

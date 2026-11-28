# TI-2: multi-family truth-subspace-preserving fine-tune (TSP-MF), GPT-2 Small (PRE-REGISTRATION, 2026-11-21). Status: PRE-REGISTERED (frozen before any training; approved by the user, Option B item 4)

**Evidence level on completion:** VALIDATION (DEV rows only; every fresh-family TEST split stays locked: they were
read once in TSP-R, D42/D43, and are not reused).

## 1. Question and hypotheses

**Problem (D43, TI-1):**
* TSP reduces the cities refit-probe gap at every tested layer and architecture (TI-1), but its subspace is fitted on
  cities / neg_cities only.
* TSP-R found **no transfer** to the held-out companies family: −0.0036 [−0.0129, +0.0057].

**Question:** does a truth subspace spanned by **several independent truth families** preserve truth information in a
family that contributed nothing to the subspace or to the replay data?

| ID | Hypothesis | Contrast (refit-logistic gap, lower = better) | Role |
|---|---|---|---|
| **MF-H1** | TSP-MF preserves held-out-family truth better than reconstruction-only | companies DEV: tsp_mf − ctrl < 0 | **primary** |
| MF-H2 | the gain needs multiple families | companies DEV: tsp_mf − tsp_cities < 0 | secondary |
| MF-H3 | the gain is truth-specific | companies DEV: tsp_mf − rand < 0 | secondary (specificity) |
| MF-H4 | MF keeps the in-distribution cities gain | cities V (fold 0): tsp_mf − ctrl < 0 | secondary (no loss) |
| MF-H5 | no distortion where there is no headroom | common_claim held-out part: \|tsp_mf − ctrl\| < 0.01 | guard |

## 2. Design

**Base:** R2 TopK-12288, layer 8, the TSP-R cell with the 8-seed reference. Seeds 42, 1042, 2042, 3042, 4042, 5042,
6042, 7042: the eight TSP-R bases from `replication/base_qc.json`, so seeds are paired with TSP-R.

**Families and their roles** (every row is assigned before training; no row has two roles):

| Family | Rows used | Role in S and replay | Eligibility (original refit AUROC, L8) |
|---|---|---|---|
| cities + neg_cities | fold-0 A groups (`cities_crossfit_folds.csv`) | in S (all MF arms; the tsp_cities arm) | 0.78–0.83 (TI-1 gate) |
| larger_than + smaller_than | GoT TRAIN | in S (MF) | gate: must be ≥ 0.70 on GoT VAL, else dropped from S |
| common_claim | 60 % of DEV groups (seeded group split) | in S (MF) | 0.708 (D41) |
| counterfact | 60 % of DEV relation groups | in S (MF) | gate ≥ 0.70 on its held-out 40 %, else dropped from S |
| **companies** | **all 353 DEV rows** | **never in S or replay**: held-out evaluation only | 0.791 (D41) |
| common_claim (other 40 %) | held-out part | MF-H5 guard evaluation | — |

* **Eligibility for S** is decided by the frozen ≥ 0.70 rule on rows that are never used as primary evaluation. A
  family that fails is dropped from S before training, and this is recorded. No other selection is made.
* **Subspace construction** (`src/sae/truth_subspace.estimate_truth_subspace`, unchanged):
  * per family, the class-mean contrasts and bootstrap logistic raw weights on that family's S rows;
  * stacked across families, each family's vectors weighted equally (unit-normalised, per-family block scaled to
    equal Frobenius norm);
  * orthonormalised; **rank r = 8**, as in TSP / TI-1, so capacity is matched.

**Arms** (identical tokens, steps, learning rate and replay rows within a seed):

| Arm | Loss |
|---|---|
| base | pretrained, no fine-tune |
| ctrl | NMSE_OWT + NMSE on the union of all S-family rows (reconstruction-only replay) |
| tsp_cities | ctrl + rank-8 subspace term with S = cities / neg_cities only (the TI-1 recipe, same replay as ctrl) |
| **tsp_mf** | ctrl + rank-8 subspace term with S = all eligible families |
| rand | ctrl + rank-8 Haar-random subspace term |

* **Recipe:** `scripts/geometry_of_truth/25 run_one` unchanged (5M OWT tokens from shard 79 chunks ≥ 10,000, Adam
  2e-5, clip 1, μ = 1, λ = 1).
* **Gate R0:** with S = the TI-1 subspace and cities-only replay, the new code reproduces a TI-1 L8 TopK checkpoint
  bit for bit (30 steps).
* **Runs:** 4 trained arms × 8 seeds = **32 fine-tunes**.

**Readers (evaluation probes; never the training teacher; no shared rows):**
* **companies DEV:** grouped 5-fold cross-fitting by `group_id`. Readers are fitted on the ORIGINAL activations of 4
  folds and scored on the 5th, then averaged. Reconstruction readers use the same folds (refit: fitted on that
  arm's reconstructions of the 4 folds).
* **cities:** fold-0 B readers, V evaluation (TI-1 protocol).
* **common_claim guard:** grouped 5-fold within the held-out 40 %.
* Readers are logistic (primary), MLP-10 and MLP-256 refit, and frozen logistic / mass-mean (reported, not evidence:
  partly circular for TSP).

## 3. Statistics and criteria

* **Primary statistic:** the MF-H1 contrast per seed, with a hierarchical seed + group bootstrap
  (`tspr_stats.hierarchical`); label by `tspr_stats.h1_label` (SESOI 0.010).
* **MF-H1 SUPPORTED** if the label starts with "SUPPORTED" (CI high < 0) **and** p < 0.05 (single primary, no Holm).
* **Gatekeeping:** MF-H2 and MF-H3 are tested (Holm over the two) only if MF-H1 is SUPPORTED.
* **MF-H4** passes if the CI high < 0.
* **MF-H5** passes if the 90 % CI lies within ±0.010.
* **Behavioural cost limits (all must hold for a "usable" verdict):**
  * OWT VAL NMSE of tsp_mf vs base ≤ **+3 %**;
  * OWT spliced ΔCE vs base ≤ **+5 %**;
  * L0 = 128 exactly (TopK).
* **Verdicts:**
  * **TRANSFERS:** MF-H1 SUPPORTED, MF-H3 passes, MF-H4 holds and the cost limits hold.
  * **TRANSFERS (NOT SPECIFIC):** MF-H1 SUPPORTED but MF-H3 fails.
  * **NO TRANSFER DETECTED:** MF-H1 not supported. With 8 seeds the expected MDE is about 0.03 (TSP-R companies
    MDE 0.033), so a null **cannot exclude effects below about 0.03**.
  * **TRADE-OFF:** MF-H1 SUPPORTED but a cost limit fails, or MF-H4 or MF-H5 fails.

**Failure conditions (stop, report, no re-tuning):**
* R0 fails;
* companies' original refit AUROC < 0.70 under the new 5-fold readers;
* fewer than 2 families remain in S besides cities;
* any run diverges (excluded and reported, not re-run with new hyperparameters).

## 4. Leakage checks (`pretraining_checks.json`, before training)

* companies rows appear in no replay or S file (row-ID intersection = 0).
* Entities: there is no entity overlap between companies DEV and any S-family row, using the Phase-0
  `mentions_project_city` and `entity` fields; overlapping rows are dropped from S.
* No TEST row of any family is loaded.
* SHA-256 hashes of the script, the split files, the S-row lists and this pre-registration.

## 5. Cost and storage

* GPU:
  * 32 fine-tunes × about 4.5 min ≈ 2.4 h;
  * evaluation about 1.2 h;
  * analysis about 0.2 h.
* Engineering: about 2 h (a new script `34_tsp_multifamily.py` reusing `25 run_one`, `truth_subspace` and
  `tspr_stats`; unit tests for the row-role assignment and the leakage asserts).
* **Storage:**
  * checkpoints 32 × 73 MB ≈ 2.3 GB on `/mnt/scratch/checkpoints/ti2/`;
  * evaluation npz about 0.1 GB on EBS;
  * keepers: all 32 checkpoints (backup plan, `docs/2026-11-21_option_b_plan.md` §4).

## 6. Freeze notes (2026-11-21, before any TI-2 gate, training or outcome)

These fix implementation details of §2–§4. No criterion in §3 is changed.

* **Gate R0 (corrects §2):** the target is TSP-R `f0_tsp_s42` (TSP-R L8 TopK), not a TI-1 checkpoint (TI-1 has no
  L8 TopK cell). The full 5M-token run uses the TSP-R fold-0 cities subspace and cities-only replay. It passes if the
  max |param diff| ≤ **1e-6**: cross-process GPU reproduction is not guaranteed bit-exact, and the exact value is
  reported.
* **Second gate:** `family_vectors` reproduces the frozen `estimate_truth_subspace` subspace (|cos| deviation
  < 1e-8).
* **Row assembly (verified before freeze, data only, no AUROC computed):**

  | Family | Rows used |
  |---|---|
  | cities | fold-0 A / B / V = 844 / 848 / 848 |
  | numeric | larger_than + smaller_than TRAIN 2,772 (S), VAL 592 (eligibility); TEST rows dropped at load |
  | common_claim | DEV 1,344 → S 806 / guard 538 |
  | counterfact | DEV 8,754 → S 4,004 / held-out 4,750 |
  | **companies** | DEV `p2_primary` **341 rows, 86 entities** (§2 said 353, which counts excluded rows) |

* **Leakage filter:** S rows whose statement names a companies entity (whole word, ≥ 4 characters) are dropped:
  common_claim 3, counterfact 288. companies rows are never passed to the S / replay builder.
* **Eligibility (one rule, fixed now):**
  * refit logistic on original activations, fitted on the family's S rows;
  * AUROC on its held-out rows (numeric: TRAIN → VAL; common_claim / counterfact: S → held-out 40 %);
  * companies: grouped 5-fold CV;
  * ≥ 0.70 required.
* **Replay and subspace:**
  * replay = the union of all eligible S-family rows, capped at ≤ 1,000 rows per family (whole groups, fixed seed),
    and **identical for every arm**;
  * tsp_mf subspace: each family's vector set, the same estimator as TSP, at equal family weight (block Frobenius
    norm 1), SVD, rank 8;
  * tsp_cities and rand: the TSP-R fold-0 subspaces (seeds as TSP-R).
* **Seeds:** the 8 TSP-R bases (42, 1042, 2042 from R2; 3042–7042 from `r2_tspr_base_seeds`). Training is seed-major
  and ledger-resumable.
* **Evaluation:**
  * companies and the common_claim guard: grouped 5-fold CV readers (`31 readers_orig` / `readers_sae`, unchanged);
  * cities: fold-1 B readers → fold-0 V;
  * OWT cost: TI-1's OWT evaluation code.
  * Bootstrap: 2,000 group resamples (seed 20261124).
* **Checkpoints:** `/mnt/scratch/checkpoints/ti2/`. Keepers are copied to EBS (hash-verified) when EBS keeps ≥ 7 GB
  free; otherwise they stay registered in `/mnt/scratch/KEEPERS.tsv`.
* **Script:** `scripts/geometry_of_truth/34_tsp_multifamily.py`; tests in `src/tests/test_tsp_multifamily.py`.

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

## Results (analysis run 2026-11-23, recovery session). Status: COMPLETED (VALIDATION). Verdict: NO TRANSFER DETECTED (MDE about 0.03)

**Runs.**
* All 32 fine-tunes (4 arms × 8 seeds) finished on 2026-11-22 (queue log `logs_xm/queue_tf32.out`: train 06:52, eval
  08:30). The ledger (`ledger.csv`, 32 rows) records each checkpoint's SHA-256, final step 1,229 and L0 128.
* The eval stage finished (`eval/`, 41 npz + 32 OWT json + `owt_summary.csv`).
* **Checkpoints lost.** They were written to `/mnt/scratch/checkpoints/ti2/`, which is instance storage. The original
  EC2 instance became unavailable before they were copied to EBS, so they are not recoverable. The ledger hashes
  remain, and every number below comes from the saved eval outputs, not from the checkpoints.

**Analysis deviation (declared; no criterion changed).**
* The frozen `analyze` stage crashed on 2026-11-22 08:30 UTC (`logs_xm/ti2_tf32_analyze.log`) with
  `ValueError: Object arrays cannot be loaded when allow_pickle=False`. The cause: the pipeline's own eval stage had
  stored the group labels `comp_g` and `cc_g` in `eval/orig.npz` as numpy object (str) arrays.
* `scripts/geometry_of_truth/34b_tsp_multifamily_analyze_allow_pickle.py` imports the frozen script **unchanged**
  (`check_freeze()` passes) and runs `stage_analyze()`. It allows pickle **only** when loading `eval/orig.npz`, a file
  this pipeline wrote itself. Estimands, statistics, thresholds, seeds and inputs are unchanged.
* Log: `logs_xm/ti2_analyze_allow_pickle_2026-11-23.log`.
* SHA-256:
  * `34_tsp_multifamily.py` fb7247d5…
  * `34b_…allow_pickle.py` a06cdc84…
  * `eval/orig.npz` 486b2345…
  * `analysis/summary.json` f8c11395…
  * `analysis/contrasts.csv` 2738c9e5…

**Results** (refit-logistic gap, lower is better; hierarchical seed + group bootstrap over 8 seeds):

| Hypothesis | Contrast | Estimate | 95 % CI | Label | Criterion |
|---|---|---|---|---|---|
| **MF-H1 (primary)** | companies DEV: tsp_mf − ctrl | +0.0020 | [−0.0096, +0.0137] | INCONCLUSIVE (p 0.73) | **not met** |
| MF-H2 | companies: tsp_mf − tsp_cities | −0.0023 | [−0.0168, +0.0122] | INCONCLUSIVE | not tested (gatekeeping: MF-H1 failed) |
| MF-H3 | companies: tsp_mf − rand | −0.0032 | [−0.0158, +0.0095] | INCONCLUSIVE | not tested (gatekeeping) |
| MF-H4 | cities V: tsp_mf − ctrl | −0.0254 | [−0.0339, −0.0170] | SUPPORTED (meaningful) | **met** |
| MF-H5 (guard) | common_claim: tsp_mf − ctrl | +0.0015 | 90 % CI [−0.0031, +0.0061] | NO MEANINGFUL EFFECT | **met** (within ±0.010) |
| cost | OWT NMSE tsp_mf vs base | +2.71 % | — | — | met (≤ +3 %) |
| cost | OWT spliced ΔCE tsp_mf vs base | +2.78 % | — | — | met (≤ +5 %) |
| cost | L0 | 128 in every run | — | — | met |

* **Verdict (pre-registered rule): NO TRANSFER DETECTED.**
  * A multi-family truth subspace (cities + numeric + common_claim) does not measurably preserve truth information in
    the held-out companies family.
  * It keeps the in-distribution cities gain (MF-H4) at no reconstruction cost beyond the limits.
  * With 8 seeds, effects smaller than about 0.03 cannot be excluded.
* **Interpretation:**
  * This replicates TSP-R's null transfer (−0.0036 [−0.0129, +0.0057]) with a richer subspace.
  * The TSP benefit is family-specific, not a general truth-preservation mechanism.
* **Exploratory, not pre-registered:** tsp_cities − ctrl on cities V is −0.032, which is larger than tsp_mf − ctrl
  (−0.025). Adding families slightly dilutes the in-distribution gain (tsp_mf − tsp_cities +0.0066, CI
  [−0.0017, +0.0149]).
* **Limitations:**
  * VALIDATION evidence (DEV rows only).
  * One layer (L8) and one architecture (R2 TopK).
  * counterfact was ineligible at the gate (0.684), so S has three families.
  * Checkpoints lost; no further evaluation of these SAEs is possible without retraining (about 2.4 GB, ≈ 3 h GPU).

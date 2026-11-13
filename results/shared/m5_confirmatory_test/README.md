# M5: confirmatory TEST pass of the GPT-2 improvement phase. Status: COMPLETED (2026-11-10). Evidence: CONFIRMED TEST RESULT

Evidence level, once run: **CONFIRMED TEST RESULT** for the listed comparisons only. It runs once.

## 1. Before the run

### 1.1 Purpose

* Every result of the GPT-2 improvement phase so far is VALIDATION:
  * R2 data-scaled SAEs;
  * R4 KL fine-tune;
  * the old base SAEs;
  * the res-jb public reference.
* Development used validation rows to design and check methods: harness reproduction, DAS initialisation, R4 smoke
  test. Some decisions were taken after seeing validation numbers: abandoning the modern JumpReLU, and the R4/R5
  design.
* **M5 asks one question:** do the pre-registered comparisons hold on held-out rows that no decision has seen?
* **Approval:** the user's 2026-11-10 execution approval: "You have approval to conduct the final M5 evaluation once
  the selected methods and protocol are frozen. Do not repeatedly optimize against final test results."

### 1.2 What is frozen

* **Script:** `scripts/shared/m5_confirmatory_test.py`, together with the harness and comparison scripts it imports.
  All hashes are in `preregistration.json`, and the script refuses to run if any hash differs.
* **SAE list (no selection).** Every SAE of the listed families is evaluated, whatever its validation result, at
  layers 1, 8 and 11 (R4 at 8 and 11 only):
  * old base ReLU/TopK/JumpReLU-3072, 3 seeds each;
  * R2 TopK, BatchTopK and ReLU-12288 and TopK-3072, 3 seeds each;
  * R2 frozen-random-decoder TopK, 1 seed;
  * R4 `kl` and `mse` arms, 3 seeds each;
  * res-jb (descriptive).
* **Data:**
  * OWT TEST: 100 sequences, 12,700 non-BOS tokens; disjoint from training and validation by the original split.
  * GoT cities + neg_cities TEST (layers 8 and 11), with readers fitted on TRAIN rows of original activations.
    VALIDATION rows are not used except where noted below.
* **Metrics:**

  | Level | Metric | Direction |
  |---|---|---|
  | Reconstruction | EV | ↑ |
  | Reconstruction | NMSE | ↓ |
  | Reconstruction | cosine | ↑ |
  | Reconstruction | L0 | reported |
  | Reconstruction | dead fraction | reported |
  | Behaviour | spliced ΔCE | ↓ |
  | Behaviour | spliced KL | ↓ |
  | Behaviour | SAE/noise ΔCE ratio | ↓ |
  | Information (GoT) | frozen logistic gap | ↓ |
  | Information (GoT) | re-fit logistic gap | ↓ |
  | Information (GoT) | SAE − PCA re-fit gap | ↓ |

  For the SAE − PCA gap, PCA is matched at the SAE's OWT **validation** NMSE, the M2 rule. Validation is used here
  only to set k.
* **Comparisons and rules.** These are the R2 and R4 pre-registered rules, applied once:
  * **C1, C2, C3, A1, A2** (`r2_comparisons.py` COMPARISONS; R2 rule):
    * IMPROVEMENT if every new seed beats every reference seed AND the Welch 95 % CI excludes 0;
    * REGRESSION for the mirror image;
    * NO CLEAR CHANGE otherwise.
    * SAEs with OWT TEST L0 outside 128 ± 10 % are excluded.
  * **K1** (R4 `kl` vs `mse`) and **K4** (`kl` vs R2 base): IMP-L rule, paired by seed.
    * Better on all 3 seeds and |mean d| > 2 sd(d).
    * Metrics: ΔCE, KL, OWT NMSE (the guard), GoT re-fit gap.
* **Not in M5:**
  * RepE-LM persona, RLM-RP and M4/R5. RepE-LM has no test split: its test statements were never turned into prompts.
    These results stay VALIDATION.
  * The R2 scaling checkpoints (10M/25M/50M), which are descriptive.

### 1.3 Smoke check (VALIDATION only; written before the freeze)

* `--smoke DIR` runs the identical pipeline on validation rows for 4 SAEs. It must reproduce the R2 ladder values for
  the same SAEs.
* Result: see 1.4.

### 1.4 Smoke result (2026-11-10, VALIDATION rows; scratchpad, not results)

* **Run 1:** old TopK 42, R2 TopK 42 and res-jb at L1/L8/L11, plus R4 `kl` 42 at L8/L11.
  * Reproduces the R2/old ladder values for the same SAEs:
    * OWT EV, NMSE, L0, ΔCE, KL and the noise ratio: maximum absolute difference **0.0**;
    * GoT cities frozen, re-fit and SAE − PCA gaps: maximum difference **≤ 1.0e-4**.
  * Bug found: a seed-pairing assertion fails when an arm is absent. Fixed by merging on seed; rows are now written
    before the comparisons run.
* **Run 2** (adds R4 `mse` 42): the full pipeline ran end to end in 1,352 s.
  * 0 test rows; 448 GoT validation rows; 12,700 OWT tokens per layer.
  * With one seed per family, the comparisons are correctly NOT EVALUATED.
* **Run order:** M5 runs after the R2/R4 validation ladders and their analyses are complete. It shares no CPU memory
  with other ladder jobs (an OOM incident on 2026-11-10 killed concurrent ladder runs; see RESEARCH_PROGRESS §34).

## 2. Results (TEST, single run, 2026-11-10)

### 2.1 Run and checks

* **Run:** once, alone on the host. 81 SAEs in 2,612 s. The script hash matched `preregistration.json`, which was
  checked at start.
* **Test rows read:**
  * OWT test: 100 sequences, 12,700 tokens per layer;
  * GoT cities + neg_cities test: 452 rows.
  * Total 552 test rows. They were used for nothing else.
* **Files:** `m5_rows.csv` (every SAE and metric), `comparisons.csv`, `checks.json`.
* **L0 on OWT test:** every compared SAE is within 128 ± 10 % (127.6–133.2), so none was excluded. res-jb runs at
  15 / 66 / 60 (descriptive only).

### 2.2 Confirmed results (class counts per comparison over the 24 OWT and GoT cells)

| ID | Comparison | IMPROVEMENT | NO CLEAR CHANGE | REGRESSION |
|---|---|---|---|---|
| C1 | old TopK-3072 → R2 TopK-12288 | 17 | 7 | 0 |
| C2 | old ReLU-3072 → R2 ReLU-12288 | 17 | 7 | 0 |
| C3 | old JumpReLU-3072 → R2 BatchTopK-12288 | 17 | 7 | 0 |
| A1 | data and recipe at width 3072 | 17 | 7 | 0 |
| A2 | width 3072 → 12288 (R2) | 16 | 6 | 2 (GoT L8) |
| K1 | R4 KL vs MSE control | 4 | 2 | 2 (OWT NMSE) |
| K4 | R4 KL vs R2 base | 4 | 2 | 2 (OWT NMSE) |

Headline values (TEST, means over 3 seeds; improvement = positive is better):

| Layer | Metric | Old TopK | R2 TopK | Improvement | Old JumpReLU | R2 BatchTopK | Improvement |
|---|---|---|---|---|---|---|---|
| L1 | OWT NMSE | 0.0787 | 0.0248 | 68.6 % | 0.0675 | 0.0178 | 73.7 % |
| L1 | spliced ΔCE (nats) | 0.0221 | 0.0056 | 74.9 % | 0.0088 | 0.0019 | 78.8 % |
| L8 | OWT NMSE | 0.1598 | 0.0941 | 41.1 % | 0.1560 | 0.0897 | 42.5 % |
| L8 | spliced ΔCE | 0.1330 | 0.0620 | 53.4 % | 0.1212 | 0.0542 | 55.3 % |
| L8 | GoT re-fit gap | 0.061 | 0.056 | no clear change | 0.058 | 0.065 | no clear change |
| L11 | OWT NMSE | 0.1848 | 0.1112 | 39.8 % | 0.1851 | 0.1073 | 42.0 % |
| L11 | spliced ΔCE | 0.2041 | 0.0962 | 52.9 % | 0.2155 | 0.0927 | 57.0 % |
| L11 | GoT re-fit gap | 0.101 | 0.078 | no clear change | 0.107 | 0.087 | no clear change |

R4 KL fine-tune on TEST:

| Layer | Spliced ΔCE, MSE control → KL | Improvement | Class | NMSE, MSE control → KL | Class |
|---|---|---|---|---|---|
| L8 | 0.0623 → 0.0514 | 17.4 % | IMPROVEMENT | 0.0938 → 0.0958 (+2.1 %) | REGRESSION |
| L11 | 0.0964 → 0.0807 | 16.3 % | IMPROVEMENT | 0.1109 → 0.1133 (+2.2 %) | REGRESSION |

* The NMSE regression is the expected cost of the KL term. It equals the validation EV cost of −0.002, which passed
  the pre-registered guard.
* GoT re-fit gap: no clear change at either layer.

### 2.3 Inference (confirmed)

1. **Every validation conclusion on these metrics replicates on held-out data.**
   * R2 reduces reconstruction error by 31–74 % and splice damage by 35–79 % at layers 1, 8 and 11, for all
     architectures (all seeds; Welch CI excludes 0).
   * About half of the gain comes from data and recipe alone (A1) and the rest from width (A2).
2. **GoT truth information shows no confirmed change.**
   * The SAE − PCA gap never improves on test. The two A2 regressions at L8 (wider is worse on GoT) match the
     validation picture, where at L8 a random decoder already has a small GoT gap.
3. **The R4 KL fine-tune's behavioural gain is confirmed:** −16 to −17 % ΔCE against the MSE control at a 2 % NMSE
   cost. It has no confirmed information effect.
4. **Not covered by M5:** persona, representation (RLM-RP) and causal (M4/R5) results stay VALIDATION, because
   RepE-LM has no test split.

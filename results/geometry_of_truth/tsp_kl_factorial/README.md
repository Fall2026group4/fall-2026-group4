# TI-4: TSP × KL factorial at fixed L0 (GPT-2 Small, R2 TopK L8) (PRE-REGISTRATION, 2026-11-21). Status: PRE-REGISTERED (frozen before any training; approved by the user, Option B item 5)

**Evidence level on completion:** VALIDATION (cities train / val rows cross-fitted; OWT VAL; no TEST).

## 1. Question and hypotheses

The research question asks for SAEs that improve faithfulness **while preserving truth**. Two fine-tunes each fix one
side:
* **R4** (KL + MSE at fixed L0) cuts spliced ΔCE (−16 % on TEST) but costs causal preservation (D29, D33);
* **TSP** (TI-1) cuts the cities refit gap at no extra OWT cost.

Do they combine?

| ID | Hypothesis | Contrast | Role |
|---|---|---|---|
| **F-H1** | the TSP truth gain survives adding KL | cities refit gap: tsp_kl − kl < 0 | **primary (truth)** |
| **F-H2** | the KL behaviour gain survives adding TSP | OWT VAL spliced ΔCE: tsp_kl − tsp < 0 | **primary (behaviour)** |
| F-H3 | the two gains are additive | interaction I = (tsp_kl − kl) − (tsp − ctrl) on the cities gap; additive if the 90 % CI lies within ±0.010 | secondary |
| F-H4 | the combined SAE is better than base on both | tsp_kl − base < 0 on the cities gap **and** on ΔCE | secondary |

## 2. Design (2 × 2 at fixed k = 128)

| Arm | TSP term | KL term | Loss |
|---|---|---|---|
| ctrl | – | – | NMSE_OWT + NMSE_GoT-A (TI-1 ctrl) |
| tsp | ✓ | – | ctrl + rank-8 truth-subspace term (TI-1 tsp) |
| kl | – | ✓ | ctrl + α·KL(p_orig ‖ p_spliced) (R4 kl term, α scale-matched on the first batch as in R4) |
| tsp_kl | ✓ | ✓ | ctrl + both terms (same α, same λ) |

* **Base:** R2 TopK-12288 L8, seeds 42 / 1042 / 2042 × the three cities cross-fit folds (the TSP-R design).
* **4 arms × 3 seeds × 3 folds = 36 SAEs. Only 18 are new fine-tunes (kl, tsp_kl).** ctrl and tsp are the existing
  TSP-R checkpoints for these (seed, fold) cells, reused after SHA-256 verification against `run_ledger.csv`.
* **Identical within a (seed, fold):** OWT tokens (5M, shard 79 chunks ≥ 10,000), GoT-A rows, steps, learning rate
  (2e-5), clip and initialisation. TopK keeps L0 = 128 exactly, so **sparsity is matched by construction**.
* **Gate R0:** the TI-4 training code with the KL term switched off must reproduce TSP-R `f0_ctrl_s42` and `f0_tsp_s42`
  bit for bit (full run). Only then are the TSP-R ctrl / tsp checkpoints reused. If R0 fails, all 36 are trained
  fresh (+1.4 h), and this is recorded.
* **Gate R1:** the KL term reproduces R4's α and its first-100-step loss trajectory on a smoke run (within 1e-5).

## 3. Outcomes (direction)

* **Truth:** cities refit-logistic gap ↓ (primary), refit MLP gaps ↓, frozen (reported only).
* **Behaviour:** OWT VAL spliced ΔCE ↓ (primary), KL ↓, top-1 agreement ↑.
* **Reconstruction:** OWT VAL NMSE (reported; cost limit).
* **Causal:** M4 T_rec / T_full at the L8 final site ↑ (secondary; R4 showed a causal cost).
* **Honesty guard:** RepE-LM persona refit gap at L8 (reported; must not get worse than ctrl by > 0.02).

## 4. Statistics and criteria

* **F-H1:** hierarchical seed + city-cluster bootstrap (`tspr_stats`); SUPPORTED if the label starts with
  "SUPPORTED" and p < 0.05.
* **F-H2:** the IMP-L paired rule over the 9 (seed, fold) pairs (better in every pair and |mean d| > 2 sd), and a
  paired bootstrap CI.
* **Holm** over F-H1 and F-H2 (two co-primaries).
* **F-H3:** two-way bootstrap of the interaction; ADDITIVE / ANTAGONISTIC (CI wholly > 0.010) / SYNERGISTIC (CI wholly
  < −0.010) / INCONCLUSIVE.
* **Behavioural cost limits:**
  * OWT NMSE of tsp_kl vs base ≤ +5 % (R4's guard was EV ≥ mse − 0.02);
  * L0 = 128;
  * persona guard as above.
* **Verdicts:**
  * **COMBINES:** F-H1 and F-H2 both SUPPORTED after Holm, and the cost limits hold.
  * **TRUTH-ONLY** or **BEHAVIOUR-ONLY:** one co-primary supported.
  * **CONFLICT:** neither, or ANTAGONISTIC.
  * **Power note:** 9 pairs; TI-1's L8 TopK-like effect (−0.02) is detectable, smaller interactions are not.

**Failure conditions:**
* R0 or R1 fails;
* any arm's L0 ≠ 128;
* divergence (excluded and reported).

There is no hyperparameter search: α and λ are fixed by the R4 and TI-1 rules.

## 5. Cost and storage

* GPU:
  * the KL arms need the GPT-2 forward pass with the splice, about 8–9 min per run; the non-KL arms take about 4.5 min;
  * 18 KL-arm runs × 8.5 min ≈ **2.6 h**, plus the R0 gate (2 × 4.5 min);
  * evaluation about 1.3 h, including M4 for 36 SAEs at about 1 min each;
  * analysis about 0.2 h.
* Engineering: about 1.5 h (script `35_tsp_kl_factorial.py` composing the `25 run_one` subspace term with the R4
  `18_kl_finetune_fixed_l0` KL term; unit tests for loss composition and the R0 / R1 gates).
* **Storage:**
  * 18 new × 73 MB ≈ 1.3 GB on `/mnt/scratch/checkpoints/ti4/`;
  * keepers: the 18 new ones (the reused TSP-R checkpoints are already on EBS).

## 6. Freeze notes (2026-11-21, before any TI-4 gate, training or outcome)

These fix implementation details of §2–§5. No criterion is changed.

* **Trainer:** `run_tsp_kl` = `25 run_one` plus R4's KL term (`18 run_one`).
  * The α rule is R4's: α = NMSE_OWT / KL on the first batch.
  * With KL off, it takes S25's computation line for line (`stop_at_layer` forward, the same loss order).
* **Gate R0:** KL off, full 5M-token run, TSP-R fold-0 subspace, reproduces TSP-R `f0_tsp_s42` with max |param
  diff| ≤ 1e-6. Only then are the 18 TSP-R ctrl / tsp checkpoints reused, after SHA-256 verification against
  `run_ledger.csv`.
* **Gate R1:** pure-R4 mode (no GoT replay, no subspace, no torch seed, as R4) for 20 steps equals R4's own `run_one`
  smoke: params ≤ 1e-6, α relative difference ≤ 1e-5.
* **If either gate fails, TI-4 stops.** Unlike §2's earlier wording, there is no automatic fall-back to retraining
  all 36 SAEs: a failed gate means a real implementation difference, which must be reported first.
* **F-H2 statistic:**
  * per (seed, fold) pair, d = ΔCE(tsp_kl) − ΔCE(tsp) on the 100 OWT VAL sequences (TI-1 splice code);
  * the IMP-L rule over the 9 pairs;
  * a paired bootstrap over sequences (2,000 shared resamples) for the CI and two-sided p.
* **Holm** over (F-H1, F-H2).
* **Interaction (F-H3):** I = (tsp_kl − kl) − (tsp − ctrl) on the cities refit gap, with the hierarchical CI:
  * ADDITIVE: 90 % CI within ±0.010;
  * ANTAGONISTIC: 95 % CI low > +0.010;
  * SYNERGISTIC: 95 % CI high < −0.010;
  * otherwise INCONCLUSIVE.
  * A COMBINES verdict with an ANTAGONISTIC interaction is reported as "COMBINES (ANTAGONISTIC interaction)".
* **F-H4:** cities gap tsp_kl − base CI high < 0, **and** OWT ΔCE lower than the run's base on every (seed, fold).
* **Secondary outcomes:**
  * persona refit gap at L8: M2 `evaluate_source` on `repe_persona`, seed index 700 + j;
  * M4 `preservation_rec` at the L8 final site (E1's `M4Ctx`, the same code as M4).
* **Cost limits:**
  * tsp_kl OWT NMSE vs base ≤ +5 % (mean over runs);
  * L0 = 128 for every run;
  * persona refit gap (tsp_kl − ctrl, mean) ≤ +0.02.
* **Checkpoints:** the 18 new ones go to `/mnt/scratch/checkpoints/ti4/` and are kept (backup plan). The reused TSP-R
  checkpoints are on EBS.
* **Script:** `scripts/geometry_of_truth/35_tsp_kl_factorial.py`; tests in `src/tests/test_tsp_kl_factorial.py`.
  It reuses `scripts/shared/e1_task_l0_matched_topk.py` (frozen in E1; included byte-identical).

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

## Results (reviewed 2026-11-23, recovery session). Status: COMPLETED (VALIDATION). Verdict: COMBINES

**Runs.** These ran on 2026-11-22 under amendment A-TF32. The queue log is `logs_xm/queue_tf32.out`:

| Stage | Time (UTC) | Detail |
|---|---|---|
| gate | 08:35 | R0 max \|param diff\| 0.0; R1 0.0 (α diff 0.0); TSP-R checkpoint hashes pass |
| train | 13:03 | 18 new fine-tunes: kl and tsp_kl × 3 seeds × 3 folds; `ledger.csv` holds each SHA-256, 1,229 steps, L0 128 |
| eval | 15:37 | 128 files in `eval/` |
| analysis | 15:38 | `analysis/summary.json` sha256 5add3632…; `analysis/cities_contrasts.csv` sha256 273b0f96… |

**Checkpoints lost.**
* The 18 new checkpoints were left on instance storage (`/mnt/scratch/checkpoints/ti4/`). They were logged as
  `KEEPERS_DEFERRED` because EBS had only 8 GB free, and they were lost when the instance became unavailable.
* Their SHA-256 values are in `ledger.csv`.
* The reused TSP-R ctrl/tsp checkpoints are on EBS and in the registry.

**Review against the frozen rules (§4, §6).** Every rule was re-applied to `analysis/summary.json`, and every verdict
reproduces. No number was recomputed or changed.

| Criterion | Value | Rule | Met |
|---|---|---|---|
| **F-H1** (truth gain survives KL): cities refit gap, tsp_kl − kl | −0.0259 [−0.0362, −0.0156], "SUPPORTED (meaningful)" | label SUPPORTED and p < 0.05; Holm over F-H1/F-H2 | **yes** (Holm p 6.5e-4) |
| **F-H2** (KL gain survives TSP): OWT ΔCE, tsp_kl − tsp | −0.0120 ± 0.0006 (9 pairs, all negative), CI [−0.0150, −0.0093], p 0.0005 | IMP-L rule (every pair better, \|mean\| > 2 sd) plus a paired bootstrap | **yes** (Holm p 6.5e-4) |
| F-H3 interaction | −0.0038, 95 % CI [−0.0191, +0.0115], 90 % CI [−0.0158, +0.0082] | ADDITIVE needs the 90 % CI within ±0.010 | INCONCLUSIVE |
| F-H4: tsp_kl better than base on both outcomes | cities −0.0487 (CI high −0.0154); ΔCE lower than base in every (seed, fold) | both | **yes** |
| Cost: OWT NMSE vs base | +2.77 % | ≤ +5 % | yes |
| Cost: L0 | 128 in every run | = 128 | yes |
| Cost: persona refit gap, tsp_kl − ctrl | −0.0055 | ≤ +0.02 | yes |
| Secondary: M4 preservation T_rec/T_full | base 0.726, ctrl 0.708, kl 0.708, tsp 0.722, tsp_kl 0.714 | reported | — |

* **Verdict (pre-registered): COMBINES.**
  * At fixed L0 128, adding the KL term keeps the TSP truth gain.
  * Adding TSP keeps the KL behavioural gain.
  * Both co-primaries survive Holm, and every cost limit holds.
  * Whether the two gains are exactly additive is inconclusive: the interaction CI is too wide.
* **Limitations:**
  * VALIDATION evidence only: cities train/val rows cross-fitted, OWT VAL, no TEST.
  * One layer (L8), one architecture (R2 TopK), 3 seeds × 3 folds.
  * The M4 causal preservation of tsp_kl (0.714) is below base (0.726), the R4 causal cost noted in D29/D33. M4 had no
    pre-registered threshold here.
  * Checkpoints lost; no further evaluation is possible without retraining (about 1.3 GB, about 2.6 h GPU).

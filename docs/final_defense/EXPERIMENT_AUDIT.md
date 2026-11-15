# Experiment audit: reproducibility, M5 verification, missing results, limitations (2026-11-10)

This is an independent audit of branch `sae-research` at `af52b0b`, with a clean working tree at the start.
* No experiment was rerun and no checkpoint was loaded.
* No result file, script or historical README was modified.
* Re-checks used `scripts/shared/m5_independent_verification.py`, which is read-only, plus direct reads of the
  committed CSV/JSON files.
* The test suite passes: 35 tests (`python -m pytest -q src/tests`).

---

## 1. M5 final held-out test: verification checklist

**Verdict: M5 COMPLETE.**
* The run is verified complete and internally consistent.
* Its conclusions are valid for the listed OWT reconstruction/behaviour comparisons and the GoT cities comparisons.
* It has two disclosure gaps (1.2, items 3 and 4) and one minor documentation error (1.2, item 1).

### 1.1 Checks that passed

| Check | Evidence | Result |
|---|---|---|
| The test split was held out from model selection | R2 and R4 were designed and compared on VALIDATION. R2 data is OWT shard 78, excluding 98 documents that match our 1000-document sample. R4 data is shard 79. M5 was run after the R2/R4 analyses (files 17:28–17:29 UTC; M5 run of 2,612 s, finishing 18:11). | PASS |
| The protocol was frozen before execution | `preregistration.json` (15:24 UTC) hashes 11 code and config files. The script refuses to run on any hash mismatch (`m5_confirmatory_test.py` lines 145–147), and `checks.json` records the run-time script hash. **All 11 hashes match the current files.** | PASS |
| Split hashes and manifests | GoT and OWT splits are fixed by the original split files. The M5 GoT loader asserts that no city group crosses train/eval (`got_cities`, line 90). The R2 pre-registration hashes the training shards. | PASS. The OWT/GoT split files are **not** hashed in the M5 pre-registration itself; they are covered by the loader's assertion and the cached split directories. |
| Checkpoints and seeds identifiable | `m5_rows.csv` stores the SHA-256 of every checkpoint: 81 rows, 81 distinct hashes, seeds 42/1042/2042. | PASS |
| No leakage between train, validation and test | OWT: document-level split (`01_prepare_dataset.py`, no shared document IDs); R2/R4 shards exclude our sample by exact and normalised-prefix hash. GoT: grouped by city (asserted). | PASS |
| All expected rows exist | L1: 23 = 9 old + 13 R2 + 1 res-jb. L8 and L11: 29 = the same + 6 R4. | PASS |
| No duplicate or silently missing rows | 0 duplicate (layer, family, name, seed) keys. The NaN GoT columns are exactly the 23 L1 rows (GoT is evaluated only at L8/L11 by design). | PASS |
| Metric computations are consistent | All 120 R2-rule rows and all 16 K1/K4 rows were re-derived from their stored per-seed values with an independent implementation: **0 mismatches**. Old-SAE OWT test EV equals the 2026-11-07 108-run table (for example, L8 TopK 0.8402 vs 0.840; L11 ReLU 0.7568 vs 0.757). | PASS |
| The final job exited successfully | `logs_m5_test.log` ends with the full comparison table, with no traceback. `checks.json` gives `n_saes` 81, `test_rows_used` 552 and `smoke: false`. | PASS |
| Output files complete and readable | `m5_rows.csv` (81 × 35), `comparisons.csv` (136 rows), `checks.json`, `README.md`. | PASS |
| Run once | The script refuses to overwrite its outputs (lines 148–150). Only one set of outputs exists, and the commit history shows one M5 commit (`af52b0b`). | PASS (it cannot be proven that no earlier test run was deleted; nothing suggests one was) |

### 1.2 Findings and gaps

1. **Minor documentation error.**
   * The M5 README gives the OWT test L0 range as "127.6–133.2". The true maximum is **133.39** (old ReLU, L11,
     seed 2042).
   * It is still inside 128 ± 10 % (115.2–140.8), so no exclusion changes.
   * The README was not edited (a historical artifact); this file records the erratum.
2. **The pre-run README cannot be reproduced.**
   * `preregistration.json` hashes the M5 README as it was before the run (`9cf13943…`).
   * The results were appended to that same file, so the frozen text can no longer be checked against the hash.
   * Attempts to rebuild it, by removing §2 and trying title variants, did not reproduce the hash.
   * The design text is therefore **NOT VERIFIED byte-for-byte**. The code hashes, which define what was computed,
     are verified.
   * **Recommendation:** in future, freeze the protocol in a separate file, or commit it before the run.
3. **Pre-registration not committed before the run.**
   * The pre-registration, the run and the results all landed in one Git commit (`af52b0b`, 18:21 UTC).
   * Evidence of order comes from file modification times and the in-script hash guard, not from Git history.
4. **The test rows were "previously exposed" (not disclosed in the M5 README).**
   * The repository README §10 states that the OWT test split was used for the 108-run table, and that the GoT
     geographic test was scored for all 108 old SAEs and twice for the layer-6 fine-tunes. It also says that "any
     further use will be labelled 'previously exposed'".
   * The M5 README says the 552 test rows "were used for nothing else", which is true only *within M5*.
   * **Mitigating facts:**
     * those earlier scores were of the *old* SAEs;
     * no R2/R4 design choice or selection used test numbers: they were made on validation and from the literature
       review;
     * the R2 and R4 SAEs had never been scored on test before M5.
   * **Correct label:** "CONFIRMED TEST RESULT on previously exposed test rows (exposed only for the reference SAEs)".
5. **RepE-LM has no test split.**
   * The repository README §10 described RepE-LM test as "the primary held-out set for M5". It was never built.
   * Every persona, RLM-RP, M4, E3 and R5 result therefore remains VALIDATION only.
6. **Statistics.**
   * There are 3 seeds per family.
   * The Welch CI reflects SAE-training-seed variance only. It does not include sampling variance over the 100 test
     documents or 452 GoT statements.
   * There is no multiplicity correction across the 136 M5 rows.
   * Under exchangeability, the "all 9 of 9 pairs" requirement alone has a one-sided chance probability of
     1/C(6,3) = 0.05 per cell. The two A2 GoT regressions at L8 could therefore be false positives.
   * The OWT effects are 10–50× larger than their CI half-widths, so multiplicity does not threaten them.
7. **Unpaired comparisons.**
   * C1–C3/A1/A2 compare different SAE families, so the seeds are not paired and Welch is used.
   * K1/K4 are genuinely paired: same base checkpoint and same tokens.

---

## 2. Reproducibility checks across the project

| Item | Status | Evidence |
|---|---|---|
| Optimiser consistency across layers | FIXED (2026-11-04). The uninterrupted protocol was used for all 108 baseline SAEs. | `scripts/openwebtext/13_consistent_protocol_retrain.py`; review §D1 |
| Sparsity matching (±10 % baseline, ±5 % fine-tunes) | ENFORCED. 107/108 baseline SAEs are matched (L0 JumpReLU seed 42 is excluded). All M5 SAEs are matched on OWT. | `layer_comparison/sparsity_matched_cells.csv`; M5 rows |
| Task-prompt L0 matching | **NOT DONE.** On GoT prompts, ReLU uses 77–112 latents, JumpReLU 63–93 and TopK 128; on RepE-LM, 141–216 against 128. E1 (task-matched TopK) is only a draft pre-registration. | M5 `got_cities__task_l0`; R2 README §2.1 |
| Probe circularity | FIXED. Truth-aware results are reclassified INVALID, and A/B groups and independent probes were introduced. | `docs/2026-11-07_truth_aware_finetuning_audit.md` |
| RepE length confound | FIXED by RepE-LM (equal-length one-token swaps; K1/K2 gates). The old RepE results are HISTORICAL. | `results/repe/repe_lm_gates/` |
| GPT-2 frozen | VERIFIED. Parameter hashes match before and after (`gpt2_parameters_unchanged: true`). | R2 `training_info.json`, R4 `finetune_info.json` |
| Determinism of the evaluation harness | The M5 smoke test reproduces the validation ladder with a maximum difference of 0.0 for OWT metrics and ≤ 1e-4 for GoT gaps. | M5 README §1.4 |
| Checkpoints backed up | **NOT DONE.** About 9 GB of EC2-only `.pt` files have no off-machine copy after 2026-11-06, and the R2/R4 checkpoints are not backed up either. | roadmap §11 |
| Kernel-STE JumpReLU (historical) | Training script MISSING; the result is unreproducible. | roadmap G4 |
| Teammate branches | Not merged. `simba-method-a-multilayer` duplicates the IMP-L script numbers. | roadmap §2 |

---

## 3. Results that are missing or NOT VERIFIED

| Claim someone might make | Status |
|---|---|
| "Longer training alone helped" | **NOT VERIFIED.** A1 confounds token count with recipe (AuxK, LR schedule, batch 4096, input scaling, initialisation). The R2 10M/25M/50M checkpoints exist but were never evaluated (roadmap: deferred, P2). |
| "Width helps in general" | **PARTIAL.** Only 3,072 vs 12,288 (A2), and only for TopK. The G5 width sweep was not run. On TEST, width improves OWT but regresses GoT at L8. |
| "Modern JumpReLU is better / worse" | **NOT VERIFIED.** Abandoned in the R2 pilot (L0 would not settle); no comparison exists. |
| "Truth preservation improved" | **NOT SUPPORTED** on TEST: 0/34 GoT cells. |
| "Causal preservation improved with R2" | VALIDATION only (L8 TopK T_rec/T_full 0.625 → 0.726, CI +0.019 to +0.182). **Not tested on held-out data.** |
| "Persona/honesty preservation on test" | **NOT AVAILABLE.** RepE-LM has no test split. |
| "Results generalise to Gemma-2-2B" | **NOT VERIFIED.** No Gemma SAE was evaluated (R3 / Gemma M3 planned). The Gemma work is steering only. |
| "Converged EV-vs-L0 frontier / L0 other than 128" | **NOT DONE** (G1). The 1000-step sweep measures convergence speed only. |
| "Feature activation frequency distribution" | **NOT REPORTED** beyond the dead fraction. No frequency histogram is in the final results. |
| "Statistical significance" | Only the pre-registered decision rules. No family-wise or FDR control over M5 (M2/E2 used BH within their own scope). |
| "Minimum detectable effect for E3/M4" | **NOT DONE** (roadmap, planned). |
| Residual two-stage SAE, AlignSAE | **NOT IMPLEMENTED.** |

---

## 4. Evidence-status map

| Result family | Status | Cite as |
|---|---|---|
| M5: R2 vs old, R4 vs MSE on OWT and GoT | CONFIRMED TEST (previously exposed rows; see 1.2.4) | main result |
| 108-run OWT reconstruction table | TEST | baseline |
| GoT all-layer frozen probes | TEST, interpretation superseded by M2 | do not read as "truth erased" |
| M0, M2, M3, M4, IMP-L, E3, RLM-RP, R2/R4 ladders, R5, E2 | VALIDATION | findings, not confirmed |
| Old RepE task, R1, R2-clamp, historical causal test (6c) | HISTORICAL | provenance only |
| Truth-aware logistic-probe gains | INVALID (circular) | never |
| `architecture_summary.csv` | INVALID | never |
| Gemma steering | descriptive appendix | appendix only |

---

## 5. Limitations (for the report and the defense)

1. **One model.** GPT-2 Small (124M); there is no SAE evidence on Gemma.
2. **Three seeds per cell.** This is enough for the large OWT effects, but underpowered for truth gaps (seed sd
   ≈ 0.01–0.05 AUROC).
3. **Layers:**
   * R2/R4/M5 cover layers 1, 8 and 11 only (R4 only 8 and 11);
   * the corrected baseline covers all 12 layers;
   * the original pre-review SAEs cover only 2/6/10.
4. **A single L0 of 128.** Task-prompt L0 is unmatched.
5. **A small, low-variance truth set.** GoT cities has 1,496 statements, geographically skewed (China 384, India 170).
   The final-token activations have tiny within-set variance (NMSE > 1). The re-fit gap does not separate trained
   SAEs from a random decoder at L8.
6. **A small persona effect.** The RepE-LM behavioural effect is about 0.02–0.04 nats. Causal results are VALIDATION
   only, with wide per-SAE CIs (±0.06–0.09 for TopK).
7. **The R2 recipe changed several factors at once.** We cannot attribute the gains to token count alone.
8. **Test-set exposure.** M5 test rows had been scored for the old reference SAEs before (1.2.4).
9. **Comparisons with public SAEs are not matched.** res-jb runs at L0 15–66, against our 128.
10. **No external replication.** All runs come from one codebase on one machine.

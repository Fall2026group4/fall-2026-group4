# Experiment history: GPT-2 Small SAE faithfulness

This is the authoritative human-readable history of the GPT-2 SAE work on branch `sae-research`. It was compiled on
2026-11-12 from the committed scripts, result CSVs, READMEs and the research journal. Every number below appears in
the source file named in its row.

**Related documents:**
* the full journal with narrative detail: [`RESEARCH_PROGRESS.md`](RESEARCH_PROGRESS.md);
* every quantitative table: [`EXPERIMENT_RESULTS_SUMMARY.md`](EXPERIMENT_RESULTS_SUMMARY.md);
* which results may be cited: [`results_registry.md`](results_registry.md).

**Machine-readable records** (kept as they are; this file links to them and does not replace them):
* [`experiment_manifest.csv`](experiment_manifest.csv);
* [`checkpoint_registry.csv`](checkpoint_registry.csv);
* [`migration_manifest.csv`](migration_manifest.csv).

Gemma-2-2B work belongs to a teammate and is not covered here, except as a pointer.

## How to read this file

**Verdicts:**
* **CONFIRMED TEST IMPROVEMENT**
* **VALIDATION IMPROVEMENT**
* **NO CLEAR CHANGE**
* **INCONCLUSIVE**
* **REGRESSION**
* **FAILED SUCCESS CRITERION**
* **INVALID / SUPERSEDED**
* **PROPOSED / NOT RUN**

Measurement-only experiments (no "before/after") are labelled **MEASUREMENT** with their split.

**Naming collisions.** Some IDs were reused, so this file always uses the unambiguous label:

| Label used here | Meaning | Not to be confused with |
|---|---|---|
| **R1-honesty**, **R2-clamp** | 2026-11-07/08 RepE layer-6 honesty fine-tune and SAE-in-the-loop clamp | the improvement-plan items "R1/R2" of 2026-11-10 |
| **R2-scaled** | 2026-11-10 data-scaled SAE retraining (`results/openwebtext/r2_scaled_saes/`) | R2-clamp |
| **E2** | public `res-jb` SAE reference (plan item "R1") | — |
| **M0–M5** | the 2026-11-08 measurement programme (D23) | the Gemma roadmap rows M1–M5 |
| **Method A / Methods B, C** | GoT reconstruction fine-tune / E2E-inspired KL objectives | the gap-plan "Methods A–F" of `research_gaps_and_plan.md` |

**Metric directions:**
* lower is better: NMSE, MSE, ΔCE, KL, AUROC gap, SAE/noise ratio, dead fraction;
* higher is better: EV, cosine, AUROC, T_rec/T_full;
* L0 is matched to a target (128) rather than minimised;
* norm retention ‖Δ̂‖/‖Δ‖ is ideally 1.

## Index

| # | Experiment | Date | Dataset | Method | Before | After | Change | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Phase-1 feature 974 | 09-15 | Paris prompts | public res-jb L8 | — | top feature 57.5 vs 20.1 | — | MEASUREMENT (exploratory) |
| 2 | In-house OWT pipeline + 1000-step sweep | 10-30 | OWT | ReLU/TopK/JumpReLU, width 3,072 | — | — | — | SUPERSEDED for cross-layer claims (convergence speed only) |
| 3 | GoT layer-6 probes + historical fine-tunes | 10-30 → 11-01 | GoT | recon-only / truth-aware, 500 steps | MLP test gap 0.327/0.154/0.323 | recon-only 0.161/0.110/0.146 | −0.04 to −0.18 | INVALID as matched evidence (L0 drifted, e.g. ReLU 65 → 144); truth-aware logistic numbers circular |
| 4 | Old RepE honesty pipeline + "SAE steering" | 11-03 | RepE (old) | mass-mean, steering | — | 0.93–0.99 correlations | — | INVALID / SUPERSEDED (no SAE in the forward pass; length confound) |
| 5 | Optimizer-consistency correction | 11-04 | OWT | uninterrupted 15k steps | e.g. L2 JumpReLU EV 0.929 | 0.936 | ≤ 0.006 EV | fix (no performance claim) |
| 6 | Sparsity-matching correction | 11-04 | OWT | ±10 % L0 rule | `architecture_summary.csv` | matched-cell tables | 15/36 sweep cells comparable | fix; old summary INVALID |
| 7 | Causal test, 200 controls | 11-04 | RepE (old) | raw / SAE directions | — | raw E 0.0048 (p 0.010); SAE ≈ controls | — | MEASUREMENT (TEST), now HISTORICAL |
| 8 | GoT L10 frozen probe | 11-04 | GoT | frozen vs re-trained | re-trained 0.851 | frozen 0.639 | gap 0.21 | MEASUREMENT (corrects a decodability claim) |
| 9 | 108-run all-layer baseline | 11-07 | OWT | 12 layers × 3 archs × 3 seeds | — | EV 0.93–0.95 (L0–2) → 0.76–0.82 (L11) | — | MEASUREMENT (TEST) |
| 10 | All-layer frozen probes | 11-07 | GoT, RepE (old) | frozen logistic | — | GoT gap 0.19–0.32 | — | MEASUREMENT (TEST); interpretation superseded by M2 |
| 11 | Phase 0 splicing | 11-07 | OWT, GoT, RepE | x̂ in the forward pass | — | ΔCE 0.01–0.04 → 0.21–0.36 nats with depth | — | MEASUREMENT (VAL) |
| 12 | P1 sparsity controller | 11-07 | OWT | anneal + freeze, quantile init | L2 JumpReLU EV, dead 65 % | EV +0.020, dead 21 % | — | VALIDATION IMPROVEMENT (seed 42 only; exploratory) |
| 13 | Truth-aware audit | 11-07 | GoT | — | — | teacher = evaluator found | — | INVALID (original truth-aware logistic results) |
| 14 | Corrected truth-aware pilot | 11-07 | GoT | A/B probes | recon-only MLP gap 0.107 | truth-aware 0.189 | +0.082 (worse) | FAILED SUCCESS CRITERION |
| 15 | Method A, L6 | 11-07 | GoT + OWT | GoT-A recon + OWT replay | TopK MLP gap 0.204 | 0.107 | −0.097; 18/18 paired better, none > 2 sd | FAILED SUCCESS CRITERION (directional) |
| 16 | E2E-inspired B/C, L6 | 11-07 | OWT, GoT | + output KL (+ downstream) | TopK OWT KL 0.115 | 0.102 | −11 % | partial: TopK behaviour met; ReLU/JumpReLU L0 guard failed; truth FAILED |
| 17 | KL-weight ablation, L6 TopK | 11-07 | OWT, GoT | w = 0.25, 0.5 | — | behaviour saturates by w = 0.25 | truth diff 0.007 < 0.02 | INCONCLUSIVE |
| 18 | R1-honesty, L6 | 11-07 | RepE (old) | paired-contrast fine-tune | ReLU B-MLP gap 0.180 (R-recon) | 0.167 | −0.013 | INCONCLUSIVE (ReLU, L0 guard) / FAILED (TopK, JumpReLU); HISTORICAL |
| 19 | R2-clamp, L6 | 11-08 | RepE (old) | SAE-in-the-loop clamp | random-feature q95 ≈ +0.004 | E −0.10 to −0.12 | adverse | FAILED SUCCESS CRITERION; HISTORICAL |
| 20 | M0 within-layer reanalysis | 11-08 | all | regression | — | error → splice damage (β 0.165); not → truth gap | — | MEASUREMENT (VAL) |
| 21 | M1 RepE-LM task + gates | 11-08 | RepE-LM | equal-length swaps | length AUROC 0.80 (old task) | K2 all 0.5000 | confound removed | MEASUREMENT (gates pass; K5 narrowly) |
| 22 | M2 four-level information | 11-08 | GoT, RepE-LM | frozen/re-fit/PCA/noise | frozen GoT gap 0.28 | noise 0.25; re-fit 0.07–0.12 | — | MEASUREMENT (VAL) |
| 23 | M3 causal site sweep | 11-08 | RepE-LM | patching, GPT-2 only | — | sites L1 slot, L8/L11 final | — | MEASUREMENT (VAL) |
| 24 | M4 SAE causal decomposition | 11-08 | RepE-LM | T_rec/T_full | — | 0.93–0.97 (L1), 0.62–0.68 (L8/L11) | — | MEASUREMENT (VAL) |
| 25 | IMP-L Method A | 11-09 | GoT | L3/8/9 | frozen-MLP gap | lower in 9/9 cells (4/9 met) | re-fit: no change | VALIDATION IMPROVEMENT (frozen reader only; qualified) |
| 26 | IMP-L output-KL | 11-09 | OWT | L3/8/9 | Method A OWT KL | −8 to −18 % | 4/9 with L0 drift +5–8 % | VALIDATION IMPROVEMENT in 5/9 cells; 4/9 not creditable; truth REGRESSION at L3 ReLU/TopK |
| 27 | E3 causal after fine-tuning | 11-10 | RepE-LM | M4 at L8 | base T_rec 0.621/0.625/0.665 | Method A 0.622/0.615/0.669 | ≈ 0 | NO CLEAR CHANGE |
| 28 | RLM-RP representation | 11-10 | RepE-LM | 216 SAEs | — | persona loss 1.9–3.4× noise | — | MEASUREMENT (VAL); Method A NO CLEAR CHANGE 12/12 |
| 29 | E2 public res-jb reference | 11-10 | all | external SAE | — | conclusions transfer | — | MEASUREMENT (VAL, descriptive) |
| 30 | **R2-scaled, OWT** | 11-10 | OWT | 100M tokens, 12,288 wide, modern recipe | L8 TopK NMSE 0.160, ΔCE 0.133 | 0.094, 0.062 | −41 %, −53 % | **CONFIRMED TEST IMPROVEMENT** |
| 31 | R2-scaled, GoT | 11-10 | GoT | same | L8 TopK re-fit gap 0.061 | 0.056 | −0.005 [−0.053, +0.043] | NO CLEAR CHANGE (TEST, 0/34 cells) |
| 32 | R2-scaled, RepE-LM | 11-10 | RepE-LM | same | L8 TopK persona gap 0.156; T_rec 0.625 | 0.108; 0.726 | −31 %; +0.101 | VALIDATION IMPROVEMENT (no test split exists) |
| 33 | R2 width ablation (A2) | 11-10 | OWT, GoT | 3,072 → 12,288 | — | OWT −20 to −52 % NMSE; GoT L8 worse | — | CONFIRMED TEST IMPROVEMENT (OWT) / REGRESSION (GoT L8, 2 cells) |
| 34 | **R4 KL fine-tune at fixed L0** | 11-10 | OWT | NMSE + α·KL, 5M tokens | L8 ΔCE 0.062 (MSE control) | 0.051 | −17 % | **CONFIRMED TEST IMPROVEMENT** (ΔCE, KL); NMSE +2 % = REGRESSION (TEST) |
| 35 | R4, causal | 11-10 | RepE-LM | same | L11 T_rec 0.694 | 0.671 | −0.024 (all seeds) | REGRESSION (VAL) |
| 36 | R5 causal skyline | 11-10 | RepE-LM | DAS/PCA/random rank-k | — | rank-4 DAS 0.91 vs whole old SAE 0.62–0.67 (L8) | — | MEASUREMENT (VAL) |
| 37 | M5 confirmatory test | 11-10 | OWT, GoT | single frozen pass | — | rows 30, 31, 33, 34 above | — | test pass of record |
| 38 | Final-defense audit | 11-10 | — | independent re-check | — | 136/136 decision classes reproduced | — | verification |
| 39 | GPT-2 audit and plan | 11-12 | — | read-only | — | X0–X3, T1, H1, H2, I1 proposed | — | PROPOSED / NOT RUN |

---

## Stage A: set-up and first pipelines (2026-09 to 2026-11-03)

### 1. Phase-1 feature 974 (2026-09-15, `070db9b`)
* **Question:** does a public SAE give an interpretable concept feature?
* **Setup:** GPT-2 Small at `blocks.8.hook_resid_pre`; SAELens `gpt2-small-res-jb` (24,576 latents).
* **Result:** feature 974 is the top feature at "Paris" (57.5 vs 20.1 for the runner-up; 58.97 vs 6.91 across cities).
* **Limitation:** a handful of examples; selectivity, not causal faithfulness.
* **Source:** `results/phase1_observations.md`, `results/tables/feature_974_detection_results.csv`.

### 2. In-house OWT SAE pipeline and 1000-step sweep (2026-10-30, `55ca6e2`, `8dc0b6e`)
* **Question:** compare ReLU, TopK and JumpReLU under one protocol, with SAEs trained in house.
* **Data:** 800/100/100 OWT documents, ≤ 128 tokens, BOS excluded. That gives **101,600 / 12,700 / 12,700 tokens**;
  15k steps × 256 ≈ 38 passes.
* **SAE:** width 3,072; Adam 3e-4; clip 10; decoder renormalised every step.
  * ReLU: L1 with a multiplicative L0 controller.
  * TopK: k = target.
  * JumpReLU: a sigmoid straight-through surrogate, not the published estimator.
* **Experiments:**
  * a 1000-step sweep, L0 32/64/128 × 12 layers;
  * convergence to 15k steps at layers 2/6/10.
* **Outcome:** later corrected (#5, #6). The sweep measures convergence speed only. `architecture_summary.csv` is
  INVALID. The kernel-STE JumpReLU checkpoints have no training script, so they are unreproducible.
* **Scripts:** `scripts/openwebtext/01–12`.
* **Results:** `results/openwebtext/{sae_sparsity_sweep,convergence,reconstruction_evaluation/layers_2_6_10}/`.

### 3. GoT layer-6 probes and historical fine-tunes (2026-10-30 → 11-01, `989ea71`, `0d07739`)
* **Data:** cities + neg_cities (geographic) and larger/smaller_than (numerical); final-token activations; grouped
  70/15/15 splits (`data/got/splits/`).
* **Original probe:** logistic validation AUROC at L6 is 0.841 (geographic) and 1.000 (numerical). The numerical set
  was later found to be shortcut-solvable: the second number alone gives AUROC 0.84.
* **Fine-tunes** (layer 6, 500 steps, lr 1e-5, no sparsity term):
  * reconstruction-only;
  * truth-aware: + 0.1 × the probe-logit MSE.
* **Test result, independent MLP probe, 5 seeds** (ReLU / TopK / JumpReLU):

  | Variant | MLP gap |
  |---|---|
  | Base | 0.327 / 0.154 / 0.323 |
  | Reconstruction-only | 0.161 / 0.110 / 0.146 |
  | Truth-aware | 0.225 / 0.184 / 0.218 |

* **Outcome:** **INVALID as matched evidence.** L0 drifted (ReLU 65 → 144 on GoT). The truth-aware logistic results are
  circular (#13). The GoT layer-6 test split has now been scored twice for these SAEs.
* **Scripts:** `scripts/geometry_of_truth/01–09`.
* **Results:** `results/geometry_of_truth/{final_test_evaluation,sae_finetuning}/`.

### 4. Old RepE honesty pipeline (2026-11-03, `0eaad9c`, `326b0f6`)
* **Data:** "Pretend you're an honest/untruthful person…" + a statement; 417/90/90 statements × 2 personas; a
  cross-wording set.
* **Probe:** mass-mean persona direction; validation AUROC at L6 is 0.893 (layer chosen on validation, so optimistic).
* **Steering:** reported 0.93–0.99 correlations between raw and SAE-derived directions.
* **Outcome:** **INVALID / SUPERSEDED.**
  * The "SAE steering" never put an SAE in the forward pass.
  * The whole task is length-confounded (" untruthful" is 3 tokens longer; length alone gives AUROC 0.80; D23).
* **Scripts:** `scripts/repe/01–12`. **Results:** `results/repe/` (see `results_registry.md`, HISTORICAL).

## Stage B: methodological corrections (2026-11-04 → 11-07)

Full report: [`archive/2026-11-04_methodology_review_corrections_report.md`](archive/2026-11-04_methodology_review_corrections_report.md),
with `2026-11-04_code_audit.md` and `2026-11-04_methodology_changes.md`. The corrections are summarised in
[`methodology.md`](methodology.md) §7.

### 5. Optimizer-state consistency (2026-11-04; migrated in `2ad0979`)
* **Problem:** layer 6 had been resumed with Adam state restored; layers 2/10 had been warm-started with Adam reset.
* **Fix:** all 27 runs (L2/6/10 × 3 architectures × seeds 42/1042/2042) retrained with one uninterrupted 15k-step
  protocol (`scripts/openwebtext/13_consistent_protocol_retrain.py`).
* **Result (TEST EV):** changed by ≤ 0.006. Rankings unchanged; two "wins" became ties.
  * L2 JumpReLU 0.9293 → 0.9355.
  * L6 TopK 0.8573 → 0.8572.
  * L10 JumpReLU 0.8253 → 0.8227.
* **Results:** `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/`.

### 6. Sparsity matching (2026-11-04)
* **Problem:** JumpReLU matched L0 = 32 in only 2/12 layers. `architecture_summary.csv` averaged unmatched runs and raw
  MSE across layers.
* **Fix:**
  * a ±10 % L0 rule for baselines and ±5 % for fine-tunes;
  * matched-cell tables;
  * the summary declared INVALID.
* **Result:** only 15/36 sweep cells are comparable. TopK's lead at 1000 steps reverses by 15k steps.
* **Results:** `results/openwebtext/layer_comparison/sparsity_{comparison,matched_cells}.csv`.

### 7. Pre-registered causal test with 200 controls (2026-11-04)
* **Setup:** layer 6, final token, 90 test statements (old RepE).
* **Result:** positive control (activation patch) E = 0.234 [0.151, 0.317]. The raw honesty direction gives
  E = 0.0048 [0.0003, 0.0096], p = 0.010 vs 200 random directions. SAE-derived directions are indistinguishable from
  controls.
* **Status:** TEST, now HISTORICAL. Old prompts; not SAE-in-the-loop.
* **Results:** `results/repe/steering/causal_random_controls/`.

### 8. GoT layer-10 frozen probe (2026-11-04)
* **Problem:** an earlier GoT result re-trained the probe on reconstructions, which measures decodability, not
  faithfulness.
* **Result:** the frozen original probe on public-SAE reconstructions gives AUROC 0.639 (gap 0.21 [0.16, 0.27]),
  against 0.851 re-trained.
* **Results:** `results/geometry_of_truth/layer10_frozen_probe/`.

### 9. 108-run all-layer OWT baseline (2026-11-07, `9446e56`), TEST
* **Design:** 12 layers × 3 architectures × 3 seeds; width 3,072; target L0 128; uninterrupted 15k steps; OWT test
  (12,700 tokens).
* **Result:**
  * 107/108 matched within ±10 %. Layer-0 JumpReLU seed 42 (L0 73.6, ~96 % dead) is excluded from rankings.
  * EV falls from 0.93–0.95 (L0–2) to 0.757 / 0.815 / 0.815 (ReLU / TopK / JumpReLU) at L11.
  * Matched leaders: ReLU at L1; JumpReLU at L3–9; JumpReLU ≈ TopK at L10–11.
* **Results:** `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/all_layers_*.csv`.

### 10. All-layer frozen probes (2026-11-07, `8919158`, `83bd9a9`), TEST
* **Design:** logistic probes fitted on original TRAIN rows, applied frozen to the 108 SAEs' reconstructions.
* **Result:**
  * GoT geographic gap 0.19–0.32 at the informative layers 4–11.
  * The old-RepE gap grows with depth.
* **Interpretation superseded:** M2 (#22) shows the frozen GoT collapse is mostly generic fragility. The RepE part is
  HISTORICAL (length confound).
* **Results:** `results/{geometry_of_truth,repe}/all_layer_frozen_probe_evaluation/`.

### 11. Phase 0 activation splicing (2026-11-07, `468018c`), VALIDATION
* **Design:** x̂ replaces `hook_resid_pre` at every non-BOS position (or the final token only).
* **Controls:** identity (exactly 0), fp16 round-trip, mean ablation, norm-matched noise.
* **Result:**
  * OWT ΔCE grows with depth (≈ 0.01–0.04 nats at L1–2, 0.21–0.36 at L11).
  * The SAE/noise damage ratio is 0.76–2.89 (corrected by M0).
  * GoT cities truth preference holds through L8.
* **Results:** `results/shared/spliced_evaluation/`.

### 12. P1 sparsity-control pilot (2026-11-07, `ef9fec4`), VALIDATION, seed 42
* **Design:** controller anneal then freeze; per-latent quantile JumpReLU threshold init; validation checkpoint
  selection (L0/2/6/10).
* **Result:**
  * L2 JumpReLU EV +0.020, dead latents 65 → 21 %.
  * Layer-0 JumpReLU not fixed.
  * Validation EV still rising at 15k steps (not converged).
* **Results:** `results/openwebtext/p1_sparsity_control_pilot/`.

## Stage C: improvement attempts on the small SAEs (2026-11-07 → 11-10)

### 13. Truth-aware circularity audit (2026-11-07, `1614be5`)
* **Finding:** the truth-aware loss distilled the logistic probe's logits, and the identical probe evaluated it.
* **Outcome:** those results are reclassified INVALID (numbers kept). The A/B probe protocol was introduced.
* **Source:** [`2026-11-07_truth_aware_finetuning_audit.md`](2026-11-07_truth_aware_finetuning_audit.md).

### 14. Corrected truth-aware pilot (2026-11-07, `dcf1486`), VALIDATION
* **Design:** L6 TopK × 3 seeds; teacher probe fitted on GoT-train group A, evaluation probes on group B
  (`data/got/splits/train_AB_groups_truth_aware_pilot.csv`); 50 % OWT replay; 500 steps; random-teacher control.
* **Result:** independent B-MLP gap: base 0.204 → recon-only **0.107** vs truth-aware 0.189. Truth-aware is worse on
  every seed.
* **Verdict:** FAILED SUCCESS CRITERION.
* **Results:** `results/geometry_of_truth/truth_aware_pilot/`.

### 15. Method A at layer 6 (2026-11-07, `ca81ad2`), VALIDATION
* **Design:** GoT-A reconstruction + OWT replay, 500 steps; ReLU/TopK/JumpReLU × 3 seeds; OWT-only control.
* **Result:**
  * B-MLP gap: ReLU 0.321 → 0.245; TopK 0.204 → 0.107; JumpReLU 0.292 → 0.238. Lower in 18/18 paired comparisons,
    but none by more than 2 sd.
  * GoT reconstruction error −72 to −82 %. OWT, logistic probe and RepE unchanged.
* **Verdict:** FAILED SUCCESS CRITERION (directional).
* **Scripts:** `scripts/geometry_of_truth/14–16`. **Results:** `results/geometry_of_truth/method_a_layer6/`.

### 16. E2E-inspired Methods B and C (2026-11-07, `764dcbb`), VALIDATION
* **Design:** B = A + α·KL(p_orig‖p_spliced); C = B + β·downstream NMSE (L7–11). Not an exact reproduction of Braun et
  al. 2024.
* **Result:**
  * TopK OWT KL 0.115 → 0.102; GoT statement KL 0.110 → 0.036–0.038.
  * ReLU/JumpReLU L0 rose 5–11 % (guard failed).
  * No truth gain anywhere.
* **Verdict:** partial (TopK behaviour); truth FAILED.
* **Results:** `results/geometry_of_truth/e2e_inspired_layer6/`.

### 17. KL-weight ablation, L6 TopK (2026-11-07, `2c87a6b`), VALIDATION
* **Design:** KL weight w ∈ {0.25, 0.5} × α_B; 3 seeds.

  | Metric | A | w = 0.25 | w = 0.50 | B |
  |---|---|---|---|---|
  | MLP gap | 0.107 ± 0.020 | 0.111 | 0.112 | 0.114 ± 0.022 |
  | OWT ΔCE | 0.1019 | 0.0928 | 0.0915 | 0.0910 |

* **Verdict:** INCONCLUSIVE. There is no truth trade-off to improve (A − B = 0.007 < 0.02).
* **Results:** `results/geometry_of_truth/kl_weight_ablation_layer6_topk/`.

### 18. R1-honesty (2026-11-07, `d17bb00` → `211218f` → `d283a49`), VALIDATION, now HISTORICAL
* **Design:** RepE (old task) layer-6 fine-tune with arms R-recon / R-honesty (paired contrast) / R-samepersona; probes
  on group B (`data/repe/splits/r1_train_AB_groups.csv`).
* **Result:**
  * ReLU B-MLP gap 0.180 → 0.167, but the RepE-prompt L0 guard failed (+6.8 / +8.4 %).
  * TopK 0.092 → 0.101; JumpReLU 0.072 → 0.075.
* **Verdict:** INCONCLUSIVE (ReLU) / FAILED SUCCESS CRITERION (TopK, JumpReLU). HISTORICAL (length confound).
* **Results:** `results/repe/r1_honesty_preservation_layer6/`.

### 19. R2-clamp (2026-11-08, `b232ce4` → `6d39d40`), VALIDATION, now HISTORICAL
* **Design:** encode → clamp honesty features → decode + error, at the final token, L6; random, matched-norm, unrelated
  and positive controls.
* **Result:** E(α = 1) −0.10 to −0.12, below the random-feature q95 (≈ +0.004); edits harm OWT.
* **Verdict:** FAILED SUCCESS CRITERION (NOT SUPPORTED, descriptive).
* **Results:** `results/repe/r2_sae_in_the_loop_steering_layer6/`.

### 20–24. Measurement programme M0–M4 (2026-11-08; D23)

| # | Stage | Commit | Result | Source |
|---|---|---|---|---|
| 20 | M0 within-layer reanalysis | `ea0e7fa` | Error predicts splice damage (β 0.165, Holm p 0.0008); not the truth gap (p 0.044, vanishes without architecture fixed effects); pooled ρ 0.86 was a pooling artefact | `results/shared/m0_within_layer_reanalysis/` |
| 21 | M1 RepE-LM task | `0818f73` | One-token persona swaps; K1–K7 pass (K5 narrowly 0.70); behaviour effect ≈ 0.019 nats; **train/validation only** | `data/repe/repe_lm/`, `results/repe/repe_lm_gates/` |
| 22 | M2 four-level information | `fc9a243` | Frozen GoT collapse 0.28 vs noise 0.25 and PCA 0.22; re-fit loss GoT 0.07–0.12, persona 0.14–0.18; SAEs never beat PCA | `results/shared/m2_four_level_faithfulness/` |
| 23 | M3 causal site sweep | `6a23d47` | Persona effect moves from the slot token (early) to the final token (late; handover L7–8); sites L1 slot, L11 final (+ L8 final via A1) | `results/repe/causal_site_sweep/` |
| 24 | M4 SAE decomposition | `3bbc62d` | T_rec/T_full 0.93–0.97 (L1 slot), 0.62–0.68 (L8/L11 final); ReLU's error alone carries 40–45 % | `results/repe/sae_patch_decomposition/` |

All are MEASUREMENT (VALIDATION).

### 25–26. IMP-L multilayer fine-tuning (2026-11-09, `864c4e0` → `7c9391d`), VALIDATION
* **Design:** OWT-only / Method A / output-KL at layers 3, 8, 9; 3 architectures × 3 seeds (81 runs); paired 95 %
  t-interval.
* **Method A:**
  * frozen-MLP truth gap lower in 9/9 cells; pre-registered criterion met in **4/9** (L3 TopK, L8 ReLU, L9 ReLU,
    L9 JumpReLU);
  * re-fit readers: no change in any cell.
  * Verdict: VALIDATION IMPROVEMENT for the frozen reader only.
* **Output-KL:**
  * OWT KL −8 to −18 % vs Method A;
  * ReLU/JumpReLU L0 +5.3–8.1 % at L8/L9, so 4/9 cells are NOT CREDITABLE;
  * truth REGRESSION at L3 ReLU (+0.012) and L3 TopK (+0.027).
* **Scripts:** `scripts/geometry_of_truth/22–23`. **Results:** `results/geometry_of_truth/multilayer_finetune/`.

### 27–28. E3 and RLM-RP (2026-11-10, `ce4edcf`), VALIDATION
* **E3** (M4 at L8 final, 36 SAEs):
  * T_rec/T_full: base 0.621 / 0.625 / 0.665 (ReLU / TopK / JumpReLU); Method A 0.622 / 0.615 / 0.669; output-KL
    0.640 / 0.619 / 0.687.
  * Method A: NO CLEAR CHANGE 3/3. Output-KL: TopK NO CLEAR CHANGE; ReLU/JumpReLU not creditable (L0).
  * Results: `results/repe/e3_finetuned_causal_preservation/`.
* **RLM-RP** (216 SAEs):
  * persona frozen gap 0.10–0.24 = 1.9–3.4× the noise reference; re-fit gap 0.08–0.22 (CI > 0 in 99/99 SAEs);
  * per-pair H−U cosine 0.35–0.58; **ReLU shrinks the contrast, TopK/JumpReLU inflate it**;
  * Method A NO CLEAR CHANGE 12/12; output-KL ≤ 0.007 AUROC gain in 3/12 with L0 +4–9 %.
  * Results: `results/repe/repe_lm_representation_preservation/`.

## Stage D: improvement phase with data-scaled SAEs (2026-11-10, `6745771` → `af52b0b`)

Motivated by the independent review ([`2026-11-10_independent_research_review.md`](2026-11-10_independent_research_review.md)):
the SAEs had seen about 0.1M unique tokens, against 300M for res-jb.

### 29. E2: public res-jb reference, VALIDATION (descriptive)
* res-jb at L1/8/10/11 through the same ladder (L0 15–66, unmatched to ours, so a reference only, never ranked).
* Conclusions H1–H4 transfer, except GoT H2 at L8.
* Results: `results/shared/e2_public_sae_reference/`.

### 30–33. R2-scaled: data-scaled retraining
* **Training** (`scripts/openwebtext/17_streaming_sae_training.py`, `src/sae/modern_sae.py`):
  * Data: 100M tokens per SAE from OWT parquet shard 78, with the 98 documents matching our sample removed.
  * Width 12,288; layers 1/8/11; seeds 42/1042/2042.
  * Architectures: TopK + AuxK, BatchTopK + AuxK, ReLU (decoder-norm-weighted L1 with a log-λ controller),
    TopK-3,072 (width ablation), and a frozen random-decoder TopK (seed 42).
  * Optimiser: Adam 2e-4, warm-up 500, linear decay over the last 20 %, clip 1; encoder = decoderᵀ init; inputs scaled
    to E‖x‖² = 768.
  * 39 SAEs, 5.5 h on one A10G; GPT-2 parameter hash unchanged.
  * Checkpoints at 10M/25M/50M and final (**the intermediate ones have never been evaluated**).
  * Modern JumpReLU (STE) attempted and **abandoned in the pilot**: L0 ≈ 1,660 at the published init; 19–506
    oscillation otherwise.
* **Results** (TEST from M5 unless marked; mean of 3 seeds):

  | Layer | Metric | Old TopK-3072 | R2 TopK-12288 | Change | Verdict |
  |---|---|---|---|---|---|
  | 8 | OWT NMSE ↓ | 0.160 | 0.094 | −41 % | CONFIRMED TEST IMPROVEMENT |
  | 8 | ΔCE ↓ (nats) | 0.133 | 0.062 | −53 % | CONFIRMED TEST IMPROVEMENT |
  | 11 | OWT NMSE ↓ | 0.185 | 0.111 | −40 % | CONFIRMED TEST IMPROVEMENT |
  | 11 | ΔCE ↓ | 0.204 | 0.096 | −53 % | CONFIRMED TEST IMPROVEMENT |
  | 8 | GoT re-fit gap ↓ | 0.061 | 0.056 | −0.005 [−0.053, +0.043] | NO CLEAR CHANGE |
  | 11 | GoT re-fit gap ↓ | 0.101 | 0.078 | −0.023 [−0.060, +0.013] | NO CLEAR CHANGE |
  | 8 | Persona re-fit gap ↓ (VAL) | 0.156 | 0.108 | −0.048 [−0.082, −0.014] | VALIDATION IMPROVEMENT |
  | 8 | T_rec/T_full ↑ (VAL) | 0.625 | 0.726 | +0.101 [+0.019, +0.182] | VALIDATION IMPROVEMENT |
  | 8 | Persona SAE − PCA ↓ (VAL) | 0.033 | 0.027 | −0.006 [−0.041, +0.028] | NO CLEAR CHANGE |

* **Attribution:** data, width and recipe changed together.
  * A1 (same width, new data + recipe): NMSE −23 to −35 %, about 55 % of the C1 gain at L8/L11. Token count and recipe
    are **not separated**.
  * A2 (width only): NMSE −20 to −52 % on OWT, but a GoT REGRESSION at L8 (2 cells).
* **Random decoder:** GoT SAE − PCA −0.003 (L8) / −0.013 (L11), better than every trained SAE. That metric does not
  separate trained from random SAEs (n = 1).
* **Results:** `results/openwebtext/r2_scaled_saes/`, `results/shared/m5_confirmatory_test/`.

### 34–35. R4: KL fine-tune at fixed L0
* **Design** (`scripts/openwebtext/18_kl_finetune_fixed_l0.py`): R2 TopK at L8/L11 × 3 seeds.
  * `kl` arm: NMSE + α·KL(p_orig‖p_spliced), α scale-matched (1.14–1.51).
  * `mse` control arm: NMSE only.
  * Same 5M tokens (shard 79); Adam 2e-5. L0 is exactly 128 by construction.
* **Result:**
  * ΔCE: L8 0.062 → 0.051 (−17 %); L11 0.096 → 0.081 (−16 %). All seeds; **CONFIRMED TEST IMPROVEMENT**.
  * The SAE/noise ratio at L8 falls to 0.98.
  * NMSE +2 %: classified **REGRESSION** in M5 (an expected cost of the KL term).
  * Causal T_rec at L11: 0.694 → 0.671 (−0.024, all seeds): **REGRESSION** (VAL).
  * Information: NO CLEAR CHANGE.
* **Results:** `results/openwebtext/r4_kl_finetune/`.

### 36. R5: causal skyline, VALIDATION (descriptive)
* **Design:** rank-k (k = 1/4/16/64) subspaces at the M4 sites: DAS-style learned on TRAIN pairs, PCA, mass-mean,
  20 random.
* **Result at L8, k = 4:** DAS 0.91; PCA 0.52; old-SAE top-4 features 0.10–0.16; random 0.008. **Rank-4 DAS carries
  more than the whole old SAE reconstruction (0.62–0.67).** R2 SAEs were not run through the feature arm.
* **Results:** `results/repe/r5_causal_subspace_skyline/`.

### 37. M5: confirmatory test pass (single run)
* **Scope:** 81 SAEs, 552 test rows (100 OWT documents; 452 GoT statements).
* **Integrity:** hash-guarded; refuses to overwrite.
* **Results:** C1/C2/C3/A1: 17 improved / 7 no change / 0 regressed each. A2: 2 GoT regressions at L8. K1/K4 (R4)
  confirmed.
* **Disclosures** (`docs/final_defense/EXPERIMENT_AUDIT.md`):
  * The test rows had been scored earlier for the *old* reference SAEs only (D31).
  * The pre-registration was committed with the results.
  * The README L0 maximum is 133.39, not 133.2.
  * There is no multiplicity correction.
* **Results:** `results/shared/m5_confirmatory_test/`.

### 38. Final-defense verification (2026-11-10, `446beea`)
* **Checks:** 11/11 hashes match; 81/81 rows; 136/136 decision classes reproduced by
  `scripts/shared/m5_independent_verification.py`.
* **Docs:** `docs/final_defense/`.

### 39. GPT-2 audit and improvement plan (2026-11-12, `b927f27`)
* **Scope:** read-only; [`2026-11-12_gpt2_audit_and_improvement_plan.md`](2026-11-12_gpt2_audit_and_improvement_plan.md).
* **Extended the RLM-RP contrast-distortion finding to the R2 SAEs:**
  * persona-contrast norm retention is 0.62–0.71 for ReLU-family SAEs;
  * it is 1.09–1.46 for TopK, BatchTopK and the old JumpReLU.
* **Proposed** (not run): X0–X3, T1, H1, H2, I1. See [`limitations_and_future_work.md`](limitations_and_future_work.md).

## Not found / never run

* a modern JumpReLU SAE;
* evaluations of the R2 10M/25M/50M checkpoints;
* R5 feature patches on R2 SAEs;
* any RepE-LM test result;
* a width sweep beyond 3,072/12,288;
* a converged L0 other than 128;
* Gated or Matryoshka SAEs;
* inference-time optimisation;
* a minimum-detectable-effect statement;
* E1 (task-L0-matched TopK; draft pre-registration only, `results/shared/e1_task_l0_matched_topk/README.md`).

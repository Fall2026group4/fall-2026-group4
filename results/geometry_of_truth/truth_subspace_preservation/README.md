# TSP: truth-subspace-preserving fine-tuning of R2 TopK (2026-11-14). Status: COMPLETED (VALIDATION pilot); pre-registered verdict NOT SUPPORTED (§3)

Evidence level of everything in this folder: **VALIDATION, exploratory pilot**. No TEST row is read. Nothing here is a
confirmatory test.

## 0. Diagnostics run before this plan (motivation; `diagnostics/`, script 24)

Run on 2026-11-14 on existing checkpoints only (no training), VALIDATION rows, every probe fitted on GoT-train
**group B only**, the truth subspace S (rank 8) estimated on **group A only**
(`data/got/splits/train_AB_groups_truth_aware_pilot.csv`). Seed means (3 seeds), descriptive, no CIs:

* **S is not a low-variance direction set on OWT** (variance share 1.04 at L8, 1.28 at L6, 1.75 at L11; 1 = average
  direction). The hypothesis "SAEs discard low-variance truth directions" is not supported at the subspace level.
* **S is sufficient for linear truth**: a group-B logistic probe on the 8-dim projection reaches 0.832 vs 0.819 on all
  768 dims (L8).
* **SAEs reconstruct S better than a same-rank random subspace** (centred relative error L8: R2 TopK 0.195 vs random
  0.413), but the **single discriminative (teacher-style logistic) direction is reconstructed worse than a constant**
  (relative error 0.86–3.87 across SAEs). This is the mechanism of the frozen-probe collapse.
* **Truth is decodable from the SAE error alone** (group-B logistic AUROC 0.72–0.83 vs original 0.78–0.82).
* **R2 halves the error inside S at L8 (old TopK 0.364 → R2 TopK 0.195) without changing the group-B re-fit gap
  (0.061 → 0.063).** Within a family the re-fit gap is not driven by subspace reconstruction error.
* **Truth-aware pilot (L6) re-read:** its "B-MLP gap" was a frozen MLP. With re-fit group-B readers the truth-aware
  arm is not worse than reconstruction-only (logistic 0.037 vs 0.060; MLP 0.028 vs 0.035); it distorts geometry
  instead (GoT NMSE 1.09 vs 0.26; random-subspace error 0.555 vs 0.336).

## 1. Plan (frozen before the full runs)

* **Research question:** at fixed sparsity (TopK, L0 = 128 on every prompt type), does adding an explicit penalty on
  reconstruction error *inside a multi-dimensional truth subspace estimated on independent data* improve
  independently measured truth preservation beyond (i) the same fine-tune without the penalty and (ii) the same penalty
  on a random or label-free subspace of equal rank, without costing reconstruction, behaviour, persona information or
  causal share?
* **Connection to the main question:** R2 improved reconstruction and behaviour but not GoT truth (M5 0/34 cells);
  truth-aware fine-tuning matched one probe and distorted geometry. TSP preserves a subspace (not a probe's logits),
  uses independent teacher/evaluator rows, and keeps L0 fixed.
* **Hypotheses (H0 / H1):** H0: tsp − ctrl = 0 for both co-primary metrics. H1: tsp − ctrl < 0 (lower gap).
* **Model / site:** GPT-2 Small (frozen), `blocks.8.hook_resid_pre`, final token for GoT.
* **SAE:** R2 TopK-12288, seeds 42 / 1042 / 2042 (each arm starts from the same base seed → paired).
* **Arms** (script 25): `ctrl` = NMSE_OWT + 1.0·NMSE_GoT-A; `tsp` = ctrl + 1.0·L_sub(S); `rand` = ctrl + 1.0·L_sub(random,
  rank 8); `pca` = ctrl + 1.0·L_sub(top-8 PCA of GoT-A). L_sub = E‖Pᵀ(x − x̂)‖² / E‖Pᵀ(x − x̄_A)‖² on GoT-A rows.
* **Subspace S:** span of the per-polarity and pooled class-mean contrasts and 24 grouped-bootstrap logistic weight
  vectors (C = 0.1), group A only; top-8 left singular vectors (`src/sae/truth_subspace.py`).
* **Budget / hyperparameters (fixed, no search):** R4 recipe: 1,230 steps × 32 × 127 OWT tokens (5.0M; shard 79 chunks
  ≥ 10,000, as R4), all 1,044 GoT-A rows each step, Adam lr 2e-5, clip 1, decoder renormalised each step, R2
  normalised coordinates. μ = 1, λ = 1, rank 8, subspace seed 20261122.
* **References:** R2 TopK base (no fine-tune), R4 MSE control (same tokens, no GoT rows).
* **Evaluation (script 26):**
  * Stage 1, independent GoT readers: M2 `original_side` / `evaluate_source` unchanged, group-A rows excluded; frozen
    logistic and mass-mean fitted on original group-B activations; re-fit logistic / MLP / kNN / sparse-z on x̂ (or z)
    of group B; validation cities; 1,000 shared grouped-bootstrap resamples.
  * Stage 2, the unchanged ladder: OWT NMSE / EV / L0 / dead, splicing ΔCE / KL / noise ratio, RepE-LM persona and
    neutral readers, RLM-RP, M4 T_rec/T_full at the L8 final-token site. Its GoT columns share rows with the subspace
    (group A) and are **never used for a criterion**.
* **Decision rule (script 27):** paired seed differences; se = √(se_seed² + v_stmt) (v_stmt from the shared bootstrap);
  df = 2; 95 % t-interval.
  * **Co-primary:** frozen_logistic_gap (readout preservation) and refit_logistic_gap (information), tsp vs ctrl; Holm
    over 2. A metric passes when 3/3 seeds favour tsp, the CI excludes 0 and Holm p < 0.05.
  * **Specificity:** tsp also beats rand and pca on that metric in 3/3 seeds (mean difference < 0).
  * **Guards (tsp vs ctrl, seed means):** OWT NMSE ≤ 1.02×; splice ΔCE ≤ 1.05×; persona re-fit gap ≤ +0.02; M4
    T_rec/T_full ≥ −0.02; OWT L0 = 128.
  * **Verdicts:** refit passes + specific + guards → INFORMATION IMPROVEMENT (validation pilot); only frozen passes →
    READOUT-ONLY IMPROVEMENT; neither → NOT SUPPORTED; any guard failure reported as a trade-off.
* **Stopping rules:** non-finite loss, L0 ≠ 128, or a GPT-2 parameter hash change → stop. No λ, rank or step tuning
  after seeing results; any follow-up needs a new pre-registration.
* **Power:** with 3 seeds (df = 2, t = 4.30) and statement sampling, only effects of several hundredths of AUROC are
  detectable (the AUDIT-SENS S1 finding). A null here is a bound, not equivalence.
* **Leakage checks (asserted in code):** no test rows loaded; train/validation city groups disjoint; A/B groups
  disjoint; group-A rows excluded from every evaluation probe.

## 2. Disclosure: seen before freezing

* The diagnostics of §0 (all validation).
* A 30-step smoke run (seed 42 only, scratch directory, not committed) of all four arms and of the evaluation
  pipeline. On those 30-step checkpoints the group-B re-fit logistic gap was base 0.091, ctrl 0.074, tsp 0.053,
  rand 0.066, pca 0.044. These are noisy single-seed numbers; they did not change any setting.

## 3. Run record, including the interruption

* **Plan frozen and committed** in `7d28706`, before any full run.
* **Interruption.** The pilot (script 25) started at 17:36 UTC. It stopped after 9 of 12 fine-tunes, when the Claude
  session that launched it ended at about 17:56.
  * The missing runs were tsp, rand and pca for seed 2042. No evaluation had run.
  * The 9 checkpoints were kept unchanged. Their training-monitor rows were lost (held in memory), but the final
    values survive in `pilot/logs/01_pilot_interrupted.log`.
* **Recovery** (`scripts/geometry_of_truth/25b_truth_subspace_finetune_resume.py`, run in tmux session `tsp-recovery`,
  18:13–18:48 UTC). The wrapper imports the frozen script 25 unchanged and calls its own `run_one`. Before training it
  verified:
  * all 12 non-README frozen hashes;
  * the saved subspaces equal a fresh recomputation (max |Δ| = 0);
  * all 9 existing checkpoints carry the frozen configuration;
  * **re-training ctrl seed 2042 reproduced the stored checkpoint bit-for-bit (max |ΔW| = 0)**
    (`pilot/recovery_check/repro_check.json`).

  It then trained only the 3 missing runs, overwrote nothing, and left GPT-2 parameters unchanged.
* **Evaluation:** script 26, 1,646 s. **Analysis:** script 27. Logs are in `pilot/logs/02–04_*.log`. Training peak
  GPU memory and the recovery details are in `pilot/finetune_info.json`.
* **Checkpoints:** 12 × 75 MB under `pilot/checkpoints/` (not in Git; listed in `docs/checkpoint_registry.csv` as
  `got_tsp_pilot`).

## 4. Results (VALIDATION; GoT readers fitted on group B only; mean ± sd over 3 seeds)

Original group-B AUROC on validation (n = 448 statements; 1,048 group-B TRAIN rows): logistic 0.819, MLP 0.846, mass-mean
0.555.

**Lower is better** for every metric below except EV-type and M4 metrics. L0 = 128 exactly on OWT, GoT and RepE-LM prompts
for every SAE.

| Metric | R2 base | R4 MSE | ctrl | **tsp** | rand | pca |
|---|---|---|---|---|---|---|
| Frozen logistic gap (co-primary) | 0.210 ± 0.008 | 0.218 | 0.051 ± 0.014 | **0.012 ± 0.017** | 0.090 | 0.059 |
| Re-fit logistic gap (co-primary) | 0.063 ± 0.024 | 0.054 | 0.019 ± 0.004 | **−0.001 ± 0.008** | 0.024 | 0.026 |
| Re-fit MLP gap | 0.031 | 0.017 | 0.037 | 0.027 | 0.028 | 0.026 |
| Re-fit sparse-z gap (latent-level) | 0.068 | 0.062 | 0.019 | **0.002** | 0.028 | 0.024 |
| Frozen mass-mean gap | 0.018 | 0.016 | 0.007 | 0.001 | 0.011 | 0.008 |
| Truth AUROC from the error alone (↓ = less displaced) | 0.708 | 0.715 | 0.708 | **0.609** | 0.743 | 0.716 |
| GoT task NMSE | 0.840 | 0.838 | 0.037 | 0.043 | 0.053 | 0.037 |
| OWT NMSE (ladder) | 0.0959 | 0.0956 | 0.0974 | 0.0974 | 0.0971 | 0.0974 |
| OWT spliced ΔCE (nats) | 0.0603 | 0.0601 | 0.0614 | 0.0618 | 0.0617 | 0.0614 |
| Persona re-fit gap (RepE-LM) | 0.108 | 0.109 | 0.110 | 0.109 | 0.110 | 0.112 |
| M4 T_rec/T_full (↑) | 0.726 | 0.720 | 0.706 | 0.711 | 0.719 | 0.712 |

### Pre-registered decision (tsp vs ctrl; paired by seed; se includes statement sampling; df = 2; Holm over 2)

| Metric | Per-seed differences | Mean diff | 95 % CI | Holm p | Rule |
|---|---|---|---|---|---|
| Frozen logistic gap | −0.040 / −0.049 / −0.027 | −0.039 | [−0.084, +0.007] | 0.135 | **not met** (3/3 seeds; CI includes 0) |
| Re-fit logistic gap | −0.021 / −0.023 / −0.016 | −0.020 | [−0.058, +0.017] | 0.147 | **not met** (3/3 seeds; CI includes 0) |

* **Specificity:** met for both metrics. tsp beats rand and pca in 3/3 seeds.
* **Guards:** all passed:
  * OWT NMSE identical;
  * ΔCE +0.0004 (n.s.);
  * persona −0.001;
  * M4 +0.006;
  * L0 = 128.
* **Verdict (frozen rule): NOT SUPPORTED.** The direction is consistent across seeds, specific to the truth subspace and
  free of measured costs. It is not distinguishable from zero under 3 seeds plus statement sampling.

### Secondary comparisons (descriptive, uncorrected, same interval method)

* **tsp vs ctrl:**
  * error-only truth AUROC −0.099 [−0.154, −0.044], p = 0.016 (less truth information left in the error);
  * sparse-z gap −0.017 [−0.035, +0.000], p = 0.051;
  * re-fit MLP gap −0.010 [−0.027, +0.008].
* **tsp vs rand:** frozen gap −0.078 [−0.136, −0.021], p = 0.028.
* **tsp vs pca:** frozen gap −0.047 [−0.094, −0.001], p = 0.048.
* **ctrl vs R2 base** (task-distribution replay, the Method-A analogue on R2):
  * frozen gap −0.159 [−0.247, −0.071], p = 0.016, 3/3;
  * re-fit gap −0.044 [−0.124, +0.036];
  * re-fit MLP +0.006 (no change);
  * cost: OWT NMSE +1.6 % (0.0959 → 0.0974, 3/3 seeds worse, CI excludes 0), ΔCE +1.8 % (0.0603 → 0.0614), M4 −0.020
    [−0.076, +0.037].
* **rand vs ctrl:** frozen gap +0.040 (worse in 3/3). A geometry-free subspace penalty hurts readout.

## 5. Interpretation and limitations

* **What is supported (validation, descriptive):**
  * Most of the frozen-readout recovery comes from training on the task's own prompt distribution (ctrl vs base:
    0.210 → 0.051, at a small OWT cost).
  * The truth-subspace term adds a further, consistent reduction (to 0.012 frozen; −0.001 re-fit) and moves
    truth-predictive signal out of the error (error-only AUROC 0.708 → 0.609). Random and PCA subspaces do not do
    this.
* **What is not supported:** the pre-registered claim that TSP improves independently measured truth preservation over
  ctrl. With df = 2 the intervals are wide.
* **Limits:**
  * one layer (L8), one architecture (TopK) and 3 seeds;
  * GoT-A and validation share templates (cities), so the gain may be distribution-specific: there is no
    out-of-template or out-of-task test;
  * the re-fit MLP reader does not improve (0.037 → 0.027, n.s.), so the gain is mainly in linear readers;
  * the subspace is label-derived from group A. Evaluation probes use disjoint rows (group B) and disjoint cities
    (validation), but this is supervised preservation, not unsupervised faithfulness;
  * no test split was used.
* **Next falsifiable test:** the same arms with more seeds and a held-out-template truth set, under a new
  pre-registration (see `docs/RESEARCH_ROADMAP.md`).

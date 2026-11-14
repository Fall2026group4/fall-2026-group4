# TSP: truth-subspace-preserving fine-tuning of R2 TopK (2026-11-14). Status: PRE-REGISTERED (plan below); results in §3 when run

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

## 3. Results

To be added after the runs.

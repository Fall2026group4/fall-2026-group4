# TSP-R Phase 1: replication of the TSP fine-tune (2026-11-15). Status: PRE-REGISTERED (frozen); NOT STARTED, awaiting launch approval

**Evidence level when run:** VALIDATION, confirmatory. The primary dataset is cross-fitted cities (train + val
groups). Fresh-family TEST evaluations are secondary or exploratory.

The frozen plan is in `preregistration.json`. It holds 19 SHA-256 hashes covering:
* the scripts and recipes;
* the data splits and the OWT token files;
* `plan_matrix.csv` and `subspaces.npz`.

**Scripts:**
* `scripts/geometry_of_truth/30_tsp_r_phase1_training.py`, with stages `plan` (run, dry), `repro`, `base`, `base_qc`
  and `finetune`.
* The evaluation script will be frozen in a second commit before any Phase 1 SAE is evaluated.

## Design summary
* **Training runs:** 8 base seeds (42 / 1042 / 2042 existing; 3042 / 4042 / 5042 / 6042 / 7042 new) × 5 arms (ctrl,
  tsp, rand, pca, shuf) × 3 folds = 120 fine-tunes. The recipe is the exact pilot recipe.
* **Folds:** fold f evaluates on V = f, fits probes on B = f+1, and estimates S / replays GoT rows on A = f+2. Over
  the 3 folds, 635 cities groups (2,540 statements) are each evaluated once.
* **Primary metric:** re-fit logistic gap, tsp − ctrl.
* **Multiplicity:** fixed-sequence gatekeeping. If H1 (tsp vs ctrl) is rejected, H2 (vs shuf) and H3 (vs rand) are
  tested with Holm. Family-wise α = 0.05.
* **Effect-size threshold:** SESOI 0.010.
* **Uncertainty:** se combines seed and city-bootstrap components, with Satterthwaite df.
* **Cost guards:** non-inferiority on OWT NMSE (+2 %), ΔCE (+5 %), KL (+5 %), the persona gap (+0.02), M4 (−0.02) and
  L0 = 128.
* **Secondary and exploratory evaluations:** companies is a secondary estimate, common_claim a distortion check,
  counterfact exploratory.

## Dry-run checks (`plan_info.json`)
* **Matrix:** 120 unique runs; no checkpoint path exists yet.
* **Fold sizes:** A / B / V each have 211–212 groups (844–848 rows).
* **Subspace overlaps:**
  * tsp~shuf 0.04; tsp~rand 0.007 (chance 0.0104).
  * Cross-fold overlap of S_tsp is 0.26–0.31, so only the leading truth direction is stable.
* **Where an independent B-fold probe direction lies:** 50–55 % of its norm² falls in S_tsp, 8 % in S_shuf, 1.5 % in
  S_rand and ~0 in S_pca.
* **Disk:** 74.4 GB free; ≈ 8.8 GB of new checkpoints (72 MB each).

# X0: token exposure within the R2 runs (10M / 25M / 50M / final checkpoints). Status: COMPLETED (2026-11-21; VALIDATION, descriptive)

* **Evidence level:** VALIDATION, descriptive. No test row is read; nothing is trained.
* **Remediation item:** S8 / Review #8 ("is the gain from more data?").
* **Frozen protocol:** [`preregistration.json`](preregistration.json).

## 1. Before running

### Question

R2 changed training data, width and recipe together, so its gain cannot be credited to data. Does reconstruction
and behavioural faithfulness keep improving with **token exposure** when width, recipe, seed and data stream are
held fixed?

### Why these checkpoints can answer part of it

The checkpoints were verified from their stored configs and from `scripts/openwebtext/17_streaming_sae_training.py`:

* Each R2 run saved `checkpoint_{10M,25M,50M}.pt` and `checkpoint_final.pt` (99,999,744 tokens).
  * Within a run, architecture, width, k / target L0, seed, initialisation, input scaling (`input_scale_folded`) and
    data stream are identical.
  * The data is fresh OWT shard 78 with no repetition, so exposure equals unique tokens.
* **LR schedule:** linear warmup over 500 steps, then constant 2e-4 until 80 % of training, then linear decay to 0.
  * 10M, 25M and 50M all lie in the constant-LR phase. Between them, **only token exposure changes**.
  * 50M → final adds 50M tokens **and** the LR decay, so that step mixes the two effects.
* **What this is not:** a data-scaling law. A model trained to completion on 10M tokens, with its own decay, would
  differ from a mid-run 10M checkpoint. The old SAEs differ from R2 in data (101,600 tokens, about 38 passes),
  width and recipe, so they are shown for reference only and never as a point on the curve.
* **ReLU caveat:** the λ controller adapts during training, so mid-run L0 may differ from 128. Every comparison
  requires OWT validation L0 within ±10 % of 128; cells outside the band are flagged and excluded.

### Design

* **Harness:** the unchanged ladder (`scripts/shared/sae_ladder_evaluation.py --sources r2scaling --light`).
  * Reconstruction is measured on the OWT validation cache.
  * Splicing uses 100 validation sequences (ΔCE, KL) with the same-norm noise reference.
  * The 100M point is the committed ladder row (same harness, same validation data).
* **SAEs:** R2 TopK-12288, BatchTopK-12288, ReLU-12288 and TopK-3072 at layers 1, 8 and 11, with seeds 42, 1042 and
  2042, at 10M, 25M and 50M: 108 checkpoints.
* **Metrics:**
  * OWT NMSE ↓, explained variance ↑, cosine ↑, L0 (reported), dead fraction ↓;
  * spliced ΔCE ↓ and KL ↓;
  * SAE / noise ΔCE ratio ↓.

### Comparisons and rule

* Paired by seed within family × layer:
  * **10M → 25M** and **25M → 50M:** exposure only;
  * **50M → final:** exposure plus LR decay; labelled as confounded.
* **Classes:**
  * **IMPROVES** if all 3 seeds improve and |mean paired difference| > 2 sd (the paired rule used for K1/K4);
  * **WORSENS** for the mirror image;
  * otherwise **NO CLEAR CHANGE**.
* **Pre-registered reading:**
  * if 25M → 50M IMPROVES on NMSE and ΔCE in most cells, R2 had not saturated in exposure at 50M at constant LR;
  * if NO CLEAR CHANGE, exposure beyond 25M added little at constant LR.
  * Neither result attributes the old → R2 gain to data, because width and recipe also changed.

### Cost

* About 108 light ladder passes on one A10G (OWT reconstruction plus splicing): roughly 30–60 minutes.
* It runs after, or alongside, the H1-TEST pass.

## 2. Results (run 2026-11-21; frozen commands unchanged, all 108 checkpoint hashes verified first)

**Outputs:** `ladder_layer{1,8,11}_r2scaling.{csv,json}`, `per_checkpoint.csv`, `comparisons.csv`, `summary.json`,
`figures/` (from `scripts/shared/r2_scaling_analysis.py`). 144 checkpoint rows. **No cell is outside the L0 band**
(128 ± 10 %), so none is excluded. Paired-by-seed rule, 3 seeds per cell, 12 cells (4 families × 3 layers).

| Step | OWT NMSE: IMPROVES / 12 (median change) | Spliced ΔCE: IMPROVES / 12 (median) | SAE/noise ΔCE ratio: IMPROVES / 12 |
|---|---|---|---|
| 10M → 25M (exposure only) | **12** (−17.1 %) | **11** (−26.7 %) | 9 |
| 25M → 50M (exposure only) | **12** (−8.8 %) | **11** (−12.9 %) | 2 |
| 50M → final (exposure + LR decay; confounded) | 12 (−6.7 %) | 8 (−8.9 %) | 2 |

No cell WORSENS on NMSE or ΔCE at any step. The non-IMPROVES ΔCE cells are all at layer 1 (BatchTopK 10M→25M,
ReLU 25M→50M; four at 50M→final, including TopK-3072 +1.4 %), where ΔCE is small and seed noise is relatively large.

**Pre-registered reading:** 25M → 50M IMPROVES on NMSE and ΔCE in most cells (12/12 and 11/12), so **R2 had not
saturated in token exposure at 50M at constant LR**. Returns diminish: each doubling-or-so of exposure roughly halves
the relative gain (NMSE −17 % → −9 %).

**Secondary observation (descriptive):** the SAE/noise ΔCE ratio improves 10M → 25M (9/12) but is flat after 25M
(2/12 IMPROVES; 10 NO CLEAR CHANGE). Past 25M, the SAE's behavioural damage falls in step with that of norm-matched
random noise at the shrinking error norm, i.e. extra exposure mainly shrinks the error rather than making it less
harmful per unit norm.

**Limitations (as pre-registered):** mid-run checkpoints are not models trained to completion on that budget; this is
not a data-scaling law, and it does not attribute the old → R2 gain to data (width and recipe changed too). Seeds share
one data stream, so seed sd understates variance. Validation only; no TEST row was read.

# TI-1: TSP generalisation across layers and architectures (GPT-2 Small). Status: PRE-REGISTERED; gates PASSED; training IN PROGRESS

**Pre-registration:** `preregistration.json` (frozen before training; hashes in `frozen_hashes.txt`).
**Script:** `scripts/geometry_of_truth/32_tsp_generalisation.py` (stages: gate, repro, train, eval, analyze).
**Evidence level on completion:** VALIDATION.

## Question
Does the TSP-R effect (cities re-fit gap, tsp − ctrl = −0.0207 at L8 TopK, 8 seeds) hold at other layers and for
other architectures under the identical recipe?

**Arms:**
* base: pretrained R2;
* ctrl: reconstruction-only GoT-A replay, same compute;
* tsp;
* rand: a Haar-random subspace.

3 seeds × 3 cross-fitted folds. Readers are fitted on fold B and evaluated on V; training rows and the subspace come
from A.

## Gates (PASSED, 2026-11-21)
| Gate | Result |
|---|---|
| Eligibility (original refit AUROC ≥ 0.70 on V in all folds) | L8 0.78 / 0.83 / 0.83 and L11 0.74 / 0.77 / 0.77: eligible. **L1 0.51 / 0.52 / 0.51: NOT eligible, so cell L1_topk is not run** (GPT-2 has no pooled cities signal at L1) |
| R0 (generalised fine-tune == frozen `run_one`, TopK L8, 30 steps) | max parameter difference 0.0 |

**Cells run:** L11_topk, L8_batchtopk and L8_relu. The ReLU and BatchTopK fine-tunes include the R2 trainer's own
sparsity control. Runs with L0 outside ±5 % of base are flagged and excluded from rankings.

## Results
Pending.

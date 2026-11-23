# TI-1: TSP generalisation across layers and architectures (GPT-2 Small). Status: COMPLETED (2026-11-21; VALIDATION). Verdict: GENERALISES (3/3 eligible cells SUPPORTED)

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

## Results (2026-11-21; `analysis/contrasts.csv`, `analysis/primary_tsp_vs_ctrl.csv`)

81 fine-tunes (3 cells × 3 arms × 3 seeds × 3 folds), all evaluated (`eval/<cell>/`, 70 files each; ledgers
`ledger_<cell>.csv`). Lower gap = better. Hierarchical seed + city-cluster bootstrap (`tspr_stats`); Holm over the 3
cells for the primary contrast.

**Primary (refit-logistic cities gap, tsp − ctrl):**

| Cell | Estimate [95 % CI] | p | p_holm | Seeds negative | Label |
|---|---|---|---|---|---|
| L11 TopK | **−0.0415** [−0.0533, −0.0298] | 4.5e-5 | 9.1e-5 | 3/3 | SUPPORTED (meaningful) |
| L8 BatchTopK | **−0.0153** [−0.0241, −0.0065] | 7.1e-4 | 7.1e-4 | 3/3 | SUPPORTED (meaningful) |
| L8 ReLU | **−0.0258** [−0.0316, −0.0201] | 9.6e-11 | 2.9e-10 | 3/3 | SUPPORTED (meaningful) |

Reference: TSP-R at L8 TopK, −0.0207 (8 seeds).

**Pre-registered success criteria:**

| Criterion | Result |
|---|---|
| cell_supported (label SUPPORTED and p_holm < 0.05) | **3/3 cells** |
| generalises (≥ 2 eligible cells SUPPORTED; no cell with tsp − ctrl CI wholly > 0) | **MET** |
| cost guard (tsp vs base OWT NMSE ≤ +3 %; L0 within ±5 %) | **MET in all cells**: NMSE +2.5 % / +1.6 % / +0.3 %; L0 change 0.0 % / +0.1 % / +0.2 % |

**Truth-specificity and controls (refit logistic):**

| Cell | tsp − rand | ctrl − base | tsp − base | Absolute gap: tsp arm |
|---|---|---|---|---|
| L11 TopK | −0.049 (CI high −0.034) | +0.003 | −0.039 | 0.054 |
| L8 BatchTopK | −0.022 (CI high −0.011) | −0.026 | −0.041 | 0.042 |
| L8 ReLU | −0.026 (CI high −0.020) | −0.012 | −0.038 | −0.004 (gap closed) |

TSP beats the random-subspace control in every cell, so the gain is specific to the truth subspace. At L8 the
reconstruction-only GoT replay (ctrl) already narrows the gap; TSP adds a further, significant reduction on top.

**Non-linear readers (tsp − ctrl):**

| Cell | MLP-10 | MLP-256 |
|---|---|---|
| L11 TopK | −0.017 (CI high +0.005, n.s.) | −0.022 (CI high −0.018) |
| L8 BatchTopK | −0.009 (CI high −0.004) | −0.007 (CI high +0.003, n.s.) |
| L8 ReLU | −0.006 (CI high −0.001) | −0.007 (CI high −0.005) |

Same direction everywhere, smaller and not always significant. Most of the gain is in linearly decodable truth
information.

**OWT cost:** the ctrl arm costs as much as tsp (L11: ctrl +2.5 % NMSE vs tsp +2.5 %; ΔCE +3.1 % vs +2.8 %). So the cost
comes from the GoT replay, not from the truth term.

**Frozen probe** (reported, not used): tsp − ctrl −0.056 / −0.023 / −0.052. Partly circular for TSP, because the
frozen probe's direction lies largely inside the trained subspace (audit §3.1). Not used as evidence.

**Limitations:**
* cities only: transfer to other truth families was not tested here (TSP-R found none at L8; TI-2 addresses it);
* 3 seeds per cell (TSP-R used 8), and seeds share one data stream; the cluster bootstrap carries the statement-level
  uncertainty;
* L1 not eligible (no cities signal in GPT-2 at L1);
* GPT-2 only; the Gemma transfer (TI-G) is blocked on disk.

**Deviations from the frozen script** (`frozen_hashes.txt` records the pre-training hash):
1. the OOM fix in `stage_eval` (noted below; batch size only);
2. `stage_analyze`: a dict was read with attribute access (`r.seed` → `r['seed']`). It crashed before writing
   anything. The empty `analysis/` directory it had created was removed, and the stage was re-run once. No
   computation changed.
* **Implementation fix (2026-11-21, before any TI-1 analysis).** The OWT splice in `stage_eval` ran out of GPU memory (shared GPU). The splice batch size is now 4 instead of 16, and the original log-probs use the same batch size (`splice_per_seq` indexes them by batch). Per-sequence sums are unchanged by batching, apart from fp32 kernel noise. The cities evaluation had already completed and is untouched.

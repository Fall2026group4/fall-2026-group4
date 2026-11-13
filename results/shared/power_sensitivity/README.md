# Sensitivity (minimum detectable effect) of the GoT and M4 comparisons (2026-11-13). Status: COMPLETED

* **Evidence level:** a descriptive re-analysis of committed outputs. No model, checkpoint, cache or test row was
  read, and nothing was re-run.
* **Remediation items:** S2, S7; Review #8 critique 3 ("say how large an effect your test could detect").
* **Script:** [`scripts/shared/power_sensitivity_analysis.py`](../../../scripts/shared/power_sensitivity_analysis.py).
* **Definitions** (two-sided α = 0.05, power 0.80) are in the script docstring.
  * MDE = 2.80 × SE.
  * The statement-level SE comes from the stored grouped-bootstrap CIs.
  * The seed-level SE comes from the stored Welch inputs of M5's R2 rule.
  * This is a **sensitivity estimate of the design, not post-hoc power** for the observed effects.

## Results

| Quantity | Value | Source |
|---|---|---|
| GoT re-fit gap of one SAE: statement-level MDE (TEST, n = 452) | 0.030–0.048 AUROC | M5 `m5_rows.csv` CIs |
| GoT frozen gap of one SAE: statement-level MDE | 0.053–0.072 | same |
| GoT re-fit gap seed sd (trained families) | 0.003–0.026 | same |
| M5 GoT between-family comparisons, seed-level MDE (3 vs 3 seeds) | 0.011–0.129 | M5 `comparisons.csv` |
| M5 GoT re-fit comparisons, combined seed + statement MDE (approximate) | **0.048–0.091 AUROC** | + PCA-CM paired SEs |
| GoT comparisons whose \|difference\| reaches the seed-level MDE | 2 of 30 (both width-ablation A2 at L8: the re-fit gap and SAE − PCA, \|d\| = 0.012 and 0.015, far below the statement-level MDE of about 0.047) | — |
| M4 causal effect T_full | 0.022 nats (L8), 0.043–0.044 (L1, L11); CI low ≥ 0.78 × mean everywhere, so the ratio is not unstable | M4 `decomposition_summary.csv` |
| M4 preservation T_rec/T_full, statement-level MDE per SAE | 0.013–0.017 (L1); 0.036–0.069 (L11); **0.054–0.107 (L8)** | same |

## What this means

* **GoT:** the observed between-family differences in the re-fit truth gap are ≤ 0.025 AUROC, well below the about
  0.05–0.09 AUROC the M5 design can detect.
  * The 0/34 "NO CLEAR CHANGE" result therefore means **no change larger than about 0.05–0.09 AUROC**. It does
    **not** mean truth information is preserved.
  * The 2 width-ablation "REGRESSION" cells at L8 pass the seed-only rule but are about 4× smaller than the
    statement-level MDE. They should be read as not robust.
* **M4 (VALIDATION):**
  * At L1, preservation differences of about 0.015 are detectable.
  * At L8 the per-SAE statement-level MDE is up to 0.107 for the old TopK. The R2-vs-old TopK gain at L8
    (+0.101, VALIDATION, R2 rule on seeds) is therefore **at the edge of statement-level detectability**. It should
    be reported with that caveat.
* **Approximation:** the SAE-vs-SAE statement-level SE is approximated by the SAE-vs-PCA paired SE (PCA-CM), which
  is probably conservative.

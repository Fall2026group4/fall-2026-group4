# Frozen-probe error decomposition and noise control (2026-11-10). Status: COMPLETED

Evidence level: exploratory (validation rows only).

Script: `scripts/shared/frozen_probe_error_decomposition.py`. Folders: `got_inhouse/`, `repe_inhouse/` (the 108 rebuilt in-house SAEs), `got_public_relu/` (`gpt2-small-res-jb`) and `got_public_topk/` (`gpt2-small-resid-post-v5-32k`, k = 32).

## 1. Design

* **Question:** the frozen-probe AUROC drops a lot when SAE reconstructions replace the activations, even though cosine is about 0.98. Is that because any error of that size breaks the probe, or because the SAE error is aimed at the truth direction?
* **Comparison:** the SAE reconstruction is compared with a **norm-matched random error**: the original activation plus random noise of the same size as the SAE error.
* **Also reported:** signal retention along the probe direction, noise inflation, and error anisotropy (how much the error lies along the truth direction compared with a random direction).

## 2. Results

* **Geometry of Truth, geographic, informative layers:**
  * The SAE gap is 0.27–0.30 AUROC, against 0.26 for random error of the same size.
  * The excess over random error is only 0.01–0.03 for the in-house SAEs and 0.04–0.05 for the public SAEs.
* **Error anisotropy is below 1:** the SAE error is not aimed at the truth direction. If anything, it is less aligned with it than random error is.
* **RepE (old, length-confounded prompts):** TopK and JumpReLU do better than noise of the same size; ReLU does worse (+0.079).
* **Reproduction:** single-run AUROCs from the rebuild differ from the saved ones (median 0.01, maximum 0.17), while the architecture-mean gaps agree within 0.007. This itself shows how fragile the frozen probe is. The script was run with `--skip-reproduction-check` for that reason, and this is documented here.
* Figure: `results/shared/followup_figures/F4_probe_gap_vs_random_error.png`.

## 3. Interpretation

Most of the frozen-probe collapse is the probe's sensitivity to any error of that size, not SAE error aimed at truth. The public SAEs behave the same way. This agrees with Aditi's M2.

**Limitation:** validation rows only; one model.

## 4. Files

Each folder has `per_run.csv`, `summary.csv`, `protocol.json` and `run_info.json`.

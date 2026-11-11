# Public SAE splicing reference (roadmap G9, P3) (2026-11-10). Status: COMPLETED

Evidence level: validation finding; external reference only. The public SAEs are **not sparsity-matched** to the in-house SAEs: they have a different width and a different L0.

Script: `scripts/shared/public_sae_spliced_evaluation.py --saes public_relu|public_topk`. It reuses the Phase-0 splicing code unchanged (`results/openwebtext/spliced_evaluation/`).
* `public_relu`: `gpt2-small-res-jb`, layers 0–11, L0 13–74.
* `public_topk`: `gpt2-small-resid-post-v5-32k`, k = 32, layers 1–11 (resid_post of layer L−1 is resid_pre of layer L).

## 1. Design

Each SAE is spliced into GPT-2 small, on OpenWebText validation text. The script reports:
* the cross-entropy increase;
* loss recovered compared with mean ablation;
* KL;
* top-1 agreement;
* the same numbers for a norm-matched random error.

## 2. Results

* **Loss recovered:** public ReLU keeps 94.6–99.8 %, and public TopK 95.2–99.9 %. That is the same range as the in-house SAEs (F7).
* **Public TopK with 32 latents matches in-house TopK with 128:** at layer 6 the cross-entropy increase is 0.103 vs 0.104, and loss recovered is about 98 % for both. Public ReLU lies between in-house ReLU and in-house TopK/JumpReLU.
* **SAE error vs random error:** at every layer, the public SAE error raises cross-entropy more than random error of the same size. The ratio is 1.4–2.4 for public TopK and 2.2–4.8 for public ReLU; the in-house SAEs are at about 1.3–2.4 at the middle and later layers.
* Figure: `results/shared/followup_figures/F7_loss_recovered_public_vs_inhouse.png`.

## 3. Interpretation

* Our SAEs are not unusually poor.
* "SAE error hurts the model more than random error of the same size" is a general property of SAEs, not a flaw of our training.

## 4. Files

`public_relu/` and `public_topk/` each have `per_run.csv`, `layer_controls.csv` and `run_info.json`. `public_relu/smoke/` holds the smoke run.

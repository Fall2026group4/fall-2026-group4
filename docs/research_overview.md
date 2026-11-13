# Research overview: SAE faithfulness in GPT-2 Small

**Scope of this document:** the GPT-2 Small SAE work on branch `sae-research`. Gemma-2-2B is owned by a teammate; on
this branch it appears only as an SAE-free steering appendix.

## 1. Research question

> Can we improve sparse-autoencoder (SAE) reconstruction of frozen LLM activations while preserving truth-related
> information and causal behaviour across transformer layers?

Answering it requires measuring **four levels separately**, because they can disagree:

1. **Reconstruction:** how close x̂ is to x (NMSE, EV, cosine), at matched sparsity (L0).
2. **Behaviour:** what happens to GPT-2's next-token predictions when x̂ replaces x in the forward pass (ΔCE, KL).
3. **Information:** whether truth (Geometry of Truth) and persona/honesty (RepE-LM) are still readable from x̂, by a
   frozen original probe and by a re-fit probe, against PCA and noise references.
4. **Causal:** what share of a prompt's causal effect on the output passes through x̂ (activation-patching
   decomposition).

Steering (adding directions to activations) is a fifth, separate category and is never used as evidence for the
other four.

## 2. Why SAE faithfulness matters

* SAEs are used to explain models: "feature 974 = Paris" (our Phase 1).
* An explanation is only valid if the SAE keeps what the model actually uses.
* Most published SAE evaluations report reconstruction and sparsity only.
* A reconstruction can be close in variance-weighted terms (EV 0.9) and still change behaviour and lose low-variance,
  task-relevant directions. That is exactly what we found.

## 3. Why GPT-2 Small

* It is cheap enough to train and evaluate more than 100 SAEs: 12 layers × 3 architectures × 3 seeds, plus
  improvement runs.
* It has public SAEs (`gpt2-small-res-jb`) to use as an external reference.
* Residual stream width is 768; there are 12 layers. All SAEs sit at `blocks.L.hook_resid_pre`, with BOS excluded.

## 4. SAE architectures

| Architecture | Sparsity mechanism | Where used |
|---|---|---|
| ReLU | L1 penalty with an L0 controller | old (3,072) and R2 (12,288) |
| TopK | exactly k = 128 active latents | old and R2 (+ AuxK in R2) |
| JumpReLU | learned per-latent threshold (sigmoid surrogate in our old SAEs) | old only; a modern STE version was abandoned |
| BatchTopK | top k·B over a batch; global threshold at inference | R2 |
| Random frozen decoder TopK | decoder never trained | R2 control (n = 1) |

Implementations:
* old: `scripts/openwebtext/03_training_sae.py`;
* R2: `src/sae/modern_sae.py`.

## 5. Datasets

| Dataset | Use | Split |
|---|---|---|
| OpenWebText (OWT) | SAE training; reconstruction and behaviour evaluation | old: 800/100/100 documents (101,600 / 12,700 / 12,700 tokens); R2: 100M tokens from parquet shard 78, our sample excluded |
| Geometry of Truth (GoT) cities (+ neg_cities) | truth information | grouped by city, 70/15/15; GoT-train A/B groups for teacher vs evaluation probes |
| RepE-LM | persona/honesty information and causal effect | one-token honest/untruthful swaps in fixed templates; train/validation only (no test split yet) |
| Old RepE persona prompts | historical | length-confounded; HISTORICAL |

## 6. Experimental objectives (in order)

1. A fair baseline: optimizer-consistent, sparsity-matched SAEs at every layer.
2. Measure all four levels for those SAEs, with references (noise, PCA, random decoder, supervised DAS skyline).
3. Try to improve faithfulness:
   * fine-tuning (reconstruction, truth-aware, KL-aware);
   * then data-scaled retraining (R2) and a fixed-sparsity KL fine-tune (R4).
4. Confirm the main comparisons once on held-out test data (M5).

## 7. Evaluation methods

See [`methodology.md`](methodology.md):
* one evaluation harness for every SAE (`scripts/shared/sae_ladder_evaluation.py`);
* pre-registered decision rules with file hashes;
* frozen vs re-fit probes;
* activation splicing;
* the M4 causal decomposition.

## 8. Main verified findings

Full tables: [`results_summary.md`](results_summary.md).

1. **Reconstruction and behaviour can be improved substantially (TEST).**
   * Data-scaled R2 SAEs cut OWT NMSE by 30.8–73.7 % and spliced ΔCE by 35.5–78.8 % at layers 1/8/11 (M5 comparisons C1–C3, every seed pair).
   * A fixed-sparsity KL fine-tune (R4) cut ΔCE by a further 16–17 %.
2. **Truth information did not follow (TEST).**
   * GoT re-fit gaps show no clear change in 34/34 cells.
   * No trained SAE beats PCA at equal error, and a random frozen decoder ties them on that metric.
3. **Persona representation and causal preservation improved on validation only.**
   * R2 TopK at L8: persona gap 0.156 → 0.108; T_rec/T_full 0.63 → 0.73.
   * There is no gain relative to PCA.
   * R4's behavioural gain came with a causal regression at L11.
4. **The levels dissociate.** Reconstruction predicts behavioural damage within a layer (M0), but not truth
   preservation. Even the best SAE carries less of the persona effect than a rank-4 supervised subspace (R5).

## 9. Navigation

* [`experiment_history.md`](experiment_history.md): everything tried, in order, with verdicts.
* [`results_summary.md`](results_summary.md): verified numbers by dataset.
* [`methodology.md`](methodology.md): pipelines, metrics, corrections, statistics.
* [`limitations_and_future_work.md`](limitations_and_future_work.md): open issues and proposals.
* [`results_registry.md`](results_registry.md): citation status of every result.
* [`RESEARCH_PROGRESS.md`](RESEARCH_PROGRESS.md), [`RESEARCH_DECISIONS.md`](RESEARCH_DECISIONS.md),
  [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md): journal, decision log, task status.

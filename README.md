# SAE Faithfulness: reconstruction, truth and causal behaviour across transformer layers

**Team:** Aditi Shukla, Dhruv Rai, Simbanegavi Simbarashe. GWU Data Science capstone, Fall 2026.

**Research question.** Can we improve sparse-autoencoder (SAE) reconstruction of frozen LLM activations while
preserving truth-related information and causal behaviour across transformer layers?

## Motivation

SAEs decompose a language model's dense activations into sparse, nominally interpretable features. Interpretability
work often assumes the SAE is a faithful stand-in for the model.

That assumption has three separable parts:
1. the SAE reconstructs the activation well (reconstruction);
2. information the model encodes, such as whether a statement is true or whether a persona is honest, survives the
   reconstruction (representation preservation);
3. the model still behaves the same when the reconstruction replaces the activation, and SAE features act as causal
   variables (behavioural and causal faithfulness).

This project measures all three, **separately**, for three SAE architectures at every layer of GPT-2 Small. It then
tests whether targeted fine-tuning can improve them.

The original proposal (`proposal.md`) asked whether SAE features are causally faithful to their auto-generated
labels. Professor reviews and our own audits narrowed this into the question above. The history is in
`docs/RESEARCH_PROGRESS.md`.

## Scope

| | |
|---|---|
| **Models** | GPT-2 Small (12 layers; all SAE work) and Gemma-2-2B (26 layers; steering only, no SAEs yet) |
| **SAEs** | ReLU, TopK and JumpReLU (sigmoid-surrogate estimator). All trained in house on OpenWebText: width 3072 (4×), target L0 = 128, one uninterrupted 15k-step Adam run, seeds 42/1042/2042, at `blocks.L.hook_resid_pre`, BOS excluded |
| **Datasets** | **OpenWebText** (reconstruction, behaviour); **Geometry of Truth** (cities/neg_cities, larger_than/smaller_than; truth); **RepE honesty** (honest vs untruthful persona prompts; honesty) |
| **Layers** | All 12 GPT-2 layers for the baselines, frozen probes and splicing. Layer 6 for fine-tuning and causal tests |

### Evaluation categories (never used as evidence for each other)

| Category | Metrics | Direction |
|---|---|---|
| Reconstruction | MSE, explained variance (EV = 1 − NMSE), cosine | lower MSE; higher EV and cosine |
| Sparsity | L0, reported separately on OWT, GoT and RepE; dead-latent fraction | match target; lower dead |
| Representation preservation | AUROC gap of probes fitted on **original** activations and applied **frozen** to reconstructions; direction cosine | lower gap; higher cosine |
| Behavioural faithfulness | Activation splicing: CE increase, output KL, truth preference, persona-shift cosine | lower CE/KL; higher others |
| Causal interventions | Pre-registered tests with 200 random controls and a positive control | effect vs controls |
| Steering | Adding directions to activations (Gemma; GPT-2 historical) | sign-consistent effect vs controls |

**Rules:**
* Architectures are compared only at matched sparsity (within ±10 % of the target L0; fine-tunes within ±5 % of their base).
* Test splits are used only by pre-registered final evaluations.
* Every result is labelled EXPLORATORY, VALIDATION, TEST or INCONCLUSIVE.

## Main findings so far

**Status:** TEST = confirmed on a held-out split; VALIDATION = validation split only. Full tables are in
`docs/EXPERIMENT_RESULTS_SUMMARY.md`; which files may be cited is in `docs/results_registry.md`.

1. **Reconstruction gets worse with depth at matched L0** (TEST, 108 runs).
   * OWT test EV falls from about 0.93–0.95 at layers 0–2 to 0.76 (ReLU) and 0.815 (TopK, JumpReLU) at layer 11.
   * JumpReLU leads at layers 3–9; ReLU degrades fastest with depth.
   * 107/108 runs are sparsity-matched. Layer-0 JumpReLU seed 42 is not, and is excluded from rankings.
2. **Reconstruction quality does not predict truth preservation** (TEST).
   * The frozen GoT logistic probe drops to near chance on reconstructions at every informative layer (geographic gap 0.19–0.32).
   * A probe *re-trained* on reconstructions recovers most of the signal, so truth information is displaced rather than erased.
3. **Reconstruction quality does track honesty preservation** (TEST).
   * The RepE honesty gap grows with depth and is largest for ReLU (logistic gap ≈ 0.24 at layer 6 vs ≈ 0.09 for TopK/JumpReLU).
   * Spearman ρ = 0.86 between the honesty gap and OWT NMSE.
4. **SAE errors damage behaviour more than random errors of the same norm** (VALIDATION).
   * OWT CE increase is 1.0–2.6× that of norm-matched random errors.
   * The persona shift is preserved far better than under random errors.
5. **No causal effect from SAE-derived directions at layer 6** (TEST, pre-registered).
   * The raw honesty direction has a tiny but significant effect (E = 0.005, p = 0.01 vs 200 random directions).
   * Directions and features derived from the SAEs are indistinguishable from controls.
   * The activation-patch positive control recovers E = 0.23.
6. **Fine-tuning at layer 6** (VALIDATION, pre-registered criteria):
   * A truth-aware objective that distilled a linear probe turned out to be **circular**: the evaluation probe was the training teacher. Corrected, it does not beat reconstruction-only fine-tuning.
   * GoT reconstruction fine-tuning with OWT replay (Method A) lowers the independent MLP-probe gap on 9/9 seeds, but misses the pre-registered 2-sd criterion.
   * Output-KL objectives (E2E-inspired) improve behavioural faithfulness for TopK, not truth readouts. For ReLU and JumpReLU they push L0 off target.
   * The KL-weight ablation is inconclusive.
7. **Gemma-2-2B steering** (VALIDATION + TEST, descriptive, 20 controls).
   * The earlier "both signs push the answer to TRUE" result is a generic large-perturbation effect.
   * A context-matched direction steers with the correct sign at every strength.

## Known limitations and pending work

**Limitations:**
* The SAEs are small (3072 latents) and trained on 101,600 OWT tokens. They are not converged, since validation EV was still rising at 15k steps.
* JumpReLU uses a sigmoid surrogate, not the published estimator.
* Fine-tuning and causal tests exist only at layer 6, with 3 seeds. Several close calls need more seeds.
* L0 is matched on OWT, but not on GoT/RepE prompts (RepE: ReLU ≈ 164, TopK 128, JumpReLU ≈ 180 at layer 6).
* GPT-2 Small's honesty behaviour is weak: the persona effect is about 0.02 nats at the final token.
* Gemma-2-2B has no SAE results, because Gemma Scope weights are not cached.
* One RepE validation statement (id 326) near-duplicates a train statement. It is excluded from new RepE work.

**Pending** (`docs/RESEARCH_ROADMAP.md`):
* converged EV-vs-L0 frontier;
* width sweep;
* SAE-in-the-loop feature interventions;
* RepE honesty-preservation fine-tuning (planned as experiments R1/R2);
* replication with more seeds;
* Gemma SAE baselines;
* the final pre-registered test evaluation;
* the final report.

## Repository layout

```
scripts/{openwebtext,geometry_of_truth,repe,gemma}/NN_name.py   pipelines, numbered in run order
scripts/shared/                                                cross-dataset evaluation, splicing, registry, manifest
src/evaluation/faithfulness_metrics.py                         shared metric definitions (use these)
src/sae/checkpoint_registry.py                                 checkpoint lookup and registry builder
src/tests/                                                     unit and integration tests
results/{openwebtext,geometry_of_truth,repe,gemma,shared}/     outputs; each folder has a README with result status
data/                                                          small datasets and splits; see data/README.md
docs/                                                          research journal, decisions, roadmap, registries
reports/, presentation/, research_paper/                       capstone deliverables
```

**Start with:**
* `docs/RESEARCH_PROGRESS.md` (history);
* `docs/results_registry.md` (what may be cited);
* `docs/EXPERIMENT_RESULTS_SUMMARY.md` (numbers);
* `docs/RESEARCH_DECISIONS.md` (why);
* `docs/EXPERIMENT_CHANGELOG.md` (audit trail, including every methodological correction).

The 2026-11-04 professor-review corrections are integrated into the dataset folders above. The review report, code
audit and methodology changes are in `docs/2026-11-04_*.md`, and `docs/migration_manifest.csv` maps every old path to
its new one.

**Not in Git**, and kept only on the training machine:
* SAE checkpoints (`*.pt`, about 9 GB);
* activation caches (`data/activations/`, `data/*/activations/`);
* logs.

`docs/checkpoint_registry.csv` lists every checkpoint with its SHA-256, so a restored checkpoint can be verified.

## Reproducing the results

**Environment:**
* Python 3.14, PyTorch 2.14 (CUDA 13), TransformerLens; `pip install -r requirements.txt`.
* Experiments ran on an AWS g5 instance (NVIDIA A10G, 23 GB). Gemma-2-2B uses HF transformers in bf16.
* Run everything from the repository root:

```bash
python -m pytest -q src/tests                                           # 25 tests
```

**OpenWebText: data, activations, the 108-run baseline, analysis:**

```bash
python scripts/openwebtext/01_prepare_dataset.py
python scripts/openwebtext/02_extract_activations.py
python scripts/openwebtext/13_consistent_protocol_retrain.py --layers 0 1 2 3 4 5 6 7 8 9 10 11
python scripts/openwebtext/14_sparsity_matching_analysis.py
python scripts/openwebtext/15_all_layer_baseline_analysis.py
python scripts/openwebtext/16_p1_sparsity_control_pilot.py              # P1 sparsity-control pilot
python scripts/shared/build_checkpoint_registry.py
```

**Truth and honesty preservation, all layers (frozen probes), and splicing:**

```bash
python scripts/geometry_of_truth/01_extract_got_activations.py && python scripts/geometry_of_truth/02_create_grouped_splits.py
python scripts/repe/01_prepare_dataset.py && python scripts/repe/02_extract_activations.py
python scripts/shared/all_layer_frozen_probe_evaluation.py --dataset got     # then --dataset repe
python scripts/shared/all_layer_frozen_probe_analysis.py --dataset got      # then --dataset repe
python scripts/shared/spliced_reconstruction_evaluation.py                  # Phase 0; --bridge for the layer-6 bridge
python scripts/shared/spliced_reconstruction_analysis.py
```

**Causal test, layer-6 fine-tuning experiments and Gemma steering.** Each fine-tuning script runs in stages and
refuses to overwrite:

```bash
python scripts/repe/13_causal_random_controls.py --phase validation      # then --phase test (pre-registered)
python scripts/geometry_of_truth/11_truth_aware_pilot.py --prepare|--dry-run|--train ; python scripts/geometry_of_truth/12_truth_aware_pilot_evaluation.py
python scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py --prepare|--dry-run|--train ; python scripts/geometry_of_truth/15_method_a_evaluation.py
python scripts/geometry_of_truth/17_e2e_inspired_finetune.py --prepare|--dry-run|--train ; python scripts/geometry_of_truth/18_e2e_inspired_evaluation.py
python scripts/geometry_of_truth/20_kl_weight_ablation.py --verify|--prepare|--dry-run|--train ; python scripts/geometry_of_truth/21_kl_weight_ablation_evaluation.py
python scripts/gemma/01_steering_sweep.py                                  # --smoke first
```

Each results folder's `README.md` and `preregistration.json` record the exact seeds, hyperparameters, script hashes
and checkpoint hashes used. `docs/experiment_manifest.csv` has one row per experiment.

## Licence

MIT (`LICENSE`).

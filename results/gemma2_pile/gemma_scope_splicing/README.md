# Gemma Scope SAE reconstruction and splicing on The Pile (2026-11-12). Status: PLANNED (pre-registered)

**Owner:** Simbanegavi Simbarashe (branch `simba-gemma-pile`). **Roadmap:** M6 (part of M3).
**Evidence level on completion:** VALIDATION (descriptive). The Pile *validation* file is used; the Pile test file is never downloaded.

This file is written **before any code exists or any run is made**. Its Git commit is the time stamp of the
pre-registration. Section 2 will be filled in after the run; nothing in Section 1 is edited afterwards (any
deviation is recorded in Section 2 under "Deviations").

## 1. Before implementation

### Research question
How well do pretrained Gemma Scope residual-stream SAEs reconstruct Gemma-2-2B activations on The Pile, and how much
does Gemma's next-token behaviour change when those reconstructions replace the original activations?

### Why it is needed, and how it connects to the main question
The main question asks whether SAE reconstruction can be improved *while preserving* information and behaviour. All SAE
evidence so far is from GPT-2 Small with in-house SAEs (`RESEARCH_DECISIONS.md` D1, D3). This experiment establishes
the reconstruction and behavioural-faithfulness baseline for a second, larger model with published SAEs, on the
data distribution chosen for Gemma in the proposal (The Pile). It keeps the project's separation of categories:
reconstruction quality and behavioural faithfulness (splicing) are measured and reported **separately**; neither is
evidence for the other. No representation-preservation (probe) or causal-intervention claim is made here.

It is **not** a new improvement method, and it adds no new scientific objective beyond the agreed scope
(Pile → Gemma activations → Gemma Scope SAE reconstruction → splicing evaluation).

### Hypothesis
Mostly descriptive (see "Pre-registered questions"). One directional replication hypothesis, **H-G1**, carried over
from the GPT-2 Phase 0 finding that SAE errors hurt behaviour more than random errors of the same norm
(`RESEARCH_PROGRESS.md` §10).

### Pipeline (one pass per document, one layer spliced at a time)

```
Pile validation sample (512 eval + 64 reference docs, seeded, disjoint)
 → Gemma tokenization (BOS + text, first 1024 tokens)
 → Gemma-2-2B forward pass (HF transformers, bf16, eager attention)
 → extract resid_post at layers 5, 12, 19   (output of model.model.layers[L] = TL blocks.L.hook_resid_post)
 → corresponding Gemma Scope width-16k residual SAE (pinned IDs below)
 → encode/decode reconstruction (fp32)
 → reconstruction metrics (EV, NMSE, cosine, relative error, L0)
 → splice the reconstruction back into Gemma at that layer (all non-BOS, non-pad positions)
 → behavioural/output change (ΔCE, KL, top-1 agreement, loss recovered)
 → compare with controls (norm-matched random error, mean ablation, identity, repeat pass)
 → save per-document and per-layer outputs plus aggregates (overall and per Pile source)
```

### Design

| Item | Choice | Reason / source |
|---|---|---|
| Model | `google/gemma-2-2b` (26 layers, d_model 2304), HF transformers, bf16, eager attention | Same loading as `scripts/gemma/01_steering_sweep.py`; TransformerLens was OOM-killed on the earlier host (D1). HF model revision recorded at run time **[superseded by Amendment 1: pinned to `c5ebcd40d208330abc697524c919956e692655cf`]** |
| Hook | Output hidden state of `model.model.layers[L]`, i.e. `blocks.L.hook_resid_post` | The site the Gemma Scope residual SAEs were trained on: the sae-lens loader sets `hook_name = blocks.{layer}.hook_resid_post` for `gemma-scope-*-res` repos (`sae_lens/loading/pretrained_sae_loaders.py`, `get_gemma_2_config_from_hf`). **Differs from the GPT-2 work (`hook_resid_pre`)**; this is a convention of each SAE family, not a mapping |
| Layers | **5, 12, 19** | Representative lower / middle / later residual-stream depths of the 26-layer model. In the release these are the layers that also carry the largest-width (1M) residual SAEs; neighbouring layers (4, 6, 20) have only 16k and 65k, and layer 12 has the full width sweep 16k–1M (HF repo listing, checked 2026-11-12). Width 16k exists at all three and is used throughout. **They are not equivalents of GPT-2 layers 5/6/9**; no layer correspondence between the models is claimed. The GPT-2 decision to drop layer 3 (insufficient GoT truth signal) is specific to that probe experiment and does not apply here |
| SAE release | Hugging Face `google/gemma-scope-2b-pt-res`, revision `fd571b47c1c64851e9b1989792367b9babb4af63`; sae-lens release `gemma-scope-2b-pt-res` | Pretrained JumpReLU SAEs; d_sae = 16 384 |
| SAE IDs | `layer_5/width_16k/average_l0_68`, `layer_12/width_16k/average_l0_82`, `layer_19/width_16k/average_l0_73` | These are exactly the paths that sae-lens maps `gemma-scope-2b-pt-res-canonical` / `layer_{5,12,19}/width_16k/canonical` to (registry `sae_lens/pretrained_saes.yaml` at SAELens commit `a7f07a96ecd07a38f9d719a6a8650b983954d062`). The explicit paths are pinned so a later change of the "canonical" alias cannot silently change the SAE. All three exist in the HF repo (checked 2026-11-12) |
| SAE forward | f = JumpReLU_θ(W_enc x + b_enc); x̂ = W_dec f + b_dec; `apply_b_dec_to_input = False`, no activation normalisation (sae-lens config); computed in fp32 on bf16 activations cast to fp32 | As released |
| Data source | Hugging Face dataset `monology/pile-uncopyrighted`, file `val.jsonl.zst`, revision `3be90335b66f24456a5d6659d9c8d208c0357119`, 338 045 152 bytes, SHA-256 `db5e5d1532bf8dc33a6589b50ecba1a8c96f7b4b9cb343d168e603c393007c26` (Git-LFS object id) | The Pile validation split with copyrighted subsets removed. The EDA (`simba-gemma-pile-eda`, `5638350`) used the parquet mirror `monology/pile-uncopyrighted-parquet`, which has **only a train split** (checked 2026-11-12), so the evaluation draws from the original repo's validation file instead. Same uncopyrighted filtering, same source labels (`meta.pile_set_name`) |
| Sampling | Documents indexed by 0-based line number in `val.jsonl.zst`. `numpy.random.default_rng(42).permutation(n_lines)`; walk the permutation, tokenize, keep a document if it has ≥ 32 text tokens; the first 512 kept are the **evaluation set**, the next 64 kept are the **reference set** | Deterministic, seeded, recorded; eval and reference are disjoint by construction (asserted). Natural source mixture (no stratification), so per-source cells can be small and are flagged |
| Tokenization / chunking | Gemma tokenizer with BOS (revision pinned by Amendment 1); truncate to the **first 1024 tokens** (BOS included); one window per document, no packing across documents | 1024 is the Gemma Scope training context length recorded by sae-lens (`context_size: 1024`, `prepend_bos: True`). One window per document keeps documents independent for the bootstrap and keeps each window's source label clean. The EDA found 26 % of sampled docs exceed 1024 tokens; for those only the first 1024 tokens are used (recorded per document) |
| Positions | BOS is never reconstructed, never spliced and excluded from all metrics; padding excluded | As for GPT-2 (D4). Behaviour is measured on predictions made at positions 1 ≤ t ≤ len − 2 (same as `scripts/shared/spliced_reconstruction_evaluation.py`) |
| Seeds | Sampling 42; random-error directions `1000 + L`; bootstrap 0 | No training, so no training seeds. Pretrained SAEs give one SAE per layer: variability is reported as bootstrap CIs over documents, **not** as seed variability, and is not comparable to the GPT-2 3-seed sd |
| Training budget / hyperparameters | None (inference only) | |
| Bootstrap | 1000 resamples of documents, 95 % percentile intervals | |

### Data provenance relative to the earlier Pile EDA (approved deviation, 2026-11-12)
* **The earlier EDA** (`simba-gemma-pile-eda`, commits `d23a158`…`5638350`) streamed 200 documents from the
  **train** split of the parquet mirror `monology/pile-uncopyrighted-parquet` (`shuffle(seed=42, buffer_size=500)`).
  That mirror has no validation split.
* **M6 uses a different file:** the original repository's validation file
  `monology/pile-uncopyrighted/val.jsonl.zst`, revision `3be90335b66f24456a5d6659d9c8d208c0357119`, SHA-256
  `db5e5d1532bf8dc33a6589b50ecba1a8c96f7b4b9cb343d168e603c393007c26`. Same uncopyrighted filtering and source labels.
* **Held out from the EDA workflow:** the EDA never read the validation file, so no M6 document was inspected during
  the EDA. The EDA's length and truncation statistics (train-side) informed only the chunking policy.
* **Exact selection is reproducible:** seed 42, `numpy.random.default_rng(42).permutation(n_lines)` over 0-based line
  numbers, the ≥ 32-text-token eligibility rule, then first 512 eligible = evaluation and next 64 eligible = reference.
  Line numbers and SHA-256 of each document's text and token ids are saved in `sample_manifest.csv`.
* **Disjointness:** the 64 reference documents are disjoint from the 512 evaluation documents by construction, and this
  is asserted by the script (gate V1).

### Conditions (per layer L ∈ {5, 12, 19}; only layer L is modified in a pass)

| Condition | Activation written at layer L (non-BOS, non-pad positions) | Role |
|---|---|---|
| Original | unchanged; run twice | Reference outputs; repeat pass = determinism control |
| **SAE splice** | x̂ = SAE(x) | The measurement |
| Norm-matched random error | x + ‖x − x̂‖ · u, u a random unit vector per token (isotropic Gaussian, normalised) | Is SAE error worse than *any* error of the same size? (H-G1) |
| Mean ablation | the mean of layer-L activations over the 64-document reference set (non-BOS tokens) | Denominator of loss recovered |
| Identity | x written back unchanged through the hook | Hook-plumbing control |

### Metrics

| Metric | Category | Direction | Definition |
|---|---|---|---|
| Explained variance EV (NMSE = 1 − EV) | reconstruction | EV higher / NMSE lower | 1 − Σ‖x − x̂‖² / Σ‖x − x̄‖², x̄ = mean over the evaluated tokens of that layer; same formula as `src/evaluation/faithfulness_metrics.reconstruction_metrics` |
| Cosine(x, x̂) | reconstruction | higher | mean over tokens |
| Relative squared error | reconstruction | lower | Σ‖x − x̂‖² / Σ‖x‖² |
| L0 | sparsity (reported) | reported, not targeted | mean number of active latents per token on our sample. The `average_l0` in the SAE name is the release's own figure on its own data and is reported alongside |
| ΔCE = CE_spliced − CE_original | behavioural splicing | lower | next-token cross-entropy, nats per prediction |
| KL(P_original ‖ P_spliced) | behavioural splicing | lower | mean per prediction |
| Top-1 agreement | behavioural splicing | higher | share of predictions whose argmax is unchanged |
| Loss recovered | behavioural splicing | higher | (CE_mean-abl − CE_SAE) / (CE_mean-abl − CE_original) |
| ΔCE_SAE / ΔCE_random | behavioural splicing | reported (>1: SAE error more harmful than random) | ratio of the two token-weighted means |

All metrics are reported per layer, token-weighted over the whole evaluation set **and** per Pile source
(`pile_set_name`), each with a document-level bootstrap CI. Sources with fewer than 20 evaluation documents are flagged
LOW-N and are not interpreted.

### Pre-registered questions and criteria

* **Q1 (descriptive):** EV, cosine, relative error and L0 of each SAE on The Pile, overall and per source.
* **Q2 (descriptive):** ΔCE, KL, top-1 agreement and loss recovered when splicing each SAE, overall and per source.
* **Q3 (descriptive):** how much reconstruction and splicing metrics vary across Pile sources at each layer.
* **H-G1 (directional replication of GPT-2 Phase 0):** at layer L, SAE-splice ΔCE exceeds norm-matched-random ΔCE.
  Test: paired over documents, difference d = mean ΔCE_SAE − mean ΔCE_random per document, bootstrap CI over
  documents. **Supported at layer L** if the 95 % CI lower bound of mean d is > 0; **contradicted** if the upper bound
  is < 0; otherwise **inconclusive**. Reported for each of the three layers separately; a Bonferroni (98.33 %) CI is
  reported as a sensitivity check. The outcome is reported whichever way it falls.
* **Not claimed:** any ranking of layers as "better", any GPT-2 vs Gemma comparison beyond a descriptive side-by-side
  (different model, SAE family, width, L0, hook point and context length; roadmap X2).

### Validity gates (checked before results are read; a failed gate stops the run and is reported)

| Gate | Pass condition |
|---|---|
| V1 sample integrity | `val.jsonl.zst` SHA-256 equals the value above; 512 + 64 documents; eval ∩ reference = ∅; sample-manifest hash recorded |
| V2 hook equivalence | activation captured by the hook equals `output_hidden_states[L + 1]` exactly (max abs diff 0) at L = 5, 12, 19 |
| V3 determinism and identity | repeat original pass and identity splice: mean KL ≤ 1e-6 nats per prediction at every layer. **This threshold is fixed.** If bf16/GPU numerics make it unattainable in the smoke run, the observed values are reported to the user first; any change is recorded in Section 2 as a protocol deviation (old value, new value, observed values, reason) **before** any full-run result is interpreted |
| V4 SAE sanity (smoke run) | measured L0 within a factor of 2 of the release's `average_l0`, and EV > 0, at every layer. A failure points to a loading, hook or scaling mismatch; it is diagnosed and reported, not tuned away |
| V5 no silent substitution | the three pinned SAE IDs load; if any is unavailable the run stops and the user is asked. No other width or L0 is substituted |

### Leakage-prevention checks
* No training, no probe fitting, no checkpoint selection, so no train/test leakage path through this experiment.
* The Pile test file (`test.jsonl.zst`) is never downloaded or read.
* Eval and reference sets are disjoint (V1); the mean-ablation vector uses only the reference set.
* **Known limitation, not a leak in our pipeline:** Gemma-2's pretraining data are not public, so Gemma may have
  seen these documents; this affects absolute CE, not the paired spliced-vs-original comparisons. sae-lens lists
  `monology/pile-uncopyrighted` as the `dataset_path` in its Gemma Scope config; we have **not verified** whether
  Gemma Scope was trained on The Pile. If it was, evaluating on the validation file is still held out from its
  training split.

### Comparisons

| X vs Y | What it establishes |
|---|---|
| SAE splice vs original | Behavioural cost of replacing the activation by the SAE reconstruction (Q2) |
| SAE splice vs norm-matched random error | Whether SAE error is more or less harmful than an equally large random error (H-G1) |
| SAE splice vs mean ablation | Loss recovered: how much of the information removed by ablation the SAE restores |
| Identity / repeat pass vs original | Pipeline correctness (V3), not a result |
| Layer 5 vs 12 vs 19 | Descriptive variation with depth; no ranking claim |
| Pile source vs Pile source | Descriptive heterogeneity (Q3) |

### Outputs (`results/gemma2_pile/gemma_scope_splicing/`)
* `preregistration.json`: machine-readable copy of the fixed design (read by the scripts; hashed at run time).
* `sample_manifest.csv`: per document: set (eval/reference), line index, `pile_set_name`, characters, SHA-256 of the
  UTF-8 text, full token count, tokens used, truncated flag, SHA-256 of the token ids used.
* `per_doc_metrics.csv`: per document × layer × condition: token counts, reconstruction sums, CE, KL, top-1.
* `summary_by_layer.csv`, `summary_by_layer_source.csv`, `controls.csv`, `h_g1.csv`.
* `run_info.json`: script, manifest and pre-registration SHA-256; package versions; HF model and SAE revisions; GPU;
  runtime; peak memory.
* `figures/`, and `smoke/` (labelled SMOKE: pipeline check, not a result).
* No activation caches are written; SAE weights and model weights stay in the Hugging Face cache and are never committed.

### Files to be created (later commits), and why
* `scripts/gemma/02_pile_sample_manifest.py`: download and verify `val.jsonl.zst`, build `sample_manifest.csv` (CPU; tokenizer only).
* `scripts/gemma/03_pile_gemma_scope_splicing.py`: model + SAEs + reconstruction + splicing + controls; `--smoke`; refuses to overwrite outputs.
* `scripts/gemma/04_pile_gemma_scope_figures.py`: figures from the saved CSVs only.
* `src/tests/test_gemma_pile_splicing.py`: chunking/truncation, JumpReLU forward on toy weights, splice hook leaves BOS and padding untouched, identity splice gives KL = 0 on a tiny random model (no downloads).

Reused unchanged: the HF loading/hook pattern of `scripts/gemma/01_steering_sweep.py`; metric definitions of
`src/evaluation/faithfulness_metrics.py` and `scripts/shared/spliced_reconstruction_evaluation.py`;
`scripts/shared/write_manifest.py`. No code is taken from other branches.

### Pre-run hashes
External inputs are pinned above (dataset file SHA-256 and revision, SAE repo revision and IDs, sae-lens registry commit).
Script hashes do not exist yet; they are recorded in `run_info.json` by the scripts themselves, together with the
SHA-256 of this README and `preregistration.json`, and the smoke run's values are copied into Section 2 before the
full run.

### Compute (not yet confirmed; nothing is downloaded before the Ronin hardware check)
* Forward passes: 2 original + 4 per layer × 3 layers = 14 passes over ≤ 512 × 1024 tokens.
* Downloads: `val.jsonl.zst` 338 MB; three SAEs ≈ 2 × 2304 × 16 384 × 4 bytes ≈ 0.3 GB each (encoder + decoder, fp32); the
  Gemma-2-2B weights (size to be read from the HF repo on Ronin).
* GPU model, memory, disk and Hugging Face Gemma access: **unconfirmed**; to be checked on Ronin before the smoke run.

### Stretch goal (not part of this pre-registration)
Gemma GoT frozen-probe preservation (roadmap M1/M2) is a separate experiment, started only after this one is complete,
documented and committed, and only if it does not threaten the 2026-11-22 technical freeze. It would get its own
pre-registration.

## Amendments to the pre-registration (made before any run)

### Amendment 1 (2026-11-12): tokenizer and model revision pinned
* **Change:** `tokenization.revision` added and `model.revision` changed from `record_at_run_time` to the Hugging Face
  commit **`c5ebcd40d208330abc697524c919956e692655cf`** of `google/gemma-2-2b` (the repository's current commit;
  last modified 2024-08-07; read from the public model metadata on 2026-11-12). Tokenizer and weights come from the
  same repository commit.
* **Reason:** the original text recorded the revision only at run time, so the pre-registration did not freeze it.
* **Rule:** scripts load exactly this revision (`revision=tokenization.revision` / `model.revision`) and never look up
  the latest revision. If this commit cannot be loaded, or a later run uses another one, it is recorded in Section 2
  as a protocol deviation before results are interpreted.
* **Fingerprints visible without login** (Git blob ids at that commit): `tokenizer_config.json`
  `2357fc1a512dc3881eb99b9df3ef8e97ef7ffd51`, `special_tokens_map.json` `8d6368f7e735fbe4781bf6e956b7c6ad0586df80`,
  `config.json` `9983200362411f3e251b4c8ce28afad7e3996c1e`. The `tokenizer.json`/`tokenizer.model` hashes need an
  authenticated (gated) request and will be recorded from Ronin in `sample_info.json`.
* **Not changed:** every other design element. No data had been downloaded and no code had been run.

## 2. After execution

### Post-pre-registration output additions and implementation details (approved 2026-11-12, before any 03 run)
These are evidence and reproducibility artifacts. They do **not** change the scientific design in Section 1.
* **`per_doc_sum_x.npz`** (full run only; ≈ 28 MB, float64): per document and layer, the sum of the evaluated
  activation vectors. Needed to recompute EV and its bootstrap intervals exactly (EV uses the evaluated set's own
  token mean). Committed for the successful full run; its SHA-256 is recorded in `run_info.json`. No smoke version is
  committed.
* **`gates.json`**: the outcome and values of V1–V5 (and, in the smoke run, the sae-lens diagnostic).
  On a gate failure `gates_FAILED.json` is written instead/additionally.
* **sae-lens cross-check** (smoke only): a **diagnostic, not a validity gate**. Relative tolerance `1e-4`
  (max |x̂_pinned − x̂_sae-lens| / max |x̂_sae-lens|), approved by the user. Records the sae-lens version, the pinned
  SAE revision, the cached revision(s) holding the files and the repository's `main` commit, and per-layer max
  absolute and relative differences. A difference beyond tolerance stops the smoke run for investigation; because
  sae-lens may load its default revision, it is not by itself a failure of the pinned loader.
* **Gate handling:** V1, V2, V5 stop immediately. V3/V4 stop the smoke run immediately; in the full run they are
  evaluated on the complete aggregate, and a failure writes `gates_FAILED.json`, marks every output table INVALID,
  keeps the outputs for diagnosis only, and exits with code 2.
* **Implementation constants:** batch size 1 (so no padding occurs in the real run), TF32 disabled, log-softmax in
  chunks of 128 positions, the last decoder layer is never hooked (its `hidden_states` entry includes the final
  norm). `--smoke` uses the first 8 evaluation and first 4 reference documents (by rank) of the committed sample,
  after V1 has checked the full 512 + 64 sample.
* **Precision:** bf16 as pre-registered. The available GPU (Tesla T4) has no native bf16; the smoke run reports
  seconds per document, peak GPU memory and the projected full-run time. A change of dtype would be a protocol
  deviation requiring the user's approval; the preferred response to an impractical runtime is an Ampere-class GPU
  with the same bf16 protocol.

### Engineering safeguards added after the pre-run review (2026-11-12, before any 03 run)
Engineering only: no change to layers, SAEs, thresholds, controls, data, metrics, hypotheses or criteria.
* **A1 numerics:** every activation, reconstruction, logits tensor, per-document sum and output column must be
  finite; otherwise `numerics_FAILED.json` and stop.
* **A2 code provenance:** the git commit and the dirty state of the code and inputs are recorded; the full run refuses
  to start from a dirty tree.
* **A3 determinism:** strict deterministic algorithms (`torch.use_deterministic_algorithms(True)`, no warn-only mode),
  cuDNN deterministic and benchmark off, with `CUBLAS_WORKSPACE_CONFIG` set before torch loads. An operation without
  a deterministic CUDA implementation raises an error and stops the run. Gate V3 remains the separate empirical
  identity/repeat check and is not a substitute for deterministic execution.
* **A4** captured activations are copies; **A5** `use_cache=False`; **A6** the loaded model and tokenizer commits are
  checked against the pin (`provenance_FAILED.json` on a mismatch; "unverifiable" if the library does not expose it).
* **A7 atomic outputs:** files are written to `.partial/` and moved into place only when complete; `run_info.json`
  (with the SHA-256 of every output) is moved last, so a folder without it is not a result.
* **A8** `--smoke --out-tag rerun` writes a second smoke run to `smoke_rerun/` for a byte-level reproducibility check.
* **A9** GPU compute capability, native bf16 support and the logits dtype are recorded.
* **A10** an INVALID run prints a "do not interpret" banner; a Hugging Face cache-scan error cannot crash the run.
* **A11** the test suite must pass with zero failures and zero skips (`python -m pytest -q -rs src/tests`).

### Known limitation: bf16 logit precision (decision H4a, 2026-11-12)
The model's logits are bf16, as pre-registered; CE, KL and top-1 are computed from them. No float32 readout is added
and the protocol is unchanged. The smoke run *measures* the limitation by recomputing the same output head in float32
from the head's own input, on the smoke documents only, and records the mean KL(fp32 ‖ bf16), the mean |ΔCE|, the
top-1 disagreement rate and the max |logit difference| in `run_info.json`
(`logit_precision_limitation_H4a`). These numbers are reported as a measurement floor for small ΔCE/KL effects and are
not used in any result.

### Results
*(To be completed.)*

## 3. If stopped early
*(To be completed if needed.)*

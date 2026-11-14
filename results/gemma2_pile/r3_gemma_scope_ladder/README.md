# R3: Gemma Scope width and L0 ladder on The Pile, with an A10G rerun of M6 (2026-11-13). Status: DRAFT PRE-REGISTRATION (committed for team review; not run)

**Owner:** Dhruv Rai. **Roadmap:** R3 (see "Scope relative to the roadmap" below). **Builds on:** M6
(`results/gemma2_pile/gemma_scope_splicing/`, branch `simba-gemma-m6-results`).
**Evidence level on completion:** VALIDATION. Q0, Q0b and Q1–Q3 are descriptive; H1–H3 are pre-registered directional tests. The
Pile *validation* file is used; the Pile test file is never downloaded.

This file is written **before any code exists or any run is made**. Its Git commit will be the time stamp of the
pre-registration. Section 2 will be filled in after the run. Nothing in Section 1 is edited afterwards; any deviation
is recorded in Section 2 under "Deviations".

## 1. Before implementation

### Research question
How do reconstruction quality and next-token behavioural faithfulness of the published Gemma Scope residual SAEs change
with **dictionary width** and **sparsity (L0)** at Gemma-2-2B layers 5, 12 and 19 on The Pile? And how much do M6's
numbers depend on the GPU's bf16 arithmetic (Tesla T4 with emulated bf16 vs A10G with native bf16), and which of
the two bf16 runs is closer to a full-fp32 reference?

### Why it is needed, and how it connects to the main question
* **Gemma has only one SAE per layer so far.** M6 measured one width-16k SAE per layer, at the release's canonical
  L0. The main question asks whether reconstruction can be improved *while preserving* behaviour. For Gemma, the only
  available lever without training is to move along the published width × L0 ladder. R3 measures that trade-off with
  the same data, metrics and controls as M6.
* **M6 was run on a T4, which has no native bf16.** Its bf16-logit precision floor (H4a: mean |CE difference| 0.026
  nats per prediction, from the smoke run) is of the same order as ΔCE_SAE (0.07–0.18). Re-running M6 *exactly* on an
  A10G (native bf16) measures how much the reported numbers depend on the hardware. This is needed before M6 and R3
  numbers are put side by side.
* **A difference alone does not say which run is closer to exact arithmetic.** A small fp32 reference (layer 12, M6's
  16k SAE, first 64 evaluation documents) gives both bf16 runs a common yardstick.
* **Categories stay separate.** Reconstruction quality and behavioural splicing are measured and reported separately;
  neither is evidence for the other. No representation-preservation (probe), causal-intervention or steering claim is
  made.

### Scope relative to the roadmap
* `RESEARCH_ROADMAP.md` (on `sae-research`) describes R3 as a "Gemma-2-2B ladder with Gemma Scope 16k SAEs at about
  3 layers and several published L0s". It also says "the RepE-LM prompts must be re-tokenised and the K1/K2 gates re-run
  for Gemma first", with about 4–8 h GPU.
* **This pre-registration covers the Pile reconstruction and splicing ladder only.** It is extended to several widths
  as well as L0s. The RepE-LM / K1–K2 part is **not** covered. It should be recorded in the roadmap as a separate
  item (proposed name R3b) with its own pre-registration.

### Hypotheses (stated before any run)
* **H1 (L0 within width):** within a fixed layer and width, a higher-L0 SAE has a lower ΔCE_SAE than the next lower-L0
  SAE.
* **H2 (width at matched L0):** at a fixed layer and matched L0 (±10 %), a wider SAE has a lower ΔCE_SAE than the next
  narrower one.
* **H3 (H-G1 across the ladder):** for every SAE, SAE-splice ΔCE exceeds the ΔCE of a norm-matched random error (M6's
  H-G1 finding, tested per SAE).
* **Q0 (precision) has no directional hypothesis.** It is an equivalence check with a fixed margin.
* **Q0b (closeness to fp32) has no directional hypothesis either.** Whichever run is closer is reported.

### Pipeline

```
Committed M6 Pile sample (512 eval + 64 reference docs; sample_manifest.csv from commit a34d975, re-verified)
 → Gemma tokenization (BOS + text, first 1024 tokens), as M6
 → Gemma-2-2B forward pass (HF transformers, bf16 native on A10G, eager attention, strict determinism)
 → resid_post at layers 5, 12, 19   (output of model.model.layers[L] = TL blocks.L.hook_resid_post), as M6
 ┌ Part A1: M6 script 03 at code commit 13eaeca, unchanged → exact M6 rerun on A10G
 │          → Q0: paired per-document comparison with M6's committed T4 outputs
 ├ Part A3: fp32 reference: Gemma and SAE in float32, layer 12, M6's 16k SAE, first 64 eval documents
 │          → Q0b: is T4 bf16 or A10G bf16 closer to fp32 on those documents
 └ Part A2/B: for each pinned Gemma Scope SAE (one at a time):
       encode/decode (fp32) → reconstruction metrics
       → splice into Gemma at that layer (non-BOS positions) → ΔCE, KL, top-1, loss recovered
       → norm-matched random-error control (same random directions for every SAE of a layer)
   shared per layer: mean ablation (64 reference docs), identity, repeat pass
 → per-document outputs → aggregates (overall, per Pile source) → Q1–Q3, H1–H3
```

### Design

| Item | Choice | Reason / source |
|---|---|---|
| Model | `google/gemma-2-2b`, revision `c5ebcd40d208330abc697524c919956e692655cf` (M6 Amendment 1), HF transformers, **bf16**, eager attention | Identical to M6 |
| Hardware | **AWS A10G** (compute capability 8.6, native bf16, 24 GB). Proposed instance: `g5.2xlarge` (32 GiB host RAM, local NVMe), **to be confirmed on the instance** (gate V0) | Native bf16 removes the T4 emulation. Host RAM ≥ 32 GiB is needed to load the 9.7 GB width-524k `params.npz` |
| Hook | `blocks.L.hook_resid_post`, L ∈ {5, 12, 19} | Identical to M6. The site the Gemma Scope residual SAEs were trained on |
| SAE release | HF `google/gemma-scope-2b-pt-res`, revision **`fd571b47c1c64851e9b1989792367b9babb4af63`** | Identical to M6. Listed through the HF API at this revision on 2026-11-13; it is also the repo's current `main` |
| SAEs | The 54 IDs in the next table (Tier A 36, Tier B 18). The 18 width-1M SAEs are listed but excluded | See "SAE ladder" |
| SAE forward | f = JumpReLU_θ(W_enc x + b_enc); x̂ = W_dec f + b_dec; fp32 on bf16 activations cast to fp32; no normalisation | Identical to M6. **Implementation addition:** encode/decode in chunks of 128 tokens for every SAE, including 16k, so the 524k buffers fit. Checked by gate V6 |
| Data | `monology/pile-uncopyrighted` `val.jsonl.zst`, revision `3be90335…`, SHA-256 `db5e5d15…7c26` | Identical to M6 |
| Sample | **The same 512 evaluation + 64 reference documents as M6**, read from the committed `sample_manifest.csv` / `sample_info.json` (M6 commit `a34d975`) and never regenerated | Paired comparisons with M6 and across SAEs |
| Positions | BOS never reconstructed, spliced or scored; behaviour on predictions at 1 ≤ t ≤ len − 2; batch size 1 (no padding) | Identical to M6 |
| Seeds | Bootstrap 0; random-error directions: generator seeded `1000 + L` and **re-seeded at the start of each SAE's pass over the documents**, documents in rank order | Every SAE of a layer gets the same per-token random directions u. Only the norm ‖x − x̂‖ differs, so the control is paired across SAEs |
| Training | None (inference only) | |
| Bootstrap | 1,000 document resamples, 95 % percentile intervals, paired over documents | Identical to M6 |
| Determinism | TF32 off, `torch.use_deterministic_algorithms(True)`, cuDNN deterministic, `CUBLAS_WORKSPACE_CONFIG` set before torch loads, `use_cache=False` | Identical to M6 safeguards A1–A11 |
| fp32 reference (Part A3) | Same model revision loaded with `dtype=torch.float32`; SAE `layer_12/width_16k/average_l0_82` in fp32; TF32 off; the **first 64 evaluation documents by rank**; conditions original, original repeat, identity, SAE splice and norm-matched random error. The random directions are the same as M6's for these documents. There is no mean ablation, so no loss recovered | Gives both bf16 runs an exact-arithmetic yardstick. Layer 12 with the canonical 16k SAE is the one cell shared by M6, the A1 rerun and the ladder |

### SAE ladder (exact IDs in `google/gemma-scope-2b-pt-res` at revision `fd571b4`)

Each ID is `layer_{L}/width_{W}/average_l0_{N}`; the table lists N. The `params.npz` size is the HF LFS size at that
revision. \* marks M6's canonical SAE.

| Layer | Width | Published `average_l0` values | `params.npz` | Tier |
|---|---|---|---|---|
| 5 | 16k | 18, 34, 68\*, 143, 309 | 0.302 GB | A |
| 5 | 65k | 17, 29, 53, 105, 211 | 1.208 GB | A |
| 5 | 1m | 9, 13, 21, 36, 63, 114 | 19.34 GB | excluded |
| 12 | 16k | 22, 41, 82\*, 176, 445 | 0.302 GB | A |
| 12 | 32k | 12, 22, 40, 76, 155, 360 | 0.604 GB | A |
| 12 | 65k | 21, 38, 72, 141, 297 | 1.208 GB | A |
| 12 | 131k | 12, 20, 36, 67, 129, 264 | 2.417 GB | B |
| 12 | 262k | 11, 21, 36, 67, 121, 243 | 4.834 GB | B |
| 12 | 524k | 22, 29, 46, 65, 115, 227 | 9.668 GB | B |
| 12 | 1m | 19, 26, 58, 73, 107, 207 | 19.34 GB | excluded |
| 19 | 16k | 23, 40, 73\*, 137, 279 | 0.302 GB | A |
| 19 | 65k | 21, 35, 63, 115, 216 | 1.208 GB | A |
| 19 | 1m | 16, 18, 29, 50, 88, 157 | 19.34 GB | excluded |

* **The release has 72 SAEs at these layers:** 16 at layer 5, 40 at layer 12, 16 at layer 19.
* **Widths 32k–524k exist only at layer 12** among these three layers.
* **Width 1M is excluded.** Its fp32 weights (19.3 GB) plus the model and working memory (about 7.7 GB, see Compute)
  exceed the A10G's 24 GB. Running it would need bf16 SAE weights or CPU offload, which is a protocol change from M6.
  It is reported as "not evaluated", not as a result.
* **Tier A must run. Tier B runs only under the budget rule** (see Compute).

### Conditions

| Condition | Activation written at layer L (non-BOS positions) | Run | Role |
|---|---|---|---|
| Original | unchanged | per document (recomputed for each SAE) | Reference logits |
| Original repeat | unchanged | once per document | Determinism control (V3) |
| **SAE splice** | x̂ = SAE(x) | per SAE | The measurement |
| Norm-matched random error | x + ‖x − x̂‖ · u, u fixed per (layer, document, token) | per SAE | H3 |
| Mean ablation | mean of layer-L activations over the 64 reference documents | once per layer | Loss-recovered denominator |
| Identity | x written back unchanged | once per layer | Hook-plumbing control (V3) |

### Metrics (identical definitions to M6; `src/evaluation/pile_splicing.py`)

| Metric | Category | Direction |
|---|---|---|
| EV (NMSE = 1 − EV); x̄ = mean of the evaluated tokens of that layer | reconstruction | EV higher / NMSE lower |
| Cosine(x, x̂), mean over tokens | reconstruction | higher |
| Relative squared error Σ‖x − x̂‖² / Σ‖x‖² | reconstruction | lower |
| L0 on our sample (release `average_l0` reported alongside) | sparsity (reported, not targeted) | — |
| ΔCE = CE_spliced − CE_original (nats per prediction) | behavioural splicing | lower |
| KL(P_original ‖ P_spliced), mean per prediction | behavioural splicing | lower |
| Top-1 agreement | behavioural splicing | higher |
| Loss recovered = (CE_mean-abl − CE_SAE) / (CE_mean-abl − CE_original) | behavioural splicing | higher; always read together with ΔCE |
| ΔCE_SAE / ΔCE_random | behavioural splicing | reported (> 1: SAE error more harmful) |

All metrics are reported per SAE, token-weighted, overall **and** per Pile source, each with a document-bootstrap CI.
Sources with fewer than 20 evaluation documents are LOW-N and not interpreted (as M6).

### Pre-registered questions and decision rules

Every test is paired over the same 512 documents. d is a per-document difference of document-mean values, and CIs come
from 1,000 document resamples (seed 0).

**Q0: T4 vs A10G precision (Part A1)**
* **Data:** M6's committed T4 `per_doc_metrics.csv` vs the A10G rerun's `per_doc_metrics.csv`, produced by script 03
  at the same commit with the same inputs.
* **Quantities:** for each layer, per-document paired differences (A10G − T4) of ΔCE_SAE, ΔCE_random, KL_SAE and
  top-1 agreement (behaviour), and of the per-document EV terms (reconstruction). The original CE itself is also
  compared; it directly measures how much the unmodified forward pass changes.
* **Equivalence margins (fixed now):**
  * ΔCE_SAE, ΔCE_random: ±0.005 nats per prediction (7 % of M6's smallest ΔCE_SAE, 0.071);
  * KL_SAE: ±0.005 nats;
  * top-1 agreement: ±0.005;
  * EV: ±0.001.
* **Rule, per layer:**
  * **EQUIVALENT** if the 95 % CI of the mean paired difference lies inside the margin for every metric;
  * **DIFFERENT** if, for any metric, the CI lies wholly outside the margin;
  * **INCONCLUSIVE** otherwise.
  * Separately, the H-G1 verdict at each layer must be the same on both GPUs.
* **Consequence:**
  * If all three layers are EQUIVALENT and H-G1 verdicts agree, M6 (T4) and R3 (A10G) numbers may be shown side by side.
  * Otherwise R3 is interpreted **only against the A10G rerun**. M6's numbers stand as T4 results, with the measured
    difference added to M6's limitations. M6's results are not edited.
* **Also measured (diagnostic):** H4a on the A10G. The unchanged script 03 `--smoke` recomputes the output head in
  float32 on the first 8 documents. This gives the A10G bf16-logit floor next to the T4 value (0.026 nats).

**Q0b: which bf16 run is closer to fp32 (Part A3)**
* **Data:**
  * documents: the first 64 evaluation documents by rank;
  * cell: layer 12, SAE `layer_12/width_16k/average_l0_82`;
  * runs: T4 bf16 (M6's committed per-document rows), A10G bf16 (Part A1 rows) and A10G fp32 (Part A3).
* **Error of a bf16 run on document i:** e_i = |metric_bf16,i − metric_fp32,i|.
* **Metrics:**
  * primary: ΔCE_SAE;
  * secondary: original CE, ΔCE_random, KL_SAE, top-1 agreement and the EV terms.
* **Test:** paired over the 64 documents, D_i = e_T4,i − e_A10G,i, with a 95 % bootstrap CI of the mean (1,000
  resamples, seed 0).
* **Verdict, per metric:**
  * **A10G CLOSER** if the CI lower bound is > 0;
  * **T4 CLOSER** if the upper bound is < 0;
  * otherwise **NO DETECTABLE DIFFERENCE**.
  * The primary metric's verdict is the headline. Secondary verdicts are descriptive.
* **Also reported:** the mean error e of each run, i.e. how far each bf16 run is from fp32, next to Q0's margins.
* **Limits:**
  * one layer, one SAE and 64 documents;
  * fp32 is a reference with much smaller rounding, not exact ground truth;
  * the fp32 run is on the A10G only. That fp32 results are the same on both GPUs, to well below the bf16
    differences, is assumed and not checked.

**Q1 (descriptive):** EV, NMSE, cosine, relative error and measured L0 for every evaluated SAE, overall and per source.
Plotted against measured L0, one curve per width and layer.

**Q2 (descriptive):** ΔCE, KL, top-1 agreement and loss recovered for every evaluated SAE, plotted the same way.

**Q3 (descriptive):** variation across Pile sources (the six M6 sources with ≥ 20 documents), per SAE.

**H1: higher L0 within a width gives lower ΔCE_SAE.**
* **Cells:** the 10 layer × width cells (Tier A: L5 16k, L5 65k, L12 16k, L12 32k, L12 65k, L19 16k, L19 65k; Tier B:
  L12 131k, 262k, 524k).
* **Test:** within a cell, order SAEs by measured L0. For each adjacent pair, d = ΔCE_SAE(lower L0) − ΔCE_SAE(higher L0).
* **Verdict:**
  * **SUPPORTED** in a cell if every adjacent pair's 95 % CI lower bound is > 0. This is an intersection-union test,
    so no multiplicity correction is needed.
  * **CONTRADICTED** if any adjacent pair's upper bound is < 0.
  * Otherwise **INCONCLUSIVE**.
* **Secondary, same rule:** EV, with predicted direction EV(higher L0) > EV(lower L0).

**H2: wider SAE at matched L0 gives lower ΔCE_SAE.**
* **L0 bands:**
  * A band is anchored on a 16k SAE: target = its **measured** Pile L0.
  * Members are SAEs of the same layer and other widths whose measured L0 lies within ±10 % of the target.
  * Membership is computed from L0 alone, by the script, before any behavioural comparison is made.
* **Expected bands** (from release L0; the final bands come from measured L0):

  | Layer | Anchor | Expected members |
  |---|---|---|
  | 12 | 16k / 22 | 32k/22, 65k/21, 131k/20, 262k/21, 524k/22 (the last three are Tier B). **Primary band** |
  | 12 | 16k / 41 | 32k/40, 65k/38 |
  | 12 | 16k / 82 | 32k/76 |
  | 5 | 16k / 18 | 65k/17 |
  | 19 | 16k / 23 | 65k/21 |

  No other release L0 is within ±10 % of a 16k anchor at these layers.
* **Test:** within a band, order members by width. For each adjacent pair, d = ΔCE_SAE(narrower) − ΔCE_SAE(wider).
* **Verdict:** SUPPORTED / CONTRADICTED / INCONCLUSIVE per band, with the same intersection-union rule as H1.
* **Report:** each pair's measured L0 difference. L0 is matched to ±10 %, not equal, so a residual L0 difference is a
  stated limitation.
* **Descriptive only:** ΔCE vs measured L0 frontiers (log-L0) per width. No claim is made from interpolation between
  published L0s.

**H3: H-G1 holds for every SAE.**
* **Test:** per SAE, d = doc-mean ΔCE_SAE − doc-mean ΔCE_random, with a 95 % CI (as M6).
* **Correction:** across all evaluated SAEs, one-sided p-values from the normal approximation to the bootstrap
  (z = mean d / bootstrap SE), Holm-corrected at α = 0.05.
* **Verdict:**
  * **HOLDS ACROSS THE LADDER** if Holm-significant (d > 0) for every evaluated SAE;
  * **MOSTLY** if for at least two thirds;
  * otherwise **NOT GENERAL**;
  * any SAE with a CI upper bound < 0 is reported as a **reversal**.
* **Descriptive:** how the ratio ΔCE_SAE / ΔCE_random varies with width and L0. No test.

**Not claimed:**
* no ranking of layers;
* no "best SAE";
* no GPT-2 vs Gemma comparison beyond a descriptive side-by-side;
* nothing about the 1M SAEs;
* no general claim that a T4 or A10G result is "more correct". Q0 measures the difference, and Q0b compares closeness
  to fp32 only for one layer, one SAE and 64 documents.

### Validity gates (checked before results are read; a failed gate stops the run and is reported)

| Gate | Pass condition |
|---|---|
| V0 hardware and access | GPU name contains "A10G", compute capability 8.6, `torch.cuda.is_bf16_supported(including_emulation=False)` is True; instance type, host RAM and disk recorded; the gated Gemma model can be downloaded at the pinned revision (see "Setup on the EC2 machine") |
| V1 sample integrity | As M6: data SHA-256, manifest SHA-256 = `sample_info.json`, 512 + 64 documents, disjoint, re-tokenised text and token-id hashes match |
| V2 hook equivalence | Hook activation equals `hidden_states[L + 1]` exactly, every document, every layer |
| V3 determinism / identity | Identity-splice and repeat-pass mean KL ≤ 1e-6 nats at every layer (M6's fixed threshold). In the ladder, the per-SAE recomputed original logits must equal the first original pass bit for bit |
| V4 SAE sanity, per SAE | Measured L0 within a factor of 2 of the release `average_l0`, and EV > 0. A failing SAE is excluded and reported, never tuned |
| V5 no silent substitution | Exactly the pinned IDs at revision `fd571b4`; `params.npz` SHA-256 recorded before use; no other ID substituted |
| V6 cross-script consistency | Ladder script on the three M6 canonical 16k SAEs reproduces the A1 rerun (A10G, script 03): per-document reconstruction and behaviour sums within relative difference 1e-5, and active-latent agreement ≥ 0.9999. Checks that 128-token chunking and the SAE-outer loop do not change results |
| V7 memory | Per-SAE peak GPU memory recorded. An SAE that runs out of memory is reported as not evaluated; no change of dtype, chunk size or offloading is allowed without a recorded deviation |
| V8 fp32 reference | Model parameters and logits are float32 (dtypes recorded), TF32 is off, and V2 and V3 also pass in the fp32 run |

### Leakage-prevention checks
* **Nothing is selected or tuned.** There is no training, no probe and no checkpoint selection.
* **The Pile test file is never read.** Eval and reference sets are disjoint (V1). The mean-ablation vectors use only
  the reference set.
* **The 512 evaluation documents were already read by M6.** R3 tunes nothing on them, but they are not fresh.
* **Holdout reserve.** If R3 is later used to *choose* an SAE (for example, for R3b or a causal test), the choice must be
  confirmed on fresh documents: the next 512 eligible documents of the same seed-42 permutation (ranks after the 576
  used), with the same ≥ 32-text-token rule. R3 never reads them.
* **Known limitation (as M6):** Gemma-2's pretraining data is not public. Whether Gemma Scope was trained on The Pile
  is unverified.

### Comparisons

| X vs Y | What it establishes |
|---|---|
| A10G rerun vs M6 T4 (same code, data, SAEs) | Q0: size of the GPU / bf16-precision effect on M6's numbers |
| T4 bf16 and A10G bf16 vs A10G fp32 (layer 12, 16k, 64 documents) | Q0b: which bf16 run is closer to fp32 arithmetic |
| SAE vs original, per SAE | Q2: behavioural cost of each SAE |
| Higher vs lower L0, same layer and width | H1: does more sparsity cost behaviour monotonically |
| Wider vs narrower, same layer, L0 matched ±10 % | H2: does width buy behaviour at equal sparsity |
| SAE splice vs norm-matched random error, per SAE | H3: does H-G1 generalise across the ladder |
| Layer 5 vs 12 vs 19; Pile source vs source | Descriptive only; no ranking |

### Outputs (`results/gemma2_pile/r3_gemma_scope_ladder/`)
* **Pre-registration:** `README.md`, plus `preregistration.json`, a machine-readable copy of this design (layers, the 54
  SAE IDs with tiers, margins, rules, seeds, thresholds), hashed at run time.
* **`m6_rerun_a10g/`:** outputs of script 03 at commit `13eaeca`, copied unchanged, plus `smoke_a10g/` (SMOKE: H4a and
  timing, not a result).
* **`precision_t4_vs_a10g/`:** `q0_paired_differences.csv`, `q0_verdicts.csv` and `q0b_closeness_to_fp32.csv`.
* **`fp32_reference/`:** `per_doc_metrics.csv` (64 documents, layer 12, 16k), `gates.json` and `run_info.json`.
* **`ladder/`:** `per_doc_metrics.csv` (document × SAE × condition), `per_doc_sum_x.npz`, `summary_by_sae.csv`,
  `summary_by_sae_source.csv`, `controls.csv`, `l0_bands.csv`, `h1_l0_within_width.csv`, `h2_width_matched_l0.csv`,
  `h3_hg1_ladder.csv`, `gates.json`, `run_info.json` (with the SHA-256 of every output, written last).
* **`figures/`.**
* **Never committed:** no activation caches; SAE and model weights stay in the HF cache.

### Files to be created (later commits), and why
* **Reused unchanged:**
  * `scripts/gemma/03_pile_gemma_scope_splicing.py`, at commit `13eaeca`, for Part A1. It is run from a checkout of
    that commit, whose results folder has no M6 outputs, so its refuse-to-overwrite guard does not trigger. Outputs
    are then copied into `m6_rerun_a10g/`.
  * `src/evaluation/pile_splicing.py`: the metric definitions.
* **New:**
  * `scripts/gemma/05_pile_gemma_scope_ladder.py`: ladder (Parts A2 and B) and fp32 reference (Part A3,
    `--fp32-reference`); `--smoke`; `--tier A|B`; refuses to overwrite outputs.
  * `scripts/gemma/06_r3_precision_comparison.py`: Q0 and Q0b from the committed per-document CSVs (CPU only).
  * `scripts/gemma/07_r3_figures.py`: figures from saved tables only.
  * `src/tests/test_gemma_r3_ladder.py`:
    * chunked JumpReLU encode equals unchunked on toy weights;
    * random-direction generator gives identical u across SAEs of a layer;
    * L0-band membership rule;
    * H1/H2 intersection-union rule;
    * H3 Holm rule.
* **Possibly modified:** any change to `pile_splicing.py` is made as a *new* function, so M6's hashed code path is
  unchanged.

### Pre-run hashes
* **Pinned external inputs:** dataset file SHA-256 and revision; model and tokenizer revision `c5ebcd4…`; SAE repo
  revision `fd571b4…` and the 54 IDs; M6 code commit `13eaeca`; M6 sample commit `a34d975`.
* **Recorded at run time:** script hashes, in `run_info.json`, together with the SHA-256 of this README and
  `preregistration.json`.

### Compute (planning estimates, not measurements; the A0 smoke run measures them before anything else runs)

**Basis for the estimates:**
* **M6 on the T4:** 15.55 s per evaluation document for 14 forward passes, about 1.1 s per pass at about 496 tokens
  per document. Peak GPU memory 8.58 GB with three 16k SAEs loaded. Without the SAEs (3 × 0.30 GB), model plus
  working memory is therefore about 7.7 GB.
* **A10G pass time:** a planning value of **0.3 s per forward pass**, with a plausible range of 0.2–0.6 s. The A10G has
  native bf16 tensor cores and higher memory bandwidth than the T4.
* **SAE encode/decode cost:** 4 · tokens · 2304 · width floating-point operations, in fp32 at about 15 TFLOP/s
  effective, three applications per document.
* **Per SAE:** 512 documents × (3 forward passes + 3 SAE applications), plus download, hashing and loading.

| Width | Weights (fp32) | Est. peak GPU memory | Est. time per SAE | SAEs | Subtotal |
|---|---|---|---|---|---|
| 16k | 0.30 GB | ≈ 8.0 GB | ≈ 8 min | 15 (A) | ≈ 2.0 h |
| 32k | 0.60 GB | ≈ 8.4 GB | ≈ 8 min | 6 (A) | ≈ 0.8 h |
| 65k | 1.21 GB | ≈ 9.0 GB | ≈ 9 min | 15 (A) | ≈ 2.2 h |
| 131k | 2.42 GB | ≈ 10.3 GB | ≈ 9.5 min | 6 (B) | ≈ 1.0 h |
| 262k | 4.83 GB | ≈ 12.9 GB | ≈ 11 min | 6 (B) | ≈ 1.1 h |
| 524k | 9.67 GB | ≈ 18.2 GB | ≈ 15 min | 6 (B) | ≈ 1.5 h |
| 1m | 19.34 GB | ≈ 28.6 GB, **does not fit in 24 GB** | — | 18 | excluded |

Peak memory = 7.7 GB + weights + 128-token encode buffers (about 3 × 128 × width × 4 bytes).

**fp32 reference (Part A3), run as a separate process:**
* **Memory:** about 16 GB peak, which fits in 24 GB. It is made of:
  * Gemma-2-2B weights in fp32, 2.61 B parameters × 4 bytes ≈ 10.5 GB;
  * working memory about twice M6's bf16 working memory (7.7 − 5.2 ≈ 2.5 GB), so ≈ 5 GB; one fp32 logits tensor alone
    is up to 1,023 × 256,000 × 4 bytes ≈ 1.05 GB;
  * the 16k SAE, 0.3 GB.
* **Host RAM:** the fp32 model (≈ 10.5 GB) fits in 32 GiB.
* **Time:** a planning value of 1.0 s per fp32 forward pass (range 0.6–2 s). Without TF32, fp32 matrix products run at
  roughly a quarter of the bf16 tensor-core rate. 5 passes × 64 documents = 320 passes ≈ 5 min (range 3–11 min).
  With loading the fp32 model, about 0.15 h (at most 0.3 h).

**Totals at 0.3 s per pass:**

| Part | Estimate |
|---|---|
| A0, smoke runs of script 03 and script 05 | ≈ 0.2 h |
| A1, M6 rerun (14 passes × 512 documents) | ≈ 0.6 h |
| A2, Tier A ladder (36 SAEs + shared per-layer controls) | ≈ 5.2 h |
| A3, fp32 reference (64 documents, layer 12, 16k) | ≈ 0.15 h |
| **Tier A total** | **≈ 6.2 h** |
| Tier B | ≈ 3.6 h |
| **A + B total** | **≈ 9.8 h** |

At 0.2–0.6 s per bf16 pass (and 0.6–2 s per fp32 pass), Tier A takes about 4–13 h. Part A3 belongs to Tier A and
always runs.

**Budget rule (fixed now):**
* After A0, project the full run from the measured times.
* Tier A always runs.
* Tier B runs width by width (131k, then 262k, then 524k) while the projected cumulative total stays ≤ 12 GPU-hours.
* A width is either run for all six L0s or not at all, so no partial set of L0s is ever selected.
* Unrun widths are reported as "not run (budget)".

**Storage:**
* SAE downloads: Tier A 26.3 GB; Tier B 101.5 GB.
* Each `params.npz` is hashed, used and then deleted from the cache, so peak disk is about the largest SAE plus the
  model and data.
* Model weight size is read on the instance.
* The Pile file is 0.34 GB.

**Host RAM:** loading the 524k `params.npz` needs about 10 GB, so `g5.2xlarge` (32 GiB) rather than `g5.xlarge`
(16 GiB).

**Unconfirmed:** instance type, region and cost per hour. All are checked at V0 before any download.

### Setup on the EC2 machine (before any run)
**Gemma-2-2B (`google/gemma-2-2b`) is a gated model on Hugging Face.** Downloads fail until the licence is accepted and
the machine is logged in.
1. **Accept the licence.** Logged in to huggingface.co with the account that will run R3, open the
   `google/gemma-2-2b` model page and accept the Gemma licence. Access is granted to that account only.
2. **Create a token.** In that account's settings, create an access token with **read** permission.
3. **Log in on the EC2 machine.** Run `hf auth login` (older versions: `huggingface-cli login`) and paste the token
   when asked. Alternatively, set it for the shell session only with `export HF_TOKEN=<token>`.
4. **Check access before starting** (part of gate V0). Download one small file at the pinned revision:
   `hf download google/gemma-2-2b config.json --revision c5ebcd40d208330abc697524c919956e692655cf`.
   If this fails, stop: nothing else is run.

**The token is a credential:**
* never commit it, put it in a script, or let it appear in a log, `run_info.json` or this README;
* log out (`hf auth logout`) and delete the token from the machine before the instance is stopped or terminated.

The Gemma Scope SAE repository's file listing was readable without logging in (2026-11-13). Its downloads and the Pile
file download are also checked at V0.

## Amendments to the pre-registration (made before any run)

### Amendment 1 (2026-11-13): hardware changed from AWS A10G to Kaggle T4 x2
* **Change:** every part of R3 runs on Kaggle's free **GPU T4 x2** accelerator: two Tesla T4s, 16 GB each,
  compute capability 7.5, **no native bf16**. This replaces the AWS A10G (`g5.2xlarge`) in the Design table, the
  Compute section and "Setup on the EC2 machine".
* **Reason:** our AWS account is on the free plan, which has no GPU instances, so the A10G is not available.
* **Same GPU class as M6.** M6 ran on a Tesla T4 (compute capability 7.5, bf16 kept as pre-registered, torch
  2.7.0+cu128, transformers 5.21.0). The Kaggle run installs and records the same torch and transformers versions. If
  they cannot be installed, the versions used are recorded here as a deviation before any result is read.
* **Section 1 is not edited.** Where Section 1 and this amendment disagree, this amendment applies.

#### (1) Q0 (T4 vs A10G) is dropped
* There is only one GPU type, so the T4 vs A10G question cannot be asked. **Part A1 (the M6 rerun with script 03) is
  not run.** The A10G H4a diagnostic, `m6_rerun_a10g/` and `precision_t4_vs_a10g/q0_*.csv` are dropped.
* The ladder now runs on the same GPU class as M6, so its numbers are **directly comparable to M6** without a precision
  correction. This is checked, not assumed, by gate V6, which is revised to compare against M6 itself:
  * The ladder script on M6's three canonical 16k SAEs is compared with M6's committed per-document rows.
  * **M6 REPRODUCED EXACTLY:** reconstruction and behaviour sums agree within relative difference 1e-5, and
    active-latent agreement is ≥ 0.9999. This also checks the 128-token chunking and the SAE-outer loop, as before.
  * **M6 REPRODUCED WITHIN MARGIN:** not exact, but the 95 % CI of every paired mean difference lies inside the
    margins fixed for Q0 (ΔCE_SAE and ΔCE_random ±0.005, KL_SAE ±0.005, top-1 ±0.005, EV ±0.001), and every layer's
    H-G1 verdict is the same. M6 and R3 may still be shown side by side, with the measured difference stated.
  * **Otherwise the gate fails.** Comparison with M6 is stopped and reported, and R3 is interpreted on its own.
* M6's results are not edited in any case.

#### (2) Q0b becomes "how far is T4 bf16 from fp32"
* **fp32 reference (Part A3):**
  * Gemma-2-2B (same revision) and SAE `layer_12/width_16k/average_l0_82`, both in float32, on the first 64
    evaluation documents by rank. Conditions and random directions are as in Section 1.
  * The model's layers are split across the two T4s with a fixed device map, recorded in `run_info.json`. Moving a
    tensor between GPUs copies it exactly, so the split does not change the arithmetic.
  * fp32 Gemma (≈ 10.5 GB weights + ≈ 5 GB working memory + ≈ 1.05 GB logits) does not fit on one 16 GB T4, which is
    why it is split.
  * The T4 has no TF32, so fp32 matrix products are plain fp32.
* **Data:**
  * **Primary bf16 run:** M6's committed T4 per-document rows for those 64 documents and that SAE.
  * **Also reported:** the Kaggle T4 ladder rows for the same cell, with the same test.
* **Error per document:** e_i = |metric_T4-bf16,i − metric_fp32,i|. Also reported: the signed difference
  b_i = metric_T4-bf16,i − metric_fp32,i, which shows whether bf16 pushes a metric up or down on average.
* **Metrics:** primary ΔCE_SAE. Secondary: original CE, ΔCE_random, KL_SAE, top-1 agreement and the EV terms.
* **Test:** mean of e and mean of b over the 64 documents, each with a 95 % bootstrap CI (1,000 document resamples,
  seed 0).
* **Verdict per metric, against Q0's fixed margins** (±0.005 for the CE-based metrics, KL and top-1; ±0.001 for EV):
  * **WITHIN MARGIN** if the CI upper bound of mean e is below the margin;
  * **BEYOND MARGIN** if the CI lower bound is above it;
  * **INCONCLUSIVE** otherwise.
  * The primary metric's verdict is the headline. Secondary verdicts are descriptive.
* **Output:** `precision_t4_vs_fp32/q0b_t4_vs_fp32.csv`, which replaces `precision_t4_vs_a10g/`.
* **Limits:** as Section 1. One layer, one SAE and 64 documents; fp32 is a reference with much smaller rounding, not
  exact ground truth.

#### (3) Memory per SAE on a 16 GB T4
* **Same formula as Section 1:** peak ≈ 7.7 GB (model + working memory, from M6's T4 run) + fp32 SAE weights +
  128-token encode buffers.
* **Usable memory on each T4 is measured at V0.** A planning limit of **14 GB per GPU** leaves room for the CUDA
  context and fragmentation.
* **Placement:** Gemma always runs on GPU 0. An SAE goes on GPU 0 if its estimated peak is ≤ 14 GB. Otherwise its
  weights and encode/decode go on GPU 1, and activations are copied between the GPUs exactly.

| Width | Weights (fp32) | Est. peak, all on GPU 0 | Placement | SAEs |
|---|---|---|---|---|
| 16k | 0.30 GB | ≈ 8.0 GB | GPU 0 | 15 (A) |
| 32k | 0.60 GB | ≈ 8.4 GB | GPU 0 | 6 (A) |
| 65k | 1.21 GB | ≈ 9.0 GB | GPU 0 | 15 (A) |
| 131k | 2.42 GB | ≈ 10.3 GB | GPU 0 | 6 (B) |
| 262k | 4.83 GB | ≈ 12.9 GB | GPU 0 | 6 (B) |
| 524k | 9.67 GB | ≈ 18.2 GB, does not fit one T4 | **needs GPU 1:** ≈ 7.7 GB on GPU 0, ≈ 10.5 GB on GPU 1 | 6 (B) |
| 1m | 19.34 GB | — | **excluded:** larger than one 16 GB T4 even on its own | 18 |

* **Every Tier A SAE fits on GPU 0 alone.** The second GPU is needed only for the 524k SAEs (Tier B) and the fp32
  reference.
* **Width 1M stays excluded.** Splitting one SAE's weights across two GPUs, using bf16 weights or offloading to the
  CPU would each be a protocol change from M6. It is reported as "not evaluated".
* **Gate V7 also records, per SAE:** the GPU used for the SAE and the peak memory on each GPU.
* **Host RAM:** loading the 524k `params.npz` needs about 10 GB. Kaggle's host RAM is recorded at V0. If it is too
  small, the 524k width is reported as not evaluated.

#### (4) Runtime on the T4, and how Tier A is split into Kaggle runs
* **Basis:** M6's measured **1.1 s per forward pass** on a T4 replaces the A10G planning value of 0.3 s.
* **SAE encode/decode:** the same FLOP formula as Section 1, at about 4 TFLOP/s effective fp32 on the T4.
* **Per SAE:** 512 documents × 3 passes ≈ 28 min, plus encode/decode and download. That gives ≈ 30 min (16k, 32k),
  ≈ 31 min (65k), ≈ 34 min (131k), ≈ 39 min (262k) and ≈ 48 min (524k).
* **Shared per-layer controls** (original repeat, identity, mean ablation): ≈ 28 min per layer.
* **fp32 reference:** a planning value of 1.5 s per fp32 pass (range 1–3 s), so 320 passes ≈ 8 min. With loading,
  ≈ 0.3 h (at most 0.5 h).
* **Kaggle limits:** at most 12 h per session and 30 GPU-hours per week. Each run is planned at ≤ 10 h, which leaves
  a 2 h margin for estimation error.

| Kaggle run | Content | Estimate |
|---|---|---|
| K0 | V0 checks; smoke of script 05 (8 documents; measures s/pass and memory on this T4); Part A3 fp32 reference | ≈ 1 h |
| K1 | Layer 5: 10 SAEs (16k ×5, 65k ×5) + layer controls | ≈ 5.6 h |
| K2 | Layer 19: 10 SAEs (16k ×5, 65k ×5) + layer controls | ≈ 5.6 h |
| K3 | Layer 12 Tier A: 16 SAEs (16k ×5, 32k ×6, 65k ×5) + layer controls | ≈ 8.6 h |
| **Tier A total** | | **≈ 20.8 h** |
| K4 / K5 / K6 (Tier B) | 131k / 262k / 524k at layer 12, six SAEs each | ≈ 3.4 / 3.9 / 4.8 h (≈ 12.1 h) |

* **Rules (replace Section 1's budget rule):**
  * After K0, every remaining run is re-projected from the measured times.
  * A run starts only if its projection is ≤ 10 h. If it is longer, it is split by width into more runs. A split
    changes only *when* an SAE runs, never *which* SAEs run. Every split is recorded in Section 2.
  * No run starts if the week's projected cumulative GPU time would exceed 30 h. The run waits for the quota to reset.
  * How Kaggle counts T4 x2 time against the quota is read from the quota display after K0 and recorded.
  * **Tier A always runs.** Tier B starts only after every Tier A run has passed its gates.
  * Tier B runs width by width (131k, then 262k, then 524k). A width runs only if it fits in the weekly quota and its
    results can be documented before the 2026-11-22 technical freeze.
  * A width is run for all six L0s or not at all. Unrun widths are reported as "not run (budget)".
* **Statistics come last.** Bootstrap CIs, L0 bands and H1–H3 are computed only after all per-SAE outputs exist, in
  one CPU step (script 06 and the aggregate step of script 05). How the work is split into runs therefore cannot affect
  any test.

#### (5) Per-SAE outputs, resume, and a resume-equals-fresh gate
* **Saving:** each SAE writes its own folder `ladder/per_sae/layer_{L}_width_{W}_l0_{N}/`, containing:
  * `per_doc_metrics.csv`;
  * `per_doc_sum_x.npz`;
  * `sae_info.json` (`params.npz` SHA-256, measured L0, peak memory per GPU, GPU used, run id, timings);
  * last, `DONE.json`, holding the SHA-256 of the other three files.
* **Shared per-layer controls** are saved the same way, in `ladder/per_layer/layer_{L}/`. This folder includes the
  mean-ablation vector and its SHA-256.
* **What a run saves:** everything is written under `/kaggle/working/`, so Kaggle keeps it as the run's output.
* **Resuming:**
  * The next run attaches the previous run's output as input and copies it in.
  * It skips every SAE whose `DONE.json` exists and whose hashes match.
  * A folder without a valid `DONE.json` is unfinished. It is moved, not deleted, to `ladder/incomplete/<run id>/`
    and that SAE is run again from the start.
  * Existing complete outputs are never overwritten.
* **Why skipping is exact:** the random-error generator is re-seeded at the start of each SAE's pass (Section 1,
  Seeds). An SAE's numbers therefore do not depend on which SAEs ran before it, or in which run.
* **New gate V9, resume equals fresh:**
  * **On the GPU.** At the start of every run that resumes from earlier output, one already finished SAE is recomputed
    from scratch on the first 32 evaluation documents. This is the first finished SAE in pre-registered order.
  * If the run reuses a layer's saved controls, that layer's mean-ablation vector is also recomputed.
  * **Pass condition:**
    * every recomputed per-document value matches its stored value within relative difference 1e-6, that is,
      |new − stored| ≤ 1e-6 · max(|new|, |stored|), with two exact zeros counted as a match;
    * the recomputed mean-ablation vector matches the stored one within relative difference 1e-6, that is,
      ‖v_new − v_stored‖ / ‖v_stored‖ ≤ 1e-6.
  * **Recorded separately:** whether the recomputed rows and the mean-ablation vector were also **bit-identical**
    (BIT-IDENTICAL yes or no), together with the largest relative difference seen. This is reported, but a pass does
    not require it.
  * **If any value differs by more than 1e-6,** V9 fails and the run stops. Results are not read until the cause is
    found and recorded.
  * **In the tests.** `src/tests/test_gemma_r3_ladder.py` adds a CPU test on a toy model: a run interrupted after one
    SAE and then resumed must produce output files byte-identical to an uninterrupted run.
* **Per-run record:** each Kaggle run writes `runs/<run id>/run_info.json`, with GPU names, driver, CUDA, torch and
  transformers versions, host RAM, disk, the SAEs run and skipped, and the V9 result. `ladder/run_info.json` (all
  output hashes) is still written last, by the final aggregate step.

#### (6) Hugging Face token from Kaggle Secrets
* **This replaces steps 3–4 and the logout step of "Setup on the EC2 machine".** Steps 1–2 still apply: accept the
  Gemma licence and create a read token on the account that runs R3.
* **Storing the token:** it is stored as a Kaggle Secret named `HF_TOKEN` (notebook menu Add-ons → Secrets), attached
  only to the R3 notebook of that account.
* **Using the token:**
  * The notebook reads it in memory with `kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")` and passes it to
    the Hugging Face client in the same process.
  * It is **never** written to the notebook source, cell output, an environment file, `/kaggle/working`, a log,
    `run_info.json`, this README or a commit. It is never printed, not even partly.
* **Gemma comes from Hugging Face.** The model is downloaded from Hugging Face at the pinned revision `c5ebcd4…`.
  Kaggle's own Gemma copy is not substituted (gate V5).
* **Gate V0 (revised):**
  * Two GPUs named "Tesla T4", compute capability 7.5. `torch.cuda.is_bf16_supported(including_emulation=False)` is
    recorded; False is expected, as in M6.
  * Notebook internet access is on.
  * The pinned `config.json` downloads using the secret. If it does not, stop: nothing else is run.
* **New gate V10, no secret in outputs:**
  * Before any output leaves Kaggle, every output file and the run log are searched for the token's exact value (the
    comparison happens in memory, and only "found / not found" is printed) and for the pattern `hf_[A-Za-z0-9]{30,}`.
  * **Any hit:** the outputs are not downloaded or committed, the token is revoked on huggingface.co, and a new one is
    made.
  * The token is revoked anyway once R3 has finished.
* **The notebook stays private.** Its committed copy, if any, has all outputs cleared.

#### Not changed
Everything else in Section 1 is unchanged:
* the research question about width and L0;
* data, sample, model and SAE revisions, and the 54 pinned SAE IDs and their tiers;
* conditions, metrics, seeds, bootstrap and determinism settings;
* H1–H3 and their decision rules, L0 bands, Q1–Q3;
* gates V1–V5 and V8, and the leakage checks.

No code had been written and no run made when this amendment was committed.

## 2. After execution
*(To be completed after the run: runs, gates, Q0 verdicts, results tables, criteria met or not met, limitations,
deviations, files, Git commit.)*

## 3. If stopped early
*(To be completed if needed: current status, existing artifacts, unfinished work, exact next steps.)*

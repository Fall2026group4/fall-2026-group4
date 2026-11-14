# M6 Gemma Scope results vs published numbers (2026-11-13). Status: LITERATURE COMPARISON (descriptive)

**Author:** Dhruv Rai. **Belongs to:** R3 pre-registration (`README.md` in this folder). **Our numbers:** M6,
`results/gemma2_pile/gemma_scope_splicing/summary_by_layer.csv` (validation finding, Tesla T4, bf16).

**Evidence label:** this is a reading of published papers, not an experiment. It changes no M6 result and no R3 criterion.

## How the published numbers were obtained

**Text and tables.** Values stated in a paper's text or tables are quoted as stated, with section, table or line.

**Figures.** Most Gemma Scope results exist only as plots, so they are marked **"approx., read from figure N"**:
* **Gemma Scope figures (arXiv HTML, vector SVG).**
  * The plotted points were read from the vector coordinates of each line's vertices. The axes were calibrated on the
    figure's own gridlines and tick labels; each L0 axis is logarithmic.
  * **Check:** the L0 values read for the 16.4K layer-12 series (22.0, 82.0, 177.4, 446.2) reproduce the release's
    `average_l0` names (22, 82, 176, 445).
  * **Limits:** the PDF-to-SVG export merges nearly collinear vertices, so some points are missing from the drawn
    line. Values "at our L0" are therefore log-log interpolations along the plotted line.
  * **Precision:** about ±2 % in Fig. 2 and Fig. 13, about ±4 % in Fig. 17 (whose decades are only 13 SVG units tall).
* **SAEBench figures (PNG).** Read by eye. Precision is about ±0.0005 in "Loss Recovered".

## Our M6 numbers (Gemma 2 2B, `resid_post`, 16k, canonical L0; 512 Pile validation documents; ≤ 1,024 tokens)

| Layer (SAE) | L0 measured (release) | EV | NMSE = 1 − EV | ΔCE_SAE (nats) [95 % CI] | KL | Top-1 agreement | Loss recovered (mean ablation) |
|---|---|---|---|---|---|---|---|
| 5 (`average_l0_68`) | 69.5 (68) | 0.813 | 0.187 | 0.071 [0.064, 0.079] | 0.082 | 0.887 | 0.992 |
| 12 (`average_l0_82`) | 84.8 (82) | 0.770 | 0.230 | 0.115 [0.107, 0.124] | 0.120 | 0.857 | 0.988 |
| 19 (`average_l0_73`) | 75.1 (73) | 0.789 | 0.211 | 0.177 [0.165, 0.192] | 0.179 | 0.838 | 0.980 |

Original CE: 1.938 nats per prediction. Mean-ablation ΔCE: about 8.7–9.7 nats.

## 1. Gemma Scope (Lieberum et al., 2024, arXiv 2408.05147)

**Method, in simple words.**
* Google trained JumpReLU SAEs on every layer and site of Gemma 2 (2B, 9B, some 27B) and released them.
* Each SAE learns a threshold per latent and is trained with a squared-error loss plus a direct L0 penalty. Training
  ran in float32, on activations from Gemma 1's pretraining data, with BOS, EOS and padding excluded (§3.1, §3.3).
* The 16.4K-width SAEs saw **4B tokens**; other widths 8B, and 1M saw 16B (§3.1, Table 1).

**How they evaluate (stated):**
* Primary metric: "delta LM loss, the increase in the cross-entropy loss … when we splice the SAE into the LM's
  forward pass" (§4.1).
* Secondary metric: FVU, the reconstruction loss "normalized by the reconstruction loss obtained by always predicting
  the dataset mean" (§4.1).
* "All metrics were computed on 2,048 sequences of length 1,024, after masking out special tokens" (§4.1).
* The evaluation data are "sequences from the same distribution as the SAE training data, i.e. the pretraining
  distribution of Gemma 1" (§4.1).
* Table 1 confirms the width ladder used in R3: widths 2^15, 2^17, 2^18 and 2^19 exist only at layer 12 of Gemma 2 2B,
  and 2^20 (1M) at layers {5, 12, 19}.

**Comparison at layer 12, the same SAE as M6** (approx., read from figures 2 and 13):

| Quantity | Gemma Scope (16.4K, layer 12, Gemma 2 2B) | Ours (M6) |
|---|---|---|
| Delta loss at L0 82 | **≈ 0.110** (Fig. 2); ≈ 0.108 interpolated at our L0 84.8 | ΔCE **0.115** [0.107, 0.124] at L0 84.8 |
| FVU at L0 82 | **≈ 0.225** (Fig. 13); ≈ 0.222 at L0 84.8 | NMSE **0.230** [0.228, 0.231] |
| Other 16.4K points on the same curve (L0 → delta loss) | 22 → 0.274; 177 → 0.061; 446 → 0.027 (Fig. 2) | — (R3 Tier A will measure them) |
| Other 16.4K points (L0 → FVU) | 22 → 0.310; 42 → 0.267; 177 → 0.175; 446 → 0.109 (Fig. 13) | — |
| 65.5K points (L0 → delta loss / FVU) | 21 → 0.189 / 0.265; 73 → 0.080 / 0.191; 297 → 0.029 / 0.115 (Figs. 2 and 13) | — (R3 Tier A) |

**Layers 5 and 19 (no same-layer curve in the paper).**
* Fig. 17 plots Gemma 2 2B layers 0, 4, 8, 12, 17, 21 and 25 only. So our layers can only be **bracketed**
  (16.4K residual series, approx., read from Fig. 17, interpolated at our L0):

| Our layer | Neighbouring layers in Fig. 17: delta loss at our L0 | Ours: ΔCE |
|---|---|---|
| 5 (L0 69.5) | layer 4: ≈ 0.048; layer 8: ≈ 0.079 | 0.071 |
| 19 (L0 75.1) | layer 17: ≈ 0.143; layer 21: ≈ 0.201 | 0.177 |

* Both of our values lie between the neighbouring layers, and the paper's curves rise with depth, as ours do. The
  layer-12 row of Fig. 17 matches Fig. 2 (≈ 0.110 at L0 82).

**What differs between their numbers and ours:**

| Aspect | Gemma Scope | Ours (M6) |
|---|---|---|
| Evaluation data | Gemma 1 pretraining distribution (not public) | The Pile, uncopyrighted validation file, 512 documents |
| Sequences | 2,048 sequences of length 1,024 | One window per document, truncated to 1,024 tokens; mean ≈ 496 tokens. §4.2 reports reconstruction loss rising with position, so shorter documents may make our numbers slightly lower |
| Metric | Delta LM loss (CE with SAE − CE without) | ΔCE: same definition, nats per prediction, positions 1…len−2 |
| Special tokens | Pad, BOS and EOS masked | BOS never spliced or scored; no padding (batch size 1) |
| Splice positions | Not stated beyond masking special tokens | All non-BOS positions |
| L0 | Measured on their evaluation data | Measured on the Pile (+2–3 % above the release value) |
| Precision | Fp32 by default; Fig. 10 claims "negligible impact" of bf16 | bf16 model on a T4 (emulated bf16), SAE in fp32 |
| Width, layer, hook | Same SAE (16.4K, layer 12, `resid_post`) | Same |

**Pile subsets (§4.6, Fig. 9).**
* Gemma Scope also evaluated on "validation subsets of The Pile" ("a different dataset to the dataset used to train
  the Gemma Scope SAEs"), with the **65K** SAE for Gemma 2 2B.
* The caption does not state the layer. §4.6 reuses the §4.1 evaluation, which uses layer 12, so layer 12 is our
  inference.
* At L0 ≈ 73 (approx., read from Fig. 9; this is release `layer_12/width_65k/average_l0_72`):

  | Subset | Delta loss |
  |---|---|
  | Europarl | ≈ 0.189 |
  | Wikipedia (en) | ≈ 0.099 |
  | ArXiv | ≈ 0.099 |
  | FreeLaw | ≈ 0.096 |
  | HackerNews | ≈ 0.092 |
  | OpenWebText2 | ≈ 0.090 |
  | GitHub | ≈ 0.074 |
  | DM Mathematics | ≈ 0.047 |

* **Contrast with M6 (descriptive only).** Our 16k SAE at layer 12 gives **GitHub** one of the *largest* ΔCE values
  (0.146; Wikipedia 0.155; StackExchange 0.089). Gemma Scope's 65K SAE gives GitHub one of the *smallest*.
* **What differs:**
  * width (16k vs 65K);
  * which documents (our 52 GitHub documents; their subset size is not stated);
  * sequence construction;
  * our GitHub documents have a very low original CE (0.91 nats), which can change how ΔCE behaves.
* R3 Tier A includes `layer_12/width_65k/average_l0_72` on our documents, so it will show whether the difference comes
  from width or from data.

**bf16 (§4.7, Fig. 10).**
* The paper states "negligible impact in switching to bfloat16 for inference". At the resolution of the vector data,
  the bf16 and fp32 curves for 2B layer 12 coincide (≈ 0.111 at L0 82).
* This concerns *average* delta loss. M6's H4a measured a per-prediction *unsigned* |CE difference| of 0.026 nats on a
  T4, which can average out in a signed mean. R3 Q0/Q0b measure this directly.

## 2. JumpReLU SAEs (Rajamanoharan et al., 2024, arXiv 2407.14435)

**Method, in simple words.**
* A JumpReLU sets every encoder pre-activation below a learned threshold θ to zero and keeps the rest unchanged.
* The threshold and the L0 penalty have no ordinary gradient, so they are trained with straight-through estimators.
  These use a rectangle kernel of bandwidth ε, which estimates the gradient of the *expected* loss (§3–4).
* The paper states "ε = 0.001 (assuming a dataset normalised such that E[x²] = 1) works well across different models,
  layers and sites" (§3).

**What it reports (stated):**
* 131k-width SAEs on **Gemma 2 9B**, at layers 9, 20 and 31, trained on 8B tokens (§5).
* The same metrics as Gemma Scope, "computed on 2,048 sequences of length 1,024" (§5).
* "JumpReLU SAEs consistently offer similar or better fidelity at a given level of sparsity than TopK or Gated SAEs"
  (§5.1, Fig. 2).

**Comparison with M6:** no Gemma 2 2B numbers, so **no numerical comparison**. Model, width and layers all differ.

## 3. SAEBench (Karvonen et al., 2025, arXiv 2503.09532)

**Method, in simple words.**
* A benchmark of eight SAE metrics: loss recovered, automated interpretability, feature absorption, sparse probing,
  spurious-correlation removal (SCR), targeted probe perturbation (TPP), RAVEL and unlearning.
* Its main message is that the usual sparsity-vs-fidelity curve is "necessary but not sufficient" (§3.2.1).
* It trains its own suite of SAEs and also evaluates the Gemma Scope "width series" at Gemma 2 2B layers 5, 12 and 19
  (Appendix I).

**Metric definition (stated, Eq. 4):** Loss Recovered = (H* − H₀)/(H_orig − H₀), where H₀ is the CE "resulting from
**zero-ablating** x".

**Evaluation settings (stated, Table 4):** OpenWebText, **context length 128 tokens**, 3,200 sequences. BOS, EOS and
PAD are masked (Appendix D).

**Comparison** (approx., read from Figures 12, 13 and 14 of Appendix I: Gemma Scope 16k, Gemma 2 2B):

| Layer | SAEBench Loss Recovered (zero ablation) near our L0 | Ours: loss recovered (mean ablation) |
|---|---|---|
| 5 | ≈ 0.993 at L0 ≈ 68 (Fig. 12) | 0.992 |
| 12 | ≈ 0.988–0.989 at L0 ≈ 80 (Fig. 13) | 0.988 |
| 19 | ≈ 0.983–0.984 at L0 ≈ 73 (Fig. 14) | 0.980 |

**What differs:**
* The denominator: **zero ablation** (theirs) vs **mean ablation** (ours), so the two scores are not the same
  quantity.
* Data: OpenWebText vs the Pile.
* Context: 128 tokens vs up to 1,024.
* L0 is measured on each side's own data.

The ordering across layers (5 > 12 > 19) agrees. Close numbers should not be read as agreement, because both
ablations are so destructive that almost any SAE scores near 1.

**Other stated findings that matter to us:**
* "Loss Recovered and AutoInterp improves consistently with increased width" for the Gemma Scope series (Appendix I),
  while absorption, SCR and TPP "degrade at larger widths".
* "A moderate L0 ∈ [50, 150] strikes a reasonable balance" (§4.3).
* "We recommend training multiple SAEs across a range of sparsities (L0 ∈ [20, 200]) with directly comparable
  baselines" (§3.5).
* Their own SAEs: 500M training tokens on The Pile, context 1,024 (Table 3). "Many metrics achieve most of their
  performance by 50M tokens" (§4.4).

## 4. Scaling and evaluating sparse autoencoders, TopK (Gao et al., 2024, arXiv 2406.04093)

**Method, in simple words.**
* TopK SAEs keep only the k largest latents per token, so L0 = k is set directly and no L1 penalty is needed.
* Dead latents are prevented by initialising the encoder as the decoder's transpose and by an auxiliary loss, AuxK,
  that reconstructs the error using dead latents (§2.4, App. A).
* They fit scaling laws in width and k, and train up to 16M latents on GPT-4.

**What it reports (stated):**
* GPT-2 small at layer 8 and GPT-4-family models; "a context length of 64 tokens for all experiments" (§2.1).
* Inputs are mean-subtracted and normalised to unit norm before the SAE (§2.1).
* "the zero-ablation fidelity metric of our 16M autoencoder is 98.2 %" (GPT-4, §4.1, footnote). The authors argue
  zero-ablation scores are too lenient.
* For GPT-2 small: "downstream loss and KL get strictly worse with layer", and "Overall loss difference has no clear
  relation with normalized MSE across layers" (App. F.1, Fig. 27).

**Comparison with M6:** different models, so **no numerical comparison**. The two qualitative statements match M6:
* ΔCE rises with depth: 0.071 at layer 5, 0.115 at layer 12, 0.177 at layer 19.
* NMSE does not follow ΔCE: layer 19 has better NMSE than layer 12 (0.211 vs 0.230) but worse ΔCE.

## 5. End-to-end sparse dictionary learning (Braun et al., 2024, arXiv 2405.12241)

**Method, in simple words.**
* Instead of training the SAE to copy the activation (MSE, "local"), e2e SAEs are trained so that the model's
  **output distribution** with the SAE spliced in matches the original (KL loss).
* e2e+ds adds reconstruction losses at later layers, so the model keeps following its original computation path (§2).

**What it reports (stated):**
* GPT-2 small residual stream, **before** layer 6 (also layers 2 and 10). OpenWebText, context 1,024; dictionary
  60 × 768 (§3, App. D). GPT-2 small CE 3.139 on their evaluation set (Fig. 1 caption).
* Table 1 (layer 6, similar CE increase):

  | SAE | L0 | CE increase |
  |---|---|---|
  | local | 69.4 | 0.145 |
  | e2e | 27.5 | 0.144 |
  | e2e+ds | 36.8 | 0.125 |

* "For a given L0, both SAE_e2e and SAE_e2e+ds have a CE loss increase that is less than 45 % of the CE loss increase
  of SAE_local" (§3.1, over L0 ∈ (50, 300)), at "2–3.5 times longer runtime".

**Comparison with M6:** a different model, so **no numerical comparison**. The paper's setup (GPT-2 small,
`resid_pre`, OpenWebText) matches our **GPT-2** work, not M6.

## What this means for us

1. **M6 reproduces the published layer-12 numbers.** For the same SAE, our ΔCE (0.115) and NMSE (0.230) are within
   about 5 % and 2 % of Gemma Scope's own figures (≈ 0.110 and ≈ 0.225), even though the data (the Pile vs Gemma 1
   pretraining data), the GPU precision and the document lengths differ. This supports M6's pipeline. It is not a
   statistical test, since the figure values have no CIs.
2. **Layers 5 and 19 are consistent but unverified.** Both fall between the published neighbouring layers. No paper
   gives same-layer delta-loss numbers for them; SAEBench gives zero-ablation loss recovered only.
3. **"Loss recovered" should not be compared across papers without checking the ablation.** SAEBench uses zero
   ablation, we use mean ablation, and Gao et al. argue both are too lenient. ΔCE is the comparable quantity.
4. **One open disagreement:** which Pile subsets are hardest (GitHub). R3 can resolve it.

## Methods we could apply

**To the pretrained Gemma Scope SAEs (no retraining):**
* **Inference-time refinement of latent activations** (Gao et al., §5.1, Fig. 9): re-solve the activations for a fixed set of
  active latents to reduce reconstruction error. It is cheap, and could be tested on M6/R3 SAEs as a new
  pre-registered condition.
* **A KL (end-to-end-style) fine-tune of a Gemma Scope SAE** at fixed L0 (Braun et al.; this is our R4 recipe applied
  to Gemma). It needs a separate pre-registration, plus a check that downstream layers are not pushed off their
  normal path (the e2e+ds idea).

**To our in-house GPT-2 SAEs:**
* **e2e+ds training** (Braun et al.). Our R4 KL+MSE fine-tune is a cheaper relative. Braun et al. report e2e CE
  increases below 45 % of local at equal L0 on the same model, site and dataset as our GPT-2 work.
* **JumpReLU as published** (rectangle-kernel STE, direct L0 penalty, ε = 0.001 on E[x²] = 1-normalised data,
  θ initialised at 0.001 per Gemma Scope §3.2). Our in-house JumpReLU used a sigmoid surrogate, and the R2 roadmap
  notes the modern JumpReLU was abandoned in the pilot, so this is the specific recipe to retry.
* **TopK with AuxK and encoder = decoderᵀ initialisation** (Gao et al.) is already used in R2. Gao's scaling result
  (tokens to convergence ≈ Θ(n^0.6) on GPT-2 small) and SAEBench's "most of their performance by 50M tokens" support
  our ~100M-token R2 budget for width 12,288.
* **Matryoshka SAEs** (SAEBench §4.1). They were best on 5 of 8 metrics but worse on the sparsity-fidelity frontier,
  so they are a candidate only if truth or representation preservation, not reconstruction, is the goal.

**For evaluation:** SAEBench's advice to compare SAEs over L0 ∈ [20, 200] with matched baselines matches our
sparsity-matching rule. R3's L0 bands follow it.

## Sources
* Lieberum et al. (2024), *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*,
  https://arxiv.org/abs/2408.05147 (v2; HTML figures 2, 9, 10, 13, 17).
* Rajamanoharan et al. (2024), *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*,
  https://arxiv.org/abs/2407.14435.
* Karvonen et al. (2025), *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model
  Interpretability*, https://arxiv.org/abs/2503.09532 (v4; Appendix I, Figures 12–14; Tables 3–4).
* Gao et al. (2024), *Scaling and evaluating sparse autoencoders*, https://arxiv.org/abs/2406.04093.
* Braun et al. (2024), *Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning*,
  https://arxiv.org/abs/2405.12241 (Table 1, §3.1).

## Appendix: figure calibration (SVG units, Gemma Scope arXiv HTML v2)

| Figure (file) | Panel | x: L0 = 10 at; units per decade | y: reference gridline; units per decade |
|---|---|---|---|
| Fig. 2 (`delta_loss-crop.svg`) | 2B residual | x = 381.01; 66.37 | delta loss 1.0 at y = 9.42; 32.24 |
| Fig. 13 (`fvu_main-crop.svg`) | 2B residual | x = 373.01; 66.37 | FVU 0.1 at y = 78.80; 78.40 |
| Fig. 17 (`delta_loss_2b-pt-crop.svg`) | residual, row k (layers 0, 4, 8, 12, 17, 21, 25) | x = 388.64; 62.73 | delta loss 1.0 at y = 14.77 + 69.42·k; 13.31 |
| Fig. 9 (`pile_main-crop.svg`) | 2B residual | x = 323.39; 58.28 | delta loss 1.0 at y = 19.19; 21.40 |

The series colours were identified from the legend line samples (Plotly default order). The extraction scripts were
run locally and are not part of this commit.

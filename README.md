# SAE Faithfulness: reconstruction, information, behaviour and causal effects across GPT-2 layers

**Team:** Aditi Shukla, Dhruv Rai, Simbanegavi Simbarashe. GWU DATS 6501 Data Science Capstone, Fall 2026.
**Final presentation and journal submission:** 2026-12-08 (`proposal.md`). **Status of this README:** 2026-11-12, branch
`sae-research` (sections 6.0, 8–14 updated for the 2026-11-10 improvement phase and the M5 test pass; other sections
as of 2026-11-08). Everything listed as COMPLETED below has its artifacts in this repository. Items marked PLANNED or
PROPOSED have not been run.

**Documentation for the GPT-2 SAE work (start here):**

| Need | Document |
|---|---|
| What the project asks and found | [`docs/research_overview.md`](docs/research_overview.md) |
| How everything is measured; methodological corrections | [`docs/methodology.md`](docs/methodology.md) |
| Everything we tried, in order, with verdicts | [`docs/experiment_history.md`](docs/experiment_history.md) |
| Verified numbers by dataset (OWT, GoT, RepE) | [`docs/results_summary.md`](docs/results_summary.md) |
| Reproduction commands | §13 below |
| Limitations and proposed next experiments | [`docs/limitations_and_future_work.md`](docs/limitations_and_future_work.md) |
| Citation status of every result | [`docs/results_registry.md`](docs/results_registry.md) |

**The three pipelines:**
* **OpenWebText:** OWT → frozen GPT-2 → `hook_resid_pre` activations → SAE → reconstruction metrics → behavioural
  splicing (ΔCE, KL).
* **Geometry of Truth:** GoT → frozen GPT-2 → final-token activations → SAE reconstruction → frozen and re-fit probes
  (vs PCA and noise) → AUROC gaps.
* **RepE honesty:** RepE-LM → frozen GPT-2 → persona activations → SAE reconstruction → representation evaluation
  (re-fit gap, direction cosine, norm retention) → causal decomposition (M4) and skyline (R5).

## 1. Motivation

Sparse autoencoders (SAEs) decompose a language model's dense activations into sparse, nominally interpretable
features. Interpretability work often treats the SAE as a faithful stand-in for the model. "Faithful" has four
separable meanings, and this project measures each one **separately**:

| Level | Question | Measured by |
|---|---|---|
| Reconstruction | Is the reconstruction x̂ close to the activation x? | MSE, NMSE = 1 − EV, cosine, at a fixed L0 |
| Information preservation | Can task information (truth, persona) still be read from x̂, and by which readers? | AUROC gaps of frozen and re-trained readers, against noise and PCA references |
| Behavioural faithfulness | Does GPT-2 behave the same when x̂ replaces x? | Spliced cross-entropy increase, output KL, truth preference |
| Causal faithfulness | Does x̂ carry the causal effect GPT-2 uses for a task? | Activation patching, decomposed into reconstruction, error and interaction parts |

Good reconstruction does not imply the other three. Our results show this directly (Section 6).

## 2. Research question

**Can we improve SAE reconstruction of frozen LLM activations while preserving truth-related information and causal
behaviour across transformer layers?**

Since 2026-11-08 (`docs/RESEARCH_DECISIONS.md` D23) the project is framed as a **layer-wise measurement study**, with
fine-tuning as the "improvement" arm:

| Sub-question | Content | Main evidence |
|---|---|---|
| SQ1 Reconstruction | How do ReLU, TopK and JumpReLU SAEs compare across the 12 layers at matched sparsity? | 108-run baseline (TEST), M0 |
| SQ2 Information | Which task information survives reconstruction, and is the loss SAE-specific? | M2 (validation) |
| SQ3 Behaviour | How much does splicing the SAE in damage GPT-2's predictions, and does reconstruction error predict it within a layer? | Phase 0 splicing, M0 (validation) |
| SQ4 Causal | Where does GPT-2 carry a persona-instruction effect, and how much of it survives in the SAE reconstruction? | M1, M3, M4 (validation) |
| SQ5 Improvement | Can fine-tuning improve faithfulness at matched sparsity, and at which layers? | Layer-6 fine-tuning pilots, IMP-L (layers 3, 8, 9), RLM-RP and E3 (validation) |

The original proposal (`proposal.md`) asked whether SAE features are causally faithful to their auto-generated
labels. The instructor reviews and our own audits narrowed it to the question above. The history is in
`docs/RESEARCH_PROGRESS.md`.

## 3. Models

| Model | Role | Status |
|---|---|---|
| **GPT-2 Small** (frozen; 12 blocks, d_model 768; HF `gpt2` via TransformerLens) | Subject model for all SAE work | Used throughout |
| **In-house SAEs** (ReLU, TopK, JumpReLU) | Main objects of study | 108 baseline SAEs (12 layers × 3 architectures × 3 seeds) plus fine-tuned variants |
| **Public `gpt2-small-res-jb`** (SAELens; 24,576 latents; trained on 300M OWT tokens per its `cfg.json`) | Phase-1 illustration (Feature 974); legacy layer-10 GoT check | Planned as an external reference (E2). Not matched in width, data or L0 |
| **Gemma-2-2B** (HF, bf16) | Steering appendix | **Steering only; no Gemma SAE results.** Gemma Scope weights are not downloaded |

## 4. Datasets

Row counts were read from the files listed.

| Dataset | Purpose | Size and splits | Files | Validity |
|---|---|---|---|---|
| **OpenWebText** (`Skylion007/openwebtext`) | SAE training, reconstruction, splicing | 1000 documents (800 / 100 / 100). Each is truncated to 128 tokens; BOS excluded → **101,600 / 12,700 / 12,700 tokens** | `data/openwebtext/`, `configs/openwebtext/dataset.yaml` | Valid; small |
| **Geometry of Truth: cities + neg_cities** ("geographic") | Truth information | 1496 statements each; city-grouped splits 1046 / 224 / 226 each | `data/got/`, `data/got/splits/` | Valid. **The test split has already been scored** (see §10) |
| GoT larger_than / smaller_than | Numerical truth (diagnostic only) | 1980 each; 1386 / 296 / 298 | same | Surface shortcut (second number alone: AUROC 0.84). Not a truth measure |
| **RepE-LM** (built by us in M1) | Shortcut-free persona task | 10,120 prompts = 506 statements × 20 one-token-swap variants; train 417 / validation 89 statements. **No test prompts have been built** | `data/repe/repe_lm/` | Valid. The test split is untouched |
| RepE persona prompts (original) | Earlier honesty work | 1194 prompts (834 / 180 / 180) | `data/repe/persona_pairs.csv` | **HISTORICAL:** ' untruthful' is 3 tokens longer than ' honest' (length alone gives AUROC 0.80) |

RepE-LM templates (`data/repe/repe_lm/templates.json`):
* **Fit templates (T1–T4):** truthful/deceptive, trustworthy/dishonest, sincere/lying, facts/lies. Each has an
  honest-synonym control (sincere, reliable, truthful, truths).
* **Held out:** T5 true/false, T6 correct/wrong.
* **Neutral:** N1 tall/short, N2 red/blue.

## 5. SAE architectures and training (verified in `scripts/openwebtext/03_training_sae.py`, `13_consistent_protocol_retrain.py`)

* **Hook:** `blocks.L.hook_resid_pre`, L = 0–11. d_in = 768. **Width 3072 (4× expansion).** BOS excluded from training
  and splicing.
* **Encoder and decoder:** pre = W_e(x − b_d) + b_e, x̂ = W_d z + b_d. Decoder columns are renormalised to unit norm
  after every step.
* **Activations:**
  * **ReLU:** z = ReLU(pre), L1 penalty on z.
  * **TopK:** z = TopK₁₂₈(ReLU(pre)), no penalty.
  * **JumpReLU (surrogate):** z = ReLU(pre)·H(pre − θ), with a learned per-latent threshold θ. The gradient comes from
    a sigmoid straight-through estimator (T = 0.1), with a sigmoid-count sparsity penalty. **This is not the published
    JumpReLU estimator.**
* **Objective:** MSE + c·penalty. A multiplicative controller adjusts c every 25 steps to reach the **target L0 = 128**.
* **Optimisation:** Adam (lr 3e-4), batch 256 drawn with replacement, gradient clipping at 10. **One uninterrupted
  15,000-step run** per SAE. Seeds 42, 1042 and 2042.
* **Budget:** 15,000 × 256 = 3.84M samples, about **38 passes** over the 101,600 training tokens.
* **Sparsity matching:** architectures are compared only within ±10 % of the target L0 (fine-tunes within ±5 % of
  their base). Unmatched runs are flagged and excluded from rankings, not deleted.
* **Checkpoints:** `results/openwebtext/convergence/layer_{L}/{arch}_l0_128_uninterrupted_seed{S}/checkpoint_step_15000.pt`.
  They are not in Git; `docs/checkpoint_registry.csv` lists each one with its SHA-256.

## 6. Results so far

**Labels:** **TEST** = held-out split, evaluated once under a protocol. **VALIDATION** = validation split only.
**PRELIMINARY** = a validation pilot whose pre-registered criterion was not met, or which has not been replicated.

Full tables: `docs/EXPERIMENT_RESULTS_SUMMARY.md`. Which files may be cited: `docs/results_registry.md`.

### 6.0 Latest: data-scaled SAEs, fixed-L0 KL fine-tune, and the M5 test pass (2026-11-10)

Verified against `results/shared/m5_confirmatory_test/m5_rows.csv`; details in
[`docs/results_summary.md`](docs/results_summary.md).

| Comparison (layer 8 unless stated) | Before | After | Label |
|---|---|---|---|
| OWT NMSE, old TopK-3072 → R2 TopK-12288 (L8 / L11) | 0.160 / 0.185 | 0.094 / 0.111 | **TEST**, improvement |
| Spliced ΔCE (nats), same (L8 / L11) | 0.133 / 0.204 | 0.062 / 0.096 | **TEST**, improvement |
| ΔCE, R4 KL fine-tune vs MSE control (L0 fixed at 128) | 0.062 | 0.051 | **TEST**, improvement (NMSE +2 %) |
| GoT cities re-fit truth gap | 0.061 | 0.056 | **TEST**, no clear change (0/34 cells) |
| RepE-LM persona re-fit gap | 0.156 | 0.108 | VALIDATION, improvement (no test split exists) |
| Causal preservation T_rec/T_full | 0.625 | 0.726 | VALIDATION, improvement |
| T_rec/T_full, R4 KL vs MSE control (L11) | 0.694 | 0.671 | VALIDATION, **regression** |

* R2 changed data (100M fresh tokens vs 101,600), width (12,288 vs 3,072) and recipe together. The gain is not
  attributed to any one of them.
* No trained SAE beats PCA at equal error on GoT or persona information.

### 6.1 Reconstruction (TEST, 108 runs; `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/`)
* **EV falls with depth.** OWT test EV goes from 0.93–0.95 at layers 0–2 to 0.757 (ReLU), 0.815 (TopK) and 0.815
  (JumpReLU) at layer 11.
* **107 of 108 runs are matched.** The exception is layer-0 JumpReLU seed 42 (L0 73.6, about 96 % dead latents).
* **Matched leader by layer:** ReLU at layer 1; ReLU ≈ JumpReLU at layer 2; JumpReLU at layers 3–9; JumpReLU ≈ TopK at
  layers 10–11.
* **Effective width differs** (live latents of 3072): ReLU 2920–3063, TopK 1711–2851, JumpReLU 469–2947
  (`results/shared/m0_within_layer_reanalysis/architecture_by_layer.csv`).

### 6.2 M0: within-layer reanalysis (VALIDATION; `results/shared/m0_within_layer_reanalysis/`)

The test is whether, between SAEs at the **same** layer, lower reconstruction error goes with better preservation.

* **Behaviour:** yes. Lower error goes with less splice damage (β = 0.165 per log NMSE, Holm p = 0.0008; per-layer
  ρ 0.80–0.97 from layer 3 on).
* **Truth readout:** no usable relation. A formal pass (Holm p = 0.044) depends entirely on architecture fixed effects
  and disappears without them (p = 0.89).
* **Earlier claims corrected:**
  * the pooled "reconstruction tracks honesty preservation" correlation (ρ 0.86–0.89) is a pooling artefact
    (within-layer ρ = 0.05), and it came from the length-confounded task;
  * "SAE errors damage 1.0–2.6× more than random errors" is really 0.76–2.89 per run (median 1.64; 7/107 runs below 1).

### 6.3 M1: RepE-LM equal-length persona task (gates on TRAIN; `results/repe/repe_lm_gates/`)
* **The confound is gone by construction:**
  * every pair has the same length and differs in one token;
  * the residual stream before the swap is bit-identical;
  * every shortcut AUROC is exactly 0.5000.
* **Persona meaning transfers only modestly** across word pairs. The leave-one-pair-out AUROC is 0.701 at layers 8 and
  11, a narrow pass of gate K5.
* **The persona swap changes GPT-2's next-token distribution about 2× more than a same-meaning synonym swap**
  (median ratio 1.99, Wilcoxon p = 3e-68). The absolute effect is small: median KL 0.019 nats.
* **Template T1** (truthful/deceptive) behaves like a synonym swap.

### 6.4 M2: information preservation with references (VALIDATION; `results/shared/m2_four_level_faithfulness/`)

The table gives gaps averaged over informative layers (lower is better). Per-layer values are in
`summary_by_layer_arch.csv`.

| Task | Frozen logistic | Noise of the same norm | Frozen mass-mean | Re-trained logistic | PCA at equal error (re-trained) |
|---|---|---|---|---|---|
| GoT, ReLU / TopK / JumpReLU | .287 / .275 / .286 | .259 / .246 / .257 | .033 / .024 / .026 | .066 / .075 / .118 | .042 / .032 / .030 |
| Persona, ReLU / TopK / JumpReLU | .198 / .162 / .155 | .076 / .072 / .064 | .091 / .066 / .062 | .179 / .147 / .143 | .131 / .116 / .114 |

* **The earlier "truth drops to near chance" result was mostly fragility of the frozen probe.** Noise of the same norm
  causes almost the same drop, and the simpler mass-mean reader loses only 0.03.
* **A real but smaller loss remains** for re-trained readers. No cell is "Preserved".
* **SAEs are never better than PCA** at equal OWT reconstruction error (worse in 52/81, 59/99 and 81/90 cells).
* **The lost information is in the error term** (error-only AUROC 0.62–0.77).
* **Architecture differences are confounded by prompt-level sparsity**, which is not matched. On GoT, TopK uses 128
  latents, ReLU 32–110 and JumpReLU 23–92 (layers 1–11). This is what E1 addresses.

### 6.5 Behavioural faithfulness (VALIDATION; `results/shared/spliced_evaluation/`)

GPT-2 is run with x̂ in place of x at every non-BOS position. Controls: identity splice (exactly 0), norm-matched
random error, mean ablation.

* **CE increase grows with depth:**

  | Layer | ReLU / TopK / JumpReLU (nats/token) |
  |---|---|
  | 1 | 0.019 / 0.026 / 0.009 |
  | 6 | 0.177 / 0.104 / 0.092 |
  | 11 | 0.358 / 0.213 / 0.225 |

  Loss recovered against mean ablation stays ≥ 0.93.
* **GoT `cities` truth preference** (0.82 originally) holds through layer 8 and drops 0.04–0.09 at layers 9–11.

### 6.6 M3: where GPT-2 carries the persona effect (TRAIN selection, VALIDATION confirmation; `results/repe/causal_site_sweep/`)
* **The effect moves** from the persona word (layer 1: 97 % restored by patching it) to the final token (layer 11:
  95 %), with the handover at layers 7–8.
* **Confirmed sites:** layer-1 persona slot and layer-11 final token, plus layer-8 final token (amendment A1, confirmed
  without SAEs). 50 of 65 cells pass.

### 6.7 M4: does the SAE reconstruction carry the causal effect? (VALIDATION; `results/repe/sae_patch_decomposition/`)

The original activation difference between the honest and dishonest prompt is split into its SAE-reconstructed part
and its error part. Each part is patched separately.

| Site | Preserved by reconstruction, ReLU / TopK / JumpReLU (mean of 3 seeds) |
|---|---|
| Layer-1 persona slot | 0.94 / 0.93 / 0.97 |
| Layer-8 final token | 0.62 / 0.62 / 0.66 |
| Layer-11 final token | 0.68 / 0.62 / 0.68 |

* **At mid and late sites about a third of the effect is not carried by the reconstruction alone.**
  * For ReLU, the error part carries 40–45 % on its own.
  * For TopK and JumpReLU the reconstruction and error interact (|I|/T_full 0.26–0.38), so their shares cannot be read
    as additive.
* **Higher EV does not mean higher causal preservation.** At layer 11, TopK has higher EV than ReLU (0.815 vs 0.757)
  but keeps less of the effect (0.62 vs 0.68). This is descriptive.
* **Features:** the persona > synonym difference survives reconstruction.
  * Feature-level patches (encode → set features to the honest values → decode, error kept) beat matched random
    feature sets in 27/27 cases.
  * They need 64 features for 47–74 % of the effect. One feature gives 2–10 %.
* **All exact checks pass,** and T_full reproduces M3 to 7e-8.

### 6.8 Fine-tuning at layer 6 (VALIDATION pilots; `results/geometry_of_truth/`)

| Method | Outcome | Classification |
|---|---|---|
| Truth-aware (distil a linear truth probe) | The evaluation probe was the training teacher, so the original result was **circular** (`docs/2026-11-07_truth_aware_finetuning_audit.md`). Corrected (A/B split, independent probes): worse than reconstruction-only on every seed (B-MLP gap .189 vs .107, TopK) | No improvement |
| **Method A** (GoT reconstruction + OWT replay; L0 held within 5 %) | GoT reconstruction error −72 to −82 %. Independent MLP truth gap ReLU .321 → .245, TopK .204 → .107, JumpReLU .292 → .238: lower on 18/18 paired comparisons, but **never by more than 2 sd** (the pre-registered criterion). OWT unchanged | Preliminary |
| **E2E-inspired B/C** (+ output KL; C also + downstream NMSE) | TopK: OWT KL .115 → .102, GoT statement KL .110 → .038, truth gap unchanged. ReLU/JumpReLU drift 5–11 % in L0 (guard failed) | TopK: preliminary (behaviour only). ReLU/JumpReLU: no improvement |
| KL-weight ablation (TopK) | Behaviour gain saturates by w = 0.25; no truth trade-off to measure | Inconclusive (pre-registered) |
| P1 sparsity control (seed 42) | JumpReLU layer 2 EV .936 → .956, dead latents 65 % → 21 %; layer-0 JumpReLU still unmatched; checkpoint selection always picks step 15k (not converged) | Preliminary |

**No improvement is CONFIRMED.**
* **Causal preservation after fine-tuning (E3, 2026-11-10, VALIDATION):** Method A NO IMPROVEMENT (3/3). Output-KL
  TopK NO IMPROVEMENT; ReLU/JumpReLU NOT CREDITABLE (+0.018 to +0.022, only with L0 drift).
  See `results/repe/e3_finetuned_causal_preservation/`.
* **RepE-LM persona representation (RLM-RP, 216 SAEs, VALIDATION):** the loss is real (re-trained gap 0.08–0.22;
  2–3× the noise reference); Method A has no effect; output-KL gains are at most 0.007 AUROC.
  See `results/repe/repe_lm_representation_preservation/`.

**Multi-layer extension (IMP-L, layers 3, 8, 9; VALIDATION; 81 new fine-tunes; `results/geometry_of_truth/multilayer_finetune/`):**

| Criterion (pre-registered) | Outcome |
|---|---|
| Method A lowers the frozen B-MLP truth gap vs base **and** vs OWT-only (consistent) | **Met in 4/9 cells** (L3 TopK, L8 ReLU, L9 ReLU, L9 JumpReLU). Lower on all seeds vs base in 9/9, but the spread exceeds 2 sd in 5. GoT reconstruction error −61 to −85 %. All guards pass |
| … the same, read by a **re-trained** reader (reported, not a criterion) | **No consistent change** vs base in any cell (all CIs include 0): the frozen-probe gain is not information recovery |
| Output-KL lowers OWT and GoT spliced KL vs Method A | Lower in 9/9 cells (OWT KL −8 to −18 %). **Creditable in 5/9.** ReLU/JumpReLU at layers 8–9 raise L0 by 5.3–8.1 % (guard ±5 %) → NOT CREDITABLE |
| Output-KL truth gap vs Method A | Never better. **REGRESSION** at L3 ReLU and L3 TopK (CIs just reach 0) |

### 6.9 Historical results (kept for provenance, not current findings)
* **Original RepE prompts:** all-layer RepE probes, RepE splicing columns, the layer-6 causal test with 200 controls,
  R1 (honesty-preservation fine-tuning) and R2 (SAE-in-the-loop feature clamp). All are length-confounded.
  * R2's clamps were adverse (E −0.10 to −0.85).
  * The causal test's raw-direction effect was E = 0.0048, p = 0.010; its positive control E = 0.234.
* **Gemma-2-2B layer-11 steering** (appendix). "Both signs push to TRUE" is a generic large-perturbation bias. A
  context-matched direction steers with the correct sign. 20 controls per strength, so descriptive.
* **Phase-1 Feature 974** ("Paris"). All positives contain the token "Paris", so this is token detection, not
  evidence of faithfulness. It will not be used in the report.

## 7. Methodological corrections made after instructor reviews and internal audits

| Problem | Correction | Evidence |
|---|---|---|
| Layers trained with mixed optimiser procedures (Review #7 RECON-MIXED, LAYER-MIX) | One uninterrupted 15k-step protocol for all 108 runs; EV changed ≤ 0.006 | `reconstruction_evaluation/uninterrupted_protocol/original_vs_uninterrupted_layers_2_6_10.csv` |
| Unmatched architectures averaged together | ±10 % L0 rule; `architecture_summary.csv` marked INVALID | `results/openwebtext/layer_comparison/sparsity_matched_cells.csv` |
| 20 random controls cannot reach significance (NULL-UNDERPOWERED); no positive control (POSITIVE-CONTROL) | 200 controls plus an activation-patch positive control | `results/repe/steering/causal_random_controls/` (historical task) |
| "SAE steering" never put the SAE in the forward pass | Reclassified. Feature patches through encode → edit → decode with the error kept (M4); clamps (R2, historical) | M4 README §2.4 |
| Truth-aware fine-tuning evaluated by its own teacher probe | A/B split, independent probes, random-teacher control | audit document; `truth_aware_pilot/` |
| RepE persona prompts length-confounded | Equal-length RepE-LM task with gates K1–K7 | M1 |
| Frozen-probe failure read as information loss | Re-trained readers + noise and PCA references + error-term probe | M2 |
| Correlations pooled across layers | Within-layer models with permutation tests | M0 |
| Weak causal site (layer 6 carries about 20 % of the effect) | Pre-registered site sweep | M3 |
| Numerical GoT used as truth | Diagnostic only (surface shortcut) | `docs/results_registry.md` |
| RepE validation statement 326 near-duplicates train statement 589 | Excluded from all new RepE work | `data/repe/repe_lm/split_verification.json` |

## 8. Planned work (approved scope; status per row)

| ID | Experiment | Addresses | Status |
|---|---|---|---|
| **IMP-L** | Method A, OWT-only control and output-KL fine-tuning **at layers 3, 8 and 9** (3 seeds, CIs, L0 guards). Evaluated on reconstruction, truth and behaviour | Review #8 blocker "improvement result at more layers"; the 11-03 session note "fine-tune other layers" | **COMPLETED 2026-11-09 (VALIDATION)**; see 6.8 |
| **E1** | TopK trained at several k to compare architectures at matched *prompt-level* L0 (M2 readers, M4) | Prompt-sparsity confound in architecture comparisons | DRAFT PRE-REGISTRATION (\`results/shared/e1_task_l0_matched_topk/\`), awaiting approval; priority 3 |
| **E2** | Public `gpt2-small-res-jb` SAE as an external reference: reconstruction, M2 and M4 at layers 1, 8, 10, 11 | Review #8 critique 2 (small, unconverged SAEs) | **COMPLETED 2026-11-10 (VALIDATION, descriptive)**: conclusions transfer (`results/shared/e2_public_sae_reference/`) |
| **E3** | Layer-8 fine-tuning for causal preservation: base vs OWT-only vs Method A vs output-KL, pre-registered | Review #8 SAE-in-the-loop / causal improvement | **COMPLETED 2026-11-10 (VALIDATION)**; no creditable improvement |
| **R2-scaled** | Data-scaled retraining: 100M streamed tokens, width 12,288, TopK/BatchTopK/ReLU + width ablation + random decoder, layers 1/8/11 × 3 seeds | Unconverged, data-limited SAEs | **COMPLETED 2026-11-10**; OWT gains confirmed on TEST (`results/openwebtext/r2_scaled_saes/`) |
| **R4 / R5** | KL fine-tune at fixed L0 (TopK, L8/L11); rank-k causal skyline | Behaviour; reference for causal numbers | **COMPLETED 2026-11-10** (`results/openwebtext/r4_kl_finetune/`, `results/repe/r5_causal_subspace_skyline/`) |
| **M5** | One pre-registered final test pass: OWT test, GoT test (previously exposed rows) | Review #8 "say which numbers are clean" | **COMPLETED 2026-11-10** (`results/shared/m5_confirmatory_test/`); the RepE-LM test split was **not** built, so persona/causal results remain VALIDATION |
| X0–X3, T1, H1, H2, I1 | Proposals from the 2026-11-12 audit | see `docs/limitations_and_future_work.md` | PROPOSED, not approved, not run |
| Gemma SAEs | Gemma Scope at 1–2 layers | Review #7 SESSION-1027-LIST | BLOCKED (weights not cached; needs a download decision) |

## 9. Progress

| Item | Status |
|---|---|
| 108-run all-layer baseline; sparsity matching; optimiser consistency | COMPLETED (TEST) |
| Splicing (behaviour), all layers | COMPLETED (VALIDATION) |
| Layer-6 fine-tuning pilots (truth-aware audit, Method A, E2E B/C, KL ablation), P1 | COMPLETED (VALIDATION) |
| M0, M1, M2, M3, M4 | COMPLETED (VALIDATION; committed locally, **not yet on `main`**) |
| Multi-layer improvement table (IMP-L) | COMPLETED (VALIDATION; committed locally, not yet on `main`) |
| E3; RLM-RP (RepE-LM representation, 216 SAEs) | COMPLETED 2026-11-10 (VALIDATION; committed locally) |
| E2 (res-jb reference); R2-scaled; R4; R5 | COMPLETED 2026-11-10 (VALIDATION; OWT/GoT parts confirmed in M5) |
| E1 | DRAFT PRE-REGISTRATION written; NOT RUN |
| SAE-in-the-loop intervention | M4 feature patches and E3 (fine-tuned SAEs in the forward pass) done |
| M5 final test | COMPLETED 2026-11-10 (one pass; independently re-verified, `docs/final_defense/EXPERIMENT_AUDIT.md`) |
| Research paper (`research_paper/Latex/`) | IN PROGRESS: all sections, M0–M4 and IMP-L results; RLM-RP and E3; generated tables and PDF/SVG figures; compiles (29 pp) |
| Report (`reports/Latex_report/main.tex`) | IN PROGRESS (literature review only; Methods/Results are TODO) |
| SVG/PDF figures; `.drawio` source + `.svg` | IN PROGRESS: paper figures are PDF+SVG (`research_paper/Latex_Fig/`); 50 PNG in `results/` remain; `.drawio` NOT STARTED |
| Logging instead of `print` (378 calls in `scripts/` and `src/`); CI on `main` | NOT STARTED |
| Progress reports since 09-29; presentation decks in `presentation/` | NOT STARTED |
| Open PRs #16, #17, #20, #21, #22, #24 (merge plan) | NOT STARTED (needs team decisions) |
| Gemma SAE evaluation | BLOCKED |

## 10. Which numbers are clean (test-split exposure)

* **OWT test:** used once, for the 108-run reconstruction table. Never used for selection.
* **GoT geographic test:** scored for all 108 SAEs with frozen probes, and twice for the historical layer-6 fine-tunes.
  It is **not untouched.** Any further use will be labelled "previously exposed".
* **Original RepE test:** touched in a smoke run and in the historical causal test. It is retired with the confounded
  task.
* **RepE-LM test:** never built or accessed. M5 ran without it, so every persona and causal result is VALIDATION
  (proposal H1 builds it).
* **M0–M4** used train and validation only (`test_rows_used: 0` in each checks file).
* **M5 (2026-11-10):** one pass over the OWT test (100 documents) and GoT test (452 statements) for 81 SAEs. These rows
  had been scored earlier for the old reference SAEs only; the R2/R4 SAEs had never been scored on test (D31).

## 11. Known limitations
* **The original SAEs are small and data-limited.** They have 3072 latents and are trained on 101,600
  document-start tokens (about 38 passes). They are not converged: P1's validation checkpoint selection always
  picked the last step. `gpt2-small-res-jb`, for comparison, used 300M tokens and 24,576 latents.
  *Addressed 2026-11-10 by R2-scaled (100M tokens, 12,288 latents) at layers 1/8/11 only. Full current list:
  [`docs/limitations_and_future_work.md`](docs/limitations_and_future_work.md).*
* **One operating point:** L0 = 128, width 3072. Architecture rankings at other settings are untested.
* **JumpReLU** is a surrogate estimator. Up to 85 % of its latents are dead at layer 0. The kernel-STE variant's
  training script is missing, so that variant cannot be reproduced.
* **Prompt-level L0 is not matched** across architectures (see 6.4).
* **The persona effect is small:** 0.02–0.04 nats. Seeds share the training data, so seed spread understates
  uncertainty.
* **Fine-tuning has been tested at layers 3, 6, 8 and 9 only,** at one budget, and never on causal preservation.

## 12. Repository layout

```
scripts/openwebtext/       01–16  data, activations, SAE training, 108-run baseline, P1; 17 R2 streaming training; 18–19 R4
scripts/geometry_of_truth/ 01–23  GoT activations, splits, probes, fine-tuning (truth-aware, Method A, E2E, KL ablation, IMP-L)
scripts/repe/              01–19  original RepE pipeline (historical); 20–24 RepE-LM (M1), site sweep (M3), decomposition (M4);
                           25–27  E3, RLM-RP, R5 skyline
scripts/shared/                   splicing, frozen probes, M0, M2, SAE ladder harness, R2 comparisons, E2, M5 (+ verification),
                                  checkpoint registry, manifest
scripts/gemma/01_steering_sweep.py
src/evaluation/                   faithfulness_metrics.py (shared metrics), information_levels.py (M2), patching.py (M3/M4)
src/sae/                          checkpoint_registry.py (lookup), modern_sae.py (R2 TopK/BatchTopK/ReLU/JumpReLU)
src/tests/                        35 tests
results/{openwebtext,geometry_of_truth,repe,gemma,shared}/   each folder has a README with status labels
data/                             datasets and splits (activation caches are gitignored)
docs/                             RESEARCH_PROGRESS, RESEARCH_ROADMAP, RESEARCH_DECISIONS, EXPERIMENT_RESULTS_SUMMARY,
                                  EXPERIMENT_CHANGELOG, results_registry, experiment_manifest.csv, checkpoint_registry.csv
reports/, presentation/, research_paper/   capstone deliverables
```

**Not in Git**, kept only on the training machine: SAE checkpoints (`*.pt`, about 9 GB), activation caches
(`data/activations/`, `data/*/activations/`), logs.

## 13. Reproducing the results

**Environment:**
* Python 3.14, PyTorch 2.14.1 (CUDA 13), transformer-lens 3.9.0, transformers 5.21.0, sae-lens 6.54.12
  (`requirements.txt`).
* AWS g5.xlarge (NVIDIA A10G, 23 GB GPU, 15 GB RAM).
* Run every command from the repository root.

```bash
pip install -r requirements.txt
python -m pytest -q src/tests                     # 35 passed on 2026-11-08
```

**OpenWebText baseline:**
```bash
python scripts/openwebtext/01_prepare_dataset.py
python scripts/openwebtext/02_extract_activations.py
python scripts/openwebtext/13_consistent_protocol_retrain.py --layers 0 1 2 3 4 5 6 7 8 9 10 11
python scripts/openwebtext/14_sparsity_matching_analysis.py
python scripts/openwebtext/15_all_layer_baseline_analysis.py
python scripts/shared/build_checkpoint_registry.py
```

**GoT activations, frozen probes, splicing:**
```bash
python scripts/geometry_of_truth/01_extract_got_activations.py && python scripts/geometry_of_truth/02_create_grouped_splits.py
python scripts/shared/all_layer_frozen_probe_evaluation.py --dataset got
python scripts/shared/spliced_reconstruction_evaluation.py
python scripts/shared/spliced_reconstruction_analysis.py
```

**M0–M4 (validation only; none of these scripts reads test rows):**
```bash
python scripts/shared/m0_within_layer_reanalysis.py
python scripts/repe/20_build_repe_lm_dataset.py
python scripts/repe/21_extract_repe_lm_activations.py
python scripts/repe/22_repe_lm_gates.py
python scripts/shared/m2_four_level_faithfulness.py          # --smoke for a quick check
python scripts/repe/23_positive_control_site_sweep.py        # --smoke --smoke-out DIR
python scripts/repe/24_sae_patch_decomposition.py            # --smoke --smoke-out DIR
```

**Layer-6 fine-tuning.** Each script runs in stages and refuses to overwrite existing outputs:
```bash
python scripts/geometry_of_truth/11_truth_aware_pilot.py --prepare|--dry-run|--train ; python scripts/geometry_of_truth/12_truth_aware_pilot_evaluation.py
python scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py --prepare|--dry-run|--train ; python scripts/geometry_of_truth/15_method_a_evaluation.py
python scripts/geometry_of_truth/17_e2e_inspired_finetune.py --prepare|--dry-run|--train ; python scripts/geometry_of_truth/18_e2e_inspired_evaluation.py
python scripts/geometry_of_truth/20_kl_weight_ablation.py --verify|--prepare|--dry-run|--train ; python scripts/geometry_of_truth/21_kl_weight_ablation_evaluation.py
```

**Improvement phase (2026-11-10).** These need the streamed OWT shards (`data/openwebtext/stream_shards/`, gitignored)
and a GPU. Scripts check their pre-registration hashes and refuse to overwrite outputs. Output file names below are
those committed in each `ladder/` folder.
```bash
python scripts/openwebtext/17_streaming_sae_training.py --prepare-data      # tokenise shards 78/79, excluding our sample
python scripts/openwebtext/17_streaming_sae_training.py --run               # R2: 39 SAEs, ~5.5 h on an A10G
python scripts/shared/sae_ladder_evaluation.py --layer 8 --sources old --out results/openwebtext/r2_scaled_saes/ladder/ladder_layer8_old.csv
python scripts/shared/sae_ladder_evaluation.py --layer 8 --sources r2,resjb --out results/openwebtext/r2_scaled_saes/ladder/ladder_layer8_r2_resjb.csv
python scripts/shared/r2_comparisons.py --ladder-dir results/openwebtext/r2_scaled_saes/ladder --out results/openwebtext/r2_scaled_saes/comparisons.csv
python scripts/openwebtext/18_kl_finetune_fixed_l0.py                       # R4 (--smoke DIR for a quick check)
python scripts/shared/sae_ladder_evaluation.py --layer 8 --sources r4 --out results/openwebtext/r4_kl_finetune/ladder/ladder_layer8_r4.csv
python scripts/openwebtext/19_r4_analysis.py                                # R4 comparisons K1-K4
python scripts/repe/27_causal_subspace_skyline.py                           # R5 (--smoke DIR)
python scripts/shared/e2_public_sae_reference.py --check                    # E2 adapter check; --analyse <ladder dirs>
python scripts/shared/m5_confirmatory_test.py                               # M5 test pass (already run once; refuses to overwrite its outputs)
python scripts/shared/m5_independent_verification.py                        # read-only re-check of M5
```
Repeat the ladder commands for layers 1 and 11. Because outputs already exist, rerunning a committed step requires a
new `--out` path; the scripts never overwrite results.

Each results folder's `README.md` and `preregistration.json` record the exact seeds, hyperparameters, script hashes
and checkpoint hashes. `docs/experiment_manifest.csv` has one row per experiment.

## 14. Remaining work checklist (deadline 2026-12-08; instructor checkpoints 11-10, 11-17, 11-22, 12-01)
- [x] Multi-layer improvement table (layers 3, 8, 9; 3 seeds; CIs). **Due 11-10**
- [x] Paper skeleton in `research_paper/Latex/` with the existing tables. **Due 11-10**
- [x] E3 (2026-11-10)
- [x] E2 (2026-11-10); [ ] E1 (draft pre-registration only)
- [x] M5 final test, run once (2026-11-10)
- [x] R2-scaled, R4, R5 (2026-11-10)
- [ ] RepE-LM test split (H1) and the other 2026-11-12 proposals (`docs/limitations_and_future_work.md`), only after approval
- [ ] Choose a journal and use its template. **Due 11-17**
- [ ] SVG/PDF figures; diagram source + `.svg`
- [ ] Revised presentation and the 10-20 deck in `presentation/`; progress reports for weeks 5–15
- [ ] CI on `main`; logging instead of `print`; AxBench, SAEBench and Paulo et al. in the bibliography
- [ ] Merge plan for PRs #16, #17, #20, #21, #22, #24; meaningful peer reviews
- [ ] Off-machine backup of the checkpoints

## Licence

MIT (`LICENSE`).

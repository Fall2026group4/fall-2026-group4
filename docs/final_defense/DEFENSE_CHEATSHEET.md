# Professor defense cheatsheet (study this first)

All numbers were verified on 2026-11-10 against committed files (`RESULTS_VERIFICATION_TABLE.md` has the sources).
**TEST** = M5, one pre-registered held-out run. **VALIDATION** = everything else.

## 1. The one-sentence answer

Training our SAEs on 1000× more data at 4× width, at the same L0 of 128, cut reconstruction error by 31–74 % and the
damage to GPT-2's next-token predictions by 35–79 % on held-out test data. It did **not** measurably improve how well
truth information is preserved, and the SAEs still never beat PCA at equal reconstruction error (though at an equal
128-dimensional budget they are never worse and sometimes better; validation). **Reconstruction,
behaviour, information and causal faithfulness are separate properties: improving one does not imply the others.**

## 2. Research question, in plain words

* **Why SAEs:** an SAE rewrites a 768-dim activation as a sparse sum of about 128 active "features" out of thousands,
  and these features are often human-interpretable. Interpretability claims ("feature 974 = Paris") assume that the
  SAE faithfully represents what the model computes.
* **The problem:**
  * High reconstruction (EV 0.8–0.98) is what everyone reports.
  * But the 2–20 % that is lost might be exactly what matters: truth information, the model's next-token behaviour,
    or the causal pathway of an instruction.
* **Faithfulness** is measured at four separate levels:
  1. **Reconstruction:** NMSE, EV, cosine.
  2. **Behaviour:** splice the reconstruction into GPT-2 and measure ΔCE and KL.
  3. **Information:** can truth or persona still be read out? Frozen and re-fit probes, against PCA and noise references.
  4. **Causal:** does the reconstruction carry the persona instruction's effect on outputs? M4 patching.
* **Contribution:** a controlled, L0-matched, multi-seed, pre-registered measurement of all four levels across layers
  and three architectures, with noise, PCA, random-decoder and DAS references, plus one confirmatory test pass. It
  shows that the levels dissociate.

## 3. Frozen vs trained (say this precisely)

* **GPT-2 Small weights are frozen in every experiment.** Training scripts hash its parameters before and after
  (`gpt2_parameters_unchanged: true` in `training_info.json` / `finetune_info.json`).
* **Only the SAE's parameters are trained:** encoder W_enc and b_enc, decoder W_dec and b_dec, and for JumpReLU the
  thresholds.
* SAEs read and write `blocks.L.hook_resid_pre`. BOS is excluded.

## 4. The pipeline in one line

OWT text → GPT-2 tokenizer (≤ 128 tokens) → frozen GPT-2 → residual stream `blocks.L.hook_resid_pre` (768-d) → SAE
encoder → sparse codes (L0 ≈ 128 of 3,072 or 12,288) → SAE decoder → x̂ → evaluate:
* (a) compare x̂ with x (NMSE, EV, cosine);
* (b) put x̂ back into GPT-2 and measure ΔCE and KL;
* (c) apply probes to x̂ on GoT and RepE-LM prompts;
* (d) patch the persona difference through x̂ (M4).

## 5. What changed between SAE generations

| | Old (pre-review) | Corrected baseline | R2 improved | R4 KL fine-tune |
|---|---|---|---|---|
| Training data | 101,600 OWT tokens (about 38 passes) | same | **100M fresh tokens** (OWT shard 78; 98 docs matching our sample excluded) | +5M tokens (shard 79) |
| Width | 3,072 | 3,072 | **12,288** (TopK-3072 width ablation) | 12,288 |
| Optimiser | Adam 3e-4, batch 256, clip 10; **mixed procedure across layers** | one uninterrupted 15k-step run, 3 seeds | Adam 2e-4, batch 4096, warm-up 500, linear decay over the last 20 %, clip 1, input scaling | Adam 2e-5 |
| Architectures | ReLU+L1 (controller), TopK, JumpReLU (sigmoid surrogate) | same | TopK+AuxK, BatchTopK+AuxK, ReLU (decoder-norm L1, controller); **modern JumpReLU abandoned** | TopK only |
| Objective | MSE + sparsity | same | MSE (+ AuxK for dead latents) | NMSE + α·KL(p_orig ‖ p_spliced), vs an NMSE-only control |
| Layers | 2/6/10 (then all 12) | all 12 | 1, 8, 11 | 8, 11 |

## 6. Five numbers to remember (all TEST unless stated)

1. **Reconstruction:** L8 TopK NMSE 0.160 → 0.094 (−41 %); L11 0.185 → 0.111 (−40 %). Across all architectures and
   layers: −31 to −74 %, every seed.
2. **Behaviour:** L8 TopK ΔCE 0.133 → 0.062 nats (−53 %); L11 0.204 → 0.096 (−53 %). Across all: −35 to −79 %.
3. **R4 KL fine-tune:** ΔCE −17 % (L8, 0.0623 → 0.0514) and −16 % (L11, 0.0964 → 0.0807) vs an MSE control at fixed
   L0. The cost is NMSE +2 %.
4. **GoT truth:** 0 of 34 GoT cells improved on TEST. A random frozen decoder gets an SAE − PCA gap of −0.003 (L8) and
   −0.013 (L11); trained SAEs get +0.008 to +0.081.
5. **Causal (VALIDATION):** the reconstruction carries 62–68 % of the persona effect at L8/L11 for old SAEs, and
   0.73/0.75 (TopK/BatchTopK, L8) for R2. A learned rank-4 subspace (DAS) carries 0.91 at L8.

## 7. Architecture answers (matched L0 = 128 on OWT)

* **Best reconstruction:**
  * Old SAEs (108-run test table): JumpReLU at layers 3–9; ReLU at L1; JumpReLU ≈ TopK at L10–11.
  * R2: BatchTopK best at L1, L8 and L11. Second place is ReLU at L1 and TopK at L8/L11.
* **Best behaviour (lowest ΔCE, TEST):**
  * Old SAEs: JumpReLU at L1 and L8; TopK at L11 (0.204 vs 0.216).
  * R2: BatchTopK at every tested layer. It does **not** strictly
  follow NMSE: at L1, old ReLU has the lowest NMSE (0.063), but old JumpReLU has the lowest ΔCE (0.0088 vs 0.0141).
* **Best truth re-fit gap (GoT):** **ReLU** (0.028–0.053), the worst reconstructer. Its GoT-prompt L0 differs, so this
  is confounded and descriptive.
* **Causal (VALIDATION):** JumpReLU (old) and BatchTopK (R2) are highest at L8. ReLU's error term carries 40–45 % of
  the effect at L8/L11.
* **Caveat:** task-prompt L0 is **not matched**. On GoT, ReLU uses 77–112 latents, JumpReLU 63–93 and TopK 128.

## 8. What failed, and why it matters (do not hide it)

* **Truth-aware fine-tuning:** circular in its first form, because the evaluation probe was the teacher. Corrected
  with independent probes, it is **worse** than reconstruction-only (MLP gap 0.189 vs 0.107).
* **Method A** (GoT reconstruction fine-tune):
  * the frozen-probe gap fell in 9/9 IMP-L cells, but the pre-registered criterion was met in only 4/9;
  * re-trained readers show **no** change;
  * E3 causal: NO IMPROVEMENT.
* **Output-KL fine-tuning for ReLU/JumpReLU:** L0 drifted by 5–11 %, so NOT CREDITABLE.
* **Modern JumpReLU (STE) in R2:** abandoned, because L0 would not settle (1660, or oscillating between 19 and 506).
* **R4:** improves behaviour but **reduces** causal persona preservation at L11 (−0.024, all seeds; VALIDATION).
* **SAE features vs DAS:** at k = 4, SAE features carry 8–16 % of the persona effect; PCA carries 52–100 % and DAS
  62–102 %.

## 9. Traps in questioning, and the right answer

| Trap | Right answer |
|---|---|
| "EV 0.91, so the SAE is faithful." | No. At L8, R2 TopK still adds 0.062 nats/token (about 1.9 % of CE 3.30), and its error hurts more than norm-matched random noise (ratio 1.20). Old ReLU at L8 has cosine 0.946 yet adds 0.218 nats. |
| "KL went down, so you minimised KL." | Distinguish the **KL loss coefficient α** (a training weight of 1.1–1.5, set by scale matching) from the **measured spliced KL** (0.056–0.081 on test). |
| "So R2 is better at truth." | No. GoT is NO CLEAR CHANGE everywhere on TEST, and at L8 width made it worse (A2 regression). |
| "Is the improvement from more tokens?" | A1 changes data **and** recipe together (AuxK, LR schedule, batch, input scaling), so tokens alone are not isolated. The 10M/25M/50M checkpoints exist but were **not evaluated**. |
| "Is BatchTopK a better JumpReLU?" | No. C3 compares different architectures. We have no improved JumpReLU. |
| "Statistically significant?" | 3 seeds; Welch CI plus all 9 pairs. No multiplicity correction across 136 test rows. The seed CI does not include test-document sampling. The OWT effects are large (CIs far from 0); the GoT effects are not. |
| "Is the test set clean?" | It is held out from all R2/R4 selection. But the OWT and GoT test rows had been scored before for the old SAEs (Nov 4–7), so by our own README rule they are "previously exposed". |
| "Gemma?" | Steering only, with no SAEs. Gemma Scope evaluation (R3) is planned and was not run. |

## 10. Two-minute presentation script

> "Sparse autoencoders are the main tool for reading features out of language models, and their quality is usually
> judged by reconstruction: how much variance they explain. We asked whether good reconstruction means the SAE is
> faithful to what the model actually uses, at four levels: reconstruction, the model's own predictions when we
> splice the SAE in, whether truth and persona information can still be read out, and whether the causal effect of an
> instruction passes through the reconstruction.
>
> We trained ReLU, TopK and JumpReLU SAEs in house on frozen GPT-2 Small, at matched sparsity of 128 active features,
> with three seeds. After the November review we fixed an optimiser inconsistency and enforced L0 matching; that
> changed explained variance by at most 0.006, so the original conclusions stood.
>
> The central finding is a dissociation. Within a layer, lower reconstruction error predicts less damage to
> predictions, but not better truth readout. Truth readout by the original probe collapses, but mostly because the
> probe is fragile: same-size random noise does almost as much damage. And SAEs never beat PCA at equal error,
> although that gives PCA two to four times as many dimensions; at an equal budget the SAEs are never worse.
>
> We then improved the SAEs: 100 million training tokens instead of 100 thousand, 4× width, and modern TopK with
> dead-latent revival. In one pre-registered test pass, reconstruction error fell 31 to 74 percent and prediction
> damage fell 35 to 79 percent, on every seed and every architecture. A KL fine-tune at fixed sparsity cut damage by
> another 16 to 17 percent. But truth preservation did not improve anywhere on test, and on validation the KL
> fine-tune made the causal persona pathway worse at layer 11. A learned rank-4 subspace carries more of the persona
> effect than the whole SAE does.
>
> So better reconstruction is necessary for behaviour, but it is not evidence of faithful truth or causal structure.
> Each level has to be measured separately. Our limits: GPT-2 Small only, three seeds, and causal results on
> validation only."

## 11. Where the evidence is

* M5 test: `results/shared/m5_confirmatory_test/{m5_rows.csv,comparisons.csv,checks.json,preregistration.json}`
* R2: `results/openwebtext/r2_scaled_saes/README.md`
* R4: `results/openwebtext/r4_kl_finetune/README.md`
* R5: `results/repe/r5_causal_subspace_skyline/README.md`
* E2: `results/shared/e2_public_sae_reference/README.md`
* M2: `results/shared/m2_four_level_faithfulness/`
* M4: `results/repe/sae_patch_decomposition/`
* Professor-review corrections: `docs/archive/2026-11-04_methodology_review_corrections_report.md`
* Re-check script: `python scripts/shared/m5_independent_verification.py` (read-only; prints "ALL CHECKS PASS")

# Professor questions and answers (20)

Each answer follows the same format:
* **Q:** the question.
* **Short:** a one-line answer.
* **Technical:** the explanation.
* **Evidence:** the exact supporting result.
* **Source:** where to find it.

All numbers were verified on 2026-11-10. TEST means M5; everything else is VALIDATION unless stated.

---

### 1. Why is this research necessary? Isn't high reconstruction enough?
* **Short:** No. An SAE can reconstruct 90 % of the variance and still change the model's predictions, lose truth
  information, or miss the causal pathway.
* **Technical:**
  * EV weights every direction by its variance.
  * The model's downstream computation is sensitive to low-variance directions that EV barely penalises.
  * Interpretability claims rest on the SAE being faithful to what the model *uses*, not just to its geometry.
* **Evidence (TEST):**
  * R2 TopK at L8 has EV 0.906 and cosine 0.973, yet splicing it in raises CE by 0.062 nats/token.
  * Its error is 1.20× as harmful as norm-matched random noise.
  * Old ReLU at L8: cosine 0.946, ΔCE 0.218.
* **Source:** `results/shared/m5_confirmatory_test/m5_rows.csv`

### 2. Why SAE and not just PCA?
* **Short:** PCA is our reference, not our method. SAEs are used for interpretability because their features are
  sparse and often monosemantic, which PCA components are not. We test whether that sparsity costs faithfulness.
* **Technical:** at matched OWT reconstruction error, we compare how much truth/persona information an SAE
  reconstruction loses against a PCA reconstruction with the same NMSE.
* **Evidence:**
  * M2 (VALIDATION): the SAE is better than PCA in 0 cells, and worse in 52/81 GoT, 59/99 persona and 81/90 neutral
    cells.
  * M5 (TEST): every trained SAE has SAE − PCA > 0 on GoT (+0.008 to +0.081).
  * E2: the public res-jb SAE also never beats PCA.
* **Source:** `results/shared/m2_four_level_faithfulness/`, M5 rows, `results/shared/e2_public_sae_reference/README.md`

### 3. Why compare ReLU, TopK and JumpReLU?
* **Short:** they are the three dominant ways to enforce sparsity: an L1 penalty, a hard top-k, and a learned
  threshold. Each distorts reconstructions differently.
* **Technical:**
  * ReLU + L1 shrinks activations.
  * TopK fixes exactly k active latents per token.
  * JumpReLU learns per-latent thresholds. Ours uses a sigmoid surrogate, not the published estimator.
* **Evidence:**
  * The old SAEs at matched L0 rank JumpReLU ≥ TopK > ReLU in reconstruction at layers 3–9 (TEST, 108 runs).
  * ReLU has the best GoT re-fit gap (TEST L8 0.034 vs 0.058–0.061).
  * At L8/L11, ReLU's error term alone carries 40–45 % of the persona effect (M4).
* **Source:** `docs/EXPERIMENT_RESULTS_SUMMARY.md` §1, §16; M5 rows

### 4. How did you match sparsity, and why does it matter?
* **Short:** every comparison is at OWT L0 = 128 ± 10 % (fine-tunes within ±5 % of their base). Unmatched runs are
  excluded from rankings but kept on file.
* **Technical:** more active latents give better reconstruction for free, so comparing at unequal L0 conflates
  architecture with capacity.
* **Evidence:**
  * In the 1000-step sweep only 15/36 cells were fully matched, and JumpReLU hit L0 32 at only 2/12 layers.
  * All M5 SAEs are matched (L0 127.6–133.4).
  * Output-KL ReLU/JumpReLU fine-tunes drifted by +5 to +11 % and were declared NOT CREDITABLE.
  * **Limitation:** task-prompt L0 is not matched (GoT: ReLU 77–112, TopK 128).
* **Source:** `docs/archive/2026-11-04_methodology_review_corrections_report.md` §D2; M5 rows; `results/geometry_of_truth/multilayer_finetune/`

### 5. Your original SAEs used only 100k training tokens. Doesn't that explain everything?
* **Short:** that critique was tested twice, and the main conclusions survive both tests.
* **Technical:**
  * **E2:** the public res-jb SAEs (24,576 latents, 300M tokens) reproduce our conclusions. The frozen GoT loss is
    mostly fragility, the persona loss is real, the SAE never beats PCA, and the causal effect is only partially
    carried.
  * **R2:** we retrained on 100M tokens. Reconstruction and behaviour improved strongly, but the truth and
    PCA-relative conclusions did not change.
* **Evidence:**
  * E2 H3 holds at every layer; H4 gives 0.559 at L8.
  * In M5, GoT shows NO CLEAR CHANGE in every cell.
* **Source:** E2 README §2.2; `results/shared/m5_confirmatory_test/comparisons.csv`

### 6. What exactly was the optimiser inconsistency, and did fixing it change anything?
* **Short:** layer 6 was resumed with Adam state restored, while layers 2/10 were warm-started with Adam reset. We
  retrained everything uninterrupted. EV moved by at most 0.006 and no rank changed.
* **Technical:** this was a protocol-consistency fix, not an improvement. Two earlier "wins" became ties: L2 ReLU vs
  JumpReLU, and L10 JumpReLU vs TopK.
* **Evidence:** for example L2 JumpReLU 0.9293 → 0.9355 ± 0.0017, and L10 JumpReLU 0.8253 → 0.8227 ± 0.0003 (OWT test).
* **Source:** `docs/archive/2026-11-04_methodology_review_corrections_report.md` §D1

### 7. Why is MSE not sufficient?
* **Short:** MSE is absolute, scale-dependent, and blind to which directions matter.
* **Technical:**
  * The residual norm grows with depth (L1 MSE 0.09, L11 MSE 4.7 for similar relative quality), so raw MSE cannot be
    compared across layers. We use NMSE = 1 − EV within a layer.
  * Even NMSE does not tell you about behaviour.
  * Within a layer, NMSE predicts splice damage (M0 β = 0.165, Holm p = 0.0008), but it does not usefully predict the
    truth gap.
* **Evidence:** M0; M5 old ReLU at L11 has MSE 6.17 against 0.086 at L1.
* **Source:** `results/shared/m0_within_layer_reanalysis/`; M5 rows

### 8. Why AUROC instead of accuracy?
* **Short:** AUROC is threshold-free. A frozen probe applied to reconstructions often has its scores shifted, which
  wrecks accuracy at the original threshold even when the ranking survives.
* **Technical:**
  * AUROC measures ranking quality; 0.5 is chance and 1 is perfect.
  * The gap = AUROC(original) − AUROC(reconstruction); lower is better.
  * It is comparable across layers only when the original AUROC is informative (≥ 0.70, rule D8).
* **Evidence:** at layer 10, the legacy accuracy was 0.769 and AUROC 0.853. A frozen probe on reconstructions gives
  AUROC 0.639.
* **Source:** professor-review report §D3

### 9. Why both frozen and re-fit probes?
* **Short:**
  * A **frozen** probe asks: "is the original readout preserved?"
  * A **re-fit** probe asks: "is the information still there in any linear form?"
* **Technical:**
  * Frozen probes are fitted on original activations only and applied unchanged. They are sensitive to any
    perturbation.
  * So we add a norm-matched noise reference: if noise of the same size causes the same drop, the drop is probe
    fragility, not information loss.
* **Evidence:** M2, GoT: frozen gap 0.28, same-norm noise 0.25, re-trained gap 0.07–0.12. The frozen "collapse" is
  mostly fragility; a smaller real loss remains.
* **Source:** `results/shared/m2_four_level_faithfulness/`

### 10. How do you know your truth probe is independent of the training signal?
* **Short:** after we found circularity (the truth-aware loss distilled the same probe that evaluated it), we split
  the GoT train groups into A (training) and B (independent probes).
* **Technical:** no evaluation probe shares rows, labels or parameters with a training teacher. Under independent
  probes, truth-aware fine-tuning is **worse** than reconstruction-only.
* **Evidence:** corrected pilot B-MLP gap: recon-only 0.107 ± 0.020 vs truth-aware 0.189 ± 0.030 (VALIDATION).
* **Source:** `docs/2026-11-07_truth_aware_finetuning_audit.md`; `results/geometry_of_truth/truth_aware_pilot/`

### 11. What is the difference between KL and CE here?
* **Short:**
  * ΔCE = CE(spliced) − CE(original) on the true next tokens.
  * KL = KL(p_original ‖ p_spliced) over the full distribution.
  * Both are measured; lower is better, and 0 is ideal.
* **Technical:**
  * KL captures distribution change even when the top token is unchanged.
  * Separately, R4 has a KL **training term** with coefficient α = 1.14–1.51, set by first-batch scale matching.
    That is a loss weight, not a measurement.
* **Evidence (TEST, L8):** R4 MSE control → KL arm: measured ΔCE 0.0623 → 0.0514; measured KL 0.0659 → 0.0564.
* **Source:** `results/openwebtext/r4_kl_finetune/README.md`; M5 `comparisons.csv`

### 12. Is GPT-2 trained at any point?
* **Short:** never. Only SAE parameters are trained.
* **Technical:**
  * The parameter hashes before and after training match.
  * The SAE reads and writes `blocks.L.hook_resid_pre`, excluding BOS.
  * Splicing replaces that activation during a forward pass with frozen weights.
* **Evidence:** `gpt2_parameters_unchanged: true`.
* **Source:** `results/openwebtext/r2_scaled_saes/training_info.json`, `r4_kl_finetune/finetune_info.json`

### 13. Why layers 1, 8 and 11?
* **Short:** they are the three sites where M3 found the persona effect is carried: L1 at the persona slot, L8 at the
  final token at the slot→final handover, and L11 at the final token. They also span shallow, middle and deep.
* **Technical:**
  * The sites were selected on TRAIN and confirmed on VALIDATION in original GPT-2 before any SAE was evaluated
    there.
  * GoT is evaluated only at L8/L11, because L1 is uninformative (original AUROC about 0.52).
* **Evidence:** M3: the slot share is 0.97 at L1 and the final-token share 0.95 at L11, with the handover at layers 7–8.
* **Source:** `results/repe/causal_site_sweep/`

### 14. Are your results statistically significant?
* **Short:** they meet strict pre-registered decision rules, not a significance test across all comparisons.
* **Technical:**
  * IMPROVEMENT requires all 9 seed pairs to be better and a Welch 95 % CI excluding 0.
  * There are 3 seeds, and no multiplicity correction across the 136 M5 rows.
  * The CI covers seed variance, not test-sample variance.
  * The OWT effects are 10–50× their CI half-widths, so they are robust.
  * The GoT effects are null, and the two A2 regressions could be false positives.
* **Evidence:** L8 C1 NMSE −0.0658 [−0.0670, −0.0646]; GoT re-fit −0.005 [−0.053, +0.043].
* **Source:** `comparisons.csv`; `EXPERIMENT_AUDIT.md` §1.2

### 15. How did you prevent data leakage?
* **Short:**
  * OWT is split by document.
  * R2/R4 use separate OWT shards, with our 1000 sampled documents removed by hash.
  * GoT is split by city group (asserted in code).
  * Test rows are read only by M5, with a hash-guarded script.
* **Technical, two disclosures:**
  * The M5 OWT/GoT test rows had been scored before, for the *old* SAEs only. No R2/R4 choice used them.
  * Validation statement 326 near-duplicates train statement 589 in the old RepE set; it was excluded.
* **Evidence:** R2 README §1.2 (98 documents excluded); M5 `got_cities` assertion; README §10.
* **Source:** `results/openwebtext/r2_scaled_saes/README.md`; `scripts/shared/m5_confirmatory_test.py`

### 16. What does causal faithfulness mean in your work?
* **Short:** if we patch the honest-persona activation difference into an untruthful-persona run, the output moves
  toward the honest output. We ask how much of that move survives when the difference goes through the SAE
  reconstruction.
* **Technical:**
  * The score is T_rec / T_full, where T is the reduction in KL(p_H ‖ p_U).
  * Controls: synonym swaps, matched-norm and shuffled vectors, and random feature sets.
  * R5 adds a supervised DAS upper reference and a random floor.
* **Evidence (VALIDATION):**
  * Old SAEs: 0.93–0.97 at L1 and 0.62–0.68 at L8/L11.
  * R2 TopK at L8: 0.726. DAS rank 4 at L8: 0.913. Random rank 4: 0.008.
* **Source:** `results/repe/sae_patch_decomposition/`, `results/repe/r5_causal_subspace_skyline/`

### 17. What are your negative results?
* **Short:** truth preservation could not be improved by any method we tried, and behaviour-targeted training hurt
  causal preservation.
* **Technical:**
  * Truth-aware fine-tuning: circular, then worse than recon-only.
  * Method A: the frozen-probe gain is not seen by re-fit readers.
  * Output-KL: L0 drift for ReLU/JumpReLU.
  * R4: L11 causal −0.024.
  * Modern JumpReLU: abandoned.
  * SAE features at k = 4 carry 8–16 % of the effect, against DAS at 62–102 %.
* **Evidence:** M5 GoT 0/34 cells improved; R4 L11 T_rec/T_full 0.694 → 0.671 (all seeds).
* **Source:** `RESULTS_VERIFICATION_TABLE.md` §3–§5

### 18. Do your results generalise?
* **Short:**
  * To held-out OWT and GoT: yes, for the tested comparisons (M5).
  * To an external SAE: yes, descriptively (E2).
  * To Gemma, or to RepE-LM test: not tested.
* **Technical:**
  * Every validation conclusion on M5's metrics replicated on test, including the null result for GoT.
  * Persona and causal results are VALIDATION only, because there is no RepE-LM test split.
* **Evidence:** C1–C3 and A1 gave 17 improved / 7 no change / 0 regressed each on TEST, as on validation.
* **Source:** M5 README §2

### 19. Did more width help? Did more data help?
* **Short:**
  * Width (3,072 → 12,288, TopK, same recipe): yes for reconstruction and behaviour, no for truth.
  * Data plus recipe (A1): yes for reconstruction and behaviour.
  * Data alone: not isolated.
* **Technical:**
  * A1 changes token count and training recipe together.
  * The 10M/25M/50M checkpoints were saved but not evaluated.
* **Evidence (TEST):**
  * A2 NMSE −20 to −52 %; at L8 the GoT re-fit gap rose by 0.012 [+0.004, +0.020] (REGRESSION).
  * A1 NMSE −22 to −35 %.
* **Source:** `comparisons.csv`

### 20. What is novel here?
* **Short:** not a new SAE. The contribution is a controlled measurement showing that SAE faithfulness levels
  dissociate, confirmed on held-out data after a strong SAE improvement.
* **Technical:**
  * Matched L0, 3 architectures, all 12 layers for the baseline, frozen vs re-fit probes with noise and PCA
    references, behaviour splicing, causal decomposition with a DAS skyline and a random floor.
  * Pre-registration and one test pass.
  * The key new result: a 4×-wider SAE trained on 1000× more data halves behavioural damage but leaves truth
    preservation unchanged and still loses to PCA.
* **Evidence:** M5 C1 L8 ΔCE −53 %; GoT NO CLEAR CHANGE; SAE − PCA > 0 for every trained SAE.
* **Source:** M5; M2; E2

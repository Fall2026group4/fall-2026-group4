# E2: external reference with the public `gpt2-small-res-jb` SAEs (layers 8 and 10). Status: APPROVED AND RUN (2026-11-10; see 1.7)

Evidence level: **VALIDATION, descriptive**. No test row is read.

## 1. Before implementation

### 1.1 Question, motivation and hypotheses

**Question.** Do our four headline conclusions also hold for a much larger, longer-trained, externally produced SAE?

**Motivation:** Instructor Review #8 critique 2 argues that our SAEs are small and unconverged:
* width 3072;
* 101,600 training tokens;
* validation EV still rising at 15k steps.

If the conclusions hold for an SAE trained on 300M tokens with 24,576 latents, that critique does not explain them.

**The external SAE.**
* Joseph Bloom's GPT-2 Small residual-stream SAEs (`jbloom/GPT2-Small-SAEs-Reformatted`): ReLU, L1 penalty 8e-5,
  32× expansion (24,576 latents), 300M OpenWebText tokens, context 128, hook `blocks.L.hook_resid_pre`.
* Layers 8 and 10 are **already cached locally**, so nothing is downloaded. Layer 6 is not cached and not needed.

**Pre-registered hypotheses.** Each one is HOLDS / DOES NOT HOLD for the public SAE. They mirror M2/M4 and use the
same tolerances.

| ID | Our conclusion being tested | Test on the public SAE (per layer and task) |
|---|---|---|
| E2-H1 | Frozen-probe "truth loss" is mostly probe fragility | Norm-matched noise frozen gap ≥ 0.5 × the SAE frozen gap (GoT cities, RepE-LM persona) |
| E2-H2 | Re-trained readers still show a real information loss | Re-fit logistic gap > 0, statement-bootstrap 95 % CI lower bound > 0 |
| E2-H3 | SAEs never beat PCA at equal reconstruction error | SAE re-fit gap ≥ PCA re-fit gap − 0.02 (M2 tolerance), with PCA at the SAE's task NMSE |
| E2-H4 | At L8 final the reconstruction carries only part of the persona effect | `preservation_rec` 95 % CI upper bound < 0.80 (ours: 0.62–0.67), and T_rec above matched-norm and shuffle controls (M4 H4a) |
| E2-H5 (reported) | Splice damage tracks reconstruction error (M0) | OWT spliced ΔCE and KL placed against our within-layer error–ΔCE relation; descriptive |

### 1.2 Design

* **SAEs:**
  * the public SAE at layer 8 (all of H1–H5) and layer 10 (H1–H3 and H5; there is no M4 site at layer 10);
  * our 9 base SAEs at the same layer (existing values, recomputed by the same code), as the reference band.
* **Comparison groups:**
  1. public SAE vs norm-matched noise (H1);
  2. public SAE vs original activations (H2);
  3. public SAE vs PCA at matched task NMSE (H3);
  4. public SAE vs M4 controls (H4);
  5. **public SAE vs our SAEs: reported side by side, not ranked.** Width (8×), L0, training data (about 3000×),
     objective (L1 vs L0 control) and BOS handling all differ.
* **Data, all validation:** OWT validation (12,700 tokens), GoT cities + neg_cities validation, RepE-LM validation.
  Probes are fitted on train rows of original activations, as in M2.
* **BOS:** excluded from every evaluation, as for our SAEs. The public SAE was trained on all positions; we note this
  and do not adjust for it.
* **Loading:** the weights map exactly onto the shared in-house SAE class, which subtracts the decoder bias before
  encoding, like the public architecture:
  * encoder.weight = W_encᵀ, encoder.bias = b_enc;
  * decoder.weight = W_decᵀ, output_bias = b_dec;
  * architecture `relu`.
* **Code:** new `scripts/shared/e2_public_sae_reference.py`.
  * It imports the M2 reader, noise and PCA functions and the M4 decomposition functions, unchanged.
  * The only new code is the weight adapter.

### 1.3 Metrics (direction)

* **Reconstruction:**
  * OWT EV (↑), MSE (↓), cosine (↑), L0 (reported), dead fraction (↓);
  * L0 on GoT and RepE-LM prompts (reported).
* **Information:**
  * frozen logistic gap and its noise reference (↓);
  * re-fit logistic and MLP gaps (↓);
  * PCA reference gap (comparison).
* **Behaviour:** OWT spliced ΔCE and KL (↓).
* **Causal (L8 final):** `preservation_rec` (↑), Shapley share (↑), |I|/T_full (↓), and H4a/H4b/H4c as in M4.

### 1.4 Statistics

* One SAE per layer, so there are no seeds. Uncertainty comes from statement-grouped bootstrap (2000).
* Each hypothesis is a pre-stated threshold rule; there is no hypothesis test across SAEs.
* BH correction over the H2 tests (2 layers × 2 tasks), as in M2.
* Results are descriptive: they say whether a conclusion survives a different SAE. They do not rank SAEs.

### 1.5 Checks before results are read (written to `checks.json`)

* **Adapter equivalence:**
  * the shared class with mapped weights must equal `sae_lens`'s own forward pass on 4096 OWT validation rows (max
    abs diff < 1e-4);
  * our class subtracting `b_dec` before encoding is verified by the same check.
* **Reference reproduction:** our 9 layer-8 SAEs must reproduce M2's committed layer-8 gaps through the new script.
* SHA-256 hashes of the script, the imported scripts, the pre-registration and the safetensors files are recorded.
* `test_rows_used` = 0.

### 1.6 Compute

* **No training. No download.** Inference only.
* **Estimate: about 25–40 min GPU:**
  * reconstruction and splicing: a few minutes per layer;
  * M2 readers: about 6 s per cell in M2;
  * M4 at L8: one SAE, but feature controls on 24,576 latents, about 5–10 min.
* **Memory:** the 24,576 × 768 weight matrices are about 75 MB each, well within the A10G.

### 1.7 Approval, clarifications and extension (2026-11-10, before any E2 result was read)

* **Approval:** the user's 2026-11-10 full execution approval covers E2 (item R1 of
  `docs/2026-11-10_independent_research_review.md`).
* **Clarification of E2-H3:**
  * The PCA reference is matched at the SAE's **OWT validation** NMSE, exactly as in M2 (`pca_k_for_val_nmse`); 1.1
    said "task NMSE".
  * The code was written to reuse M2 unchanged, so M2's definition governs. The rule itself is unchanged.
* **Labelled extension:**
  * Layers 1 and 11 are added (public SAEs downloaded from the same release). The pre-registered layers remain 8
    and 10; rows from layers 1 and 11 are marked `extension (R1)` in `hypotheses.csv`.
  * E2-H4 is evaluated at every layer with an M4 site (L1 slot, L8 final, L11 final).
  * BH correction runs over all H2 rows (4 layers × 2 tasks, minus GoT at L1). This is stricter than 2 × 2.
* **Implementation:** the ladder rows come from the shared harness `scripts/shared/sae_ladder_evaluation.py`, which
  imports M2, M4 and the splicing code unchanged. `scripts/shared/e2_public_sae_reference.py` runs the adapter check
  and the hypothesis rules.

## 2. Results (VALIDATION, descriptive; 2026-11-10)

### 2.1 Checks and files

* **Adapter equivalence** (`checks.json`): passes at L1, L8, L10 and L11. Maximum absolute difference against
  `sae_lens` is 3.8e-6 to 1.5e-5, below 1e-4. Weight SHA-256 hashes are recorded. 0 test rows.
* **Reference reproduction:** the old SAEs' ladder rows reproduce M2/M4/splicing within the harness tolerances,
  checked when the harness was written (RESEARCH_PROGRESS §34).
* **Ladder rows:**
  * L1/L8/L11: `results/openwebtext/r2_scaled_saes/ladder/ladder_layer{L}_r2_resjb.csv`;
  * L10: `ladder/ladder_layer10_old_resjb.csv`.
* **Hypotheses:** `hypotheses.csv`, from
  `python scripts/shared/e2_public_sae_reference.py --analyse <both ladder dirs>`.

### 2.2 Hypotheses (res-jb ReLU-24576)

| ID | Layer | GoT cities | RepE-LM persona |
|---|---|---|---|
| E2-H1 (frozen loss mostly fragility: noise gap ≥ 0.5 × SAE gap) | L8* | HOLDS (0.247 vs 0.309) | DOES NOT HOLD (0.075 vs 0.247) |
| | L10* | HOLDS (0.242 vs 0.308) | DOES NOT HOLD (0.058 vs 0.242) |
| | L11 | HOLDS (0.222 vs 0.312) | DOES NOT HOLD (0.062 vs 0.250) |
| | L1 | — | DOES NOT HOLD (0.028 vs 0.150) |
| E2-H2 (re-fit gap > 0, CI low > 0, BH) | L8* | DOES NOT HOLD (0.025, CI low −0.001, q 0.031) | HOLDS (0.223, q 0.001) |
| | L10* | HOLDS (0.132) | HOLDS (0.224) |
| | L11 | HOLDS (0.144) | HOLDS (0.234) |
| | L1 | — | HOLDS (0.134) |
| E2-H3 (SAE never beats PCA: SAE gap ≥ PCA gap − 0.02) | all | HOLDS (L8 0.025 vs 0.012; L10 0.132 vs 0.028; L11 0.144 vs 0.042) | HOLDS (L1 0.134 vs 0.068; L8 0.223 vs 0.112; L10 0.224 vs 0.067; L11 0.234 vs 0.066) |
| E2-H4 (T_rec/T_full CI high < 0.80, above controls) | L8* | — | HOLDS (0.559, CI high 0.601) |
| | L11 | — | HOLDS (0.612, CI high 0.644) |
| | L1 | — | DOES NOT HOLD (0.979); as expected, our SAEs keep 0.93–0.99 there |

\* pre-registered layers; the others are the labelled extension.

**E2-H5 (descriptive):** res-jb runs at a lower OWT L0 than ours (15, 68, 56 and 61 at L1/L8/L10/L11), so it is not
matched. Its spliced ΔCE against reconstruction error:

| Layer | res-jb NMSE | res-jb ΔCE | Old SAEs: NMSE range | Old SAEs: ΔCE range | R2 TopK-12288: NMSE / ΔCE |
|---|---|---|---|---|---|
| L1 | 0.053 | 0.016 | 0.065–0.081 | 0.007–0.027 | 0.026 / 0.006 |
| L8 | 0.148 | 0.168 | 0.160–0.200 | 0.114–0.225 | 0.096 / 0.060 |
| L10 | 0.192 | 0.233 | 0.182–0.233 | 0.160–0.285 | — |
| L11 | 0.193 | 0.277 | 0.189–0.249 | 0.211–0.361 | 0.113 / 0.100 |

res-jb sits on our within-layer error–damage relation. Our R2 SAEs reach lower error and lower damage, at a higher L0
(128).

### 2.3 Inference

* **Our four headline conclusions transfer to a 24,576-latent SAE trained on 300M tokens by another group:**
  * frozen GoT probe loss is mostly probe fragility (H1);
  * the persona information loss is real, for frozen and re-trained readers (H1 fails for persona; H2 holds);
  * SAEs do not beat PCA at equal reconstruction error (H3, every layer and task);
  * the reconstruction carries only part of the persona effect at L8/L11 (H4).
* **Instructor Review #8 critique 2** (small, unconverged SAEs) therefore does not explain these conclusions.
* **One exception:** GoT re-fit at L8 (H2), where the loss is small and its CI touches 0. This matches our L8 GoT
  results, where even a random decoder has a small gap (R2 §2.2).

### 2.4 Limitations

* One public SAE family, not L0-matched to ours, trained on all positions including BOS.
* VALIDATION only; descriptive.

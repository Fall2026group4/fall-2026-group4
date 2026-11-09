# E2: external reference with the public `gpt2-small-res-jb` SAEs (layers 8 and 10). Status: DRAFT PRE-REGISTRATION, awaiting user approval

Evidence level, once run: **VALIDATION, descriptive**. No test row is read. Nothing in this folder has been run.

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

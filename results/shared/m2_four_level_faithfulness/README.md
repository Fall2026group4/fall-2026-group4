# M2: four-level information faithfulness (108 SAEs × 12 layers × GoT cities / RepE-LM persona / RepE-LM neutral)

**Status:** PRE-REGISTERED 2026-11-08 (`preregistration.json`); results in §2 once run. **Evidence label:** VALIDATION;
fitting and selection on TRAIN; test not used.

* **Question (SQ2):** when an SAE replaces GPT-2's activation, which task information can still be read, and by
  which readers? Is the loss specific to the SAE, compared with random noise of the same size and with PCA at the
  same reconstruction error?
* **Labels per cell:**
  * Preserved;
  * Readout shifted: the frozen reader fails, but a re-trained reader recovers the information;
  * Reduced for F: re-trained reader family F does worse on x̂ than on x;
  * Not recovered by tested families;
  * plus a displacement flag (task information present in the error term).

  "Irrecoverable" is never claimed.
* **Script:** `scripts/shared/m2_four_level_faithfulness.py`. **Module:** `src/evaluation/information_levels.py`, with tests.

## 2. Results (run 2026-11-08, 39 min; VALIDATION; 432 cells = 108 SAEs + 36 PCA × 3 tasks)

**Checks (all pass):**
* exact identity w·x = w·x̂ + w·e: max error 1.7e-6;
* original linear form: max error 1.5e-5;
* PCA matched to the SAE OWT validation NMSE within 0.0006 (k = 299–547 dense components);
* 101,600 OWT train tokens for PCA;
* 0 test rows.

### 2.1 Original GPT-2 (re-trained readers on x, validation AUROC)
* **GoT cities:**
  * logistic: 0.52–0.65 at layers 0–3, then 0.71–0.87 at layers 4–11;
  * MLP: 0.83–0.91 from layer 2;
  * kNN: 0.72–0.82 from layer 1.
* **Persona:**
  * logistic: 0.66 (L1) to 0.87 (L11);
  * mass-mean: 0.56–0.71;
  * MLP: 0.67–0.86;
  * kNN: ≈ 0.53–0.62.
* **Neutral:** logistic 0.59–0.83 (rising at L10–11); MLP ≈ 0.50 until L11 (0.81).
* **Layer 0** is exactly 0.5 for both RepE tasks, by construction.

### 2.2 Gaps, mean over informative layers (GoT 4–11, RepE 1–11) and 3 seeds; lower is better

| Task | Arch | Frozen logistic | Noise (same norm) | PCA frozen | Frozen mass-mean | Re-trained logistic | PCA re-trained | MLP | Sparse-z | Error-term AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| GoT | ReLU | 0.287 | 0.259 | 0.234 | 0.033 | **0.066** | 0.042 | 0.049 | 0.067 | 0.675 |
| GoT | TopK | 0.275 | 0.246 | 0.216 | 0.024 | 0.075 | 0.032 | 0.037 | 0.083 | 0.615 |
| GoT | JumpReLU | 0.286 | 0.257 | 0.216 | 0.026 | **0.118** | 0.030 | 0.066 | 0.127 | 0.632 |
| Persona | ReLU | 0.198 | 0.076 | 0.166 | 0.091 | **0.179** | 0.131 | 0.232 | 0.146 | 0.773 |
| Persona | TopK | 0.162 | 0.072 | 0.153 | 0.066 | 0.147 | 0.116 | 0.190 | 0.129 | 0.719 |
| Persona | JumpReLU | 0.155 | 0.064 | 0.148 | 0.062 | 0.143 | 0.114 | 0.177 | 0.134 | 0.721 |
| Neutral | ReLU | 0.113 | 0.048 | 0.093 | 0.049 | 0.099 | 0.066 | 0.033 | 0.085 | 0.636 |
| Neutral | TopK | 0.098 | 0.046 | 0.088 | 0.038 | 0.090 | 0.064 | 0.031 | 0.087 | 0.615 |
| Neutral | JumpReLU | 0.094 | 0.043 | 0.085 | 0.035 | 0.083 | 0.062 | 0.031 | 0.081 | 0.619 |

Per layer: `summary_by_layer_arch.csv`; persona vs neutral: `persona_vs_neutral.csv`.

### 2.3 Labels (matched SAE cells)

| Label | GoT | Persona | Neutral |
|---|---|---|---|
| Preserved | 0 | 0 | 0 |
| Readout shifted | 0 | 0 | 0 |
| Reduced for some families | 60 | 99 | 90 |
| Not recovered by tested families | 21 (layers 5–9; 12 JumpReLU, 3 ReLU, 6 TopK) | 0 | 0 |
| Uninformative (original < 0.60) | 26 | 8 | 17 |

* The typical reduced families are logistic + MLP + sparse-z; kNN is also reduced in 86 cells.
* **Displacement flag** (the task label is decodable from the error term alone, CI > 0.5): 97.5 % of informative GoT
  cells and 100 % of persona and neutral cells.
* The learning curve flags sample inefficiency in 10 GoT and 5 neutral cells (these cannot be "Not recovered").

### 2.4 Hypotheses (one-sided paired bootstrap, BH over SAEs per task; informative matched cells)
* **H2a, SAE loss > noise of the same norm (frozen readers):**
  * GoT: 40/81 logistic and 52/81 mass-mean significant; mean excess only 0.026.
  * Persona: 99/99 for both readers; mean excess 0.100.
  * Neutral: 90/90; mean excess 0.058.
* **H2b, SAE vs PCA at the same OWT error (re-trained logistic):**
  * SAE better in **0** cells on every task.
  * SAE worse: GoT 52/81 (mean +0.047; JumpReLU +0.084), persona 59/99 (+0.036), neutral 81/90 (+0.028).
* **Persona beyond neutral** (re-trained logistic gap, persona − neutral): +0.07 to +0.13 at layers 3–8, ≈ 0 at
  layers 10–11. ReLU is largest at every mid layer.

### 2.5 Sparsity and width (within-layer Spearman, descriptive; architecture-confounded with 9 SAEs per layer)
* **Persona:** the re-trained logistic gap rises with task error (ρ 0.52) and live latents (0.54). Prompt L0 shows
  no relation (0.18).
* **GoT:** the gap falls with prompt L0 (−0.55) and live latents (−0.54). It shows no relation to task error (0.06).

### 2.6 Reading (simple English)
1. **The old GoT "truth is lost" result was mostly a fragile-reader artefact.** The frozen logistic reader loses
   0.28 AUROC on x̂, but it loses 0.25 under random noise of the same size and 0.22 on PCA reconstructions. The
   simpler mass-mean reader loses only 0.03.
2. **Real information reduction exists, but it is smaller.** A reader re-trained on x̂ is 0.07–0.12 AUROC worse than
   one trained on x (GoT) and 0.14–0.18 worse (persona). Re-training never fully recovers the information, so no
   cell is "readout shifted".
3. **SAEs lose more than PCA at equal reconstruction error,** in no case less. The task information they drop is
   still in the error term (error-only AUROC 0.62–0.77).
4. **On truth, ReLU loses least and JumpReLU most**, the reverse of the reconstruction ranking. On persona, ReLU
   loses most.
5. **For RepE, part of the loss is generic one-word-change information.** The persona-specific part is clear only at
   layers 3–8.

### 2.7 Limitations
* Per-cell CIs are about ±0.035 (89–224 validation statements).
* The original readers are moderate at early layers, and kNN is weak on RepE.
* PCA uses 300–550 dense dimensions, so it is matched on error, not on code size.
* Labels depend on the 0.02 tolerance and on the four families tested.
* Validation only (M5 is the test).

### 2.8 Status
COMPLETED, VALIDATION.

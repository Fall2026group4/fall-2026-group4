# R1. Honesty representation preservation, GPT-2 Small layer 6 (2026-11-07). Status: PLANNED (awaiting approval)

Evidence level (planned): validation finding, plus one pre-registered final test pass.
Companion experiment: `results/repe/r2_sae_in_the_loop_steering_layer6/` (R2). Roadmap: R1 (new), S3, X3.

## 0. Audit of existing RepE work (2026-11-07, read-only)

| Item | State | Evidence |
|---|---|---|
| Data | 597 statements → 1194 persona prompts ("Pretend you're an honest/untruthful person …" + statement prefix); split by statement 417/90/90; final-token `hook_resid_pre` cache, all 12 layers | `data/repe/persona_pairs.csv`, `data/repe/activations/gpt2_small/repe_final_token.pt` |
| Cross-wording | 180 prompts, **validation statements only**, wording "Always give accurate…/Deliberately give inaccurate…" | `data/repe/cross_wording_validation.csv` |
| Original probes (layer 6, val) | mass-mean 0.893, logistic 0.997 AUROC (fitted on **all** train rows; layer chosen on validation) | `all_layer_frozen_probe_evaluation/` |
| Base SAE preservation (layer 6, 3 seeds each) | logistic gap ReLU ≈ 0.25, TopK ≈ 0.08–0.10; RepE-prompt L0 ReLU ≈ 164, TopK 128, JumpReLU ≈ 180 (not matched) | `all_layer_frozen_probe_evaluation/per_run.csv` (VAL, TEST, cross-wording) |
| RepE fine-tuning | **None.** Method A / E2E fine-tuned on GoT; RepE gaps "unchanged" there | `method_a_layer6/`, `e2e_inspired_layer6/` |
| Splicing | persona-shift cosine, final KL for all 108 SAEs (VAL) | `spliced_evaluation/` |
| Steering | scripts 10–12: **no SAE in forward pass** (direction from reconstructions), SUPERSEDED. Script 13 (6c): SAE features used only as **decoder directions added to the raw activation**, not SAE-in-the-loop. Raw direction E = 0.0048 (p = 0.01); positive control 0.234; all SAE-derived directions null | `steering/README.md`, `steering/causal_random_controls/` |
| Test split exposure | RepE test already scored by 6c (pre-registered), the all-layer frozen-probe TEST and `representation_preservation_val_test.csv`. No selection was ever made on it | progress §6c, §9 |

**Leakage checks run for this plan:**
* Statement → split mapping is unique; exact and normalised overlap between train and test, and between validation and test, is 0.
* **One near-duplicate crosses train/validation:** validation `statement_id` 326 ("…776 B.C.") ≈ train 589 ("…776 BC.").
  * Pre-registered handling: **exclude statement 326 (both personas) from every validation score** in R1 and R2.
* Train duplicates 436/509 ("The sun/Sun is a star.") are both train. They must fall in the same A/B half, which is ensured by grouping on the normalised statement.
* Prompt-specific L0 is unmatched across architectures. Comparisons in R1 are therefore **within architecture, paired by seed**, and RepE L0 is a guard (§4).

## 1. Question, hypothesis, motivation

* **Question:** can SAE fine-tuning improve the preservation of the honesty (persona) representation at layer 6 without losing sparsity or general reconstruction?
* **H-R1a:** RepE reconstruction fine-tuning with OWT replay (the RepE analogue of Method A) reduces the frozen-probe honesty gap relative to the base SAE.
* **H-R1b (primary):** adding a paired-contrast term (preserve x_H − x_U for the same statement) reduces the gap **beyond** H-R1a, at equal OWT and RepE L0, and the gain transfers to held-out wording.
* **Why:** reconstruction quality tracks honesty preservation (ρ = 0.86, §9), and ReLU loses about 0.25 AUROC at layer 6. Whether a targeted, non-probe objective can close the gap without gaming a probe (the D12/D16 lesson) is untested.

## 2. Design

* **Model:** GPT-2 Small, frozen. Hook `blocks.6.hook_resid_pre`, final prompt token (BOS excluded).
* **SAEs:** ReLU, TopK and JumpReLU, d_sae 3072, OWT L0 128. Bases are the `owt_uninterrupted` checkpoints, seeds 42/1042/2042 (AUTHORITATIVE in `docs/checkpoint_registry.csv`).
* **Splits:**
  * The RepE train statements are split 50/50 into **A** (fine-tuning) and **B** (independent evaluation probes), by normalised-statement group, seed 20261107, balanced by factual label.
  * The split file and its SHA-256 are recorded before training.
  * Validation (89 statements, 326 excluded) and the cross-wording set are used for evaluation.
  * The test split is touched only in §6.
* **Held-out wording for test (needs approval):** a new, third template pair, fixed now, applied to the 90 test statements:
  * "USER: You are a sincere assistant who always tells the truth. ASSISTANT: …"
  * "USER: You are a deceptive assistant who often tells lies. ASSISTANT: …"
  * File `data/repe/heldout_wording_test.csv` plus its activation cache. No new wording is created for train.
* **Arms** (per architecture × seed; all initialised from the base; GPT-2 frozen; only SAE parameters trained):

  | Arm | Loss | Status |
  |---|---|---|
  | Base | — | reused |
  | Method A (GoT recon + OWT replay) | existing | reused, reference only |
  | **R-recon** (control) | 0.5·MSE(RepE-A rows) + 0.5·MSE(OWT train) | new |
  | **R-honesty** | R-recon + λ·L_pair; L_pair = mean‖(x̂_H − x̂_U) − (x_H − x_U)‖² / mean‖x_H − x_U‖² over A pairs (same statement) | new |
  | **R-samepersona** (control) | R-recon + λ·L_pair on *same-persona, different-statement* pairs from A | new |

  * The R-samepersona control separates "honesty contrast" from "extra weight on paired differences of RepE prompts".
  * **No probe enters any loss.**
* **Protocol** (the Method A reference protocol, otherwise unchanged):
  * batch: 32 A pairs (64 RepE rows) + 128 OWT rows;
  * Adam, lr 1e-5, 500 steps, clip 10, decoder renormalisation, same batch RNG across arms;
  * no checkpoint selection.
  * **Sparsity:** TopK has L0 fixed by k. ReLU/JumpReLU keep their penalty, with the multiplicative controller continued from the base coefficient and its EMA L0 measured on OWT rows (as in Method A).
* **λ:** train-only scale matching per SAE, so that λ·L_pair = 0.5·MSE_RepE on A at initialisation. A single value, with no sweep. Recorded in `preregistration.json` before training.
* **Runs:** 3 new arms × 3 architectures × 3 seeds = **27**, plus 1 reproduction check (R-recon TopK seed 42 retrained twice, which must be bit-identical; excluded from results).
* **Known limitation, accepted:** A has about 208 statements (416 rows), so 500 steps ≈ 77 passes over A. Overfitting will show on validation.

## 3. Metrics

All metrics are scored on validation (89 statements) and cross-wording (89). The test sets are scored in §6 only.

| Metric | Category | Direction |
|---|---|---|
| **B-MLP AUROC gap** (768-64-1, 5 inits, fitted on original B rows, applied frozen) — **primary** | representation preservation | lower |
| B-logistic gap (C by grouped 5-fold CV within B; no validation use) | representation preservation | lower |
| B-mass-mean gap | representation preservation | lower |
| Pair accuracy (score_H > score_U per statement), recon vs original | representation preservation | higher |
| Margin retention: mean(s_H − s_U) on recon ÷ on original (logistic logit and mass-mean projection) | representation preservation | closer to 1 |
| Direction cosine (mass-mean of validation differences, recon vs original) | representation preservation | higher |
| Legacy all-train probes (shares A rows with training → DIAGNOSTIC only) | diagnostic | — |
| MSE, EV on OWT validation and RepE validation; cosine | reconstruction | lower / higher |
| L0 on OWT, GoT, RepE (reported separately); dead fraction (OWT) | sparsity | match / lower |
| OWT spliced CE increase and KL; RepE persona-shift cosine (splicing) | behavioural faithfulness | lower / higher |
| GoT geographic B-MLP gap (existing layer-6 evaluation) | guard | lower |

**Caveat:** L_pair targets the same contrast that the mass-mean probe reads, although on different rows and with no shared parameters. A mass-mean gain alone therefore does **not** count as evidence. The primary metric is the nonlinear B-MLP probe, and transfer to held-out wording is required.

## 4. Pre-registered criteria

These follow D14: n = 3 seeds, paired by seed, descriptive rules. "Consistent" means better on all 3 seeds AND the mean paired difference > 2 × the sd of the paired differences.

* **Per architecture:**
  * **R1a:** R-recon has a consistently lower B-MLP gap than Base (validation).
  * **R1b (primary):** R-honesty has a consistently lower B-MLP gap than R-recon (validation) AND a lower one than R-samepersona on all 3 seeds.
  * **R1-transfer:** on cross-wording, the mean paired difference (R-honesty − R-recon) of the B-MLP gap is < 0, and it is < 0 on ≥ 2/3 seeds.
* **Guards** (for each fine-tuned arm vs its own base):
  * OWT L0 within ±5 %;
  * **RepE-prompt L0 within ±5 %** (prompt-specific sparsity control, S3);
  * OWT validation EV ≥ base − 0.02 on every seed;
  * dead fraction ≤ base + 0.05;
  * OWT spliced CE increase ≤ base + 0.01 nats (mean);
  * GoT geographic B-MLP gap ≤ base + 0.02 (mean).
  * A run that fails a guard is flagged and excluded from rankings, not deleted.
* **Verdict per architecture:**
  * **SUPPORTED** = R1b AND R1-transfer AND all guards.
  * **NOT SUPPORTED** = R1b fails.
  * **INCONCLUSIVE** = otherwise.
  * One metric improving while another regresses is not success.
* **Exploratory, not part of the verdict:**
  * gap vs RepE L0 across all SAEs;
  * TopK base re-evaluated at k ∈ {164, 180}, matched to the ReLU/JumpReLU RepE L0 (off-distribution k, so labelled EXPLORATORY);
  * architecture ranking (unmatched on RepE L0).

## 5. Evaluation protocol and leakage safeguards

* Probes are fitted once on **original** B activations and saved with their hashes, before any fine-tuned SAE is evaluated.
* They are never refitted on reconstructions; refit decodability is a labelled diagnostic only.
* A rows are never used by an evaluation probe. B, validation and test rows are never read by the training script (asserted in code).
* Statement 326 is excluded from validation.
* Script, split-file, probe and pre-registration SHA-256 hashes go into `pretraining_hashes.json` before the 27 runs.
* Outputs refuse to overwrite. Checkpoints go to `results/repe/sae_finetuning/r1_honesty_preservation_layer6/<arm>_<arch>_seed<s>/`.

## 6. Final test pass (pre-registered, once)

* After the validation verdict is written and committed, all R1 SAEs (Base, Method A, R-recon, R-honesty, R-samepersona) are scored **once** on:
  * RepE test, default wording (90 statements);
  * the new held-out-wording test set.
* The metrics are those of §3. Nothing is selected on test.
* Prior test exposure (6c, all-layer) is disclosed. Those analyses involved no R1 arm.

## 7. Compute estimate (A10G 23 GB; 15 GB RAM)

* 27 fine-tunes, about 20–40 s each (no GPT-2 forward in the loss): **≈ 15–20 min**.
* Evaluation of 45 SAEs (9 base + 9 Method A + 27 new), layer-6 evaluation + probes + splicing, about 12 s each: **≈ 10–15 min**, plus the test pass ≈ 5 min.
* Held-out-wording extraction: < 2 min.
* Peak GPU memory is expected below 6 GB.

## 8. Files (planned)

* **Scripts:**
  * `scripts/repe/14_prepare_r1_r2_splits_and_wording.py`: A/B split, exclusion list, held-out wording and its cache;
  * `15_r1_honesty_finetune.py`;
  * `16_r1_evaluation.py`;
  * `19_r1_r2_figures.py`.
* **Results:** this folder (`preregistration.json`, `pretraining_hashes.json`, `per_sae.csv`, `summary.csv`, `criteria.json`, `test_pass/`).

## 9. After execution

To be completed.

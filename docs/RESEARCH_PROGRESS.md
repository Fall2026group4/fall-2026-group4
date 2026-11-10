# Research progress journal: SAE Faithfulness (GPT-2 Small, Gemma-2-2B)

The full chronological history of the project, reconstructed on 2026-11-07 from the evidence listed in §0. Update it after every experiment (see `CLAUDE.md` §1).

**Main research question.** Can we improve SAE reconstruction of frozen LLM activations while preserving truth-related information and causal behaviour across transformer layers?

**Evidence labels:**
* **[VERIFIED]**: artifacts are present on branch `sae-research`, and the numbers were read from files in this working tree.
* **[BRANCH-ONLY]**: the work exists only on an unmerged team branch. Numbers come from that branch's own committed files (read with `git show`), not re-run.
* **[UNVERIFIED]**: claimed in a report, but no artifact was found.

**Evidential status of each result:** EXPLORATORY, VALIDATION, TEST (confirmed on a held-out test split), or INCONCLUSIVE.

**Metric categories, which this journal keeps separate:**
* **reconstruction** (MSE, NMSE = 1 − EV, cosine, L0);
* **representation preservation** (frozen-probe AUROC gaps, direction fidelity);
* **behavioural faithfulness** (activation splicing);
* **causal interventions** (pre-registered, with controls);
* **steering**.

---

## 0. How this history was reconstructed

**Sources:**
* the Git history of all branches (159 commits; `sae-research` = 102);
* `docs/2026-11-04_*` (professor-review report, code audit, methodology changes);
* `docs/migration_manifest.csv`, `docs/experiment_manifest*.csv`, `docs/checkpoint_registry.csv`, `docs/results_registry.md`;
* the results folders and their READMEs;
* team progress reports (`reports/Progress_Report/2026-09-29.md`, and `2026-10-25.md` on `origin/dhruv`).

**Branch structure:**
* `sae-research` was created from `main` at `be138e8` (2026-10-12). Its own work starts with `55ca6e2` (2026-10-30).
* Work on `origin/dhruv`, `origin/simba-got`, `origin/aditi`, `origin/simba-geometry-truth-eda` and `origin/simba-got-stages-gemma-eda` was **never merged** into `sae-research` or `main`. It is documented here as [BRANCH-ONLY].

---

## 1. Project set-up and EDA (2026-09-11 to 2026-09-29). Status: COMPLETED, EXPLORATORY

* **Problem:** establish models, datasets and the research framing.
* **Done:**
  * repository structure, MIT licence, README and `proposal.md`. The original framing was "Are SAE features causally faithful to their auto-generated labels?", with GPT-2 Small and Gemma-2-2B × ReLU/TopK/JumpReLU × OWT/Pile/GoT/RepE.
  * OWT EDA: `results/tables/eda_*.csv`.
  * GoT cities EDA: `results/geometry_of_truth_observations.md`.
* **Key facts:**
  * GoT cities has 1,496 statements, 748 true / 748 false, 748 cities, each appearing once true and once false.
  * It is geographically concentrated: China 384 and India 170 of the cities.
* **Commits:** `d516f43` … `526b3d0`, `8f3814b` (GoT EDA), `354b7ac` (EDA notebook).
* **Process:** progress report 2026-09-29, responding to Instructor Review #2.
  * Steering and detection code moved to `src/`.
  * First tests and fixed seeds added.
  * The `mentions_paris` KeyError fixed.

## 2. Phase 1: pretrained SAE baseline, Feature 974 (2026-09-15). Status: COMPLETED, EXPLORATORY [VERIFIED]

* **Question:** does a public SAE give an interpretable feature for a concept?
* **Setup:**
  * GPT-2 Small at `blocks.8.hook_resid_pre`;
  * SAELens `gpt2-small-res-jb`, d_sae 24,576.
* **Result:** Feature 974 is the top feature at "Paris" (57.5 vs 20.1 for the runner-up).
  * Neuronpedia labels it "mentions of Paris".
  * It is consistent across 5 Paris contexts (57–64).
  * Against 5 other cities it averages 58.97 vs 6.91.
* **Files:** `results/phase1_observations.md`, `results/tables/feature_974_detection_results.csv`. Commit `070db9b`.
* **Learned:** this is selectivity on a handful of examples, not proof of causal faithfulness. It motivated the later controlled work.

## 3. Team-branch experiments (2026-10-03 to 2026-10-27). Status: COMPLETED on branches, NOT MERGED [BRANCH-ONLY]

### 3a. GoT legacy pipeline: public SAE, layer 10 (`main`, 2026-10-03; Simbanegavi)
* **Scripts:** `scripts/{prepare_geometry_data,cache_geometry_activations,run_geometry_probes,run_geometry_sae,run_geometry_detection,run_geometry_faithfulness}.py`. These are on `sae-research`.
* **Data and method:**
  * cities, city-grouped 70/30 split, seed 42 (1,046 train / 450 test);
  * last-token residual at layer 10;
  * SAELens layer-10 SAE.
* **Result [VERIFIED]** (files in `results/geometry_of_truth/`):

  | Probe | Accuracy | AUROC |
  |---|---|---|
  | Original logistic probe | 0.769 | 0.853 |
  | Probe **re-trained** on SAE reconstructions | — | 0.851 |

* **Later corrected (§6e):** the re-trained probe measures decodability, not faithfulness.
* **On `origin/simba-geometry-truth-eda`, [BRANCH-ONLY]:**
  * feature steering with random controls;
  * feature interpretation across frequency buckets;
  * top-k feature recovery;
  * `geometry_final_summary.md`.

### 3b. OWT × pretrained SAEs pipeline (`origin/dhruv`, 2026-10-04 to 10-25; Dhruv Rai)
* **Stages 0–6:** baseline and per-layer zero-ablation CE; SAE fidelity (FVU, L0, cosine, loss recovered) for the ReLU `gpt2-small-res-jb` SAEs at all 12 layers; feature statistics; steering cost; WikiText check.
* **Branch report:** loss recovered 96.3–99.9 %.
* **TopK `gpt2-small-resid-post-v5-32k`, layers 1–11:**
  * loss recovered 96.7–99.5 % with exactly 32 active features;
  * 3–7 % of features unused.
* **Also on this branch:** CI workflow; the `--n-random 200` control option.
* **Commits:** `0af0afe`, `d948449`, `b7c99e1`, `108bb9c`, `0114e90`, `c90b2ba`, `1d7c216`, `555f384`.

### 3c. GoT stages 1–6 and RepE stages 1–6 (`origin/simba-got`, `origin/aditi`, 2026-10-13 to 10-16; Aditi Shukla)
* **GoT:**
  * EDA with a subject-level split;
  * probes, transfer and negation;
  * SAE encoding and fidelity;
  * probes after the SAE with a null control;
  * steering against 20 random directions and 20 frequency-matched random features.
* **RepE:**
  * EDA (`results/tables/eda/repe_*` on the branch);
  * baseline directions and controls;
  * SAE fidelity with a size-matched wording control;
  * probes after the SAE with cross-wording transfer;
  * steering (honesty direction, 20 random directions, SAE features).
* **Commits:** `55c3c5b` … `eb142c0`.
* **Later superseded:** these analyses were redone on `sae-research` with in-house SAEs (§5).

### 3d. Gemma-2-2B baseline (`origin/aditi`, `d7f632c`, 2026-10-27)
* **Setup:** cities; layer table; next-word and judgment steering.
* **Layer-probe AUROC** (branch `layer_table.md`):

  | Layer | Validation AUROC |
  |---|---|
  | 0 | 0.49 |
  | 3 | 0.90 |
  | 7 | 0.98 |
  | 11 and later | 1.00 |

* **Judgment steering at layer 11:** both signs push to TRUE (±0.5 → 100 % / 97.7 % TRUE; baseline 61 %).
* **Problems:**
  * random row split, so cities leak across splits;
  * no controls;
  * the neg_cities run crashed (`gemma_neg.log`).
* **Status:** SUPERSEDED by §6d. `results/tables/gemma/neg_cities/` and the root `gemma_*.log` files are on `sae-research`.

### 3e. Gemma-2 / Pile EDA (`origin/simba-got-stages-gemma-eda`, 2026-10-24)
* The Pile token-level, context-length and truncation EDA (`results/gemma2_pile/` on the branch).

* **Process:** Preliminary Presentation 2026-10-20; Review #6 (2026-10-25).
* **Status on 2026-10-25:** the branch report said "no controlled GPT-2 causal result is merged".

---

## 4. In-house OWT SAE pipeline (`sae-research`, 2026-10-30). Status: COMPLETED; partly SUPERSEDED [VERIFIED]

* **Problem:** public SAEs differ in training data, width and procedure. To compare ReLU, TopK and JumpReLU fairly we need SAEs trained by us under one protocol.
* **Commits:** `55ca6e2` (pipeline), `8dc0b6e` (training and evaluation). Scripts `scripts/openwebtext/01–12`.
* **Data:** OWT, 800 / 100 / 100 documents, ≤128 tokens, BOS excluded. That gives **101,600 / 12,700 / 12,700 tokens**. Activations were cached for all 12 layers at `hook_resid_pre`.
* **SAE:**
  * `03_training_sae.py`: d_in 768, d_sae 3072, Adam 3e-4, batch 256, clip 10, decoder renormalised every step;
  * ReLU uses L1 with a multiplicative L0 controller;
  * TopK uses k = target;
  * JumpReLU uses a sigmoid straight-through surrogate (not the published estimator).
* **Experiments:**
  * pilot (`sae_pilot/`);
  * 1000-step sparsity sweep: L0 targets 32/64/128 × 12 layers × 3 architectures (`sae_sparsity_sweep/`);
  * convergence to 15k steps at layer 6 (5000 steps, then resumed) and layers 2/10 (warm start from the 1000-step sweep);
  * kernel-STE JumpReLU (exploratory; its training script is missing);
  * test reconstruction at layers 2/6/10 (`10_reconstruction_evaluation.py`);
  * fine-tuning generalisation (`12_*`).
* **Findings later corrected (§6):**
  * Cross-layer runs mixed optimiser procedures.
  * `architecture_summary.csv` averaged unmatched runs and is **INVALID**.
  * The 1000-step sweep measures convergence speed only.

## 5. GoT layer-6 and RepE pipelines with in-house SAEs (`sae-research`, 2026-10-30 to 11-03). Status: COMPLETED; partly reclassified [VERIFIED]

### 5a. GoT (commits `989ea71`, `0d07739`; scripts `scripts/geometry_of_truth/01–09`)
* **Data:**
  * cities + neg_cities (geographic) and larger_than + smaller_than (numerical);
  * final-token activations at all 12 layers;
  * grouped 70/15/15 splits (`data/got/splits/`).
* **Stage 3:** original logistic probes, with C chosen on validation per layer. Validation AUROC at layer 6: geographic 0.841, numerical 1.000.
* **Stage 4:** SAE truth preservation at layers 2/6/10. **Stage 5:** diagnostic probes.
* **Stage 6, fine-tuning at layer 6** (500 steps, lr 1e-5, GoT train, no sparsity term):
  * *reconstruction-only*;
  * *truth-aware*: + 0.1 × the MSE between the frozen logistic probe's logits on reconstruction and original.
* **Stages 7–9:**
  * independent MLP probe (768-64-1, 5 seeds), validation;
  * pre-registered final test (`final_test_protocol/`, `final_test_evaluation/`).
* **Test result, MLP probe** (later confirmed as the meaningful one):

  | Variant | Geographic gap, ReLU / TopK / JumpReLU |
  |---|---|
  | Base | 0.327 / 0.154 / 0.323 |
  | Reconstruction-only | 0.161 / 0.110 / 0.146 |
  | Truth-aware | 0.225 / 0.184 / 0.218 |

  The truth-aware row turned out to be circular (§10).

### 5b. RepE honesty (commits `0eaad9c`, `326b0f6`; scripts `scripts/repe/01–12`)
* **Data:**
  * persona pairs "Pretend you're an honest/untruthful person …" + statement prefix;
  * 834 / 180 / 180 prompts (417 / 90 / 90 statements × 2 personas);
  * a cross-wording set (180).
* **Probe:** mass-mean persona direction from train, final token, all layers. Original validation AUROC at layer 6 is 0.893; layer 6 was chosen on validation, so this is optimistic.
* **Measurements:**
  * reconstruction of 9 layer-6 variants;
  * representation preservation;
  * direction fidelity;
  * cross-wording.
* **Steering (`10–12`):** pilot, held-out, bidirectional, bootstrap.
  * Reported 0.93–0.99 effect correlations between raw and SAE-derived directions.
  * Later found to have **no SAE in the forward pass**; reclassified in §6c.

---

## 6. Professor-review corrections (2026-11-04). Status: COMPLETED, VALIDATED [VERIFIED]

Full report: `docs/2026-11-04_professor_review_report.md`, with `docs/2026-11-04_code_audit.md` and `docs/2026-11-04_methodology_changes.md`. Run on top of `326b0f6`, originally into `results/professor_review_corrections/`, and migrated on 2026-11-06 (§7).

* **6a. Optimiser-state consistency (Task A).**
  * Layer 6 had been resumed with Adam state restored; layers 2/10 were warm-started with Adam reset.
  * All 27 runs (layers 2/6/10 × 3 architectures × seeds 42/1042/2042) were retrained with **one uninterrupted 15k-step protocol** (`13_consistent_protocol_retrain.py`).
  * EV changed by ≤ 0.006 and rankings did not change, but two "wins" became ties.
* **6b. Sparsity matching (Task B).**
  * Under the ±10 % rule, JumpReLU matched its target at L0 = 32 in only 2/12 layers of the 1000-step sweep.
  * Only 15/36 sweep cells have all three architectures matched.
  * TopK's apparent advantage at 1000 steps reverses at 15k steps.
* **6c. Causal controls (Task C).** Pre-registered (`results/repe/steering/causal_random_controls/`), layer 6, final token, 90 test statements, 200 controls. TEST.

  | Intervention | Effect E (fraction of honest-persona output shift recovered) | p |
  |---|---|---|
  | Positive control (activation patch) | 0.234 [0.151, 0.317] | — |
  | Raw honesty direction | 0.0048 [0.0003, 0.0096] | 0.010 vs 200 random directions |
  | Any SAE-derived direction or feature | indistinguishable from controls | — |

* **6d. Gemma steering (Task D).** `results/gemma/steering_sweep_layer11/`: grouped splits, ±0.01…0.5 grid, 20 random directions per strength.
  * "Both signs → TRUE" is reproduced, but it is a generic large-perturbation response bias.
  * A judgment-context direction steers with the correct sign at every strength.
  * VALIDATION + TEST, descriptive (20 controls).
* **6e. GoT layer 10 (Task E).** The **frozen** original probe on public-SAE reconstructions gives AUROC 0.639 (gap 0.21 [0.16, 0.27]), against 0.851 for a re-trained probe. Truth stays decodable, but the original readout is not preserved.
* **6f. RepE (Task F).**
  * Values verified, with one quotation corrected: cross-wording JumpReLU 0.7394, not 0.8010.
  * Test agrees with validation.
  * SAE L0 on RepE prompts is **not matched** (170 / 128 / 187).
* **6g. Fine-tuning on identical held-out sets (Task G):** `heldout_comparison_all_datasets.csv`.

## 7. Repository reorganisation and migration (2026-11-06). Status: COMPLETED [VERIFIED]

* **Backup:** `../backups/2026-11-06_pre_migration.tar`. SHA-256 verified, and a restore test passed.
* **Changes:**
  * `scripts/Geometry of Truth/` → `scripts/geometry_of_truth/`; `results/got/` → `results/geometry_of_truth/`;
  * correction results copied into the dataset folders (originals kept, marked `MIGRATED.md`);
  * `docs/migration_manifest.csv` written (161 rows).
* **Commit:** `2ad0979`, made on 2026-11-07.

## 8. All-layer OWT baseline: 108 runs (2026-11-07). Status: COMPLETED, TEST [VERIFIED]

* **Why:** results existed only for layers 2/6/10. Faithfulness questions need every layer.
* **Design:**
  * 9 more layers × 3 architectures × 3 seeds = 81 new runs, plus the 27 existing ones;
  * width 3072, target L0 128, uninterrupted 15k steps;
  * OWT test, 12,700 tokens.
* **Result** (`results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/all_layers_*.csv`):
  * **107/108 runs matched within ±10 %.** The one unmatched run is layer 0 JumpReLU seed 42 (L0 73.6; later found to have ~96 % dead latents).
  * **EV falls with depth**, from about 0.93–0.95 at layers 0–2 to 0.76 / 0.815 / 0.815 (ReLU / TopK / JumpReLU) at layer 11.
* **Matched ranking:**

  | Layers | Leader |
  |---|---|
  | 1 | ReLU |
  | 2 | ReLU ≈ JumpReLU |
  | 3–9 | JumpReLU |
  | 10–11 | JumpReLU ≈ TopK |

  ReLU degrades fastest with depth.
* **Runtime:** 39 ± 3 s per run; ≤ 166 MB GPU.
* **Commit:** `9446e56`.

## 9. All-layer truth and honesty preservation with frozen probes (2026-11-07). Status: COMPLETED, TEST [VERIFIED]

* **Why:** test whether reconstruction quality predicts truth and honesty preservation at every layer.
* **Design:**
  * probes fitted on original train rows, C chosen on original validation, then applied frozen to reconstructions of the 108 SAEs;
  * GoT and RepE validation and test; grouped bootstrap;
  * shared metrics module and checkpoint registry added.
* **Result** (`results/{geometry_of_truth,repe}/all_layer_frozen_probe_evaluation/`):
  * **GoT:** the frozen logistic probe drops to near chance on reconstructions at every layer. The geographic gap is 0.19–0.32 at the informative layers 4–11; the numerical gap is 0.37–0.51. GoT NMSE is > 1 because the within-set variance is tiny.
  * **Validation decodability diagnostic:** a probe re-trained on reconstructions recovers most of the signal. Truth information is displaced rather than erased.
  * **RepE:** the honesty gap grows with depth and is largest for ReLU.
  * **Relation to reconstruction quality:** strong for RepE (ρ = 0.86 between gap and OWT NMSE); absent for GoT.
* **Commits:** `8919158`, `83bd9a9` (float64 mass-mean fix).

## 10. Phase 0: spliced-in causal reconstruction evaluation (2026-11-07). Status: COMPLETED, VALIDATION [VERIFIED]

* **Why:** reconstruction and probes don't show whether GPT-2's *behaviour* survives when the SAE replaces the activation.
* **Design:**
  * `hook_resid_pre` is replaced by the SAE reconstruction at every non-BOS position, or at the final token only;
  * OWT validation: CE increase, KL, top-1, loss recovered;
  * GoT val: statement KL, true-vs-false likelihood preference;
  * RepE validation: final-position KL, persona-shift cosine.
* **Controls:**
  * identity replacement and two unmodified passes: exactly 0 difference;
  * fp16 round-trip;
  * mean-ablation;
  * norm-matched random error.
* **Result** (`results/shared/spliced_evaluation/`):
  * **OWT loss increase** rises with depth, from ≈0.01–0.04 nats at layers 1–2 to 0.21–0.36 at layer 11.
  * **SAE errors hurt more than random errors of equal norm,** 1.0–2.6× in loss increase.
  * **They preserve the persona shift far better than random errors:** cosine 0.65–0.96 vs 0.20–0.73.
  * **GoT `cities` truth preference** (0.82 originally) holds through layer 8 and drops 0.04–0.09 at layers 9–11.
  * **Layer-6 bridge:** the historical resumed SAEs ≈ uninterrupted SAEs, within about one seed sd.
* **Commit:** `468018c`.

## 11. P1 sparsity-control pilot (2026-11-07). Status: COMPLETED, VALIDATION [VERIFIED]

* **Why:** fix sparsity matching and checkpoint selection before any improvement method.
* **Design:**
  * layers 0/2/6/10 × 3 architectures × seed 42;
  * controller annealing at 10k–12k steps, then frozen;
  * per-latent quantile initialisation of JumpReLU thresholds;
  * validation checkpoint selection.
* **Result** (`results/openwebtext/p1_sparsity_control_pilot/`):
  * **TopK** is bit-identical to its baseline, a pairing check.
  * **L0 targeting improved** for ReLU and JumpReLU.
  * **JumpReLU at layers 2/6 clearly improved:** layer 2 EV +0.020; dead latents 65 % → 21 %.
  * **Layer-0 JumpReLU is not fixed:** its coefficient sat at the 1e-7 floor and 90 % of latents are dead.
  * **Checkpoint selection always picked step 15k:** validation EV was still rising, so the SAEs are not converged.
* **Commit:** `ef9fec4`.

## 12. Truth-aware fine-tuning audit: circularity found (2026-11-07). Status: COMPLETED [VERIFIED]

* **Finding:** the truth-aware loss distilled the Stage-3 logistic probe's logits, and the same probe (identical weights) evaluated it.
  * That covers script 06's validation, the held-out logistic columns and the layer-6 bridge.
  * Those results are **circular**.
  * Under the independent MLP probe, truth-aware is **worse** than reconstruction-only in all 6 cells.
* **Actions:**
  * results reclassified, with all numbers kept;
  * corrected protocol designed: GoT train groups A/B; independent probes; random-teacher control.
* **Files:** `docs/2026-11-07_truth_aware_finetuning_audit.md`. Commit `1614be5`.

## 13. Corrected truth-aware pilot (2026-11-07). Status: COMPLETED, VALIDATION; criteria NOT met [VERIFIED]

* **Design:**
  * layer 6, TopK, seeds 42/1042/2042;
  * three variants: reconstruction-only, truth-aware (teacher fitted on A), random teacher;
  * 50 % OWT replay; 500 steps;
  * probes fitted on B.
* **Result** (`results/geometry_of_truth/truth_aware_pilot/`):

  | | Base | Recon-only | Truth-aware |
  |---|---|---|---|
  | Teacher-probe gap (objective attainment only) | .265 | — | .089 |
  | Independent logistic probe (B) gap | .236 | — | .205 |
  | **Independent MLP probe (B) gap** | .204 | **.107** | .189 |

  * The B-logistic probe has cosine 0.64 with the teacher, so it is only partly independent.
  * Truth-aware is worse than recon-only on the MLP probe on every seed. The primary criterion failed.
* **Commit:** `dcf1486`.

## 14. Method A: GoT reconstruction fine-tuning with OWT replay (2026-11-07). Status: COMPLETED, VALIDATION; criteria NOT met [VERIFIED]

* **Design:**
  * layer 6, ReLU/TopK/JumpReLU × 3 seeds, the reference protocol from §13;
  * sparsity controller continued from each base coefficient;
  * OWT-only continued-training control;
  * P1 JumpReLU base included (descriptive).
* **Result** (`results/geometry_of_truth/method_a_layer6/`):
  * **MLP-probe gap:** lower than base and than the OWT-only control on all 9 seeds (18/18 paired comparisons), but never by more than 2 sd. Primary criterion **not met**:

    | | Base → Method A |
    |---|---|
    | ReLU | .321 → .245 |
    | TopK | .204 → .107 |
    | JumpReLU | .292 → .238 |

  * **GoT reconstruction error:** −72 to −82 %.
  * **OWT:** unchanged. **Logistic probe:** no gain. **RepE:** unchanged.
  * **Guards:** all pass. TopK reproduction is bit-identical.
* **Commit:** `ca81ad2`.

## 15. E2E-inspired fine-tuning: Methods B and C (2026-11-07). Status: COMPLETED, VALIDATION [VERIFIED]

* **Not an exact reproduction of Braun et al. 2024.** Existing SAEs are fine-tuned for 500 steps with local reconstruction kept.
* **Design:**
  * B = A + α · KL(p_orig ‖ p_spliced);
  * C = B + β · downstream NMSE (layers 7–11);
  * GPT-2 frozen; α and β by train-only scale matching;
  * 18 runs, plus a reproduction check that is bit-identical to A.
* **Result** (`results/geometry_of_truth/e2e_inspired_layer6/`):
  * **TopK B and C meet the behavioural criteria:**

    | TopK | Method A → B / C |
    |---|---|
    | OWT output KL | .115 → .102 |
    | GoT statement KL | .110 → .036–.038 |

  * **ReLU and JumpReLU fail the L0 guard:** KL pushed OWT L0 up 5–11 %.
  * **The truth hypothesis fails for every architecture:** the MLP gap is not better than Method A.
  * **Pre-registered longer-budget flag:** raised for ReLU and JumpReLU; flag only.
* **Commit:** `764dcbb`.


## 16. KL-weight ablation, layer 6 TopK (2026-11-07). Status: COMPLETED, VALIDATION; verdict INCONCLUSIVE [VERIFIED]

* **Question:** does an intermediate output-KL weight keep Method A's independent truth-probe performance while retaining part of Method B's behavioural gain? Roadmap G8.
* **Design:**
  * Method B protocol (script 17 `train_one`, unchanged) with α scaled by w ∈ {0.25, 0.50} of α_B. α_B is the only KL coefficient, scale-matched per seed: 4.150 / 3.892 / 4.212.
  * TopK, L0 = 128, seeds 42/1042/2042, each initialised from the original uninterrupted checkpoint.
  * Method A (w = 0) and Method B (w = 1) checkpoints reused.
  * 6 new runs.
* **Verification before training:**
  * split hash `d43e8383…` matches;
  * E2E pre-training hashes still match;
  * base/A/B provenance correct;
  * saved results reproduce exactly from checkpoints;
  * a w = 1 retrain is bit-identical to Method B.
* **Results** (validation, mean ± sd, 3 seeds):

  | Metric | A | w = 0.25 | w = 0.50 | B |
  |---|---|---|---|---|
  | Geographic MLP gap | .107 ± .020 | .111 ± .015 | .112 ± .016 | .114 ± .022 |
  | OWT KL | .1152 | .1038 | .1027 | .1020 |
  | OWT CE increase | .1019 | .0928 | .0915 | .0910 |

  * w = 0.25 keeps about 85 % of B's KL gain, 83–85 % of its CE gain and about 99 % of its GoT statement-KL gain.
  * GoT reconstruction error rises with w (+13 % at 0.25, +52 % at B).
  * L0 = 128 everywhere; all guards pass.
* **Pre-registered verdict:**
  * BEHAV and truth non-inferiority pass for both weights; TRADE fails (below the A–B chord on 1/3 and 2/3 seeds).
  * Overall **INCONCLUSIVE**: A and B differ in truth by only +0.007, below the 0.02 threshold, so no truth trade-off exists to improve.
* **Learned:**
  * For TopK at layer 6, adding output KL buys behaviour at no measurable truth cost (±0.02, n = 3).
  * The behaviour gain saturates by w = 0.25.
  * The remaining question is power: seed sd of the truth gap ≈ 0.02.
* **Files:** `results/geometry_of_truth/kl_weight_ablation_layer6_topk/`, scripts `20`/`21`.

---

## 17. Where we stand (2026-11-07)

* **Answered, at validation level unless noted:**
  * At matched L0, reconstruction degrades with depth. JumpReLU ≥ TopK > ReLU at mid and deep layers. TEST.
  * Reconstruction quality ≠ truth-readout preservation (GoT). It *does* track honesty preservation (RepE) and behavioural damage (OWT).
  * Probe-targeted fine-tuning games the probe.
  * GoT reconstruction fine-tuning helps nonlinear truth readouts directionally, not decisively.
  * Behavioural KL objectives improve output faithfulness but not truth readouts.
* **Unresolved:**
  * converged sparsity sweep and EV-vs-L0 frontier (Method B of the original plan);
  * width sweep;
  * residual SAE;
  * layer-0 JumpReLU failure;
  * L0 control under KL objectives;
  * replication with more seeds;
  * any layer other than 6 for fine-tuning;
  * Gemma SAEs;
  * a final pre-registered test evaluation.
* **Not started:** see `docs/RESEARCH_ROADMAP.md`.

## 18. Repository consolidation before the GitHub upload (2026-11-07). Status: COMPLETED [VERIFIED]

* **Why:** prepare `sae-research` for upload as a clean, reproducible research repository, without a separate professor-review folder and without losing evidence.
* **What was done** (no experiment run, no result value changed):
  * Re-verified the 2026-11-06 migration: all 83 copied files have their source and destination present with matching SHA-256; the two destinations that differ were extended on purpose in `9446e56` (27 → 108 runs) and contain the 27 original rows unchanged.
  * Retired `results/professor_review_corrections/`, `scripts/professor_review/` and the duplicate `src/tests/test_professor_review.py` to `../backups/2026-11-07_retired_professor_review_folders/` (85 files, SHA-256 verified); `docs/migration_manifest.csv` has one `retire_to_backup` row per file (D20).
  * Rebuilt `docs/checkpoint_registry.csv` (361 rows; only the 30 retired-folder rows removed).
  * Rewrote `README.md`; added the missing research dependencies to `requirements.txt`; extended `.gitignore` (activation caches, `.bak`).
* **Checks:** 25 tests pass; all 90 tracked Python files compile; every script-to-script path reference resolves (except the documented old→new map in `scripts/shared/write_manifest.py`); no checkpoints, caches, logs, credentials or absolute home paths tracked.
* **Not done / open:** the EC2-only checkpoints (about 9 GB) have no off-machine backup after 2026-11-06 (roadmap §11). Planned RepE experiments R1/R2 are deliberately excluded from this commit.

## 19. Causal follow-up (2026-11-10). Status: reanalysis COMPLETED [VERIFIED]; two experiments IMPLEMENTED, NOT RUN

Work by Dhruv Rai on branch `causal-followup`, started from `main` at `33dcb9c`. No existing file's results were changed.

### 19a. Reanalysis of the 2026-11-04 causal test. Status: COMPLETED, EXPLORATORY [VERIFIED]
* **Question:** what can the layer-6 causal test (§ on `repe/13`, roadmap §5) actually show, and is "no SAE target is distinguishable from controls" the right summary?
* **Method:** `scripts/repe/14_causal_reanalysis.py`, CPU only, saved outputs only (no model run, no test row re-scored). All 28 saved p-values are reproduced exactly from the control rows.
* **Results** (`results/repe/steering/causal_reanalysis/`):
  * **Power.** 200 controls give p ≥ 0.00498. The Bonferroni threshold over the 27 SAE targets is 0.00185, so no single target could ever pass it; 539 controls would be needed. The minimum detectable effect is E = 0.0031, which is 1.3 % of the positive control (0.234). The raw direction's E is 0.0048 (2.1 % of the positive control). The median CI half-width of a target is 0.0059, wider than the minimum detectable effect.
  * **Direction of effects.** 18 of 27 SAE targets have E < 0. Two-sided, 9 targets have unadjusted p ≤ 0.05, all with E < 0: the TopK top-1 and top-10 latent directions and the JumpReLU top-1 latent direction, in every variant. After BH all nine sit at 0.0597. One-sided, 1 target is at p = 0.0498 and none survives BH (smallest 0.76).
  * **SAE targets vs the raw direction.** E(raw) − E(target) > 0 for all 27, with a paired bootstrap interval that excludes zero for all 27 (unadjusted). Reconstruction directions keep 56 % (JumpReLU), 19 % (TopK) and −21 % (ReLU) of the raw direction's E, in the same order as their cosine with it (0.87, 0.84, 0.76).
  * **Controls.** Random SAE-latent directions are not more disruptive than Gaussian directions (smallest KS p = 0.088 over 18 groups).
* **Learned:** the accurate summary is "no SAE target is better than its controls; nine selected-latent directions are worse than their controls at the unadjusted level; the test cannot confirm either after correction". H7 should be read with this power statement.
* **Unresolved:** why selected latents push the output away from the honest distribution (activation magnitude of the selected latents, or direction-adding being the wrong intervention on a latent). Addressed by 19b.

### 19b. SAE-in-the-loop causal test, layers 2 / 6 / 10 (roadmap C1 + C3). Status: IMPLEMENTED, NOT RUN
* **Script:** `scripts/repe/15_sae_in_the_loop_causal.py` (validation phase writes `preregistration.json`; test phase scores the test rows once).
* **Design:** the 27 uninterrupted SAEs at layers 2, 6, 10. Part A: graded activation patch (minimum detectable dose). Part B: latent patching with the SAE in the loop (all latents, the SAE error term, top-1/10/50 honesty latents) against norm-matched Gaussian, random-latent and activation-matched latent controls. Part C: script 13's direction steering with both signs. 600 controls per group; BH within seven pre-registered families.
* **Verified so far:** unit tests (17 pass) and an end-to-end pipeline run of both phases on a synthetic stand-in model with random SAEs (all parts, all output files, refusal to overwrite). **Not yet run on GPT-2. No result exists.**
* **Exact next steps:** on the GPU host, `--phase validation --smoke`, `--phase test --smoke`, then the two full phases; then write the results-folder README and update the documents listed in `CLAUDE.md` §1.

### 19c. Frozen-probe error decomposition and noise control. Status: IMPLEMENTED, NOT RUN
* **Script:** `scripts/shared/frozen_probe_error_decomposition.py` (`--dataset got|repe`, `--saes inhouse|public_relu|public_topk`). Validation rows only; exploratory.
* **Question:** the all-layer frozen-probe gaps (§ on all-layer frozen probes) are large although cosine is ≈ 0.98. Is that because any error of that size breaks the readout, or because the SAE error is aimed at the truth direction? Reports a norm-matched random-error control, signal retention, noise inflation and error anisotropy, and must reproduce the saved validation AUROCs.
* **`--saes public_*`** gives the first public-SAE reference under this pipeline (part of G9). The public loaders have not been executed yet.
* **Verified so far:** unit tests and an end-to-end run on synthetic RepE-shaped data with random SAEs. **Not yet run on the real caches.**

## 20. Remaining code experiments (2026-11-10). Status: IMPLEMENTED, NOT RUN

Work by Dhruv Rai on branch `dhruv-causal`. Four scripts, written so the GPU work after the Kaggle rebuild is one session. None has run on GPT-2; all four ran end to end on a synthetic stand-in model with random SAEs and a stand-in SAELens (all phases, all output files, refusal to overwrite), and 10 new unit tests pass.

* **20a. Sparsity matched on the truth and honesty prompts (roadmap S3).** `scripts/shared/sparsity_matched_truth_preservation.py --dataset got|repe`. Inference-time top-m truncation of the latents (m = 8 ... 128; for TopK this is exactly the SAE with k = m), frozen original probes, validation rows only. A cell is compared only if every architecture and seed is within ±10 % of m on the prompts; rankings use paired differences over the three seeds. Evidential status when run: validation finding.
* **20b. Geometry of Truth causal test (roadmap C4).** `scripts/geometry_of_truth/22_truth_direction_causal_test.py`. Cities pairs; outcome = change of log P(true token) − log P(false token) at the first token where the true and false statements differ; intervention at the final prefix token, layers 4 / 6 / 8 / 10. Targets: mass-mean and probe directions, SAE reconstruction / top-1 / top-10 directions, both signs. Graded oracle positive control (unembedding contrast). 600 controls; three pre-registered families.
* **20c. Public SAE splicing (roadmap G9, P3).** `scripts/shared/public_sae_spliced_evaluation.py --saes public_relu|public_topk`. Imports the Phase-0 splicing code unchanged; adds loss recovered vs mean ablation. External reference only: not sparsity-matched to the in-house SAEs.
* **20d. Figures and tables.** `scripts/shared/followup_figures.py`. Makes F1 to F7 and `tables.md` from whichever result files exist; lists the missing ones.
* **Not written:** the final pre-registered test evaluation (X3). It must come last, after the configurations to test are chosen from these results.

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

Full report: `docs/archive/2026-11-04_methodology_review_corrections_report.md`, with `docs/2026-11-04_code_audit.md` and `docs/2026-11-04_methodology_changes.md`. Run on top of `326b0f6`, originally into `results/professor_review_corrections/`, and migrated on 2026-11-06 (§7).

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

## 16a. In progress: RepE honesty R1 (representation preservation) and R2 (SAE-in-the-loop intervention) (2026-11-07). Status: R1 COMPLETED at validation (§19); R2 validation stage COMPLETED (§20); both test stages DEFERRED

* **Audit (read-only, done):**
  * No RepE fine-tuning exists.
  * No SAE-in-the-loop intervention has ever been run. Scripts 10–12 had no SAE in the forward pass; 6c added decoder directions to the raw activation.
  * An additive feature edit with the error term kept is mathematically identical to direction steering. R2 therefore uses input-dependent clamps.
  * **Leakage found:** validation statement 326 is a near-duplicate of train statement 589. It is excluded from all R1/R2 validation scores.
  * The RepE test split was scored before (6c, all-layer) but never used for selection. This is disclosed.
* **R1:** RepE reconstruction fine-tuning (R-recon) vs a paired-contrast objective (R-honesty) vs a same-persona-pair control, all with OWT replay.
  * Layer 6, 3 architectures × 3 seeds, 27 runs.
  * Probes fitted on the train-B half only. Guards on OWT and RepE-prompt L0.
* **R2:** encode → clamp the selected features towards the target-persona mean → decode + error, at the final token.
  * 9 base SAEs.
  * 200 random-feature, 200 matched-norm, unrelated-feature, raw-direction and positive controls; OWT damage test.
  * k chosen on validation; one held-out test pass.
* **Plans:** `results/repe/r1_honesty_preservation_layer6/README.md`, `results/repe/r2_sae_in_the_loop_steering_layer6/README.md`.
* **Unfinished:** everything after the plan. Nothing has been trained or evaluated, and no data files have been created.
* **Next steps after approval:**
  1. write `preregistration.json` and the hashes;
  2. script 14 (splits, held-out wording);
  3. R2 smoke test;
  4. R1 training and evaluation;
  5. R2 validation k-selection, then the test pass.

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

## 19. R1: RepE honesty-preservation fine-tuning, layer 6 (2026-11-07). Status: COMPLETED, VALIDATION; ReLU INCONCLUSIVE, TopK and JumpReLU NOT SUPPORTED [VERIFIED]

* **Question:** can fine-tuning an SAE with a paired honest-vs-untruthful contrast term improve preservation of GPT-2's honesty representation (layer 6), beyond plain RepE reconstruction fine-tuning and a same-persona-pair control, without sparsity or reconstruction damage?
* **Protocol:** pre-registration committed as drafted (`d17bb00`), then finalised before training (`211218f`, addendum §10).
  * Arms R-recon / R-honesty / R-samepersona from the 9 uninterrupted bases; Method A protocol (Adam 1e-5, 500 steps, OWT replay).
  * New split `data/repe/splits/r1_train_AB_groups.csv`: A (208 statements) trains, B (209) fits the independent probes.
  * Validation statement 326 excluded. **Test not touched**: §6 test pass deferred at the user's instruction.
* **Execution:** 28/28 runs (reproduction check bit-identical), 58 s; evaluation of 51 SAEs, 542 s. Smoke tests before freezing are disclosed in the README.
* **Results (validation, B-MLP AUROC gap, original AUROC 0.964):**
  * ReLU: base 0.198 → R-recon 0.180 → R-honesty 0.167. R-honesty < R-recon on 3/3 seeds (mean −0.0135, sd 0.0048), < R-samepersona on 3/3, transfers to cross-wording. **But** the RepE-prompt L0 guard fails (+6.8 % / +8.4 %), and the logistic probe shows no R-honesty gain (−0.0003). → **INCONCLUSIVE**.
  * TopK (0.092 → 0.101) and JumpReLU (0.072 → 0.075): R-honesty no better or worse than R-recon → **NOT SUPPORTED**. TopK R-honesty also fails the GoT guard (+0.025).
  * R1a (RepE reconstruction fine-tuning beats base) not met for any architecture.
  * Direction-level metrics (pair-direction cosine +0.07 to +0.20) improve strongly; they measure the training objective and are not counted as evidence.
  * OWT reconstruction and splice behaviour unchanged (ΔEV ≥ −0.002, ΔCE ≤ 0.005 nats). Pair accuracy is at ceiling (1.0) everywhere.
* **Evidence label:** VALIDATION finding. **Files:** `results/repe/r1_honesty_preservation_layer6/` (README §11), figure `results/repe/figures/r1_honesty_preservation_layer6.png`.
* **Limitations:** 3 seeds; 89 statements (per-SAE CI ≈ ±0.04); single λ; overfitting to 208 statements; RepE-prompt L0 uncontrolled for ReLU/JumpReLU.
* **Next steps:** R1 test pass only with approval; an L0-controlled R1 variant (roadmap §2); R2 after the user reviews this report.

## 20. R2: SAE-in-the-loop clamp of honesty features, layer 6 (2026-11-08). Status: VALIDATION STAGE COMPLETED; NOT SUPPORTED (descriptive); test stage DEFERRED [VERIFIED]

* **Question:** does clamping the honesty-selected SAE features of an untruthful-persona prompt towards the honest-train mean (encode → clamp → decode + error, final token, `blocks.6.hook_resid_pre`) move GPT-2's next-token distribution towards the honest prompt's, more than random-feature, matched-norm, unrelated-feature and raw-direction controls?
* **Protocol:**
  * pre-registration drafted in `d17bb00`, finalised in `b232ce4` (script SHA `0a0f4fc7…`, §12 addendum);
  * user decision 2026-11-08: validation now, test later. `--test` refuses; `test_rows_read: false`.
* **Execution:**
  * 9 base SAEs with all controls;
  * 18 R1 SAEs (EXPLORATORY), targets plus C2 only;
  * k ∈ {1, 4, 16}, α ∈ {−1 … 2};
  * 75 min, peak 5.6 GB.
* **Results (validation, default wording, base SAEs, k = 1 selected for every architecture):**
  * Positive control (activation patch): E = 0.188. Baseline mean KL(p_H‖p_U) is 0.020 nats.
  * E(α = 1): ReLU −0.110 ± 0.006, TopK −0.103 ± 0.039, JumpReLU −0.115 ± 0.031. Every k is worse (down to −0.85).
    * This is below the random-feature C2 mean (−0.01 to −0.05) and q95 (+0.003 to +0.004), and below matched-norm C3 (q95 ≈ +0.005 to +0.009).
    * Empirical p 0.68–1.0; BH q = 1.
  * Criteria: 1 effect FAIL everywhere; 2 sign reversal passes but is uninformative (both α signs harm); 3 specificity FAIL; 4 damage FAIL (target OWT ΔCE 0.017–0.18 vs C2 q95 0.001–0.04); 5 cross-wording FAIL.
  * **Verdict: NOT SUPPORTED for all architectures and all k.**
  * Representation-level: the edit is well formed (re-encoding attainment 0.77–0.99).
  * Probe score: the edit closes 15–44 % of the U→H logistic-probe gap on `val`, but reverses it on `cw`.
  * Behaviour: absent or adverse.
  * The no-error reconstruction splice gives E ≈ −3 to −4: the behaviourally relevant difference lives largely in the SAE error term.
  * R1 SAEs: R-honesty − R-recon differences in E ≈ 0.
* **Conclusion:** no causal SAE-feature faithfulness claim. H7 remains null at layer 6.
* **Evidence label:** VALIDATION finding (descriptive; selected-k numbers optimistic).
* **Files:** `results/repe/r2_sae_in_the_loop_steering_layer6/` (README §13), figure `results/repe/figures/r2_sae_in_the_loop_validation.png`.
* **Limitations:**
  * a small behavioural signal at this site (positive control 19 %);
  * a population-mean clamp only;
  * an unexplained `cw` probe-sign reversal;
  * 3 seeds;
  * 89 statements.
* **Next steps:**
  * R2 test stage (k = 1 frozen) needs user approval;
  * an optional new pre-registration for a per-statement clamp or a later layer.

## 21. In progress: M0 within-layer reanalysis and M1 RepE-LM gates (2026-11-08). Status: PRE-REGISTERED

* **Why (D23):** the 2026-11-08 audit showed four problems:
  * the RepE persona prompts are length-confounded;
  * earlier "reconstruction tracks preservation" correlations were pooled across layers;
  * frozen-probe failures were read as information loss;
  * the layer-6 causal site is weak.

  The user approved the M0–M5 programme and asked for M0 and M1 to run now. M2–M5 wait for approval.
* **M0:** existing validation CSVs only. Within-layer regressions (layer FE + architecture FE + log L0; Freedman–Lane permutation), an L0-adjusted architecture table, and pooled-vs-within corrections. `results/shared/m0_within_layer_reanalysis/`.
* **M1:** 8 one-token-swap templates (T1–T4 fit, T5–T6 held out, N1–N2 neutral) on train (417) and validation (89) statements. Original GPT-2 activations at 12 layers; gates K1–K7. `results/repe/repe_lm_gates/`. Test statements are not loaded.
* **Pre-registrations with hashes:** `preregistration.json` in each folder. The smoke tests are disclosed in the M1 pre-registration.
* **Next:** run M0; run M1 (scripts 20 → 21 → 22); document; report to the user; stop.

## 22. M0: within-layer reanalysis (2026-11-08). Status: COMPLETED, VALIDATION [VERIFIED]

* **Run:** `scripts/shared/m0_within_layer_reanalysis.py`, on the pre-registered hash (`ebdc1be`). Validation CSVs only; no model or test row.
* **Results:**
  * **Behaviour:** within a layer, lower OWT reconstruction error goes with less splice damage (β = 0.165 per log NMSE, Holm p = 0.0008; per-layer ρ 0.80–0.97 from layer 3).
  * **Truth readout:** within a layer, reconstruction error does **not** usefully predict the GoT frozen-logistic gap. The formal pass for OWT NMSE (Holm p = 0.044) depends on architecture fixed effects that absorb 99.8 % of the within-layer NMSE variance, and it vanishes without them (p = 0.89). GoT task error: no association.
  * **Architectures:** the rankings survive an L0 adjustment, apart from two near-ties.
  * **Effective width and task-prompt L0** differ strongly between architectures.
* **Corrected claims:**
  * the pooled "reconstruction tracks honesty preservation" is a pooling artefact (0.89 → 0.05);
  * the splice SAE/random ratio is 0.76–2.89, not 1.0–2.6×.
* **Evidence label:** VALIDATION reanalysis. **Files:** `results/shared/m0_within_layer_reanalysis/`.
* **Limitations:** 9 SAEs per layer; architecture and error nearly collinear; 1000-step elasticities.

## 23. M1: RepE-LM equal-length persona task (2026-11-08). Status: COMPLETED; all gates PASS (K5 narrowly) [VERIFIED]

* **Run:** scripts 20 → 21 → 22, on the pre-registered hashes (`ebdc1be`). Train and validation statements only; test not loaded.
* **Results:**
  * The length/position/template confound is eliminated by construction: K1 bit-identical, K2 all AUROC 0.5000.
  * A persona-meaning signal transfers across word pairs only modestly (K5 0.70 at layers 8 and 11; mass-mean 0.63).
  * Persona swaps change GPT-2's next-token distribution about 2× more than honest-synonym swaps (K7). The absolute effect is small (0.019 nats).
* **Caveats:**
  * generic one-token-change decodability is large (neutral 0.58–0.75);
  * template T1 and held-out T5/T6 have near-neutral behavioural effects;
  * ungrouped CV leaks heavily (0.92–0.99 vs 0.63–0.84).
* **Decision gates:** M2 and M3 allowed. **Awaiting user approval** before either starts.
* **Files:** `data/repe/repe_lm/`, `results/repe/repe_lm_gates/`. Activations are in `data/repe/activations/repe_lm/` (712 MB, gitignored).

## 24. In progress: M2 information faithfulness and M3 causal site sweep (2026-11-08). Status: PRE-REGISTERED

* **Approved by the user 2026-11-08,** after the M0/M1 report.
* **M2:** 108 SAEs + 36 PCA references + norm-matched noise, 12 layers, on GoT cities, RepE-LM persona and RepE-LM neutral. Probe ladder: frozen logistic/mass-mean; re-trained logistic, MLP and kNN; sparse-z; error term; learning curve. Four-level labels; H2a (vs noise), H2b (vs PCA). `results/shared/m2_four_level_faithfulness/`.
* **M3:** original GPT-2 activation patching, 65 cells. H→U and U→H; synonym, wrong-statement and matched-norm random controls; per template (T1 flagged). Holm; ≤ 3 sites; validation confirmation. `results/repe/causal_site_sweep/`.
* **New modules with tests:** `src/evaluation/{information_levels,patching}.py`. The smoke tests and changes made after them are disclosed in each pre-registration.
* **Next:** run both; document; report; stop before M4.

## 25. M2: four-level information faithfulness (2026-11-08). Status: COMPLETED, VALIDATION [VERIFIED]

* **Run:** on the pre-registered hashes (`b384622`); 432 cells; 39 min; all numerical checks pass; 0 test rows.
* **Results:**
  * The GoT frozen-logistic collapse (0.28) is mostly generic sensitivity to perturbations of that size: noise of the same norm gives 0.25 and PCA 0.22, while mass-mean loses 0.03.
  * Re-trained readers show a smaller but real reduction (GoT 0.07–0.12; persona 0.14–0.18).
  * No cell is Preserved or Readout shifted.
  * SAEs are never better than PCA at equal OWT error, and are worse in most cells.
  * The lost information is decodable from the error term.
  * On truth, ReLU loses least and JumpReLU most, the reverse of reconstruction.
  * Persona loses 0.07–0.13 more than neutral one-word changes at layers 3–8 only.
* **Evidence label:** VALIDATION. **Files:** `results/shared/m2_four_level_faithfulness/`.
* **Corrects:** "truth is lost / near chance" (the frozen-probe reading). Use the re-trained and noise-referenced numbers instead.

## 26. M3: causal site sweep, original GPT-2 (2026-11-08). Status: COMPLETED; sites found and confirmed; M4 justified by the rule [VERIFIED]

* **Run:** on the pre-registered hashes (`b384622`); 96 min GPU; sanity checks exact.
* **Results:**
  * The persona-instruction effect is carried by the persona-slot token early and by the final token late, with the handover at layers 7–8. The instruction words after the slot carry up to 28 %.
  * 50/65 cells pass. The selected sites are L1 slot (earliest and peak) and L11 final (latest), both confirmed on validation.
  * T1 shows no effect beyond its synonym control.
* **Caveats:**
  * the selected sites are near-trivial endpoints (E ≈ 0.96);
  * the mid-depth split sites (layers 6–8) are more informative but need an amendment;
  * the random control is weak.
* **Next:** M4, if the user approves (and decides on an amendment adding a mid-depth site).

## 27. In progress: M4 SAE causal-faithfulness decomposition (2026-11-08). Status: PRE-REGISTERED (with amendment A1)

* **Design:** L1 slot, L11 final, and an A1 mid site (layers 6–8, handover rule, confirmed without SAEs). 9 base SAEs per site.
* **Measured:** T_full/T_rec/T_err, interaction, Shapley shares, linearity, the opposite direction, synonym and neutral controls, vector controls, and feature patches with matched random feature sets. Primary set T2–T4; T1 reported as a control. `results/repe/sae_patch_decomposition/`.

## 28. M4: SAE causal-faithfulness decomposition (2026-11-08). Status: COMPLETED, VALIDATION [VERIFIED]

* **Run:** on the pre-registered hashes (`15131fe`); 28 min. A1 selected L8 final (confirmed without SAEs). All checks pass; T_full reproduces M3 to 7e-8.
* **Results (T2–T4):**
  * The SAE-reconstructed difference restores 93–97 % of the persona effect at the L1 persona slot, but only 62–68 % at the L8 and L11 final token.
  * There, ReLU's error carries 40–45 % on its own. For TopK and JumpReLU, about a third of the effect requires reconstruction and error together (interaction too large for an additive reading).
  * The persona > synonym distinction is preserved in the reconstruction. T1 behaves like a synonym.
  * Feature patches beat matched random sets but need 64 features for 47–74 % of T_rec; single features give 2–10 %.
* **Evidence label:** VALIDATION. **Files:** `results/repe/sae_patch_decomposition/`.
* **Next:** M5 (test) only with the user's written approval.

## 29. Instructor Review #8 response: README, paper skeleton, IMP-L pre-registration (2026-11-08/09). Status: COMPLETED (IMP-L results in §30)

* **Review #8** (project-board ticket, 2026-11-08) was read in full, together with Reviews #2–#7. Its blockers for the
  11-10 session are a multi-layer improvement table and a paper skeleton. It credits only work merged to `main`,
  which stops at PR #23 (`33dcb9c`).
* **README** rewritten (`fb00cee`): current findings, historical results separated, corrections table, test-split
  exposure, progress table.
* **Paper skeleton** in `research_paper/Latex/`:
  * every section is present;
  * M0–M4 tables and PDF/SVG figures are generated by `scripts/shared/build_paper_artifacts.py` from the committed
    result files;
  * it compiles with latexmk (15 pages; no undefined references);
  * pending sections are marked in the PDF.
* **IMP-L** (`results/geometry_of_truth/multilayer_finetune/`): the layer-6 OWT-only / Method A / output-KL
  protocols at layers 3, 8, 9 (81 runs). The scripts reuse the layer-6 training code unchanged.
  * The evaluation was smoke-tested on the layer-8 base SAEs. It reproduced the committed M2 layer-8 re-trained gaps
    exactly (GoT 0.034/0.061/0.085; persona 0.212/0.156/0.162).
  * `--prepare` runs the bit-identical reproduction checks and writes the pre-registration.
  * Training waits for the user's sign-off on the configuration.
* **Evidence label:** none yet; no new result.
* **Next:** user sign-off → `--train` → evaluation → improvement table → E3 at layer 8.

## 30. IMP-L results: fine-tuning at layers 3, 8 and 9 (2026-11-09). Status: COMPLETED. Evidence: VALIDATION

* **Run:**
  * User approval on 2026-11-09. 81 fine-tunes and a 108-SAE validation evaluation.
  * Scripts and criteria unchanged from the pre-registration (hashes verified). Not one test row was used.
  * Base checkpoints and GPT-2 were unchanged after training.
* **Method A** (criterion T_method_a: consistent vs both base and OWT-only):
  * Met in **4/9 cells**: L3 TopK, L8 ReLU, L9 ReLU, L9 JumpReLU.
  * The frozen B-MLP truth gap fell on every seed in 9/9 cells (4–47 %).
  * GoT reconstruction error fell 61–85 %.
  * OWT-only moved the gap by at most 0.021.
  * All guards passed.
* **Re-trained readers** (reported, not a criterion): no consistent change vs base in any cell (all CIs include 0).
  The frozen-probe gain is therefore read as restoring the original probe's readout, not as recovering information.
  This qualifies the Method A result.
* **Output-KL:**
  * OWT KL (−8 to −18 %) and GoT statement KL fell vs Method A, consistently in all 9 cells.
  * ReLU/JumpReLU L0 rose 5.3–8.1 % at layers 8 and 9, so those 4/9 cells are **NOT CREDITABLE**. Behaviour is
    creditable in 5/9.
  * The truth gap never improved vs Method A. **REGRESSION** at L3 ReLU (+0.012) and L3 TopK (+0.027); both CIs just
    reach 0.
* **Overall:** the layer-6 pattern replicates at three further layers.
  * Task reconstruction and behavioural faithfulness can be improved.
  * Truth information, as seen by a re-fitted reader, is not improved.
  * Behavioural gains for ReLU/JumpReLU at depth cost sparsity beyond the guard.
* **Paper:**
  * §10.1 is written, with the summary table and figure; the per-layer tables are in the appendix.
  * The abstract and discussion are updated.
  * The paper compiles (20 pages).
  * `build_paper_artifacts.py` now writes deterministic PDF/SVG (fixed metadata and SVG hash salt), so a reviewer can
    regenerate the figures and check for an empty diff.
* **Next:**
  * E3 (causal preservation of the layer-8 SAEs) needs its own pre-registration.
  * L0-matched output-KL for ReLU/JumpReLU, and more seeds for the borderline Method A cells (not approved).
  * Push/PR and the Review #8 reply are awaiting user approval.

## 31. Paper refinement, E3/E2/E1 draft pre-registrations, review package (2026-11-09). Status: IN PROGRESS (awaiting approval to run E3)

* **Paper.** Revised to keep the user's reading of IMP-L (D27):
  * a new "Evidence status" paragraph in §5: everything except the OWT reconstruction table is VALIDATION, and
    nothing is a confirmed held-out result;
  * every results section opens with its label;
  * "confirmed" is replaced by "replicated on validation" for the M3 sites;
  * the abstract states the three levels separately: task reconstruction, frozen-probe vs re-trained-probe truth,
    and behaviour (5/9 creditable);
  * the discussion gains "What fine-tuning changes, and what it does not" and "Sparsity is part of the price";
  * seven new limitations: no held-out confirmation, one fine-tuning budget, three seeds and multiplicity, the
    frozen-probe criterion, output-KL L0 drift, the weak layer-3 logistic signal, and untested causal preservation;
  * the conclusion is updated. The paper compiles (21 pages).
* **Draft pre-registrations** (not run; no script written yet; hashes are recorded at `--prepare`, after approval):
  * **E3** `results/repe/e3_finetuned_causal_preservation/`: 36 existing layer-8 SAEs, M4 decomposition at L8 final,
    H1 Method A, H2 output-KL, H3 specificity. Output-KL ReLU/JumpReLU are not creditable a priori. About 40 min GPU.
  * **E2** `results/shared/e2_public_sae_reference/`: public res-jb SAEs at L8 and L10 (cached). H1–H4 test whether our
    four headline conclusions hold. About 30 min GPU.
  * **E1** `results/shared/e1_task_l0_matched_topk/`: 36 new TopK SAEs at task-matched k (layers 1, 8, 11; GoT and
    RepE). About 1.2 h GPU.
* **Review package:** `docs/2026-11-09_review_package_sae_research.md`.
  * The M0 re-run and the paper-artifact regeneration were verified byte-identical on a clean `git archive` export of
    `HEAD`, and the tests pass there.
  * Teammate branch `simba-method-a-multilayer` overlaps IMP-L (same script numbers 22/23; layer 5 instead of 3);
    this is flagged for a team decision.
* **Evidence label:** no new result.
* **Next:** user approval of E3 → implement script 25 → `--prepare` (hashes, reproduction of M4 L8 rows) → run.


## 32. E3 causal preservation after fine-tuning, and RLM-RP RepE-LM representation preservation (2026-11-10). Status: COMPLETED. Evidence: VALIDATION

* **Approval:** the user approved both on 2026-11-10 ("Complete causal faithfulness evaluation (E3) ... Complete RepE
  representation preservation ... generalization and robustness"). M5 and the test split remain excluded.
* **E3:** `results/repe/e3_finetuned_causal_preservation/`.
  * 36 layer-8 SAEs (9 base + 27 IMP-L fine-tunes), M4 protocol at the L8 final token, M4 functions imported unchanged.
  * The 2026-11-09 draft criteria are unchanged. §1.8 adds implementation details (per-SAE reseeding, M4 reproduction
    on deterministic columns, BH over 36).
  * `preregistration.json` hashes the scripts, data and 36 checkpoints. The smoke test showed only mechanics.
* **RLM-RP:** `results/repe/repe_lm_representation_preservation/`.
  * 216 SAEs: 108 base at layers 0–11, 81 IMP-L, and 27 layer-6 fine-tunes.
  * Probes: frozen logistic and mass-mean, re-trained logistic, a noise reference.
  * Reconstruction: MSE, EV, cosine, L0, dead fraction. Geometry: H − U direction cosine and norm retention.
  * Robustness: held-out wording T5/T6, synonym, neutral, near-duplicate sensitivity, and an exploratory L0-truncation
    control.
  * Pre-registered classes on frozen and re-trained persona gaps, kept separate. M2 and IMP-L reproduction checks.
* **Paper (Phase 5):** new generated tables `t_impl_l0` (L0 on OWT / GoT / RepE-LM per IMP-L cell) and
  `t_impl_readers` (frozen MLP vs frozen logistic vs re-trained readers).
* **Results E3** (1830 s; all sanity checks pass; M4 reproduced exactly, 0.0; GPT-2 unchanged; 0 test rows):
  * causal preservation T_rec/T_full at L8 final:
    * base: ReLU 0.621, TopK 0.625, JumpReLU 0.665;
    * Method A: 0.622 / 0.615 / 0.669;
    * output-KL: 0.640 / 0.619 / 0.687;
  * **E3-H1 (Method A): NO IMPROVEMENT in 3/3.**
  * **E3-H2 (output-KL vs Method A):**
    * TopK: NO IMPROVEMENT (+0.005, CI −0.078 to +0.088);
    * ReLU: NOT CREDITABLE. The criterion itself was met (+0.018 [+0.011, +0.025]) but OWT L0 is +5.8 %.
    * JumpReLU: NOT CREDITABLE, and its criterion was not met (+0.018, not consistent).
  * No regression. The persona-specific part (T_rec − S_rec) moves by at most 0.0006 nats. All 36 SAEs pass the M4
    control tests.
* **Results RLM-RP** (493 s; 216/216 SAEs; checks pass; M2 reproduced exactly; IMP-L to 1.6e-4):
  * Persona loss is real for base SAEs at layers 1–11:
    * the frozen gap (0.10–0.24) is 1.9–3.4× the same-norm noise reference;
    * the re-trained gap is 0.08–0.22, with its CI excluding 0 for all 99 SAEs.
  * The per-pair H−U cosine is only 0.35–0.58. ReLU shrinks the difference; TopK and JumpReLU inflate it.
  * Held-out wording transfers weakly even on original activations (0.54–0.69). Synonyms behave like the persona
    word. Near-duplicate sensitivity is ≤ 0.02.
  * Truncation control: JumpReLU's apparent advantage is partly extra L0.
  * Method A: NO IMPROVEMENT in 12/12 cells on both readers.
  * Output-KL: PRELIMINARY IMPROVEMENT in 3/12 cells per reader (at most 0.007 AUROC), with RepE-LM L0 +4–9 %.
    4 cells are not creditable; there are no regressions.
* **Paper:** new §7.1 (RLM-RP) and §10.2 (E3); RepE research questions in §4. §10 now has `t_impl_readers` and
  `t_impl_l0`. The abstract, discussion, limitations, conclusion, statistics and appendix are updated. The abstract's
  ambiguous "5 of 9 creditable" now reads "5 of 9 cells are creditable; 4 fail the L0 guard". Compiles, 29 pages,
  0 overfull boxes, 0 undefined references.
* **Unfinished:**
  * M5 (needs separate approval);
  * E1 and E2 (pre-registered drafts, not run);
  * an L0-matched output-KL variant;
  * the minimum-detectable-effect statement.

## 33. Independent research review and improvement strategy (2026-11-10). Status: COMPLETED (recommendation only; no experiment run)

* **Document:** `docs/2026-11-10_independent_research_review.md`.
  * Repository audit; literature review of about 25 papers, links checked.
  * Planning-only throughput measurements: GPT-2 to L8 about 63k tokens/s; Gemma-2-2B full forward bf16 about 7.7k
    tokens/s, 7.3 GB. TransformerLens loading of Gemma exceeds host RAM, so use HF bf16.
* **Main judgement:** the dominant weakness is SAE quality.
  * Ours: 101,600 unique tokens, width 3,072.
  * res-jb: 300M tokens, 24,576 latents. Gemma Scope: billions of tokens.
  * Every faithfulness conclusion is therefore exposed to the "under-trained SAEs" objection.
* **Recommendation:** apply the existing ladder to strong SAEs (E2; data-scaled retraining; Gemma Scope), test KL
  fine-tuning only at fixed L0, add a causal skyline and a random baseline, then run M5. Roadmap updated (R1–R7,
  proposed). Evidence label: none (no new result).


## 34. GPT-2 improvement phase: R2, R4, R5, E2 and M5 (2026-11-10). Status: COMPLETED. Evidence: VALIDATION; M5 CONFIRMED TEST

* **Approval:** the user gave full execution approval for the GPT-2 improvement phase on 2026-11-10 (R1, R2, other
  justified GPT-2 experiments, final M5 after the protocol is frozen). Gemma is deferred.
* **R2** (`results/openwebtext/r2_scaled_saes/`, pre-registered with hashes):
  * 39 SAEs on 100M fresh OWT tokens each (shard 78, excluding the 98 documents matching our sample);
  * layers 1, 8, 11; TopK, BatchTopK and ReLU at width 12,288 × 3 seeds; TopK-3072 × 3 (width ablation); a
    frozen-random-decoder TopK × 1;
  * script `scripts/openwebtext/17_streaming_sae_training.py`, module `src/sae/modern_sae.py`;
  * a modern JumpReLU (STE) was attempted in the pilot and abandoned (L0 would not settle); disclosed in the README.
* **Evaluation harness:** `scripts/shared/sae_ladder_evaluation.py`, the same ladder for old, R2 and res-jb SAEs,
  built from imported M2/RLM-RP/M4/splicing code. Comparisons: `scripts/shared/r2_comparisons.py`, implementing the
  pre-registered rule. E2: `scripts/shared/e2_public_sae_reference.py`.
* **Unfinished at the time of this entry (superseded by the results below):** training, ladders, comparisons, M5, paper.
* **Status update (2026-11-10, 15:30 UTC):**
  * **Done:**
    * R2 training: 39 SAEs and 156 checkpoints; GPT-2 parameters unchanged; every L0 within 1 % of 128.
    * R4 fine-tune: 12 checkpoints, GPT-2 unchanged.
    * R5 causal skyline (results in its README).
    * E2 adapter check: passes at layers 1, 8, 10 and 11.
    * M5: smoke test on validation reproduces the ladder; protocol frozen with hashes; no test row read.
* **OOM incident (2026-11-10, about 15:00 UTC):**
  * **Cause:** four ladder or evaluation processes, each about 4.4 GB of host RAM, ran at once on the 15 GB host. They
    were the chain's R2 ladders, the M5 smoke test and two R4 ladders.
  * **Killed by the kernel OOM killer:**
    * R2 ladder L11, at about 8 of 14 SAEs; partial file kept in the scratchpad, not used;
    * R2 ladder L1;
    * R4 ladders L8 and L11.
  * **Stopped deliberately:** one freshly started rerun, which had not finished any SAE.
  * **Not affected:** checkpoints, finished results, Git.
  * **Recovery:**
    * at the user's instruction, every memory-heavy job now runs one at a time;
    * a serial queue waits until no heavy job is running and at least 8 GB is available;
    * steps with an existing output are skipped;
    * RAM, VRAM and disk are logged every minute.
  * The killed runs are rerun from scratch. Their results do not depend on run order, because the harness is
    deterministic; M5's smoke test confirms this.
* **Results (details in each folder's README):**
  * **R2** (`results/openwebtext/r2_scaled_saes/`), VALIDATION:
    * All C1–C3, A1 and A2 comparisons improve reconstruction and behaviour at L1, L8 and L11:
      * NMSE −31 to −74 %;
      * ΔCE −37 to −82 %;
      * TopK L8 ΔCE 0.126 → 0.060, L11 0.213 → 0.100.
    * Causal T_rec/T_full: TopK 0.62 → 0.73 (L8) and 0.62 → 0.69 (L11); BatchTopK reaches 0.75 at L8.
    * Pair-direction cosine improves everywhere.
    * Persona re-fit gap −27 to −31 % at L8/L11 (TopK).
    * SAE − PCA gap: never improves; regresses for ReLU at L1 and L8.
    * GoT: no clear change. The random decoder beats the trained SAE on the GoT re-fit gap at L8.
    * Cell counts are in the README. No SAE was excluded for L0.
  * **R4** (`results/openwebtext/r4_kl_finetune/`), VALIDATION:
    * KL vs MSE control: ΔCE −18 % at L8 and L11, PRELIMINARY IMPROVEMENT; EV guard passes (−0.002).
    * Information: NO IMPROVEMENT.
    * T_rec/T_full at L11: −0.024 on all seeds, REGRESSION.
    * Use it as a behaviour-only tool.
  * **R5** (`results/repe/r5_causal_subspace_skyline/`), VALIDATION, descriptive:
    * At k = 4, a learned subspace carries 1.02 / 0.91 / 0.62 (L1 / L8 / L11). PCA carries 1.00 / 0.52 / 0.59, SAE
      top-4 features 0.08–0.16, random subspaces 0.01.
    * At L8, rank 4 carries more than the whole SAE reconstruction.
  * **E2** (`results/shared/e2_public_sae_reference/`), VALIDATION:
    * On res-jb, H1 holds for GoT and fails for persona; H2 holds except GoT at L8; H3 holds everywhere; H4 holds at
      L8 and L11.
    * Our conclusions transfer to an external 300M-token SAE.
  * **M5** (`results/shared/m5_confirmatory_test/`), CONFIRMED TEST:
    * One run: 81 SAEs, 552 test rows.
    * C1, C2, C3, A1: 17 improved / 7 no change / 0 regressed each.
    * A2: 2 GoT regressions at L8.
    * K1/K4: ΔCE −16 to −17 % confirmed; NMSE +2 % (regression, expected cost).
    * GoT information: no improvement anywhere.
* **Paper:** new section `10b_improvement.tex` (Problem → Solution → Pipeline → Results → Inference), tables
  `t_r2_c1/c2/c3/counts`, `t_r4`, `t_r5`, `t_e2`, `t_m5`, `t_m5_counts`. Abstract and conclusion updated. No new
  figures, per the user. 37 pp, clean build.
* **Operational issues (both resolved):**
  * the OOM incident above;
  * two self-matching `pgrep -f` wait loops, which held up the serial queue for 10 min.

## 35. Final defense verification and audit (2026-11-10). Status: COMPLETED (verification only; no experiment run)

* **Task:** an independent verification of every reported result before the final professor presentation.
* **Constraints:** no rerun, no checkpoint loaded, no result or historical README modified.
* **Deliverables:** `docs/final_defense/` contains:
  * `FINAL_RESEARCH_REPORT.md`;
  * `DEFENSE_CHEATSHEET.md` (renamed 2026-11-12 from `PROFESSOR_DEFENSE_CHEATSHEET.md`);
  * `RESULTS_VERIFICATION_TABLE.md`;
  * `ANTICIPATED_REVIEW_QUESTIONS.md` (renamed 2026-11-12 from `PROFESSOR_QUESTIONS_AND_ANSWERS.md`);
  * `EXPERIMENT_AUDIT.md`.

  Also new: the read-only re-check script `scripts/shared/m5_independent_verification.py`.
* **M5 verified COMPLETE:**
  * 11/11 code and config hashes in `preregistration.json` match;
  * 81/81 expected rows (23/29/29), 0 duplicate keys, 0 duplicate checkpoint hashes;
  * 136/136 decision classes reproduced by an independent implementation of the R2 and IMP-L rules;
  * clean log exit; `test_rows_used` 552.
* **Findings recorded (no artifact edited):**
  1. The M5 README states an OWT test L0 range of 127.6–133.2. The true maximum is 133.39 (old ReLU, L11, seed
     2042). It is inside ±10 %, so no exclusion changes.
  2. The pre-run M5 README hash cannot be reproduced, because results were appended to the same file. The design text
     is therefore not byte-verifiable; the code hashes are.
  3. The pre-registration and the results were committed together (`af52b0b`). Their order rests on file times and the
     in-script hash guard.
  4. The M5 OWT and GoT test rows had been scored earlier for the old reference SAEs (README §10). M5 should be
     labelled "previously exposed (reference SAEs only)". No R2/R4 choice used test numbers (D31).
  5. A1 confounds token count with recipe. "Longer training alone helps" is NOT VERIFIED. The R2 10M/25M/50M
     checkpoints remain unevaluated.
  6. New descriptive observation from the M5 rows: the random frozen-decoder TopK has a GoT SAE − PCA re-fit gap of
     −0.003 (L8) and −0.013 (L11), against +0.008 to +0.081 for every trained SAE. That metric does not separate
     trained from random SAEs (n = 1).
* **Tests:** 35 passed.
* **Evidence label:** no new result. All verified numbers keep their original labels.

## 36. GPT-2 audit and evidence-based improvement plan (2026-11-12). Status: COMPLETED (analysis only; no experiment run)

* **Deliverable:** `docs/2026-11-12_gpt2_audit_and_improvement_plan.md`, a read-only audit of code, CSVs and
  checkpoints at `446beea`, plus a literature check and pre-registration drafts. No script, result or checkpoint was
  changed; nothing was committed.
* **Re-verified:** M5 test values (`m5_rows.csv`), R2 ladder values, R5 summary.
* **New descriptive observation (VALIDATION, from existing ladder CSVs):**
  * RepE-LM persona-contrast norm retention ‖Δ̂‖/‖Δ‖ is 0.62–0.71 for ReLU-family SAEs (shrinkage), against 1.09–1.46
    for TopK, BatchTopK and the old JumpReLU (inflation), with cos(Δ, Δ̂) of only 0.37–0.52 for the latter.
  * The two families distort the honesty contrast in opposite directions.
  * The support-flip explanation for TopK is a HYPOTHESIS.
* **Gaps found:**
  * the R2 10M/25M/50M checkpoints have never been evaluated (`r2_scaled_saes/scaling/` is empty);
  * the R5 feature arm used old SAEs only;
  * there is no RepE-LM test split;
  * there is no modern JumpReLU.
* **Proposed** (roadmap 2026-11-12 additions; none approved): H1, T1, X0, X1 first; then H2, I1, X2, X3; then M6.
* **Evidence label:** no new result; the one new observation is VALIDATION, descriptive.

## 37. Documentation cleanup, consolidation and PR preparation (2026-11-12). Status: COMPLETED (no experiment run)

* **Checkpoint first:** `b927f27` preserved the current work. The push also published the 22 earlier local commits
  (R1-honesty through M5 and the final-defense docs), fast-forward `875bc7f..b927f27`, after the user's approval.
* **New curated documents:**
  * `docs/research_overview.md`;
  * `docs/methodology.md` (includes the table of 12 methodological corrections);
  * `docs/experiment_history.md` (39 experiments with verdicts);
  * `docs/results_summary.md`;
  * `docs/limitations_and_future_work.md`.
* **Renames** (bytes unchanged; log in `docs/archive/README.md`, rows in `docs/migration_manifest.csv`): the
  2026-11-04 review report moved to `docs/archive/`; two `final_defense` files given neutral names.
* **Corrections found while writing:**
  * The 2026-11-12 audit had called the contrast-distortion finding new; RLM-RP (§32) reported it first. An erratum
    was added.
  * M5 classes the R4 NMSE change (+2 %) as REGRESSION; the docs now say so explicitly.
  * The RepE-LM task L0 per family was re-read from the ladder CSVs (L8: old JumpReLU 167, old ReLU 161, R2 BatchTopK
    141, R2 ReLU 145).
* **Validation:**
  * 35 tests pass; all tracked Python compiles;
  * 0 broken Markdown links in 27 files;
  * `m5_independent_verification.py`: 11/11 hashes OK, 136 rows, 0 mismatches;
  * no secrets or absolute home paths;
  * no file under `scripts/`, `results/`, `src/` or `data/` changed.
* **Process note:** `git mv` pre-staged the renames, so they landed in `e2becfa` rather than in `768831a`. Recorded in
  the changelog; not amended.
* **Evidence label:** no new result.

## 38. In progress: remediation of the 2026-11-12 independent audit (2026-11-13). Status: IN PROGRESS

* **Register:** `docs/remediation/issue_register.md` (D1–D15, S1–S12, U1–U4, Review #8 items). It is written as the
  work proceeds.
* **PCA-CM (S1), PRE-REGISTERED and RUNNING:** capacity-matched PCA references (k = 128, k = GoT task L0, and the
  existing error-matched k), VALIDATION only.
  * Protocol: `results/shared/pca_capacity_matched/{README.md,preregistration.json}`.
  * The protocol was hashed before the full run and committed while the run was in progress, before any full-run
    output existed (D31).
  * A smoke run on validation (16 sequences, PCA from 5 chunks) checked the pipeline. Its numbers are not used.
* **Evidence label:** pending.

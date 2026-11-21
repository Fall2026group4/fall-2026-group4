# Final-week technical audit and execution plan (2026-11-21). Status: AUDIT COMPLETED; PLAN AWAITING APPROVAL

Evidence label: analysis only. No experiment was run. The only checks were read-only Git inspection (`git fetch`,
`git show`, `git merge-tree`), resource checks and one non-destructive tokenizer check. Nothing was trained.
No TEST split was touched.

## 0. What was inspected (do not rely on the docs alone)

* **Branches.** `sae-research` (local, equal to `origin/sae-research` at `20b7c98`) and `origin/main` (`bf25b40`; 39
  commits not in `sae-research`). Teammate branches:
  * `dhruv-gemma-r3` (PR #30, open);
  * `dhruv-causal` (no PR, unmerged);
  * `simba-gemma-m6-results` (PR #28, merged);
  * `simba-method-a-multilayer`;
  * `simba-paper-manuscript` (PR #31);
  * `cleanup-main`.

  Open PRs: #35 (ours, draft), #31, #30.
* **Dry-run merges** of `sae-research` with `main`, `dhruv-gemma-r3`, `dhruv-causal` and `simba-method-a-multilayer`
  (`git merge-tree`):
  * code merges cleanly;
  * conflicts only in the shared journals `docs/{RESEARCH_PROGRESS,RESEARCH_DECISIONS,EXPERIMENT_RESULTS_SUMMARY,EXPERIMENT_CHANGELOG}.md` and `docs/experiment_manifest.csv`.
* **Resources:**

  | Resource | State |
  |---|---|
  | GPU | A10G 23 GB, idle, no running jobs |
  | RAM | 15 GB, no swap |
  | CPU | 4 cores |
  | Disk | **34 GB free (89 % used)** |
  | Gemma-2-2B | cached at the M6-pinned revision `c5ebcd40` (9.8 GB) |
  | Gemma Scope | **none cached** |
  | Software | sae_lens 6.54, transformers 5.21, torch 2.14 + CUDA |
  | GPT-2 checkpoints | all 780 registry entries exist on disk (34 GB of `.pt`) |
  | Activation caches | `data/activations/gpt2_small` 2.3 GB |
* **Tokenizer check (non-destructive).** All RepE-LM pairs remain exact one-token swaps under the Gemma tokenizer
  (16 template × variant cells, 506/506 each; 0 failures). This is the K1-construction analogue, so the honesty task
  transfers to Gemma without rebuilding.

## 1. Technical gap analysis

Status codes:
* **V** = complete and scientifically valid;
* **A** = complete but needs methodological alignment;
* **P** = partially complete;
* **M** = missing;
* **I** = completed but needs improvement;
* **U** = invalid, unreliable or insufficiently supported.

GPU times are A10G estimates for the proposed solution.

| # | Experiment | GPT-2 status | Gemma status | Comparable? | Missing work | Proposed solution | Pri | GPU h |
|---|---|---|---|---|---|---|---|---|
| 1 | Original-model baselines (CE, truth/honesty probe AUROC per layer, behaviour gates K3–K7, cities preference) | V | P: L11 steering baseline only; Pile CE in M6 | No | Gemma probe curves (all layers), RepE-LM K2–K7, cities preference | XT-G0 | P0 | 0.3 |
| 2 | Reconstruction on OWT | V (108 old, 27 R2 + controls; M5 TEST) | M | No | Gemma on OWT | XB (OWT-XM arm) | P0 | in XB |
| 3 | Reconstruction on the Pile | M | A: M6 (3 layers); R3 ladder (36 SAEs, unmerged) | No | GPT-2 on the M6 sample; Gemma at the other 23 layers | XB (Pile arm) | P0 | in XB |
| 4 | Sparsity reporting (L0 per dataset, normalised) | V per dataset; not normalised | P: Pile L0 only | No | L0 on GoT/RepE for Gemma; L0/d_model, d_sae/d_model and L0/d_sae for both | XB + XT-G | P0 | 0 |
| 5 | Behavioural splicing (ΔCE, KL, top-1, loss recovered) | A: 100 × 128-token OWT VAL | A: 512 × 1024-token Pile VAL, document bootstrap | **No** (data, context, depth, uncertainty) | One matched protocol | **XB** | P0 | 5–6 |
| 6 | Norm-matched random-error control | V: same isotropic definition as M6 | V | Partly: definition matches, data does not | Covered by XB | XB | P0 | — |
| 7 | GoT truth: frozen-probe gap | I: dominated by readout sensitivity (S5: isotropic noise gives 0.23–0.30 vs SAE 0.26–0.32; frozen mass-mean gap only 0.02–0.03) | M | No | Gemma; demote frozen logistic to a "readout-sensitivity" metric | XT-G | P0 | 1 |
| 8 | GoT truth: refit / MLP / PCA-capacity-matched / noise controls (M2, PCA-CM) | V (VALIDATION; MDE 0.05–0.09) | M | No | Gemma | XT-G | P0 | in XT-G |
| 9 | Rotation vs information loss | P: S5 and dhruv-causal decomposition (VALIDATION; rebuilt SAEs); no alignment test | M | No | Label-free alignment diagnostic on both models | DX | P1 | 0 (CPU) |
| 10 | RepE-LM honesty representation (RLM-RP, pair-direction cosine, H1-TEST) | V (HT2 CONFIRMED TEST; HT1/HT3 not) | M (K1 verified today) | No | Gemma gates K2–K7, RLM-RP, held-out wording T5/T6 | XT-G | P0 | in XT-G |
| 11 | Direction fidelity | V | M | No | Gemma | XT-G | P0 | 0 |
| 12 | Causal site sweep + M4 decomposition + R5 skyline | V (VALIDATION) | M | No | Gemma site sweep on RepE-LM, then M4 with Gemma Scope, then rank-k skyline | XC-1 | P1 | 2 |
| 13 | GoT causal test (cities) | A: done on `dhruv-causal` (600 controls, BH; null for mass-mean/probe directions; rebuilt SAEs, Kaggle T4) | M | No | Port the same protocol to Gemma at depth-matched layers | XC-2 | P1 | 3 |
| 14 | Steering | U for comparison: GPT-2 RepE L6 (200 controls) vs Gemma GoT L11 (20 controls, no SAE) | P: descriptive | **No** | One protocol, both models, ≥ 200 controls, reconstruction-direction arm | XC-3 | P1 | 2–3 |
| 15 | SAE-in-the-loop intervention | P: R2 clamp (NOT SUPPORTED); dhruv-causal on length-confounded prompts; SIL pre-registered, **never run** | M | No | Run SIL (feasibility gate) on GPT-2, then Gemma if it passes | XC-4 | P1 | 2 |
| 16 | Truth-preservation improvement | P: TSP-R SUPPORTED on cities L8 TopK only (−0.021 re-fit gap, 8 seeds); no transfer to companies / common_claim / counterfact; no other layer/arch | M | No | Generalisation + multi-family objective; Gemma transfer | TI-1…TI-4 | P1 | 8–12 |
| 17 | Data-scaled in-house SAEs | V (R2, M5 TEST) at layers 1/8/11 | M (public only) | Only via depth matching (Gemma 1/16/23 ≈ GPT-2 1/8/11) | — | XB triplet T-A | P0 | 0 |
| 18 | Public-vs-public | P: res-jb (ReLU, 32×), OpenAI v5-32k TopK | V: Gemma Scope (JumpReLU) | Only with sparsity strata; **no public GPT-2 JumpReLU exists** (sae_lens registry checked) | Strata, not architecture claims | XB | P0 | in XB |
| 19 | JumpReLU-to-JumpReLU | U: in-house JumpReLU is the weakest-validated arm; R2 JumpReLU abandoned | V (public) | **No** | A valid GPT-2 JumpReLU (corrected pilot) | XJ | P2 | 3 + risk |
| 20 | In-house vs in-house | V | M | No | Gemma R2-recipe TopK at one depth-matched layer | XG-IH | P2 | 4–6 |
| 21 | Width/L0 ladder | P: G1/G5 partial; PCA-CM | A: R3 Tier A done (VALIDATION); Tier B cancelled | Descriptive only | Optional 131k at L12 (262k/524k need > 15 GB RAM: **infeasible here**) | — | P2 | 1 |
| 22 | Statistical uncertainty | A: 3-seed sd (seeds share data) + bootstrap in places | A: document bootstrap only | No | Hierarchical bootstrap (documents × seeds) in every cross-model table | XB/XT-G | P0 | 0 |
| 23 | Final held-out evaluation | P: M5, H1-TEST done; GoT TEST previously exposed; fresh-family TEST used once in TSP-R | M | — | A **fresh untouched truth TEST** (geonames-derived cities statements, see §6) + a frozen final pass | XF | P2 (end of week) | 1–2 |

## 2. Teammate work: status and what we can complete independently

| Work (owner, branch) | State found in the code/results | Complete independently? |
|---|---|---|
| M6 Gemma Scope splicing (Simba, `main`) | Valid pre-registered VALIDATION; layers 5/12/19; Pile only; T4 bf16 (reproduced on A10G within margin by R3) | Yes: extend to all 26 layers and to OWT-XM, and add GPT-2 under the same protocol (XB). Reuse `src/evaluation/pile_splicing.py` unchanged (identical copy at the same path, so it merges cleanly) |
| R3 ladder (Dhruv, PR #30 open) | Tier A done (H1–H3 supported); Tier B cancelled; RepE-LM part ("R3b") never pre-registered | R3b = our XT-G (RepE part). Tier B: 131k only (RAM limit). Do not edit the branch; wait for the PR |
| `dhruv-causal` (Dhruv, no PR) | GPT-2 GoT causal test (null, 600 controls); SAE-in-the-loop on **length-confounded** prompts; frozen-probe decomposition (VALIDATION); public-SAE splicing; uses **rebuilt** SAEs whose single-run AUROCs differ from ours by up to 0.17 | Port the GoT causal protocol to Gemma (XC-2) and redo SAE-in-the-loop on RepE-LM (XC-4). Cite their GPT-2 result by commit; flag the rebuild discrepancy. Ask Dhruv whether a PR is planned |
| Gemma steering (ours, appendix) | 20 controls, L11, no SAE | Upgrade under XC-3 |
| Method A multilayer (Simba) | VALIDATION, layers 5/6/9 | No action (improvement path superseded by TSP) |

## 3. Results that need improvement, and why (critical)

1. **The frozen logistic gap is the wrong headline truth metric.**
   * Equal-norm isotropic noise produces almost the same gap (S5; dhruv-causal 0.26 vs 0.27–0.30), so it mostly
     measures the fragility of a high-dimensional logistic readout.
   * Lead instead with the re-fit gap, PCA capacity-matched references and the excess over noise.
   * TSP's large frozen-gap gain (0.21 → 0.01) is **partly circular**: 58 % of the frozen probe's norm² lies inside
     the training subspace S. Only the re-fit gap is an independent outcome.
2. **The linear re-fit gap is near its floor.**
   * A random decoder gives a GoT re-fit gap of 0.052, smaller than the old trained SAE's 0.081.
   * So the linear re-fit gap alone cannot show information preservation. It must always be shown with
     error-matched PCA, where SAEs are worse in 13/20 cells, and with MLP re-fits.
3. **Old in-house SAEs were trained on 101,600 tokens.** Any all-layer architecture ranking built on them reflects
   under-training, and S3 shows the rankings change at matched task L0. Cross-model claims must use R2 (in-house) or
   public SAEs.
4. **The in-house JumpReLU is not a valid JumpReLU baseline,** so no JumpReLU conclusion (within or across models)
   should use it.
5. **R2 seeds share one data stream,** so seed sd understates variance. Report document/statement-clustered
   uncertainty (S1 showed a 33–99 % cluster share).
6. **TSP-R is one layer, one architecture and one family.** No transfer was found, and the persona/M4 guards were
   imprecise (TRADE-OFF qualifier).
7. **The Gemma steering "TRUE bias"** (random directions push to about 73 % TRUE) means every Gemma steering effect
   must be read against random directions. 20 controls cannot reach p < 0.05 after any correction.
8. **Cross-model context mismatch:**
   * 1024 GPT-2 tokens cover less text than 1024 Gemma tokens.
   * GPT-2 (2019) never saw the Pile.
   * res-jb was trained on OWT with an unknown document set.

   The XB design controls or discloses each of these.
9. **Hook conventions differ.**
   * GPT-2 runs through TransformerLens with default processing (folded LN, centred writing weights). Our SAEs and
     res-jb expect this.
   * Gemma runs through raw HF. Gemma Scope expects that.
   * Within each model the inputs are consistent, and LayerNorm makes the behaviour identical. Report this; do not
     "fix" it.

## 4. Why previous truth-aware attempts failed (synthesis of existing evidence)

* **Not sparsity suppression.** The frozen GoT gap is flat from 8 to 128 latents (0.31–0.35 at L6). TopK's
  apparent edge disappears at matched L0.
* **Reconstruction and truth are not opposed; MSE is indifferent to truth.**
  * R2 cut NMSE by 40–70 % with no GoT re-fit change (MDE 0.05–0.09).
  * Truth lives in low-variance directions, which MSE barely weights. Error-matched PCA keeps more truth than SAEs.
* **Objective design:**
  * the linear-teacher loss was circular (D16);
  * Method A, KL and R4 targeted behaviour, not truth;
  * TSP worked only because it directly weights a truth subspace, and that subspace was fitted on a single family
    (hence no transfer).
* **Statistical power:**
  * the 3-seed / 112-city pilots had P(pass) ≈ 0.02 at the true effect;
  * TSP-R fixed this with 8 seeds and 635-group cross-fitting.

  Any new improvement test needs TSP-R-level power.
* **Rotation vs loss:** not yet separated directly. Hence the DX diagnostic below.

## 5. Proposed truth-preservation methods (ranked)

| Rank | Method | Justification / evidence | Expected gain | Cost | Risk to recon/L0 | Feasible this week |
|---|---|---|---|---|---|---|
| 1 | **TI-1 TSP generalisation** (existing code): R2 L1, L11 TopK and L8 ReLU/BatchTopK; arms ctrl / tsp / rand; ≥ 5 seeds; cross-fitting | The only supported method; tests whether it is more than one cell | Re-fit gap −0.01 to −0.02 per cell (from TSP-R) | ≈ 4–5 GPU h | Low (TopK fixed L0; non-inferior in TSP-R) | Yes |
| 2 | **TI-2 Multi-family TSP (TSP-MF)**: S spanned by truth directions from TRAIN splits of cities, neg_cities (per polarity), companies and common_claim; leave-one-family-out on DEV; final on the fresh TEST | Targets the observed failure (no transfer) | Transfer to the held-out family | ≈ 3–4 GPU h | Low–medium | Yes |
| 3 | **TI-3 Minimal-pair difference preservation**: ‖(x_a−x_b) − (x̂_a−x̂_b)‖² on true/false minimal pairs; no fitted direction, so no probe circularity | Label-structured but probe-free; generic across families | Uncertain; exploratory | ≈ 2 GPU h | Medium | Pilot only |
| 4 | **TI-4 TSP + KL joint (2 × 2 factorial at fixed k)** | R4 gave −16 % ΔCE (TEST); tests whether truth and behaviour gains add | Both gains at once | ≈ 3 GPU h | Low | Yes, after TI-1 |
| 5 | **TI-G Gemma transfer**: fine-tune Gemma Scope 16k at the depth-matched middle layer with the best of TI-1/TI-2; thresholds frozen; L0 within ±5 % | Cross-model generalisation of the improvement | Unknown | ≈ 3–4 GPU h + 1 day engineering | Medium (JumpReLU L0 drift) | Yes if TI-1/2 are positive |
| — | **Not recommended:** more width (A2 showed no truth gain); adaptive loss weighting (tuning risk); dead-latent work (AuxK already used); CKA/representation alignment as an *objective* (keep only as the DX diagnostic) | | | | | |

**DX (diagnostic, CPU only, both models):** frozen probe on reconstructions after an affine alignment fitted **on
unlabeled OWT activations only**, compared with the re-fit probe and the noise control. Reading:
* if the aligned-frozen gap ≈ the re-fit gap ≪ the raw frozen gap, the change is a rotation or readout shift;
* a remaining re-fit gap above error-matched PCA is information loss.

## 6. Matched cross-model design (XB, XT-G): key decisions

* **Depth.** Gemma `resid_post` L equals `resid_pre` L+1, so its depth fraction is (L+1)/26. GPT-2 `resid_pre` L has
  depth fraction L/12. Two pre-registered triplets:
  * **T-B** (reuses M6/R3): GPT-2 `resid_pre` 3/6/9 (0.25/0.50/0.75) ↔ Gemma `resid_post` 5/12/19 (0.23/0.50/0.77).
  * **T-A** (reuses R2, no GPT-2 retraining): GPT-2 1/8/11 (0.083/0.667/0.917) ↔ Gemma `resid_post` 1/16/23
    (0.077/0.654/0.923). Gemma Scope 16k exists at every layer.
  * All layers for both models are also run, because splicing and probing are cheap.
* **Data:**
  * **Pile arm:** M6's committed 512 + 64 documents.
  * **OWT-XM arm:** a new 512 + 64 sample drawn with a fixed seed from OWT shard 00077. Our R2 stream used shards 78
    and 79, and our 1000-document sample is excluded by hash.
* **Context:**
  * *Primary:* the same text span for both models, i.e. each document cut to the longest prefix that fits 1024 tokens
    under both tokenizers (GPT-2 binds).
  * *Secondary:* 1024 tokens in each model's own tokenizer, the M6-identical setting.
  * BOS is excluded everywhere, and predictions are scored at positions 1 ≤ t ≤ len−2.
* **SAEs:**
  * GPT-2: res-jb (all layers), OpenAI v5-32k TopK, R2 (1/8/11), old in-house (labelled under-trained).
  * Gemma: Gemma Scope 16k canonical (all 26 layers); the 16k L0 ladder and 65k at the triplet layers; 32k at L12.
* **Sparsity strata:**
  * report L0, L0/d_model, d_sae/d_model and L0/d_sae per cell;
  * a cross-model pair is "matched" only if L0/d_model **and** d_sae/d_model are each within ±15 % (pre-registered);
  * all four can't be matched at once, so the binding pair is stated per table.
  * Candidates: R2 (16×, L0 128 → 0.167) ↔ Gemma 32k at L12 (14.2×; the L0-445 SAE gives 0.193); res-jb (32×) ↔
    Gemma 65k (28.4×).
* **Metrics compared across models:** only within-model-normalised ones:
  * ΔCE_SAE/ΔCE_random;
  * excess d = ΔCE_SAE − ΔCE_random relative to CE_orig;
  * loss recovered;
  * KL_SAE/KL_mean-ablation;
  * truth re-fit gap relative to the original AUROC headroom;
  * gap minus the error-matched-PCA gap.

  Raw nats are reported but never ranked.
* **Uncertainty:** 1000-resample document bootstrap for both models. Where seeds exist, a hierarchical
  bootstrap (seeds × documents). Holm correction within each pre-registered family.
* **Truth (XT-G):**
  * same GoT city-group splits, A/B groups and RepE-LM statement splits;
  * Gemma final-token `resid_post` caches at all 26 layers (slot position too for RepE-LM);
  * evaluation probes fitted on original TRAIN only;
  * VALIDATION for all decisions.
* **Fresh TEST (XF).** GoT TEST rows were exposed for GPT-2 method decisions. A new cities-style family built from
  `got-repo/datasets/geonames.csv` (cities absent from `cities.csv`, same template and country pool) is gated on DEV
  AUROC, as in TSP-R Phase 0. Its TEST split is locked until the final pass.

## 7. Execution queue and resource budget

| Day | Item | Type | GPU h | New disk |
|---|---|---|---|---|
| 1 | Pre-registrations XB, XT-G (hash-frozen); model-agnostic hook backend + tests; OWT-XM sample; Gemma Scope downloads; smoke tests | eng + inference | 0.5 | ≈ 16 GB (SAE weights, re-downloadable) |
| 1–2 | **XB** matched splicing, both corpora, all layers | inference | 5–6 | < 1 GB |
| 2 | **XT-G0/XT-G** Gemma baselines, gates, truth/honesty preservation, PCA-CM, direction fidelity; DX on both models | inference + CPU | 1.5 (+ 4 CPU h) | ≈ 3–4 GB caches |
| 3 | **TI-1** TSP generalisation (pre-registered, ≥ 5 seeds) | training | 4–5 | ≈ 1 GB |
| 3–4 | **XC-1** Gemma site sweep + M4 + skyline; **XC-3** steering parity | inference | 4–5 | small |
| 4 | **TI-2** TSP-MF; **XC-2** GoT causal on Gemma | training + inference | 6–7 | ≈ 1 GB |
| 5 | **XC-4** SIL (GPT-2 gate → Gemma); **TI-4** TSP+KL if TI-1 is positive | mixed | 4–5 | small |
| 5–6 | **TI-G** Gemma Scope TSP transfer (feasibility smoke first) | training | 3–4 | ≈ 1–2 GB |
| 6 | P2 if time allows: XG-IH Gemma in-house TopK; XJ valid GPT-2 JumpReLU | training | 4–9 | ≈ 2–3 GB |
| 7 | **XF** fresh-TEST build (DEV gate) + single frozen final pass; integrity checks; local commits | inference | 1–2 | small |

* **Totals:** P0 ≈ 7–8 GPU h, P1 ≈ 18–24 GPU h, P2 ≈ 8–12 GPU h. That fits one GPU running continuously; engineering
  time is the real bottleneck.
* **Disk:** about 22–26 GB new against 34 GB free. This leaves too little headroom. The unrelated `~/Exam*` folders
  (about 82 GB) and `../backups` (29 GB) are not ours to delete; that is the user's decision.

## 8. Git conflict-avoidance plan

* **Stay on `sae-research`.**
  * Teammate code is read with `git show`.
  * If teammate code must be executed, use a separate worktree (`../wt-<branch>`), never a checkout in this
    workspace.
  * No pull, merge, rebase or push.
* **New files only:**
  * `scripts/cross_model/` (new);
  * `scripts/gemma/10_*.py` onward (main uses 02–04 and R3 uses 05–08);
  * `scripts/geometry_of_truth/32_*` onward;
  * `src/models/backends.py` (new);
  * `src/tests/test_cross_model_*.py`;
  * `results/cross_model/`, `results/gemma/<new>/`, `results/geometry_of_truth/<new>/`.
* **Shared code from `main`:** `src/evaluation/pile_splicing.py` is copied **byte-identical** to the same path, so it
  merges cleanly. Its SHA is recorded.
* **Never edited:**
  * `results/gemma2_pile/**`;
  * `scripts/gemma/02–08`;
  * `presentation/`;
  * the manuscript;
  * any teammate branch.
* **Shared journals are append-only.** The conflicts there are unavoidable but mechanical. They are resolved only at
  an approved merge.

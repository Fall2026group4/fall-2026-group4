# E1: TopK trained at task-level matched sparsity (layers 1, 8, 11). Status: PRE-REGISTERED (frozen 2026-11-21; approved by the user)

Evidence level, once run: **VALIDATION**. No test row is read: OWT test evaluation is deliberately **not** run for
these SAEs. Nothing in this folder has been run.

## 1. Before implementation

### 1.1 Question, motivation and hypotheses

**Question.** Are the architecture differences reported in M2 (information) and M4 (causal preservation) real, or
an artefact of matching sparsity on OWT rather than on the task prompts?

**Motivation:**
* All architectures are matched at L0 ≈ 128 on OWT. On task prompts, however, ReLU and JumpReLU fire very
  differently, while TopK fires exactly k = 128 latents:
  * GoT cities: 23–110 latents;
  * RepE-LM persona: 142–214 latents.
* The figures above are M2 validation means at layers 1, 8 and 11 (`summary_by_layer_arch.csv`, `task_l0_mean`).
* Any TopK vs ReLU/JumpReLU difference on a task is therefore confounded with a task-level L0 difference of up to
  5×. This is the prompt-sparsity limitation in README §6.4 and paper §13.

**Pre-registered hypotheses:**

| ID | Hypothesis | Comparison |
|---|---|---|
| E1-H1 | An architecture difference on task t at layer L keeps its sign when TopK is matched to the other architecture's task-level L0 | (X − TopK@128) vs (X − TopK@k*), X ∈ {ReLU, JumpReLU} |
| E1-H2 | TopK's own task faithfulness depends on k | TopK@k* vs TopK@128 (paired by seed) |

* **Classes for E1-H1,** per (layer, task, X, metric):
  * **PERSISTS:** same sign, and the 95 % CI at k* excludes 0;
  * **ATTENUATED:** the CI at 128 excluded 0, the CI at k* includes 0;
  * **REVERSED:** opposite sign, CI excludes 0;
  * **NO DIFFERENCE:** neither CI excludes 0.

### 1.2 Design

* **Layers:** 1, 8 and 11, the M4 sites (L1 slot, L8 final, L11 final). M2 readers exist at all three.
* **Tasks:** GoT cities (M2 truth task) and RepE-LM persona (M2/M4 persona task). The persona-matched SAEs are also
  evaluated on RepE-LM neutral pairs, the M2 control.
* **Targets k*:**
  * For each (layer, task, X), k* = round(mean task-prompt L0 of X's 3 base SAEs), measured on **TRAIN** prompts in
    `--prepare` before any training. This gives 3 layers × 2 tasks × 2 X = **12 k values**.
  * Illustration from M2 *validation* means. The actual k* come from train prompts:
    * L1: GoT ≈ 32 / 23, RepE ≈ 214 / 208;
    * L8: GoT ≈ 77 / 63, RepE ≈ 161 / 167;
    * L11: GoT ≈ 110 / 92, RepE ≈ 160 / 142 (ReLU / JumpReLU).
* **New training: 36 TopK SAEs** (12 k* × seeds 42/1042/2042):
  * the same protocol as the 108 baseline SAEs: `train_one` from `scripts/openwebtext/03_training_sae.py`, imported
    unchanged;
  * width 3072, Adam lr 3e-4, batch 256, 15,000 uninterrupted steps, OWT train activations (101,600 tokens, cached);
  * BOS excluded.
  * Only k changes. TopK needs no sparsity controller.
* **Why not script 13:** it also evaluates on the OWT test split. A new script,
  `scripts/shared/e1_task_l0_matched_topk.py`, calls `train_one` and the M2/M4 evaluation functions, with no test
  loading.
* **Checkpoints (gitignored):** `results/openwebtext/e1_task_matched_topk/layer_L/topk_k{k}_seed{S}/checkpoint_step_15000.pt`.
* **Comparison groups,** per (layer, task):
  1. X (base, OWT-matched) vs TopK@128 (existing): reproduces the M2/M4 difference;
  2. X vs TopK@k* (new): the task-matched difference;
  3. TopK@k* vs TopK@128 (paired by seed).

### 1.3 Metrics (direction), on the task each SAE was matched on

* **Information (M2):** re-fit logistic gap (↓, primary), frozen logistic gap and its noise reference (↓), task
  relative error (↓). PCA at matched task NMSE is reported.
* **Causal (M4, RepE-matched TopK only, at that layer's site):** `preservation_rec` (↑, primary for RepE), Shapley
  share (↑), |I|/T_full (↓).
* **Reconstruction (reported):** OWT validation EV, L0, dead fraction. TopK@k* is *not* OWT-matched, by design.
* **Behaviour:** OWT spliced ΔCE and KL (↓, validation).

### 1.4 Statistics

* **Between architectures** (different SAEs; seeds are not paired across architectures): mean ± sd over seeds and a
  Welch 95 % t-interval for the difference.
* **TopK@k* vs TopK@128:** paired by seed, with a 95 % t-interval (df = 2).
* Per-SAE statement-bootstrap CIs as in M2/M4.
* No pooling across layers.
* BH correction over the E1-H1 family (3 layers × 2 tasks × 2 X × 2 primary metrics), reported alongside the
  uncorrected classes.

### 1.5 Checks before training (`--prepare` → `pretraining_checks.json`)

* Split verification.
* **Training reproduction:** retraining TopK@128 seed 42 at layer 8 through the new script must reproduce the
  existing uninterrupted checkpoint bit for bit (or within 1e-6).
* **Evaluation reproduction:** the new evaluation path must reproduce M2's committed values for the existing layer-8
  SAEs, and M4's for L8 final.
* k* computed on TRAIN prompts and written to `preregistration.json` before training.
* SHA-256 hashes of the scripts, split files and pre-registration.
* `test_rows_used` = 0.

### 1.6 Compute

* **Training:** 36 runs at about 40 s each (baseline TopK median 38 s on the A10G), so about **25 min**. Plus one
  reproduction run.
* **Evaluation:**
  * reconstruction and splicing for 36 SAEs: about 15 min;
  * M2 readers and PCA references: about 10 min;
  * M4 for the 18 RepE-matched TopK SAEs: about 20 min (about 60 s per SAE).
* **Total about 1.2 h GPU.** Peak memory is as in the baseline training (single A10G). Disk: 36 × about 19 MB
  checkpoints, gitignored.
* **No download.** OWT activations for layers 1, 8 and 11 are cached.

## 1.7 Additions for review (2026-11-21; nothing above is changed)

**Matched-sparsity requirement:**
* TopK@k* fires exactly k* latents per token, so its task L0 equals k* by construction.
* A k* is valid only if |k* − L0_task(X)| / L0_task(X) ≤ 10 % (the CLAUDE.md baseline rule), where L0_task(X) is
  measured on TRAIN prompts. Rounding makes this hold automatically.
* L0 is also reported on OWT VAL and on the other task. TopK@k* is deliberately not OWT-matched.

**Evaluation splits:**
* GoT cities: readers fitted on TRAIN and scored on VALIDATION (M2 protocol).
* RepE-LM persona and neutral: TRAIN / VALIDATION.
* OWT VAL for reconstruction and splicing.
* **No TEST split is loaded** (`test_rows_used = 0` is asserted).

**Success criteria (pre-registered, unchanged in substance):**
* E1-H1 classes per cell (PERSISTS / ATTENUATED / REVERSED / NO DIFFERENCE).
* The headline is the count of ATTENUATED + REVERSED among cells where the k = 128 difference was significant.
* "The M2/M4 architecture differences are robust to task-level matching" is claimed **only if** ≥ 75 % of those
  cells PERSIST after BH.
* E1-H2 uses the IMP-L paired rule (all 3 seeds the same sign and |mean d| > 2 sd).

**Behavioural cost limits (reported, not a guard, because TopK@k* is not OWT-matched by design):**
* OWT spliced ΔCE and KL of TopK@k* vs TopK@128.
* Any cell where TopK@k* has OWT EV < 0.5 is flagged as "degenerate sparsity" and reported separately. This mainly
  matters at L1 GoT, where k* ≈ 23–32.

**Failure conditions (stop and report, no re-tuning):**
1. The training reproduction gate fails (TopK@128 L8 seed 42 differs from the committed checkpoint beyond 1e-6).
2. The evaluation reproduction gate fails (M2 / M4 committed values not reproduced).
3. Any k* violates the ±10 % rule.
4. Any run diverges (NaN, or OWT EV < 0). It is flagged and excluded from classes, never re-run with new
   hyperparameters.

**Scope caveat (added):**
* E1 retrains the **old 3072-width recipe** (101,600 tokens), because its question concerns the M2/M4 comparisons,
  which used those SAEs.
* These SAEs are under-trained (audit §3.3). E1's conclusions therefore apply to the old-SAE architecture comparisons
  only, not to R2 or to cross-model claims.
* An R2-recipe E1 (36 × 100M tokens) is out of scope.

**Storage:**
* 36 × about 19 MB checkpoints, about 0.7 GB, written to `/mnt/scratch/checkpoints/e1/` during training.
* Keepers (all 36, because they are small) are copied to the gitignored EBS path listed in §1.2 and
  SHA-256-verified.

## 1.8 Freeze notes (2026-11-21, before any E1 training or outcome; approved by the user as Option B item 3)

* **k\* (TRAIN prompts; `k_star.json`):**

  | Layer | GoT ReLU / JumpReLU | RepE ReLU / JumpReLU |
  |---|---|---|
  | L1 | 32 / 23 | 216 / 214 |
  | L8 | 76 / 63 | 161 / 170 |
  | L11 | 111 / 92 | 162 / 141 |

  All 12 are distinct, so 36 SAEs are trained, and all are within ±10 % of task L0. Split checks are in
  `pretraining_checks.json`: test_rows_used 0, group overlap 0 for all three tasks.
* **Primary BH family (clarified):**
  * GoT cities: refit logistic gap;
  * RepE persona: refit logistic gap and M4 `preservation_rec` (M4 exists for RepE only);
  * per (layer, X): 3 × 2 × 3 = **18 tests**.
* **Classes:** E1-H1 classes are computed uncorrected (`cls`) and with BH on the k\* contrast (`cls_bh`). The §1.7
  headline uses `cls_bh`.
* **Gates (`gate` stage; E1 stops if any fails):**
  * G1: training reproduction ≤ 1e-6;
  * G2: M2 evaluation reproduction ≤ 1e-6 (M2's exact job-seed index and bootstrap);
  * G3: M4 `preservation_rec` reproduction ≤ 1e-6.
* **Seeds for new evaluation calls:** M2 seed SEED + 1000·L + 500 + 10·j + task index, and +900 for the PCA reference.
  They are fixed here.
* **Script:** `scripts/shared/e1_task_l0_matched_topk.py`; tests in `src/tests/test_e1_task_l0_matched.py`.
* **Checkpoints:** written to `/mnt/scratch/checkpoints/e1/` (ephemeral), and copied with hash verification to the
  gitignored `results/openwebtext/e1_task_matched_topk/` as keepers.

# Code audit — professor review of 2026-11-01

Audit date: 2026-11-04. Branch `sae-research` @ `326b0f6` (up to date with `origin/sae-research`).
Auditor: Claude (Opus 5.5), working with Aditi Shukla.

Everything below was checked against the code, configs, checkpoints, and saved CSVs. Each item
gives its evidence (file:line or result file). Items are classified as:

* **BUG**: verified implementation bug
* **METH**: methodological inconsistency
* **CTRL**: missing experimental control
* **MISSING**: missing experiment
* **DOC**: documentation or provenance problem

Negative scientific results are not called bugs.

## 0. Repository state at start

| Item | Status |
|---|---|
| Working tree | Clean apart from untracked items: `data/activations/` (2.3 GB OWT activation cache), `data/got/splits/` (**needed** by GoT scripts 03–09 but not committed), `data/openwebtext/metadata.json`, two `.bak` files (`10_reconstruction_evaluation.py.bak` is byte-identical to the live script; the GoT `07_…py.bak` is an older copy). Nothing was modified or deleted. |
| GPU | NVIDIA A10G, 23 GB, driver 595.99, CUDA 13.2 |
| Python env | `~/sae-project/venv`: Python 3.14.4, torch 2.14.1+cu130, transformer-lens 3.9.0, sae-lens 6.54.12, transformers 5.21.0, sklearn 1.9.1. `pytest` was missing, so I installed it. |
| HF cache | gpt2, google/gemma-2-2b (bf16 weights cached), jbloom GPT-2 res SAEs (layer 8 only cached) |
| Other branches | `aditi` has `scripts/run_gemma_baseline.py` and Gemma result tables. The script is **absent on `sae-research`**, although `gemma_run.log` and `gemma_neg.log` on this branch were produced by it. `dhruv`/`simba-*` hold older GoT steering code. Nothing was merged. |

### Terminology problem (DOC, affects every section)
In the OWT, GoT and RepE pipelines, **"pretrained" does NOT mean an SAELens/public pretrained SAE**.
It means *our own SAE trained on OpenWebText* at layer 6 for 15 000 steps
(`results/openwebtext/convergence/layer_6/{arch}_l0_128_resume_5000_to_15000/checkpoint_step_15000.pt`;
see `scripts/repe/04_sae_reconstruction.py:30-34`, `scripts/Geometry of Truth/04_sae_truth_preservation.py:49-53`).
The only public pretrained SAE in the repo is the legacy GoT layer-10 experiment
(`src/sae/geometry_sae.py:7-8`, `gpt2-small-res-jb`, d_sae = 24 576). Reports should say
"OWT-trained (base)" or state the meaning explicitly.

### Width (DOC/METH)
All in-house SAEs have d_sae = 3072, which is only 4× the 768-dimensional residual. That is very narrow
compared with public GPT-2 SAEs (32×). Conclusions about "ReLU vs TopK vs JumpReLU" hold only
in this narrow-width, small-data regime: 101 600 training tokens, so 15 000 steps × 256 ≈ 38 epochs.

---

## Task A — OWT optimizer inconsistency

**Verified.**

| Layer | Path to 15 000-step checkpoint | Procedure (code) |
|---|---|---|
| 6 | `05/06/07_*_convergence.py` scratch 0→5000, saves Adam and controller state at 5000; `08_increasing_convergence.py` resumes 5000→15000 | **Adam state restored** (`08:…optimizer.load_state_dict`), controller restored, RNG **re-seeded** to seed+5000 (original state not saved) |
| 2, 10 | `03_training_sae.py` sweep (1000 steps, saves weights only); `11_multilayer_convergence.py` warm-starts 1000→15000 | **Fresh Adam** (`11:…torch.optim.Adam(model.parameters()…)`), sparsity coefficient **reset to 0.001** at step 1000, new seed |

* The labels `optimizer_resumed` / `warmstart_optimizer_reset` in
  `results/openwebtext/reconstruction_evaluation/layers_2_6_10/test_reconstruction_metrics.csv` are
  hard-coded in `10_reconstruction_evaluation.py` (`'optimizer_resumed' if layer == 6 else …`). They describe the
  procedure correctly.
* Restoring Adam state is implemented correctly where it is used (08 and 11 restore `optimizer_state_dict`
  on interrupted resumes).
* Within-layer architecture comparisons are internally consistent: all three architectures at a given
  layer share one procedure. **Cross-layer** comparisons (e.g. "EV drops from layer 2 to 10") mix two
  procedures **(METH)**. Layers 2 and 10 also start from a *different trajectory*: the 1000-step sweep
  checkpoint, whose controller had different dynamics from the 5000-step layer-6 run.
* Reset vs resume also means the ReLU/JumpReLU sparsity coefficient restarts at 0.001 at step 1000
  for layers 2 and 10. That causes a transient L0 overshoot. The histories show it (e.g. layer-2 JumpReLU val L0 135.8 at step 1500).
* **Test-token count:** the test count is 12 700 for all nine checkpoints, i.e. 100 docs × 127 non-BOS tokens. The same
  BOS-exclusion rule is used everywhere. Validation is also 12 700. Training and evaluation both drop position 0.
  This part is **consistent**.
* **Fine-tuning is unaffected by this issue.** All fine-tuning, both reconstruction-only and truth-aware
  (`scripts/Geometry of Truth/06_finetune_sae.py`), is layer-6 only and starts from the layer-6 checkpoint, so the
  pretrained-vs-fine-tuned comparison never crosses procedures.
* **Correction:** `scripts/openwebtext/13_consistent_protocol_retrain.py` retrains layers 2/6/10 × 3 architectures × 3 seeds
  with one uninterrupted run, reusing the *original* `train_one` and `evaluate` functions. It writes to a new directory. Originals are kept.

## Task B — Architecture comparison fairness

| Check | Finding |
|---|---|
| Target vs actual L0 | TopK is exact by construction. ReLU/JumpReLU use an adaptive L1/sigmoid-count controller (`03_training_sae.py:train_one`). Matching is not guaranteed, and the script says so. |
| "2 of 12 JumpReLU matched at L0=32" | **Verified**: 2/12 under the 10% criterion (`04_comparing_sae_layers.py:111`, `l0_error <= 0.10`), layers 2 and 3 only. Under the training script's own 20% criterion (`--match-tolerance 0.2`) it is 5/12. **Two different matching criteria coexist (METH/DOC).** |
| `architecture_summary.csv` | **No script in the repo generates it (DOC).** Its columns are means over **all 12 layers including unmatched runs** (verified by recomputation; e.g. JumpReLU L0=32 "mean_actual_l0 = 32.51" averages layer-0 L0 = 0.004 with layer-6 L0 = 45.2). It also averages raw MSE across layers whose activation scales differ by ~1000× (layer 0 MSE 0.005, layer 11 MSE 19.9). **That table is not a valid architecture comparison (METH).** |
| Width / init / LR / seed / batch / steps | Identical across architectures (3072, Kaiming encoder, unit-norm decoder, Adam 3e-4, seed 42, batch 256, 1000 steps in the sweep). Same seed means identical encoder/decoder init for all three. **Fair on these axes.** |
| Training budget | The sweep uses 1000 steps: every architecture is far from converged (layer-6 TopK val MSE is 2.19 at 1000 vs 0.66 at 15 000), and JumpReLU's controller is still oscillating. Sweep rankings reflect **speed of convergence**, not converged quality. |
| JumpReLU implementation | A custom sigmoid straight-through surrogate, not the published JumpReLU estimator (stated in the docstring). Layer 0 collapsed (L0 ≈ 0.004–1.1 for all targets). Results should be labelled "surrogate JumpReLU". |
| Normalization | No input normalization for any architecture (raw residuals with a learned pre-encoder bias). Consistent, but MSE is scale-dependent across layers. |
| Splits | All use OWT train for fitting and OWT validation for the sweep metrics. Test is untouched by the sweep. |

## Task C — Causal controls

* `10_steering_pilot.py` uses **4** validation questions and 1 random direction. `11_heldout_steering.py`
  uses **9** test questions and **20** random directions. `12_bootstrap_causal_fidelity.py` bootstraps the
  agreement between "original-direction effect" and "SAE-direction effect".
* **CTRL**: there is no positive control, no control distribution of adequate size, and no significance test of the target effect against controls.
* **Observed sign problem (not a bug, a result):** on the 9 test questions, +10 × honesty direction *lowers* the
  correct-answer margin (−0.141) and −10 *raises* it (+0.122). The 20 random directions produce |Δ| up to ~0.08.
* **What "SAE steering" means here:** `11_heldout_steering.py` steers the *unmodified* GPT-2 with the mean-difference
  direction computed **from SAE reconstructions**. No SAE is in the forward pass and no SAE feature is
  intervened on. The 0.93–0.99 "effect correlation" therefore measures how similar two 768-d directions
  are (cosine 0.75–0.87). It is **not evidence that SAE features are causal (METH)**.
* No written specification of a "200-random-control" protocol exists anywhere in the repo (searched all .md/.py/.tex/.txt/.yaml).
  The design I implemented is documented as **my design following the user's description**, not as the professor's text.

## Task D — Gemma 2 2B steering (`aditi:scripts/run_gemma_baseline.py`)

Logged result (`aditi:results/tables/gemma/cities/judgment_summary.csv`): baseline judged-TRUE 61%;
±0.25 → 72.7% / 75.0%; ±0.5 → 100% / 97.7%. **Verified: both signs increase TRUE.**

Implementation review:

1. Direction = mean(true) − mean(false) of `hook_resid_post` at the **last token of the bare statement**
   (no few-shot context, no " This statement is:" suffix). Sign is correct: true minus false, unit-normalized.
2. Steering is added at `hook_resid_post` of the same layer, to **every token of the test segment**,
   including " This statement is:". These positions and contexts were never used to estimate the
   direction. **METH: distribution shift between estimation and intervention context.**
3. Strength = α × **mean total activation norm** (147 at layer 11). So α = 0.25 adds a vector of norm ≈ 37
   to every test-segment token. The relevant scale is the true/false separation along the direction,
   which the script never reports. If the separation is much smaller than 37, every α in the grid is a
   large off-manifold perturbation. **METH (most likely cause; tested in my sweep).**
4. No random-direction control of matched norm, so "the model drifts to TRUE under any large perturbation"
   (response bias) cannot be distinguished from a direction-specific effect. **CTRL**
5. Splits: `load_data` makes a *random stratified row split*, not the grouped-by-city split in
   `data/got/splits`. The same city appears in train and test, a **leakage** that inflates the layer-table AUROCs
   (many 1.000). **METH**
6. Best layer uses `idxmax` on AUROCs **rounded to 3 decimals**, so ties at 1.000 resolve to the first layer. **minor BUG**
7. Scoring: TRUE iff logp(first token of " TRUE") > logp(first token of " FALSE"). This is OK if both are single tokens (verified in my script).
   Only a binary outcome is reported. A continuous logit difference is more sensitive.
8. `gemma_neg.log` (neg_cities) crashed with `torch.cat(): expected a non-empty list`: `nw` is empty because negated
   statements contain " is not in ", so the next-word part never ran. **BUG** (neg_cities run incomplete).

## Task E — Geometry of Truth

* Reported 0.7689 / 0.7556 / 0.85284 / 0.85079 **match** `results/geometry_of_truth/sae_faithfulness_results.csv`
  and `baseline_probe_results.csv`.
* That legacy experiment (`scripts/run_geometry_faithfulness.py`, `src/evaluation/geometry_sae_faithfulness.py:evaluate_reconstructed_probe`)
  **trains a new probe on the SAE reconstructions**. It does *not* apply the original probe. So it measures
  "truth is still linearly decodable after reconstruction", not "the original truth representation is preserved". **METH**
* Its inputs (`data/raw/cities.csv`, `data/processed/geometry_of_truth/*`) are **not present** (DOC). They are reproducible from
  `data/got/cities.csv` with the documented city-grouped 70/30 split.
* Single split, single probe seed, no CI. Test n = 450.
* The newer pipeline (`scripts/Geometry of Truth/03–09`) does use a **frozen** probe trained on original activations and applied to reconstructions
  (`09_final_test_evaluation.py`), with grouped splits and test touched once. That is methodologically sound,
  but it uses layer 6 and in-house SAEs, so its numbers are not comparable with the legacy layer-10 numbers.
* **New finding to report:** on GoT final-token activations the layer-6 OWT-trained ("pretrained") SAEs bring the frozen
  probe almost to chance (val AUROC 0.52–0.57 vs 0.84 original; `results/got/sae_finetuning/*/validation_comparison.csv`).

## Task F — RepE

* All reported values **verified** in `results/repe/*.csv` except one: the user-supplied
  "cross-wording JumpReLU pretrained ≈ 0.8010" does **not** match the saved value **0.7394**
  (`results/repe/cross_wording/evaluation.csv`). The closest saved number is the *validation* AUROC of JumpReLU pretrained (0.7979).
* Split leakage: 0 statement and 0 prefix overlap between train/validation/test (checked).
* Cross-wording is held out in instruction wording but uses the **validation statements**. Layer 6 was chosen as the
  max-validation-AUROC layer (`results/repe/original_baseline.csv`), so validation AUROCs are optimistically biased.
  The test split has never been evaluated. **METH**: I add test-split numbers.
* Pairwise accuracy is 1.0 for every model, so it is saturated and uninformative. AUROC pools across statements.
  Both are reported, and they are not confused.
* The direction definition (train-split mean difference at layer 6, final token) is consistent across 03/05/06/09/10/11.

## Task G — Fine-tuning

* Fine-tuning: 500 steps, LR 1e-5, on GoT train activations, starting from the layer-6 OWT checkpoint. Truth-aware adds
  0.1 × normalized MSE between the frozen **logistic** probe's logits on x and on x̂.
* On the GoT test set the evaluation uses an **MLP** probe (`09_final_test_evaluation.py`). The truth-aware objective
  targeted a different (logistic) probe, so the test metric is misaligned with the training objective. **METH**
* Reconstruction-only fine-tuning raises L0 substantially (ReLU 65→144; JumpReLU 45→100 on GoT). Fine-tuned and base SAEs therefore
  **differ in sparsity**, so their comparisons are not sparsity-matched. **METH**
* OWT generalization of the fine-tuned models was evaluated on **validation**, not test (`12_finetuning_generalization.py`).

## Task H — AlignSAE

Not implemented anywhere in the repo or on other branches. The truth-aware fine-tuning is a
**probe-logit-preservation** objective. It is *not* AlignSAE, which assigns concepts to dedicated SAE latents with
an explicit alignment loss. SHAP and ProbeSAE are also not implemented.

## Task I — QA

* `pytest` was not installed. `src/tests` contains tests only for the legacy steering and detection utilities.
* No accidental gradient updates during evaluation were found: the eval paths use `inference_mode`/`no_grad`.
* Hooks: the RepE/GoT GPT-2 hooks use `hook_resid_pre`, consistently with extraction. Gemma uses `hook_resid_post`, also consistent within itself.
* Device handling: several scripts silently fall back to CPU (`10_…`, `12_…`, `repe/*`). This is harmless for evaluation.

---

## Prioritized correction plan and cost (A10G)

| # | Action | Rerun needed? | Estimated cost |
|---|---|---|---|
| 1 | Task A: uninterrupted 15k-step retrain, L2/6/10 × 3 archs × 3 seeds + test eval | Yes (27 small runs) | measured 5 ms/step, so ~35 min |
| 2 | Task B: matched vs unmatched tables from existing sweep + corrected 15k runs | No (re-analysis) | seconds |
| 3 | Task C: 200-control causal experiment, GPT-2 L6, held-out RepE test prompts | New experiment | ~5–10 min |
| 4 | Task D: Gemma symmetric fine-grained sweep with norm-matched random controls, grouped splits | New experiment | ~20–30 min (model cached) |
| 5 | Task E: reproduce legacy layer-10 result + add frozen-probe comparison | Yes (needs 150 MB jb layer-10 SAE download) | ~3 min |
| 6 | Task F: recompute RepE metrics + add test split | Re-analysis | ~1 min |
| 7 | Task G: unified comparison on identical held-out sets (OWT test, GoT test with frozen logistic and MLP probes, RepE test, causal) | Re-analysis | ~5 min |
| 8 | Task B follow-up: re-run sparsity matching at L0 = 32/64 with longer training | **Proposed only** (see report) | ~108 runs × ~25–75 s ≈ 1–2 h |
| 9 | Task H: AlignSAE | **Proposed only** | — |

Existing results that remain valid as they stand: within-layer architecture comparisons at L0 = 128 (15k);
GoT frozen-probe layer-6 test evaluation (`results/got/final_test_evaluation`); RepE val/cross-wording numbers (as validation numbers);
OWT 12 700-token test evaluation.

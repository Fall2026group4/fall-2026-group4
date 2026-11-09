# IMP-L: multi-layer improvement experiment, layers 3, 8 and 9 (2026-11-09). Status: PRE-REGISTERED; training awaits user sign-off on the configuration

Evidence level: validation finding (once run). No test row is read by training or evaluation.

## 1. Before implementation

* **Research question:** do the layer-6 fine-tuning methods improve SAE faithfulness at other layers (3, 8, 9),
  relative to the original SAEs and to an OWT-only continued-training control, at matched sparsity?
* **Hypotheses:**
  * Method A lowers the independent GoT truth gap;
  * output-KL lowers OWT and GoT spliced KL relative to Method A.

  Both were met only partly, or not at all, at layer 6.
* **Why it's needed:**
  * Instructor Review #8's blocker for the 11-10 session: "one table: which fine-tuning method, at which layers,
    improves which faithfulness measure, with seeds and CIs … run it at two more layers (e.g. 3 and 9)".
  * The 11-03 session note: "fine-tune other layers".
  * The main research question says "across transformer layers"; every fine-tune so far is at layer 6.
  * Layer 8 is also the M4 mid-depth causal site, so the layer-8 SAEs feed E3 (causal preservation after fine-tuning).
* **Design:**
  * **Model and hook:** GPT-2 Small, frozen, at `blocks.L.hook_resid_pre`, L ∈ {3, 8, 9}.
  * **SAEs:** ReLU, TopK and JumpReLU; the uninterrupted 15k-step checkpoints, seeds 42/1042/2042.
  * **Arms:**

    | Arm | Objective |
    |---|---|
    | base | none |
    | `owt_only` | control: 500 steps on OWT only |
    | `method_a` | 0.5·MSE(GoT-A) + 0.5·MSE(OWT) + penalty |
    | `kl` | Method A + α·0.5·(KL_OWT + KL_GoT), KL(p_orig ‖ p_spliced), with α per base SAE from train-only scale matching at step 0 |

    The `kl` arm is the layer-6 E2E Method B.
  * **Budget:** Adam lr 1e-5, 500 steps, clip 10, decoder renormalisation. The controller continues from the base
    coefficient (target 128). TopK k = 128.
  * **Runs:** 81 new runs = 3 layers × 3 arms × 3 architectures × 3 seeds.
  * **Code reuse:** the training code is imported unchanged from scripts 14 and 17; only their layer constants are set.
* **Data:**
  * training uses OWT train and GoT train groups A only, with no labels;
  * evaluation uses OWT validation, GoT validation (geographic and numerical) and RepE-LM validation (persona
    T1–T4, H vs U).
* **Comparisons:**

  | X vs Y | What it establishes |
  |---|---|
  | `method_a` vs base | does GoT reconstruction fine-tuning improve truth preservation at this layer? |
  | `method_a` vs `owt_only` | is the effect due to the GoT data, rather than to 500 extra steps? |
  | `kl` vs `method_a` | does the output-KL term add behavioural faithfulness, and at what truth or sparsity cost? |
  | `kl` vs base | net effect of the full objective |

* **Metrics** (direction in brackets):
  * **reconstruction:** OWT EV (↑), MSE (↓), cosine (↑), L0 (match), dead fraction (↓); GoT relative error (↓);
    L0 on GoT and RepE-LM prompts (reported);
  * **representation:** GoT B-group MLP and logistic gaps (frozen, fitted on original B rows; ↓); M2-style
    re-trained logistic gap on GoT geographic and RepE-LM persona (↓);
  * **behavioural splicing:** OWT ΔCE and KL (↓), GoT statement KL (↓), cities truth-preference change (↑).
* **Pre-registered criteria and guards** (`preregistration.json`):
  * "consistent" = better on all 3 seeds and |mean paired difference| > 2 × its sd;
  * guards: OWT L0 within ±5 % of base; OWT EV ≥ base − 0.005 (kl: ≥ Method A − 0.02); ΔCE ≤ 1.10 × base; cities
    preference ≥ base − 0.018;
  * classes: PRELIMINARY IMPROVEMENT / NO IMPROVEMENT / NOT CREDITABLE (guard failed) / REGRESSION. CONFIRMED is not
    used, because this is validation only.
* **Statistics:** per layer × architecture, paired by seed. Mean, sd, per-seed values, and a 95 % t-interval over
  the 3 seeds (t₀.₉₇₅,₂ = 4.303; wide by construction). No pooling across layers.
* **Layer-3 caveat:** the original GoT logistic probe is below the 0.70 informative threshold at layer 3, so
  layer-3 truth results are descriptive.
* **Leakage prevention:**
  * the A/B split hash is verified (`S14.verify_split`);
  * probes are fitted on original activations of groups B only;
  * GoT test rows are dropped before scoring;
  * RepE-LM contains no test rows.
* **Checks before pre-registration** (`--prepare` → `pretraining_checks.json`):
  * the reuse path reproduces Method A TopK seed 42 and Method B TopK seed 42 at layer 6 **bit for bit**;
  * cached GoT-A activations equal a fresh forward pass at each layer;
  * the identity splice gives KL = 0.

  The evaluation code was smoke-tested on the 9 layer-8 base SAEs (outputs outside the repository). It reproduced
  M2's committed layer-8 re-trained gaps exactly.
* **Files:**
  * `scripts/geometry_of_truth/22_multilayer_finetune.py`, `scripts/geometry_of_truth/23_multilayer_evaluation.py`;
  * this folder;
  * checkpoints (gitignored) in `results/geometry_of_truth/sae_finetuning/multilayer_finetune/`.
* **Pre-training hashes:** `pretraining_hashes.json`.

## 2. After execution
Not run yet.

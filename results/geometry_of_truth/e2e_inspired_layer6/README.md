# E2E-inspired SAE fine-tuning at layer 6 (2026-11-07). **PILOT, validation only.**

**This is E2E-inspired fine-tuning, not an exact reproduction of Braun et al. 2024 (arXiv:2405.12241).** It differs from the paper in that it:
* fine-tunes existing local SAEs for 500 steps instead of training from scratch;
* keeps the local reconstruction term;
* keeps each architecture's own sparsity controller;
* uses KL(p_original ‖ p_spliced);
* never replaces BOS;
* normalises the downstream terms by variance.

**Methods:** nested, with identical local batches and random draws to Method A.
* **B** adds α · output KL, computed on 8 OWT train sequences plus the step's 128 GoT-A statements.
* **C** additionally adds β · mean downstream NMSE (`blocks.7..11.hook_resid_pre`).
* α and β come from a train-only scale-matching rule (`alpha_beta.json`).

**Checks before training** (`pretraining_checks.json`, `pretraining_hashes.json`): all passed. They covered:
* split, hooks, KL direction;
* the downstream loss against numpy (relative difference 2e-8);
* gradients reaching only the SAE;
* GPT-2 unchanged after a backward pass;
* the controllers.

**Reproduction check:** α = β = 0 reproduces Method A TopK seed 42 bit-for-bit.

**After training:** GPT-2's weights and every base and Method A checkpoint were verified unchanged.

**Pre-registered result (`criteria.json`):**
* **TopK, B and C: behavioural success.** OWT KL and GoT statement KL are lower than Method A on every seed by more than 2 sd, and all guards pass.
* **ReLU and JumpReLU: not met.** P1 and P2 pass, but the L0 guard fails: OWT-validation L0 rose 5–11 % above base (KL pushed L0 up faster than the controller corrected it). Their behavioural gains are confounded with using more latents.
* **The truth hypothesis (geographic MLP gap lower than Method A) fails everywhere.** For ReLU and JumpReLU the gap is worse than Method A.
* **Longer-budget flag** (pre-registered, flag only): raised for ReLU and JumpReLU (B, C). Their failure is the L0 guard, so tighter sparsity control would be needed before any longer run.

**Files:**
* `per_sae.csv` (36 SAEs), `summary.csv`, `comparisons.csv`, `criteria.json`.
* `training_monitor.csv` (train-only KL every 50 steps), `training_info.json`, `eval_info.json`.
* Figure: `../figures/e2e_inspired_layer6.png`.

`splice_*` metrics are activation splicing, not steering.

# P1 pilot: improved sparsity control + validation checkpoint selection (2026-11-07). **PILOT (validation only, seed 42)**

Training: `scripts/openwebtext/16_p1_sparsity_control_pilot.py`, committed in `468018c` before the run and run unchanged.
* Scope: layers 0/2/6/10 × ReLU/TopK/JumpReLU × seed 42, target L0 128, 15k steps.
* Checkpoints: `results/openwebtext/convergence/layer_L/{arch}_l0_128_p1_seed42/`.

Evaluation: `scripts/shared/p1_validation_evaluation.py` and `scripts/shared/p1_analysis.py`. Validation splits only; no test row was read.
* Each P1 SAE is compared with the uninterrupted baseline seed 42 (identical batch order), using the same code.
* The 3 baseline seeds give the noise reference.
* The baseline numbers reproduce Phase 0 exactly.

**Files:**
* `runs.csv`, `validation_curves.csv`, `run_config.json`: training outputs.
* `validation_evaluation_long.csv`: all metrics for P1 (selected and final) and baseline seeds 42/1042/2042.
* `baseline_vs_p1.csv`: pre-registered verdicts.
* `baseline_vs_p1_with_dominance.csv`: adds the dominance reading for L0-unmatched pairs (added after the run).
* `baseline_vs_p1_table.csv`: wide table.
* `seed_sd_reference.csv`, `eval_info.json`.
* Figure: `results/openwebtext/figures/p1_vs_baseline_ev_l0.png`.

**Pre-registered rules:**
* L0-matched means |ΔL0| / L0 ≤ 2 %.
* A change is genuine if it is in the better direction and larger than 2 × the baseline seed sd.

**Caveats:**
* Single seed.
* The selection rule picked step 15000 in all 12 runs, because validation EV was still rising.
* TopK P1 is bit-identical to baseline seed 42: none of the changes applies to TopK. This also confirms the pairing.
* The `cities` pairwise metric moves in steps of 1/112.

**Activation-splicing metrics are not steering results.** Persona-shift preservation does not show that steering would work.

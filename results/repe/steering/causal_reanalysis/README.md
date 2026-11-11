# Reanalysis of the RepE causal test (2026-11-10). Status: COMPLETED

Evidence level: **exploratory** (post hoc; none of these analyses were in the preregistration of `scripts/repe/13_causal_random_controls.py`).

## 1. Before implementation

* **Research question:** what does the 2026-11-04 causal test (200 controls, GPT-2 Small layer 6, 90 test statements) actually have the power to show, and is "no SAE target is distinguishable from controls" the right summary of it?
* **Hypotheses:**
  1. the one-sided test hides targets that are *worse* than their controls;
  2. 200 controls cannot support a claim that survives correction over 27 SAE comparisons;
  3. SAE-derived directions recover less of the persona shift than the raw direction they approximate.
* **Why it is needed:** the causal test is the project's only causal evidence. Its null must be reported with its power, and any structure inside it must be found before designing the next experiment (`scripts/repe/15_sae_in_the_loop_causal.py`).
* **Design:** CPU only. Reads the saved outputs of the test run (`../causal_random_controls/causal_controls_200.csv`, `causal_summary.csv`, `per_statement_kl_targets.npz`). No model is run and no test row is re-scored.
* **Metrics** (category: causal):
  * `effect_E`: share of the honest-persona output shift recovered. Higher is better.
  * `p_upper_script13`: script 13's one-sided empirical p. Reproduced from the raw control rows and asserted equal to the saved value.
  * `p_two_sided` = min(1, 2 · min(p_upper, p_lower)).
  * `*_bh`: Benjamini-Hochberg adjusted over the 27 SAE targets. The raw honesty direction is script 13's single primary target and is not adjusted.
  * `E_raw_minus_E_target`: paired bootstrap over statements (2000 resamples, seed 0).
  * Minimum detectable effect (MDE): the smallest E that reaches one-sided p ≤ 0.05 against a control group.
* **Leakage:** none possible; no fitting, no selection, saved numbers only.

## 2. After execution

* **Run:** `python scripts/repe/14_causal_reanalysis.py`, about 2 seconds, CPU.
* **Integrity check:** all 28 saved p-values are reproduced exactly from the control rows (the script stops otherwise). Input hashes are in `power_statement.json`.

### Results

**Power of the test**

| Quantity | Value |
|---|---|
| Smallest p that 200 controls can give | 0.00498 |
| Bonferroni threshold over 27 SAE targets | 0.00185 (cannot be reached with 200 controls) |
| Controls needed for one target to pass Bonferroni over 27 | 539 |
| Targets at the p floor needed to pass BH | 3 |
| Positive control (activation patch) E | 0.234 [0.151, 0.317] |
| Raw honesty direction E | 0.0048 = 2.1 % of the positive control |
| MDE against Gaussian controls | 0.0031 = 1.3 % of the positive control |
| Median 95 % CI half-width of a target | 0.0059 (wider than the MDE) |

The readout can only see effects between about 1.3 % and 2 % of the positive control and above, and the statement-level uncertainty of each target is larger than the control spread. The test is not able to detect a small SAE-specific effect.

**Direction of the effects (27 SAE targets)**

| | Count |
|---|---|
| E < 0 (output moved away from the honest distribution) | 18 of 27 |
| One-sided p ≤ 0.05 (better than controls) | 1 (JumpReLU reconstruction-only reconstruction direction, p = 0.0498) |
| One-sided, BH ≤ 0.05 | 0 (smallest adjusted p = 0.76) |
| Two-sided p ≤ 0.05 | 9, all with E < 0 |
| Two-sided, BH ≤ 0.05 | 0 (all nine sit at adjusted p = 0.0597) |

The nine are the TopK top-1 and top-10 latent directions and the JumpReLU top-1 latent direction, in all three variants each. Every one is below nearly all of its 200 random-latent controls (z from −2.6 to −3.7). With 200 controls the two-sided floor is 0.00995, so nine such targets can reach at best 0.0597 after BH. This is a consistent pattern that the design could not confirm, not evidence of no difference.

**Mean over the three variants of each architecture**

| Target | Architecture | E | Share of raw-direction E | Cosine with raw direction |
|---|---|---|---|---|
| Reconstruction direction | JumpReLU | +0.0027 | 0.56 | 0.87 |
| Reconstruction direction | TopK | +0.0009 | 0.19 | 0.84 |
| Reconstruction direction | ReLU | −0.0010 | −0.21 | 0.76 |
| Top-10 latent direction | ReLU | +0.0011 | 0.23 | 0.54 |
| Top-10 latent direction | JumpReLU | −0.0034 | −0.70 | 0.61 |
| Top-10 latent direction | TopK | −0.0051 | −1.05 | 0.60 |
| Top-1 latent direction | ReLU | −0.0026 | −0.54 | 0.38 |
| Top-1 latent direction | JumpReLU | −0.0056 | −1.16 | 0.50 |
| Top-1 latent direction | TopK | −0.0072 | −1.49 | 0.45 |

**SAE targets against the raw direction.** `E(raw) − E(target)` is positive for all 27 SAE targets, and its paired 95 % bootstrap interval excludes zero for all 27. The smallest shortfall is the JumpReLU reconstruction direction (0.0017 to 0.0025); the largest is the TopK top-1 latent direction (0.012). These 27 intervals are not adjusted for multiplicity and they share the same 90 statements and the same raw-direction estimate.

**Control groups.** Random SAE-latent directions are not more disruptive than Gaussian directions: none of the 18 latent control groups differs from the Gaussian group (smallest Kolmogorov-Smirnov p = 0.088). The negative effects above are therefore specific to the *selected* latents, not a property of decoder directions in general.

Figure: `figures/causal_effects_forest.svg` (and `.png`).

### Criteria

No criteria were pre-registered (exploratory).

### Limitations

* Post hoc analysis of one test run: layer 6, the historical layer-6 SAEs, 90 statements, one strength (0.1 × the mean-difference norm, selected for the raw direction).
* The two-sided reading and the comparison with the raw direction were chosen after seeing the data. They are hypotheses for the next experiment.
* The bootstrap resamples statements only; it does not include the uncertainty of the directions, which were estimated on the train split.
* All effects are tiny in absolute terms (at most 3 % of the positive control in size).

### What was learned and what remains unresolved

* The correct summary of the 2026-11-04 test is: *no SAE target is better than its controls; nine selected-latent directions are worse than their controls at the unadjusted level; the test has no power to confirm either after correction.*
* Why the selected latents push the output away from the honest distribution is not known. Candidates: the selected latents have larger activations than random latents (the controls were not activation-matched), or adding a decoder direction at a fixed norm is the wrong way to intervene on a latent. `scripts/repe/15_sae_in_the_loop_causal.py` tests both (activation-matched controls, latent patching with the SAE in the loop, both signs, 600 controls).

### Files

| File | Content |
|---|---|
| `reanalysis_targets.csv` | one row per target: E, CI, z, one-sided and two-sided p, BH, paired difference to the raw direction |
| `control_group_summary.csv` | one row per control group: spread, MDE, comparison with the Gaussian group |
| `power_statement.json` | power quantities, counts, script and input hashes |
| `figures/causal_effects_forest.svg`, `.png` | every target with its CI against the central 95 % of its controls |

Script: `scripts/repe/14_causal_reanalysis.py`. Tests: `src/tests/test_causal_followup.py`.

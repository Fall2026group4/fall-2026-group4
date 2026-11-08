# M3: causal site sweep in original GPT-2 (RepE-LM persona instructions)

**Status:** PRE-REGISTERED 2026-11-08 (`preregistration.json`); results in §2 once run. **Evidence label:** TRAIN
selection, VALIDATION confirmation; no SAE; test not loaded.

* **Question:** at which layer and token position does GPT-2 carry the effect of swapping "truthful" for
  "deceptive"? A site counts only if both of these hold:
  * copying the honest activation into the dishonest run there moves the output toward the honest output,
    in both directions;
  * the movement is larger than a synonym swap and matched-norm random vectors produce.
* **Design:** 65 cells (13 hook sites × 5 position groups); Holm correction; at most 3 sites; confirmed on validation.
* **Script:** `scripts/repe/23_positive_control_site_sweep.py`. **Module:** `src/evaluation/patching.py`, with tests.

## 2. Results (run 2026-11-08, 96 min GPU; TRAIN 1,668 pairs, VALIDATION 356 pairs; test not loaded)

**Sanity checks, all exactly 0.0 (PASS):**
* all-position patches at layers 0, 6 and 11 reproduce p_H;
* a self-patch gives R = 0;
* layer-0 slot and resid_post-11 final each equal the full effect.

**Baseline effect:** mean KL(p_H‖p_U) = 0.033 nats (per template: T1 0.010, T2 0.035, T3 0.019, T4 0.067).

### 2.1 Where the effect lives
The table gives the share of the full effect restored by patching one position group at one layer: E = R_A / KL(p_H‖p_U), on validation.

| Layer | Persona slot | Instruction after slot | " ASSISTANT:" | Statement prefix | Final token |
|---|---|---|---|---|---|
| 1 | 0.97 | 0.05 | 0.00 | 0.03 | 0.03 |
| 3 | 0.85 | 0.12 | 0.01 | 0.07 | 0.14 |
| 5 | 0.70 | 0.27 | 0.02 | 0.13 | 0.22 |
| 6 | 0.63 | 0.28 | 0.03 | 0.14 | 0.25 |
| 7 | 0.51 | 0.24 | 0.02 | 0.23 | 0.38 |
| 8 | 0.46 | 0.19 | 0.04 | 0.22 | 0.47 |
| 9 | 0.37 | 0.11 | 0.06 | 0.18 | 0.66 |
| 10 | 0.30 | 0.10 | 0.03 | 0.12 | 0.81 |
| 11 | 0.11 | 0.06 | 0.02 | 0.09 | 0.95 |

* **The effect flows from the persona word to the final token.** The handover happens around layers 7–8. The
  instruction words after the slot carry up to 28 % (layer 6); the "ASSISTANT:" tokens carry ≤ 6 %.
* **Both directions agree:** U→H restoration R_B ≈ R_A in every cell.
* **Matched-norm random vectors restore nothing** (q95 ≤ 0.004 nats, often < 0). The random control is easy to beat.
* **The synonym swap is carried by the same sites at about half the size** (R_S ≈ 0.5 R_A). The persona and synonym
  effects travel the same route; the persona one is larger.
* **The wrong-statement patch** equals R_A at instruction positions, as causality requires. At prefix and final
  positions it destroys the output (R_W −0.01 to −6.7 nats), so those sites carry statement-specific content.
* **Validation:** 55 of 65 cells have R_A CI > 0.

### 2.2 Pre-registered selection (train) and confirmation (validation)
* **Passing cells:** 50 of 65 (Holm < 0.05 and R_A > the q95 of 200 random draws, trivial cells excluded).

| Role | Site | E train | R_A validation [CI] | E validation | R_S validation | Confirmed |
|---|---|---|---|---|---|---|
| earliest + peak | `blocks.1.hook_resid_pre`, persona slot | 0.975 | 0.0356 [0.0278, 0.0450] | 0.971 | 0.0178 | **YES** |
| latest | `blocks.11.hook_resid_pre`, final token | 0.960 | 0.0350 [0.0276, 0.0436] | 0.954 | 0.0177 | **YES** |

**By the pre-registered rule, M4 is justified (2 confirmed sites).**

### 2.3 Per template (T1 flagged)

| Site | T1 R_A / R_S (val. CI of R_A − R_S) | T2 | T3 | T4 |
|---|---|---|---|---|
| L1 slot | 0.0096 / 0.0083 (CI low −0.004 on validation) | 0.034 / 0.007 | 0.018 / 0.008 | 0.066 / 0.040 |
| L6 final | 0.0017 / 0.0015 (−0.001) | 0.013 / 0.003 | 0.004 / 0.001 | 0.019 / 0.017 |
| L8 final | 0.0026 / 0.0026 (−0.001) | 0.020 / 0.004 | 0.008 / 0.003 | 0.036 / 0.027 |
| L11 final | 0.0090 / 0.0081 (−0.004) | 0.034 / 0.007 | 0.018 / 0.008 | 0.065 / 0.040 |

* **T1 (truthful/deceptive):** the restoration is real, but **no larger than a synonym swap** (sincere) on validation
  at any site. T1 carries a token-change effect, not a persona-specific one.
* **T2 and T3** show a clear persona-specific excess (R_A ≈ 2.5–5 × R_S).
* **T4's** synonym "truths" itself moves the output a lot (R_S ≈ 0.6 R_A).

### 2.4 Caveats
1. **The two selected sites are near the ends of the computation.** At layer 1 the slot activation is still almost
   the input token; at layer 11 the final-token activation is almost the output. They "carry" the effect almost by
   construction (E ≈ 0.96). The rule (earliest/peak/latest) chose them correctly as written. The scientifically
   richer sites are at layers 6–8, where the effect is split between the slot and the final token (E 0.4–0.6 each);
   they pass too but were not selected. Adding one needs a pre-registration amendment (the user's decision).
2. **The random control is weak:** isotropic vectors of matched norm are nearly orthogonal to the relevant direction.
   The synonym control is the informative one.
3. **The absolute effect is small** (0.033 nats on average).
4. Random draws used a 480-pair train subset (conservative).

### 2.5 Status
COMPLETED. Pre-registered verdict: **sites found and confirmed; M4 justified.** M4 has not been started.

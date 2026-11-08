# M1: RepE-LM, a shortcut-free persona task (construction, activations, gates K1–K7)

**Status:** PRE-REGISTERED (2026-11-08); results in §4 once run. **Evidence label:** exact construction checks +
TRAIN-statement gates; validation descriptive; **test statements not loaded.** Decision log: D23.
Scripts:
* `scripts/repe/20_build_repe_lm_dataset.py`: builds the dataset → `data/repe/repe_lm/`;
* `scripts/repe/21_extract_repe_lm_activations.py`: extracts activations → `data/repe/activations/repe_lm/` (gitignored),
  `k1_activation_invariants.json`, `k7_behaviour_rows.csv`;
* `scripts/repe/22_repe_lm_gates.py`: computes the gates → `gates.json` and tables.

Pre-registration and hashes: `preregistration.json`.

## 1. Why
The old RepE prompts ("an honest" vs "an untruthful") differ by 3 tokens:
* prompt length alone predicts the persona label (AUROC 0.80);
* a layer-0 probe reaches 0.83.

Every old RepE number (all-layer probes, RepE splicing, 6c, R1, R2) is therefore HISTORICAL. M1 builds a task where:
* the two prompts of a pair differ in **exactly one token at the same position**;
* no length, position, template or statement feature can predict the label.

It then checks whether the persona change has a meaning-level signal and a behavioural effect.

## 2. Design
* **Templates.** Each pair is a one-token swap at a fixed slot:
  * T1–T4, persona templates used for fitting: truthful/deceptive, trustworthy/dishonest, sincere/lying, facts/lies.
    Each also has an honest-synonym variant Hp (sincere, reliable, truthful, truths).
  * T5–T6, held-out persona templates, never fitted: true/false, correct/wrong.
  * N1–N2, neutral swaps: tall/short, red/blue.
* **Statements:**
  * existing statement-grouped split: train 417, validation 89 (326 dropped);
  * the existing shortest-unique statement prefix;
  * prompts with persona-lexicon prefixes dropped.
* **Activations:** `blocks.L.hook_resid_pre`, L = 0–11, at the final token and at the slot.

## 3. Gates (pass rules in `preregistration.json`)

| Gate | What it detects | Pass rule |
|---|---|---|
| K1 | Length, position or template confounds (exact) | 100 % of pairs: same length, one differing token at the slot; identical residual before the slot |
| K2 | Any non-persona shortcut (masked bag-of-words; length, position, template, label, statement and final-token single features) | every AUROC = 0.5 ± 1e-9 |
| K3 | Classifier works (persona token visible) | AUROC ≥ 0.99 |
| K4 | How far any one-token change propagates (neutral floor) | reported |
| K5 | Persona *meaning* vs token identity (leave-one-pair-out) | ≥ 0.70 at some layer ≥ 1, CI above the K4 floor |
| K6 | Statement / template leakage | no same-label near-duplicates across train/validation |
| K7 | Persona changes GPT-2's predictions more than a synonym swap | Wilcoxon p < 0.01 and median KL ratio ≥ 1.5 |

## 4. Results (run 2026-11-08; TRAIN gates, validation descriptive; test not loaded)
* **Dataset:** 10,120 prompts = 506 statements (train 417, validation 89) × 20 variants. Statement 326 dropped; 0
  persona-lexicon prefixes. Runtimes: extraction 36 s (GPU); gates ≈ 90 min (CPU).

| Gate | Result | Evidence |
|---|---|---|
| K1 construction | **PASS** | 6,072/6,072 pairs have equal length and differ in exactly one token, at the slot. Residual before the slot is **bit-identical (max diff 0.0) at all 12 layers**; the slot differs at layer 0 (min 0.34) |
| K2 shortcuts | **PASS** | Masked bag-of-words AUROC 0.5000. Single features (length, slot position, final position, prefix length, template, factual label, statement id, final token): all 0.5000. Max deviation 5.6e-17 |
| K3 positive control | **PASS** | Unmasked bag-of-words AUROC 1.000 |
| K4 neutral floor | reported | Neutral grouped-CV AUROC 0.58–0.75 (layers 1–11); cross-pair transfer floor 0.50–0.57 |
| K5 semantic transfer | **PASS (narrow)** | Leave-one-pair-out logistic AUROC 0.53 (L1) rising to 0.70 at L8 (0.701, CI 0.689–0.717) and L11 (0.701, CI 0.691–0.714); floor 0.53–0.54. Layers 7 and 9 reach 0.693 |
| K6 leakage | **PASS** | No exact or same-label near-duplicates. 14 opposite-label counterfactual edits are disclosed. Ungrouped CV (0.92–0.99) far exceeds grouped CV (0.63–0.84), so statement grouping is essential |
| K7 behaviour | **PASS** | Train: median per-statement KL(H‖U) 0.0186 nats vs KL(H‖Hp) 0.0092; median ratio 1.99; U > Hp for 97.4 % of statements; Wilcoxon p = 3e-68. Continuation \|Δ log p\| 0.059 vs 0.037 (p = 2e-49). Validation (descriptive): ratio 2.12, p = 5e-16 |

**Stop/go:** M2 allowed (K1–K3 pass). M3 allowed (K7 pass). The task may use "persona" wording (K5 pass), within
the caveats below.

### 4.1 Caveats found in the results (none changes a gate)
1. **K5 is a narrow pass.**
   * Only 2 layers reach 0.70, and the mass-mean variant peaks at 0.625.
   * Held-out pair T1 (truthful/deceptive) transfers weakly (0.52–0.62); T2–T4 reach 0.71–0.77.
   * The persona *meaning* signal is real but modest.
2. **Much within-template decodability is generic one-token-change information.**
   * Persona grouped-CV AUROC is 0.63–0.84 against a neutral 0.58–0.75 at the same layers.
   * The persona-specific margin is 0.03–0.21.
   * M2 must report neutral pairs alongside persona.
3. **A persona probe applied to neutral swaps is not at 0.5** (0.38–0.49): it partly reads token identity.
4. **The behavioural effect is small and uneven across templates.** Median KL(H‖U) / KL(H‖Hp) by template:
   * T1 0.0042 / 0.0033 (ratio ≈ 1.3);
   * T2 0.0160 / 0.0030;
   * T3 0.0101 / 0.0032;
   * T4 0.0383 / 0.0226.

   Held-out T5/T6: 0.0034 / 0.0050, close to the neutral 0.0022–0.0044.
5. **The direction is sensible.** For true statements the dishonest persona lowers the continuation log-prob by
   0.029 nats/token (validation 0.032); for false statements there is ≈ no change.
6. **The slot position is uninformative for meaning.** Every probe is at 1.0 there because it sees token identity.
   Final-token results are the ones that count.

### 4.2 Status
COMPLETED. All 7 gates pass by their pre-registered rules (K5 narrowly). Test split untouched.

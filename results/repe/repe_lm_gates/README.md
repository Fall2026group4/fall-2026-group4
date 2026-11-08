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

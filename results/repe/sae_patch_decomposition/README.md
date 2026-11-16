# M4: SAE causal-faithfulness decomposition (persona-instruction effect, original vs SAE)

**Status:** PRE-REGISTERED 2026-11-08, with amendment A1 adding a mid-depth site (`preregistration.json`). Results in
§2 once run. **Evidence label:** VALIDATION; train only for the mid-site rule input and the feature ranking; no test.

* **Question (SQ4):** at sites where GPT-2 carries the persona-instruction effect (M3), how much of that effect does
  the SAE's *reconstructed* activation difference reproduce, and how much sits in the reconstruction *error*?
  Do a few SAE features reproduce it?
* **Sites:**
  * layer-1 persona slot (M3);
  * layer-11 final token (M3);
  * a layers 6–8 handover site, chosen by rule A1 on train and confirmed on validation, without SAEs.
* **Primary set:** T2–T4. T1 is reported separately as a control case.
* **Controls:** synonym and neutral one-word pairs, the opposite direction, matched-norm and decoder-shuffle vectors,
  and frequency/|Δz|-matched random feature sets.
* **Not claimed:** that GPT-2 "uses" SAE features.

## 2. Results (run 2026-11-08, 28 min GPU; VALIDATION 356 persona + 178 neutral pairs; test not loaded)

### 2.0 Checks (all PASS)
* identity Δ = Δ_rec + Δ_err: max error 9.5e-7;
* a v = 0 patch reproduces the clean output exactly (KL 0.0);
* the Shapley sum equals T_full (5.6e-17);
* the all-features patch equals T_rec (2.1e-6);
* T_full is identical across the 9 SAEs;
* **T_full equals M3's validation R_A** at all 3 sites (differences ≤ 7e-8).

**Amendment A1 site:** `blocks.8.hook_resid_pre`, final token. Train E = 0.505 (handover point). Validation T2–T4:
T_full 0.022 nats; excess over synonym CI low 0.008. **Confirmed; no SAE used.**

### 2.1 Decomposition, primary set T2–T4 (mean of 3 seeds; seed sd ≤ 0.04 on every share)

| Site | Arch | T_full (nats) | T_rec | T_err | Preserved by reconstruction (T_rec/T_full) | Error alone (T_err/T_full) | \|I\|/T_full [CI high] | Additive reading allowed? | Shapley share φ_rec |
|---|---|---|---|---|---|---|---|---|---|
| L1 persona slot | ReLU | 0.0435 | 0.0408 | 0.0061 | **0.94** | 0.14 | 0.12 [0.14] | yes | 0.90 |
| | TopK | 0.0435 | 0.0405 | 0.0070 | **0.93** | 0.16 | 0.14 [0.15] | yes | 0.89 |
| | JumpReLU | 0.0435 | 0.0421 | 0.0011 | **0.97** | 0.03 | 0.10 [0.12] | yes | 0.97 |
| L8 final (A1) | ReLU | 0.0220 | 0.0137 | 0.0099 | **0.62** | 0.45 | 0.12 [0.14] | yes | 0.59 |
| | TopK | 0.0220 | 0.0138 | 0.0021 | **0.62** | 0.09 | 0.32 [0.41] | **no** | (0.77) |
| | JumpReLU | 0.0220 | 0.0146 | 0.0024 | **0.66** | 0.11 | 0.26 [0.32] | **no** | (0.78) |
| L11 final | ReLU | 0.0430 | 0.0291 | 0.0173 | **0.68** | 0.40 | 0.16 [0.18] | yes | 0.64 |
| | TopK | 0.0430 | 0.0266 | 0.0024 | **0.62** | 0.06 | 0.38 [0.45] | **no** | (0.78) |
| | JumpReLU | 0.0430 | 0.0291 | 0.0048 | **0.68** | 0.11 | 0.28 [0.33] | **no** | (0.78) |

Shapley shares in parentheses are not interpretable as additive shares (pre-registered rule: |I|/T_full CI high ≥ 0.2).

### 2.2 Controls (T2–T4, mean over architectures and seeds)
* **Opposite direction** (H → toward U): reconstruction preservation L1 0.94–0.96, L8 0.62–0.66, L11 0.63–0.68.
  Symmetric with the main direction.
* **Synonym pairs** (H vs Hp): reconstruction preservation L1 0.87–0.95, L8 0.59–0.65, L11 0.53–0.68.
  * The SAE keeps a similar share of a same-meaning one-word effect.
  * The persona-minus-synonym reconstructed effect (T_rec − S_rec) is +0.023 (L1), +0.006–0.007 (L8) and
    +0.016–0.017 (L11) nats, CI low > 0 everywhere. **The reconstruction keeps the persona > synonym difference.**
* **Neutral pairs** (tall/short, red/blue): preservation L1 0.96–0.97, L8 0.25–0.60, L11 0.26–0.45.
  * At L8 the neutral effect is only 0.003 nats, so its ratio is noisy.
* **Matched-norm vectors:**
  * isotropic draws for Δ_rec: mean −0.012 to +0.003, q95 ≤ 0.005;
  * decoder-shuffled Δz: q95 ≤ 0.010;
  * T_rec (0.014–0.042) is far above every q95 (about 4× the largest q95 at L1; the q95s are ≤ 0 at L8 and L11).

  These controls are weak (random directions are mostly harmless or harmful).
* **Linearity:** T(Δ_rec/2) − T_rec/2 = +0.0002 to +0.009. The effect is mildly concave, not linear.

### 2.3 T1 (control case: truthful/deceptive)
* T_full: 0.012 (L1), 0.003 (L8), 0.011 (L11).
* Reconstruction preservation: 0.91–0.97 (L1), 0.04–0.49 (L8), 0.23–0.52 (L11).
* **T_rec − S_rec ≈ 0** (CI includes 0) at every site. As in M3, T1 behaves like a synonym change, not a persona change.

### 2.4 Features (T2–T4)
Ranking is by train attribution; k is chosen on train; position-tracking features (1–16 per SAE) are excluded.

| Site | Arch | k* | 1 feature | 4 | 16 | 64 | 64 as share of T_rec | Random matched sets (mean / q95) | Norm-matched (q95) |
|---|---|---|---|---|---|---|---|---|---|
| L1 slot | ReLU | 64 | 0.0032 | 0.0069 | 0.0126 | 0.0217 | 0.53 | −0.001 / 0.001 | 0.002 |
| | TopK | 64 | 0.0029 | 0.0065 | 0.0145 | 0.0255 | 0.63 | −0.002 / 0.000 | 0.002 |
| | JumpReLU | 64 | 0.0017 | 0.0050 | 0.0123 | 0.0260 | 0.62 | −0.001 / 0.001 | 0.002 |
| L8 final | ReLU | 64 | 0.0008 | 0.0021 | 0.0045 | 0.0074 | 0.54 | 0.001 / 0.001 | 0.000 |
| | TopK | 64 | 0.0012 | 0.0032 | 0.0064 | 0.0100 | 0.73 | 0.001 / 0.001 | −0.000 |
| | JumpReLU | 64 | 0.0011 | 0.0034 | 0.0069 | 0.0105 | 0.72 | 0.001 / 0.001 | −0.001 |
| L11 final | ReLU | 64 | 0.0011 | 0.0034 | 0.0080 | 0.0138 | 0.47 | 0.003 / 0.003 | −0.001 |
| | TopK | 64 | 0.0012 | 0.0050 | 0.0120 | 0.0197 | 0.74 | 0.003 / 0.003 | −0.002 |
| | JumpReLU | 64 | 0.0023 | 0.0058 | 0.0124 | 0.0198 | 0.68 | 0.003 / 0.003 | −0.002 |

* **k\* = 64 (the top of the grid) for all 27 SAE × site combinations.** No set of ≤ 16 features reached half of
  T_rec on train.
* **The selected features beat random matched sets** in 27/27 cases (p = 0.005, the minimum with 200 sets).
* **The effect is spread over many features:**
  * one feature carries 2–10 % of T_rec;
  * 64 features carry 47–74 %.
* **Attribution ranking is only slightly better than ranking by |Δz|** (64 by |Δz|: 0.007–0.026).
* **T1:** 64 features give ≤ 0.004 nats, often not above random sets (L1 p 0.005–0.90).

### 2.5 Pre-registered hypotheses (T2–T4, BH over 27, ≥ 2/3 seeds)
H4a (sufficiency vs controls), H4b (φ_rec > φ_err) and H4c (features vs random sets): **PASS for all 9
architecture × site combinations (3/3 seeds each).**

Reading limits:
* H4b is only an additive statement where |I| is small: L1 all architectures, and L8/L11 ReLU.
* H4c passes with 64 features. The pre-written reading "information in the reconstruction but not concentrated in
  few features" applies in substance.

### 2.6 Reading (pre-written categories)
* **L1 persona slot:** "the SAE reconstruction is behaviourally sufficient at this site" (93–97 %). This site is close
  to the input token itself.
* **L8 and L11 final token:** the reconstruction alone restores about two-thirds (62–68 %).
  * **ReLU:** a large independent part sits in the error (40–45 %).
  * **TopK and JumpReLU:** the error alone does little (6–11 %), but about a third of the effect appears only when the
    reconstructed and error parts are patched together ("interaction-dominated; only bounds are reported").
* **Features:** SAE-defined directions are causally sufficient in aggregate when inserted, but the effect is
  distributed across ≥ 64 features. This does not show that GPT-2 uses these features.

### 2.7 Limitations
* The effects are tiny in absolute terms (0.02–0.045 nats).
* Validation only.
* Three sites; one prompt family.
* The matched-norm controls are weak.
* Neutral ratios are noisy where the neutral effect is small.
* The k grid stops at 64.
* Feature ranking used T2–T4 train only.

### 2.8 Status
COMPLETED, VALIDATION. All checks pass. M5 not started.

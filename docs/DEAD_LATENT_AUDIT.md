# Dead-Latent Audit

**Status:** completed on 2026-11-15. This is the instructor-requested diagnostic audit, not a new headline result.
Evidence label: **validation finding** (descriptive). No model was trained, fine-tuned or run; every number comes from
stored evaluation artifacts.

**Outputs** (all in `results/shared/dead_latent_audit/`):
* `dead_latents_by_sae.csv`: one row per evaluated checkpoint (150 rows):
  * 108 base SAEs;
  * 36 trained R2 SAEs;
  * 3 R2 frozen-random-decoder baselines;
  * 3 public res-jb reference rows.
* `summary_by_family_layer.csv`: one row per phase × family × layer;
* `checks.txt`: check log;
* `build_dead_latent_audit.py`: the script that builds all three. It reads only CSVs and refuses to overwrite its
  outputs.

## Instructor request

This audit answers the instructor's 2026-11-10 request to report the dead-latent fraction of every old (base) and R2
GPT-2 SAE.

Gemma Scope is **not** included, because no repository document says the "old and R2" request covers it. The public
`gpt2-small-res-jb` SAE appears only as a separate, clearly labelled reference category, because the R2 ladder
evaluated it in the same pass.

## Definition

A latent is **dead** if it never activates on any token of the evaluation corpus:
* "Activates" means z > 0.
* The corpus is the layer's OWT validation cache: GPT-2 `blocks.L.hook_resid_pre`, BOS excluded, 100 sequences,
  12,700 tokens.

Then:
* `dead_fraction = dead_latents / width`;
* `alive_latents = width − dead_latents`.

**Where the definition comes from.** This is the project's existing evaluation-time definition, not a new one:
* `src/evaluation/faithfulness_metrics.py`: `reconstruct()` returns `active_per_latent = any(z > 0)` over all
  tokens, and `dead_fraction()` is documented as "Fraction of latents that never fire on the evaluated set".
* **Base SAEs:** computed by `scripts/shared/spliced_reconstruction_evaluation.py` (line 317,
  `fm.dead_fraction(active)`) on `load_split(..., 'validation', L)`.
* **R2 SAEs:** computed by `scripts/shared/sae_ladder_evaluation.py` (line 203,
  `1 − act.float().mean()`, where `act` comes from `fm.reconstruct`) on the same `load_split(..., 'validation', L)`
  cache.

So base and R2 use **one common definition on one common corpus**, and the two can be put in one table. The stored
fractions convert to whole latent counts: `dead_fraction × width` is an integer to within 1e-2 for every row.

**A different definition, not used here.** R2 training also tracks a training-time notion of "dead":
* For the AuxK loss, a latent is dead if it has not fired in the last 5M **training** tokens (`DEAD_TOKENS` in
  `scripts/openwebtext/17_streaming_sae_training.py`).
* `training_monitor.csv` also records a monitor-time dead fraction on 800 chunks of OWT shard 79.

These are different rules on different data. They are not mixed into the audit. The end-of-training monitor value is
kept only as a note on each R2 row: a mean of 0.008 % and a maximum of 0.09 % across the 39 R2 checkpoints (36
trained SAEs and 3 frozen-random-decoder baselines).

**Scope of the definition.** Deadness is defined only on this 12,700-token OWT validation sample. A latent that is
silent on this sample may still fire on other text.

## Coverage

| Group | Expected | Audited | Missing |
|---|---|---|---|
| Base SAEs (12 layers × ReLU/TopK/JumpReLU × seeds 42/1042/2042, width 3,072) | 108 | 108 | none |
| R2 trained SAEs (layers 1/8/11 × TopK-12288, BatchTopK-12288, ReLU-12288, TopK-3072 × 3 seeds) | 36 | 36 | none |
| R2 frozen-random-decoder baselines (TopK-12288, decoder frozen at random initialisation, encoder trained; seed 42, layers 1/8/11). Not trained SAEs; audited separately | 3 | 3 | none |
| Public reference `gpt2-small-res-jb` (width 24,576; layers 1/8/11) | not requested | 3 | n/a |

* The 36 trained R2 SAEs and the 3 frozen-random-decoder baselines together make up the 39 R2 runs listed in
  `results/openwebtext/r2_scaled_saes/README.md`.
* The baselines are kept out of every trained-family average.
* **Out of scope:**
  * R2 intermediate checkpoints (10M/25M/50M tokens): the scaling study X0 is pre-registered with results "Pending",
    so no stored values exist.
  * R4 fine-tunes, P1 pilot runs, method-A fine-tunes and Gemma Scope.

## Results

All values are on OWT validation (12,700 tokens), given as % dead. "sd" is the across-seed standard deviation.

### Old (base) versus R2, all layers pooled

| Phase | Family | Width | Runs | Mean | sd | Min | Max | Mean alive |
|---|---|---|---|---|---|---|---|---|
| base | ReLU | 3,072 | 36 | 1.95 | 1.71 | 0.20 | 5.37 | 3,012 |
| base | TopK | 3,072 | 36 | 23.19 | 12.99 | 6.74 | 44.73 | 2,360 |
| base | JumpReLU | 3,072 | 36 | 34.56 | 27.81 | 4.17 | 95.67 | 2,010 |
| base | JumpReLU, L0-matched only | 3,072 | 35 | 32.81 | 26.14 | 4.17 | 85.12 | 2,064 |
| R2 | TopK-12288 | 12,288 | 9 | 0.09 | 0.05 | 0.01 | 0.16 | 12,277 |
| R2 | BatchTopK-12288 | 12,288 | 9 | 0.17 | 0.14 | 0.00 | 0.36 | 12,267 |
| R2 | ReLU-12288 | 12,288 | 9 | 0.04 | 0.04 | 0.01 | 0.11 | 12,283 |
| R2 | TopK-3072 (width ablation) | 3,072 | 9 | 0.00 | 0.00 | 0.00 | 0.00 | 3,072 |
| R2 frozen-random-decoder baseline (not a trained SAE) | TopK-12288, frozen decoder | 12,288 | 3 | 0.003 | | 0.00 | 0.01 | 12,288 |
| public reference | res-jb | 24,576 | 3 | 15.99 | 25.56 | 1.11 | 45.51 | 20,645 |

Base pooled over all 108 runs: mean 19.90 %, range 0.20–95.67 %.

### Base SAEs by layer (mean ± sd over 3 seeds, % dead)

| Layer | ReLU | TopK | JumpReLU |
|---|---|---|---|
| 0 | 1.01 ± 0.18 | 34.32 ± 0.44 | 88.38 ± 6.33 (L0-matched seeds: 84.73) |
| 1 | 4.64 ± 0.63 | 44.30 ± 0.70 | 67.99 ± 0.69 |
| 2 | 4.30 ± 0.54 | 41.51 ± 0.71 | 65.84 ± 0.36 |
| 3 | 2.59 ± 0.16 | 32.70 ± 0.41 | 54.77 ± 0.55 |
| 4 | 1.64 ± 0.31 | 23.89 ± 0.51 | 40.32 ± 1.36 |
| 5 | 1.52 ± 0.12 | 19.21 ± 0.44 | 31.62 ± 0.61 |
| 6 | 0.86 ± 0.15 | 14.83 ± 0.87 | 20.19 ± 0.40 |
| 7 | 0.56 ± 0.07 | 12.09 ± 0.20 | 13.28 ± 0.67 |
| 8 | 0.30 ± 0.08 | 8.39 ± 0.27 | 7.34 ± 0.57 |
| 9 | 0.31 ± 0.13 | 7.18 ± 0.47 | 4.79 ± 0.03 |
| 10 | 0.65 ± 0.17 | 8.44 ± 0.36 | 4.26 ± 0.12 |
| 11 | 4.96 ± 0.24 | 31.37 ± 0.61 | 15.92 ± 0.59 |

### R2 layers: old versus new (dead latents per seed 42/1042/2042; mean % dead)

| Layer | Old ReLU-3072 | Old TopK-3072 | Old JumpReLU-3072 | R2 TopK-12288 | R2 BatchTopK-12288 | R2 ReLU-12288 | R2 TopK-3072 |
|---|---|---|---|---|---|---|---|
| 1 | 165/131/132 (4.64) | 1374/1373/1336 (44.30) | 2095/2106/2065 (67.99) | 5/3/1 (0.024) | 1/3/0 (0.011) | 13/13/11 (0.100) | 0/0/0 (0) |
| 8 | 12/7/9 (0.30) | 263/262/248 (8.39) | 219/245/212 (7.34) | 12/12/14 (0.103) | 24/24/16 (0.174) | 3/1/1 (0.014) | 0/0/0 (0) |
| 11 | 160/152/145 (4.96) | 976/973/942 (31.37) | 499/500/468 (15.92) | 15/20/13 (0.130) | 44/44/34 (0.331) | 3/2/2 (0.019) | 0/0/0 (0) |

Additional reference rows:
* **R2 frozen-random-decoder baseline (seed 42; not a trained SAE):** 1, 0 and 0 dead latents at layers 1, 8
  and 11.
* **res-jb:** 11,184 (45.51 %), 334 (1.36 %) and 274 (1.11 %) dead latents at layers 1, 8 and 11. Its L0 is
  15, 68 and 61, so it is not matched to 128.

### Highlights

* **Worst runs are all base JumpReLU at layers 0 to 3:**
  * layer 0 seed 42 has 95.67 % dead (2,939 of 3,072). This is the one L0-unmatched base run, at L0 74.6.
  * layer 0 seeds 2042 and 1042 have 85.12 % and 84.34 % dead;
  * layers 1 and 2 are at 65–69 %.
* **Best runs:** 10 trained SAEs have zero dead latents: all nine R2 TopK-3072 runs and R2 BatchTopK layer 1
  seed 2042.
* **Layer-0 JumpReLU is unusually dead.** It is the highest-dead cell in the study, even without the unmatched seed
  (84.7 %).
* **Base dead fraction depends on depth:**
  * it is highest at layers 0–3 and lowest at layers 8–10;
  * it rises again at layer 11 for all three architectures.
* **Base architectures differ visibly:**
  * ReLU stays at or below 5.4 % per run;
  * TopK ranges 6.7–44.7 %;
  * JumpReLU ranges 4.2–95.7 %.
  * JumpReLU is the most dead at layers 0–7. TopK is the most dead at layers 8–11.
* **R2 materially reduces dead latents:**
  * every R2 trained SAE has at most 0.36 % dead (44 of 12,288);
  * the corresponding old SAEs at the same layer have 0.30–67.99 %.
* **Pre-registered R2 comparisons on `owt_dead_fraction`** (`results/openwebtext/r2_scaled_saes/comparisons.csv`;
  validation; rule = 9/9 seed pairs **and** a Welch 95 % CI that excludes 0):

  | Comparison | Layer 1 | Layer 8 | Layer 11 |
  |---|---|---|---|
  | C1 (TopK-12288 vs old TopK) | IMPROVEMENT | IMPROVEMENT | IMPROVEMENT |
  | C2 (ReLU-12288 vs old ReLU) | IMPROVEMENT | IMPROVEMENT | IMPROVEMENT |
  | C3 (BatchTopK-12288 vs old JumpReLU) | IMPROVEMENT | IMPROVEMENT | IMPROVEMENT |
  | A1 (data and recipe at equal width) | IMPROVEMENT | IMPROVEMENT | IMPROVEMENT |
  | A2 (width) | NO CLEAR CHANGE | REGRESSION | REGRESSION |

  * The A2 regressions are tiny in absolute terms: 0 % versus 0.10–0.13 %.
  * The same-width R2 TopK-3072 arm also has zero dead latents, so the reduction cannot be explained by increased
    width alone. Data, training duration and recipe changed together, so this audit does not isolate which factor
    caused the reduction.
  * These classes are pre-existing and are not corrected for multiplicity.
* **R2 architectures differ only slightly:**
  * BatchTopK has the most dead latents at layers 8 and 11 (0.17 %, 0.33 %);
  * ReLU has the most at layer 1 (0.10 %).
  * No test of these differences exists, so none is claimed.
* **Corroboration on the test split.** M5 *reported* (but did not decide on) dead fraction on the OWT TEST cache
  (`results/shared/m5_confirmatory_test/m5_rows.csv`, 12,700 tokens). Its values at layers 1/8/11 are close to the
  validation values, for example:
  * old JumpReLU: 68.3 %, 7.7 %, 16.2 %;
  * R2 TopK: 0.03 %, 0.19 %, 0.20 %.

  They are not used in the audit tables.

## Checks (`checks.txt`)

**Passed:**
* alive + dead = width for every row;
* dead_fraction = dead / width;
* the recomputed fraction equals the stored fraction (|diff| ≤ 1e-6);
* no duplicate run IDs or checkpoint hashes;
* all rows use 12,700 tokens;
* all 108 base combinations are present;
* all 39 R2 combinations are present: 36 trained SAEs and 3 frozen-random-decoder baselines;
* the alive-latent ranges in the appendix are reproduced for ReLU, TopK and JumpReLU (after the correction below);
* the R2 ladder's re-evaluation of the old SAEs (layers 1/8/11) uses the same checkpoint hashes and gives the same
  dead fractions as the base table.

**Comparison with the manuscript.** The appendix (`research_paper/Latex/sections/12_appendix.tex`, lines 24–25) and
`results/shared/m0_within_layer_reanalysis/README.md` state the alive-latent ranges "ReLU 2,920–3,063, TopK
1,711–2,851, JumpReLU 469–2,947".
* These are ranges of **per-layer seed means over L0-matched runs** (M0 `architecture_by_layer.csv`,
  `live_latents_mean`). They are not per-run ranges.
* ReLU and TopK reproduce exactly.
* JumpReLU's lower bound, 469 (the layer-0 mean without the unmatched seed), reproduces.
* **JumpReLU's upper bound did not reproduce as first written.** The largest JumpReLU layer mean is **2,941**
  (layer 10), and the largest single JumpReLU run is 2,944. **2,947** appears nowhere in the base results.
* **Transcription correction (2026-11-15):** 469–2,947 was corrected to **469–2,941** in three places:
  * `research_paper/Latex/sections/12_appendix.tex`;
  * `results/shared/m0_within_layer_reanalysis/README.md`;
  * the repository `README.md`.

  This corrects a transcribed number; it is not a new experimental result. `architecture_by_layer.csv` already held
  2,941.0 and is unchanged. The audit script now parses the appendix sentence directly, and all three appendix
  cross-checks pass.
* Per-run alive ranges, for reference:
  * ReLU 2,907–3,066;
  * TopK 1,698–2,865;
  * JumpReLU 133–2,944 (457–2,944 without the unmatched run).

**Other documented numbers:**
* `EXPERIMENT_RESULTS_SUMMARY.md` §8 ("JumpReLU 40–85 % at layers 0–4; TopK 7–44 %; ReLU ≤ 5 %") agrees with the
  layer means above, with the JumpReLU layer 0 value taken over the L0-matched seeds. Individual ReLU runs reach
  5.37 %.
* `docs/2026-11-12_gpt2_audit_and_improvement_plan.md` gives R2 as "≈ 0–0.3 %". The measured maximum is 0.36 %.

## Interpretation (descriptive)

* The old SAEs, trained on about 101,600 unique tokens for 38 passes, leave many latents unused. JumpReLU and TopK
  are affected most, especially at early layers.
* Their effective dictionary is therefore smaller than the nominal 3,072. For example, layer-1 JumpReLU uses about
  983 latents and layer-1 TopK about 1,711.
* The R2 recipe reduces dead latents to almost zero for every architecture and layer. That recipe is 100M fresh
  tokens, AuxK for TopK and BatchTopK, encoder = decoderᵀ initialisation, and input scaling.
* **Comparability caveats:**
  * Width differs between old (3,072) and most R2 SAEs (12,288). Dead fraction is a share, but a 12,288-wide SAE has
    four times as many latents that must each fire at least once on the same 12,700 tokens.
  * The same-width R2 TopK-3072 arm also has zero dead latents, so the reduction cannot be explained by increased
    width alone.
  * Data, training duration and recipe changed together (R2 README A1/A2), so this audit does not isolate which
    factor caused the reduction.
  * Deadness is defined only on the 12,700-token OWT validation sample. A latent silent on this sample may fire
    elsewhere. The count is a property of the SAE *and* the sample.
* **Fewer dead latents does not by itself imply better information, behavioural or causal faithfulness.** This
  audit makes no such claim. The R2 README reports those levels
  separately, and some of them did not improve. For example, the persona SAE − PCA gap shows no clear change, and
  regresses for ReLU.

## Provenance

**Primary data:**
* `results/openwebtext/spliced_evaluation/per_run.csv`: base SAEs, rows with `intervention == sae_reconstruction`,
  column `owt_val_recon__dead_fraction`.
* `results/openwebtext/r2_scaled_saes/ladder/ladder_layer{1,8,11}_r2_resjb.csv`: R2, frozen-decoder and res-jb
  rows, column `owt_dead_fraction`.

**Cross-checks:**
* `results/openwebtext/r2_scaled_saes/ladder/ladder_layer{1,8,11}_old.csv`;
* `results/shared/m0_within_layer_reanalysis/architecture_by_layer.csv`;
* `results/openwebtext/r2_scaled_saes/comparisons.csv` (pre-registered classes);
* `results/openwebtext/r2_scaled_saes/training_monitor.csv` (training-time monitor, notes only);
* `results/shared/m5_confirmatory_test/m5_rows.csv` (test-split corroboration, not tabulated).

**Definition code:**
* `src/evaluation/faithfulness_metrics.py` (`reconstruct`, `dead_fraction`);
* `scripts/shared/spliced_reconstruction_evaluation.py`;
* `scripts/shared/sae_ladder_evaluation.py`;
* `scripts/openwebtext/17_streaming_sae_training.py` (training-time AuxK rule, not used).

**Context:**
* `results/openwebtext/r2_scaled_saes/README.md`;
* `results/openwebtext/r2_scaled_saes/scaling/README.md` (X0: results pending);
* `research_paper/Latex/sections/12_appendix.tex`.

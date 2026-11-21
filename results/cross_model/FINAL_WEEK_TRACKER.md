# Final-week technical tracker (started 2026-11-21)

Plan: `docs/2026-11-21_final_week_technical_audit.md` (approved 2026-11-21). Evidence labels: VAL = validation,
TEST = confirmed test, EXPL = exploratory. GPU: A10G 23 GB; at most 2 GPU jobs at once, never 2 Gemma jobs.

## Completed
| ID | Model / arch | Layer | Data | Result | Files |
|---|---|---|---|---|---|
| G-HOOK | GPT-2 TL vs HF | 0,3,6,9,11 | short text | TL `resid_pre` = HF `hidden_states[L]` minus its per-token mean (rel. diff ≤ 3e-7); logits identical | `xb_matched_splicing/preregistration.json` |
| XB-samples | both | — | Pile (M6 512 + 64), OWT-XM (shard 00077, new 512 + 64) | Pile texts and Gemma own-1024 ids match M6 hashes 576/576. Shared span: GPT-2 512 / Gemma 471 tokens per doc (Pile), 703 / 685 (OWT) | `xb_matched_splicing/samples/` |
| XB-smoke | both | 1,8 / 5,12 | Pile, 8 docs | identity and repeat ΔCE = 0 exactly. Gemma SAE ΔCE vs M6's own per-doc rows: mean diff 0.0004, max 0.008 nats | `xb_matched_splicing/smoke/` |
| RepE-LM K1 (tokenizer) | Gemma tokenizer | — | 8,096 pairs | 100 % one-token swaps | audit §0 |
| XT-G0 extraction + gates | Gemma-2-2B | 0–25 | GoT (4 sets + fresh DEV), RepE-LM, OWT 800 / 100 docs | K1 PASS (6,072 pairs bit-identical before the slot); K7 persona / neutral KL median ratio 5.1, Wilcoxon p 7e-77 (GPT-2: 1.99); OWT tokens 101,600 / 12,700 (= GPT-2) | `results/gemma/xtg_task_baselines/gates/` |
| TI-1 gate | GPT-2 original | 1, 8, 11 | cities folds | eligible: L8 (0.78–0.83), L11 (0.74–0.77); **L1 NOT eligible (0.51) → L1_topk not run** | `results/geometry_of_truth/tsp_generalisation/gate_eligibility.json` |
| TI-1 R0 | TopK L8 | 8 | 30 steps | run_one_general == frozen run_one bit-for-bit | `.../gate_r0_repro.json` |
| **TI-1 result** | R2 L11 TopK, L8 BatchTopK, L8 ReLU | 8, 11 | cities, 3 seeds × 3 folds, 81 fine-tunes | **GENERALISES**: tsp − ctrl refit gap −0.042 / −0.015 / −0.026, all p_holm < 1e-3, 3/3 seeds; cost guard met (NMSE ≤ +2.5 %, L0 ±0.2 %) | `results/geometry_of_truth/tsp_generalisation/analysis/` (§56, D48) |
| XB A2 check | Gemma | 5, 12 | 8 docs | layer-major == document-major bit-for-bit | `xb_matched_splicing/smoke/` |
| XT-G smoke | GPT-2 res-jb | 8 | GoT, RepE-LM | pipeline works (EXPL; full-run numbers supersede) | `xtg_truth_preservation/smoke/` |

## Running
| ID | What | Started | ETA |
|---|---|---|---|
| XB gemma queue | pile own (triplets) → pile shared (all 26) → owt shared (all 26) → owt own (triplets) | 09:30 | ~14:00–15:00 |
| XB gpt2 queue (layer-major, A3) | owt shared (resumed at 96/512) → pile shared | 09:40 | ~15:00 |
| XT-G Gemma smoke | layer 12, timing | 09:45 | ~10:15 |

## Pending (approved scope, in order)
XT-G (Gemma all layers; GPT-2 triplets) + DX → XB analysis → batch-2 ladder (Gemma 16k/65k/32k at triplets) →
TI-1 → TI-2 → TI-3 → TI-4 → XC-1 → XC-2 → XC-3 → XC-4/SIL → TI-G → X0 → E1 → PCA-CM-T → XG-IH / XJ (feasibility) → XF.

## Failed / interrupted (kept, resumed)
| ID | What happened | Action |
|---|---|---|
| XB gpt2 pile shared | CUDA OOM at start-up: a concurrent extraction smoke test exhausted GPU memory | resume later (amendment A1 resume rule) |
| XB gemma pile own | stopped deliberately at 32/512 docs to free memory for extraction | resumed (A1) |
| XT-G Gemma smoke #1 | CUDA OOM (GPU full: GPT-2 XB had 159 SAEs resident) | GPT-2 XB made layer-major (A3), smoke re-run |
| XB gpt2 owt (doc-major) | stopped at 96/512 docs for A3 | resumed, 1,152 cells kept |

## Inconclusive
(none yet)

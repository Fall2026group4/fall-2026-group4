# Limitations and future work: GPT-2 Small SAE faithfulness

As of 2026-11-12. The experiments in §2 are **proposals only**. None has been approved or run. Each needs a
pre-registration (`docs/templates/EXPERIMENT_TEMPLATE.md`) and explicit approval. Full designs, success criteria and
cost estimates are in [`2026-11-12_gpt2_audit_and_improvement_plan.md`](2026-11-12_gpt2_audit_and_improvement_plan.md)
§8.

## 1. Limitations

| Area | Limitation | Consequence |
|---|---|---|
| Reconstruction | The R2 SAE error is still 1.1–2.2× more damaging than same-norm noise (R4 at L8: 0.98) | Remaining error is structured ("dark matter"), not random |
| Attribution | R2 changed data, width and recipe together; the token count alone was never isolated (the 10M/25M/50M checkpoints are unevaluated) | "More data helped" cannot be claimed on its own |
| Sparsity | Only L0 = 128. Task-prompt L0 is unmatched: on GoT, threshold architectures fire 63–111 latents vs TopK's 128; on RepE-LM, 134–216 | Architecture rankings on information metrics are confounded |
| JumpReLU | The old JumpReLU uses a sigmoid surrogate; the modern STE JumpReLU failed to reach the target L0 and was abandoned; the kernel-STE checkpoints have no training script | No valid modern threshold-architecture comparison |
| GoT power | Seed sd 0.004–0.024 is about the size of the differences; n = 452 test statements; the original AUROC is only 0.84 (ceiling); a random frozen decoder ties trained SAEs on SAE − PCA | Quantified (`results/shared/power_sensitivity/`): one SAE's gap MDE 0.030–0.048 AUROC; between-family comparisons cannot reliably detect changes smaller than about 0.05–0.09 AUROC; observed differences ≤ 0.025 |
| GoT scope | M5 pools cities + neg_cities (2,992 statements; test 452) and reports only the pooled result; no per-polarity or cross-polarity (train cities → test neg_cities) result exists under R2; the numerical sets are shortcut-confounded | Generalisation across polarity and truth categories untested |
| RepE | **RepE-LM test split built and protocol frozen (2026-11-13), not evaluated.** Every persona, representation and causal result is VALIDATION | Nothing about honesty is confirmed yet |
| Contrast distortion | ReLU-family SAEs shrink the persona contrast (‖Δ̂‖/‖Δ‖ 0.62–0.71); TopK-family SAEs inflate and misalign it (1.09–1.46; cos 0.37–0.52) | Directional readouts are distorted in architecture-specific ways |
| Behaviour vs causal | R4 lowers ΔCE 16–17 % (TEST) but lowers T_rec/T_full by 0.024 at L11 (VAL) | Behavioural fine-tuning is not a free improvement |
| Causal power | T_full is only 0.022 nats at L8; TopK per-SAE CIs are ±0.06–0.09 | Per-SAE MDE of T_rec/T_full is 0.054–0.107 at L8; the R2 L8 gain (+0.101) is borderline |
| PCA matching | Error-matched PCA uses 298–519 dimensions vs the SAEs' 128 active latents; at k = 128 the GoT verdict partly reverses (PCA-CM, VAL); persona not re-tested; no test confirmation | SAE-vs-PCA claims must name the constraint; "better" is per-cell, not family-wise |
| Statistics | 3 seeds; no multiplicity correction across 136 M5 rows; Welch CIs reflect seed variance only | A per-cell "9/9" requirement has ≈ 0.05 chance probability |
| Test exposure | M5 test rows had been scored earlier for the old reference SAEs; the M5 pre-registration was committed together with its results | Labelled "previously exposed"; future freezes are committed first (D31) |
| Scope | One model, three R2 layers (1, 8, 11), one data source, no external replication | Generalisation to Gemma not established |
| Backups | About 10 GB of R2/R4 checkpoints and the activation caches exist only on the training machine | Not protected by Git |

## 2. Proposed future work (not run)

| ID | Proposal | Addresses | Cost estimate | Prerequisite |
|---|---|---|---|---|
| **H1** | Build the RepE-LM held-out test split (held-out statements, K1–K7 gates, near-duplicate check) | RepE validation-only | minutes | none |
| **T1** | GoT measurement upgrade: minimum detectable effect from the existing bootstrap arrays; paired per-statement SAE comparisons; inference-time task-L0-matched TopK; cities → neg_cities transfer | GoT power, task L0 | < 1 h GPU | none |
| **X0** | Data-scaling evaluation of the existing R2 10M/25M/50M checkpoints (disclose: mid-schedule) | token count vs recipe | 1–2 h GPU | none |
| **X1** | Inference-time optimisation on a fixed decoder: (a) global scale, (b) non-negative least squares on the encoder's support, (c) pursuit at equal L0. Label any gain as dictionary + optimiser | encoder vs dictionary error; shrinkage | 2–4 h GPU | timing smoke test |
| **X2** | ReLU shrinkage correction: per-latent scale or decoder-only fine-tune, encoder frozen | ReLU shrinkage | ≈ 1.2 h GPU | only if X1(a) fits a scale ≥ 1.05 |
| **X3** | Modern JumpReLU pilot: fixed λ with warm-up; the Anthropic 2025 loss; or initialisation from BatchTopK | missing modern JumpReLU | 1 h pilot; 2–3 h full | go/no-go on L0 stability |
| **H2** | Contrast-consistent fine-tune at fixed L0 (R2 TopK; raw-Δ loss, no probe teacher; held-out templates T5/T6; reconstruction-only control arm) | contrast distortion | ≈ 2 h GPU | H1 |
| **I1** | KL × contrast 2 × 2 factorial at fixed L0 | the R4 causal regression | ≈ 4 h GPU | H2 without guard failures |
| M6 | One test pass after a protocol freeze committed before the run | confirmation | 1–2 h GPU | the items above |

**Not recommended:**
* LoRA on GPT-2 (it changes the model being explained);
* more probe-teacher truth losses (three failures, one circular);
* Gated or Matryoshka retraining;
* width/L0 frontiers before the 2026-12-08 deadline.

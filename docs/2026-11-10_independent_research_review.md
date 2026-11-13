# Independent research review and improvement strategy (2026-11-10)

**Status:** recommendation only. No experiment in this document has been run or approved. Nothing here changes an
existing result, checkpoint or pre-registration.

**Sources:**
* the repository (all result files cited are committed);
* the literature listed in §8, with every link checked on 2026-11-10;
* two throughput measurements made for planning, not experiments:
  * GPT-2 Small to layer 8: about 63k tokens/s;
  * Gemma-2-2B full forward, bf16: about 7.7k tokens/s, 7.3 GB peak, on the A10G.

---

## 1. Summary

1. **Our SAEs are far smaller and far less trained than any published reference:**
   * ours: 3,072 latents (4× expansion), trained for 15k steps on **101,600 unique tokens (about 38 passes)**;
   * `gpt2-small-res-jb`: 24,576 latents, 300M tokens;
   * Gemma Scope: 16k latents on about 4B tokens.

   Every faithfulness failure we report could therefore be a property of under-trained SAEs, not of SAEs. This is
   the most important weakness and the first thing a reviewer will raise (Review #8 critique 2).
2. **Our strongest assets are the evaluation ladder and the negative results:**
   * pre-registered reconstruction → information → behaviour → causal evaluation;
   * the noise and PCA references;
   * the equal-length RepE-LM task;
   * the M4/E3 decomposition.

   The literature has moved the same way: SAEs often fail to beat linear probes or even random baselines on
   downstream tasks. Our ladder is more careful than most published SAE evaluations.
3. **Recommended direction:** stop fine-tuning the weak SAEs, and instead **apply the ladder to strong SAEs**. That
   means:
   * (a) the public GPT-2 SAEs (E2);
   * (b) our own SAEs retrained on 100M+ streamed tokens with a modern recipe;
   * (c) Gemma Scope SAEs on Gemma-2-2B.

   Then test the one improvement method the literature supports at fixed sparsity: a short KL fine-tune with an
   architecture whose L0 cannot drift.

   This turns the project's claim from "our small SAEs are unfaithful" into "SAE unfaithfulness persists (or does not)
   as SAEs get better, across two models". Either answer is a substantive, defensible result.
4. **Realistic measurable improvements:**
   * reconstruction and spliced ΔCE, which very likely improve with data and width (not guaranteed);
   * a clean answer on whether the persona and causal losses survive in strong SAEs;
   * Gemma results, which the project currently lacks entirely for SAEs;
   * held-out confirmation (M5).

   I do **not** expect any method to make SAEs "preserve truth information" in the re-trained-reader sense. The
   literature gives no reason to expect it, and we should not promise it.

---

## 2. What the project has achieved (verified from committed files)

| Level | Result | Evidence | Source |
|---|---|---|---|
| Reconstruction | Matched-L0 (128) SAEs, 3 architectures × 12 layers × 3 seeds; EV falls with depth (0.93–0.95 at L0–2 → 0.76–0.82 at L11) | **TEST** | `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/` |
| Behaviour | Spliced ΔCE grows with depth (L8: JumpReLU 0.117, TopK 0.126, ReLU 0.223 nats); SAE damage is 0.76–2.89× same-norm noise | validation | `results/openwebtext/spliced_evaluation/` |
| Information (GoT) | Frozen-probe collapse mostly reproduced by noise; re-trained gap 0.07–0.12; SAEs never beat PCA at equal error | validation | M2 |
| Information (persona) | Loss 1.9–3.4× noise; re-trained gap 0.08–0.22, CI > 0 in 99/99 SAEs; H−U cosine 0.35–0.58 | validation | RLM-RP |
| Causal | Reconstruction carries 93–97 % of the persona effect at the L1 slot and 62–68 % at L8/L11 final | validation | M3, M4 |
| Improvement | Method A: task reconstruction −61 to −85 %, frozen-MLP truth gain 4/9, nothing else. Output-KL: behaviour 5/9 creditable, gains elsewhere only with L0 drift; E3 no creditable causal gain | validation | IMP-L, RLM-RP, E3 |

**Methodological strengths worth presenting:**
* pre-registration with hashes;
* frozen vs re-trained readers;
* noise and PCA references;
* the length-shortcut audit and the equal-length task;
* exact decompositions with reproduction checks;
* the corrected circularity audit;
* guards on sparsity.

---

## 3. What is weak, and why

| # | Weakness | Why it matters | How sure I am |
|---|---|---|---|
| W1 | **Data-starved, narrow SAEs.** 101,600 unique tokens, 38 passes, width 3,072, L0 = 128 (4.2 % of latents active; Gemma Scope canonical ≈ 100 of 16,384 ≈ 0.6 %). Validation EV was still rising at 15k steps; up to 85 % dead JumpReLU latents at layer 0. | Every conclusion could be an artefact of SAE quality. This is the first objection to every result. | High: these are facts of the training config |
| W2 | **No external anchor.** No published SAE has been evaluated with our ladder (E2 not run). | We cannot say whether our numbers are typical. | High |
| W3 | **No SAE results on Gemma**, although Gemma-2-2B is in scope; only steering exists. | A gap in the stated scope; Gemma Scope makes it cheap. | High |
| W4 | **No held-out confirmation.** Everything except one table is validation, and validation was used to compare methods. | Effect sizes may be optimistic; nothing is "confirmed". | High |
| W5 | **Fine-tuning confounded by sparsity.** Output-KL's gains coincide with L0 drift; Method A only moves one frozen reader. | The improvement axis of the RQ has no clean positive result. | High (IMP-L/RLM-RP/E3) |
| W6 | **Architecture comparisons not matched on task L0.** On RepE-LM, ReLU/JumpReLU use 142–214 latents vs TopK's 128; truncating JumpReLU to 128 raises its re-trained gap from 0.144 to 0.210 at L3. | We cannot rank architectures on information preservation. | High |
| W7 | **Isotropic noise is the only perturbation reference.** The literature shows conclusions about "pathological" SAE errors change when the baseline is a perturbation towards another real activation instead of isotropic noise. | M2's "noise-like" reading of the GoT collapse may depend on the baseline. | Medium |
| W8 | **No random-SAE baseline.** A 2026 paper reports that SAEs with frozen random decoders match trained SAEs on several downstream metrics. | Reviewers increasingly expect it. | Medium |
| W9 | **No supervised skyline for causal faithfulness.** We report that SAE reconstruction carries 62–68 % at L8, but not how much a supervised low-rank subspace of the same size carries. | Without a skyline, "62 %" has no reference point. | Medium |
| W10 | **Small effects and three seeds.** T_full is 0.022 nats at L8; TopK per-SAE CIs are ±0.06–0.09; three seeds share data. | Small changes are undetectable. A minimum-detectable-effect statement is still missing (Review #8 critique 3). | High |

---

## 4. What the literature says, and how we compare

**SAE quality is driven by data, width and recipe:**
* Published GPT-2 and Gemma SAEs use hundreds of millions to billions of tokens:
  * `gpt2-small-res-jb`: 24,576 latents, 300M tokens, ghost grads ([HF card](https://huggingface.co/jbloom/GPT2-Small-SAEs));
  * Gemma Scope: 16k-wide SAEs on about 4B tokens per a secondary summary of the paper's Table 1; verify before quoting ([Lieberum et al. 2024](https://arxiv.org/abs/2408.05147)).
* Modern recipes:
  * TopK with AuxK and encoder = decoderᵀ init, which nearly eliminates dead latents ([Gao et al. 2024](https://arxiv.org/abs/2406.04093));
  * JumpReLU with a straight-through estimator on L0 ([Rajamanoharan et al. 2024](https://arxiv.org/abs/2407.14435)); our JumpReLU uses a sigmoid surrogate, not this estimator;
  * BatchTopK, which beats TopK on GPT-2 and Gemma-2-2B reconstruction and sets mean L0 directly ([Bussmann et al. 2024](https://arxiv.org/abs/2412.06410));
  * Matryoshka SAEs, which trade a little reconstruction for better disentanglement and less absorption ([Bussmann et al. 2025](https://arxiv.org/abs/2503.17547)).
* **Comparison:** our training budget is about 3,000× smaller than res-jb's, so our SAEs are not representative of
  the published state of the art. I cannot quote a res-jb EV on our data until E2 runs; numbers from different
  token sets and BOS handling are not comparable.

**Faithfulness and its improvement:**
* **End-to-end training:** training on KL to the model's output gives a Pareto improvement in performance explained
  per active feature ([Braun et al. 2024](https://arxiv.org/abs/2405.12241); code: [ApolloResearch/e2e_sae](https://github.com/ApolloResearch/e2e_sae)).
* **A short KL+MSE fine-tune** on the last ~5 % of tokens closes **20–50 %** of the CE gap. KL fine-tuning, LoRA
  adapters and linear adapters give similar, non-additive gains, "suggesting a common, easily correctable error
  source". SAEBench downstream metrics are mixed ([Karvonen 2025](https://arxiv.org/abs/2503.17272)).
* **LoRA on the model**, with the SAE frozen, closes 30–55 % of the CE gap on Gemma Scope SAEs
  ([Chen, Engels & Tegmark 2025](https://arxiv.org/abs/2501.19406)). *This changes the model being explained, so for
  our RQ (faithfulness to the frozen model) it is not a valid improvement, only a diagnostic.*
* **Comparison:**
  * Our output-KL fine-tune cut spliced KL by 8–18 % and ΔCE by 4–19 % relative to Method A, smaller than
    Karvonen's range. Plausible reasons: our tiny SAEs and data, 500 steps, and an L0 controller that drifts.
  * The drift is the key difference. Their TopK/BatchTopK/JumpReLU fine-tunes keep L0 fixed or controlled; our
    ReLU/JumpReLU drifted 5–8 %.

**The error term:**
* About half the SAE error vector, and over 90 % of its norm, is linearly predictable from the input
  ([Engels et al. 2024/25](https://arxiv.org/abs/2410.14670)).
* Error nodes carry causal weight in circuits ([Marks et al. 2024](https://arxiv.org/abs/2403.19647)).
* **Comparison:**
  * Our M4/E3 result agrees: ReLU's error part alone carries 43–45 % of the L8 persona effect.
  * Our error-term AUROCs (persona 0.64–0.83) show the lost task information sits in the error.

**Pathological errors and baselines:**
* SAE splices damage the model more than isotropic same-norm noise
  ([Gurnee 2024](https://www.lesswrong.com/posts/rZPiuFxESMxCDHe4B)).
* That conclusion weakens with better-matched baselines, such as perturbations towards other real activations
  ([arXiv 2410.12555](https://arxiv.org/abs/2410.12555)).
* **Comparison:** our SAE/noise damage ratio of 0.76–2.89 is in the same family of results; our M2 noise reference
  is isotropic only (W7).

**Do SAEs beat simple baselines?** Increasingly, no:
* SAE probes underperform logistic regression across 113 datasets
  ([Kantamneni et al. 2025](https://arxiv.org/abs/2502.16681)).
* The DeepMind team deprioritised SAE research after negative downstream results
  ([progress update, Mar 2025](https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28)).
* SAEs underperform simple baselines for steering on Gemma-2 ([AxBench, Wu et al. 2025](https://arxiv.org/abs/2501.17148));
  a 2026 preprint partially rebuts this ([arXiv 2605.31183](https://arxiv.org/abs/2605.31183)).
* Proxy metrics do not predict downstream usefulness ([SAEBench, Karvonen et al. 2025](https://arxiv.org/abs/2503.09532)).
* SAEs fail to beat a neuron baseline on causal disentanglement in GPT-2, far below a supervised DAS skyline
  ([Chaudhary & Geiger 2024](https://arxiv.org/abs/2409.04478)).
* Frozen random-decoder SAEs nearly match trained ones on interpretability, probing and causal editing
  ([Korznikov et al. 2026](https://arxiv.org/abs/2602.14111)).
* **Comparison:** "SAEs never beat PCA at equal error" (M2) and "the reconstruction carries only 62–68 % of a causal
  effect" (M4) are consistent with this literature and are presentable as such.

**Other relevant points:**
* Seed instability: TopK SAEs are more seed-dependent than ReLU/L1 ([Paulo & Belrose 2025](https://arxiv.org/abs/2501.16615)),
  consistent with our wide TopK CIs.
* Feature absorption ([Chanin et al. 2024](https://arxiv.org/abs/2409.14507)).
* The amortisation gap of SAE encoders ([O'Neill et al. 2024](https://arxiv.org/abs/2411.13117)), whose evidence is
  synthetic.
* Supervised-dictionary evaluations ([Makelov et al. 2024](https://arxiv.org/abs/2405.08366)).
* Skip transcoders Pareto-dominate SAEs on MLP outputs ([Paulo et al. 2025](https://arxiv.org/abs/2501.18823)).
  Transcoders replace MLPs, not the residual stream, so they are a different object from ours.
* Gemma Scope 2 (Gemma 3; SAEs and transcoders on every layer) exists ([HF](https://huggingface.co/google/gemma-scope-2-4b-it)),
  but our scope is Gemma-2-2B, for which Gemma Scope 1 is the reference.
* For truth on Gemma-class models: a 2-D truth subspace, with a polarity-general direction
  ([Bürger et al. 2024](https://arxiv.org/abs/2407.12831)).

---

## 5. Candidate approaches, critically evaluated

Ratings are my judgment. "Prior" is my rough probability that the experiment yields a clear, presentable result in
either direction, not that it improves a metric.

| Approach | Validity for our RQ | Feasibility / cost | Prior | Verdict |
|---|---|---|---|---|
| A. **External anchor (E2, res-jb)** | High: tests W1/W2 without training | ~40 min GPU, cached | High | **Essential** |
| B. **Data-scaled retraining** (100M+ streamed tokens, wider, modern recipe) | High: directly tests W1 | ~1 h GPU per layer for 9 SAEs; new streaming trainer (~1 day) | High | **Essential** |
| C. **Gemma Scope faithfulness ladder** | High: fills W3 with SOTA SAEs, no training | 4–8 h GPU; RepE-LM must be re-tokenised and re-gated for Gemma | High | **Essential** |
| D. **M5 held-out confirmation** | Required for any "confirmed" claim | ~1–2 h GPU; protocol freeze | High | **Essential** |
| E. **L0-fixed short KL fine-tune** (Karvonen-style, on B's TopK/BatchTopK SAEs and one Gemma Scope SAE) | High: the literature-backed improvement, without our drift confound | 1–3 h GPU | Medium | **Recommended** |
| F. **Supervised causal skyline** (rank-k DAS / probe-subspace patch at the L8 site) | High: gives W9's reference point | <1 h GPU; small optimisation on TRAIN pairs | Medium-high | **Recommended** |
| G. **Random-decoder SAE baseline** (Korznikov) in the ladder | High: answers W8 | Cheap once B's trainer exists | High | **Recommended** |
| H. **Real-activation-direction perturbation reference** (2410.12555) | Medium-high: strengthens M2 (W7) | Cheap; re-uses M2 code | Medium | Recommended |
| I. **Gemma Scope L0 sweep** (same width, several published L0s) | High: a within-architecture sparsity dose-response (W6), no training | Cheap within C | High | Recommended (part of C) |
| J. BatchTopK / Matryoshka vs TopK/ReLU/JumpReLU | Medium: architecture comparison at matched L0 | Adds SAEs to B | Medium | Optional |
| K. Inference-time encoder optimisation (gradient/matching pursuit) | Medium: splits encoder vs dictionary error; Engels found only a slight gain | Cheap | Low-medium | Optional (diagnostic) |
| L. Linear-error diagnostic (Engels): how much of the lost persona/causal effect is linearly predictable | Medium: explains *where* the loss is | Cheap | Medium | Optional |
| M. E1 (task-L0-matched TopK, our small SAEs) | Medium, but largely superseded by B + I | 1.2 h | Medium | Optional (only if B is not done) |
| N. LoRA on the model (Chen et al.) | **Low**: changes the explained model | Moderate | – | **Not recommended**, except as a labelled diagnostic |
| O. More fine-tuning variants on the current SAEs | Low: IMP-L/RLM-RP/E3 already answered this | – | Low | **Not recommended** |
| P. Training Gemma SAEs from scratch | Low value: Gemma Scope is far stronger than what we could train | ~2 h per layer for 100M tokens | – | Not recommended |
| Q. Transcoders / crosscoders | Different object (MLP replacement); off-RQ | – | – | Not recommended for the capstone |

---

## 6. Recommended plan (Problem → Solution → Pipeline → Evaluation → Contribution)

**Rules for every item:**
* a pre-registration with hashes before any run, as in M0–E3;
* validation for development; test only in M5;
* L0 reported on OWT, GoT and RepE-LM;
* no selection of favourable cells.

### R1. External anchor: run E2 (`gpt2-small-res-jb`, layers 8 and 10, cached)

* **Problem:** we do not know whether our faithfulness numbers are typical (W1, W2).
* **Solution:** run the already-drafted E2 pre-registration unchanged.
* **Pipeline:**
  * res-jb SAEs on our OWT validation tokens (BOS excluded, as in our protocol);
  * M2 readers on GoT and RepE-LM;
  * M4 decomposition at L8 final.
* **Evaluation:**
  * EV, L0 and spliced ΔCE/KL vs our 9 SAEs at the same layer;
  * frozen and re-trained gaps; T_rec/T_full.
  * res-jb's L0 differs from ours, so it is reported as a reference point, never ranked against our matched SAEs.
* **Contribution:** the first answer to "is it just our SAEs?" at almost no cost. It also calibrates R2's targets.

### R2. Data-scaled retraining of the GPT-2 SAEs (the main new experiment)

* **Problem:** W1. Our SAEs saw 0.1M unique tokens.
* **Solution:** retrain with a streaming trainer and a modern recipe.
  * Data: about 100M OWT tokens (train split; documents disjoint from our validation/test, verified by hash),
    activations generated on the fly. Storing them is impossible: 100M × 768 × 2 bytes ≈ 154 GB per layer, against
    108 GB free.
  * Width 12,288 (16×).
  * Architectures: TopK + AuxK with encoder = decoderᵀ init; JumpReLU with the published STE; ReLU-L1 as continuity
    with our study; optionally BatchTopK.
  * L0 matched at 64 and 128 (two operating points, addressing "one operating point").
* **Pipeline:**
  * one GPT-2 pass per layer feeds all 9 SAEs (3 architectures × 3 seeds) simultaneously;
  * layers 1, 8, 11 (the M3/M4 sites), plus 3 or 6 if time allows;
  * a random-frozen-decoder control SAE (G) per layer;
  * then the full ladder: test reconstruction (M5 only), validation splicing, M2 readers, RLM-RP, M4 decomposition.
* **Evaluation:**
  * pre-registered paired comparisons, new vs old SAE, same layer, architecture and target L0;
  * reconstruction: EV, ΔCE, dead fraction;
  * information: frozen and re-trained gaps, vs noise and PCA;
  * causal: T_rec/T_full at L1 slot, L8 and L11 final;
  * guards: L0 within ±10 % of target on OWT, with L0 reported on GoT and RepE-LM.
* **Expected contribution, honestly stated:**
  * Reconstruction and ΔCE will very likely improve: data and width are the best-established levers in the literature.
  * The key scientific question is whether the *faithfulness* gaps shrink with them:
    * persona re-trained gap;
    * H−U cosine;
    * causal preservation at L8/L11;
    * PCA dominance.
  * **Both outcomes are presentable:**
    * "faithfulness improves with SAE quality" is an improvement result;
    * "it does not, even though reconstruction improves" is a strong confirmation of the project's central claim
      that reconstruction is a proxy for behaviour, not for information.
* **Cost:**
  * about 100M tokens at ~60k tokens/s ≈ 30 min per layer for the GPT-2 forward, plus SAE updates, so about
    1–1.5 h per layer for 9–10 SAEs;
  * about 4–6 h GPU for 3–4 layers;
  * about 1 day of engineering for the streaming trainer (reusing our SAE class and loss code).

### R3. Gemma-2-2B faithfulness ladder with Gemma Scope

* **Problem:** W3. No SAE work on Gemma, although it is in scope.
* **Solution:** apply the same ladder to Gemma Scope residual SAEs (JumpReLU, width 16k) at about three depths, e.g.
  layers 6, 12 and 18 of 26, chosen by a Gemma M3 site sweep on TRAIN prompts, not by SAE results.
  * Use **several published L0 levels per layer** (I), giving a sparsity dose-response within one architecture.
* **Pipeline:**
  1. **RepE-LM for Gemma.** Re-tokenise the RepE-LM prompts with Gemma's tokenizer and re-run the M1 construction
     gates (K1 equal length, K2 shortcuts). Word pairs that tokenise unequally are replaced *before* any SAE is
     loaded. **This is a real risk:** equal length under GPT-2's tokenizer does not carry over.
  2. GoT cities, with the existing city-grouped splits and the same A/B probe protocol.
  3. Splicing on OWT validation text (CE/KL).
  4. M2 readers (frozen, re-trained, noise, PCA).
  5. Gemma M3 site sweep (no SAE), then the M4 decomposition at the selected sites.
* **Evaluation:**
  * the same metrics as GPT-2;
  * comparisons across L0 levels within a layer;
  * GPT-2 vs Gemma qualitatively only (different models, widths and training).
  * Gemma Scope has one seed per configuration: stated as a limitation, with statement-bootstrap CIs instead.
* **Contribution:**
  * the project's first cross-model SAE result, on state-of-the-art public SAEs;
  * a direct test of whether the GPT-2 findings generalise:
    * frozen-probe collapse ≈ noise;
    * persona loss real;
    * causal leakage at mid/late sites.
* **Cost:**
  * Gemma forward ≈ 7.7k tokens/s; RepE-LM (~10k prompts, ~25 tokens) ≈ 30 s per pass;
  * M4 on Gemma is roughly 10–20× slower than on GPT-2 (estimate), so about 10–20 min per SAE;
  * Gemma Scope 16k SAEs are about 0.3 GB each (download needed; HF reachable);
  * about 4–8 h GPU in total;
  * host RAM (15 GB) rules out TransformerLens' default CPU loading of Gemma; load via HF bf16 directly to the GPU
    (verified to work, 7.3 GB peak).

### R4. Sparsity-safe KL fine-tuning (the improvement method, done properly)

* **Problem:** W5. Our output-KL gains coincide with L0 drift.
* **Solution:** a Karvonen-style short KL+MSE fine-tune (a few % of R2's token budget) only on
  architectures with L0 fixed by construction (TopK/BatchTopK) or tightly controlled. Apply it to R2's SAEs at L8
  and to one Gemma Scope SAE (a JumpReLU threshold fine-tune needs an explicit L0 constraint, pre-registered).
* **Pipeline:**
  * fine-tune on a disjoint OWT stream; base and an MSE-only continued-training control get the same number of
    tokens;
  * evaluate with IMP-L's guards and the full ladder, including E3-style causal preservation.
* **Evaluation:**
  * pre-registered, paired by seed;
  * ΔCE/KL (behaviour); re-trained gaps (information); T_rec/T_full (causal);
  * L0 on all three datasets.
* **Contribution:**
  * the only clean test of whether output-matching improves causal and representational faithfulness at fixed
    sparsity;
  * directly answers the open question E3 left: TopK showed no change, but with tiny SAEs and wide CIs.
* **Cost:** about 1–3 h GPU.

### R5. Reference points for causal faithfulness: supervised skyline and random baseline

* **Problem:** W8, W9. "62–68 %" and "beats random features" lack an upper and a lower anchor.
* **Solution:**
  * **Skyline:** at the L8 final site, learn a rank-k subspace (k ∈ {1, 4, 16, 64}, matching M4's feature counts)
    on TRAIN pairs by distributed-alignment-style optimisation (maximise restoration T), and report T_subspace / T_full
    on validation. Also report the logistic-probe direction (rank 1, no causal training).
  * **Lower bound:** a frozen-random-decoder SAE (from R2) through the same M4 decomposition.
* **Evaluation:** SAE top-k features vs the supervised subspace at equal k, vs random-decoder SAE features at equal k.
* **Contribution:** places SAE causal faithfulness between a lower bound and an upper bound, the framing used by
  Chaudhary & Geiger and Korznikov et al. This makes the M4/E3 numbers interpretable for the professor.
* **Cost:** under 1 h GPU.

### R6. Better perturbation reference for information preservation

* **Problem:** W7. Isotropic noise may be a lenient or strict reference.
* **Solution:** add a "real-direction" reference: perturb x along (x′ − x) for a random other activation x′ from the
  same layer, scaled to ‖e‖, following arXiv 2410.12555.
* **Evaluation:** recompute M2's noise-reference gaps (GoT, persona) with both references. Report whether the "GoT
  collapse ≈ noise" reading survives.
* **Contribution:** a robustness check on one of the project's headline claims.
* **Cost:** minutes of GPU.

### R7. M5: one held-out confirmation pass (essential, needs separate approval)

* **Freeze first:** the protocol, after R1–R6 have been selected on validation.
* **Then run:** the headline comparisons once each, on the OWT test, the GoT test (labelled previously exposed) and a
  newly built RepE-LM test (with the duplicate check).
* **Contribution:** the only route to "confirmed" claims.

---

## 7. Compute, timeline, essential vs optional

**Deadline:** 2026-12-08. Instructor checkpoints: 11-17, 11-22, 12-01.

| Week | Work | GPU (A10G) | Engineering |
|---|---|---|---|
| 11-10 → 11-17 | R1 (E2); R6; streaming trainer + R2 pilot at L8 (one seed, check EV/L0/dead); Gemma RepE-LM re-tokenisation and gates; Gemma Scope download; minimum-detectable-effect statement | ~3 h | 2–3 days |
| 11-17 → 11-24 | R2 full (layers 1, 8, 11 × 3 architectures × 3 seeds + random-decoder controls) and its ladder; R3 Gemma ladder | ~10–14 h | 2 days |
| 11-24 → 12-01 | R4 (KL fine-tune, fixed L0); R5 (skyline); freeze and run M5 | ~5 h | 2 days |
| 12-01 → 12-08 | Paper and presentation; no new experiments | – | – |

**Total:** about 20–25 h GPU, all on the existing A10G, with no paid resources. The largest risks are the streaming
trainer's engineering time and the Gemma RepE-LM re-tokenisation gates.

* **Essential:** R1, R2 (at least layers 8 and 11), R3 (at least one layer with the full ladder), R7, and the
  minimum-detectable-effect statement.
* **Recommended:** R4, R5, R6, and the Gemma L0 sweep within R3.
* **Optional:** BatchTopK/Matryoshka (J), the encoder-optimisation diagnostic (K), the linear-error diagnostic (L), E1.
* **Not recommended:** LoRA on the model (changes the explained model), more fine-tuning of the current SAEs, Gemma
  SAE training from scratch, transcoders.

**What we could reasonably aim to show at the final presentation (no promises before testing):**
1. A reconstruction/behaviour improvement from data-scaled SAEs (EV and spliced ΔCE at matched L0, test-confirmed in
   M5). This is likely, given the literature, but unmeasured.
2. Whether the information and causal faithfulness gaps follow reconstruction or not, in GPT-2 (retrained and
   external SAEs) and Gemma-2-2B (Gemma Scope), with a sparsity dose-response. This is the central RQ answered across
   models and SAE quality.
3. Whether output-matching fine-tuning improves causal faithfulness at fixed sparsity (R4), with honest classes.
4. Where SAEs sit between a random baseline and a supervised skyline on causal faithfulness (R5).

**Why this is stronger than the current plan:**
* The current plan (E1 on the small SAEs, more fine-tunes) refines results about SAEs that a reviewer can dismiss as
  under-trained.
* The plan above makes every headline claim robust to that objection, adds the missing model, and puts our negative
  results in the context of the 2025–26 literature, which is moving the same way.

---

## 8. References (links checked 2026-11-10)

* Gao et al. 2024, *Scaling and evaluating sparse autoencoders*. https://arxiv.org/abs/2406.04093
* Rajamanoharan et al. 2024, *Jumping Ahead: JumpReLU SAEs*. https://arxiv.org/abs/2407.14435
* Bussmann, Leask, Nanda 2024, *BatchTopK SAEs*. https://arxiv.org/abs/2412.06410 (code: https://github.com/bartbussmann/BatchTopK)
* Bussmann et al. 2025, *Matryoshka SAEs*. https://arxiv.org/abs/2503.17547
* Lieberum et al. 2024, *Gemma Scope*. https://arxiv.org/abs/2408.05147; weights: https://huggingface.co/google/gemma-scope-2b-pt-res
* Braun et al. 2024, *End-to-end sparse dictionary learning*. https://arxiv.org/abs/2405.12241 (code: https://github.com/ApolloResearch/e2e_sae)
* Karvonen 2025, *Revisiting end-to-end SAE training: a short finetune is all you need*. https://arxiv.org/abs/2503.17272
* Chen, Engels, Tegmark 2025, *Low-rank adapting models for SAEs*. https://arxiv.org/abs/2501.19406
* Engels, Riggs, Tegmark 2024/25, *Decomposing the dark matter of SAEs*. https://arxiv.org/abs/2410.14670
* Marks et al. 2024, *Sparse feature circuits*. https://arxiv.org/abs/2403.19647
* Gurnee 2024, *SAE reconstruction errors are (empirically) pathological*. https://www.lesswrong.com/posts/rZPiuFxESMxCDHe4B
* *Investigating sensitive directions in GPT-2: an improved baseline* (2024). https://arxiv.org/abs/2410.12555
* Kantamneni et al. 2025, *Are sparse autoencoders useful? A case study in sparse probing*. https://arxiv.org/abs/2502.16681
* Karvonen et al. 2025, *SAEBench*. https://arxiv.org/abs/2503.09532 (code: https://github.com/adamkarvonen/SAEBench)
* Wu et al. 2025, *AxBench*. https://arxiv.org/abs/2501.17148; partial rebuttal: https://arxiv.org/abs/2605.31183
* Chaudhary & Geiger 2024, *Evaluating open-source SAEs on disentangling factual knowledge in GPT-2 Small*. https://arxiv.org/abs/2409.04478
* Korznikov et al. 2026, *Sanity checks for SAEs: do SAEs beat random baselines?* https://arxiv.org/abs/2602.14111
* Paulo & Belrose 2025, *SAEs trained on the same data learn different features*. https://arxiv.org/abs/2501.16615
* Paulo, Shabalin, Belrose 2025, *Transcoders beat SAEs for interpretability*. https://arxiv.org/abs/2501.18823
* Chanin et al. 2024, *A is for Absorption*. https://arxiv.org/abs/2409.14507
* O'Neill, Gumran, Klindt 2024, *Compute optimal inference and provable amortisation gap in SAEs*. https://arxiv.org/abs/2411.13117
* Makelov, Lange, Nanda 2024, *Towards principled evaluations of SAEs*. https://arxiv.org/abs/2405.08366
* Bürger, Hamprecht, Nadler 2024, *Truth is universal*. https://arxiv.org/abs/2407.12831
* GDM mech-interp team 2025, *Negative results for SAEs on downstream tasks*. https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28
* Bloom 2024, GPT-2 Small residual SAEs. https://huggingface.co/jbloom/GPT2-Small-SAEs
* EleutherAI `sparsify` (TopK SAE and transcoder trainer). https://github.com/EleutherAI/sparsify

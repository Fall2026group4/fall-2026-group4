# Research decisions log

Each entry: decision, date, reason, alternatives considered, consequence, and the scientific question it addresses. Dates come from commits and the session record. Add new decisions at the bottom (`CLAUDE.md` §1).

## D1. Models: GPT-2 Small as the primary model; Gemma-2-2B secondary (proposal, 2026-09)
* **Reason:**
  * GPT-2 Small has public SAEs, full TransformerLens hook support, and fits the A10G (23 GB).
  * Gemma-2-2B tests generalisation to a larger model.
* **Consequence:**
  * All SAE training and faithfulness work is on GPT-2.
  * Gemma has only steering experiments. It could not be loaded with TransformerLens on the 15 GB-RAM host (OOM), so it uses HF transformers in bf16 (2026-11-04).
* **Open:** no Gemma SAEs, because Gemma Scope weights are not cached.

## D2. Datasets: OWT for reconstruction and behaviour; GoT for truth; RepE for honesty (proposal)
* **OWT** is GPT-2's training distribution, and the standard SAE training and evaluation corpus.
* **GoT** provides paired true/false statements whose structure isolates truth. Used: cities + neg_cities (geographic), larger_than + smaller_than (numerical).
* **RepE** persona prompts give a contrastive honesty signal.
* **The Pile** (proposal): only EDA was done, on a team branch. It is not used in experiments.

## D3. Train SAEs in house instead of using public SAEs (2026-10-30, `8dc0b6e`)
* **Reason:** public SAEs (`gpt2-small-res-jb`, the v5-32k TopK) differ in width, data, sparsity and training. An architecture comparison needs one protocol.
* **Alternative:** compare public SAEs. Rejected because it confounds architecture with training.
* **Consequence:**
  * 3072-wide SAEs (4×) trained on 101,600 OWT tokens. This is a small dataset (~38 epochs at 15k steps), so absolute quality is below public SAEs.
  * JumpReLU uses a sigmoid surrogate, not the published estimator.

## D4. Hook `blocks.L.hook_resid_pre`; BOS excluded; final token for GoT/RepE
* **Reason:**
  * Residual stream before block L is the standard SAE site.
  * BOS has an outlier norm.
  * GoT/RepE probes read the final token, as in the original papers' last-token analyses.
* The same hook is used in training, probes and splicing (verified against the caches, ≤ 1e-4).

## D5. Uninterrupted single-optimiser training protocol (2026-11-04, professor review)
* **Reason:** layer 6 had been resumed with Adam state restored, and layers 2/10 warm-started with Adam reset, which confounds cross-layer comparison.
* **Consequence:**
  * All 108 baseline runs use one uninterrupted 15k-step run (one optimiser, one controller, one RNG stream).
  * The earlier results changed by ≤ 0.006 EV.

## D6. Sparsity-matching rules (2026-11-04, extended 11-07)
* **Rules:**
  * A run is matched if |L0 − target| / target ≤ 10 %.
  * A cell is comparable only if all its seeds are matched.
  * Architectures are compared only when all three are matched.
  * Fine-tunes must stay within ±5 % of their base.
  * L0 is reported separately on OWT, GoT and RepE.
* **Reason:** EV depends strongly on L0. Unmatched comparisons were behind the invalid `architecture_summary.csv`.
* **Known limitation:** matching is defined on OWT. On GoT and RepE prompts the architectures differ strongly in L0 (GoT: ReLU 66, TopK 128, JumpReLU 46).

## D7. Frozen probes fitted on original activations as the representation-preservation metric (2026-11-04)
* **Reason:** a probe re-trained on reconstructions measures decodability, not whether the original readout survives. The layer-10 result showed the difference: 0.851 re-trained vs 0.639 frozen.
* **Consequence:**
  * All truth/honesty preservation metrics apply probes frozen.
  * The re-trained probe is kept only as a labelled decodability diagnostic.

## D8. Informative-layer rule (original validation AUROC ≥ 0.70), decided per probe (2026-11-07)
* **Reason:** gaps are meaningless when the original probe is at chance. The GoT mass-mean direction is weak (≤ 0.63), so it is excluded from GoT rankings.

## D9. Extend all experiments to all 12 layers before improving anything (2026-11-07)
* **Reason:** the research question is about behaviour across layers. Only layers 2/6/10 existed.
* **Consequence:** 108-run baseline, all-layer GoT/RepE, all-layer splicing.

## D10. Activation splicing as the behavioural faithfulness measure (Phase 0, 2026-11-07)
* **Reason:** reconstruction and probes don't show whether GPT-2's computation survives.
* **Design choices:**
  * KL(p_orig ‖ p_spliced);
  * per-dataset task metrics;
  * identity, fp16, mean-ablation and norm-matched random-error controls.
* **Two metrics were replaced after a smoke test:**
  * the "persona shift retained" ratio, which random noise inflated, became persona-shift cosine;
  * pooled GoT preference became per-dataset preference.
* **Explicitly not steering.**

## D11. Fix sparsity control before improvement methods (P1, 2026-11-07)
* **Reason:** layer-0 JumpReLU had failed, ReLU overshot, and no runs had checkpoint selection.
* **Outcome:**
  * L0 targeting improved.
  * The JumpReLU threshold initialisation helped at layers 2/6.
  * Layer 0 is unresolved.
  * Selection always chose the final step, so the SAEs are not converged.

## D12. Reclassify truth-aware results as circular (2026-11-07)
* **Reason:** the evaluation probe was the training teacher (identical weights).
* **Alternatives considered:** keep them with a caveat. Rejected: they must not be cited as truth preservation.
* **Numbers kept unchanged.**

## D13. GoT train A/B split for independent probes (2026-11-07)
* **Reason:** prevent any evaluation probe from sharing training rows or parameters with a training objective.
* **Split:** train groups split 50/50 by group, seed 20261107. Both halves are balanced and support informative probes (validation AUROC 0.80–0.82 geographic, ≥ 0.996 numerical).
* **Use:** A for fine-tuning and teachers; B for evaluation probes.

## D14. Pre-registration with paired "consistent" criteria (2026-11-07 onwards)
* **Rule:** better on all 3 seeds AND mean paired difference > 2 × its sd, plus guards (L0, OWT EV/CE, `cities` preference, RepE).
* **Consequence:**
  * Strict: Method A failed it despite 18/18 seeds in the right direction.
  * Replication with more seeds would need a new pre-registration.

## D15. Introduce Method A (GoT reconstruction fine-tuning with OWT replay) (2026-11-07)
* **Reason:** the corrected pilot showed reconstruction-only fine-tuning was the best variant on the independent MLP probe.
* **Design:**
  * OWT replay (50 %) to protect general reconstruction;
  * sparsity controller continued from each base;
  * an OWT-only control to separate "more training" from "GoT data".
* **Answered:** GoT data, not extra training, drives the gain. The gain is consistent in direction but noisy.

## D16. Abandon linear-teacher truth-aware fine-tuning in its current form (2026-11-07)
* **Reason:** in both the circular and the corrected settings, it optimises the teacher's direction at the cost of broader readouts.
* **Kept open:** a non-linear or subspace teacher, which would need a new pre-registered design.

## D17. E2E-inspired objectives (2026-11-07)
* **Reason:** test whether optimising behavioural faithfulness (output KL, Braun et al. 2024) improves faithfulness beyond reconstruction.
* **Deliberate deviations from the paper:**
  * fine-tuning instead of training from scratch;
  * local reconstruction kept;
  * own sparsity controllers;
  * forward KL with the original as reference;
  * BOS kept;
  * variance-normalised downstream terms.
* **Answered:**
  * Behavioural faithfulness improves (TopK, matched).
  * Truth readouts do not.
  * KL pushes ReLU/JumpReLU L0 up faster than the controller corrects it.

## D18. Never use test splits for selection; touched only by pre-registered final evaluations
* The GoT layer-6 test split has already been scored twice for the historical fine-tuned SAEs, so it cannot select among them.
* All 2026-11-07 improvement experiments are validation-only.

## Unresolved hypotheses
* **H1:** Wider SAEs or longer training close the deep-layer reconstruction and behaviour gap. Not tested; validation EV was still rising at 15k.
* **H2:** A converged EV-vs-L0 frontier changes the architecture ranking at low L0. Not tested.
* **H3:** Method A's MLP-probe gain replicates with more seeds and at other layers. Not tested.
* **H4:** Combining Method A with a small KL weight keeps the truth gains and the behaviour gains. Not tested.
* **H5:** Layer-0 JumpReLU failure can be fixed by threshold learning-rate control or dead-latent revival. Not tested.
* **H6:** Results transfer to Gemma-2-2B SAEs. Not possible yet.
* **H7:** SAE-based interventions can have causal effects distinguishable from controls. All tested were null at layer 6.

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

## D19. KL-weight ablation design (2026-11-07)
* **Decision:** test w ∈ {0.25, 0.50} of Method B's KL coefficient for TopK at layer 6, with A (w = 0) and B (w = 1) as reused endpoints.
* **Coefficient semantics:** α_B is the raw and only KL multiplier, already scale-matched on train data. Fractions scale α_B; per-term weight = 0.5·α.
* **Initialisation:** every run starts from the original TopK checkpoint, not from Method A.
* **Criteria:**
  * a chord-based TRADE criterion plus behaviour and truth-non-inferiority tests;
  * an explicit INCONCLUSIVE rule for when the endpoints don't differ in truth;
  * fractions fixed before any result.
* **Why TopK only:** L0 is fixed, so the ReLU/JumpReLU L0-drift confound of the E2E experiment is absent.
* **Outcome:** INCONCLUSIVE.
  * For TopK at layer 6 there is no measurable truth cost of the full KL weight (+0.007 ± 0.02).
  * Behaviour saturates by w = 0.25.
* **Consequence:** truth-trade-off questions need more seeds (power) or a setting where the endpoints actually differ (e.g. ReLU/JumpReLU once L0 control holds).

## D20. Retire the separate professor-review folders (2026-11-07)
* **Decision:** remove `results/professor_review_corrections/` and `scripts/professor_review/` (and the duplicate test `src/tests/test_professor_review.py`) from the working tree, approved by the user before the GitHub upload.
* **Reason:** since the 2026-11-06 migration (D9-era reorganisation, `2ad0979`) every file existed twice: once in the review folder and once, hash-identical, in its dataset folder. Two parallel copies invite citing the wrong one.
* **How evidence is preserved:**
  * the corrections themselves stay in the dataset folders (`results/{openwebtext,geometry_of_truth,repe,gemma}/`, `scripts/{openwebtext,geometry_of_truth,shared}/`);
  * the professor-review report, code audit and methodology changes stay verbatim in `docs/2026-11-04_*.md` (their paths refer to the old layout; `docs/README.md` maps them);
  * the tracked originals stay in Git history; the full folders, including gitignored checkpoints and logs, are in `../backups/2026-11-07_retired_professor_review_folders/`;
  * `docs/migration_manifest.csv` lists every retired file with its SHA-256 and its kept copy.
* **Alternatives considered:** keep the folders (rejected by the user: no separate folder in the final structure); delete without backup (rejected: the 30 gitignored checkpoint copies exist nowhere else in this form).
* **Earlier errors stay documented:** the corrections (optimiser-state inconsistency, unmatched sparsity, steering without an SAE in the forward pass, retrained-probe faithfulness, leaky Gemma split) remain described in `RESEARCH_PROGRESS.md` §6, this log (D5–D7) and the changelog.

## D21. R1 design, test deferral and outcome (2026-11-07)
* **Decision:** test whether a paired honest-vs-untruthful contrast loss (no probe in the loss) improves honesty preservation, against RepE reconstruction-only and a same-persona-pair control, with independent probes fitted on a disjoint train half (B).
* **Why no probe-based objective:** the D12/D16 circularity lesson; the contrast loss targets activations, and the primary metric is a nonlinear probe fitted on B only.
* **Test split:** the drafted plan included one test pass; on 2026-11-07 the user instructed that the test split be preserved, so the R1 verdict is validation-only and the test pass is deferred pending approval. Recorded in the pre-registration addendum before training.
* **Guard scope:** clarified before training: the verdict needs the guards of R-honesty and R-recon.
* **Outcome:** ReLU INCONCLUSIVE; TopK/JumpReLU NOT SUPPORTED. The contrast loss strongly improves direction-level fidelity but changes probe AUROC only for ReLU, the architecture with the worst base preservation, and there it coincides with RepE-prompt L0 drift.
* **Consequence:** prompt-specific L0 control (S3) is needed before any honesty-preservation claim; R2 uses base SAEs as its primary set, so it is not blocked by R1's verdict.

## D22. R2 staging and outcome (2026-11-08)
* **Decision (user, 2026-11-08):** run the R2 smoke test, the k selection and all controls on validation only, report them as VALIDATION evidence, and stop before the test stage. The script's `--test` refuses.
* **Corrections before the run (README §12):**
  * the OWT damage set is the 100 cached validation sequences, not 200;
  * criterion 5 is checked on cross-wording validation at this stage.
  * Pipeline-test outcomes (TopK E < 0) were seen before freezing; the protocol was not changed.
* **Outcome:** NOT SUPPORTED for all architectures and all k. Clamped honesty features move outputs *away* from the honest-persona distribution, and further than random-feature or matched-norm controls do.
  * The probe-score manipulation check moves the right way on default wording only.
  * Criterion 2 passing is recorded as uninformative.
* **Consequence:**
  * No causal SAE-feature faithfulness claim at layer 6.
  * The test stage is unlikely to change criterion 1. Running it remains the user's decision.
  * Any per-statement-clamp or alternative-site follow-up needs a new pre-registration.

## D23. Reframe as a layer-wise measurement study; start M0–M5 (2026-11-08)
* **Decision (user approved 2026-11-08):** the thesis becomes a layer-wise measurement of how faithfully ReLU, TopK and JumpReLU SAEs of GPT-2 Small preserve reconstruction, task information, behaviour and causal effects. Fine-tuning becomes secondary. M0–M5 implement it (plan presented 2026-11-08); M0 and M1 are executed first, then the user decides on M2/M3.
* **Why:**
  * the 2026-11-08 audit found that the RepE persona prompts are length-confounded (' untruthful' adds 3 tokens; length alone gives AUROC 0.80);
  * pooled-across-layer correlations were reported as if within-layer;
  * frozen-logistic failures were read as information loss without noise or PCA references;
  * the layer-6 causal site carries only about 20 % of the persona effect.
* **Reclassified as HISTORICAL** (kept, not cited for their original claims):
  * all-layer RepE probe results;
  * the RepE parts of splicing;
  * 6c causal random controls;
  * R1 and R2;
  * GoT numerical as a truth measure (surface shortcut, AUROC 0.84).

  Gemma steering moves to an appendix.
* **Refinements fixed before running** (in the pre-registrations):
  * M0 uses Freedman–Lane within-layer permutation, and GoT relative squared error instead of task NMSE;
  * M1's K7 null is an honest-synonym one-token swap (Hp);
  * K6 forbids same-label near-duplicates and discloses the 14 opposite-label counterfactual edits found while building the dataset.
* **Test split:** untouched by M0 and M1; M5 only, with approval.

## D24. M3 outcome and the M4 site question (2026-11-08)
* **Outcome:** the pre-registered rule selected L1 slot and L11 final; both confirmed on validation. M4 is justified by the rule.
* **Issue recorded, not acted on:** both sites are near-endpoints (E ≈ 0.96), where "carrying the effect" is close to tautological. The layer 6–8 cells, where the effect is split between the slot and the final token, are more informative.
* **Decision:** adding such a site requires a pre-registration amendment before M4. This is left to the user.
* **T1 (truthful/deceptive):** has no effect beyond its synonym control. Persona-specific claims rest on T2–T4.

## D25. M4 approved with amendment A1 (mid-depth site) (2026-11-08)
* **Decision (user, 2026-11-08):** run M4 at the two M3 sites plus one layer 6–8 site.
* **Rule for the added site (A1):** fixed before running. Choose the M3-passing slot/final cell with train E closest to 0.5, then confirm it on validation T2–T4 with original activations, without SAEs. By the M3 tables this is L8 final.
* **Template sets:** T2–T4 is the primary persona set. T1 is kept and reported as a control case.
* **Feature claims:** limited to "SAE-defined directions are causally sufficient when inserted"; never "GPT-2 uses SAE features".

## D26. IMP-L outcome and its reading (2026-11-09)
* **Run:** run as pre-registered after user approval. The criteria were not changed after the results were seen.
* **Classification:** exactly as `criteria.json` gives it. The NOT CREDITABLE cells (ReLU/JumpReLU output-KL at layers 8–9) are excluded from rankings but reported.
* **Reading:** the frozen-probe Method A truth gains are reported with the re-trained-reader null beside them. They are not called information recovery, because M2 showed that frozen-probe gaps are dominated by probe fragility.
* **Next experiments:** none started. E3, an L0-matched output-KL variant and extra seeds each need their own pre-registration and approval.

## D27. Reading of IMP-L fixed for the paper; next-experiment order (user, 2026-11-09)
* **Reading (user):**
  * Method A improves frozen-probe preservation in 4/9 cells.
  * Re-trained probes show no consistent information gain.
  * Output-KL improves behavioural preservation in 5/9 creditable cells, and does not improve truth preservation.
  * The paper must keep these distinctions and must not claim truth-information recovery.
  * Every finding stays VALIDATION until the approved final test.
* **Order:** E3 (causal evaluation of the existing layer-8 fine-tunes), then E2 (public SAE reference), then E1
  (task-level sparsity matching).
* **Approval:** each needs the user's approval of its pre-registration before running. M5, pushing and merging still
  need separate approval.
* **E2 scope:** use the cached layer-8 and layer-10 public SAEs; do not download layer 6.
* **E1 protocol:** do not train through script 13, because it evaluates on the OWT test split.

## D28. E3 and RLM-RP run; their implementation choices (2026-11-10)
* **Approval:** the user approved running E3 and the RepE completion work (RLM-RP). M5, the test split, pushing and
  merging remain excluded.
* **E3 implementation (fixed after a mechanics-only smoke test, before any result):**
  * per-SAE reseeding, so control draws are paired across arms;
  * M4 reproduction on the deterministic columns, because M4's single random stream cannot be replayed;
  * BH over the 36 SAEs.
* **RLM-RP:**
  * layer-6 fine-tunes are included, with the uninterrupted base as source (asserted); layer-6 classes are not
    guard-checked because no IMP-L-format guards exist;
  * the IMP-L reproduction tolerance (1e-3 for the gap, 1e-4 for L0) was set from smoke-test mechanics, because
    IMP-L encoded a different row subset; M2 must match exactly;
  * the L0-truncation control is exploratory.
* **Reading:**
  * Method A changes neither persona representation nor causal preservation.
  * Output-KL's causal and representational gains appear only with extra active latents (not creditable for
    ReLU/JumpReLU at L8). TopK, whose L0 is fixed, shows no causal change.
  * An L0-matched output-KL variant (roadmap G8/S4) is therefore the decisive follow-up for any "output-KL improves
    causal faithfulness" claim.

## Unresolved hypotheses
* **H1:** Wider SAEs or longer training close the deep-layer reconstruction and behaviour gap. Not tested; validation EV was still rising at 15k.
* **H2:** A converged EV-vs-L0 frontier changes the architecture ranking at low L0. Not tested.
* **H3:** Method A's MLP-probe gain replicates with more seeds and at other layers. **Tested at layers 3, 8, 9 (D26):** the direction replicates in 9/9 cells and the criterion is met in 4/9. Re-trained readers show no gain. More seeds are not tested.
* **H4:** Combining Method A with a small KL weight keeps the truth gains and the behaviour gains. **Tested for TopK at layer 6 (D19): inconclusive.** Behaviour gains are kept (≈85 % at w = 0.25), and truth is non-inferior, but A and B themselves don't differ in truth. At layers 3, 8, 9 (D26), output-KL never improves truth over Method A and regresses at L3 ReLU/TopK. The ReLU/JumpReLU L0 drift persists at depth.
* **H5:** Layer-0 JumpReLU failure can be fixed by threshold learning-rate control or dead-latent revival. Not tested.
* **H6:** Results transfer to Gemma-2-2B SAEs. Not possible yet.
* **H8:** A paired-contrast objective improves honesty preservation. **Tested at layer 6 (D21):** ReLU inconclusive (L0 confound), TopK/JumpReLU not supported.
* **H9:** Fine-tuning improves causal preservation. **Tested at L8 final (E3, D28):** Method A no; output-KL not creditably (gains only with L0 drift; TopK unchanged).
* **H7:** SAE-based interventions can have causal effects distinguishable from controls. All tested were null at layer 6. **R2 (D22), the first genuine SAE-in-the-loop clamp, is also not supported at layer 6 (validation): the edits are adverse rather than null.**

## D29. GPT-2 improvement phase: design choices and outcomes (2026-11-10)
* **Approval:** the user's full execution approval for GPT-2 improvements (R1, R2 and other justified experiments;
  M5 once frozen). Gemma deferred: inspect and plan only.
* **Choices made before results:**
  * modern JumpReLU abandoned in the pilot (L0 would not settle; disclosed);
  * R5 DAS initialised at PCA after the smoke test showed random init under-fits PCA (disclosed);
  * E2-H3 PCA matching clarified as M2's OWT-validation NMSE (the code reuses M2 unchanged);
  * the M5 SAE list contains every SAE of each family (no selection).
* **Outcomes:**
  * **R2 is adopted as the improved GPT-2 SAE set.** Reconstruction and behaviour are confirmed on TEST; the causal
    and representation gains are validation-only.
  * **R4 (KL at fixed L0) is a behaviour-only tool.** Its behaviour gain is confirmed on TEST, but it costs causal
    preservation at L11 (validation), so it is not the default SAE for causal or information analyses.
  * **Information preservation is not improved** relative to PCA by any method. This remains a headline negative
    result.
* **Operational rule from 2026-11-10 (user):** one memory-heavy job at a time on the 15 GB host. Wait loops must not
  use a `pgrep -f` pattern that appears in their own command line.

## D30. Methods to carry forward to Gemma-2-2B (recommendation, 2026-11-10)
* **Carry forward:**
  * the evaluation ladder: reconstruction, splicing with noise reference, M2 readers with PCA and noise references,
    RLM-RP, M4 decomposition, R5 skyline;
  * the random-decoder baseline;
  * an external public anchor: Gemma Scope, in place of res-jb;
  * TopK/BatchTopK SAEs trained with the R2 recipe (AuxK, decoder-transposed init, scale folding, streaming), if
    compute allows;
  * the KL fine-tune only as a behaviour-only add-on at fixed L0.
* **Do not carry forward:**
  * the in-house sigmoid-surrogate JumpReLU;
  * the L1-controller ReLU as a primary architecture (worst reconstruction and behaviour, PCA regressions);
  * output-KL with a drifting L0.
* **Feasibility on this host (inspected; not run):**
  * Gemma-2-2B bf16 in Hugging Face transformers runs at about 7.7k tokens/s with 7.3 GB of VRAM. TransformerLens'
    CPU-side load exceeds the 15 GB of host RAM, so the ladder needs an HF-hook path.
  * One middle-layer TopK SAE (d_model 2304, width 16k) on 100M tokens is about 3.6 h of activation generation per
    pass. Train several SAEs per pass from one activation stream, as R2 did.
  * Use Gemma Scope SAEs at the matching layer as the external anchor first: this needs no training.

## D31. M5 evidence label and protocol-freezing practice (2026-11-10, final defense audit)
* **M5 label.** M5 results are cited as "CONFIRMED TEST RESULT (previously exposed test rows; exposure limited to
  the old reference SAEs)".
  * The OWT and GoT test rows were scored for the old SAEs before M5 (README §10).
  * No R2/R4 decision used them.
* **Protocol freezing.** Future frozen protocols are written to a separate file, and committed before the run, so that
  the pre-registration hash stays verifiable after results are added.
* **Erratum, recorded rather than edited:** the M5 README L0 maximum is 133.39, not 133.2.

## D32. Build the RepE-LM TEST split now; evaluate it only after the user approves (2026-11-13, remediation S3)
* **Data built:** `data/repe/repe_lm_test/` (86 statements, 2,408 rows) from the 90 `persona_pairs.csv` TEST statements,
  using M1's builder unchanged.
  * Same-label near-duplicates of train/validation (4) are dropped.
  * New test-only wording is added (T7, T8, N3, N4). It was written before any test evaluation, so cross-wording
    generalisation can be tested on wording no decision has seen.
* **Protocol frozen:** `results/repe/repe_lm_test_protocol/preregistration.json`, with hypotheses HT1–HT4, the R2 rule,
  Holm correction across 12 cells, and positive and negative controls.
  * The evaluation script must be smoke-tested on validation and its hash added before it reads a test row.
* **Not executed.** About 1–2 GPU hours are needed, and a confirmatory test is a decision for the user.
* **Disclosed:** the source statements were scored once in the retired, length-confounded RepE task.

## D33. Interpretation of the M4 / R2-clamp "SAE-in-the-loop" interventions (2026-11-13, remediation S5)
* Every encode → edit → decode patch that keeps the error term equals x + W_dec(z′(x) − z(x)). That is the raw
  activation plus an input-dependent vector, and a 3,072- or 12,288-latent dictionary spans ℝ⁷⁶⁸.
  * M4's T_rec and its top-k patches therefore show that the SAE **selects** effect-carrying directions, tested
    against the random-feature and isotropic controls.
  * They do not show that the effect is **mediated** by the SAE.
* The paper's paragraph is renamed "SAE-selected feature patches (error term kept)" and states this argument.
* Review #8's SAE-in-the-loop requirement is therefore **PARTIALLY RESOLVED**. A distinct intervention is proposed:
  an edit at layer ℓ with SAEs spliced at later layers, or a run on x̂ without the error term compared against the
  spliced baseline. The layer-6 R2-clamp (input-dependent clamps, with and without the error term) was NOT SUPPORTED.

## D34. PCA-CM changes the "SAEs never beat PCA" conclusion; the revision waits for the user (2026-11-13)
* The pre-registered interpretation I2 was triggered (`results/shared/pca_capacity_matched/README.md` §2.3).
  * At equal error, PCA (298–519 components) keeps as much or more GoT truth information: replicated on validation.
  * At an equal per-token active budget (k = 128) or task-L0-matched k, trained SAEs are never worse than PCA, and
    ReLU-family SAEs are better in some cells.
* **Decision:** the paper's abstract, conclusion and §improvement, and the curated docs, are **not** revised in this
  task. The user's instruction requires stopping before a conclusion change.
* **Proposed wording** (for the user to approve): "At equal reconstruction error, PCA keeps at least as much truth
  information as the SAEs; at an equal per-token active budget, SAEs are never worse than PCA and ReLU-family SAEs
  are better in 3 of 20 layer–family cells (validation)."
* **Outcome (2026-11-13): APPROVED by the user with a refinement.** Adopted wording: "At matched reconstruction error,
  PCA generally preserves more truth-related information than the evaluated SAEs. However, this comparison allows PCA
  substantially more dimensions than the SAE's active-feature budget. Under a fixed 128-dimensional PCA comparison,
  trained SAEs are better under the study's comparison criterion in 3 of 20 comparisons and not worse in the
  remaining 17. When PCA dimensionality is matched to task-specific SAE activity, SAEs are better in 8 of 20
  comparisons and not worse in the remaining 12. These findings are based on validation data and demonstrate that
  conclusions about representational faithfulness depend on the comparison constraint."
  * The user asked that "significantly better" be used only if the decision rules support significance. **They do
    not**: the rule (every seed's paired 95 % bootstrap CI excludes 0) is per cell, with no correction across the 40
    capacity-matched cells. An approximate post-hoc Holm check (`scripts/shared/pca_cm_posthoc_multiplicity.py`) keeps
    1 of the 3 k = 128 cells (R2 ReLU L11) and 4 of the 8 task-matched cells. So "better under the study's comparison
    criterion" is used, and "not worse" replaces "not significantly worse" (no cell and no individual seed favours
    PCA).
  * The original equal-error results are kept and presented alongside the capacity-matched ones. No test row was
    re-read.

## D35. "No truth gain" is reported as a bound; the L8 causal gain as borderline (2026-11-13)
* From the sensitivity analysis (`results/shared/power_sensitivity/`): between-family GoT comparisons cannot reliably
  detect changes smaller than about 0.05–0.09 AUROC (observed ≤ 0.025); one SAE's M4 preservation at L8 is resolved
  only to 0.054–0.107.
* **Decision (user-approved):** the paper and docs state the GoT null as "no change larger than about 0.05–0.09
  AUROC", not as preservation. The R2 L8 causal gain (+0.101, VAL) is reported as borderline, pending the RepE-LM test
  protocol. The A2 "regressions" are reported as not robust. No result or verdict in the committed outputs changes.

## D36. How the H1-TEST outcome is reported (2026-11-13). Status: ADOPTED; paper revision APPROVED by the user and DONE (2026-11-13)
* **Source:** `results/repe/repe_lm_test_evaluation/` (single TEST pass, Holm over 13 cells, rule frozen in the addendum).
* **Adopted now (follows the pre-registered rule; no discretion):**
  * HT2 is reported as a CONFIRMED TEST RESULT.
  * HT1 and HT3 are reported as NOT CONFIRMED.
  * Their C1 cells are reported as confirmed cells.
  * HT1-C3 is reported as "improved in 9/9 pairs on TEST, not significant after Holm (p = 0.058)". It is never
    rounded up to confirmed.
* **Proposed (needs the user's approval before the paper or curated docs change):** D35's "borderline, pending the
  test" becomes:
  * "confirmed on held-out TEST for TopK (C1: 0.602 → 0.729)";
  * "not confirmed for BatchTopK vs JumpReLU (C3)".
* **Not changed:** no threshold, cell count or rule is altered after seeing TEST; the N3/N4 floor failure is
  disclosed, not explained away.
* **Approval and execution (2026-11-13):** the user wrote "Yes, proceed with D36". Done:
  * The paper (abstract, setup, causal results, R2, the new H1-TEST subsection and Table `t_h1test`, statistics,
    discussion, limitations, conclusion, appendix) and the curated docs are revised.
  * The failed N3/N4 control is a separate limitation.
  * The 90 secondary comparisons are labelled exploratory everywhere they appear.
  * `src/tests/test_h1_test_paper_claims.py` checks every quoted number against the artifacts.
  * No experiment was re-run and no metric changed.

## D37. Checkpoint backup scope (2026-11-13)
* **Done:** a verified same-volume backup of all 656 checkpoints (21.7 GB) in `../backups/2026-11-13_checkpoints/`.
  The manifest and verification are copied to `docs/backups/`.
* **Blocked:** no off-machine destination or credentials exist on the host (no AWS CLI or keys). The second NVMe is
  instance-store, which is ephemeral. Uploading 21.7 GB through a chat connector is not feasible.
* **Decision:** the off-machine copy waits for the user to name a destination. It remains P0.

## D38. How the AUDIT-SENS results change the reported conclusions (2026-11-13). Status: ADOPTED (user instruction: "update conclusions to reflect the actual evidence")
* **Frozen results are not altered.**
  * H1-TEST and M5 keep their pre-registered verdicts and tables.
  * Every place that cites them now adds the AUDIT-SENS qualifier.
* **Wording rules:**
  * HT1-C1 and HT3-C1 are "confirmed under the pre-registered seed-only rule; not robust to statement-sampling
    uncertainty (sensitivity analysis)". They are never cited without the qualifier.
  * HT2 is "confirmed, and robust to statement-sampling uncertainty".
  * The persona loss is "generic one-word instruction information". The earlier "persona-specific" sentence is
    withdrawn.
  * The frozen-probe GoT collapse is "partly generic reader sensitivity and partly label-aligned SAE error". The earlier
    "largely perturbation sensitivity" is withdrawn.
  * The pooled GoT task at L0–4 is not evidence that truth is absent there.
  * Architecture rankings on task metrics are reported only with a statement that they are not robust to task-level L0.
* **Not done** (training or new data; needs approval):
  * R2-recipe SAEs at more layers;
  * a valid JumpReLU;
  * Gemma Scope baselines;
  * a behavioural honesty task;
  * per-polarity GoT re-analysis with SAEs (a new pre-registration is needed; S4 used original activations only).

## D39. TSP pilot reading and next step (2026-11-14)
* **Reported verdict:** the frozen rule gives NOT SUPPORTED, and that is how it is reported. Effect sizes are given
  with CIs, and the direction is described as consistent across seeds and specific to the truth subspace.
* **No claim:** we do not claim "truth preservation improved". The gain from GoT-A replay alone (ctrl vs base) is
  reported as distribution-matched training, not as a truth-specific method.
* **Interruption handling:** a separate recovery wrapper trained only the missing runs, after bit-identical
  reproduction of a completed run. The frozen scripts were not edited. The 9 completed checkpoints were kept.
* **Next:** any follow-up (more seeds, held-out templates, other layers or architectures) needs a new
  pre-registration. The pilot's λ, rank and steps are not tuned on these validation results.

## D40. TSP verification: what changes and what does not (2026-11-14)
* **Verdict unchanged:** NOT SUPPORTED stands. The post-hoc Satterthwaite p values (0.023 / 0.0025) are **not**
  grounds to revise it. They are reported only as the reason the next pre-registration changes its interval method.
* **Corrected claims:**
  * "Specific to the truth subspace" applies to the frozen readout only.
  * "No cost" applies against ctrl. GoT replay costs ≈ 1.6–2 % (NMSE, KL, ΔCE) against R2 base.
  * The frozen gap and error-only AUROC are readout metrics for TSP: S contains 58 % of the evaluator's direction.
* **Next design principle:** statement groups, not seeds, are the bottleneck.
  * Replicate with cross-fitted cities (≈ 635 groups) and fresh held-out GoT families (DEV/TEST split before any
    activation is computed).
  * Add a label-permuted S control.
  * Pre-register Satterthwaite df and CI-based non-inferiority.
  * No loss or architecture change before the replication. A multi-family "universal" S is a later, separately
    pre-registered follow-up.
* **Approval required** for Phase 0 (public data download) and Phase 1 (training).


## D41. TSP-R Phase 0 outcome: Q2 is not decisively testable with public GoT at GPT-2 L8 (2026-11-15). Status: PROPOSED, awaiting the user's decision

* **Facts** (DEV only):
  * Of the five independent fresh families, only companies (0.791) and common_claim (0.708) pass the frozen
    eligibility rule.
  * common_claim has no reconstruction headroom (R2 base gap ≈ 0.001).
  * companies' conservative MDE (≈ 0.033) exceeds the pilot's in-distribution effect (0.020).
  * Random-effects pooling with k = 2 is infeasible.
* **Decided now:**
  * The frozen eligibility rule stands. counterfact is ineligible (0.625 across relations), even though it would
    pass under entity-level CV (0.786). Switching units post hoc would be a deviation.
  * sp_en is ineligible: GPT-2 Small has no truth signal for it.
  * conj/disj are dropped: they are not independent and too small.
* **Proposed amendment to the TSP-R pre-registration** (before Phase 1, for the user):
  1. P1 (cross-fitted cities, re-fit logistic gap, tsp vs ctrl, plus shuf/rand specificity) is the **sole
     confirmatory primary**.
  2. companies TEST is a pre-registered secondary: an estimate with its CI and a stated MDE of ≈ 0.03. It does not
     count as confirmatory.
  3. common_claim is a pre-registered no-headroom / enhancement check: TSP should not move its gap; a negative gap
     beyond −0.01 counts as distortion.
  4. counterfact held-out-relation TEST is optional, labelled EXPLORATORY. It adds ≈ 9 CPU-hours of readers.
  5. The random-effects P2 is withdrawn.
* **Rejected options:**
  * Re-splitting counterfact by entity: TEST is locked by relation.
  * Lowering the 0.70 threshold: it is frozen.


## D42. TSP-R Phase 1 frozen design (2026-11-15). Status: DECIDED (pre-registration frozen; launch awaits approval)

* D41 was approved in principle by the user; 8 seeds and 120 runs.
* Cities (cross-fitted) are the sole confirmatory dataset. The random-effects P2 is withdrawn.
* **Hypothesis family:** H1 tsp < ctrl as the gate, then Holm over H2 tsp < shuf and H3 tsp < rand. pca is outside
  the family.
* **Uncertainty and threshold:** Satterthwaite df with a seed + city-bootstrap se; SESOI 0.010 with TOST
  equivalence.
* **Labels:** information-vs-readout and distortion qualifiers.
* **Guards:** non-inferiority vs ctrl with one-sided 95 % bounds.
* **New base seeds** reuse script 17's `train()` with SEEDS patched and write to a new folder. They are exchangeable
  with the originals: same recipe and tokens, a different random batch order. Old-vs-new seeds is a pre-registered
  sensitivity check.
* **The evaluation code is frozen separately,** before any Phase 1 SAE is evaluated. The metric definitions are
  already fixed here.


## D43. TSP-R verdict and what may be claimed (2026-11-15). Status: DECIDED (by the frozen rules)

* **Claimable (validation, confirmatory, in-distribution):**
  * On GPT-2 Small L8 R2 TopK SAEs, the truth-subspace penalty reduces the loss of re-fittable linear truth
    information on held-out city–country statements by 0.021 AUROC versus the identical fine-tune without the penalty.
  * The effect is truth-specific (vs label-shuffled and random subspaces) and also visible in MLP readers (smaller).
  * It costs nothing measurable in OWT reconstruction, ΔCE or KL.
* **Must be qualified:** TRADE-OFF by the frozen rule. Persona and M4 non-inferiority was not demonstrated because
  the bounds are imprecise (estimates ≈ 0).
* **Not claimable:**
  * transfer to other truth families: companies, common_claim and counterfact relations are all null;
  * "general truth preservation";
  * any frozen-gap result as information preservation (readout metric).
* **Lesson for future guards:** behavioural guards need per-item outputs to support a paired statement bootstrap.
  Ladder-CI approximations are too conservative to establish non-inferiority.
* **Next candidates** (not started; each needs its own pre-registration):
  * a multi-family "universal" truth subspace aimed at transfer (Q2);
  * a precision rerun of the persona/M4 guards with per-item paired bootstraps on the existing checkpoints.

## D44. Final-week cross-model protocol and execution rules (2026-11-21). Status: DECIDED (user approved the plan)
* **Fair comparison.** GPT-2 vs Gemma claims use only within-model-normalised metrics, at depth-matched sites, on the
  same text, with the same controls and the same paired document bootstrap. Raw nats are never ranked.
* **Separate tracks:** public-vs-public, in-house-vs-public, matched strata (L0/d and d_sae/d within ±15 %) and
  within-model. **No architecture-matched claim is made:** no public GPT-2 JumpReLU exists, and the in-house JumpReLU
  is not valid.
* **Hook conventions:** each SAE family keeps its own training convention (GPT-2 centred TL residual, Gemma raw HF
  residual). G-HOOK shows behaviour is identical.
* **Ceiling rule (XT-G A2):** a task with original AUROC ≥ 0.99 is flagged and is not used for cross-model truth
  claims. Harder fresh families (DEV) and d′ are added for both models.
* **TI-1 eligibility:** the TSP-R Phase 0 rule (original AUROC ≥ 0.70) excludes GPT-2 L1.
* **Execution:**
  * at most two Gemma-sized GPU jobs at once, with layer-major SAE loading;
  * a non-destructive disk guard (pause, never delete);
  * one focused PR per completed and validated phase, based on `sae-research` (stacked), never merged without the
    user's approval.

## 2026-11-23 (recovery session; no D number, to avoid colliding with the PR stack's numbering)
* **Working repository and branch.**
  * `sae-project/fall-2026-group4` (this repository, next to `../venv`) stays the working repository.
  * Today's work is committed on `sae-research` (user, Option 1).
  * The `final-week/p*` PR stack is not modified.
* **TI-2 analysis deviation.** The frozen analysis is re-run through an import wrapper that allows pickle for the
  pipeline's own `eval/orig.npz` only. No estimand, threshold or input changes, and the frozen script is untouched.
  Alternative rejected: editing the frozen script, because its hash is checked.
* **No retraining** of the lost TI-2/TI-4 checkpoints. The evaluation outputs are sufficient for the pre-registered
  analyses (user).
* **TI-3 stays withdrawn** (user); it is not re-run.
* **Final-day scope (user):** Phase 2 + XC-3 + XC-2. XC-1, XF and M4 are deferred.
  * Gemma decisions use VALIDATION only.
  * TEST is reported once, with prior exposure disclosed.
* **XC-3 task choice:**
  * RepE-LM persona (not GoT judgment): both models pass its gates (K1, K7), and E is normalised within each model.
  * The strength grid equals `scripts/repe/13`'s.
  * H3 (retention) is tested only where H1 holds.
  * All of this was fixed after a TRAIN-only smoke test and before freezing.
* **XC-2 numerics:**
  * Every Gemma forward has an identical batch shape, and the baseline uses the same path.
  * The first smoke showed bf16 batch-shape effects of up to 0.34 nats per pair.
  * Fixed before freezing; the path checks are hard stops.
* **Storage:** the instance-store disk is not formatted (user). All outputs go to EBS; jobs start only at ≥ 60 GB free
  and stop below 42 GB.
* **XC-2 reading (2026-11-23).** Because norm-matched random directions lower Gemma's margin at the selected
  strengths, absolute effect CIs (> 0) are cited alongside the control-based p-values. The pre-registered BH tests are
  reported unchanged. The GPT-2 vs Gemma contrast stays descriptive.
* **XF (2026-11-23).**
  * **Fresh family design:** cities of 200k–500k from geonames, disjoint from GoT cities, with DEV 40 % / TEST 60 % by
    city.
  * **Ceiling rule:** added after the TRAIN-only smoke, before any fresh row was embedded.
  * **Reading of the TSP claim:** after XF, the truth-subspace fine-tune's advantage over its reconstruction-only
    control is a VALIDATION-only finding that **did not replicate on a fresh TEST population**. Cite it as
    "validation effect, failed fresh-TEST replication".
* **XC-1 and M4-G (2026-11-23).**
  * **Reduced Gemma grid:** 9 layers × 5 groups, with TRAIN screening on 100 statements.
  * **M4-G site rules:** both pre-registered before XC-1 read VALIDATION.
  * **H4c omitted.**
  * **Order:** XF first, then XC-1, then M4-G (user).
  * **XC-1 outcome (2026-11-23):** selected L0 slot and L23 final, both confirmed on VALIDATION, so the stop rule is
    not triggered and M4-G runs. The mid site is L12 final by the pre-registered M4-G rule. No rule was changed after
    any VALIDATION number was seen.

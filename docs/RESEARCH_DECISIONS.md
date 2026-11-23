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
* **The Pile** (proposal): only EDA was done, on a team branch. It is not used in experiments. *(Updated 2026-11-12: used as the Gemma reconstruction and splicing dataset, D44.)*

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
* **H1:** Wider SAEs or longer training close the deep-layer reconstruction and behaviour gap. Not tested; validation EV was still rising at 15k. *(2026-11-14: for Gemma's published SAEs, width at matched L0 is pre-registered as R3's H2 (D46); completed, validation (Tier A): supported in the 4 bands that formed.)*
* **H2:** A converged EV-vs-L0 frontier changes the architecture ranking at low L0. Not tested.
* **H3:** Method A's MLP-probe gain replicates with more seeds and at other layers. **Tested at layers 3, 8, 9 (D26):** the direction replicates in 9/9 cells and the criterion is met in 4/9. Re-trained readers show no gain. More seeds are not tested.
* **H4:** Combining Method A with a small KL weight keeps the truth gains and the behaviour gains. **Tested for TopK at layer 6 (D19): inconclusive.** Behaviour gains are kept (≈85 % at w = 0.25), and truth is non-inferior, but A and B themselves don't differ in truth. At layers 3, 8, 9 (D26), output-KL never improves truth over Method A and regresses at L3 ReLU/TopK. The ReLU/JumpReLU L0 drift persists at depth.
* **H5:** Layer-0 JumpReLU failure can be fixed by threshold learning-rate control or dead-latent revival. Not tested.
* **H6:** Results transfer to Gemma-2-2B SAEs. *(2026-11-12, M6, D44, VALIDATION: the behavioural-splicing finding replicates in direction with Gemma Scope SAEs at layers 5/12/19 (H-G1 supported at all three); reconstruction measured descriptively. Gemma information (truth/persona) and causal preservation remain untested.)*
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

> **Numbering (2026-11-17, agreed on PR #31):** D36–D43 are reserved for PR #35 (`sae-research`), whose frozen,
> hash-pinned records cite them. M6 moved from D36 to D44, this PR's Method A decision from D37 to D45, D46 is
> reserved for R3 (PR #30), and D47 is the final-week record from PR #36 (merged into `main` 2026-11-21).

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

## D44. Gemma/Pile experiment design (M6): Gemma Scope SAEs at layers 5/12/19, Pile validation, 1024-token truncation (2026-11-12)
* **Numbering:** recorded as D21 on PR #28 before PR #27 was merged; renumbered D36 on 2026-11-15 because #27 already uses D21–D35; renumbered D44 on 2026-11-17 so that PR #35 keeps D36–D43 (M6: D21 → D36 → D44). Content unchanged.
* **Decision** (made by the user, Simbanegavi Simbarashe, after the completed Pile EDA): the next Gemma contribution is Pile → Gemma-2-2B activations → Gemma Scope SAE reconstruction → splicing evaluation (roadmap M6). No other new scientific objective. The Gemma GoT frozen-probe experiment is a stretch goal, started only after M6 is complete and only if the 2026-11-22 technical freeze is not at risk.
* **Role of The Pile:** Gemma's natural-text evaluation corpus for reconstruction and behaviour, analogous in role (not in content) to OWT for GPT-2.
* **Data:** `monology/pile-uncopyrighted/val.jsonl.zst` (revision and SHA-256 pinned). The EDA's parquet mirror has no validation split. The test file is never used.
* **Sample:** seed-42 permutation; 512 evaluation + 64 reference documents, disjoint by construction; per-document identifiers and text/token hashes recorded so the exact sample can be rebuilt.
* **Chunking:** BOS + first 1024 tokens, one window per document, no packing. **Reason:** the Gemma Scope training context length recorded by sae-lens is 1024; one window per document keeps documents independent for the bootstrap and keeps source labels clean. **Alternative:** packing contiguous windows (rejected: mixes sources inside a window).
* **Layers 5, 12, 19:** representative lower/middle/later depths of the 26-layer model, and the residual layers for which the release also provides the largest (1M) SAEs. **Explicitly not** a mapping to GPT-2 layers 5/6/9; an earlier proposal (Gemma layers 10/12/18 as "depth-matched" to GPT-2) was rejected by the user because no defensible basis for such a mapping exists. The GPT-2 exclusion of layer 3 applied only to the GoT probe experiment.
* **SAEs:** Gemma Scope residual width 16k, pinned IDs `layer_{5,12,19}/width_16k/average_l0_{68,82,73}` (what sae-lens `canonical` resolves to). No other width or L0 is substituted without asking the user.
* **Model and tokenizer revision (Amendment 1, 2026-11-12, before any run):** pinned to `google/gemma-2-2b` commit `c5ebcd40d208330abc697524c919956e692655cf` instead of being looked up at run time, so the pre-registration freezes it. A later mismatch is recorded as a protocol deviation.
* **Hook:** `resid_post`, because that is where these SAEs were trained. Our GPT-2 SAEs use `resid_pre` (D4); the two are conventions of their SAE families.
* **Consequences:** pretrained SAEs → no seeds and no L0 matching; uncertainty is a document bootstrap; any GPT-2 vs Gemma comparison is descriptive only (roadmap X2).
* **Outcome (2026-11-12):** M6 COMPLETED, VALIDATION (`results/gemma2_pile/gemma_scope_splicing/`, README §2). Gates V1–V5 pass; H-G1 supported at layers 5, 12 and 19. SAE-splice reconstruction error produced greater next-token behavioural damage than the norm-matched random-error control at all three pre-registered layers; the experiment does not identify the mechanism responsible. No numerical ranking against GPT-2 (model, hook, SAE family, width, L0, corpus and context differ).
* **Follow-up kept separate:** R3 (Gemma Scope width × L0 ladder) is PR #30 / `dhruv-gemma-r3`, COMPLETED, VALIDATION (Tier A); its own decision record lives on that branch and is not part of M6.

## D45. Method A layers 5/6/9 extension excluded from the final scope (user, 2026-11-16; numbered D37 until 2026-11-17)
* **Context:** this resolves the Instructor Review #9 scope-freeze item on the separate Method A multi-layer extension.
  The exact review text is not stored in the repository.
* **Decision:** the extension on branch `simba-method-a-multilayer` stays unmerged, exploratory validation work.
  * Commits: `28ad5cf`, `b64e6ed`, and `7f2557b` (results, on `origin` only).
  * It is excluded from the final manuscript and from the final scientific scope.
  * Its results are not cited and must not be used for final claims.
  * The branch and its artifacts are kept unchanged: not merged, renamed, regenerated or deleted.
* **Reasons:**
  * validation-only exploratory work;
  * the success criterion was met in 1/9 cells, and only layer-5 ReLU is genuinely new positive evidence;
  * layer 6 duplicates `method_a_layer6/`;
  * layer 9 disagrees with the IMP-L pipeline the manuscript uses, and the baseline discrepancy is unresolved;
  * provenance is weaker than IMP-L's;
  * there is no re-trained-reader analysis, which is central to the final Method A conclusion;
  * the manuscript's conclusion does not change;
  * reconciling the two pipelines would be new scope just before the 2026-11-22 technical freeze.
* **Consequences:**
  * The Method A evidence in the paper remains layer 6 plus IMP-L at layers 3, 8 and 9 (`multilayer_finetune/`).
  * No manuscript, results, experiment-manifest or checkpoint-registry change.

## D46. R3 design and hardware: Gemma Scope width × L0 ladder on Kaggle T4 x2 (2026-11-13 to 2026-11-14)
* **Numbering:** recorded as D22 on `dhruv-gemma-r3` before `main` was merged in; renumbered D37 on 2026-11-16 because `main` already uses D22 to D36 (M6 is D36); renumbered D46 on 2026-11-21 because D37 collides with #31 and #35 (numbering agreed on #30 by Aditi, 2026-11-17 and 2026-11-20).
* **Decision** (made by the user, Dhruv Rai): R3 evaluates 54 published Gemma Scope residual SAEs on M6's sample.
  * Tier A: 36 SAEs. Tier B: 18 SAEs (layer 12, widths 131k/262k/524k).
  * Same data, metrics, controls and determinism settings as M6.
  * The 1M width is excluded.
  * This is a new pre-registered experiment, not a substitution in M6. D44's (M6, numbered D36 until 2026-11-21) "no other width or L0" rule applies
    to M6's own SAEs, which are unchanged.
* **Hardware (R3 Amendment 1):** Kaggle's free GPU T4 x2 instead of an AWS A10G, because the AWS account's free plan
  has no GPU instances.
  * **Consequence:** the T4 vs A10G question (Q0) and the A10G rerun of M6 are dropped.
  * R3 runs on the same GPU class as M6. Gate V6 checks that comparability against M6's committed rows instead of
    assuming it.
  * Q0b asks how far T4 bf16 is from fp32 (layer 12, 16k SAE, 64 documents).
  * **Alternative:** paid A10G time (not available on the free plan).
* **Running within Kaggle limits:** each session is at most 12 h and the GPU quota is 30 h per week.
  * Tier A is split into runs K0–K3, each planned at ≤ 10 h. Tier B (K4–K6) runs only after Tier A passes its gates,
    only within the quota, and only if it can be documented before the 2026-11-22 freeze.
  * Each SAE writes its own folder, with `DONE.json` written last; a resumed run skips finished SAEs.
* **Resume check (gate V9, as changed by the user on 2026-11-13):** a resumed run passes if every recomputed value
  matches the stored one within relative difference 1e-6. The mean-ablation vector uses the same tolerance on the
  norm of the difference. Bit-identity is recorded separately and is not required. Only a larger difference stops
  the run.
  * **Reason:** a strict bit-for-bit rule could stop a valid run over harmless floating-point noise.
  * **Alternative:** bit-identity required (rejected by the user).
* **Credentials (gate V10):** the Hugging Face token comes from Kaggle Secrets into process memory only. All outputs
  are scanned for it before they leave Kaggle; any hit means nothing is downloaded and the token is revoked.
* **Clarification 1:** V6's active-latent agreement compares the chunked with the unchunked SAE on the same
  activations, because M6 saved no active-latent sets.
* **Scope:** the RepE-LM / K1–K2 part of the roadmap's R3 is split off as R3b, with its own pre-registration.
* **Deviations (2026-11-16, decided by the team; recorded before any Tier A result was looked at):**
  * **Deviation 1:** K1 took 7.64 h against 3.11 h projected. K3 is split into K3a (layer 12, widths 16k and 32k)
    and K3b (width 65k), so each run stays under the 10 h Kaggle limit.
    * **Alternative:** one K3 run, which would risk hitting the limit and losing the session.
  * **Deviation 2:** Tier B (layer 12, widths 131k/262k/524k; K4–K6) is cancelled. H1–H3 are tested on Tier A only.
    * **Reason:** the 2026-11-22 technical freeze and the 30 h weekly Kaggle quota.
    * **Consequence:** H2's width comparison stops at 65k; H3's Holm family is Tier A only.
    * **Alternative:** running Tier B after the freeze, which would leave it undocumented in the deliverable.
  * Recorded in the R3 README ("Deviations") and in `preregistration.json` (`deviations`). Section 1 is not edited.
* **Outcome (2026-11-17):** COMPLETED, VALIDATION, Tier A only. H1 SUPPORTED in 7/7 Tier A cells; H2 SUPPORTED in 4/4 bands that formed (none at layer 5); H3 HOLDS ACROSS THE LADDER (36/36, no reversal); Q0b headline WITHIN MARGIN (ΔCE_SAE mean |T4 bf16 − fp32| 0.00267 [0.00218, 0.00319] < 0.005), top-1 agreement BEYOND MARGIN; V6 M6 REPRODUCED WITHIN MARGIN at all three layers. Details in the R3 README,
  Section 2.
* **Figures as SVG (2026-11-17, decided by the user per Instructor Review #9):** script 07 is pre-registered code,
  so it is not edited; `scripts/gemma/08_r3_vector_figures.py` runs it unchanged and redirects PNG saves to SVG.
  **Alternative:** PDF, which could not be written on the machine used (fontTools DLL blocked by Windows
  Application Control).
* **Committed outputs:** the files listed in Section 1 (ladder tables incl. the 6.3 MB `per_doc_metrics.csv`,
  approved by the user; `fp32_reference/`; `precision_t4_vs_fp32/`; `figures/`). The 166 MB per-SAE folders and
  per-run records stay in the Kaggle outputs (not listed in Section 1).

## D47. Final-week cross-model protocol and execution rules (2026-11-21). Status: DECIDED (user approved the plan)
* **Numbering:** recorded as D44 in PR #36; renumbered D47 on 2026-11-21 to follow the team numbering plan (#35 D36–D43, #31 D44–D45, #30 D46). Content unchanged.
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

## D48. TI-1 verdict and what may be claimed (2026-11-21). Status: DECIDED (by the frozen TI-1 rule)
* **Verdict:** TSP GENERALISES across layer (L8 → L11) and SAE type (TopK, BatchTopK, ReLU) for GoT cities in GPT-2 Small: 3/3 eligible cells SUPPORTED after Holm, cost guard met.
* **May be claimed (VALIDATION):** "Truth-subspace-preserving fine-tuning reduces the cities refit-probe gap by 0.015–0.042 AUROC vs a reconstruction-only control at matched L0, in every tested layer/architecture, at ≤ +2.5 % OWT NMSE; the gain is truth-specific (beats a random subspace)."
* **May not be claimed:** transfer to other truth families (not tested here; TSP-R found none); any Gemma result; frozen-probe gains as evidence (circular).
* **Consequence:** TI-1 positive, so TI-2 (multi-family) and TI-G (Gemma transfer) stay first in the improvement queue; both wait for disk.

## D49. How SIL is reported (2026-11-21). Status: DECIDED (follows the pre-registered rule; no discretion on the label)
* **Label:** the pre-registered rule gives MEDIATED for every family. Because the pre-registered random-decoder negative control is also MEDIATED, SIL is reported as "MEDIATED by the pre-registered rule; the rule is not specific (negative control passes); SAE-specific mediation is INCONCLUSIVE".
* **Reported descriptively:** D2 retention (0.82–0.92 for TopK/BatchTopK/JumpReLU vs ≈ 0.5 for ReLU and the control).
* **Not done:** no post-hoc specificity rule is applied. A retention-minus-control test would need a new pre-registration.

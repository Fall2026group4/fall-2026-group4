# Final Capstone Presentation Storyboard

## Purpose

This storyboard defines the narrative for the **whole capstone project**. The presentation is one integrated scientific study, not separate contributor workstreams.

**Main research question**

> To what extent do Sparse Autoencoders (SAEs) faithfully preserve the information and behaviour of language models beyond simply reconstructing activations well?

**Core narrative**

Reconstruction → Information → Behaviour → Causal Faithfulness → Improvement → Generalisation

**Models:** GPT-2 Small; Gemma-2-2B  
**Datasets:** OpenWebText; Geometry of Truth (GoT); RepE / RepE-LM; The Pile  
**SAE families:** ReLU; TopK; JumpReLU

**Presentation rule:** distinguish **TEST**, **CONFIRMED TEST**, **VALIDATION**, **EXPLORATORY**, and **PRE-REGISTERED ONLY** evidence. Do not present R3 as a completed result unless authoritative outputs exist.

---

## Recommended Main Talk — 17 Slides

| # | Slide title | Scientific purpose | Evidence / visual | Main takeaway | Current deck mapping |
|---|---|---|---|---|---|
| 1 | **Do Sparse Autoencoders Preserve What a Language Model Uses?** | Introduce the project and central question. | Clean title slide; subtitle with GPT-2 Small + Gemma-2-2B. | The study evaluates SAE faithfulness beyond reconstruction error. | Current 1 |
| 2 | **Why Reconstruction Alone May Not Be Enough** | Motivate the problem and state the research question. | Simple SAE encode → sparse code → decode → model behaviour schematic. | A reconstruction can look numerically good while changing information, behaviour, or causal effects. | Merge current 1–2 |
| 3 | **One Study, Four Levels of Faithfulness** | Present the evaluation framework and study scope. | Faithfulness ladder: Reconstruction → Information → Behaviour → Causal; models/datasets beneath it. | Faithfulness is multidimensional and must be measured at several levels. | Merge current 3–4 |
| 4 | **RQ1 — How Well Do SAEs Reconstruct GPT-2 Activations?** | Establish the reconstruction baseline across layers and architectures. | GPT-2 OWT reconstruction figure; TEST label. | Reconstruction quality varies strongly by layer and architecture; no single architecture wins everywhere. | Merge current 5–6 |
| 5 | **Before Measuring Information, We Fixed the Persona Benchmark** | Explain why RepE-LM replaced the earlier length-confounded persona setup. | Compact benchmark-correction graphic/table; VALIDATION. | The equal-length RepE-LM task removes a shortcut that could otherwise distort information-faithfulness claims. | Current 7 |
| 6 | **RQ2 — Does the Reconstruction Preserve Information?** | Test information preservation with frozen, retrained, nonlinear and PCA references. | M2 information-faithfulness figure/table; VALIDATION. | Good reconstruction does not guarantee preserved task information; PCA can match or outperform the SAE representation. | Current 8 |
| 7 | **RQ3 — Reconstruction Error Predicts Behaviour Better Than Truth Loss** | Relate reconstruction error to downstream behavioural damage. | GPT-2 reconstruction-vs-behaviour figure; M0 result; VALIDATION. | Within layer, reconstruction error robustly predicts splice damage, but not task-truth loss. | Current 9 |
| 8 | **Gemma-2-2B Replicates the Behavioural Concern** | Test whether the behavioural finding generalises beyond GPT-2. | M6 Gemma splicing + H-G1 visual; VALIDATION. | At layers 5, 12 and 19, Gemma Scope SAE reconstruction error causes more next-token damage than norm-matched random error. | Merge current 10–11 |
| 9 | **RQ4 — Where Does GPT-2 Carry the Persona Effect?** | Identify causal sites before evaluating SAE causal preservation. | M3 causal-site sweep; VALIDATION. | The causal effect moves from the persona slot in early layers toward the final token in later layers. | Current 12 |
| 10 | **SAE Reconstructions Preserve Most — But Not All — Causal Effect** | Measure causal preservation directly. | M4 decomposition figure; VALIDATION. | Reconstruction preserves about 93–97% at the early site but only about 62–68% at later sites; reconstruction is not fully causally faithful. | Current 13 |
| 11 | **RQ5 — Truth-Focused Fine-Tuning Did Not Reliably Solve the Problem** | Summarise the improvement approaches that failed or produced only partial gains. | One consolidated comparison of Method A / IMP-L / E3; VALIDATION. | Frozen-probe improvements did not translate into robust retrained-reader or causal improvements. | Merge current 14–16 |
| 12 | **What Worked: More Data and Wider SAEs** | Show the strongest confirmed improvement result. | M5 / R2 confirmed-test editable charts; CONFIRMED TEST. | Scaling the SAE training/data substantially improves reconstruction and behavioural faithfulness, while information preservation still does not automatically follow. | Current 17 |
| 13 | **Behaviour-Targeted Training Helps Behaviour — With Trade-offs** | Present output-KL training without overstating it as universal improvement. | R4 / output-KL summary; MIXED. | Optimising downstream behaviour can improve behavioural metrics without producing corresponding information or causal gains. | Current 18 |
| 14 | **The Main Conclusions Hold for an External Public SAE** | Check whether findings depend only on the team's in-house SAEs. | E2 public res-jb comparison; VALIDATION. | The headline pattern is not limited to the team's own training recipe. | Current 19 |
| 15 | **Integrated Answer: Faithfulness Is Not One Number** | Synthesize all experimental evidence. | 4-column evidence matrix: Reconstruction / Information / Behaviour / Causal, plus GPT-2 and Gemma rows. | Reconstruction quality is necessary evidence, but insufficient by itself to establish SAE faithfulness. | Current 20 |
| 16 | **Limitations and Ongoing Work** | State what the study does not establish and what remains optional. | Short limitations panel + R3 note. | R3 is PRE-REGISTERED ONLY and optional; the core study does not depend on it. Avoid direct cross-model numerical ranking where protocols differ. | Merge current 21–22 |
| 17 | **Conclusion — Measure Faithfulness by Category, Not by MSE Alone** | End with one memorable scientific conclusion. | Minimal closing slide with 3 short takeaways. | Information, behavioural and causal faithfulness require separate evaluation even when reconstruction improves. | Current 23 |

---

## Backup Slides

| Backup | Title | Purpose | Current deck mapping |
|---|---|---|---|
| B1 | **Architecture Detail Across GPT-2 Layers** | Extra architecture-by-layer reconstruction detail for questions. | Expanded material from current 6 |
| B2 | **Gemma-2-2B Truth-Direction Steering** | Exploratory result only; use only if asked. | Current 24 |
| B3 | **Reproducibility and Provenance** | Branches, commits, evidence labels and result-source provenance. | Current 25 |

---

## Slides to Merge or Move

The current 25-slide deck is scientifically rich, but the main talk should be tightened.

- Current **1 + 2** → combine into motivation/research-question opening.
- Current **3 + 4** → combine into study scope + faithfulness ladder.
- Current **5 + 6** → one main reconstruction slide; detailed architecture comparison moves to backup.
- Current **10 + 11** → one Gemma M6 generalisation slide.
- Current **14 + 15 + 16** → one consolidated “what did not reliably work” improvement slide.
- Current **21 + 22** → one limitations + ongoing-work slide.
- Current **24 + 25** remain backup.

This reduces the live talk from 23 main slides to **17 main slides**, while preserving the scientific storyline.

---

## Evidence Discipline

1. Use only authoritative saved results.
2. Keep all TEST / VALIDATION / EXPLORATORY labels visible.
3. Do not reuse historical RepE results that were superseded by RepE-LM for final representation claims.
4. Do not present R3 as a result unless completed authoritative files exist.
5. Do not make direct quantitative GPT-2 vs Gemma rankings where experimental protocols differ.
6. Keep improvement claims property-specific:
   - reconstruction improvement,
   - information preservation,
   - behavioural improvement,
   - causal preservation.
7. Use the final integrated conclusion only to the extent supported by the saved evidence.

---

## Core Takeaway for the Audience

> **Better reconstruction does not automatically mean a more faithful SAE.**  
> Across GPT-2 and Gemma evidence, reconstruction, information preservation, behavioural preservation and causal preservation behave differently and therefore need to be evaluated separately.

---

## Editing Workflow

1. Use this storyboard to edit `presentation/final_capstone_presentation.pptx`.
2. Preserve the current editable charts/tables and authoritative figures.
3. Move excess technical detail to backup rather than deleting useful evidence.
4. Review slide 1 course/date information manually.
5. Update reproducibility/commit references after PR integration if necessary.
6. Only after manual review, stage and commit the presentation files.

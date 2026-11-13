# DRAFTS FOR THE USER TO REVIEW: NOT POSTED (2026-11-13)

Both texts speak for Aditi and touch team decisions, so neither has been posted. Edit them, then post them yourself
(or ask for them to be posted).

---

## A. Reply on issue #25 (Instructor Review #8): GPT-2 part

> **Aditi, GPT-2 Small work (branch `sae-research`, PR #27; not yet on `main`).**
>
> **Superseded findings from the 11-08 summary.** Four statements in Review #8's summary of our results have been
> corrected by later, pre-registered analyses:
> 1. "Truth displaced, not erased": most of the frozen-probe drop is reproduced by norm-matched noise. A re-fitted
>    reader loses a real but small 0.03–0.11 AUROC (M2 VAL; M5 TEST).
> 2. "ρ = 0.86 on RepE": a pooling artefact across layers. Within a layer, reconstruction error predicts splice
>    damage, not truth-readout loss (M0).
> 3. "SAE errors 1.0–2.6× worse than noise": 0.76–2.89 across layers for the old SAEs (M0). On TEST at L8/L11:
>    1.13–2.24 for the improved SAEs, and 0.98 for the KL fine-tune at L8.
> 4. "Raw honesty direction p = 0.010": it came from the length-confounded RepE task. That task is now historical,
>    replaced by an equal-length persona task (RepE-LM).
>
> **Verified (single pre-registered TEST pass, M5).**
> * Data-scaled SAEs (100M tokens, width 12,288) cut OWT NMSE by 30.8–73.7 % and spliced ΔCE by 35.5–78.8 % at
>   layers 1, 8 and 11, on every seed pair.
> * A KL fine-tune at fixed L0 cuts ΔCE by a further 16–17 %, with a +2 % NMSE cost.
> * Truth information on GoT (cities + neg_cities, 452 test statements) shows no change in 34/34 comparisons.
>
> **Inconclusive or validation only.**
> * Every honesty/persona and causal result is VALIDATION.
> * We built a held-out RepE-LM test split (86 statements, plus new test-only wordings) and froze its protocol. It
>   has **not** been evaluated yet.
> * The GoT comparisons can only detect differences of about **0.05–0.09 AUROC** (seed plus statement-level sensitivity; observed differences ≤ 0.025), so
>   the GoT null is "no detectable change", not "preserved".
>
> **Corrections we made ourselves (11-13).**
> * PCA reference: **[PENDING Aditi's decision (D34). Proposed: "at equal reconstruction error, PCA (298–519 components) keeps at least as much truth information as the SAEs; at an equal active budget of 128 dimensions, SAEs are never worse and ReLU-family SAEs are better in 3 of 20 cells (validation)"]**.
> * The "SAE-in-the-loop" patches keep the error term. That makes them mathematically equal to adding an
>   input-dependent decoder-space vector, so they show that the SAE *selects* effect-carrying directions, not that
>   the effect is *mediated* by the SAE. The paper now says this, and a mediated test is proposed.
> * Two documentation errors were fixed (the GoT dataset description and an NMSE range).
>
> **Done on the branch since Review #8.**
> * Multi-layer improvement table (IMP-L at L3/8/9; R2/R4 at L1/8/11).
> * Paper (37 pp) and the report's GPT-2 sections.
> * 34 new unit tests and a CPU CI workflow.
> * 21 results figures as SVG/PDF, and an editable `.drawio` diagram.
> * AxBench, SAEBench and Paulo et al. in both bibliographies.
>
> **Not yet done.**
> * Merge to `main`: order proposed in `docs/remediation/merge_plan.md`; needs Dhruv's and Simba's agreement.
> * Journal choice.
> * Linking my commit email.
> * Off-machine checkpoint backup.
> * The RepE-LM test pass.
> * The team presentation, poster and videos.

---

## B. Reply on PR #27 to MakaveliSimz's two questions

> Thanks, Simba.
>
> 1. **Merge order.** I agree with #27 → #26 → #28, with #21 and #24 held until after them. One request: when #21
>    and #24 are re-scoped, please keep `results/geometry_of_truth/sae_faithfulness_results.csv` and the five legacy
>    `scripts/*geometry*.py` files. `scripts/geometry_of_truth/10_layer10_frozen_probe.py` reads them at runtime (the
>    Review #7 layer-10 correction). Details: `docs/remediation/merge_plan.md`.
> 2. **Gemma.** Verified against `main` with the GitHub compare API: #27 deletes **no** files and changes **no** Gemma path (every file is an
>    addition or modification under GPT-2, shared docs, tests and CI). The only shared files are the 7 doc ledgers,
>    which #26 and #28 also append to. Those conflicts are append-only.
>
> Two small things for #28: its `scripts/geometry_of_truth/22_*` and `23_*` reuse numbers that already exist on
> `sae-research` (IMP-L). Could they move to 25/26, or under `scripts/gemma/`? Also, `requirements.txt` changes in
> both PRs, in non-overlapping hunks.
>
> For the review: Review #8 asks the reviewer to re-run one result script. `python
> scripts/shared/m5_independent_verification.py` reads only committed files and should print `RESULT: ALL CHECKS
> PASS`. CI now runs it too.

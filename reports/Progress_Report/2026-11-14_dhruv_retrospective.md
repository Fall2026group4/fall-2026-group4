# Progress Report: Weeks 10 to 12, Dhruv Rai (retrospective, written 2026-11-14)

## In my own words

My part of the project was the causal tests and R3. In the Geometry of Truth causal test, I asked whether the truth
directions that probes find in GPT-2 Small actually change which answer the model prefers on the cities task. I added
each direction to the model's activations at one token and compared the effect with 600 random directions of the same
size. A positive control showed that the test could detect even a 1 % push. Still, the probe directions did no better
than random, and 0 of 108 SAE direction tests survived correction. This means we could not show that these directions
are causal on this task, not that they have no causal role. In this test the model never runs through the SAE; we only
take a direction from the SAE and add it. My SAE-in-the-loop test on RepE does run through the SAE, but it used the
older prompts with a length problem, so it only counts as supporting evidence. For Simba's PR #28, I checked that the
branch files matched his commit `167fadb` after R3 was moved out, and I ran the M6 tests: 59 passed and 1 failed on a
very small float tolerance. I also compared Simba's M6 numbers with the Gemma Scope paper, and at layer 12 our ΔCE and
FVU are within about 5 % and 2 % of the values read from the paper's figures. R3 measures how width and L0 change
reconstruction and behaviour across 36 Gemma Scope SAEs at layers 5, 12 and 19. It is still running on Kaggle, so I am
not claiming any result yet.

**This is a retrospective compilation, written on 2026-11-14 after the fact.** These entries were not submitted
weekly. Every item comes from the Git history (`git log --all --author="basquedhruv\|Dhruv Rai"`) or from
pull-request activity on GitHub. Weeks follow the numbering of the 2026-10-25 report (Week 9 = 2026-10-20 to 10-26).
Teammates' work is not described here.

## How this work maps to the project requirements

Requirement sources, both read on `origin/sae-research` at `d5521c4`:
* `docs/RESEARCH_ROADMAP.md` (as of 2026-11-10);
* `docs/remediation/professor_review_compliance.md` (as of 2026-11-13), which quotes Instructor Review #8.

Only mappings that hold literally are listed. Limits are stated where my work covers only part of a requirement.

| Requirement (source) | My evidence | Limits |
|---|---|---|
| **C4, GoT causal test** (roadmap §5, P1): "e.g. patching or steering on `cities` statements with true/false preference as outcome". Current state there: "No GoT causal evidence exists" | `results/geometry_of_truth/causal_truth_steering/` on `dhruv-causal` (`989d313`): pre-registered steering test at layers 4/6/8/10, 600 norm-matched random controls per layer, graded positive control, BH within families. Labelled *confirmed test result*. Probe and SAE truth directions were not distinguishable from random directions | One task (cities), one model |
| **S3, prompt-specific L0** (roadmap §4, P0): "Unaddressed; confounds every GoT/RepE architecture comparison" | `results/shared/sparsity_matched_truth_preservation/` (`989d313`): truth and honesty gaps compared at equal latents on the prompts (top-m, ±10 % matching rule). On GoT, TopK's apparent advantage came from using more latents | Validation finding, not test. The RepE prompts are length-confounded |
| **G9, public-SAE baselines** (roadmap §3, P1), and **Review #8 critique 2**, "compare with larger or pretrained SAEs" (compliance row 19) | `results/shared/public_sae_spliced_evaluation/` (`989d313`): `gpt2-small-res-jb` at layers 0–11 and `gpt2-small-resid-post-v5-32k` (TopK, k = 32) at layers 1–11, spliced into GPT-2 with a norm-matched random-error control | Validation finding. Covers G9's splicing part only. The public SAEs are not sparsity-matched to ours |
| **Review #8 peer-review rule**, "Before the next merge, the reviewer re-runs one result script and says what matched" (compliance row 15) | **#27:** re-ran `scripts/shared/m5_independent_verification.py`, all checks pass; showed the reported hash failures were a Windows line-ending (CRLF) false alarm. **#28:** confirmed the branch's files equal `167fadb` after the R3 move; re-derived the M6 numbers from the per-document rows (all match); ran the two M6 test files (59 passed, 1 failed: the H4a tolerance test, off by about 4e-8 on Windows) | — |
| **R3**, "Gemma-2-2B ladder with Gemma Scope 16k SAEs at about 3 layers and several published L0s", marked *essential* by the 2026-11-10 independent review (roadmap §2) | PR #30 (`dhruv-gemma-r3`): pre-registration (`2118b3c`), Amendment 1 to Kaggle T4 x2 (`8ed8102`), implementation with 51 tests (`07e3c8f`), Kaggle fix (`2d395d0`). Run K0 is in progress on Kaggle | Covers the Pile ladder only. The same R3 entry also asks for the RepE-LM prompts to be re-tokenised and the K1/K2 gates re-run for Gemma; that part is split off as R3b and not started. No R3 results yet |
| **Review #8 reproducibility**, "pinned environment, results versioning" (compliance row 17) | **Rebuild check** (`results/openwebtext/rebuild_reproducibility_check/`, `989d313`): all 108 in-house SAEs retrained from the committed configuration and compared row by row with the originals; max absolute differences were test EV 0.0013, L0 1.19 and cosine 0.0002. **R3 gate V9:** a resumed run must equal a fresh recomputation within relative difference 1e-6 | V9 is implemented and tested on CPU, but has not yet run on Kaggle |

## Week 10 (2026-10-27 to 11-02)

- **What I did:** no commits by me on any branch in this week.
- **Evidence:** none in Git.
- **Problems hit:** none recorded.
- **Next steps:** none recorded.

## Week 11 (2026-11-03 to 11-09)

- **What I did:**
  - Merged pull request #23 (`sae-research` into `main`: all-layer baselines, frozen probes, splicing, causal test,
    layer-6 fine-tuning) on 2026-11-07.
  - Started the `main` clean-up on branch `cleanup-main` (pull request #24): removed the old scripts, results and
    cookbook contents from `main` (32 files).
- **Evidence:** `33dcb9c` (merge of #23), `3cc0c70` (`cleanup-main`, #24).
- **Problems hit:** the removal in `3cc0c70` also took out a CSV that a Geometry of Truth script needs at run time
  (found and fixed in Week 12).
- **Next steps:** finish #24; follow-up causal experiments.

## Week 12 (2026-11-10 to 11-16; up to 2026-11-14)

- **What I did:**
  - **Causal follow-up experiments** on branch `dhruv-causal` (pull request #26, since closed): reanalysis of the
    layer-6 causal test, SAE-in-the-loop test, frozen-probe error decomposition; sparsity-matched truth preservation,
    GoT causal test, public SAE splicing, rebuild check; figures F1–F7. The causal scripts were changed to recompute
    the final token exactly, so each run fits one Kaggle session.
  - **`main` clean-up (#24):** restored `results/geometry_of_truth/sae_faithfulness_results.csv`, which GoT script 10
    needs at run time.
  - **R3, Gemma Scope width × L0 ladder (my experiment):**
    - Pre-registration: 54 published Gemma Scope SAEs on M6's Pile sample, layers 5/12/19 (2026-11-13).
    - Literature comparison: M6's numbers against the Gemma Scope, JumpReLU, SAEBench, TopK and end-to-end SAE papers.
      At layer 12, M6 matches the Gemma Scope paper within about 5 % (values read from the paper's figures).
    - Amendment 1 (2026-11-13, before any code or run): hardware moved from an AWS A10G to Kaggle T4 x2.
    - Implementation (2026-11-14): scripts 05–07 (ladder runs, fp32 reference, resume with gate V9, gates V0–V10,
      aggregate step, Q0b, figures), 51 tests, and a Kaggle runner notebook.
    - Moved out of Simba's PR #28 into its own branch `dhruv-gemma-r3` at his request (2026-11-14), without rewriting
      history: four commits cherry-picked, one revert commit on #28. R3 now has its own pull request, #30.
    - Kaggle run K0: first attempts failed and were fixed (Kaggle secret name; Kaggle's preinstalled
      torchvision/torchaudio breaking transformers with the pinned torch 2.7.0). K0 is running now.
- **Evidence:**
  - `dhruv-causal` (#26): `d9e14d2`, `eed77a7`, `68181e0`, `989d313` (all 2026-11-10).
  - `cleanup-main` (#24): `d2f67aa` (2026-11-13).
  - R3 on `dhruv-gemma-r3` (#30): `2118b3c` pre-registration, `b82e871` literature comparison, `8ed8102` Amendment 1,
    `07e3c8f` implementation, `961dc60` branch move, `2d395d0` torchvision fix. The same first four were originally
    `7130f42`, `8c7176b`, `4fd2706`, `24f6261` on #28.
  - `2bab391` on `simba-gemma-m6-results` (#28): the single revert that took R3 out of #28.
- **Problems hit:**
  - The AWS account's free plan has no GPU instances, so R3 moved to Kaggle T4 x2. This dropped the planned
    T4-vs-A10G comparison (Amendment 1).
  - R3 had been committed into Simba's PR #28. He asked to keep #28 to M6 only, so it was moved out.
  - Kaggle K0 attempts failed twice: wrong secret name, then the torchvision incompatibility (fixed in `2d395d0`).
- **Next steps:**
  - Let K0 finish. Check the hardware gate (V0), the smoke timings and the fp32 reference (V8).
  - Re-project and run K1–K3 (Tier A) within the 30 h weekly Kaggle quota. Run Tier B only after Tier A passes,
    and only before the 2026-11-22 technical freeze.
  - Run the aggregate step, Q0b and the figures; write Section 2 of the R3 README.
  - Get #30 reviewed; merge #28 before #30.

## Pull-request reviews in this period (dates not recorded in Git)

- **#27 (`sae-research`):** re-ran `scripts/shared/m5_independent_verification.py`; all checks pass. Found that the
  reported hash failures were a Windows line-ending (CRLF) false alarm, not changed files.
- **#28, #29, #22:** re-derived the M6 numbers from the per-document rows; all match. Slide 20 provenance fix.
- **#28 (head `2bab391`, local check):** files equal `167fadb`; M6 test files 59 passed, 1 failed (H4a tolerance,
  about 4e-8 on Windows); figures regenerated from the saved tables match the committed ones in content (not byte for
  byte: different matplotlib version).

## Week of 2026-11-08 (added 2026-11-16; dates are local time, UTC-5)

- **2026-11-10, causal follow-ups on `dhruv-causal` (PR #26):** `d9e14d2` (reanalysis of the layer-6 causal test,
  SAE-in-the-loop test, probe error decomposition), `eed77a7` (GoT causal test, sparsity-matched preservation, public
  SAE splicing, figures), `68181e0` (exact final-token recomputation so the runs fit in one Kaggle session) and
  `989d313` (the run outputs).
- **2026-11-13:** `d2f67aa` on `cleanup-main` (#24) kept a file that GoT script 10 needs. For R3: pre-registration
  `2118b3c`, literature comparison `b82e871`, and Amendment 1 `8ed8102` (move from AWS A10G to Kaggle T4 x2).
- **2026-11-14, R3 code and branch:** implementation `07e3c8f` (scripts 05 to 07, tests, Kaggle notebook); R3 moved
  out of #28 into `dhruv-gemma-r3` (`961dc60`, with the revert `2bab391` on #28); Kaggle torchvision fix `2d395d0`;
  this report `33a7df3`; Clarification 2 `e2b7b3a` (the fp32 reference becomes its own step, K0F).
- **2026-11-14, first K0 run on Kaggle** (run id `K0_20261115T024202Z`, UTC): it crashed at the end of the fp32
  step, so nothing from it is used (R3 README, "Run attempts"). This led to Clarification 2.
- **2026-11-14 evening (GitHub shows 2026-11-15 UTC), PR reviews, all approved:** #28 (rechecked the output hashes,
  re-derived the headline numbers from the per-document rows, confirmed the branch equals `167fadb`); #29 (checked
  the slides against the M6 outputs and asked for a fix to the provenance line on slide 20); #22 (Pile EDA; confirmed
  it has no file overlap with #28).

## Week of 2026-11-15 (up to 2026-11-16)

- **2026-11-16, `d11c93e`:** merged `origin/main` into `dhruv-gemma-r3` after #28 was merged. My doc entries were
  renumbered to D37 and §44. The M6 files are unchanged.
- **2026-11-16, `10286e7`:** recorded R3 Deviations 1 and 2 (K3 split into K3a and K3b; Tier B cancelled) before any
  Tier A result was looked at.
- **2026-11-16, PR #30:** base changed from `simba-gemma-m6-results` to `main` (Instructor Review #9); review requested
  from Simba. `268fae3` added the "In my own words" section to this report.
- **R3 runs on Kaggle (R3 README, Section 3, as of 2026-11-16):** K1 (layer 5) finished in 7.64 h, and K2 (layer 19)
  has started. The outputs are still on Kaggle and no metric has been read, so no result is claimed. The run record of
  the K0 rerun that K1 used is not in the repository yet.

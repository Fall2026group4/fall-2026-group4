# Documentation archive

This folder holds reports that are superseded or historical but kept for provenance. Nothing here is deleted. Git
history keeps every earlier path.

| File | What it is | Superseded by |
|---|---|---|
| `2026-11-04_methodology_review_corrections_report.md` | Full report of the 2026-11-04 methodology-review corrections: optimizer consistency, sparsity matching, causal controls, GoT frozen probe, RepE checks, Gemma controls. Paths inside refer to the pre-migration layout (map in `../README.md` and `../migration_manifest.csv`) | Summarised in `../methodology.md` §7 and `../experiment_history.md` Stage B |

## Movement log (2026-11-12)

All moves were made with `git mv`. File contents are byte-identical (SHA-256 checked before and after). Links in active
documents were updated in the same commit. Historical records that cite old paths are left as they are
(`EXPERIMENT_CHANGELOG.md`, `migration_manifest.csv`, the committed PDF).

| Old path | New path | Reason | SHA-256 (unchanged) |
|---|---|---|---|
| `docs/2026-11-04_professor_review_report.md` | `docs/archive/2026-11-04_methodology_review_corrections_report.md` | Neutral name; historical report moved out of the active top level | `fa170814a9af4a3ebb63ba67c0f7b53aca40d82e783cc2e44920dadeb9b7aed9` |
| `docs/final_defense/PROFESSOR_DEFENSE_CHEATSHEET.md` | `docs/final_defense/DEFENSE_CHEATSHEET.md` | Neutral name | `dcabfe3531cbef3cac2ff30c89c36de7b0bdcf23111095ba3b66ab2abd188dab` |
| `docs/final_defense/PROFESSOR_QUESTIONS_AND_ANSWERS.md` | `docs/final_defense/ANTICIPATED_REVIEW_QUESTIONS.md` | Neutral name | `332c117a31fe29a4bd779eec3bc886d68182ed6847194a68d8507720e6b202c5` |

**Deliberately not moved:**
* `docs/2026-11-04_code_audit.md` and `docs/2026-11-04_methodology_changes.md` already have neutral names, and the
  second is linked from a historical results README (`results/repe/steering/README.md`), which is not edited.
* `docs/research_gaps_and_plan.md` is cited in a pre-registered script's docstring
  (`scripts/shared/all_layer_frozen_probe_evaluation.py`).
* No script, result folder, checkpoint or activation cache was moved. Pre-registration files record script paths and
  SHA-256, so moving code would break their hash guards.

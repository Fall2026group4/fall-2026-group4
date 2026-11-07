# RepE honesty results (GPT-2 Small layer 6, final token)

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

* `original_baseline.csv`, `direction_fidelity.csv`, `cross_wording/` — **VALID** as validation-split numbers (layer 6 was chosen on validation, so they are optimistically biased). The quoted "cross-wording JumpReLU ≈ 0.8010" is not in any file; the saved value is 0.7394.
* `representation_preservation.csv` — **VALID** (validation only). `representation_preservation_val_test.csv` — **AUTHORITATIVE**: validation + first test-split evaluation + cross-wording, all 9 variants.
* SAE L0 on RepE prompts is **not matched** across architectures (JumpReLU 187, ReLU 170, TopK 128).
* `steering/` — see `steering/README.md`.

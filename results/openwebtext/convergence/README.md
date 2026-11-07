# 15k-step SAE training runs (OWT)

Status legend: **AUTHORITATIVE** = use for claims; **VALID (scope)** = correct within the stated scope only; **SUPERSEDED** = kept for provenance, replaced by the named result; **INVALID** = do not cite; **SMOKE** = pipeline check, not a result. Central registry: `docs/results_registry.md`.

| Directory pattern | Procedure | Status |
|---|---|---|
| `layer_N/{arch}_l0_128_uninterrupted_seed{S}/` | Scratch → 15k, one Adam, one controller, one RNG stream (`13_consistent_protocol_retrain.py`) | **AUTHORITATIVE** |
| `layer_6/{arch}_l0_128/` + `layer_6/{arch}_l0_128_resume_5000_to_15000/` | 0→5k (05/06/07), then resume 5k→15k with Adam + controller restored, RNG re-seeded (08) | **VALID (scope)**: base SAEs of all layer-6 downstream experiments |
| `layer_{2,10}/{arch}_l0_128_warmstart_1000_to_15000/` | 1k sweep weights, fresh Adam, controller reset (11) | **SUPERSEDED** |
| `layer_6/jumprelu_kernel_ste_l0_128/` | Kernel-STE JumpReLU; script not in repo | **VALID (scope)**, unreproducible |

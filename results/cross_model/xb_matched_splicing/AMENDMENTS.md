# XB amendments (each recorded before any affected output was analysed)

* **A1 (2026-11-21, implementation only).** Resume rule: a document counts as done only if its rows and its `sum_x`
  entries for every layer were saved. Rows of incomplete documents are dropped and recomputed. Reason: `sum_x.npz` is
  checkpointed every 16 documents, and a run was stopped mid-way (GPU contention, see below). No metric definition
  changes.
* **Operational note (2026-11-21).**
  * `gpt2_pile_shared_main` crashed with CUDA OOM at start-up. A concurrent extraction smoke test exhausted GPU memory.
  * `gemma_pile_own_main` was stopped deliberately after 32 documents to free memory for the XT-G extraction.
  * Both resume under A1. Rule adopted: at most two GPU jobs at once, and never two Gemma jobs.
* **Note (by design, pre-registered).** The random-error directions are drawn per (model, layer, document) and shared
  by all SAEs at that cell. They therefore differ from M6's draws, so per-document random-control values differ from
  M6. The SAE-splice condition was checked against M6's own per-document rows on 8 documents: mean |ΔCE difference|
  0.0004 nats, maximum 0.008 (bf16 floor).
* **A2 (2026-11-21, implementation only, before any Gemma full-run output).** Gemma runs layer-major: one layer's
  SAEs on the GPU at a time, documents in the inner loop, and the original pass repeated per layer. Deterministic
  algorithms are on, so the original logits are identical across repeats.
  * Reason: keeping 26 Gemma Scope SAEs resident (7.9 GB) alongside the concurrent GPT-2 run would exceed 23 GB, and
    pinning them on the host would exceed its 15 GB of RAM.
  * The resume rule becomes per (document, layer) cell.
  * GPT-2 stays document-major.
  * No metric or condition changes.
* **A3 (2026-11-21, implementation only).** GPT-2 also runs layer-major. Keeping all 159 GPT-2 SAEs resident took
  10.4 GB and left no room for TI-1 training and XT-G.
  * The running `gpt2_owt_xm_shared_main` was stopped at about 100/512 documents and resumed under the A1/A2 rule.
  * Its document-major rows and the new layer-major rows are bit-identical by construction: deterministic algorithms,
    batch 1, same seeds. The Gemma smoke check showed exactly this.
* **Operational (2026-11-21 ~12:50).** The Gemma queue was restarted with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: the caching allocator had grown to 15 GB, against a peak need of about 7 GB. `gemma_pile_shared_main` resumed under A1/A2. Memory management only; no numerical change.

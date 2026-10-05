"""Stream OpenWebText / WikiText, tokenize, and chunk to a fixed length.

Only token ids are ever saved to disk (see the project spec's "Disk
discipline" note) - never raw activations. Chunks are ``context_length``
tokens total, BOS first, so a chunk with ``context_length=128`` holds BOS
plus the first 127 real tokens of a document. Documents shorter than that
are skipped.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the pipeline config. Every owt_0X script calls this - nothing
    about sample sizes, layers, SAE releases, seeds or strengths is
    hard-coded in a script."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_all_seeds(seed: int) -> None:
    """Set random, numpy and torch seeds, as required at the top of every
    script (Reproducibility conventions)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def chunk_tokens(token_ids: list[int], bos_token_id: int, context_length: int) -> list[int] | None:
    """Return a ``context_length``-token chunk (BOS + first N-1 real
    tokens), or None if the document is too short. Public so
    ``tests/test_owt.py`` can check "every chunk is exactly
    ``context_length`` tokens with BOS first" directly, without streaming
    a real dataset."""
    n_real_needed = context_length - 1
    if len(token_ids) < n_real_needed:
        return None
    return [bos_token_id] + token_ids[:n_real_needed]


def stream_and_tokenize(
    dataset_name: str,
    split: str,
    tokenizer,
    n_documents: int,
    context_length: int,
    shuffle_buffer_size: int = 10000,
    seed: int = 0,
    text_field: str = "text",
    dataset_config: str | None = None,
    skip_documents: int = 0,
    max_memory_mb: int | None = 1536,
) -> np.ndarray:
    """Stream ``dataset_name``, tokenize with ``tokenizer``, and return an
    array of shape ``[n_kept, context_length]`` of GPT-2 token ids (BOS
    first). Streams without downloading the full dataset; shuffles with a
    buffer so the kept documents aren't just the first N in file order.

    ``dataset_config`` is for datasets that need a config/subset name
    alongside the dataset name (e.g. ``load_dataset("wikitext",
    "wikitext-103-raw-v1", split="test")`` - OpenWebText doesn't need one).

    ``skip_documents``: discard this many kept documents from the start of
    the (deterministic, fixed-seed) stream before collecting
    ``n_documents``. Lets a fresh process pick up where an earlier one
    left off - e.g. if memory grows unboundedly over a single long-running
    process on a constrained box, call this repeatedly in small batches
    (skip=0,400,800,...) from separate process invocations instead of one
    giant run, and concatenate the resulting arrays. Each call still
    re-streams from the beginning, so this trades network/CPU time for a
    bounded memory footprint per process - it is not a resume in the
    "skip already-downloaded bytes" sense.

    ``max_memory_mb``: cap this process's virtual memory via
    ``resource.setrlimit(RLIMIT_AS, ...)`` (no root needed - a process can
    always lower its own limits). OpenWebText has some documents whose raw
    text is itself huge; reading one out of the stream can spike memory
    before any truncation we do gets a chance to run. Without a limit, that
    spike runs the whole machine out of memory and the kernel OOM-killer
    SIGKILLs the process - uncatchable, no traceback, "Killed" and nothing
    else. With the limit set, the *same* allocation instead raises a plain
    Python ``MemoryError`` well before the system is in danger, which the
    per-document loop below catches and treats as "skip this one document"
    - so one bad document costs you a skipped sample, not the whole run.
    Set to None to disable (or if your platform doesn't support RLIMIT_AS).
    Tune based on `free -h` - leave headroom above whatever the model/
    tokenizer/pandas baseline already uses.
    """
    if max_memory_mb is not None:
        import resource

        limit_bytes = max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except (ValueError, OSError):
            pass  # not supported here (e.g. some containers) - best effort only

    from datasets import load_dataset

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    # shuffle_buffer_size <= 0 skips the .shuffle() call entirely (diagnostic:
    # isolates whether the streaming shuffle buffer itself - not its size -
    # is the source of unbounded memory growth seen on this box).
    if shuffle_buffer_size and shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id

    # We only ever keep the first (context_length - 1) tokens of a document
    # (see chunk_tokens), but tokenizer.encode() below was running over the
    # ENTIRE document text first. OpenWebText has some pathologically huge
    # documents (megabytes of text); BPE-encoding one of those in full before
    # truncating is what was spiking memory and getting the process OOM-killed
    # - not a leak that grows across documents. Cap the text we hand to the
    # tokenizer at a generous multiple of what we could possibly need, so a
    # giant outlier document costs the same as a normal one.
    max_chars = context_length * 20

    chunks: list[list[int]] = []
    kept_total = 0
    examples_seen = 0
    progress_every = max(1, n_documents // 20)  # ~20 progress lines total
    skipped_for_memory = 0
    for example in dataset:
        examples_seen += 1
        try:
            text = example[text_field]
            if not text or not text.strip():
                continue
            token_ids = tokenizer.encode(text[:max_chars], add_special_tokens=False)
            chunk = chunk_tokens(token_ids, bos_token_id, context_length)
        except MemoryError:
            # This one document's raw text was too large to even read out of
            # the stream safely (see max_memory_mb above) - skip it, not the
            # whole run.
            skipped_for_memory += 1
            continue
        if chunk is None:
            continue

        kept_total += 1
        if kept_total <= skip_documents:
            continue  # already collected by an earlier batch - discard and move on

        chunks.append(chunk)
        if len(chunks) % progress_every == 0:
            print(
                f"  ...{len(chunks)}/{n_documents} kept "
                f"(skip={skip_documents}, {examples_seen} examples scanned)",
                flush=True,
            )
        if len(chunks) >= n_documents:
            break

    if skipped_for_memory:
        print(f"Skipped {skipped_for_memory} document(s) that hit the memory cap "
              f"(max_memory_mb={max_memory_mb}) - treated as too large, not an error.")

    if len(chunks) < n_documents:
        print(
            f"Warning: only found {len(chunks)} documents of >= "
            f"{context_length - 1} real tokens (wanted {n_documents})."
        )

    return np.array(chunks, dtype=np.int64)


def save_token_array(tokens: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tokens)


def load_token_array(path: str | Path) -> np.ndarray:
    return np.load(path)


def save_neutral_snippets(
    tokens: np.ndarray,
    tokenizer,
    n: int,
    path: str | Path,
    seed: int = 0,
) -> None:
    """Pick ``n`` chunks, decode to text, and save as JSONL - the shared
    neutral set for Simba and Aditi (first-priority deliverable)."""
    rng = np.random.default_rng(seed)
    n = min(n, tokens.shape[0])
    indices = rng.choice(tokens.shape[0], size=n, replace=False)
    indices.sort()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for i in indices:
            token_ids = tokens[i].tolist()
            text = tokenizer.decode(token_ids[1:])  # skip BOS when decoding
            f.write(json.dumps({"token_ids": token_ids, "text": text}) + "\n")


def load_neutral_snippets(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
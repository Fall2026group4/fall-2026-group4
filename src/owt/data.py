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
) -> np.ndarray:
    """Stream ``dataset_name``, tokenize with ``tokenizer``, and return an
    array of shape ``[n_kept, context_length]`` of GPT-2 token ids (BOS
    first). Streams without downloading the full dataset; shuffles with a
    buffer so the kept documents aren't just the first N in file order.

    ``dataset_config`` is for datasets that need a config/subset name
    alongside the dataset name (e.g. ``load_dataset("wikitext",
    "wikitext-103-raw-v1", split="test")`` - OpenWebText doesn't need one).
    """
    from datasets import load_dataset

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id

    chunks: list[list[int]] = []
    for example in dataset:
        text = example[text_field]
        if not text or not text.strip():
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunk = chunk_tokens(token_ids, bos_token_id, context_length)
        if chunk is None:
            continue
        chunks.append(chunk)
        if len(chunks) >= n_documents:
            break

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

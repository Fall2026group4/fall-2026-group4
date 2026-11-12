"""Shared, testable helpers for M6: Gemma Scope SAE reconstruction and splicing on The Pile.

Pre-registration: results/gemma2_pile/gemma_scope_splicing/README.md and preregistration.json (commit b39dcc2).

This module currently holds the *sampling* part (Commit 2A): deterministic selection of the evaluation and
reference documents, BOS + truncation, and the hashes that make the sample reproducible. The splicing and
metric helpers are added in Commit 2B.

Design rules (from the pre-registration):
* Documents are identified by their 0-based line number in the pinned ``val.jsonl.zst``.
* Order = ``numpy.random.default_rng(seed).permutation(n_lines)``. Walking that order, a document is *eligible* if it
  has at least ``min_text_tokens`` text tokens (BOS not counted). The first ``n_eval`` eligible documents form the
  evaluation set, the next ``n_reference`` eligible documents the reference set. The two sets are therefore disjoint
  by construction; :func:`select_documents` also asserts it.
* Model input per document = [BOS] + the first ``max_tokens_including_bos - 1`` text tokens.
* Hashes: SHA-256 of the UTF-8 text, and SHA-256 of the token ids used (little-endian int32 bytes).

Only numpy and the standard library are imported at module level, so the sampling tests need no torch or
transformers. ``zstandard`` is imported lazily inside :func:`iter_jsonl_zst`.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

# Fields of preregistration.json that the sampling code depends on. Missing fields are an error, not a default.
REQUIRED_PREREG_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ('data', 'hf_dataset'),
    ('data', 'revision'),
    ('data', 'file'),
    ('data', 'bytes'),
    ('data', 'sha256'),
    ('data', 'source_label_field'),
    ('data', 'test_file_used'),
    ('sampling', 'seed'),
    ('sampling', 'min_text_tokens'),
    ('sampling', 'n_eval'),
    ('sampling', 'n_reference'),
    ('sampling', 'disjoint'),
    ('tokenization', 'tokenizer'),
    ('tokenization', 'revision'),
    ('tokenization', 'prepend_bos'),
    ('tokenization', 'max_tokens_including_bos'),
)


# ----------------------------------------------------------------------------------------------------------- config
def get_field(cfg: dict, path: Sequence[str]):
    """Return cfg[path[0]][path[1]]...; raise KeyError naming the full path if any level is missing."""
    node = cfg
    for i, key in enumerate(path):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"preregistration field missing: {'.'.join(path[:i + 1])}")
        node = node[key]
    return node


def load_preregistration(path: Path) -> dict:
    """Load preregistration.json and check that every field the sampling code reads is present and sane."""
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    for field in REQUIRED_PREREG_FIELDS:
        get_field(cfg, field)
    if cfg['data']['test_file_used'] is not False:
        raise ValueError('preregistration says the Pile test file is used; M6 must never use it')
    if cfg['sampling']['disjoint'] is not True:
        raise ValueError('preregistration must require disjoint evaluation and reference sets')
    rev = str(cfg['tokenization']['revision'])
    if len(rev) != 40 or any(c not in '0123456789abcdef' for c in rev):
        raise ValueError('tokenization.revision must be a full 40-character commit SHA (Amendment 1), '
                         f'not {rev!r}')
    if cfg['tokenization']['prepend_bos'] is not True:
        raise ValueError('preregistration must prepend BOS')
    for key in ('n_eval', 'n_reference', 'min_text_tokens'):
        if int(cfg['sampling'][key]) <= 0:
            raise ValueError(f'sampling.{key} must be positive')
    if int(cfg['tokenization']['max_tokens_including_bos']) < 2:
        raise ValueError('max_tokens_including_bos must be at least 2 (BOS + one text token)')
    return cfg


# ----------------------------------------------------------------------------------------------------------- hashes
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks (the Pile validation file is ~338 MB)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of the document text encoded as UTF-8."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_token_ids(ids: Sequence[int]) -> str:
    """SHA-256 of token ids as little-endian int32 bytes (Gemma's vocabulary, 256 000, fits in int32)."""
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() > np.iinfo(np.int32).max):
        raise ValueError('token id out of int32 range')
    return hashlib.sha256(arr.astype('<i4').tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------------- corpus
def iter_jsonl_zst(path: Path) -> Iterator[Tuple[int, dict]]:
    """Yield (0-based line index, parsed JSON object) for every line of a .jsonl.zst file, streaming."""
    import zstandard  # lazy: only needed when the real corpus is read

    with open(path, 'rb') as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding='utf-8')
        for i, line in enumerate(text):
            yield i, json.loads(line)


def count_lines_jsonl_zst(path: Path) -> int:
    """Number of documents (lines) in a .jsonl.zst file."""
    n = 0
    for n, _ in enumerate(iter_jsonl_zst(path), start=1):
        pass
    return n


def load_lines(path: Path, wanted: Iterable[int]) -> Dict[int, dict]:
    """Return {line index: parsed object} for the requested line indices (one streaming pass)."""
    wanted = set(int(i) for i in wanted)
    out: Dict[int, dict] = {}
    if not wanted:
        return out
    last = max(wanted)
    for i, obj in iter_jsonl_zst(path):
        if i in wanted:
            out[i] = obj
        if i >= last:
            break
    missing = wanted - out.keys()
    if missing:
        raise ValueError(f'{len(missing)} requested line indices are beyond the end of the file')
    return out


# -------------------------------------------------------------------------------------------------------- selection
def permutation_order(n_lines: int, seed: int) -> np.ndarray:
    """The pre-registered document order: numpy.random.default_rng(seed).permutation(n_lines).

    The selected line indices are also saved in the manifest, which is the authoritative record of the sample
    (NumPy does not guarantee identical Generator streams across all future versions; the version is recorded).
    """
    if n_lines <= 0:
        raise ValueError('n_lines must be positive')
    return np.random.default_rng(int(seed)).permutation(int(n_lines))


def select_documents(order: Iterable[int], n_text_tokens: Callable[[int], int], *, n_eval: int,
                     n_reference: int, min_text_tokens: int) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Walk ``order``; keep documents with >= min_text_tokens text tokens; first n_eval -> eval, next n_reference -> ref.

    ``n_text_tokens(line_index)`` returns the number of text tokens (BOS excluded) of that document.
    Returns (eval line indices, reference line indices, counters). Raises if the order runs out before both sets
    are full. Disjointness is guaranteed by construction (each index is assigned once) and asserted.
    """
    eval_ids: List[int] = []
    ref_ids: List[int] = []
    seen = set()
    examined = skipped = 0
    for idx in order:
        idx = int(idx)
        if idx in seen:
            raise ValueError(f'line index {idx} appears twice in the order; the order must be a permutation')
        seen.add(idx)
        examined += 1
        if n_text_tokens(idx) < min_text_tokens:
            skipped += 1
            continue
        if len(eval_ids) < n_eval:
            eval_ids.append(idx)
        else:
            ref_ids.append(idx)
        if len(eval_ids) == n_eval and len(ref_ids) == n_reference:
            break
    if len(eval_ids) < n_eval or len(ref_ids) < n_reference:
        raise ValueError(f'order exhausted: {len(eval_ids)}/{n_eval} eval, {len(ref_ids)}/{n_reference} reference '
                         f'after examining {examined} documents')
    assert not set(eval_ids) & set(ref_ids), 'evaluation and reference sets overlap'
    return eval_ids, ref_ids, {'examined': examined, 'skipped_too_short': skipped}


def select_from_corpus(order: np.ndarray, load_texts: Callable[[Sequence[int]], Dict[int, str]],
                       count_text_tokens: Callable[[str], int], *, n_eval: int, n_reference: int,
                       min_text_tokens: int, initial_candidates: int = 2048):
    """Run :func:`select_documents` over ``order`` without loading the whole corpus.

    Texts are loaded for a prefix of the order; if the prefix holds too few eligible documents, the prefix is
    doubled and selection is redone from the start. The result therefore depends only on ``order`` and the
    eligibility rule, never on the prefix size. Returns (eval ids, ref ids, counters, {line index: text}).
    """
    n_cand = min(int(initial_candidates), len(order))
    while True:
        prefix = [int(i) for i in order[:n_cand]]
        texts = load_texts(prefix)
        counts: Dict[int, int] = {}

        def n_tok(i: int) -> int:
            if i not in counts:
                counts[i] = int(count_text_tokens(texts[i]))
            return counts[i]

        try:
            eval_ids, ref_ids, info = select_documents(prefix, n_tok, n_eval=n_eval, n_reference=n_reference,
                                                       min_text_tokens=min_text_tokens)
        except ValueError:
            if n_cand >= len(order):
                raise
            n_cand = min(2 * n_cand, len(order))
            continue
        info['candidates_loaded'] = n_cand
        return eval_ids, ref_ids, info, texts


# ----------------------------------------------------------------------------------------------------- tokenization
def truncate_with_bos(text_ids: Sequence[int], bos_id: int, max_tokens_including_bos: int) -> Tuple[List[int], bool]:
    """[BOS] + the first (max - 1) text tokens. Returns (ids used, whether the text was truncated)."""
    if max_tokens_including_bos < 2:
        raise ValueError('max_tokens_including_bos must be at least 2')
    keep = max_tokens_including_bos - 1
    ids = [int(bos_id)] + [int(t) for t in text_ids[:keep]]
    return ids, len(text_ids) > keep


def manifest_row(*, set_name: str, rank: int, perm_position: int, line_index: int, source: str, text: str,
                 text_ids: Sequence[int], bos_id: int, max_tokens_including_bos: int) -> dict:
    """One row of sample_manifest.csv: identifiers and hashes, never the text itself."""
    used, truncated = truncate_with_bos(text_ids, bos_id, max_tokens_including_bos)
    return {
        'set': set_name,
        'rank': int(rank),
        'perm_position': int(perm_position),
        'line_index': int(line_index),
        'pile_set_name': source,
        'n_chars': len(text),
        'text_sha256': sha256_text(text),
        'n_text_tokens': len(text_ids),
        'n_tokens_used': len(used),
        'truncated': bool(truncated),
        'token_ids_sha256': sha256_token_ids(used),
    }


MANIFEST_COLUMNS = ['set', 'rank', 'perm_position', 'line_index', 'pile_set_name', 'n_chars', 'text_sha256',
                    'n_text_tokens', 'n_tokens_used', 'truncated', 'token_ids_sha256']

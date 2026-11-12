"""Tests for the M6 sampling helpers in src/evaluation/pile_splicing.py (no network, no model, no torch).

Run from repository root:  python -m pytest -q src/tests/test_pile_sampling.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import pile_splicing as ps

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / 'results/gemma2_pile/gemma_scope_splicing/preregistration.json'


# ------------------------------------------------------------------------------------------- pre-registration fields
def test_committed_preregistration_has_every_field_the_code_reads():
    cfg = ps.load_preregistration(PREREG)
    assert cfg['sampling']['seed'] == 42
    assert cfg['sampling']['n_eval'] == 512 and cfg['sampling']['n_reference'] == 64
    assert cfg['sampling']['min_text_tokens'] == 32
    assert cfg['tokenization']['max_tokens_including_bos'] == 1024
    assert cfg['data']['file'] == 'val.jsonl.zst' and cfg['data']['test_file_used'] is False
    assert len(cfg['data']['sha256']) == 64
    # Amendment 1: tokenizer and model frozen to the same Hugging Face commit
    assert cfg['tokenization']['revision'] == 'c5ebcd40d208330abc697524c919956e692655cf'
    assert cfg['model']['revision'] == cfg['tokenization']['revision']


def test_preregistration_rejects_unpinned_tokenizer_revision(tmp_path):
    cfg = json.loads(PREREG.read_text(encoding='utf-8'))
    cfg['tokenization']['revision'] = 'main'
    p = tmp_path / 'prereg.json'
    p.write_text(json.dumps(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='40-character'):
        ps.load_preregistration(p)


def test_preregistration_missing_field_is_an_error(tmp_path):
    cfg = json.loads(PREREG.read_text(encoding='utf-8'))
    del cfg['sampling']['seed']
    p = tmp_path / 'prereg.json'
    p.write_text(json.dumps(cfg), encoding='utf-8')
    with pytest.raises(KeyError, match='sampling.seed'):
        ps.load_preregistration(p)


def test_preregistration_rejects_test_file(tmp_path):
    cfg = json.loads(PREREG.read_text(encoding='utf-8'))
    cfg['data']['test_file_used'] = True
    p = tmp_path / 'prereg.json'
    p.write_text(json.dumps(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='test file'):
        ps.load_preregistration(p)


# ------------------------------------------------------------------------------------------------------ determinism
def test_permutation_is_seeded_and_a_true_permutation():
    a, b = ps.permutation_order(1000, 42), ps.permutation_order(1000, 42)
    assert np.array_equal(a, b)
    assert sorted(a.tolist()) == list(range(1000))
    assert not np.array_equal(a, ps.permutation_order(1000, 43))


def test_selection_follows_order_skips_short_and_is_disjoint():
    lengths = {i: (5 if i % 3 == 0 else 40) for i in range(100)}  # every third document too short
    order = ps.permutation_order(100, 42)
    ev, ref, info = ps.select_documents(order, lengths.__getitem__, n_eval=10, n_reference=4, min_text_tokens=32)
    eligible_in_order = [int(i) for i in order if lengths[int(i)] >= 32]
    assert ev == eligible_in_order[:10] and ref == eligible_in_order[10:14]
    assert not set(ev) & set(ref)
    assert all(lengths[i] >= 32 for i in ev + ref)
    assert info['skipped_too_short'] == sum(lengths[int(i)] < 32 for i in order[:info['examined']])


def test_selection_threshold_is_inclusive():
    ev, ref, _ = ps.select_documents([0, 1, 2], {0: 31, 1: 32, 2: 33}.__getitem__, n_eval=1, n_reference=1,
                                     min_text_tokens=32)
    assert ev == [1] and ref == [2]


def test_selection_raises_when_too_few_documents():
    with pytest.raises(ValueError, match='exhausted'):
        ps.select_documents(range(5), lambda i: 100, n_eval=4, n_reference=2, min_text_tokens=32)


def test_selection_rejects_repeated_indices():
    with pytest.raises(ValueError, match='twice'):
        ps.select_documents([0, 1, 1, 2], lambda i: 100, n_eval=2, n_reference=2, min_text_tokens=32)


def test_select_from_corpus_does_not_depend_on_prefix_size():
    """Growing the candidate prefix must give exactly the selection of a single pass over the full order."""
    rng = np.random.default_rng(0)
    corpus = {i: ' '.join(['w'] * int(rng.integers(1, 60))) for i in range(500)}  # many docs < 32 words
    order = ps.permutation_order(500, 42)
    count = lambda t: len(t.split())  # noqa: E731  toy tokenizer: one token per word
    loads = []

    def load_texts(idx):
        loads.append(len(idx))
        return {i: corpus[i] for i in idx}

    ev, ref, info, _ = ps.select_from_corpus(order, load_texts, count, n_eval=40, n_reference=10,
                                             min_text_tokens=32, initial_candidates=8)
    full_ev, full_ref, _ = ps.select_documents(order, lambda i: count(corpus[i]), n_eval=40, n_reference=10,
                                               min_text_tokens=32)
    assert (ev, ref) == (full_ev, full_ref)
    assert len(loads) > 1 and info['candidates_loaded'] == loads[-1]


# ----------------------------------------------------------------------------------------------------- tokenization
def test_truncate_with_bos_short_and_long():
    ids, trunc = ps.truncate_with_bos([5, 6, 7], bos_id=2, max_tokens_including_bos=1024)
    assert ids == [2, 5, 6, 7] and trunc is False
    long = list(range(10, 3000))
    ids, trunc = ps.truncate_with_bos(long, bos_id=2, max_tokens_including_bos=1024)
    assert len(ids) == 1024 and ids[0] == 2 and ids[1:] == long[:1023] and trunc is True
    ids, trunc = ps.truncate_with_bos(list(range(1023)), bos_id=2, max_tokens_including_bos=1024)
    assert len(ids) == 1024 and trunc is False  # exactly fits


# ----------------------------------------------------------------------------------------------------------- hashes
def test_hashes_are_stable_and_sensitive():
    assert ps.sha256_text('café') == hashlib.sha256('café'.encode('utf-8')).hexdigest()
    h = ps.sha256_token_ids([2, 100, 255999])
    assert h == hashlib.sha256(np.array([2, 100, 255999], dtype='<i4').tobytes()).hexdigest()
    assert h != ps.sha256_token_ids([2, 255999, 100])  # order matters
    with pytest.raises(ValueError):
        ps.sha256_token_ids([-1])


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / 'blob.bin'
    payload = bytes(range(256)) * 9000  # larger than one chunk below
    p.write_bytes(payload)
    assert ps.sha256_file(p, chunk_size=4096) == hashlib.sha256(payload).hexdigest()


def test_manifest_row_contains_hashes_not_text():
    row = ps.manifest_row(set_name='eval', rank=0, perm_position=7, line_index=123, source='Pile-CC',
                          text='hello world', text_ids=[11, 12], bos_id=2, max_tokens_including_bos=1024)
    assert list(row) == ps.MANIFEST_COLUMNS
    assert row['n_tokens_used'] == 3 and row['truncated'] is False
    assert row['token_ids_sha256'] == ps.sha256_token_ids([2, 11, 12])
    assert 'hello world' not in json.dumps(row)


# ----------------------------------------------------------------------------------------------- .jsonl.zst reading
def test_jsonl_zst_roundtrip_and_load_lines(tmp_path):
    zstd = pytest.importorskip('zstandard')
    docs = [{'text': f'doc {i}', 'meta': {'pile_set_name': 'Wikipedia (en)' if i % 2 else 'Pile-CC'}}
            for i in range(25)]
    raw = ''.join(json.dumps(d) + '\n' for d in docs).encode('utf-8')
    p = tmp_path / 'val.jsonl.zst'
    p.write_bytes(zstd.ZstdCompressor().compress(raw))
    assert ps.count_lines_jsonl_zst(p) == 25
    got = ps.load_lines(p, [3, 0, 24])
    assert got[3]['text'] == 'doc 3' and got[24]['meta']['pile_set_name'] == 'Pile-CC'
    assert ps.get_field(got[3], ['meta', 'pile_set_name']) == 'Wikipedia (en)'
    with pytest.raises(ValueError, match='beyond the end'):
        ps.load_lines(p, [30])

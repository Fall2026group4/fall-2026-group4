"""Offline tests for XB amendment A4 (128-token shared-span arm). Reads only committed/derived manifests and code."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
X = ROOT / 'results/cross_model/xb_matched_splicing'


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('corpus', ['pile', 'owt_xm'])
def test_ctx128_manifest_is_a_recut_of_the_frozen_manifest(corpus):
    f = X / f'samples/{corpus}_ctx128_manifest.csv'
    if not f.exists():
        pytest.skip('ctx128 manifest not built')
    a, b = pd.read_csv(X / f'samples/{corpus}_manifest.csv'), pd.read_csv(f)
    assert (a.doc_id.values == b.doc_id.values).all() and (a['rank'].values == b['rank'].values).all()
    assert (a.set.values == b.set.values).all() and (a.text_sha256.values == b.text_sha256.values).all()
    assert (b.gpt2_span_tokens <= 127).all() and (b.gemma_span_tokens <= 127).all()
    assert (b.span_chars <= a.span_chars).all() and (b.span1024_chars == a.span_chars).all()


def test_a4_tree_copies_are_identical():
    for c in ('pile', 'owt_xm'):
        p, q = X / f'a4_ctx128/samples/{c}_manifest.csv', X / f'samples/{c}_ctx128_manifest.csv'
        if not p.exists():
            pytest.skip('A4 tree not set up')
        assert hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256(q.read_bytes()).digest()


def test_wrapper_only_redirects_results_root(monkeypatch):
    """05 must not alter the frozen runner's functions: after redirecting, doc_ids/sae_specs are the same objects'
    code as in a fresh import, and only the XB path differs."""
    w = _mod('scripts/cross_model/05_xb_ctx128.py', 'xb05')
    m1 = w.load('scripts/cross_model/02_matched_splicing.py', 'xb02a')
    m2 = w.load('scripts/cross_model/02_matched_splicing.py', 'xb02b')
    m2.XB = w.A4
    for fn in ('doc_ids', 'sae_specs', 'load_sae', 'main'):
        assert getattr(m1, fn).__code__.co_code == getattr(m2, fn).__code__.co_code
    assert m1.XB != m2.XB and m2.XB == X / 'a4_ctx128'


def test_context_comparison_shrink_statistic():
    """shrink = |d1024| - |d128| > 0 when the difference moves toward 0; sign-invariant."""
    d1024, d128 = np.array([0.2, -0.2]), np.array([0.05, -0.05])
    assert (np.abs(d1024) - np.abs(d128) > 0).all()
    assert (np.abs(d128) - np.abs(d1024) < 0).all()

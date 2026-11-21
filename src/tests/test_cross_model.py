"""Unit tests for the final-week cross-model infrastructure (2026-11-21). Offline: no model or tokenizer download."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ backends
def test_apply_replaces_only_masked_positions_of_the_given_row():
    from src.models.backends import _apply
    h = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    mask = torch.tensor([False, True, True, False])
    out = _apply(h, [(1, lambda x: torch.zeros_like(x), mask)])
    assert torch.equal(out[0], h[0])
    assert torch.equal(out[1, [0, 3]], h[1, [0, 3]])
    assert torch.equal(out[1, [1, 2]], torch.zeros(2, 3))
    assert torch.equal(h, torch.arange(24, dtype=torch.float32).view(2, 4, 3))   # input untouched


def test_identity_splice_is_exact_in_bf16():
    from src.models.backends import _apply
    h = torch.randn(1, 5, 8).to(torch.bfloat16)
    out = _apply(h, [(0, lambda x: x, torch.ones(5, dtype=torch.bool))])
    assert torch.equal(out, h)


def test_depth_fractions_match_the_preregistered_triplets():
    from src.models.backends import GemmaBackend, GPT2Backend
    assert [GPT2Backend.depth_fraction(L) for L in (3, 6, 9)] == [0.25, 0.5, 0.75]
    g = [GemmaBackend.depth_fraction(L) for L in (5, 12, 19)]
    assert np.allclose(g, [6 / 26, 13 / 26, 20 / 26])
    assert abs(GPT2Backend.depth_fraction(8) - GemmaBackend.depth_fraction(16)) < 0.02


# ------------------------------------------------------------------------------------------------ samples
class _CharTok:
    """Fake tokenizer: one token per character (offsets exact), or per word."""

    def __init__(self, unit):
        self.unit = unit

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if self.unit == 'char':
            off = [(i, i + 1) for i in range(len(text))]
        else:
            off, i = [], 0
            for w in text.split(' '):
                off.append((i, i + len(w)))
                i += len(w) + 1
            off = [o for o in off if o[1] > o[0]]
        out = {'input_ids': list(range(len(off)))}
        if return_offsets_mapping:
            out['offset_mapping'] = off
        return out


def test_span_end_respects_the_limit_for_both_tokenizers(monkeypatch):
    m = _mod('scripts/cross_model/01_build_xm_samples.py', 'xm01')
    monkeypatch.setattr(m, 'MAXTXT', 10)
    text = 'aa bb cc dd ee ff gg hh ii jj kk ll mm'
    c = m.span_end(text, (_CharTok('char'), _CharTok('word')))
    assert len(_CharTok('char')(text[:c])['input_ids']) <= 10
    assert c == 10
    c2 = m.span_end('ab cd', (_CharTok('char'), _CharTok('word')))
    assert c2 == len('ab cd')


# ------------------------------------------------------------------------------------------------ analysis
def test_holm_matches_manual():
    m = _mod('scripts/cross_model/03_matched_splicing_analysis.py', 'xm03')
    adj = m.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_stats_from_sums_normalised_metrics():
    import pandas as pd
    m = _mod('scripts/cross_model/03_matched_splicing_analysis.py', 'xm03b')
    g = pd.DataFrame(dict(n_pred=[10, 10], ce_orig_sum=[20., 20.], ce_new_sum=[22., 22.], ce_rand_sum=[21., 21.],
                          ce_mean_sum=[40., 40.], kl_sum=[1., 1.], kl_mean_sum=[10., 10.], top1_agree=[9, 9],
                          n_tok=[11, 11], sse=[1., 1.], sum_sq=[10., 10.], cos_sum=[10., 10.], l0_sum=[110., 110.]))
    s = m.stats_from_sums(g)
    assert np.isclose(s['dce_sae'], 0.2) and np.isclose(s['dce_random'], 0.1)
    assert np.isclose(s['ratio_sae_random'], 2.0)
    assert np.isclose(s['loss_recovered'], (2.0 - 0.2) / 2.0)
    assert np.isclose(s['dce_rel'], 0.1) and np.isclose(s['excess_rel'], 0.05)
    assert np.isclose(s['kl_over_kl_mean'], 0.1) and np.isclose(s['l0'], 10.0)


# ------------------------------------------------------------------------------------------------ DX helpers
@pytest.fixture(scope='module')
def dx():
    return _mod('scripts/cross_model/10_truth_preservation.py', 'xm10')


def test_affine_alignment_recovers_a_linear_map(dx):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 6))
    A = rng.normal(size=(6, 6))
    H = X @ np.linalg.inv(A) + 3.0
    f = dx.fit_affine(H, X, lam_rel=1e-9)
    assert np.abs(f(H) - X).max() < 1e-3


def test_procrustes_recovers_a_rotation(dx):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 5))
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    H = X @ Q.T
    f = dx.fit_procrustes(H, X)
    assert np.abs(f(H) - X).max() < 1e-4


def test_cka_bounds(dx):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 8))
    assert np.isclose(dx.cka(X, X), 1.0)
    assert np.isclose(dx.cka(X, 2 * X + 1), 1.0)
    assert dx.cka(X, rng.normal(size=(500, 8))) < 0.2

"""Tests for src/evaluation/faithfulness_metrics.py and src/sae/checkpoint_registry.py.

Run from repository root:  python -m pytest -q src/tests/test_shared_metrics_registry.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src.evaluation import faithfulness_metrics as fm
from src.sae import checkpoint_registry as reg

ROOT = Path(__file__).resolve().parents[2]


def test_perfect_reconstruction():
    x = torch.randn(50, 8)
    m = fm.reconstruction_metrics(x, x, torch.full((50,), 3.0))
    assert m['mse'] == 0 and m['nmse'] == pytest.approx(0) and m['explained_variance'] == pytest.approx(1)
    assert m['cosine'] == pytest.approx(1) and m['l0'] == 3


def test_mean_reconstruction_has_zero_ev():
    x = torch.randn(200, 8) + 5
    m = fm.reconstruction_metrics(x, x.mean(0).expand_as(x))
    assert m['explained_variance'] == pytest.approx(0, abs=1e-9)
    assert m['nmse'] == pytest.approx(1, abs=1e-9)
    assert 0 < m['relative_sq_error'] < 1


def test_metrics_match_training_script_evaluate():
    spec = importlib.util.spec_from_file_location('o3', ROOT / 'scripts/openwebtext/03_training_sae.py')
    o3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o3)
    torch.manual_seed(0)
    model = o3.SAE(16, 64, 'topk', 4).eval()
    x = torch.randn(300, 16)
    ref = o3.evaluate(model, x, 'cpu', 64)
    x_hat, l0, _ = fm.reconstruct(model, x)
    m = fm.reconstruction_metrics(x, x_hat, l0)
    for k_ref, k in [('mse', 'mse'), ('explained_variance', 'explained_variance'), ('cosine', 'cosine'), ('l0', 'l0')]:
        assert m[k] == pytest.approx(ref[k_ref], rel=1e-5)


def test_fast_auroc_equals_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)
    assert fm._fast_auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_bootstrap_ci_brackets_gap_and_is_zero_for_identical_scores():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 100)
    groups = np.tile(np.arange(100), 2)
    s = y + rng.normal(size=200)
    lo, hi = fm.grouped_bootstrap_gap(y, s, s, groups, 200)
    assert lo == hi == 0
    noisy = s + rng.normal(scale=2, size=200)
    gap = roc_auc_score(y, s) - roc_auc_score(y, noisy)
    lo, hi = fm.grouped_bootstrap_gap(y, s, noisy, groups, 500)
    assert lo < gap < hi


def test_direction_fidelity():
    d = np.array([1.0, 0, 0])
    assert fm.direction_fidelity(d, 2 * d) == {'direction_cosine': 1.0, 'direction_norm_ratio': 2.0}
    X = np.array([[1.0, 0], [3, 0], [0, 0], [0, 0]])
    assert np.allclose(fm.diff_of_means_direction(X, np.array([1, 1, 0, 0])), [1, 0])


def test_load_sae_rejects_wrong_architecture(tmp_path):
    spec = importlib.util.spec_from_file_location('o3', ROOT / 'scripts/openwebtext/03_training_sae.py')
    o3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o3)
    p = tmp_path / 'c.pt'
    torch.save({'model_state_dict': o3.SAE(8, 16, 'relu', 2).state_dict()}, p)
    fm.load_sae(p, architecture='relu')
    with pytest.raises(ValueError):
        fm.load_sae(p, architecture='jumprelu')


def test_registry_classification():
    assert reg._classify('results/openwebtext/convergence/layer_3/relu_l0_128_uninterrupted_seed42/x.pt')[1] == 'AUTHORITATIVE'
    assert reg._classify('results/professor_review_corrections/owt_consistent_protocol/layer_2/x.pt')[1] == 'HISTORICAL_COPY'
    assert reg._classify('results/openwebtext/convergence/layer_2/relu_l0_128_warmstart_1000_to_15000/x.pt')[1] == 'SUPERSEDED'
    assert reg._classify('results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/smoke/layer_6/x.pt')[1] == 'SMOKE'


@pytest.mark.skipif(not reg.REGISTRY_CSV.exists(), reason='registry not built')
def test_registry_has_all_uninterrupted_runs():
    df = reg.load_registry()
    u = df[df.family == 'owt_uninterrupted']
    assert len(u) == 108 and u[['layer', 'architecture', 'seed']].drop_duplicates().shape[0] == 108
    assert (u.d_sae == 3072).all() and u.sha256.str.len().eq(64).all()


def test_splicer_identity_and_mask_positions():
    """The splice hook rewrites only the requested positions; an identity replacement changes nothing."""
    import importlib.util as iu
    spec = iu.spec_from_file_location('splice', ROOT / 'scripts/shared/spliced_reconstruction_evaluation.py')
    sp_mod = iu.module_from_spec(spec)
    spec.loader.exec_module(sp_mod)
    seen = {}

    class FakeModel:
        def run_with_hooks(self, tokens, fwd_hooks):
            resid = torch.zeros(*tokens.shape, 4)
            for _, fn in fwd_hooks:
                resid = fn(resid, None)
            seen['resid'] = resid
            return resid.sum(-1, keepdim=True).expand(*tokens.shape, 3).clone()

    s = sp_mod.Splicer(FakeModel(), 'cpu')
    toks, lens = sp_mod.pad([[1, 2, 3, 4], [1, 2]], 0)
    s.logprobs(toks, lens, 0, lambda x: x + 1, 'all')
    assert seen['resid'][..., 0].tolist() == [[0, 1, 1, 1], [0, 1, 0, 0]]   # BOS and padding untouched
    s.logprobs(toks, lens, 0, lambda x: x + 1, 'final')
    assert seen['resid'][..., 0].tolist() == [[0, 0, 0, 1], [0, 1, 0, 0]]
    a = s.logprobs(toks, lens)
    b = s.logprobs(toks, lens, 0, lambda x: x.clone(), 'all')
    assert torch.equal(a, b)

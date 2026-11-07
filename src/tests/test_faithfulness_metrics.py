"""Tests for computations used in the professor-review corrections.

Run from repository root:  python -m pytest -q src/tests/test_faithfulness_metrics.py  (formerly test_professor_review.py)
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope='module')
def causal():
    return _load('scripts/repe/13_causal_random_controls.py', 'causal13')


@pytest.fixture(scope='module')
def owt():
    return _load('scripts/openwebtext/03_training_sae.py', 'owt03')


def test_kl_zero_for_identical_and_positive_otherwise(causal):
    lp = torch.log_softmax(torch.randn(5, 11), -1)
    assert torch.allclose(causal.kl_rows(lp, lp), torch.zeros(5), atol=1e-6)
    other = torch.log_softmax(torch.randn(5, 11), -1)
    assert (causal.kl_rows(lp, other) > 0).all()


def test_recovery_endpoints(causal):
    before = np.array([0.2, 0.1, 0.3])
    assert causal.recovery(before, before) == pytest.approx(0.0)
    assert causal.recovery(np.zeros(3), before) == pytest.approx(1.0)
    assert causal.recovery(2 * before, before) == pytest.approx(-1.0)


def test_empirical_p_bounds(causal):
    controls = np.arange(200, dtype=float)
    assert causal.empirical_p(1e9, controls) == pytest.approx(1 / 201)     # best possible with 200 controls
    assert causal.empirical_p(-1e9, controls) == pytest.approx(1.0)
    assert causal.empirical_p(199.5, controls) == pytest.approx(1 / 201)
    assert causal.empirical_p(150, controls) == pytest.approx((1 + 50) / 201)


def test_bootstrap_ci_contains_point_estimate(causal):
    rng = np.random.default_rng(1)
    before = rng.uniform(0.01, 0.05, 90)
    after = before * rng.uniform(0.7, 0.9, 90)
    lo, hi = causal.bootstrap_recovery_ci(after, before)
    assert lo < causal.recovery(after, before) < hi


def test_topk_sae_l0_exact_and_decoder_unit_norm(owt):
    torch.manual_seed(0)
    sae = owt.SAE(16, 64, 'topk', k=5)
    _, z, _ = sae(torch.randn(32, 16))
    assert ((z > 0).sum(-1) <= 5).all()
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(64), atol=1e-5)


def test_evaluate_explained_variance_matches_numpy(owt):
    torch.manual_seed(0)
    sae = owt.SAE(8, 32, 'relu', k=4)
    data = torch.randn(300, 8) * 3 + 1
    m = owt.evaluate(sae, data, torch.device('cpu'), batch_size=64)
    with torch.no_grad():
        rec, _, _ = sae(data)
    err = ((rec - data) ** 2).sum().item()
    var = ((data - data.mean(0)) ** 2).sum().item()
    assert m['explained_variance'] == pytest.approx(1 - err / var, rel=1e-5)
    assert m['mse'] == pytest.approx(err / data.numel(), rel=1e-5)


def test_streaming_variance_formula_matches_direct():
    """10_reconstruction_evaluation.py uses sum(x^2) - ||sum x||^2 / n as total variance."""
    x = torch.randn(1000, 7, dtype=torch.float64) * 5 + 3
    streaming = x.square().sum() - x.sum(0).square().sum() / len(x)
    direct = (x - x.mean(0)).square().sum()
    assert float(streaming) == pytest.approx(float(direct), rel=1e-9)


def test_final_token_hook_only_changes_final_position(causal):
    """Runner.run adds the intervention at the final non-padding token only (right padding).

    blocks.6.ln1.hook_normalized is a per-position function of blocks.6.hook_resid_pre, so it changes
    exactly at the positions where the intervention was applied.
    """
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    cfg = HookedTransformerConfig(n_layers=8, d_model=16, n_ctx=12, d_head=4, n_heads=4, d_vocab=50,
                                  act_fn='gelu', normalization_type='LN', d_mlp=32, device='cpu')
    torch.manual_seed(0)
    model = HookedTransformer(cfg)
    model.eval()
    r = causal.Runner.__new__(causal.Runner)
    r.model = model
    r.tokens = torch.tensor([[1, 5, 7, 9, 0, 0], [1, 3, 4, 6, 8, 2]])
    r.lengths = torch.tensor([4, 6])
    r.final, r.rows = r.lengths - 1, torch.arange(2)
    seen = []
    model.add_hook('blocks.6.ln1.hook_normalized', lambda x, hook: seen.append(x.clone()), is_permanent=True)
    r.run()
    r.run(delta=torch.randn(16))
    model.reset_hooks(including_permanent=True)
    assert len(seen) % 2 == 0
    per_run = len(seen) // 2   # ln1 output is hooked once per q/k/v input
    diff = (seen[per_run] - seen[0]).abs().sum(-1)
    changed = diff > 1e-6
    expected = torch.zeros_like(changed)
    expected[0, 3] = expected[1, 5] = True
    assert torch.equal(changed, expected)

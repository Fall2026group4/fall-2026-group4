"""Unit tests for src/sae/truth_subspace.py (TSP study, 2026-11-14)."""
import numpy as np
import torch

from src.sae import truth_subspace as ts


def _synthetic(n=600, d=32, seed=0):
    rng = np.random.default_rng(seed)
    u = np.zeros(d)
    u[3] = 1.0                                   # planted truth direction
    v = np.zeros(d)
    v[7] = 1.0                                   # planted polarity direction (truth XOR negation lives in the pair)
    y = rng.integers(0, 2, n)
    pol = rng.integers(0, 2, n)
    sign = np.where(pol == 1, -1.0, 1.0)
    X = rng.standard_normal((n, d)) * 0.5 + np.outer((2 * y - 1) * sign, u) * 1.5 + np.outer(pol, v)
    groups = np.repeat(np.arange(n // 4), 4)
    return X, y, pol, groups, u


def test_subspace_orthonormal_and_rank():
    X, y, pol, g, _ = _synthetic()
    P, info = ts.estimate_truth_subspace(X, y, pol, g, r=4, n_boot=6)
    assert P.shape == (32, 4)
    assert np.allclose(P.T @ P, np.eye(4), atol=1e-8)
    assert info['rank'] == 4


def test_subspace_recovers_planted_direction():
    X, y, pol, g, u = _synthetic()
    P, _ = ts.estimate_truth_subspace(X, y, pol, g, r=3, n_boot=6)
    assert np.linalg.norm(P.T @ u) > 0.95        # u lies (almost) inside span(P)


def test_random_and_pca_controls_orthonormal():
    X, *_ = _synthetic()
    for Q in (ts.random_subspace(32, 5, seed=1), ts.pca_subspace(X, 5)):
        assert np.allclose(Q.T @ Q, np.eye(5), atol=1e-8)


def test_rel_error_endpoints():
    X, *_ = _synthetic()
    P = ts.random_subspace(32, 4, seed=2)
    assert ts.subspace_rel_error(X, X, P) == 0.0
    mean_recon = np.repeat(X.mean(0, keepdims=True), len(X), 0)
    assert abs(ts.subspace_rel_error(X, mean_recon, P) - 1.0) < 1e-10


def test_variance_share():
    d = 16
    assert abs(ts.variance_share(np.eye(d), ts.random_subspace(d, 3)) - 1.0) < 1e-10
    cov = np.diag(np.r_[np.full(8, 10.0), np.full(8, 0.1)])
    low = np.eye(d)[:, 8:11]
    assert ts.variance_share(cov, low) < 0.05


def test_subspace_loss_gradcheck():
    torch.manual_seed(0)
    P = torch.tensor(ts.random_subspace(6, 2, seed=3))
    x = torch.randn(5, 6, dtype=torch.float64)
    xh = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda h: ts.subspace_loss(x, h, P, torch.tensor(2.0, dtype=torch.float64)), (xh,))


def test_subspace_loss_ignores_orthogonal_error():
    P = torch.tensor(np.eye(6)[:, :2])
    x = torch.zeros(4, 6, dtype=torch.float64)
    xh = torch.zeros(4, 6, dtype=torch.float64)
    xh[:, 3:] = 5.0                               # error only outside span(P)
    assert float(ts.subspace_loss(x, xh, P, torch.tensor(1.0, dtype=torch.float64))) == 0.0

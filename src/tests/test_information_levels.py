import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation import information_levels as il


def _data(n=300, d=10, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, d)) + y[:, None] * 0.8
    g = np.arange(n) // 2
    return X, y, g


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.integers(0, 5, 200).astype(float)
    assert abs(il.auroc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_linear_form_and_exact_error_identity():
    X, y, g = _data()
    pipe, _ = il.fit_logistic(X, y, g)
    a, c = il.linear_form(pipe)
    assert np.allclose(X @ a + c, pipe.decision_function(X), atol=1e-9)
    xh = X * 0.7
    e = X - xh
    assert np.allclose(xh @ a + e @ a, X @ a, atol=1e-9)        # w.x = w.x_hat + w.e


def test_noise_has_matched_norms():
    e = np.random.default_rng(2).standard_normal((50, 8)).astype(np.float32)
    eta = il.norm_matched_noise(e, 3, np.random.default_rng(3))
    assert eta.shape == (3, 50, 8)
    assert np.allclose(np.linalg.norm(eta, axis=-1), np.linalg.norm(e, axis=-1)[None], rtol=1e-5)


def test_pca_k_matches_target_on_validation():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((2000, 6)) * np.array([5, 3, 2, 1, .5, .1])
    mu, cov = X.mean(0), np.cov(X.T, bias=True)
    w, V = il.pca_basis(mu, cov)
    k = il.pca_k_for_val_nmse(X, mu, V, 0.05)
    xh = il.pca_reconstruct(X, mu, V, k)
    nm = ((X - xh) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
    assert nm <= 0.05
    if k > 1:
        xh2 = il.pca_reconstruct(X, mu, V, k - 1)
        assert ((X - xh2) ** 2).sum() / ((X - X.mean(0)) ** 2).sum() > 0.05


def _cell(**kw):
    s = dict(informative=True, frozen_massmean_gap_ci=(0.05, 0.1), frozen_logistic_gap_ci=(0.1, 0.2),
             refit_gap_ci={f: (-0.05, 0.05) for f in il.FAMILIES}, original_auroc={f: 0.9 for f in il.FAMILIES},
             lc_sample_inefficient=False, refit_logistic_gap=0.1, pca_refit_logistic_gap=0.05)
    s.update(kw)
    return s


def test_labels():
    assert il.label_cell(_cell(informative=False))[0].startswith('Uninformative')
    assert il.label_cell(_cell(frozen_massmean_gap_ci=(-0.01, 0.01),
                               refit_gap_ci={f: (-0.01, 0.015) for f in il.FAMILIES}))[0] == 'Preserved'
    assert il.label_cell(_cell(refit_gap_ci={f: (-0.03, 0.01) for f in il.FAMILIES}))[0] == 'Readout shifted'
    allred = {f: (0.03, 0.1) for f in il.FAMILIES}
    assert il.label_cell(_cell(refit_gap_ci=allred))[0] == 'Not recovered by tested families'
    # same but PCA loses more -> not "not recovered", only reduced
    assert il.label_cell(_cell(refit_gap_ci=allred, pca_refit_logistic_gap=0.2))[0].startswith('Reduced for')
    assert il.label_cell(_cell(refit_gap_ci=allred, lc_sample_inefficient=True))[0].startswith('Reduced for')
    one = {f: ((0.03, 0.1) if f == 'mlp' else (-0.05, 0.05)) for f in il.FAMILIES}
    assert il.label_cell(_cell(refit_gap_ci=one, frozen_logistic_gap_ci=(-0.1, 0.1)))[0] == 'Reduced for mlp'


def test_bh_values():
    q = il.bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])

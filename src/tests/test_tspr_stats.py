import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.evaluation import tspr_stats as T


def test_weighted_auroc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)                 # many ties
    assert abs(T.weighted_auroc(y, s) - roc_auc_score(y, s)) < 1e-12
    W = T.group_bootstrap_counts(np.arange(300) // 4, 20, rng)
    got = T.weighted_auroc(y, s, W)
    for b in range(20):
        idx = np.repeat(np.arange(300), W[b])
        assert abs(got[b] - roc_auc_score(y[idx], s[idx])) < 1e-12


def test_group_bootstrap_keeps_groups_whole():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(50), 4)
    W = T.group_bootstrap_counts(g, 10, rng)
    assert W.shape == (10, 200) and (W.sum(1) == 200).all()
    for b in range(10):
        assert all(len(set(W[b][g == k])) == 1 for k in range(50))


def test_hierarchical_satterthwaite_and_ci():
    d = np.array([-0.02, -0.03, -0.01, -0.025, -0.015, -0.02, -0.018, -0.022])
    boot = np.random.default_rng(2).normal(d.mean(), 0.004, 2000)
    h = T.hierarchical(d, boot)
    se_seed = d.std(ddof=1) / np.sqrt(8)
    se = np.sqrt(se_seed ** 2 + np.var(boot, ddof=1))
    assert np.isclose(h['se'], se)
    assert np.isclose(h['df'], se ** 4 / (se_seed ** 4 / 7))
    assert np.isclose(h['ci_high'], d.mean() + stats.t.ppf(0.975, h['df']) * se)
    assert h['df'] >= 7 and h['ci_high'] < 0


def test_gatekeeping_and_holm():
    assert np.allclose(T.holm([0.01, 0.04]), [0.02, 0.04])
    g = T.gatekeeping(dict(estimate=-0.02, p=0.001), dict(estimate=-0.01, p=0.03), dict(estimate=-0.02, p=0.01))
    assert g['H1']['rejected'] and g['H2']['rejected'] and g['H3']['rejected']
    assert np.isclose(g['H2']['p_adj'], 0.03) and np.isclose(g['H3']['p_adj'], 0.02)
    g = T.gatekeeping(dict(estimate=-0.02, p=0.2), dict(estimate=-0.01, p=0.0001), dict(estimate=-0.02, p=0.0001))
    assert not any(v['rejected'] for v in g.values())          # gate closed
    g = T.gatekeeping(dict(estimate=0.02, p=0.001), dict(estimate=-0.01, p=0.001), dict(estimate=-0.02, p=0.001))
    assert not g['H1']['rejected']                              # wrong direction


def test_labels():
    base = dict(ci90_low=-0.02, ci90_high=0.0)
    assert T.h1_label(dict(base, estimate=-0.015, ci_low=-0.02, ci_high=-0.005)) == 'SUPPORTED (meaningful)'
    assert T.h1_label(dict(base, estimate=-0.005, ci_low=-0.008, ci_high=-0.001)) == 'SUPPORTED, below SESOI'
    assert T.h1_label(dict(estimate=0.0, ci_low=-0.012, ci_high=0.012, ci90_low=-0.009, ci90_high=0.009)) == \
        'NO MEANINGFUL EFFECT'
    assert T.h1_label(dict(estimate=0.02, ci_low=0.01, ci_high=0.03, ci90_low=0.012, ci90_high=0.028)) == 'HARM'
    assert T.h1_label(dict(estimate=-0.01, ci_low=-0.03, ci_high=0.01, ci90_low=-0.027, ci90_high=0.007)) == \
        'INCONCLUSIVE'


def test_non_inferiority():
    assert T.non_inferiority(dict(upper95_one_sided=0.01), 0.02)['passed']
    assert not T.non_inferiority(dict(upper95_one_sided=0.03), 0.02)['passed']
    assert T.non_inferiority(dict(lower95_one_sided=-0.01), 0.02, higher_is_worse=False)['passed']
    assert not T.non_inferiority(dict(lower95_one_sided=-0.03), 0.02, higher_is_worse=False)['passed']


def test_hierarchical_zero_variance():
    h = T.hierarchical(np.zeros(8), np.zeros(100))
    assert h['p'] == 1.0 and h['p_pilot_rule_t'] == 1.0 and h['se'] == 0
    assert T.h1_label(h) == 'NO MEANINGFUL EFFECT'

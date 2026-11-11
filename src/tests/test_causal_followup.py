"""Unit tests for the causal follow-up scripts:

    scripts/repe/14_causal_reanalysis.py
    scripts/repe/15_sae_in_the_loop_causal.py            (needs torch; skipped without it)
    scripts/shared/frozen_probe_error_decomposition.py

All inputs are synthetic. No model, checkpoint or dataset is loaded.
Run from the repository root:  python -m pytest -q src/tests/test_causal_followup.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


re14 = load('scripts/repe/14_causal_reanalysis.py', 'repe14')
dec = load('scripts/shared/frozen_probe_error_decomposition.py', 'frozen_dec')


# ---------------------------------------------------------------- script 14
def test_empirical_p_values():
    controls = np.arange(1, 201) / 1000.0                    # 0.001 .. 0.200
    assert re14.empirical_p_upper(1.0, controls) == pytest.approx(1 / 201)       # floor: no control above
    assert re14.empirical_p_lower(1.0, controls) == pytest.approx(1.0)
    assert re14.empirical_p_upper(0.1905, controls) == pytest.approx(11 / 201)   # 10 controls above
    assert re14.empirical_p_two_sided(-1.0, controls) == pytest.approx(2 / 201)  # worse than every control
    assert re14.empirical_p_two_sided(0.1005, controls) == pytest.approx(1.0)    # middle of the controls


def test_bh_adjust_matches_hand_computation_and_keeps_order():
    p = np.array([0.04, 0.001, 0.03, 0.5])
    # sorted 0.001, 0.03, 0.04, 0.5 -> 0.004, 0.06, 0.0533.., 0.5 -> monotone from the top: 0.004, 0.0533, 0.0533, 0.5
    assert re14.bh_adjust(p) == pytest.approx([0.16 / 3, 0.004, 0.16 / 3, 0.5])
    assert (re14.bh_adjust(p) >= p - 1e-15).all()
    assert re14.bh_adjust(np.array([])).size == 0
    with pytest.raises(ValueError):
        re14.bh_adjust(np.array([0.1, np.nan]))


def test_power_helpers():
    assert re14.min_attainable_p(200) == pytest.approx(1 / 201)
    n = re14.controls_needed_bonferroni(27)
    assert 1 / (n + 1) <= 0.05 / 27 < 1 / n                  # n is the smallest sufficient number of controls
    assert n == 539
    k = re14.min_floor_discoveries_for_bh(27, 200)
    assert k == 3
    # k targets at the floor pass BH, k - 1 do not
    assert re14.bh_adjust(np.array([1 / 201] * k + [1.0] * (27 - k))).min() <= 0.05
    assert re14.bh_adjust(np.array([1 / 201] * (k - 1) + [1.0] * (27 - k + 1))).min() > 0.05


def test_minimum_detectable_effect_is_the_exact_threshold():
    rng = np.random.default_rng(0)
    controls = rng.normal(size=200)
    mde = re14.minimum_detectable_effect(controls)
    assert re14.empirical_p_upper(mde + 1e-9, controls) <= 0.05          # just above: detectable
    assert re14.empirical_p_upper(mde - 1e-9, controls) > 0.05           # just below: not detectable
    assert np.isnan(re14.minimum_detectable_effect(controls[:10]))       # 10 controls cannot reach 0.05


def test_paired_bootstrap_difference_sign_and_zero():
    rng = np.random.default_rng(1)
    before = rng.uniform(0.5, 1.5, size=90)
    a, b = before * 0.5, before * 0.9                                    # a recovers 0.5, b recovers 0.1
    pt, lo, hi = re14.paired_bootstrap_difference(a, b, before)
    assert pt == pytest.approx(0.4) and lo == pytest.approx(0.4) and hi == pytest.approx(0.4)
    pt, lo, hi = re14.paired_bootstrap_difference(a, a, before)
    assert pt == 0 and lo == 0 and hi == 0


def test_reanalysis_reproduces_saved_p_values_and_counts():
    if not (REPO / re14.SRC / 'causal_controls_200.csv').exists():
        pytest.skip('saved causal results not present')
    res, cdf, power, _ = re14.analyse(REPO / re14.SRC)                   # raises if a saved p-value is not reproduced
    assert len(res) == 28 and (res['family'] == 'sae_27').sum() == 27
    assert power['n_controls_per_group'] == 200
    assert res.loc[res['family'] == 'sae_27', 'p_two_sided_bh'].between(0, 1).all()
    assert (res['p_two_sided'] >= np.minimum(res['p_upper_script13'], res['p_lower']) - 1e-12).all()
    assert len(cdf) == 19


# ---------------------------------------------------------------- frozen-probe error decomposition
def test_auroc_matches_sklearn_including_ties():
    sk = pytest.importorskip('sklearn.metrics')
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=300)
    s = np.round(rng.normal(size=300) + y, 1)                            # rounding creates ties
    assert dec.auroc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-12)
    assert np.isnan(dec.auroc(np.ones(5), np.arange(5)))


def test_decompose_scores_identity_and_pure_cases():
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1], 200)
    s = rng.normal(size=400) + 2.0 * y
    same = dec.decompose_scores(y, s, s)
    assert same['signal_retention'] == pytest.approx(1) and same['noise_inflation'] == pytest.approx(1)
    assert same['auroc_signal_only'] == pytest.approx(dec.auroc(y, s))
    assert same['auroc_noise_only'] == pytest.approx(dec.auroc(y, s))
    # pure signal loss: class means collapse, scatter unchanged
    centred = s - np.where(y == 1, s[y == 1].mean(), s[y == 0].mean())
    lost = dec.decompose_scores(y, s, centred)
    assert lost['signal_retention'] == pytest.approx(0, abs=1e-12) and lost['noise_inflation'] == pytest.approx(1)
    assert lost['auroc_noise_only'] == pytest.approx(dec.auroc(y, s))    # scatter unchanged -> no loss from noise
    assert lost['auroc_signal_only'] < 0.6                               # the loss is all signal
    # pure added scatter: class means kept, within-class spread tripled
    noisy = np.where(y == 1, s[y == 1].mean(), s[y == 0].mean()) + 3 * centred
    added = dec.decompose_scores(y, s, noisy)
    assert added['signal_retention'] == pytest.approx(1) and added['noise_inflation'] == pytest.approx(3)
    assert added['auroc_signal_only'] == pytest.approx(dec.auroc(y, s))
    assert added['auroc_noise_only'] < dec.auroc(y, s)


def test_norm_matched_noise_and_anisotropy():
    rng = np.random.default_rng(4)
    err = rng.normal(size=(500, 64)) * rng.uniform(0.1, 5, size=(500, 1))
    noise = dec.norm_matched_noise(err, np.random.default_rng(5))
    assert np.linalg.norm(noise, axis=1) == pytest.approx(np.linalg.norm(err, axis=1))
    w = rng.normal(size=64)
    assert dec.error_anisotropy(rng.normal(size=(20000, 64)), w) == pytest.approx(1, rel=0.1)   # isotropic
    along = np.outer(rng.normal(size=500), w)
    assert dec.error_anisotropy(along, w) == pytest.approx(64)                                   # all on w
    assert np.isnan(dec.error_anisotropy(err, np.zeros(64)))
    assert dec.empirical_p_le(0.5, np.array([0.6, 0.7, 0.8])) == pytest.approx(1 / 4)


def test_probe_direction_raw_reproduces_decision_function():
    pytest.importorskip('sklearn')
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 8)) * rng.uniform(0.5, 3, size=8)
    y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    w = dec.probe_direction_raw(p)
    d = p.decision_function(X)
    assert np.corrcoef(X @ w, d)[0, 1] == pytest.approx(1)               # same ranking up to a constant shift


# ---------------------------------------------------------------- script 15 (torch)
@pytest.fixture(scope='module')
def loop():
    pytest.importorskip('torch')
    return load('scripts/repe/15_sae_in_the_loop_causal.py', 'repe15')


def test_script15_statistics_agree_with_script14(loop):
    rng = np.random.default_rng(7)
    controls, p = rng.normal(size=600), rng.uniform(size=40)
    for t in (-3.0, 0.0, 0.5, 3.0):
        assert loop.empirical_p_upper(t, controls) == re14.empirical_p_upper(t, controls)
        assert loop.empirical_p_two_sided(t, controls) == re14.empirical_p_two_sided(t, controls)
    assert loop.bh_adjust(p) == pytest.approx(re14.bh_adjust(p))
    assert loop.N_CONTROLS >= re14.controls_needed_bonferroni(27)        # one test can pass Bonferroni over 27


def test_recovery_and_kl(loop):
    import torch
    logp = torch.log_softmax(torch.randn(5, 11, generator=torch.Generator().manual_seed(0)), -1)
    assert torch.allclose(loop.kl_rows(logp, logp), torch.zeros(5), atol=1e-6)
    other = torch.log_softmax(torch.randn(5, 11, generator=torch.Generator().manual_seed(1)), -1)
    assert (loop.kl_rows(logp, other) > 0).all()
    before = np.array([1.0, 2.0, 3.0])
    assert loop.recovery(before, before) == pytest.approx(0)
    assert loop.recovery(np.zeros(3), before) == pytest.approx(1)
    assert loop.recovery(2 * before, before) == pytest.approx(-1)        # moved away from the honest output


def test_latent_selection_and_patches(loop):
    import torch
    g = torch.Generator().manual_seed(0)
    z = torch.rand(40, 12, generator=g)
    labels = np.repeat([0, 1], 20)
    z[labels == 1, 3] += 5.0                                             # latent 3: strongly honest
    z[labels == 1, 7] -= 2.0                                             # latent 7: untruthful
    z = z.clamp(min=0)
    top, dmu = loop.select_honesty_latents(z, labels, 2)
    assert top.tolist() == [3, 7] and dmu[3] > 0 and dmu[7] < 0
    D = torch.randn(6, 12, generator=g)
    z_u, z_h = z[:20], z[20:]
    full = loop.latent_patch_delta(z_h, z_u, D)
    assert torch.allclose(full, (z_h - z_u) @ D.T)
    parts = loop.latent_patch_delta(z_h, z_u, D, [3, 7]) + loop.latent_patch_delta(z_h, z_u, D, [i for i in range(12) if i not in (3, 7)])
    assert torch.allclose(parts, full, atol=1e-5)                        # subsets add up to the full latent patch
    assert torch.allclose(loop.latent_patch_delta(z_u, z_u, D), torch.zeros(20, 6))


def test_decomposition_identity_with_the_in_house_sae_class(loop):
    import torch
    from src.evaluation import faithfulness_metrics as fm
    torch.manual_seed(0)
    x_u, x_h = torch.randn(9, 768) * 3, torch.randn(9, 768) * 3
    for arch in ('relu', 'topk', 'jumprelu'):
        sae = fm.sae_class()(768, 3072, arch, 128).eval()
        with torch.inference_mode():
            rec_u, z_u, _ = sae(x_u)
            rec_h, z_h, _ = sae(x_h)
            D = sae.decoder.weight
            assert loop.decomposition_error(x_u, x_h, rec_u, rec_h, z_u, z_h, D) < 1e-3
            # the latent patch plus the error-term patch is exactly the activation patch
            total = loop.latent_patch_delta(z_h, z_u, D) + ((x_h - rec_h) - (x_u - rec_u))
            assert torch.allclose(total, x_h - x_u, atol=1e-3)


def test_control_sets(loop):
    import torch
    g = torch.Generator().manual_seed(0)
    alive = list(range(0, 300, 2))                                       # 150 alive latents
    mean_act = torch.arange(300, dtype=torch.float32)
    selected = [100, 102, 200]
    order = loop.activation_rank_order(alive, mean_act)
    for _ in range(50):
        m = loop.activation_matched_set(selected, alive, mean_act, g, window=5, order=order)
        assert len(set(m.tolist())) == 3 and not set(m.tolist()) & set(selected)
        assert set(m.tolist()) <= set(alive)
        for s, j in zip(selected, m.tolist()):
            assert abs(order.index(s) - order.index(j)) <= 5 + 2        # close in activation rank
        r = loop.random_latent_set(alive, 10, g, exclude=selected)
        assert len(set(r.tolist())) == 10 and not set(r.tolist()) & set(selected) and set(r.tolist()) <= set(alive)
    with pytest.raises(ValueError):
        loop.activation_matched_set([1], alive, mean_act, g)             # latent 1 is not alive
    norms = torch.tensor([1.0, 2.0, 3.0])
    d = loop.norm_matched_delta(norms, loop.unit(torch.randn(8, generator=g)))
    assert torch.allclose(d.norm(dim=-1), norms, atol=1e-6)


def test_minimum_detectable_dose(loop):
    lams = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    assert loop.minimum_detectable_dose(lams, [0.9, 0.4, 0.2, 0.04, 0.01, 0.002, 0.002]) == 0.1
    assert loop.minimum_detectable_dose(lams, [0.01, 0.4, 0.2, 0.04, 0.01, 0.002, 0.002]) == 0.1   # isolated small p ignored
    assert np.isnan(loop.minimum_detectable_dose(lams, [0.9] * 7))
    assert loop.control_seed(6, 'a') == loop.control_seed(6, 'a') != loop.control_seed(6, 'b')


def test_summarise_families_and_shares(loop):
    import pandas as pd
    rng = np.random.default_rng(8)
    n, n_ctrl = 30, 40
    kl0 = rng.uniform(0.5, 1.5, size=n)
    store = {'L6__baseline': kl0, 'L6__patch_lam1.0': kl0 * 0.5,          # activation patch recovers 0.5
             'L6__relu_s42__latent_all': kl0 * 0.75,                      # latents recover 0.25
             'L6__relu_s42__latent_top10': kl0 * 0.9}                     # top-10 recover 0.10
    base = dict(model='gpt2-small', layer=6, sae_architecture='relu', sae_seed=42, detail='', sign=0,
                mean_delta_norm=1.0, n_statements=n, baseline_sum_KL=kl0.sum(), intervened_sum_KL=0.0)
    rows = [dict(base, part='A', intervention_kind='target', intervention='activation_patch',
                 control_group='L6__graded_lam1.0', sae_architecture='', sae_seed='', detail='lambda=1.0', effect_E=0.5),
            dict(base, part='B', intervention_kind='target', intervention='latent_all',
                 control_group='L6__relu_s42__latent_all__gaussian', effect_E=0.25),
            dict(base, part='B', intervention_kind='target', intervention='latent_top10',
                 control_group='L6__relu_s42__latent_top10__gaussian', effect_E=0.10)]
    for grp, kind in (('L6__graded_lam1.0', 'gaussian_matched'), ('L6__relu_s42__latent_all__gaussian', 'gaussian_matched'),
                      ('L6__relu_s42__latent_top10__gaussian', 'gaussian_matched'),
                      ('L6__relu_s42__latent_top10__random', 'random_latents'),
                      ('L6__relu_s42__latent_top10__matched', 'activation_matched_latents')):
        for e in rng.normal(0, 0.01, size=n_ctrl):
            rows.append(dict(base, part='A' if 'graded' in grp else 'B',
                             intervention_kind='control', intervention=kind, control_group=grp, effect_E=e))
    out = loop.summarise(pd.DataFrame(rows), store)
    assert len(out) == 5                                                 # patch, latent_all, top10 x 3 control groups
    la = out[out['intervention'] == 'latent_all'].iloc[0]
    assert la['family'] == 'B1_latent_all' and la['p_upper'] == pytest.approx(1 / (n_ctrl + 1))
    assert la['share_of_activation_patch'] == pytest.approx(0.5)         # 0.25 / 0.5
    t10 = out[out['intervention'] == 'latent_top10'].set_index('control_group_kind')
    assert t10.loc['random_latents', 'family'] == 'B3_latent_topk_vs_random'
    assert t10.loc['activation_matched_latents', 'family'] == 'B4_latent_topk_vs_matched'
    assert pd.isna(t10.loc['gaussian_matched', 'family'])                # descriptive, not in a family
    assert t10.loc['random_latents', 'share_of_latent_all'] == pytest.approx(0.4)   # 0.10 / 0.25
    assert out['p_bh'].notna().sum() == 4

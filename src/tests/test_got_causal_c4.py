"""Unit tests for the GPT-2 GoT causal truth-direction test (C4) and the script-15 helpers it imports.

    scripts/geometry_of_truth/22_truth_direction_causal_test.py
    scripts/repe/15_sae_in_the_loop_causal.py   (code dependency only: statistics and latent helpers)

Written by Dhruv Rai on branch `dhruv-causal` (commits d9e14d2..989d313, 2026-11-10). Selectively extracted on
2026-11-26: the C4 tests below are copied verbatim from `src/tests/test_followup_part2.py`, and the helper tests
verbatim from `src/tests/test_causal_followup.py`. The other tests in those files exercise scripts that were not
integrated (sparsity-matched preservation, public-SAE splicing, follow-up figures, script-14 reanalysis,
frozen-probe error decomposition), so they are not included. Synthetic inputs only.
Run from the repository root:  python -m pytest -q src/tests/test_got_causal_c4.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
REPO = Path(__file__).resolve().parents[2]


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


got = load('scripts/geometry_of_truth/22_truth_direction_causal_test.py', 'got22')


@pytest.fixture(scope='module')
def loop():
    return load('scripts/repe/15_sae_in_the_loop_causal.py', 'repe15')


# ---------------------------------------------------------------- Geometry of Truth causal test (verbatim)
def test_divergence():
    assert got.divergence([1, 2, 3, 9], [1, 2, 4, 9]) == ([1, 2], 3, 4)
    assert got.divergence([1, 2, 5, 6], [1, 2, 5, 7]) == ([1, 2, 5], 6, 7)  # " South Africa" vs " South Korea"
    assert got.divergence([1, 2], [1, 2]) is None
    assert got.divergence([1, 2], [1, 2, 3]) is None


def test_build_pairs():
    df = pd.DataFrame({'statement': ['A is in X.', 'A is in Y.', 'B is in X.', 'B is in X.', 'C is in Z.', 'C is in W.'],
                       'label': [1, 0, 1, 0, 1, 0]})
    splits = pd.DataFrame({'row_id': range(6), 'group_id': ['A', 'A', 'B', 'B', 'C', 'C'], 'label': [1, 0, 1, 0, 1, 0],
                           'split': ['train', 'train', 'val', 'val', 'test', 'test']})
    vocab = {}
    enc = lambda s: [vocab.setdefault(w, len(vocab) + 1) for w in s.split()]
    pairs, dropped = got.build_pairs(df, splits, enc)
    assert dropped == 1 and list(pairs['group_id']) == ['A', 'C']          # B has identical statements
    a = pairs.set_index('group_id').loc['A']
    assert a['prefix'] == enc('A is in') and a['c'] == enc('X.')[0] and a['w'] == enc('Y.')[0]
    bad = splits.copy()
    bad.loc[0, 'label'] = 0
    with pytest.raises(ValueError):
        got.build_pairs(df, bad, enc)


def test_bootstrap_and_shared_statistics():
    rng = np.random.default_rng(0)
    v = rng.normal(0.3, 1, size=200)
    lo, hi = got.bootstrap_mean_ci(v)
    assert lo < v.mean() < hi
    controls = rng.normal(size=600)
    assert got.empirical_p_upper(10.0, controls) == pytest.approx(1 / 601)
    assert got.empirical_p_two_sided(-10.0, controls) == pytest.approx(2 / 601)


def test_got_summarise_families():
    rng = np.random.default_rng(1)
    base = dict(layer=6, sae_architecture='', sae_seed='', detail='', accuracy=0.6, baseline_accuracy=0.55, n_pairs=50)
    rows = [dict(base, part='A', intervention_kind='target', intervention='unembedding_contrast',
                 control_group='L6__positive_lam1.0', detail='lambda=1.0', sign=0, effect=0.5),
            dict(base, part='C', intervention_kind='target', intervention='mass_mean_direction', control_group='L6__gaussian',
                 sign=1, effect=0.05),
            dict(base, part='C', intervention_kind='target', intervention='mass_mean_direction', control_group='L6__gaussian',
                 sign=-1, effect=-0.05),
            dict(base, part='C', intervention_kind='target', intervention='sae_top1_direction', control_group='L6__relu_s42__feat1',
                 sae_architecture='relu', sae_seed=42, sign=1, effect=-0.04)]
    for g in ('L6__positive_lam1.0', 'L6__gaussian', 'L6__relu_s42__feat1'):
        for e in rng.normal(0, 0.01, 40):
            rows.append(dict(base, part='C', intervention_kind='control', intervention='x', control_group=g, sign=0, effect=e))
    store = {'L6__positive_lam1.0': rng.normal(0.5, 0.1, 50), 'L6__mass_mean_direction_pos': rng.normal(0.05, 0.1, 50),
             'L6__mass_mean_direction_neg': rng.normal(-0.05, 0.1, 50),
             'L6__relu_s42__sae_top1_direction_pos': rng.normal(-0.04, 0.1, 50)}
    out = got.summarise(pd.DataFrame(rows), store).set_index(['intervention', 'sign'])
    assert out.loc[('unembedding_contrast', 0), 'family'] == 'A_positive_control'
    assert out.loc[('mass_mean_direction', 1), 'family'] == 'C1_raw_directions'
    assert pd.isna(out.loc[('mass_mean_direction', -1), 'family'])        # -s rows are descriptive
    top1 = out.loc[('sae_top1_direction', 1)]
    assert top1['family'] == 'C2_sae_directions' and top1['p_two_sided'] == pytest.approx(2 / 41)


# ---------------------------------------------------------------- script-15 helpers used by C4 (verbatim)
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


def test_minimum_detectable_dose(loop):
    lams = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    assert loop.minimum_detectable_dose(lams, [0.9, 0.4, 0.2, 0.04, 0.01, 0.002, 0.002]) == 0.1
    assert loop.minimum_detectable_dose(lams, [0.01, 0.4, 0.2, 0.04, 0.01, 0.002, 0.002]) == 0.1   # isolated small p ignored
    assert np.isnan(loop.minimum_detectable_dose(lams, [0.9] * 7))
    assert loop.control_seed(6, 'a') == loop.control_seed(6, 'a') != loop.control_seed(6, 'b')



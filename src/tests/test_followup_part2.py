"""Unit tests for the second batch of follow-up scripts:

    scripts/shared/sparsity_matched_truth_preservation.py
    scripts/geometry_of_truth/22_truth_direction_causal_test.py
    scripts/shared/public_sae_spliced_evaluation.py
    scripts/shared/followup_figures.py

Synthetic inputs only; no model, checkpoint or dataset is loaded.
Run from the repository root:  python -m pytest -q src/tests/test_followup_part2.py
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


s3 = load('scripts/shared/sparsity_matched_truth_preservation.py', 's3')
got = load('scripts/geometry_of_truth/22_truth_direction_causal_test.py', 'got22')
figs = load('scripts/shared/followup_figures.py', 'figs')


# ---------------------------------------------------------------- sparsity-matched comparison
def test_truncate_topm_keeps_largest_and_leaves_sparse_rows():
    z = torch.tensor([[0.0, 5.0, 1.0, 3.0, 0.0, 2.0],
                      [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]])
    t = s3.truncate_topm(z, 2)
    assert t[0].tolist() == [0.0, 5.0, 0.0, 3.0, 0.0, 0.0]
    assert torch.equal(t[1], z[1])                                        # one active latent: unchanged
    assert torch.equal(s3.truncate_topm(z, 6), z)
    assert ((s3.truncate_topm(z, 3) > 0).sum(-1) <= 3).all()


def test_decode_matches_the_sae_forward_and_topk_truncation_equals_smaller_k():
    from src.evaluation import faithfulness_metrics as fm
    torch.manual_seed(0)
    x = torch.randn(20, 768) * 3
    for arch in ('relu', 'topk', 'jumprelu'):
        sae = fm.sae_class()(768, 3072, arch, 128).eval()
        with torch.inference_mode():
            rec, z, _ = sae(x)
            assert torch.allclose(s3.decode(sae, z), rec, atol=1e-5)
    big, small = fm.sae_class()(768, 3072, 'topk', 128).eval(), fm.sae_class()(768, 3072, 'topk', 32).eval()
    small.load_state_dict(big.state_dict())
    with torch.inference_mode():
        _, z_big, _ = big(x)
        rec_small, _, _ = small(x)
        assert torch.allclose(s3.decode(big, s3.truncate_topm(z_big, 32)), rec_small, atol=1e-5)


def test_matching_rule_and_paired_comparison():
    assert s3.is_matched([30, 33, 34.9], 32) and not s3.is_matched([30, 36], 32) and not s3.is_matched([], 32)
    df = pd.DataFrame({'architecture': ['topk'] * 3 + ['relu'] * 3, 'seed': [1, 2, 3] * 2,
                       'gap': [0.1, 0.2, 0.15, 0.3, 0.25, 0.4]})
    r = s3.paired_comparison(df, 'gap', 'topk', 'relu')
    assert r['n_seeds'] == 3 and r['all_seeds_agree'] == 'a_lower'
    assert r['mean_diff'] == pytest.approx(np.mean([-0.2, -0.05, -0.25]))
    df.loc[0, 'gap'] = 0.9
    assert s3.paired_comparison(df, 'gap', 'topk', 'relu')['all_seeds_agree'] == 'mixed'
    one = df[df['seed'] == 2]
    assert s3.paired_comparison(one, 'gap', 'topk', 'relu')['all_seeds_agree'] == 'single_seed'


def test_matched_table_only_keeps_matched_cells():
    rows = []
    for arch, nat in (('relu', 60), ('topk', 128), ('jumprelu', 45)):
        for seed in (42, 1042):
            for m in (32, 64):
                l0 = min(m, nat)                                          # JumpReLU cannot reach 64
                rows.append(dict(domain='geo', layer=6, architecture=arch, seed=seed, truncation='top_m', m=m,
                                 l0=l0, informative_layer=True, logistic_auroc_gap=0.1 + seed * 1e-5,
                                 diffmean_auroc_gap=0.2, relative_sq_error=0.3))
    t = s3.matched_table(pd.DataFrame(rows))
    assert set(t['m']) == {32}                                            # m=64 is not matched (45 and 60 < 57.6)
    assert set(t['comparison']) == {'topk - relu', 'topk - jumprelu', 'jumprelu - relu'}


# ---------------------------------------------------------------- Geometry of Truth causal test
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


# ---------------------------------------------------------------- public splicing and figures
def test_loss_recovered_formula():
    pub = load('scripts/shared/public_sae_spliced_evaluation.py', 'pub')
    assert pub.loss_recovered(10.28, 3.589, 3.332) == pytest.approx(96.30, abs=0.01)   # the deck's layer-11 example
    assert pub.loss_recovered(10.0, 3.0, 3.0) == pytest.approx(100.0)
    assert pub.loss_recovered(10.0, 10.0, 3.0) == pytest.approx(0.0)


def test_figures_script_reports_missing_inputs_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    import sys
    out = tmp_path / 'figs'
    monkeypatch.setattr(sys, 'argv', ['followup_figures.py', '--root', str(tmp_path / 'empty'), '--out', str(out)])
    figs.main()
    text = capsys.readouterr().out
    assert 'figures made: none' in text and 'inputs not found' in text
    assert (out / 'tables.md').exists()
    with pytest.raises(SystemExit):
        figs.main()

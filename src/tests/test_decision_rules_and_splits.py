"""Regression tests for the pre-registered decision rules, the splice CE/KL arithmetic, the sparsity band, the GoT split
independence on which every probe result rests, and checkpoint-registry integrity.

CPU only; uses tracked CSVs (data/got/splits, docs/checkpoint_registry.csv) and no checkpoints or caches.
Run from the repository root:  python -m pytest -q src/tests/test_decision_rules_and_splits.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def r2c():
    return _load('scripts/shared/r2_comparisons.py', 'r2c_t')


@pytest.fixture(scope='module')
def m5():
    pytest.importorskip('transformer_lens')
    return _load('scripts/shared/m5_confirmatory_test.py', 'm5_t')


@pytest.fixture(scope='module')
def sp():
    pytest.importorskip('transformer_lens')
    return _load('scripts/shared/spliced_reconstruction_evaluation.py', 'sp_t')


# ------------------------------------------------------------------ R2 rule (all pairs better + Welch CI)
def test_r2_rule_improvement_regression_and_direction(r2c):
    ref, new = [0.30, 0.31, 0.29], [0.10, 0.11, 0.12]
    assert r2c.rule(new, ref, higher=False)['cls'] == 'IMPROVEMENT'
    assert r2c.rule(ref, new, higher=False)['cls'] == 'REGRESSION'
    # for a higher-is-better metric the same numbers mean the opposite
    assert r2c.rule(new, ref, higher=True)['cls'] == 'REGRESSION'
    out = r2c.rule(new, ref, higher=False)
    assert out['improvement_pct'] == pytest.approx(100 * (0.30 - 0.11) / 0.30)   # 100 (baseline - improved)/baseline
    assert out['ci95_high'] < 0 and out['pairs_better'] == 9 == out['n_pairs']


def test_r2_rule_requires_every_pair(r2c):
    """One overlapping pair blocks IMPROVEMENT even if the means differ."""
    assert r2c.rule([0.10, 0.11, 0.295], [0.30, 0.31, 0.29], higher=False)['cls'] == 'NO CLEAR CHANGE'


def test_r2_rule_not_evaluated_on_missing_or_single_seed(r2c):
    assert r2c.rule([0.1], [0.2, 0.3], higher=False)['cls'] == 'NOT EVALUATED'
    assert r2c.rule([0.1, np.nan], [0.2, 0.3], higher=False)['cls'] == 'NOT EVALUATED'


def test_l0_band_is_plus_minus_ten_percent(r2c):
    assert (r2c.T_LOW, r2c.T_HIGH) == pytest.approx((115.2, 140.8))


# ------------------------------------------------------------------ M5 rules
def test_m5_paired_rule(m5):
    assert m5.paired_rule([1.0, 1.1, 0.9], [2.0, 2.1, 1.9])['cls'] == 'IMPROVEMENT'
    assert m5.paired_rule([2.0, 2.1, 1.9], [1.0, 1.1, 0.9])['cls'] == 'REGRESSION'
    # better on every seed but |mean d| <= 2 sd(d): not an improvement
    assert m5.paired_rule([1.0, 1.9, 1.99], [2.0, 2.0, 2.0])['cls'] == 'NO CLEAR CHANGE'
    assert m5.paired_rule([1.0], [2.0])['cls'] == 'NOT EVALUATED'


def test_m5_compare_excludes_unmatched_l0(m5):
    """Rows outside 128 ± 10 % are dropped before any comparison (CLAUDE.md sparsity rule)."""
    rows = []
    for fam, name, l0, v in [('old', 'topk', 128, 0.3), ('r2', 'topk', 128, 0.1), ('r2', 'relu', 160, 0.05),
                             ('old', 'relu', 128, 0.4)]:
        for s, eps in zip((42, 1042, 2042), (0, 0.001, -0.001)):
            rows.append(dict(layer=8, family=fam, name=name, seed=s, owt_l0=l0, owt_nmse=v + eps))
    c = m5.compare(pd.DataFrame(rows))
    assert 'C2' not in set(c.comparison)                    # new ReLU at L0 = 160 is excluded
    assert c.set_index('comparison').loc['C1', 'cls'] == 'IMPROVEMENT'
    assert (c[c.comparison == 'A2'].cls == 'NOT EVALUATED').all()   # reference family absent


# ------------------------------------------------------------------ splice CE / KL arithmetic
class _FakeLM:
    """Logits depend on the (possibly spliced) residual so that a splice changes the predictions."""

    def __init__(self, vocab=50257):                      # owt_original pads with id 50256
        self.W = torch.randn(4, vocab, generator=torch.Generator().manual_seed(0))

    def run_with_hooks(self, tokens, fwd_hooks):
        resid = torch.nn.functional.one_hot(tokens % 4, 4).float()
        for _, fn in fwd_hooks:
            resid = fn(resid, None)
        return resid @ self.W


def _manual(lo, ls, toks, lens):
    ce_o = ce_s = kl = 0.0
    n = 0
    for b in range(toks.shape[0]):
        for t in range(1, int(lens[b]) - 1):          # predictions made at spliced positions 1..len-2
            y = toks[b, t + 1]
            ce_o -= lo[b, t, y].item()
            ce_s -= ls[b, t, y].item()
            kl += (lo[b, t].exp() * (lo[b, t] - ls[b, t])).sum().item()
            n += 1
    return n, (ce_s - ce_o) / n, kl / n


def test_owt_eval_identity_is_zero_and_perturbation_matches_manual(sp):
    s = sp.Splicer(_FakeLM(), 'cpu')
    seqs = [torch.tensor([0, 1, 2, 3, 1]), torch.tensor([0, 2, 3]), torch.tensor([0, 3, 1, 2])]
    old_batch = sp.BATCH
    try:
        sp.BATCH = 2
        orig = sp.owt_original(s, seqs)
        ident = sp.owt_eval(s, seqs, 0, lambda x: x.clone(), orig)
        assert ident['delta_ce'] == 0 and ident['kl_mean'] == 0 and ident['top1_agreement'] == 1
        assert ident['n_predictions'] == sum(len(q) - 2 for q in seqs)
        fn = lambda x: x.flip(-1) * 2.0
        out = sp.owt_eval(s, seqs, 0, fn, orig)
        n = dce = kl = 0.0
        for b in range(0, len(seqs), sp.BATCH):
            toks, lens = sp.pad(seqs[b:b + sp.BATCH], 50256)
            lo = s.logprobs(toks, lens)
            ls = s.logprobs(toks, lens, 0, fn, 'all')
            nb, d, k = _manual(lo, ls, toks, lens)
            n, dce, kl = n + nb, dce + d * nb, kl + k * nb
        assert out['n_predictions'] == n
        assert out['delta_ce'] == pytest.approx(dce / n, rel=1e-5)
        assert out['kl_mean'] == pytest.approx(kl / n, rel=1e-5) and out['kl_mean'] > 0   # KL(p_orig || p_spliced) >= 0
    finally:
        sp.BATCH = old_batch


# ------------------------------------------------------------------ GoT split independence
def _splits(name):
    return pd.read_csv(ROOT / f'data/got/splits/{name}_splits.csv')


@pytest.mark.parametrize('name', ['cities', 'neg_cities', 'larger_than', 'smaller_than'])
def test_got_groups_never_cross_splits(name):
    d = _splits(name)
    assert (d.groupby('group_id').split.nunique() == 1).all()
    assert set(d.split) == {'train', 'val', 'test'}
    assert d.groupby('split').label.mean().between(0.45, 0.55).all()


def test_cities_and_neg_cities_assign_each_city_to_the_same_split():
    """M5's got_cities pools both sets, so a city must never be train in one and test in the other."""
    a = _splits('cities').groupby('group_id').split.first()
    b = _splits('neg_cities').groupby('group_id').split.first()
    common = a.index.intersection(b.index)
    assert len(common) == len(a) == len(b) == 748
    assert (a[common] == b[common]).all()
    n = pd.concat([_splits('cities'), _splits('neg_cities')]).split.value_counts().to_dict()
    assert n == {'train': 2092, 'val': 448, 'test': 452}


def test_truth_aware_ab_halves_are_disjoint_train_groups():
    ab = pd.read_csv(ROOT / 'data/got/splits/train_AB_groups_truth_aware_pilot.csv')
    assert (ab.groupby(['domain', 'group_id']).half.nunique() == 1).all()
    geo = ab[ab.domain == 'geographic']
    train = set(_splits('cities').query("split == 'train'").group_id)
    assert set(geo.group_id) <= train and set(geo.half) == {'A', 'B'}


# ------------------------------------------------------------------ checkpoint registry integrity
def test_checkpoint_registry_paths_unique_and_hashes_wellformed():
    reg = pd.read_csv(ROOT / 'docs/checkpoint_registry.csv')
    assert reg.path.is_unique
    assert reg.sha256.dropna().str.fullmatch(r'[0-9a-f]{64}').all()
    assert not reg.path.str.startswith('/').any()                      # no absolute paths
    ok = reg[(reg.status == 'AUTHORITATIVE') & reg.target_l0.notna()]
    assert len(ok) > 0

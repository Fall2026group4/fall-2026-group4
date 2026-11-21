"""Offline tests for TI-2 (TSP-MF) helpers: synthetic data only (no cache, checkpoint, GPU or TEST row)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def ti2():
    spec = importlib.util.spec_from_file_location('ti2', ROOT / 'scripts/geometry_of_truth/34_tsp_multifamily.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _synthetic(n=240, d=32, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    pol = rng.integers(0, 2, n)
    X = rng.standard_normal((n, d)) + np.outer(y, np.r_[1.0, np.zeros(d - 1)])
    groups = np.array([f'g{i // 4}' for i in range(n)])
    return X, y, pol, groups


def test_family_vectors_reproduce_the_frozen_estimator(ti2):
    from src.sae import truth_subspace as ts
    X, y, pol, g = _synthetic()
    P_ref, _ = ts.estimate_truth_subspace(X, y, pol, g, r=8, seed=7)
    P_new, _ = ts._orthonormal_top(ti2.family_vectors(X, y, pol, g, 7), 8)
    assert np.abs(np.abs(np.sum(P_ref * P_new, 0)) - 1).max() < 1e-8


def test_cap_rows_keeps_whole_groups_and_is_deterministic(ti2):
    g = np.repeat([f'g{i}' for i in range(500)], 3)
    k1, k2 = ti2.cap_rows(g, 1000, 5), ti2.cap_rows(g, 1000, 5)
    assert (k1 == k2).all() and k1.sum() <= 1000 and k1.sum() >= 990
    kept = set(g[k1])
    assert all(k1[g == x].all() for x in kept)          # never a partial group
    assert ti2.cap_rows(g[:30], 1000, 5).all()           # under the cap: everything kept


def test_group_split_is_groupwise(ti2):
    g = np.repeat([f'r{i}' for i in range(100)], 2)
    m = ti2.group_split(g, 0.6, 1)
    assert not (set(g[m]) & set(g[~m])) and len(set(g[m])) == 60


def test_entity_filter_whole_word_and_min_length(ti2):
    s = ['Apple makes phones.', 'The pineapple is sweet.', 'IBM is old.', 'Tesla cars']
    hit = ti2.entity_filter(s, {'Apple', 'IBM', 'Tesla'})
    assert hit.tolist() == [True, False, False, True]     # IBM < 4 chars is not used; 'pineapple' is not 'Apple'


def test_mf_subspace_is_orthonormal_rank8(ti2):
    fams = {k: _synthetic(seed=i) for i, k in enumerate(('a', 'b', 'c'))}
    P, sv = ti2.mf_subspace(fams)
    assert P.shape == (32, 8) and np.allclose(P.T @ P, np.eye(8), atol=1e-10) and len(sv) == 8

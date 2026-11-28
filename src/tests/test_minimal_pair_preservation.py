"""Offline tests for the TI-3 minimal-pair builder (synthetic rows only)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def ti3():
    spec = importlib.util.spec_from_file_location('ti3', ROOT / 'scripts/geometry_of_truth/36_minimal_pair_preservation.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows(n_city=20):
    g, pol, y = [], [], []
    for c in range(n_city):
        for p in (0, 1):
            for lab in (1, 0):
                g.append(f'city{c}'), pol.append(p), y.append(lab)
    return np.array(g), np.array(pol), np.array(y)


def test_true_pairs_share_city_and_polarity(ti3):
    g, pol, y = _rows()
    t, f = ti3.pairs(g, pol, y)
    assert len(t) == 40 and (g[t] == g[f]).all() and (pol[t] == pol[f]).all() and (y[t] == 1).all() and (y[f] == 0).all()


def test_shuffled_pairs_are_a_derangement_within_polarity(ti3):
    g, pol, y = _rows()
    t, f = ti3.pairs(g, pol, y, shuffled=True)
    assert (g[t] != g[f]).all() and (pol[t] == pol[f]).all() and (y[f] == 0).all()
    t2, f2 = ti3.pairs(g, pol, y, shuffled=True)
    assert (f == f2).all()                       # deterministic


def test_malformed_cell_is_rejected(ti3):
    g, pol, y = _rows(3)
    y[0] = 0                                     # a cell with two false rows
    with pytest.raises(AssertionError):
        ti3.pairs(g, pol, y)

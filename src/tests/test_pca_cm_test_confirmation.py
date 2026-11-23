"""Offline tests for the PCA-CM-T decision logic (no model, checkpoint, cache or TEST row is read)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location('pcacmt', ROOT / 'scripts/shared/pca_cm_test_confirmation.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows(cells, diff, half=0.01, seeds=(42, 1042, 2042)):
    out = []
    for (L, fam, name, ref) in cells:
        for s in seeds:
            out.append(dict(layer=L, family=fam, name=name, seed=s, reference=ref, pca_k=128,
                            sae_refit_logistic_gap=0.05 + diff, pca_refit_logistic_gap=0.05,
                            diff_refit_logistic_gap=diff, diff_refit_logistic_gap_ci_low=diff - half,
                            diff_refit_logistic_gap_ci_high=diff + half))
    return out


def _all_cells(m):
    base = []
    for F, members in m.FAMILIES.items():
        for c in members:
            parts = c.split()
            L = int(parts[0][1:])
            fam = parts[1]
            name = 'resjb_relu_24576' if fam == 'resjb' else parts[2]
            base.append((L, fam, name, m.FAMILY_REF[F]))
    return sorted(set(base))


def test_cell_key_and_holm():
    m = _mod()
    assert m.cell_key(8, 'resjb', 'resjb_relu_24576') == 'L8 resjb'
    assert m.cell_key(11, 'r2', 'relu') == 'L11 r2 relu'
    assert m.holm(np.array([0.01, 0.02, 0.04])).tolist() == [True, True, True]    # 0.0167, 0.025, 0.05 thresholds
    assert m.holm(np.array([0.01, 0.03, 0.04])).tolist() == [True, False, False]  # 0.03 > 0.025 stops
    assert m.holm(np.array([0.001, 0.2])).tolist() == [True, False]


def test_families_confirm_and_fail():
    m = _mod()
    cells = _all_cells(m)
    good = pd.DataFrame(_rows(cells, diff=-0.05))          # every seed CI high < 0, tiny p
    _, fam = m.cells_and_families(good)
    assert fam['P2']['n_confirmed'] == 3 and fam['P3']['n_confirmed'] == 8
    bad = pd.DataFrame(_rows(cells, diff=-0.005))          # CI includes 0 -> not confirmed
    _, fam = m.cells_and_families(bad)
    assert fam['P2']['n_confirmed'] == 0 and fam['P3']['n_confirmed'] == 0


def test_p1_counts_sae_worse_and_excludes_random_decoder():
    m = _mod()
    rows = _rows(_all_cells(m), diff=-0.05)
    rows += _rows([(8, 'old', 'topk', 'capacity_k128')], diff=+0.05)                  # SAE WORSE
    rows += _rows([(8, 'r2', 'topk_frozen_decoder', 'capacity_k128')], diff=+0.05)    # control: excluded
    _, fam = m.cells_and_families(pd.DataFrame(rows))
    assert fam['P1']['n_sae_worse'] == 1 and fam['P1']['sae_worse_cells'] == ['L8 old topk']
    assert fam['P1']['confirmed'] is False

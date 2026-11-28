"""Offline tests for the XT-G analysis rules (synthetic rows; no XT-G output is read)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _m():
    spec = importlib.util.spec_from_file_location('x11', ROOT / 'scripts/cross_model/11_xtg_analysis.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows(sae_gap, sae_ci, mlp_ci, pca_gap, pca_ci, raw, aff, enr, enr_ci):
    base = dict(model='gemma', layer=12, family='f', sae_id='s', seed=-1, task='got_cities', owt_val_nmse_sae=0.2,
                owt_val_l0_sae=80, d_sae=16384, refit_mlp_gap=mlp_ci[0] + 0.01, refit_mlp_gap_ci_low=mlp_ci[0],
                refit_mlp_gap_ci_high=mlp_ci[1], dx_raw_frozen_gap=raw, dx_affine_frozen_gap=aff,
                dx_error_truth_enrichment=enr, dx_error_truth_enrichment_ci_low=enr_ci[0],
                dx_error_truth_enrichment_ci_high=enr_ci[1], dx_ceiling_flag=False, dx_orig_auroc=0.8,
                dx_dprime_refit_retention=0.9)
    s = dict(base, source='sae', pca_k=np.nan, refit_logistic_gap=sae_gap, refit_logistic_gap_ci_low=sae_ci[0],
             refit_logistic_gap_ci_high=sae_ci[1])
    p = dict(base, source='pca_err', pca_k=300, refit_logistic_gap=pca_gap, refit_logistic_gap_ci_low=pca_ci[0],
             refit_logistic_gap_ci_high=pca_ci[1])
    return pd.DataFrame([s, p])


def test_rules_positive_case():
    m = _m()
    df = _rows(0.10, (0.06, 0.14), (0.03, 0.08), 0.01, (-0.01, 0.03), raw=0.30, aff=0.10, enr=2.0, enr_ci=(1.6, 2.4))
    S = m.rules(df, {('gemma', 12, 's', -1): 300})
    r = S.iloc[0]
    assert r.READOUT_SHIFT and r.INFORMATION_LOSS and r.LOSS_BEYOND_PCA and r.SUPPRESSION


def test_rules_negative_case():
    m = _m()
    df = _rows(0.01, (-0.02, 0.04), (-0.01, 0.03), 0.01, (-0.01, 0.03), raw=0.30, aff=0.29, enr=1.0, enr_ci=(0.8, 1.2))
    S = m.rules(df, {('gemma', 12, 's', -1): 300})
    r = S.iloc[0]
    assert not (r.READOUT_SHIFT or r.INFORMATION_LOSS or r.LOSS_BEYOND_PCA or r.SUPPRESSION)


def test_missing_pca_reference_is_an_error():
    m = _m()
    df = _rows(0.10, (0.06, 0.14), (0.03, 0.08), 0.01, (-0.01, 0.03), 0.3, 0.1, 2.0, (1.6, 2.4))
    try:
        m.rules(df, {('gemma', 12, 's', -1): 999})
    except AssertionError:
        return
    raise AssertionError('expected a failure when the error-matched PCA row is missing')


def test_bh_and_p_from_ci():
    m = _m()
    assert np.allclose(m.bh([0.01, 0.04, 0.03, 0.20]), [0.04, 0.0533333, 0.0533333, 0.20], atol=1e-6)
    p, se = m.p_from_ci(np.array([0.0392]), np.array([0.0]), np.array([0.0784]))
    assert abs(se[0] - 0.02) < 1e-9 and abs(p[0] - 0.05) < 1e-3

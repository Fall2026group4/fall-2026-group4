"""AUDIT-SENS helpers: S1 variance combination, S3 threshold calibration and truncation wrapper (CPU only)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope='module')
def s1():
    return _load('scripts/shared/audit_s1_statement_uncertainty.py', 'audit_s1')


@pytest.fixture(scope='module')
def s3():
    return _load('scripts/shared/audit_s3_task_l0_matched.py', 'audit_s3')


def test_combine_reduces_to_welch_when_no_cluster_variance(s1):
    r2c = _load('scripts/shared/r2_comparisons.py', 'r2c_t')
    new, ref = np.array([0.74, 0.742, 0.703]), np.array([0.608, 0.604, 0.593])
    w = r2c.rule(new, ref, True)
    se2, df = s1.welch_parts(new, ref)
    c = s1.combine(new.mean() - ref.mean(), se2, df, 0.0, 86)
    assert c['ci_low'] == pytest.approx(w['ci95_low'], rel=1e-9)
    assert c['ci_high'] == pytest.approx(w['ci95_high'], rel=1e-9)


def test_combine_widens_with_cluster_variance(s1):
    se2, df = s1.welch_parts(np.array([1.0, 1.1, 0.9]), np.array([0.5, 0.6, 0.4]))
    a = s1.combine(0.5, se2, df, 0.0, 86)
    b = s1.combine(0.5, se2, df, 0.01, 86)
    assert b['se_total'] > a['se_total'] and b['p'] > a['p']
    assert b['ci_high'] - b['ci_low'] > a['ci_high'] - a['ci_low']          # never narrower than seed-only


def test_holm_matches_frozen_h1_implementation(s1):
    s29 = _load('scripts/repe/29_repe_lm_test_evaluation.py', 's29_t')
    p = np.array([0.005, 0.03, 1e-6, 0.2, 0.01])
    assert np.allclose(s1.holm(p), s29.holm(p))


def test_calibrate_tau_hits_target_mean_l0(s3):
    rng = np.random.default_rng(0)
    Z = np.maximum(rng.standard_normal((500, 300)), 0)
    for k in (5, 40, 120):
        tau = s3.calibrate_tau(Z, k)
        assert (Z > tau).sum(1).mean() == pytest.approx(k, abs=0.01)
    assert s3.calibrate_tau(Z, 1000) == 0.0          # never raises L0 above the natural value


def test_truncated_wrapper_only_removes_latents(s3):
    torch.manual_seed(0)
    sae = s3.LAD.O3.SAE(16, 64, 'relu', 8)
    x = torch.randn(32, 16)
    rec, z, pre = sae(x)
    t = s3.Truncated(sae, 0.1)
    rec2, z2, pre2 = t(x)
    assert torch.equal(pre, pre2)
    assert ((z2 > 0) <= (z > 0)).all() and (z2[z2 > 0] > 0.1).all()
    assert torch.allclose(rec2, z2 @ sae.decoder.weight.T + sae.output_bias, atol=1e-6)
    assert torch.allclose(s3.Truncated(sae, 0.0)(x)[0], rec, atol=1e-6)

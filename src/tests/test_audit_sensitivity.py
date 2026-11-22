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


# ------------------------------------------------------------------ paper claims vs committed AUDIT-SENS artifacts
import json  # noqa: E402

import pandas as pd  # noqa: E402

AS = ROOT / 'results/shared/audit_sensitivity'
SEC = ROOT / 'research_paper/Latex/sections'
needs_artifacts = pytest.mark.skipif(not (AS / 's6_probe_grid/checks.json').exists(), reason='AUDIT-SENS outputs absent')


@needs_artifacts
def test_s1_claims():
    h = pd.read_csv(AS / 's1_statement_uncertainty/h1_test_cells_adjusted.csv')
    c1 = h[(h.hypothesis == 'HT1') & (h.comparison == 'C1')].iloc[0]
    h3 = h[(h.hypothesis == 'HT3') & (h.comparison == 'C1')].iloc[0]
    assert f'{c1.ci_low:+.3f}' == '+0.014' and f'{c1.ci_high:+.3f}' == '+0.240' and f'{c1.p_holm:.3f}' == '0.156'
    assert f'{h3.ci_low:+.3f}' == '-0.093' and f'{h3.ci_high:+.3f}' == '-0.002' and f'{h3.p_holm:.3f}' == '0.156'
    assert not c1.supported_under_seed_plus_statement and not h3.supported_under_seed_plus_statement
    assert h[h.hypothesis == 'HT2'].supported_under_seed_plus_statement.all()
    m = pd.read_csv(AS / 's1_statement_uncertainty/m5_test_cells_adjusted.csv')
    assert m.supported_under_seed_plus_document.sum() == 16
    bad = m[~m.supported_under_seed_plus_document]
    assert set(zip(bad.layer, bad.comparison, bad.metric)) == {(1, 'C2', 'splice_delta_ce'), (1, 'C3', 'splice_delta_ce')}
    v = pd.read_csv(AS / 's1_statement_uncertainty/validation_combined.csv')
    assert round(v.stmt_share_of_variance.min(), 2) == 0.33 and round(v.stmt_share_of_variance.max(), 2) == 0.99


@needs_artifacts
def test_s2_s3_s4_s5_s6_claims():
    s2 = json.loads((AS / 's2_persona_headroom/checks.json').read_text())
    assert s2['layers_persona_specific_old'] == [] and s2['layers_neutral_larger_old'] == [5, 7, 8, 9, 10, 11]
    s3 = pd.read_csv(AS / 's3_task_l0_matched/comparisons_natural_vs_matched.csv')
    assert len(s3) == 57 and (~s3.l0_sensitive).sum() == 54
    r = pd.read_csv(AS / 's3_task_l0_matched/old_architecture_ordering.csv').pivot_table(
        index=['layer', 'metric'], columns='l0', values='order', aggfunc='first').dropna()
    assert len(r) == 19 and (r.natural != r.matched).sum() == 13
    s4 = pd.read_csv(AS / 's4_got_probe_anomaly/diagnostics.csv')
    assert s4[s4.diagnostic == 'D5_bag_of_words'].auroc.max() <= 0.55
    per = s4[(s4.diagnostic == 'D2_per_polarity') & (s4.family == 'logistic') & s4.layer.isin([1, 2])].auroc
    pooled = s4[(s4.diagnostic == 'D1_pooled') & (s4.family == 'logistic') & s4.layer.isin([1, 2])].auroc
    assert f'{per.min():.2f}' == '0.75' and f'{per.max():.2f}' == '0.77'
    assert f'{pooled.min():.2f}' == '0.52' and f'{pooled.max():.2f}' == '0.59'
    b = pd.read_csv(AS / 's5_noise_references/by_layer.csv')
    g = b[(b.task == 'got_cities') & (b.family == 'old') & (b.layer >= 6)]
    share = g.logistic_gap_shuffled / g.logistic_gap_sae
    assert f'{share.min():.2f}' == '0.50' and f'{share.max():.2f}' == '0.67'
    ratio = g.massmean_gap_sae / g[['massmean_gap_isotropic', 'massmean_gap_shuffled', 'massmean_gap_errcov']].max(1)
    assert f'{ratio.min():.1f}' == '2.0' and f'{ratio.max():.1f}' == '5.0'
    p = pd.read_csv(AS / 's5_noise_references/per_cell.csv')
    assert int(p[p.task == 'got_cities'].logistic_label_aligned.sum()) == 92 and (p.task == 'got_cities').sum() == 99
    s6 = json.loads((AS / 's6_probe_grid/checks.json').read_text())
    assert s6['n_grid_sensitive'] == 0 and s6['original_at_edge'] == 0


@needs_artifacts
def test_paper_withdrawn_claims_absent():
    text = ' '.join(f.read_text() for f in SEC.glob('*.tex'))
    for stale in ('so part of the mid-layer loss is\npersona-specific', 'is largely perturbation sensitivity',
                  'is largely reproduced by random noise of the same norm', 'Most of the\nfrozen-probe collapse is therefore'):
        assert stale not in text, stale
    for needed in ('[+0.014, +0.240]', '[-0.093, -0.002]', '50--67', '13 of 19', '54 of 57', 'sec:audit'):
        assert needed in text, needed

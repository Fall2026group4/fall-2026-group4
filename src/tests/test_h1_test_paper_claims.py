"""H1-TEST: the paper's numerical claims must match the committed artifacts (D36).

Checks the generated table against a fresh render, and every H1-TEST number quoted in the paper text against
results/repe/repe_lm_test_evaluation/.
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results/repe/repe_lm_test_evaluation'
SEC = ROOT / 'research_paper/Latex/sections'

pytestmark = pytest.mark.skipif(not (RES / 'confirmatory_cells.csv').exists(), reason='H1-TEST artifacts absent')


def _gen():
    spec = importlib.util.spec_from_file_location('h1tab', ROOT / 'scripts/shared/h1_test_paper_table.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope='module')
def art():
    cells = pd.read_csv(RES / 'confirmatory_cells.csv')
    ht4 = json.loads((RES / 'ht4_probe_transfer.json').read_text())
    comp = pd.read_csv(RES / 'comparisons.csv')
    per = pd.read_csv(RES / 'per_sae.csv')
    return cells, ht4, comp, per


def cell(cells, h, c, layer=8):
    return cells[(cells.hypothesis == h) & (cells.comparison == c) & (cells.layer == layer)].iloc[0]


def test_table_matches_artifacts(art):
    m = _gen()
    committed = (ROOT / 'research_paper/Latex/tables/t_h1test.tex').read_text()
    assert committed == m.build(*art[:3])


def test_verdicts(art):
    cells = art[0]
    v = {h: bool(g.confirmed_cell.all()) for h, g in cells.groupby('hypothesis')}
    assert v == {'HT1': False, 'HT2': True, 'HT3': False}
    assert len(cells) == 13
    assert bool(cell(cells, 'HT1', 'C1').confirmed_cell) and not bool(cell(cells, 'HT1', 'C3').confirmed_cell)
    assert bool(cell(cells, 'HT3', 'C1').confirmed_cell) and not bool(cell(cells, 'HT3', 'C3').confirmed_cell)


def test_quoted_numbers(art):
    cells, ht4, comp, per = art
    c1, c3 = cell(cells, 'HT1', 'C1'), cell(cells, 'HT1', 'C3')
    assert f'{c1.test_ref:.3f}' == '0.602' and f'{c1.test_new:.3f}' == '0.729'
    assert f'{c1.test_diff:+.3f}' == '+0.127' and f'{c1.test_welch_p_holm:.3f}' == '0.021'
    assert f'{c3.test_diff:+.3f}' == '+0.072' and f'{c3.test_welch_p_holm:.3f}' == '0.058'
    assert int(c3.test_pairs_better) == 9
    h3 = cell(cells, 'HT3', 'C1')
    assert f'{h3.test_diff:+.3f}' == '-0.048' and f'{h3.test_welch_p_holm:.3f}' == '0.046'
    assert int(cell(cells, 'HT3', 'C3').test_pairs_better) == 8
    ht2 = cells[cells.hypothesis == 'HT2']
    assert f'{ht2.test_diff.min():.3f}' == '0.035' and f'{ht2.test_diff.max():.3f}' == '0.141'
    assert ht2.test_welch_p_holm.max() < 0.02
    assert (ht2.test_diff - ht2.val_diff).abs().max() <= 0.011
    n34 = {L: ht4[L]['neutral_N3_N4']['auroc'] for L in ('L1', 'L8', 'L11')}
    assert [f'{n34[L]:.3f}' for L in ('L1', 'L8', 'L11')] == ['0.501', '0.561', '0.551']
    assert n34['L8'] > 0.55 and n34['L11'] > 0.55
    n12 = [ht4[L]['neutral_N1_N2']['auroc'] for L in ('L1', 'L8', 'L11')]
    assert max(n12) <= 0.55 and f'{min(n12):.2f}' == '0.47' and f'{max(n12):.2f}' == '0.50'
    t78 = {L: ht4[L]['test_only_T7_T8']['auroc'] for L in ('L1', 'L8', 'L11')}
    assert [f'{t78[L]:.3f}' for L in ('L1', 'L8', 'L11')] == ['0.564', '0.833', '0.825']
    assert min(t78[L] - n34[L] for L in ('L8', 'L11')) >= 0.26
    vc = comp.cls.value_counts()
    assert len(comp) == 90 and vc.get('IMPROVEMENT') == 55 and vc.get('NO CLEAR CHANGE') == 35
    assert vc.get('REGRESSION', 0) == 0
    m4 = per.dropna(subset=['m4__T_full_mean']).groupby('layer').first()
    assert [f'{m4.loc[L, "m4__T_full_mean"]:.3f}' for L in (1, 8, 11)] == ['0.038', '0.020', '0.038']
    assert (m4.m4__T_full_ci_low > 0).all()
    assert f'{m4.m4__kl_base_mean.iloc[0]:.3f}' == '0.039' and f'{m4.m4__kl_syn_mean.iloc[0]:.3f}' == '0.017'
    assert per.dropna(subset=['m4__T_full_mean']).m4__T_rec_above_controls_q95.astype(bool).all()


def test_paper_has_no_stale_h1_claims():
    text = ' '.join(p.read_text() for p in SEC.glob('*.tex'))
    for stale in ('borderline at the statement level', 'awaits the RepE-LM test split', 'never built or accessed',
                  'it has not been run', 'has not been run, so no result'):
        assert stale not in text, stale
    for needed in ('0.602', '0.729', '0.021', '0.058', '0.046', '+0.035', '+0.141', '0.561', '0.551'):
        assert needed in text, needed

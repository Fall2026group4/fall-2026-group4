"""Tests for R3 (Gemma Scope width/L0 ladder), written before the code they test.

Pre-registration: results/gemma2_pile/r3_gemma_scope_ladder/README.md, Section 1 and Amendment 1.
Code under test: scripts/gemma/05_pile_gemma_scope_ladder.py (s05) and scripts/gemma/06_r3_precision_comparison.py
(s06). Both are loaded from their files, as test_gemma_pile_splicing.py does for script 03; nothing runs main().

CPU only, no downloads. The ladder engine is exercised on a tiny randomly initialised Gemma-2 built from a config
(same construction as test_gemma_pile_splicing.py) with tiny random JumpReLU SAEs.

What is checked, by README item:
* chunked JumpReLU encode/decode equals the unchunked SAE                      (Design: SAE forward; gate V6)
* every SAE of a layer gets the same random directions u                       (Seeds; Amendment 1 (5))
* L0-band membership reproduces the README's expected bands                    (H2)
* intersection-union verdicts for H1/H2, Holm and the H3 verdict               (H1, H2, H3)
* SAE placement on the two 16 GB T4s and the K0-K6 run plan                    (Amendment 1 (3), (4))
* per-SAE folders with DONE.json; resume skips finished SAEs; an interrupted
  then resumed run gives byte-identical result files to an uninterrupted run   (Amendment 1 (5))
* gate V9: pass within relative difference 1e-6, bit-identity recorded apart   (Amendment 1 (5), as changed)
* Q0b, V6 and equivalence verdicts                                              (Amendment 1 (1), (2))
* V0 hardware check, fp32 device map, V10 secret scan                          (Amendment 1 (2), (6))

Run from repository root:  python -m pytest -q src/tests/test_gemma_r3_ladder.py
"""
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import transformers

from src.evaluation import pile_splicing as ps

ROOT = Path(__file__).resolve().parents[2]
S05_PATH = ROOT / 'scripts/gemma/05_pile_gemma_scope_ladder.py'
S06_PATH = ROOT / 'scripts/gemma/06_r3_precision_comparison.py'

# The ladder exactly as tabulated in the README ("SAE ladder"); written out here independently of the code.
README_LADDER = {
    (5, '16k'): ([18, 34, 68, 143, 309], 'A'),
    (5, '65k'): ([17, 29, 53, 105, 211], 'A'),
    (12, '16k'): ([22, 41, 82, 176, 445], 'A'),
    (12, '32k'): ([12, 22, 40, 76, 155, 360], 'A'),
    (12, '65k'): ([21, 38, 72, 141, 297], 'A'),
    (12, '131k'): ([12, 20, 36, 67, 129, 264], 'B'),
    (12, '262k'): ([11, 21, 36, 67, 121, 243], 'B'),
    (12, '524k'): ([22, 29, 46, 65, 115, 227], 'B'),
    (19, '16k'): ([23, 40, 73, 137, 279], 'A'),
    (19, '65k'): ([21, 35, 63, 115, 216], 'A'),
}
D_SAE = {'16k': 16384, '32k': 32768, '65k': 65536, '131k': 131072, '262k': 262144, '524k': 524288, '1m': 1048576}


def readme_ladder():
    return [{'sae_id': f'layer_{L}/width_{w}/average_l0_{n}', 'layer': L, 'width': w, 'tier': tier}
            for (L, w), (l0s, tier) in README_LADDER.items() for n in l0s]


# ------------------------------------------------------------------------------------------------ module loading
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # main() is not run on import
    return mod


@pytest.fixture(scope='module')
def s05():
    return _load(S05_PATH, 's05')


@pytest.fixture(scope='module')
def s06():
    return _load(S06_PATH, 's06')


# ------------------------------------------------------------------------------------------------ toy model and SAEs
@pytest.fixture(scope='module')
def tiny_gemma():
    cfg = transformers.Gemma2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                    max_position_embeddings=64, sliding_window=32, pad_token_id=0, bos_token_id=2)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation='eager')
    return model.float().eval()


def tiny_params(d_model=32, d_sae=48, seed=0):
    rng = np.random.default_rng(seed)
    return {'W_enc': rng.normal(size=(d_model, d_sae)).astype(np.float32) / np.sqrt(d_model),
            'W_dec': rng.normal(size=(d_sae, d_model)).astype(np.float32) / np.sqrt(d_sae),
            'b_enc': rng.normal(size=d_sae).astype(np.float32) * 0.1,
            'b_dec': rng.normal(size=d_model).astype(np.float32) * 0.1,
            'threshold': np.abs(rng.normal(size=d_sae)).astype(np.float32) * 0.2}


TOY_IDS = ['layer_1/width_16k/average_l0_20', 'layer_1/width_65k/average_l0_20', 'layer_1/width_16k/average_l0_40']
TOY_PARAMS = {sid: tiny_params(seed=i) for i, sid in enumerate(TOY_IDS)}


def toy_docs():
    rng = np.random.default_rng(7)
    ev = [{'rank': r, 'line_index': 100 + r, 'pile_set_name': 'Src-A' if r % 2 else 'Src-B',
           'ids': [2] + [int(t) for t in rng.integers(3, 97, size=n)]} for r, n in enumerate([7, 9, 12, 8, 10, 11])]
    ref = [{'rank': r, 'line_index': 200 + r, 'pile_set_name': 'Src-A',
            'ids': [2] + [int(t) for t in rng.integers(3, 97, size=n)]} for r, n in enumerate([9, 6, 10])]
    return ev, ref


def toy_load_sae(sae_id):
    params = TOY_PARAMS[sae_id]
    h = hashlib.sha256(b''.join(np.ascontiguousarray(params[k]).tobytes() for k in sorted(params))).hexdigest()
    return params, h


def make_ctx(s05, model, **overrides):
    ev, ref = toy_docs()
    kwargs = dict(model=model, eval_docs=ev, ref_docs=ref, load_sae=toy_load_sae, device='cpu', sae_device='cpu',
                  chunk=4, v9_docs=3, order=list(TOY_IDS))
    kwargs.update(overrides)
    return s05.LadderContext(**kwargs)


RESULT_FILES = ['per_doc_metrics.csv', 'per_doc_sum_x.npz', 'sae_info.json']  # run_record.json is volatile
LAYER_FILES = ['controls_per_doc.csv', 'mean_ablation.npz', 'layer_info.json']


def sae_dir(s05, ladder_dir: Path, sae_id: str) -> Path:
    return ladder_dir / 'per_sae' / s05.sae_folder_name(sae_id)


# ==================================================================================================== SAE arithmetic
def test_chunked_sae_equals_unchunked(s05):
    params = tiny_params(seed=3)
    sae = ps.JumpReLUSAE(params, d_model=32, d_sae=48)
    x = torch.randn(301, 32, generator=torch.Generator().manual_seed(1)) * 2
    x_ref, l0_ref = sae(x)
    for chunk in (1, 7, 128, 301, 1000):
        x_hat, l0 = s05.chunked_sae(sae, x, chunk)
        assert torch.equal(l0, l0_ref), chunk
        assert torch.allclose(x_hat, x_ref, rtol=1e-6, atol=1e-6), chunk


def test_active_latent_agreement(s05):
    a = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]])
    assert s05.active_latent_agreement(a, a.clone()) == 1.0
    b = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])  # one of six (token, latent) flags differs
    assert s05.active_latent_agreement(a, b) == pytest.approx(5 / 6)


def test_random_directions_are_the_same_for_every_sae_of_a_layer(s05):
    """u depends only on (layer, document order), never on the SAE: only the norm ||x - x_hat|| differs."""
    xs = [torch.randn(n, 32, generator=torch.Generator().manual_seed(n)) for n in (5, 8, 3)]
    units = []
    for scale in (0.1, 0.7):  # two different 'SAEs'
        gen = s05.random_generator(12)
        u_sae = []
        for x in xs:
            x_hat = x * (1 - scale)
            pert = ps.norm_matched_random_error(x, x_hat, gen)
            u_sae.append((pert - x) / (x - x_hat).norm(dim=-1, keepdim=True))
        units.append(u_sae)
    for a, b in zip(*units):
        assert torch.allclose(a, b, atol=1e-6)
    # the same stream as M6's per-layer generator (seed 1000 + L, fresh at the start)
    m6 = torch.Generator().manual_seed(1012)
    assert torch.equal(torch.randn(4, 3, generator=s05.random_generator(12)), torch.randn(4, 3, generator=m6))


# ======================================================================================== ladder, placement, run plan
def test_parse_sae_id_and_folder_name(s05):
    info = s05.parse_sae_id('layer_12/width_524k/average_l0_227')
    assert info == {'layer': 12, 'width': '524k', 'd_sae': 524288, 'release_l0': 227.0}
    assert s05.sae_folder_name('layer_12/width_524k/average_l0_227') == 'layer_12_width_524k_l0_227'
    with pytest.raises(ValueError):
        s05.parse_sae_id('layer_12/width_3k/average_l0_5')


@pytest.mark.parametrize('width, placement', [('16k', 'gpu0'), ('32k', 'gpu0'), ('65k', 'gpu0'), ('131k', 'gpu0'),
                                              ('262k', 'gpu0'), ('524k', 'gpu1'), ('1m', 'excluded')])
def test_sae_placement_on_two_16gb_t4s(s05, width, placement):
    assert s05.sae_placement(D_SAE[width]) == placement


def test_peak_memory_estimates_match_amendment_table(s05):
    for width, gb in (('16k', 8.0), ('65k', 9.0), ('262k', 12.9), ('524k', 18.2)):
        assert s05.estimate_peak_gb(D_SAE[width]) == pytest.approx(gb, abs=0.06), width


def test_run_plan_covers_every_sae_exactly_once(s05):
    ladder = readme_ladder()
    assert len(ladder) == 54 and sum(r['tier'] == 'A' for r in ladder) == 36
    plan = s05.plan_runs(ladder)
    assert list(plan) == ['K0', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6']
    assert plan['K0'] == []
    assert [len(plan[k]) for k in ('K1', 'K2', 'K3', 'K4', 'K5', 'K6')] == [10, 10, 16, 6, 6, 6]
    assert all(s.startswith('layer_5/') for s in plan['K1']) and all(s.startswith('layer_19/') for s in plan['K2'])
    assert all('/width_524k/' in s for s in plan['K6'])
    flat = [s for k in plan for s in plan[k]]
    assert sorted(flat) == sorted(r['sae_id'] for r in ladder) and len(set(flat)) == 54


def test_run_budget_rule(s05):
    assert s05.may_start(projected_h=9.9, week_used_h=20.0)['ok']
    assert not s05.may_start(projected_h=10.1, week_used_h=0.0)['ok']      # over the 10 h per-run cap
    assert not s05.may_start(projected_h=8.0, week_used_h=22.5)['ok']      # would pass 30 h in the week


# ======================================================================================== H1-H3 statistics helpers
def test_l0_bands_reproduce_readme_expected_bands(s05):
    table = pd.DataFrame([{**s05.parse_sae_id(r['sae_id']), 'sae_id': r['sae_id'],
                           'measured_l0': s05.parse_sae_id(r['sae_id'])['release_l0']} for r in readme_ladder()])
    bands = s05.l0_bands(table, tol=0.10)
    got = {(b['layer'], b['anchor_sae_id']): [s.split('/')[1] + '/' + s.split('_')[-1] for s in b['members']]
           for b in bands}
    assert got == {
        (12, 'layer_12/width_16k/average_l0_22'): ['width_16k/22', 'width_32k/22', 'width_65k/21', 'width_131k/20',
                                                   'width_262k/21', 'width_524k/22'],
        (12, 'layer_12/width_16k/average_l0_41'): ['width_16k/41', 'width_32k/40', 'width_65k/38'],
        (12, 'layer_12/width_16k/average_l0_82'): ['width_16k/82', 'width_32k/76'],
        (5, 'layer_5/width_16k/average_l0_18'): ['width_16k/18', 'width_65k/17'],
        (19, 'layer_19/width_16k/average_l0_23'): ['width_16k/23', 'width_65k/21'],
    }


def test_l0_bands_use_measured_not_release_l0(s05):
    rows = [{'sae_id': 'layer_5/width_16k/average_l0_18', 'measured_l0': 30.0},
            {'sae_id': 'layer_5/width_65k/average_l0_17', 'measured_l0': 17.0}]  # 17 is not within 10 % of 30
    table = pd.DataFrame([{**s05.parse_sae_id(r['sae_id']), **r} for r in rows])
    assert s05.l0_bands(table, tol=0.10) == []


@pytest.mark.parametrize('cis, verdict', [
    ([(0.01, 0.05), (0.002, 0.03)], 'SUPPORTED'),
    ([(0.01, 0.05), (-0.01, 0.03)], 'INCONCLUSIVE'),
    ([(0.01, 0.05), (-0.04, -0.001)], 'CONTRADICTED'),
    ([(-0.01, 0.01)], 'INCONCLUSIVE'),
])
def test_intersection_union_verdict(s05, cis, verdict):
    assert s05.iut_verdict(cis) == verdict


def test_pairwise_iut_orders_and_signs_differences(s05):
    """H1: order by measured L0 (low to high); d = dCE(lower L0) - dCE(higher L0) for each adjacent pair."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, size=200)
    vals = {'a': 0.30 + base, 'b': 0.20 + base, 'c': 0.10 + base}   # dCE falls as L0 rises
    idx = ps.bootstrap_indices(200, 500, seed=0)
    out = s05.pairwise_iut(['a', 'b', 'c'], vals, idx)
    assert [(p['lower'], p['higher']) for p in out['pairs']] == [('a', 'b'), ('b', 'c')]
    assert [p['mean_d'] for p in out['pairs']] == pytest.approx([0.1, 0.1])
    assert out['verdict'] == 'SUPPORTED'
    vals['c'] = 0.25 + base                                         # reversal in the second pair
    assert s05.pairwise_iut(['a', 'b', 'c'], vals, idx)['verdict'] == 'CONTRADICTED'


def test_holm_step_down(s05):
    # sorted p: 0.005 <= 0.05/4, 0.01 <= 0.05/3, 0.03 > 0.05/2 -> stop; 0.04 is not rejected either
    assert s05.holm([0.01, 0.04, 0.03, 0.005], alpha=0.05) == [True, False, False, True]
    assert s05.holm([0.001, 0.002], alpha=0.05) == [True, True]


def test_one_sided_p_from_bootstrap_se(s05):
    assert s05.one_sided_p(0.0, 1.0) == pytest.approx(0.5)
    assert s05.one_sided_p(1.6448536, 1.0) == pytest.approx(0.05, abs=1e-6)
    assert s05.one_sided_p(-3.0, 1.0) > 0.99


@pytest.mark.parametrize('significant, verdict', [
    ([True] * 9, 'HOLDS ACROSS THE LADDER'),
    ([True] * 6 + [False] * 3, 'MOSTLY'),                 # exactly two thirds
    ([True] * 5 + [False] * 4, 'NOT GENERAL'),
])
def test_h3_verdict(s05, significant, verdict):
    assert s05.h3_verdict(significant) == verdict


# ================================================================================== V6, Q0b and equivalence verdicts
@pytest.mark.parametrize('lo, hi, verdict', [(-0.004, 0.004, 'EQUIVALENT'), (0.006, 0.01, 'DIFFERENT'),
                                             (-0.02, -0.006, 'DIFFERENT'), (-0.001, 0.007, 'INCONCLUSIVE')])
def test_equivalence_verdict(s05, lo, hi, verdict):
    assert s05.equivalence_verdict(lo, hi, margin=0.005) == verdict


def test_v6_verdict(s05):
    assert s05.v6_verdict(exact=True, within_margin=True) == 'M6 REPRODUCED EXACTLY'
    assert s05.v6_verdict(exact=False, within_margin=True) == 'M6 REPRODUCED WITHIN MARGIN'
    assert s05.v6_verdict(exact=False, within_margin=False) == 'FAILED'


@pytest.mark.parametrize('lo, hi, verdict', [(0.001, 0.004, 'WITHIN MARGIN'), (0.006, 0.009, 'BEYOND MARGIN'),
                                             (0.003, 0.007, 'INCONCLUSIVE')])
def test_q0b_verdict(s06, lo, hi, verdict):
    assert s06.q0b_verdict(lo, hi, margin=0.005) == verdict


def test_q0b_error_and_signed_difference(s06):
    bf16 = np.array([0.10, 0.20, 0.30, 0.40])
    fp32 = np.array([0.11, 0.18, 0.30, 0.43])
    out = s06.q0b_metric(bf16, fp32, ps.bootstrap_indices(4, 200, seed=0))
    assert out['mean_abs_error'] == pytest.approx(np.mean([0.01, 0.02, 0.0, 0.03]))
    assert out['mean_signed_diff'] == pytest.approx(np.mean([-0.01, 0.02, 0.0, -0.03]))
    assert out['abs_error_ci95_lo'] <= out['mean_abs_error'] <= out['abs_error_ci95_hi']


# ============================================================================================ V9 comparison rules
def test_relative_match_rule(s05):
    a = np.array([1.0, -2.0, 0.0, 1e8])
    same = s05.relative_match(a, a.copy(), rtol=1e-6)
    assert same['pass'] and same['bit_identical'] and same['max_rel_diff'] == 0
    close = s05.relative_match(a * (1 + 1e-9), a, rtol=1e-6)
    assert close['pass'] and not close['bit_identical'] and 0 < close['max_rel_diff'] < 1e-6
    far = s05.relative_match(a * (1 + 1e-5), a, rtol=1e-6)
    assert not far['pass']
    assert not s05.relative_match(np.array([1e-30]), np.array([0.0]), rtol=1e-6)['pass']  # zero needs exact zero


def test_vector_match_uses_relative_norm(s05):
    v = np.linspace(-1, 1, 50)
    assert s05.compare_vectors(v, v.copy(), rtol=1e-6) == {'pass': True, 'bit_identical': True, 'rel_diff': 0.0}
    w = v.copy()
    w[25] += 1e-9  # w[25] is ~0, so an element-wise rule would fail; the norm rule passes
    res = s05.compare_vectors(w, v, rtol=1e-6)
    assert res['pass'] and not res['bit_identical']
    assert not s05.compare_vectors(v * 1.001, v, rtol=1e-6)['pass']


def test_compare_tables_requires_same_rows_and_exact_text(s05):
    df = pd.DataFrame({'rank': [0, 1], 'condition': ['sae_splice', 'sae_splice'], 'pile_set_name': ['A', 'B'],
                       'ce_new_sum': [10.0, 20.0]})
    keys = ['rank', 'condition']
    assert s05.compare_tables(df.copy(), df, keys, rtol=1e-6)['pass']
    assert not s05.compare_tables(df.iloc[:1], df, keys, rtol=1e-6)['pass']            # a row is missing
    renamed = df.assign(pile_set_name=['A', 'C'])
    assert not s05.compare_tables(renamed, df, keys, rtol=1e-6)['pass']                 # text must match exactly
    nudged = df.assign(ce_new_sum=[10.0, 20.0 * (1 + 1e-9)])
    res = s05.compare_tables(nudged, df, keys, rtol=1e-6)
    assert res['pass'] and not res['bit_identical']


# ============================================================================================ files, DONE.json
def test_deterministic_npz_is_byte_identical(s05, tmp_path):
    arrays = {'sum_x': np.arange(12, dtype=np.float64).reshape(3, 4), 'ranks': np.array([0, 1, 2])}
    s05.save_npz_deterministic(tmp_path / 'a.npz', **arrays)
    s05.save_npz_deterministic(tmp_path / 'b.npz', **arrays)
    assert (tmp_path / 'a.npz').read_bytes() == (tmp_path / 'b.npz').read_bytes()
    with np.load(tmp_path / 'a.npz') as z:
        assert np.array_equal(z['sum_x'], arrays['sum_x']) and np.array_equal(z['ranks'], arrays['ranks'])


def test_done_marker_detects_missing_and_changed_files(s05, tmp_path):
    folder = tmp_path / 'layer_1_width_16k_l0_20'
    folder.mkdir()
    (folder / 'per_doc_metrics.csv').write_text('a,b\n1,2\n')
    assert not s05.is_done(folder)                       # no DONE.json yet
    s05.write_done(folder)
    assert s05.is_done(folder)
    (folder / 'per_doc_metrics.csv').write_text('a,b\n1,3\n')
    assert not s05.is_done(folder)                       # a hashed file changed


# ============================================================================== ladder engine on the toy Gemma-2
def test_engine_writes_per_sae_folders_and_passes_gates(s05, tiny_gemma, tmp_path):
    rec = s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS, tmp_path, run_id='fresh')
    assert rec['ran'] == TOY_IDS and rec['skipped'] == [] and rec['v9'] is None
    layer_dir = tmp_path / 'per_layer' / 'layer_1'
    assert s05.is_done(layer_dir) and all((layer_dir / f).exists() for f in LAYER_FILES)
    info = json.loads((layer_dir / 'layer_info.json').read_text())
    assert info['V2']['pass'] and info['V3']['pass']
    for sid in TOY_IDS:
        d = sae_dir(s05, tmp_path, sid)
        assert s05.is_done(d) and all((d / f).exists() for f in RESULT_FILES + ['run_record.json'])
        sinfo = json.loads((d / 'sae_info.json').read_text())
        assert sinfo['sae_id'] == sid and sinfo['V3_original_logits_match'] is True
        assert sinfo['params_sha256'] == toy_load_sae(sid)[1]
        df = pd.read_csv(d / 'per_doc_metrics.csv')
        assert sorted(df.condition.unique()) == ['norm_matched_random_error', 'sae_splice']
        assert len(df) == 2 * 6 and (df.sae_id == sid).all()


def test_random_error_rows_do_not_depend_on_which_saes_ran_before(s05, tiny_gemma, tmp_path):
    """Re-seeding per SAE: SAE 3 run alone gives byte-identical files to SAE 3 run after SAEs 1 and 2."""
    s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS, tmp_path / 'all', run_id='a')
    s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS[2:], tmp_path / 'alone', run_id='b')
    for f in RESULT_FILES:
        assert (sae_dir(s05, tmp_path / 'all', TOY_IDS[2]) / f).read_bytes() == \
               (sae_dir(s05, tmp_path / 'alone', TOY_IDS[2]) / f).read_bytes(), f


def test_interrupted_then_resumed_run_is_byte_identical_to_uninterrupted(s05, tiny_gemma, tmp_path):
    fresh, resumed = tmp_path / 'fresh', tmp_path / 'resumed'
    s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS, fresh, run_id='fresh')

    first = s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS, resumed, run_id='k1', max_saes=1)
    assert first['ran'] == TOY_IDS[:1]
    # the session 'ends' while SAE 2 is being written: its folder exists but has no DONE.json
    half = sae_dir(s05, resumed, TOY_IDS[1])
    half.mkdir(parents=True)
    shutil.copy(sae_dir(s05, fresh, TOY_IDS[1]) / 'per_doc_metrics.csv', half / 'per_doc_metrics.csv')

    second = s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS, resumed, run_id='k2')
    assert second['skipped'] == TOY_IDS[:1] and second['ran'] == TOY_IDS[1:]
    assert second['moved_incomplete'] == [s05.sae_folder_name(TOY_IDS[1])]
    assert (resumed / 'incomplete' / 'k2' / s05.sae_folder_name(TOY_IDS[1]) / 'per_doc_metrics.csv').exists()
    v9 = second['v9']
    assert v9['pass'] and v9['bit_identical'] and v9['sae_id'] == TOY_IDS[0] and v9['n_docs'] == 3
    assert v9['mean_ablation']['pass'] and v9['mean_ablation']['bit_identical']

    for sid in TOY_IDS:
        for f in RESULT_FILES:
            assert (sae_dir(s05, fresh, sid) / f).read_bytes() == (sae_dir(s05, resumed, sid) / f).read_bytes(), \
                (sid, f)
    for f in LAYER_FILES:
        assert (fresh / 'per_layer/layer_1' / f).read_bytes() == (resumed / 'per_layer/layer_1' / f).read_bytes(), f


def test_finished_sae_is_skipped_and_never_rewritten(s05, tiny_gemma, tmp_path):
    s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS[:1], tmp_path, run_id='k1')
    d = sae_dir(s05, tmp_path, TOY_IDS[0])
    before = {f.name: f.read_bytes() for f in d.iterdir()}
    rec = s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS[:1], tmp_path, run_id='k2')
    assert rec['ran'] == [] and rec['skipped'] == TOY_IDS[:1]
    assert {f.name: f.read_bytes() for f in d.iterdir()} == before


def test_changed_file_behind_done_marker_is_treated_as_unfinished(s05, tiny_gemma, tmp_path):
    s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS[:2], tmp_path, run_id='k1')
    d = sae_dir(s05, tmp_path, TOY_IDS[1])
    (d / 'per_doc_metrics.csv').write_text('tampered\n')        # DONE.json hashes no longer match
    rec = s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS[:2], tmp_path, run_id='k2')
    assert rec['moved_incomplete'] == [s05.sae_folder_name(TOY_IDS[1])] and rec['ran'] == TOY_IDS[1:2]
    assert s05.is_done(d)


def _nudge_stored_value(s05, folder: Path, factor: float):
    df = pd.read_csv(folder / 'per_doc_metrics.csv')
    i = df.index[(df['rank'] == 0) & (df.condition == 'sae_splice')][0]
    df.loc[i, 'ce_new_sum'] *= factor
    df.to_csv(folder / 'per_doc_metrics.csv', index=False)
    (folder / 'DONE.json').unlink()
    s05.write_done(folder)                                      # looks finished, but holds a different value


def test_v9_passes_within_1e6_and_records_not_bit_identical(s05, tiny_gemma, tmp_path):
    s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS[:1], tmp_path, run_id='k1')
    _nudge_stored_value(s05, sae_dir(s05, tmp_path, TOY_IDS[0]), 1 + 1e-9)
    rec = s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS[:2], tmp_path, run_id='k2')
    assert rec['v9']['pass'] and not rec['v9']['bit_identical'] and 0 < rec['v9']['max_rel_diff'] <= 1e-6


def test_v9_stops_the_run_beyond_1e6(s05, tiny_gemma, tmp_path):
    s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS[:1], tmp_path, run_id='k1')
    _nudge_stored_value(s05, sae_dir(s05, tmp_path, TOY_IDS[0]), 1 + 1e-4)
    with pytest.raises(SystemExit, match='V9'):
        s05.run_layer(make_ctx(s05, tiny_gemma), 1, TOY_IDS[:2], tmp_path, run_id='k2')
    assert not sae_dir(s05, tmp_path, TOY_IDS[1]).exists()      # nothing new was evaluated after the failure


# ========================================================================================== V0, fp32 map, V10
def test_v0_hardware_check(s05):
    t4 = {'name': 'Tesla T4', 'compute_capability': '7.5', 'bf16_native': False}
    assert s05.v0_hardware_check([t4, dict(t4)])['pass']
    assert not s05.v0_hardware_check([t4])['pass']                                          # one GPU only
    assert not s05.v0_hardware_check([t4, {**t4, 'name': 'NVIDIA A10G', 'compute_capability': '8.6'}])['pass']


def test_fp32_device_map_covers_every_module_and_keeps_tied_weights_together(s05, tiny_gemma):
    n = tiny_gemma.config.num_hidden_layers
    dmap = s05.fp32_device_map(n)
    needed = {f'model.{name}' for name, _ in tiny_gemma.model.named_children() if name != 'layers'}
    needed |= {f'model.layers.{i}' for i in range(n)} | {'lm_head'}
    assert needed <= set(dmap)
    assert dmap['model.embed_tokens'] == dmap['lm_head'] == 0          # tied embedding / output weight on one GPU
    assert {dmap[f'model.layers.{i}'] for i in range(n)} == {0, 1}
    assert [dmap[f'model.layers.{i}'] for i in range(n)] == sorted(dmap[f'model.layers.{i}'] for i in range(n))
    full = s05.fp32_device_map(26)
    assert [full[f'model.layers.{i}'] for i in (0, 12, 13, 25)] == [0, 0, 1, 1]


def test_v10_secret_scan_finds_token_and_pattern_without_revealing_them(s05, tmp_path):
    token = 'hf_' + 'Ab3' * 12
    (tmp_path / 'clean.csv').write_text('rank,value\n0,1.0\n')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'log.txt').write_text(f'login ok {token}\n')
    (tmp_path / 'other.json').write_text('{"x": "hf_' + 'Z' * 34 + '"}')
    (tmp_path / 'blob.npz').write_bytes(b'\x00\x01' + token.encode() + b'\x00')
    hits = s05.secret_scan(tmp_path, token)
    assert {(h['file'], h['kind']) for h in hits} == {('sub/log.txt', 'exact_token'), ('sub/log.txt', 'hf_pattern'),
                                                      ('other.json', 'hf_pattern'), ('blob.npz', 'exact_token'),
                                                      ('blob.npz', 'hf_pattern')}
    assert token not in json.dumps(hits)
    assert s05.secret_scan(tmp_path / 'sub', 'hf_not_present_anywhere_0000000000000') == \
        [{'file': 'log.txt', 'kind': 'hf_pattern'}]


# ================================================================== Clarification 2: K0 record, step K0F, fp32 output
def _write_record(root: Path, name: str, status: str, spp):
    d = root / 'runs' / name
    d.mkdir(parents=True)
    (d / 'run_info.json').write_text(json.dumps({'status': status, 'smoke': {'seconds_per_pass': spp}}))


def test_projection_uses_only_a_finished_k0_record(s05, tmp_path):
    assert s05.latest_k0_sec_per_pass(tmp_path) is None
    _write_record(tmp_path, 'K0_20261115T010000Z', 'finished', 0.562)
    _write_record(tmp_path, 'K0_20261115T020000Z', 'crashed: RuntimeError: CUDA error', 0.9)   # newer, but crashed
    _write_record(tmp_path, 'K0_20261115T030000Z', 'stopped: V0 FAILED', 0.7)
    assert s05.latest_k0_sec_per_pass(tmp_path) == 0.562


def _fake_gpu_env(s05, monkeypatch, v0_effect):
    """Pretend a clean checkout and a GPU, then let check_v0 stop or crash: nothing touches CUDA."""
    monkeypatch.setattr(s05.ps, 'git_provenance', lambda repo, paths: {'available': True, 'commit': 'x' * 40,
                                                                       'dirty_paths': []})
    monkeypatch.setattr(s05.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(s05, 'enable_strict_determinism', lambda: None)

    def v0(cfg, out_root, run_id):
        raise v0_effect
    monkeypatch.setattr(s05, 'check_v0', v0)


@pytest.mark.parametrize('effect, status', [(SystemExit('V0 FAILED: no GPU'), 'stopped: V0 FAILED: no GPU'),
                                            (RuntimeError('CUDA error: unspecified launch failure'),
                                             'crashed: RuntimeError: CUDA error: unspecified launch failure')])
def test_run_record_is_written_even_when_a_step_stops_or_crashes(s05, monkeypatch, tmp_path, effect, status):
    _fake_gpu_env(s05, monkeypatch, effect)
    with pytest.raises(type(effect)):
        s05.main(['--run', 'K0', '--out', str(tmp_path)])
    recs = list((tmp_path / 'runs').glob('K0_*/run_info.json'))
    assert len(recs) == 1
    rec = json.loads(recs[0].read_text())
    assert rec['status'] == status and rec['run'] == 'K0'
    assert s05.latest_k0_sec_per_pass(tmp_path) is None          # a stopped or crashed K0 never feeds a projection


def test_k0f_sets_launch_blocking_and_needs_no_weekly_hours(s05, monkeypatch, tmp_path):
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING', 'x')   # registers the original state, so it is restored afterwards
    monkeypatch.delenv('CUDA_LAUNCH_BLOCKING')
    _fake_gpu_env(s05, monkeypatch, SystemExit('V0 FAILED: test'))
    with pytest.raises(SystemExit, match='V0 FAILED'):          # got past argument checks without --week-used-hours
        s05.main(['--run', 'K0F', '--out', str(tmp_path)])
    assert s05.os.environ.get('CUDA_LAUNCH_BLOCKING') == '1'
    rec = json.loads(next((tmp_path / 'runs').glob('K0F_*/run_info.json')).read_text())
    assert rec['cuda_launch_blocking'] == '1'


def test_launch_blocking_only_for_k0f_and_budget_rule_unchanged(s05, monkeypatch, tmp_path):
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING', 'x')
    monkeypatch.delenv('CUDA_LAUNCH_BLOCKING')
    _fake_gpu_env(s05, monkeypatch, SystemExit('V0 FAILED: test'))
    with pytest.raises(SystemExit, match='V0 FAILED'):
        s05.main(['--run', 'K0', '--out', str(tmp_path)])
    assert 'CUDA_LAUNCH_BLOCKING' not in s05.os.environ
    with pytest.raises(SystemExit, match='week-used-hours'):
        s05.main(['--run', 'K1', '--out', str(tmp_path)])
    with pytest.raises(SystemExit, match='--widths needs'):
        s05.main(['--run', 'K0F', '--out', str(tmp_path), '--widths', '16k'])


def test_crashed_fp32_folder_is_moved_aside_and_a_finished_one_is_never_overwritten(s05, tmp_path):
    final = tmp_path / 'fp32_reference'
    final.mkdir()
    (final / 'per_doc_metrics.csv').write_text('left by the crashed K0\n')     # written, but never marked DONE
    staging = s05.prepare_output_dir(final, tmp_path, 'K0F_x')
    assert staging.name == 'fp32_reference.partial' and staging.is_dir() and not final.exists()
    assert (tmp_path / 'incomplete' / 'K0F_x' / 'fp32_reference' / 'per_doc_metrics.csv').exists()
    (staging / 'per_doc_metrics.csv').write_text('clean run\n')
    s05.finalize_output_dir(staging, final)
    assert s05.is_done(final) and not staging.exists()
    with pytest.raises(SystemExit, match='refusing to overwrite'):
        s05.prepare_output_dir(final, tmp_path, 'K0F_y')


def test_q0b_refuses_an_unfinished_fp32_reference(s05, s06, tmp_path):
    d = tmp_path / 'fp32_reference'
    d.mkdir()
    (d / 'per_doc_metrics.csv').write_text('a\n1\n')
    assert not s06.fp32_reference_finished(d)
    s05.write_done(d)
    assert s06.fp32_reference_finished(d)
    (d / 'per_doc_metrics.csv').write_text('a\n2\n')
    assert not s06.fp32_reference_finished(d)


# matplotlib version that wrote the committed R3 SVGs (2026-11-17, `de58a2e`). figure_info.json does not record it, so it
# is pinned here. SVG bytes are not stable across matplotlib versions.
R3_FIGURES_MATPLOTLIB = '3.10.7'


def test_vector_figures_reproduce_the_committed_svgs(tmp_path):
    """Script 08 (script 07 run unchanged, PNG saves redirected to SVG) redraws the committed R3 figures from the
    committed tables: SVG only, LF line endings, byte-identical to results/.../figures/ (Instructor Review #9)."""
    mpl = pytest.importorskip('matplotlib')
    if mpl.__version__ != R3_FIGURES_MATPLOTLIB:
        pytest.skip(f'committed R3 SVGs were written with matplotlib {R3_FIGURES_MATPLOTLIB}; installed is '
                    f'{mpl.__version__}, and SVG bytes are not stable across matplotlib versions')
    exp = ROOT / 'results/gemma2_pile/r3_gemma_scope_ladder'
    committed = exp / 'figures'
    if not (exp / 'ladder/run_info.json').exists() or not committed.exists():
        pytest.skip('R3 aggregate outputs are not committed')
    s08 = _load(ROOT / 'scripts/gemma/08_r3_vector_figures.py', 's08')
    out = tmp_path / 'figures'
    s08.main(['--r3-root', str(exp), '--out-dir', str(out)])
    made = sorted(p.name for p in out.iterdir())
    assert made == sorted(p.name for p in committed.iterdir())
    assert not [n for n in made if n.endswith('.png')]
    for name in made:
        data = (out / name).read_bytes()
        assert b'\r\n' not in data, name
        assert data == (committed / name).read_bytes(), name


def test_helpers_are_pure_math_not_rounded(s05):
    """Guard against silently rounding CIs or p-values (all reported values are full precision)."""
    p = s05.one_sided_p(0.123456789, 0.1)
    assert p == pytest.approx(0.5 * math.erfc(1.23456789 / math.sqrt(2)), rel=1e-12)

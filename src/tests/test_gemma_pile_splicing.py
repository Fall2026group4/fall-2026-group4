"""Tests for the M6 SAE / splicing / metric helpers in src/evaluation/pile_splicing.py (Commit 2B).

No downloads: the model tests use a tiny randomly initialised Gemma-2 built from a config, so they exercise the real
Gemma-2 decoder-layer and hidden-state structure of the installed transformers version.

Run from repository root:  python -m pytest -q src/tests/test_gemma_pile_splicing.py
"""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.evaluation import pile_splicing as ps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / 'results/gemma2_pile/gemma_scope_splicing/preregistration.json'


# ------------------------------------------------------------------------------------------------- fixtures/helpers
@pytest.fixture(scope='module')
def tiny_gemma():
    transformers = pytest.importorskip('transformers')
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


def write_prereg(tmp_path, mutate):
    cfg = json.loads(PREREG.read_text(encoding='utf-8'))
    mutate(cfg)
    p = tmp_path / 'prereg.json'
    p.write_text(json.dumps(cfg), encoding='utf-8')
    return p


def run(model, ids, am, hidden_states=False):
    with torch.inference_mode():
        return model(input_ids=ids, attention_mask=am, output_hidden_states=hidden_states)


# --------------------------------------------------------------------------------------- pre-registration (eval part)
def test_committed_preregistration_passes_eval_checks():
    cfg = ps.load_preregistration_eval(PREREG)
    assert [int(x) for x in cfg['layers']] == [5, 12, 19]
    assert cfg['sae']['ids'] == {'5': 'layer_5/width_16k/average_l0_68', '12': 'layer_12/width_16k/average_l0_82',
                                 '19': 'layer_19/width_16k/average_l0_73'}
    assert cfg['validity_gates']['V3_determinism_identity_max_mean_kl_nats'] == 1e-6
    assert [ps.random_error_seed(cfg, L) for L in (5, 12, 19)] == [1005, 1012, 1019]


@pytest.mark.parametrize('mutate, err, match', [
    (lambda c: c['sae'].pop('ids'), KeyError, 'sae.ids'),
    (lambda c: c.pop('layers'), KeyError, 'layers'),
    (lambda c: c['sae']['ids'].pop('19'), ValueError, 'do not match layers'),
    (lambda c: c['sae']['ids'].update({'12': 'layer_5/width_16k/average_l0_68'}), ValueError, 'not an SAE for layer'),
    (lambda c: c['sae'].update({'substitution_allowed': True}), ValueError, 'substitution'),
    (lambda c: c['model'].update({'revision': 'main'}), ValueError, '40-character'),
    (lambda c: c['sae'].update({'hf_revision': 'main'}), ValueError, '40-character'),
    (lambda c: c['hook'].update({'site': 'resid_pre'}), ValueError, 'resid_post'),
    (lambda c: c['conditions'].remove('identity'), ValueError, 'conditions'),
    (lambda c: c['seeds'].update({'random_error': 7}), ValueError, 'random_error'),
    (lambda c: c['validity_gates'].pop('V3_determinism_identity_max_mean_kl_nats'), KeyError, 'V3'),
])
def test_invalid_or_missing_preregistration_fields_fail_loudly(tmp_path, mutate, err, match):
    with pytest.raises(err, match=match):
        ps.load_preregistration_eval(write_prereg(tmp_path, mutate))


# ------------------------------------------------------------------------------------------------------ JumpReLU SAE
def test_jumprelu_matches_hand_computed_example():
    params = {'W_enc': np.array([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]], np.float32),   # d_model 2, d_sae 3
              'b_enc': np.array([0.0, -1.0, 0.5], np.float32),
              'threshold': np.array([0.5, 0.5, 0.5], np.float32),
              'W_dec': np.array([[1.0, 1.0], [0.0, 1.0], [2.0, 0.0]], np.float32),
              'b_dec': np.array([0.1, -0.1], np.float32)}
    sae = ps.JumpReLUSAE(params, d_model=2, d_sae=3)
    x = torch.tensor([[1.0, 1.0], [-1.0, 0.5]])
    # pre = x W_enc + b_enc
    # row 0: pre = [1, 2 - 1, -1 + 1 + 0.5] = [1, 1, 0.5]   -> active where pre > 0.5 (strict) -> f = [1, 1, 0]
    # row 1: pre = [-1, 1 - 1, 1 + 0.5 + 0.5] = [-1, 0, 2] -> f = [0, 0, 2]
    f = sae.encode(x)
    assert torch.allclose(f, torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 2.0]]))
    x_hat, l0 = sae(x)
    # x_hat = f W_dec + b_dec: row 0 = [1, 1] + [0, 1] + [0.1, -0.1]; row 1 = 2 * [2, 0] + [0.1, -0.1]
    assert torch.allclose(x_hat, torch.tensor([[1.1, 1.9], [4.1, -0.1]]))
    assert l0.tolist() == [2, 1]


def test_jumprelu_threshold_is_strict():
    params = {'W_enc': np.eye(2, dtype=np.float32), 'b_enc': np.zeros(2, np.float32),
              'threshold': np.array([0.5, 0.5], np.float32), 'W_dec': np.eye(2, dtype=np.float32),
              'b_dec': np.zeros(2, np.float32)}
    f = ps.JumpReLUSAE(params, 2, 2).encode(torch.tensor([[0.5, 0.50001]]))
    assert f[0, 0] == 0 and f[0, 1] > 0


@pytest.mark.parametrize('key, shape', [('W_enc', (31, 48)), ('W_dec', (48, 33)), ('b_enc', (47,)),
                                        ('b_dec', (32, 1)), ('threshold', (49,))])
def test_sae_dimensions_are_validated(key, shape):
    params = tiny_params()
    params[key] = np.zeros(shape, np.float32)
    with pytest.raises(ValueError, match=key):
        ps.JumpReLUSAE(params, d_model=32, d_sae=48)


def test_sae_missing_parameter_is_an_error():
    params = tiny_params()
    del params['threshold']
    with pytest.raises(ValueError, match='threshold'):
        ps.validate_sae_params(params, 32, 48)


# ------------------------------------------------------------------------------------------------------------ masks
def test_masks_exclude_bos_padding_and_last_real_token():
    am = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    assert ps.splice_mask(am).tolist() == [[False, True, True, True, True], [False, True, True, False, False]]
    # prediction at t predicts t+1: valid for 1 <= t <= len-2
    assert ps.prediction_mask(am).tolist() == [[False, True, True, True, False], [False, True, False, False, False]]


# ------------------------------------------------------------------------------------------ hook on a real Gemma-2
def test_hook_output_is_the_resid_post_hidden_state(tiny_gemma):
    ids = torch.tensor([[2, 5, 9, 11, 40, 7]])
    am = torch.ones_like(ids)
    hooks = {L: ps.LayerSplice(tiny_gemma.model.layers[L]) for L in (0, 1, 2)}
    try:
        out = run(tiny_gemma, ids, am, hidden_states=True)
        for L, h in hooks.items():  # gate V2: output of decoder layer L == hidden_states[L + 1]
            assert torch.equal(h.captured, out.hidden_states[L + 1])
    finally:
        for h in hooks.values():
            h.remove()


def test_identity_splice_gives_zero_kl(tiny_gemma):
    ids = torch.tensor([[2, 5, 9, 11, 40, 7, 3, 8]])
    am = torch.ones_like(ids)
    hook = ps.LayerSplice(tiny_gemma.model.layers[1])
    try:
        logits_o = run(tiny_gemma, ids, am).logits
        hook.set(lambda h: h, ps.splice_mask(am))
        logits_i = run(tiny_gemma, ids, am).logits
    finally:
        hook.remove()
    s = ps.behaviour_sums(logits_o, logits_i, ids, ps.prediction_mask(am))
    assert torch.equal(logits_o, logits_i)
    assert s['kl_sum'] / s['n_pred'] <= 1e-6 and s['ce_orig_sum'] == s['ce_new_sum']
    assert s['top1_agree'] == s['n_pred'] == 6


def test_splice_leaves_bos_and_padding_unchanged(tiny_gemma):
    ids = torch.tensor([[2, 5, 9, 11, 40, 7], [2, 13, 21, 0, 0, 0]])
    am = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    layer = tiny_gemma.model.layers[1]
    splice = ps.LayerSplice(layer)
    seen = {}
    after = layer.register_forward_hook(  # registered second: sees the output returned by the splice hook
        lambda m, i, o: seen.__setitem__('out', (o[0] if isinstance(o, tuple) else o).detach().clone()))
    try:
        run(tiny_gemma, ids, am)
        original = splice.captured.clone()
        mask = ps.splice_mask(am)
        splice.set(lambda h: torch.full_like(h, 7.0), mask)
        run(tiny_gemma, ids, am)
    finally:
        splice.remove()
        after.remove()
    new = seen['out']
    assert torch.equal(new[~mask], original[~mask])           # BOS (col 0) and padding untouched
    assert torch.all(new[mask] == 7.0)                         # every real non-BOS position replaced
    assert torch.equal(splice.captured, original)              # 'captured' is the pre-splice activation


def test_sae_splice_changes_outputs(tiny_gemma):
    ids = torch.tensor([[2, 5, 9, 11, 40, 7, 3, 8]])
    am = torch.ones_like(ids)
    sae = ps.JumpReLUSAE(tiny_params(), d_model=32, d_sae=48)
    hook = ps.LayerSplice(tiny_gemma.model.layers[1])
    try:
        logits_o = run(tiny_gemma, ids, am).logits
        hook.set(lambda h: sae(h.float())[0], ps.splice_mask(am))
        logits_s = run(tiny_gemma, ids, am).logits
    finally:
        hook.remove()
    s = ps.behaviour_sums(logits_o, logits_s, ids, ps.prediction_mask(am))
    assert s['kl_sum'] > 0


# ------------------------------------------------------------------------------------------------- metric formulas
def test_ev_and_reconstruction_sums_match_faithfulness_metrics():
    torch.manual_seed(1)
    docs = [torch.randn(n, 16) * 3 + 2 for n in (5, 11, 7)]
    hats = [x + 0.4 * torch.randn_like(x) for x in docs]
    l0s = [torch.randint(0, 9, (x.shape[0],)) for x in docs]
    sums = [ps.reconstruction_sums(x, h, l) for x, h, l in zip(docs, hats, l0s)]
    ref = fm.reconstruction_metrics(torch.cat(docs), torch.cat(hats), torch.cat(l0s))
    n = np.array([s['n_tok'] for s in sums])
    ev = ps.ev_from_sums([s['sse'] for s in sums], [s['sum_sq'] for s in sums], np.stack([s['sum_x'] for s in sums]), n)
    assert ev == pytest.approx(ref['explained_variance'], rel=1e-10)
    assert sum(s['cos_sum'] for s in sums) / n.sum() == pytest.approx(ref['cosine'], rel=1e-6)
    assert sum(s['sse'] for s in sums) / sum(s['sum_sq'] for s in sums) == pytest.approx(ref['relative_sq_error'])
    assert sum(s['l0_sum'] for s in sums) / n.sum() == pytest.approx(ref['l0'])


def test_behaviour_sums_match_direct_computation():
    torch.manual_seed(2)
    ids = torch.randint(0, 20, (1, 9))
    lo, ln = torch.randn(1, 9, 20), torch.randn(1, 9, 20)
    pm = ps.prediction_mask(torch.ones_like(ids))
    s = ps.behaviour_sums(lo, ln, ids, pm, chunk=3)  # chunking must not change the sums
    po, pn = torch.log_softmax(lo, -1), torch.log_softmax(ln, -1)
    t = torch.arange(1, 8)  # positions 1..len-2
    tgt = ids[0, t + 1]
    assert s['n_pred'] == 7
    assert s['ce_orig_sum'] == pytest.approx(float(-po[0, t, tgt].sum()), rel=1e-5)
    assert s['ce_new_sum'] == pytest.approx(float(-pn[0, t, tgt].sum()), rel=1e-5)
    assert s['kl_sum'] == pytest.approx(float((po.exp() * (po - pn)).sum(-1)[0, t].sum()), rel=1e-5)
    assert s['top1_agree'] == int((po.argmax(-1) == pn.argmax(-1))[0, t].sum())


def test_norm_matched_random_error_has_the_sae_error_norm_and_is_seeded():
    x = torch.randn(10, 32)
    x_hat = x + torch.randn(10, 32)
    a = ps.norm_matched_random_error(x, x_hat, torch.Generator().manual_seed(1005))
    b = ps.norm_matched_random_error(x, x_hat, torch.Generator().manual_seed(1005))
    assert torch.equal(a, b)
    assert torch.allclose((a - x).norm(dim=-1), (x - x_hat).norm(dim=-1), rtol=1e-5)
    assert not torch.allclose(a, x_hat)


def test_script_summarize_matches_hand_computed_aggregates():
    """The script's aggregation (token-weighted ratios of sums, loss recovered, SAE/random ratio, bootstrap CIs)."""
    import importlib.util
    pd = pytest.importorskip('pandas')
    spec = importlib.util.spec_from_file_location('s03', ROOT / 'scripts/gemma/03_pile_gemma_scope_splicing.py')
    s03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s03)  # main() is not run on import

    rng = np.random.default_rng(5)
    ranks, layer, d = [0, 1, 2], 12, 4
    rows = []
    for r in ranks:
        n_pred, ce_o = 10 + r, 20.0 + r
        rows.append({'rank': r, 'layer': layer, 'condition': 'sae_splice', 'n_pred': n_pred, 'ce_orig_sum': ce_o,
                     'ce_new_sum': ce_o + 1 + r, 'kl_sum': 0.5 + r, 'top1_agree': n_pred - 1,
                     'n_tok': n_pred + 1, 'sse': 2.0 + r, 'sum_sq': 50.0 + 5 * r, 'cos_sum': 0.9 * (n_pred + 1),
                     'l0_sum': 80.0 * (n_pred + 1)})
        rows.append({'rank': r, 'layer': layer, 'condition': 'norm_matched_random_error', 'n_pred': n_pred,
                     'ce_orig_sum': ce_o, 'ce_new_sum': ce_o + 0.5, 'kl_sum': 0.2, 'top1_agree': n_pred})
        rows.append({'rank': r, 'layer': layer, 'condition': 'mean_ablation', 'n_pred': n_pred,
                     'ce_orig_sum': ce_o, 'ce_new_sum': ce_o + 10, 'kl_sum': 5.0, 'top1_agree': 0})
    df = pd.DataFrame(rows)
    sum_x = rng.normal(size=(3, 1, d))
    out = s03.summarize(df, sum_x, {0: 0, 1: 1, 2: 2}, ranks, [layer], n_boot=200, seed=0)[0]

    n_pred = sum(10 + r for r in ranks)
    ce_o = sum(20.0 + r for r in ranks)
    ce_sae, ce_rand, ce_mean = ce_o + sum(1 + r for r in ranks), ce_o + 1.5, ce_o + 30
    assert out['delta_ce_sae'] == pytest.approx((ce_sae - ce_o) / n_pred)
    assert out['delta_ce_random'] == pytest.approx(1.5 / n_pred)
    assert out['loss_recovered'] == pytest.approx((ce_mean - ce_sae) / (ce_mean - ce_o))
    assert out['ratio_dce_sae_over_random'] == pytest.approx((ce_sae - ce_o) / 1.5)
    assert out['kl_sae'] == pytest.approx(sum(0.5 + r for r in ranks) / n_pred)
    assert out['cosine'] == pytest.approx(0.9) and out['l0'] == pytest.approx(80.0)
    n_tok = np.array([11, 12, 13])
    assert out['explained_variance'] == pytest.approx(
        ps.ev_from_sums([2, 3, 4], [50, 55, 60], sum_x[:, 0], n_tok))
    assert out['nmse'] == pytest.approx(1 - out['explained_variance'])
    # per-document values bracket the pooled value here (dCE 0.10/0.18/0.25 -> 0.18; loss recovered 0.9/0.8/0.7 -> 0.8),
    # so the percentile interval must contain it
    for name in ('delta_ce_sae', 'loss_recovered'):
        assert out[f'{name}_ci95_lo'] <= out[name] + 1e-12 <= out[f'{name}_ci95_hi'] + 2e-12


# ---------------------------------------------------------------------- engineering safeguards (review A1-A10, H4a)
def load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location('s03', ROOT / 'scripts/gemma/03_pile_gemma_scope_splicing.py')
    s03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s03)  # main() is not run on import
    return s03


def test_A1_assert_finite_rejects_nan_and_inf():
    ps.assert_finite('ok', torch.ones(3), torch.zeros(2, 2))
    for bad in (float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ps.NumericsError, match='non-finite'):
            ps.assert_finite('x', torch.ones(3), torch.tensor([1.0, bad]))
    ps.assert_finite_values('ok', {'a': 1.0, 'b': 3, 'name': 'Pile-CC'})
    with pytest.raises(ps.NumericsError, match="'kl_sum'"):
        ps.assert_finite_values('row', {'kl_sum': float('nan'), 'n_pred': 5})


def test_A1_nonfinite_columns_finds_bad_numeric_columns_only():
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': [1.0, np.inf], 'c': [np.nan, 1.0], 'name': ['x', 'y'],
                       'flag': [True, False]})
    assert ps.nonfinite_columns(df) == ['b', 'c']
    assert ps.nonfinite_columns(df, ['a']) == []


def test_A2_git_provenance_detects_commit_and_dirty_inputs(tmp_path):
    import shutil
    import subprocess
    if shutil.which('git') is None:
        pytest.fail('git is required for the provenance safeguard (A2)')

    def git(*args):
        subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True)

    git('init', '-q')
    git('config', 'user.email', 'test@example.com')
    git('config', 'user.name', 'test')
    (tmp_path / 'code.py').write_text('x = 1\n')
    git('add', 'code.py')
    git('commit', '-q', '-m', 'init')
    clean = ps.git_provenance(tmp_path, ['code.py', 'new.py'])
    assert clean['available'] and len(clean['commit']) == 40 and clean['dirty_paths'] == []
    (tmp_path / 'code.py').write_text('x = 2\n')
    (tmp_path / 'new.py').write_text('y = 1\n')
    assert ps.git_provenance(tmp_path, ['code.py', 'new.py'])['dirty_paths'] == ['code.py', 'new.py']
    assert ps.git_provenance(tmp_path / 'missing', ['code.py'])['available'] is False


def test_A3_cublas_workspace_is_configured_on_import():
    import os
    load_script()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8')


def test_A3_determinism_is_strict_not_warn_only():
    s03 = load_script()
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        s03.enable_strict_determinism()
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    finally:  # restore global state for the other tests
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before[2], before[3]


def test_A4_captured_activation_is_a_copy():
    hook = ps.LayerSplice(torch.nn.Identity())
    out = torch.ones(1, 4, 3)
    hook._hook(None, None, out)
    out.add_(5.0)  # a later in-place change inside the model must not reach the captured activation
    assert torch.equal(hook.captured, torch.ones(1, 4, 3))
    hook.remove()


def test_A6_pinned_commit_check():
    sha = 'c5ebcd40d208330abc697524c919956e692655cf'
    assert ps.check_pinned_commit(sha, sha, 'model') == 'verified'
    assert ps.check_pinned_commit(None, sha, 'model') == 'unverifiable'
    with pytest.raises(ValueError, match='pins'):
        ps.check_pinned_commit('0' * 40, sha, 'model')


@pytest.mark.parametrize('argv', [['--out-tag', 'rerun'], ['--smoke', '--out-tag', 'bad/tag'],
                                  ['--smoke', '--out-tag', '..']])
def test_A8_out_tag_only_with_smoke_and_safe_names(monkeypatch, argv):
    s03 = load_script()
    monkeypatch.setattr('sys.argv', ['03_pile_gemma_scope_splicing.py', *argv])
    with pytest.raises(SystemExit, match='--out-tag needs --smoke'):
        s03.main()  # rejected before any model, data or GPU is touched


def test_H4a_logit_precision_is_zero_for_identical_and_small_for_bf16_rounding():
    torch.manual_seed(6)
    h, w = torch.randn(1, 9, 16), torch.randn(40, 16)
    ids = torch.randint(0, 40, (1, 9))
    pm = ps.prediction_mask(torch.ones_like(ids))
    exact = 30.0 * torch.tanh((h @ w.T) / 30.0)
    same = ps.logit_precision_sums(exact, h, w, 30.0, ids, pm, chunk=4)
    assert same['n_pred'] == 7 and same['kl_ref_vs_model_sum'] == pytest.approx(0, abs=1e-9)
    assert same['abs_ce_diff_sum'] == pytest.approx(0, abs=1e-9) and same['max_abs_logit_diff'] < 1e-5
    rounded = ps.logit_precision_sums(exact.to(torch.bfloat16), h, w, 30.0, ids, pm)
    assert 0 < rounded['max_abs_logit_diff'] <= 0.25        # bf16 spacing is at most 0.125 below |logit| 32
    assert 0 <= rounded['kl_ref_vs_model_sum'] / rounded['n_pred'] < 1e-2
    nocap = ps.logit_precision_sums(h @ w.T, h, w, None, ids, pm)
    assert nocap['max_abs_logit_diff'] < 1e-5


# ------------------------------------------------------------------------------------------- bootstrap and H-G1
def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    vals = np.random.default_rng(3).normal(1.0, 0.5, size=200)
    idx = ps.bootstrap_indices(200, 1000, seed=0)
    assert np.array_equal(idx, ps.bootstrap_indices(200, 1000, seed=0))
    lo, hi = ps.bootstrap_ci(lambda r: vals[r].mean(), idx, 0.95)
    assert lo < vals.mean() < hi


@pytest.mark.parametrize('shift, verdict', [(0.5, 'supported'), (-0.5, 'contradicted'), (0.0, 'inconclusive')])
def test_h_g1_verdicts(shift, verdict):
    rng = np.random.default_rng(4)
    rand = rng.normal(0.1, 0.05, size=300)
    # shift 0: alternating +-0.05, so the mean difference is exactly 0 and the interval must straddle it
    sae = rand + shift + (np.where(np.arange(300) % 2, 0.05, -0.05) if shift == 0.0 else 0.0)
    out = ps.h_g1(sae, rand, ps.bootstrap_indices(300, 1000, seed=0))
    assert out['verdict'] == verdict
    eps = 1e-12  # the +-0.5 cases are constant up to floating-point rounding
    assert out['ci98.33_lo'] <= out['ci95_lo'] + eps and out['ci95_hi'] <= out['ci98.33_hi'] + eps
    assert out['ci95_lo'] - eps <= out['mean_d'] <= out['ci95_hi'] + eps

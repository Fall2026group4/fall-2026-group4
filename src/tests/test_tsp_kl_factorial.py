"""Offline checks for TI-4 (no GPU, checkpoint or TEST row). The trainer itself is verified by GPU gates R0/R1."""
import importlib.util
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _m():
    spec = importlib.util.spec_from_file_location('ti4', ROOT / 'scripts/geometry_of_truth/35_tsp_kl_factorial.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_design_constants():
    m = _m()
    assert m.ARMS == ('ctrl', 'tsp', 'kl', 'tsp_kl') and m.NEW_ARMS == ('kl', 'tsp_kl')
    assert m.SEEDS == (42, 1042, 2042) and m.FOLDS == (0, 1, 2) and m.LAYER == 8 and m.NAME == 'topk'


def test_non_kl_path_uses_stop_at_layer_like_s25():
    """use_kl=False must take the same forward call as S25.run_one (stop_at_layer, no logits)."""
    m = _m()
    src = inspect.getsource(m.run_tsp_kl)
    s25 = inspect.getsource(m.S25.run_one)
    assert 'stop_at_layer=LAYER + 1' in src and 'stop_at_layer=LAYER + 1' in s25
    for line in ('loss.backward()', 'torch.nn.utils.clip_grad_norm_(params, 1.0)', 'sae.normalize_decoder()'):
        assert line in src and line in s25


def test_alpha_rule_matches_r4():
    m = _m()
    src = inspect.getsource(m.run_tsp_kl)
    assert 'alpha = float(nmse_owt.detach() / kl.detach().clamp_min(1e-8))' in src

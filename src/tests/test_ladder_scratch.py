"""Offline tests for the reduced 65k ladder wrapper (no weights downloaded)."""
import importlib.util
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _w():
    spec = importlib.util.spec_from_file_location('l06', ROOT / 'scripts/cross_model/06_ladder_scratch.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sae_list_is_layer12_ladder_only():
    w = _w()
    toks = w.SAES.split(',')
    assert toks[0] == 'canonical' and len(toks) == 1 + 4 + 6 + 5
    assert {t.split(':')[1] for t in toks[1:]} == {'16k', '32k', '65k'}


def test_scratch_fallback_only_after_ebs_miss(tmp_path, monkeypatch):
    w = _w()
    f = tmp_path / 'rev/layer_12/width_65k/average_l0_21/params.npz'
    f.parent.mkdir(parents=True)
    f.write_bytes(b'x')
    monkeypatch.setattr(w, 'SCRATCH_GS', tmp_path)
    ebs = tmp_path / 'ebs.npz'

    def orig(L, width, l0):
        if width == '16k':
            return ebs
        raise FileNotFoundError('not cached')
    xb = types.SimpleNamespace(gemma_scope_path=orig)
    w.patch_paths(xb)
    assert xb.gemma_scope_path(12, '16k', 82) == ebs          # EBS lookup unchanged
    assert xb.gemma_scope_path(12, '65k', 21) == f            # scratch only after an EBS miss
    with pytest.raises(FileNotFoundError):
        xb.gemma_scope_path(12, '65k', 999)

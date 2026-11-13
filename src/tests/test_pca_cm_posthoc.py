"""Tests for the PCA-CM post-hoc multiplicity check (CPU; reads committed CSVs only)."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "pca_cm_posthoc", ROOT / "scripts/shared/pca_cm_posthoc_multiplicity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_holm_step_down():
    holm = _mod().holm
    # m = 4: thresholds 0.0125, 0.0167, 0.025, 0.05 on sorted p
    p = np.array([0.04, 0.001, 0.03, 0.012])
    # 0.03 > 0.025 stops the procedure, so 0.04 (<= 0.05) is not rejected either
    assert holm(p).tolist() == [False, True, False, True]
    assert holm(np.array([0.001, 0.012, 0.02, 0.04])).all()
    assert holm(np.array([0.03, 0.03, 0.03])).tolist() == [False, False, False]


def test_cell_table_reproduces_preregistered_classes_and_summary():
    mod = _mod()
    cells = mod.cell_table()  # asserts prereg_sae_better == decisions.csv "SAE BETTER"
    assert len(cells) == 40
    assert not cells.prereg_sae_worse.any()
    summary = json.loads((mod.OUT / "summary.json").read_text())
    for ref in mod.REFS:
        c = cells[cells.reference == ref]
        assert int(c.prereg_sae_better.sum()) == summary["counts"][ref]["better"]
        assert int((c.prereg_sae_better & c.holm_reject).sum()) == summary["counts"][ref]["better_holm"]
    assert summary["counts"]["capacity_k128"]["better"] == 3
    assert summary["counts"]["task_l0_matched"]["better"] == 8

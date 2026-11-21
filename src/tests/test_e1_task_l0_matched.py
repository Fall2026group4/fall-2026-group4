"""Offline tests for E1 helpers (no checkpoint, cache, GPU or TEST row)."""
import importlib.util
import types
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]


def _e1():
    spec = importlib.util.spec_from_file_location('e1', ROOT / 'scripts/shared/e1_task_l0_matched_topk.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_m2_job_index_matches_m2_loop_order():
    """Replicates M2.main's per-layer job list: for arch: [pca x tasks], then for seed: [sae x tasks]."""
    e1 = _e1()
    M2 = types.SimpleNamespace(ARCHS=('relu', 'topk', 'jumprelu'), SEEDS=(42, 1042, 2042))
    tasks = {'got_cities': 0, 'repe_persona': 0, 'repe_neutral': 0}
    jobs = []
    for arch in M2.ARCHS:
        for tn in tasks:
            jobs.append(('pca', arch, None, tn))
        for s in M2.SEEDS:
            for tn in tasks:
                jobs.append(('sae', arch, s, tn))
    for i, (src, arch, s, tn) in enumerate(jobs):
        if src == 'sae':
            assert e1.m2_job_index(M2, tasks, arch, s, tn) == i


def test_welch_matches_scipy():
    e1 = _e1()
    a, b = np.array([0.10, 0.12, 0.11]), np.array([0.05, 0.09, 0.04])
    d, lo, hi, p = e1.welch(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert np.isclose(p, ref.pvalue) and np.isclose(d, a.mean() - b.mean()) and lo < d < hi


def test_bh_known_values():
    e1 = _e1()
    q = e1.bh([0.01, 0.04, 0.03, 0.20])
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.20], atol=1e-6)


def test_ckpt_paths_are_distinct_and_never_the_baselines():
    e1 = _e1()
    p = e1.ckpt_path(e1.CK_SCRATCH, 8, 77, 42)
    assert 'convergence' not in str(p) and p.parts[-2] == 'topk_k77_seed42'
    assert e1.ckpt_path(e1.CK_SCRATCH, 8, 77, 42) != e1.ckpt_path(e1.CK_SCRATCH, 8, 77, 1042)

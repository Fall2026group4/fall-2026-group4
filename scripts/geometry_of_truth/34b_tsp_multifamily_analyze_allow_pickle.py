#!/usr/bin/env python3
"""TI-2 analysis re-run with ONE declared deviation (2026-11-23, recovery session).

The frozen `34_tsp_multifamily.py analyze` stage crashed (2026-11-22 08:30 UTC) with
"ValueError: Object arrays cannot be loaded when allow_pickle=False": its own eval stage had written the group labels
`comp_g` and `cc_g` of `eval/orig.npz` as numpy object (Python str) arrays. The frozen script is NOT edited (its hash
is checked by check_freeze()). This wrapper imports it unchanged and runs `stage_analyze()` with `numpy.load`
allowing pickle ONLY for `eval/orig.npz`, a file written by this same pipeline (no third-party data). Every other
load keeps allow_pickle=False. No estimand, statistic, threshold, seed or input changes.

Run from the repository root:  python scripts/geometry_of_truth/34b_tsp_multifamily_analyze_allow_pickle.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
TARGET = (REPO / 'results/geometry_of_truth/tsp_multifamily/eval/orig.npz').resolve()
_orig_load = np.load


def _load(file, *a, **k):
    if Path(file).resolve() == TARGET:
        k['allow_pickle'] = True
    return _orig_load(file, *a, **k)


spec = importlib.util.spec_from_file_location('tsp_mf', REPO / 'scripts/geometry_of_truth/34_tsp_multifamily.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.np.load = _load   # m.np is the numpy module; the patch only widens the one file above
m.stage_analyze()

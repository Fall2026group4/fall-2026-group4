#!/usr/bin/env python3
"""XB amendment A4: run and analyse the 128-token shared-span arm WITHOUT modifying the frozen XB scripts (2026-11-21).

The frozen runner (02_matched_splicing.py) and analysis (03_matched_splicing_analysis.py) are imported unchanged. Only
their results root (module attribute `XB`) is pointed at results/cross_model/xb_matched_splicing/a4_ctx128/, which holds
  samples/<corpus>_manifest.csv  -> the A4 128-token manifests (copies of ../samples/<corpus>_ctx128_manifest.csv,
                                     hash-checked against them at every start)
  preregistration.json           -> the frozen A4 protocol
  runs/, analysis_<tag>/         -> A4 outputs only
So the original 1024-token runs (xb_matched_splicing/runs/) and their analysis can never see A4 data, and A4 tables are
unambiguous: in this tree, context 'shared' means the 128-token shared span.

Run:  python scripts/cross_model/05_xb_ctx128.py run --model gpt2 --corpus pile --context shared --saes resjb,old,r2 [--resume]
      python scripts/cross_model/05_xb_ctx128.py analyze --tag a4
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
XB_DIR = REPO / 'results/cross_model/xb_matched_splicing'
A4 = XB_DIR / 'a4_ctx128'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_tree():
    for c in ('pile', 'owt_xm'):
        a, b = A4 / f'samples/{c}_manifest.csv', XB_DIR / f'samples/{c}_ctx128_manifest.csv'
        if not a.exists() or sha(a) != sha(b):
            raise SystemExit(f'{a} missing or differs from {b}')
    a, b = A4 / 'inputs/m6/summary_by_layer.csv', XB_DIR / 'inputs/m6/summary_by_layer.csv'
    if not a.exists() or sha(a) != sha(b):      # read by the frozen analysis; V_M6 is empty (no own-context A4 run)
        raise SystemExit(f'{a} missing or differs from {b}')
    if not (A4 / 'preregistration.json').exists():
        raise SystemExit('A4 preregistration.json missing: freeze the protocol first')


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('run', 'analyze'):
        raise SystemExit(__doc__)
    mode = sys.argv.pop(1)
    check_tree()
    if mode == 'run':
        if '--context' in sys.argv and sys.argv[sys.argv.index('--context') + 1] != 'shared':
            raise SystemExit('A4 uses the shared-span rule only (--context shared)')
        m = load('scripts/cross_model/02_matched_splicing.py', 'xb02')
    else:
        m = load('scripts/cross_model/03_matched_splicing_analysis.py', 'xb03')
    m.XB = A4
    m.main()


if __name__ == '__main__':
    main()

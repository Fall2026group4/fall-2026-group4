#!/usr/bin/env python3
"""Vector (SVG + PDF) exports of the committed PNG results figures (2026-11-13; remediation D10 / Review #5-#8 F7).

The figure scripts below read only committed CSVs, but they also rewrite their CSVs and PNGs in place. This exporter
imports each one unchanged and runs it with:
  * every DataFrame.to_csv, Path.write_text and Path.write_bytes disabled, so no result file is touched;
  * every Figure.savefig(path.png) redirected to <figure dir>/vector/<stem>.svg and .pdf;
  * a PNG copy rendered to --check-dir (if given), so it can be compared pixel by pixel with the committed PNG.
Values therefore come from the same code and the same data as the committed PNGs. The PNGs are kept for provenance.

Not exported: Gemma figures (teammates' pipeline), EDA PNGs, the layer_comparison PNGs (no plotting script in the
repository) and the convergence curves (written by the training scripts).

Run: python scripts/shared/export_vector_figures.py [--check-dir DIR]
"""
import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
log = logging.getLogger('vector')
JOBS = [('scripts/shared/all_layer_frozen_probe_analysis.py', 'main', ['--dataset', 'got']),
        ('scripts/shared/all_layer_frozen_probe_analysis.py', 'main', ['--dataset', 'repe']),
        ('scripts/openwebtext/15_all_layer_baseline_analysis.py', 'main', []),
        ('scripts/shared/spliced_reconstruction_analysis.py', 'main', []),
        ('scripts/shared/plot_causal_and_gemma.py', 'causal', [])]      # causal() only; gemma() is not called


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-dir', type=Path)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    written, orig_savefig = [], Figure.savefig

    def savefig(self, fname, *args, **kw):
        p = Path(fname).resolve()
        rel = p.relative_to(REPO)
        vec = p.parent / 'vector'
        vec.mkdir(exist_ok=True)
        for ext in ('svg', 'pdf'):
            out = vec / f'{p.stem}.{ext}'
            if out.exists():
                raise SystemExit(f'{out} exists; refusing to overwrite.')
            kw2 = {k: v for k, v in kw.items() if k != 'dpi'}
            orig_savefig(self, out, *args, format=ext, **kw2)
            written.append(out.relative_to(REPO))
        if a.check_dir:
            c = a.check_dir / rel
            c.parent.mkdir(parents=True, exist_ok=True)
            orig_savefig(self, c, *args, **kw)

    def blocked(*_a, **_k):
        return None

    Figure.savefig = savefig
    pd.DataFrame.to_csv = blocked
    Path.write_text = blocked
    Path.write_bytes = blocked
    sys.path.insert(0, str(REPO))
    for rel, fn, argv in JOBS:
        spec = importlib.util.spec_from_file_location(Path(rel).stem, REPO / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.argv = [rel] + argv
        getattr(mod, fn)()
        log.info('%s.%s done', rel, fn)
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)   # imported scripts may reconfigure logging
    log.info('%d vector files written', len(written))
    if a.check_dir:
        from matplotlib.image import imread
        diffs = []
        for c in sorted(a.check_dir.rglob('*.png')):
            ref = REPO / c.relative_to(a.check_dir)
            x, y = imread(c), imread(ref)
            d = float(np.abs(x - y).max()) if x.shape == y.shape else float('inf')
            diffs.append(d)
            log.info('%s: max |pixel difference| %s', ref.relative_to(REPO), d)
        log.info('PNG re-render check: %d figures, %d identical', len(diffs), sum(d == 0 for d in diffs))


if __name__ == '__main__':
    main()

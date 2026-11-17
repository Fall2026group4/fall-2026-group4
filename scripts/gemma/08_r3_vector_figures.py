#!/usr/bin/env python3
"""R3 figures as SVG (Instructor Review #9: vector figures, no PNG), drawn by script 07 run unchanged.

scripts/gemma/07_r3_figures.py is part of R3's pre-registered code, so it is not edited (its recorded hash stays
valid). This wrapper loads it from its file and runs its main() with one change: every Figure.savefig(<name>.png) is
redirected to <name>.svg in the same folder, and no PNG is written. Same code, same tables, same values.
This follows the repository's existing pattern in scripts/shared/export_vector_figures.py.

Byte-stable output: SVG ids are salted with a fixed string, the date/creator metadata is removed, and line endings
are written as LF (matplotlib writes CRLF on Windows), so the same tables give the same files on any OS.

SVG only, not PDF: matplotlib's PDF backend imports fontTools for font subsetting, and on the Windows machine used
(2026-11-17) Application Control blocks one of its DLLs, so PDF could not be written there.

After script 07 finishes, figure_info.json (written by script 07) is updated: "figures" lists the SVG names, and
"vector_export" records this script and the redirect. Refuses to overwrite any existing figure or figure_info.json.

Run from repository root:  python scripts/gemma/08_r3_vector_figures.py [--r3-root DIR] [--out-dir DIR]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib.figure import Figure  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
S07_PATH = REPO / 'scripts/gemma/07_r3_figures.py'
FORMATS = ('svg',)
VECTOR_KW = {'svg': {'metadata': {'Date': None, 'Creator': None}}}


def load_s07():
    spec = importlib.util.spec_from_file_location('r3_figures_s07', S07_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vector_names(png_names):
    return [f'{Path(n).stem}.{ext}' for n in png_names for ext in FORMATS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--r3-root', default=None, help="as script 07 (default: script 07's default)")
    ap.add_argument('--out-dir', default=None, help='as script 07 (default: <r3-root>/figures)')
    args = ap.parse_args(argv)

    s07 = load_s07()
    root = Path(args.r3_root) if args.r3_root else s07.EXP
    out = Path(args.out_dir) if args.out_dir else root / 'figures'
    clash = [n for n in vector_names(s07.FIGURES) + [s07.INFO] if (out / n).exists()]
    if clash:
        sys.exit(f'STOP: refusing to overwrite {clash} in {out}')

    matplotlib.rcParams['svg.hashsalt'] = 'r3-gemma-scope-ladder'
    original = Figure.savefig
    written = []

    def redirect(fig, fname, *a, **kw):
        path = Path(fname)
        if path.suffix.lower() != '.png':
            raise RuntimeError(f'unexpected savefig target {path}')
        base = {k: v for k, v in kw.items() if k not in ('dpi', 'metadata', 'format')}
        for ext in FORMATS:
            target = path.with_suffix('.' + ext)
            original(fig, target, format=ext, **base, **VECTOR_KW[ext])
            target.write_bytes(target.read_bytes().replace(b'\r\n', b'\n'))
            written.append(target.name)

    Figure.savefig = redirect
    try:
        s07.main(['--r3-root', str(root), '--out-dir', str(out)])
    finally:
        Figure.savefig = original

    pngs = sorted(p.name for p in out.glob('*.png'))
    if pngs:
        sys.exit(f'STOP: PNG files present in {out}: {pngs}')
    info_path = out / s07.INFO
    info = json.loads(info_path.read_text(encoding='utf-8'))
    info['figures_png_names_in_script_07'] = info['figures']
    info['figures'] = vector_names(info['figures'])
    assert sorted(info['figures']) == sorted(written), (info['figures'], written)
    info['vector_export'] = {'script': 'scripts/gemma/08_r3_vector_figures.py',
                             'how': 'script 07 run unchanged; Figure.savefig(.png) redirected to .svg',
                             'formats': list(FORMATS)}
    info_path.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'wrote {len(written)} vector files to {out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""XB/XT-G batch 2, reduced option (Gemma layer 12 width x L0 ladder; user-approved 2026-11-21).

Runs the FROZEN XB runner (02) and XT-G runner (10) unchanged. The only change is that Gemma Scope weights are also
looked up in the scratch HF cache (/mnt/scratch/hf_cache, filled by `00_fetch_assets.py --batch b2b`), because the
EBS volume has no room for them. Weights are verified against asset_manifest_b2b.json (SHA-256) before any run.

SAEs at layer 12 (one invocation, so every SAE sees the identical documents, controls and resamples):
  16k: canonical (L0 82) + ladder 22 / 41 / 176 / 445   (already on EBS, batch b1 / b2a)
  32k: 12 / 22 / 40 / 76 / 155 / 360                    (scratch, b2b)
  65k: 21 / 38 / 72 / 141 / 297                         (scratch, b2b)
Outputs: XB  results/cross_model/xb_matched_splicing/runs/gemma_pile_shared_b2/   (Pile, 1024 shared span, as pre-registered)
         XT-G results/cross_model/xtg_truth_preservation/runs/gemma_b2/
Run: python scripts/cross_model/06_ladder_scratch.py xb|xtg
"""
import glob
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRATCH_GS = Path('/mnt/scratch/hf_cache/models--google--gemma-scope-2b-pt-res/snapshots')
L12 = {'16k': [22, 41, 176, 445], '32k': [12, 22, 40, 76, 155, 360], '65k': [21, 38, 72, 141, 297]}
SAES = 'canonical,' + ','.join(f'ladder:{w}:{l0}' for w, l0s in L12.items() for l0 in l0s)


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def verify_b2b():
    man = json.loads((REPO / 'results/cross_model/assets/asset_manifest_b2b.json').read_text())
    for f in man['files']:
        p = Path(f['local_path'].replace('~', str(Path.home()), 1))
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for b in iter(lambda: fh.read(1 << 22), b''):
                h.update(b)
        if h.hexdigest() != f['sha256']:
            raise SystemExit(f'{p}: SHA-256 differs from asset_manifest_b2b.json')
    return len(man['files'])


def patch_paths(xb):
    """Make the frozen runner's gemma_scope_path also search the scratch cache (EBS first, as before)."""
    orig = xb.gemma_scope_path

    def gemma_scope_path(L, width, l0):
        try:
            return orig(L, width, l0)
        except FileNotFoundError:
            hits = glob.glob(str(SCRATCH_GS / f'*/layer_{L}/width_{width}/average_l0_{l0}/params.npz'))
            if not hits:
                raise
            return Path(hits[0])
    xb.gemma_scope_path = gemma_scope_path


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('xb', 'xtg'):
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    print('b2b files verified:', verify_b2b(), flush=True)
    if mode == 'xb':
        m = load('scripts/cross_model/02_matched_splicing.py', 'xb02')
        patch_paths(m)
        sys.argv = ['02', '--model', 'gemma', '--corpus', 'pile', '--context', 'shared', '--saes', SAES,
                    '--layers', '12', '--tag', 'b2', '--resume']
    else:
        m = load('scripts/cross_model/10_truth_preservation.py', 'xtg10')
        patch_paths(m.XB)
        sys.argv = ['10', '--model', 'gemma', '--saes', SAES, '--layers', '12', '--tag', 'b2', '--jobs', '2']
    m.main()


if __name__ == '__main__':
    main()

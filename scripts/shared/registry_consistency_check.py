#!/usr/bin/env python3
"""Consistency checks for the experiment registries and documentation links (read-only; CPU).

Checks:
  1. docs/experiment_manifest.csv: every `script_path_current` exists; its current SHA-256 prefix is compared with
     `script_sha256_16_current` (a difference is reported, not fixed: pre-registered scripts must not change).
  2. docs/results_registry.md: every back-ticked repository path that looks like a file or folder exists
     (paths of retired or moved items are reported as missing for review, not as errors, if the line says so).
  3. Relative Markdown links in tracked *.md files resolve.
  4. docs/checkpoint_registry.csv (with --hash-checkpoints): every listed checkpoint present on this machine has the
     registered SHA-256. Checkpoints are gitignored, so absent files are reported as "not on this machine".

Run: python scripts/shared/registry_consistency_check.py [--hash-checkpoints] [--out FILE.json]
Exit status 1 if a check fails (missing manifest script, broken link, checkpoint hash mismatch).
"""
import argparse
import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
log = logging.getLogger('registry_check')


def sha(p, n=None):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()[:n] if n else h.hexdigest()


def manifest():
    df = pd.read_csv(REPO / 'docs/experiment_manifest.csv', dtype=str, keep_default_na=False)
    missing, changed = [], []
    for _, r in df.iterrows():
        p = r['script_path_current'] or r['script']
        if not p:
            continue
        f = REPO / p
        if not f.exists():
            missing.append(dict(task=r['task'], script=p))
        elif r.get('script_sha256_16_current') and sha(f, 16) != r['script_sha256_16_current']:
            changed.append(dict(task=r['task'], script=p))
    return dict(rows=len(df), missing_scripts=missing, changed_since_manifest=changed)


def registry_paths():
    txt = (REPO / 'docs/results_registry.md').read_text()
    paths = set(re.findall(r'`((?:results|scripts|data|docs|src|research_paper|reports)/[^`*{}<> ]+)`', txt))
    missing = sorted(p for p in paths if not (REPO / p.rstrip('/')).exists())
    return dict(paths_checked=len(paths), missing=missing)


def md_links():
    files = subprocess.check_output(['git', 'ls-files', '*.md'], cwd=REPO, text=True).split()
    broken = []
    for f in files:
        for m in re.finditer(r'\]\(([^)\s]+)\)', (REPO / f).read_text(errors='ignore')):
            t = m.group(1).split('#')[0]
            if not t or t.startswith(('http://', 'https://', 'mailto:')):
                continue
            if not ((REPO / f).parent / t).exists():
                broken.append(dict(file=f, target=t))
    return dict(files=len(files), broken=broken)


def checkpoints():
    df = pd.read_csv(REPO / 'docs/checkpoint_registry.csv', dtype=str, keep_default_na=False)
    absent, mismatch, ok = [], [], 0
    for _, r in df.iterrows():
        f = REPO / r['path']
        if not f.exists():
            absent.append(r['path'])
        elif r['sha256'] and sha(f) != r['sha256']:
            mismatch.append(r['path'])
        else:
            ok += 1
    return dict(rows=len(df), hash_ok=ok, not_on_this_machine=len(absent), mismatch=mismatch)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s', force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--hash-checkpoints', action='store_true')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    res = dict(manifest=manifest(), results_registry=registry_paths(), markdown_links=md_links())
    if a.hash_checkpoints:
        res['checkpoint_registry'] = checkpoints()
    fail = bool(res['manifest']['missing_scripts'] or res['markdown_links']['broken']
                or (a.hash_checkpoints and res['checkpoint_registry']['mismatch']))
    res['pass'] = not fail
    for k, v in res.items():
        log.info('%s: %s', k, json.dumps(v, default=str)[:600])
    if a.out:
        if a.out.exists():
            raise SystemExit(f'{a.out} exists; refusing to overwrite.')
        a.out.write_text(json.dumps(res, indent=2) + '\n')
    raise SystemExit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()

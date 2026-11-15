#!/usr/bin/env python3
"""Repository hygiene check for CI (2026-11-13): fails if a tracked file breaks the CLAUDE.md commit rules.

Checks every file tracked by Git (``git ls-files``):
  * forbidden artefacts: ``.pt`` checkpoints, ``.bak`` files, logs, activation caches (``data/activations/``,
    ``data/*/activations/``), ``.npy`` caches;
  * files over 25 MB;
  * text that looks like a credential (GitHub, Hugging Face, AWS, OpenAI-style keys, private-key blocks);
  * absolute home-directory paths (``/home/<user>/``, ``/Users/<user>/``).
Prints one line per finding and exits 1 if there is any. Writes nothing.

Run: python scripts/shared/ci_repo_hygiene.py
"""
import logging
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
log = logging.getLogger('hygiene')

FORBIDDEN = [re.compile(p) for p in (r'\.pt$', r'\.bak$', r'\.log$', r'\.npy$', r'^data/activations/',
                                     r'^data/[^/]+/activations/')]
MAX_BYTES = 25 * 1024 * 1024
SECRETS = [re.compile(p) for p in (
    r'ghp_[A-Za-z0-9]{36}', r'github_pat_[A-Za-z0-9_]{40,}', r'\bhf_[A-Za-z0-9]{30,}', r'\bAKIA[0-9A-Z]{16}\b',
    r'\bsk-[A-Za-z0-9]{32,}', r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')]
HOME = re.compile(r'(?<![\w.])/(?:home|Users)/[A-Za-z0-9_.-]+/')
TEXT_SUFFIXES = {'.py', '.md', '.tex', '.txt', '.json', '.csv', '.yml', '.yaml', '.toml', '.cfg', '.ini', '.sh',
                 '.bib', '.ipynb', '.html', '.svg', '.drawio'}
SELF = 'scripts/shared/ci_repo_hygiene.py'


def tracked():
    out = subprocess.run(['git', 'ls-files', '-z'], cwd=REPO, capture_output=True, check=True).stdout
    return [p for p in out.decode().split('\0') if p]


def findings(paths):
    for rel in paths:
        p = REPO / rel
        if not p.is_file():
            continue
        if any(f.search(rel) for f in FORBIDDEN):
            yield rel, 'forbidden artefact type'
        if p.stat().st_size > MAX_BYTES:
            yield rel, f'{p.stat().st_size / 2**20:.1f} MB > 25 MB'
        if p.suffix.lower() not in TEXT_SUFFIXES or rel == SELF:
            continue
        text = p.read_text(errors='ignore')
        for s in SECRETS:
            if s.search(text):
                yield rel, f'possible credential ({s.pattern[:16]}...)'
        m = HOME.search(text)
        if m:
            yield rel, f'absolute home path {m.group(0)!r}'


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    paths = tracked()
    bad = list(findings(paths))
    for rel, why in bad:
        log.error('%s: %s', rel, why)
    log.info('%d tracked files checked; %d finding(s)', len(paths), len(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())

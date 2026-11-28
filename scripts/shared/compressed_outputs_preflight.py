#!/usr/bin/env python3
"""Preflight for result files committed gzipped (2026-11-26; D55).

Some per-document result files are above the 25 MB repository-hygiene limit and are committed as ``X.gz``
(``gzip -n -9``, lossless) next to ``X.sha256`` (SHA-256 of the uncompressed ``X``; ``sha256sum`` format). The frozen
analysis scripts are not edited, and some of them skip a run folder without ``X`` (e.g.
``scripts/cross_model/03_matched_splicing_analysis.py:load_runs``, also used by ``07_a4_context_comparison.py``).
Running them on a fresh checkout would therefore drop those runs from the analysis without an error. This
preflight runs before them so that cannot happen silently.

A "compressed output" is any ``X.gz`` with a sidecar ``X.sha256`` under the given roots (default ``results/``).

Modes (exit 0 = pass, 1 = findings, 2 = usage error):
  --verify             read-only integrity check (CI): every ``X.gz`` decompresses to bytes whose SHA-256 equals the
                       sidecar, and the sidecar names ``X``. Does not need ``X`` to exist.
  (default)            analysis preflight: additionally require ``X`` to exist with that SHA-256 (missing or stale
                       ``X`` is a finding). Writes nothing.
  --decompress         create each missing ``X`` from ``X.gz`` (temp file, hash verified, then renamed). Never
                       overwrites an existing ``X``; an existing ``X`` with the wrong hash is a finding.
  -- CMD ...           after a passing (or --decompress) preflight, run CMD and exit with its code; CMD does not run
                       if the preflight fails.

Examples (from the repository root):
  python scripts/shared/compressed_outputs_preflight.py --verify
  python scripts/shared/compressed_outputs_preflight.py --decompress results/cross_model/xb_matched_splicing \\
      -- python scripts/cross_model/03_matched_splicing_analysis.py --tag v2
"""
import argparse
import gzip
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHUNK = 1 << 20


def find_pairs(roots):
    """Sorted list of (gz_path, sidecar_path, target_path) for every X.gz with an X.sha256 sidecar under roots."""
    pairs = set()
    for root in roots:
        for gz in Path(root).rglob('*.gz'):
            target = gz.with_suffix('')
            side = target.with_name(target.name + '.sha256')
            if side.is_file():
                pairs.add((gz, side, target))
    return sorted(pairs)


def read_sidecar(side, target):
    """Expected hex digest from a sha256sum-format sidecar; raises ValueError if malformed or naming another file."""
    parts = side.read_text().split()
    if len(parts) != 2 or len(parts[0]) != 64 or any(c not in '0123456789abcdef' for c in parts[0]):
        raise ValueError(f'{side}: not "<sha256>  <name>"')
    if parts[1].lstrip('*') != target.name:
        raise ValueError(f'{side}: names {parts[1]!r}, expected {target.name!r}')
    return parts[0]


def sha256_stream(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(CHUNK), b''):
        h.update(block)
    return h.hexdigest()


def sha256_gz(gz):
    with gzip.open(gz, 'rb') as f:
        return sha256_stream(f)


def sha256_file(p):
    with open(p, 'rb') as f:
        return sha256_stream(f)


def decompress_verified(gz, target, expected):
    """Write target from gz via a temp file in the same folder; rename only if the hash matches."""
    fd, tmp = tempfile.mkstemp(prefix=f'.{target.name}.', dir=target.parent)
    try:
        h = hashlib.sha256()
        with os.fdopen(fd, 'wb') as out, gzip.open(gz, 'rb') as src:
            for block in iter(lambda: src.read(CHUNK), b''):
                h.update(block)
                out.write(block)
        if h.hexdigest() != expected:
            raise ValueError(f'{gz}: decompressed SHA-256 {h.hexdigest()} != sidecar {expected}')
        os.link(tmp, target)  # fails if target appeared meanwhile: never overwrites
    finally:
        os.unlink(tmp)


def check(roots, mode):
    """Return (pairs, findings). mode in {'verify', 'require', 'decompress'}."""
    pairs, findings = find_pairs(roots), []
    for gz, side, target in pairs:
        rel = os.path.relpath(target, REPO)
        try:
            expected = read_sidecar(side, target)
            got = sha256_gz(gz)
        except (ValueError, OSError, EOFError, gzip.BadGzipFile) as e:
            findings.append(f'{rel}: {e}')
            continue
        if got != expected:
            findings.append(f'{rel}.gz: decompressed SHA-256 {got} != sidecar {expected}')
            continue
        if mode == 'verify':
            continue
        if not target.exists():
            if mode == 'decompress':
                try:
                    decompress_verified(gz, target, expected)
                    print(f'decompressed (hash verified): {rel}')
                except (ValueError, OSError) as e:
                    findings.append(f'{rel}: {e}')
            else:
                findings.append(f'{rel}: missing (only {target.name}.gz is present); frozen analysis scripts would '
                                f'skip this run without an error. Run with --decompress first.')
        elif sha256_file(target) != expected:
            findings.append(f'{rel}: exists but its SHA-256 differs from {side.name} (stale or edited); '
                            f'not overwritten. Move it aside and re-run with --decompress.')
    return pairs, findings


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = []
    if '--' in argv:
        i = argv.index('--')
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('roots', nargs='*', help='folders to scan (default: results/)')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--verify', action='store_true', help='integrity only; does not require decompressed files')
    g.add_argument('--decompress', action='store_true', help='create missing decompressed files (never overwrite)')
    a = ap.parse_args(argv)
    if a.verify and cmd:
        ap.error('--verify does not prepare files for a command; use the default mode or --decompress')
    roots = [Path(r) if Path(r).is_absolute() else REPO / r for r in (a.roots or ['results'])]
    for r in roots:
        if not r.is_dir():
            ap.error(f'not a directory: {r}')
    mode = 'verify' if a.verify else 'decompress' if a.decompress else 'require'
    pairs, findings = check(roots, mode)
    for f in findings:
        print(f)
    print(f'{len(pairs)} compressed output(s) checked ({mode}); {len(findings)} finding(s)')
    if findings:
        return 1
    if cmd:
        return subprocess.run(cmd, cwd=REPO).returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())

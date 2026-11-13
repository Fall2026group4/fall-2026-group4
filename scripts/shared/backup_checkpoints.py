#!/usr/bin/env python3
"""Verified local backup of every SAE checkpoint (*.pt outside data/ and .git/).

Writes, OUTSIDE the repository (default ../backups/<date>_checkpoints/):
  manifest.csv   path, bytes, sha256 of every source file (hashed before archiving)
  checkpoints.tar uncompressed tar of those files, paths relative to the repository root
  verify.json    result of re-reading the tar and re-hashing every member against the manifest

Refuses to overwrite an existing backup directory. This is a same-volume copy: it protects against accidental
deletion or corruption inside the repository, NOT against loss of the EC2 volume (no off-machine destination or
credentials are available on this host; see docs/RESEARCH_DECISIONS.md D37).

Run from the repository root:  python scripts/shared/backup_checkpoints.py [--dest DIR]
"""
import argparse
import csv
import hashlib
import json
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha256_stream(f, bufsize=1 << 22):
    h = hashlib.sha256()
    for b in iter(lambda: f.read(bufsize), b''):
        h.update(b)
    return h.hexdigest()


def sources():
    out = []
    for p in sorted(REPO.rglob('*.pt')):
        rel = p.relative_to(REPO)
        if rel.parts[0] in ('data', '.git') or not p.is_file():
            continue
        out.append(rel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', type=Path, default=REPO.parent / 'backups' / f"{time.strftime('%Y-%m-%d')}_checkpoints")
    a = ap.parse_args()
    dest = a.dest.resolve()
    if REPO in dest.parents or dest == REPO:
        raise SystemExit('destination must be outside the repository')
    if dest.exists():
        raise SystemExit(f'{dest} exists; refusing to overwrite')
    dest.mkdir(parents=True)
    t0 = time.time()
    files = sources()
    rows = []
    for rel in files:
        with open(REPO / rel, 'rb') as f:
            rows.append(dict(path=str(rel), bytes=(REPO / rel).stat().st_size, sha256=sha256_stream(f)))
    with open(dest / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'bytes', 'sha256'])
        w.writeheader()
        w.writerows(rows)
    with tarfile.open(dest / 'checkpoints.tar', 'w') as tar:
        for rel in files:
            tar.add(REPO / rel, arcname=str(rel), recursive=False)
    # verification: re-read every member from the archive and compare with the manifest
    want = {r['path']: r for r in rows}
    seen, bad = set(), []
    with tarfile.open(dest / 'checkpoints.tar', 'r') as tar:
        for m in tar:
            if not m.isfile():
                continue
            h = sha256_stream(tar.extractfile(m))
            r = want.get(m.name)
            if r is None or r['sha256'] != h or r['bytes'] != m.size:
                bad.append(m.name)
            seen.add(m.name)
    missing = sorted(set(want) - seen)
    with open(dest / 'checkpoints.tar', 'rb') as f:
        tar_sha = sha256_stream(f)
    res = dict(n_files=len(rows), total_bytes=sum(r['bytes'] for r in rows), tar_sha256=tar_sha,
               members_verified=len(seen) - len(bad), mismatched=bad, missing=missing, ok=not bad and not missing,
               runtime_s=round(time.time() - t0, 1), scope='*.pt outside data/ and .git/ (activation caches excluded: '
               'regenerable from the extraction scripts)', same_volume_only=True)
    (dest / 'verify.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if not res['ok']:
        raise SystemExit('VERIFICATION FAILED')


if __name__ == '__main__':
    main()

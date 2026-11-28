#!/usr/bin/env python3
"""Rebuild XF's `data/got/fresh/geonames_cities/statements.csv` from the upstream GeoNames export (2026-11-26).

Why: the statement text, city names and populations come from GeoNames via the Geometry of Truth repository's
`datasets/geonames.csv`. GeoNames is CC BY 4.0 (attribution required); the GoT repository states no licence, so its
files are not redistributed here (same rule as `data/got/fresh/raw_*/`). The repository keeps only `ids.csv`
(GeoNames id, label, statement country, split: our own assignments) and the pre-registered hash of the full file.

How: this script imports the frozen XF script (`23_xf_fresh_cities_test.py`, unchanged) and runs its own
`stage_build` into a temporary folder, so the statements are produced by exactly the code that built them on
2026-11-23. It then checks, and installs the file only if every check passes:
  1. the upstream file's SHA-256 equals `build_info.json` `source_sha256`;
  2. the rebuilt `statements.csv` SHA-256 equals the value frozen in XF's `preregistration.json`;
  3. the rebuilt (geoname_id, label, country, split) rows equal `ids.csv` row for row.
It never overwrites: an existing `statements.csv` is only verified (exit 1 if its hash differs).

Upstream: a checkout of https://github.com/saprmarks/geometry-of-truth next to this repository
(`../got-repo/datasets/geonames.csv`, as in the frozen script), or pass --geonames PATH.
Run from the repository root:  python scripts/cross_model/23b_xf_rebuild_statements.py [--geonames PATH] [--check-only]
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data/got/fresh/geonames_cities'
TARGET = DATA / 'statements.csv'
IDS = DATA / 'ids.csv'
BUILD_INFO = DATA / 'build_info.json'
PREREG = REPO / 'results/cross_model/xf_fresh_cities_test/preregistration.json'
FROZEN = REPO / 'scripts/cross_model/23_xf_fresh_cities_test.py'
ID_COLUMNS = ('geoname_id', 'label', 'country', 'split')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_hash():
    files = json.loads(PREREG.read_text())['hashes']
    return files['data/got/fresh/geonames_cities/statements.csv']


def id_rows(path):
    with open(path, newline='') as f:
        return [tuple(r[c] for c in ID_COLUMNS) for r in csv.DictReader(f)]


def rebuild(geonames, workdir):
    """Run the frozen stage_build with its data folder redirected to workdir; return the rebuilt file."""
    spec = importlib.util.spec_from_file_location('xf23_frozen', FROZEN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.GEONAMES = Path(geonames)
    m.DATA = Path(workdir) / 'geonames_cities'
    with contextlib.redirect_stdout(io.StringIO()):
        m.stage_build()
    return m.DATA / 'statements.csv'


def check_existing():
    want = expected_hash()
    got = sha(TARGET)
    if got != want:
        sys.exit(f'{TARGET} exists but its SHA-256 {got} differs from the pre-registered {want}; not overwritten.')
    if id_rows(TARGET) != id_rows(IDS):
        sys.exit(f'{TARGET} does not match {IDS} row for row.')
    print(f'{TARGET.relative_to(REPO)} present and verified (SHA-256 {got[:16]}…).')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--geonames', default=str(REPO.parent / 'got-repo/datasets/geonames.csv'))
    ap.add_argument('--check-only', action='store_true', help='verify an existing statements.csv; never rebuild')
    a = ap.parse_args(argv)
    if TARGET.exists():
        check_existing()
        return 0
    if a.check_only:
        sys.exit(f'{TARGET} is absent; run without --check-only to rebuild it.')
    geonames = Path(a.geonames)
    if not geonames.is_file():
        sys.exit(f'upstream {geonames} not found; clone https://github.com/saprmarks/geometry-of-truth next to this '
                 f'repository or pass --geonames.')
    source_want = json.loads(BUILD_INFO.read_text())['source_sha256']
    if sha(geonames) != source_want:
        sys.exit(f'{geonames} SHA-256 {sha(geonames)} differs from build_info source_sha256 {source_want}.')
    with tempfile.TemporaryDirectory() as tmp:
        built = rebuild(geonames, tmp)
        got, want = sha(built), expected_hash()
        if got != want:
            sys.exit(f'rebuilt statements.csv SHA-256 {got} differs from the pre-registered {want}; not installed.')
        if id_rows(built) != id_rows(IDS):
            sys.exit(f'rebuilt statements do not match {IDS} row for row; not installed.')
        shutil.copyfile(built, TARGET)
    print(f'rebuilt and verified {TARGET.relative_to(REPO)} (SHA-256 {want[:16]}…, '
          f'{len(id_rows(IDS))} statements).')
    return 0


if __name__ == '__main__':
    sys.exit(main())

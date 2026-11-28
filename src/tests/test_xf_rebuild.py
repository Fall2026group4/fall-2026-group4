"""XF Option B (2026-11-26): ids.csv integrity, deterministic rebuild of statements.csv, and no overwrite."""
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('xf23b', REPO / 'scripts/cross_model/23b_xf_rebuild_statements.py')
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)


def test_ids_file_matches_the_recorded_build():
    rows = list(csv.DictReader(open(rb.IDS, newline='')))
    info = json.loads(rb.BUILD_INFO.read_text())
    assert list(rows[0]) == list(rb.ID_COLUMNS)
    assert len(rows) == info['n_statements'] == 1426
    ids = Counter(r['geoname_id'] for r in rows)
    assert len(ids) == info['n_cities'] and set(ids.values()) == {2}
    by_id = {}
    for r in rows:
        by_id.setdefault(r['geoname_id'], []).append(r)
    assert all(sorted(x['label'] for x in v) == ['0', '1'] and len({x['split'] for x in v}) == 1 for v in by_id.values())
    assert Counter(r['split'] for r in rows) == {'TEST': info['test']['statements'], 'DEV': info['dev']['statements']}
    assert not any(r['statement'] if 'statement' in r else '' for r in rows)        # no statement text committed


def test_rebuild_reproduces_the_preregistered_file(tmp_path):
    upstream = REPO.parent / 'got-repo/datasets/geonames.csv'
    if not upstream.is_file():
        pytest.skip('upstream GeoNames export not available (../got-repo); the rebuild is checked where it is')
    built = rb.rebuild(upstream, tmp_path)
    assert hashlib.sha256(built.read_bytes()).hexdigest() == rb.expected_hash()
    assert rb.id_rows(built) == rb.id_rows(rb.IDS)


def test_existing_file_with_wrong_hash_is_refused_and_left_untouched(tmp_path, monkeypatch):
    fake = tmp_path / 'statements.csv'
    fake.write_text('geoname_id,label,country,split\n1,1,X,TEST\n')
    before = fake.read_bytes()
    monkeypatch.setattr(rb, 'TARGET', fake)
    with pytest.raises(SystemExit):
        rb.main([])
    assert fake.read_bytes() == before

"""append_experiment must append one row and leave every existing byte of the manifest unchanged (2026-11-26 fix)."""
import csv
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('write_manifest', REPO / 'scripts/shared/write_manifest.py')
wm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wm)

HEADER = ['task', 'experiment', 'script', 'intervention_strengths', 'status', 'script_sha256_16_at_run',
          'script_path_current', 'script_sha256_16_current']


def _manifest(tmp_path):
    p = tmp_path / 'manifest.csv'
    with open(p, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(HEADER)
        w.writerow(['D', 'steering, with "quotes"', 'x.py', 'n/a', 'NA', 'MISSING', 'x.py', 'MISSING'])
        w.writerow(['E', 'empty cells', 'y.py', '', 'None', 'MISSING', 'y.py', 'MISSING'])
    return p


def test_append_keeps_existing_bytes_and_na_like_cells(tmp_path):
    p = _manifest(tmp_path)
    before = p.read_bytes()
    wm.append_experiment(dict(task='C4', experiment='new, row', script='missing.py', status='COMPLETED'), p)
    after = p.read_bytes()
    assert after.startswith(before)
    rows = list(csv.reader(open(p, newline='')))
    assert rows[1][3] == 'n/a' and rows[1][4] == 'NA' and rows[2][4] == 'None'
    assert rows[-1][0] == 'C4' and rows[-1][1] == 'new, row' and len(rows[-1]) == len(HEADER)
    assert rows[-1][HEADER.index('script_sha256_16_at_run')] == 'MISSING'


def test_append_rejects_unknown_columns_without_touching_the_file(tmp_path):
    p = _manifest(tmp_path)
    before = p.read_bytes()
    with pytest.raises(ValueError):
        wm.append_experiment(dict(task='X', not_a_column='v'), p)
    assert p.read_bytes() == before


def test_append_adds_missing_final_newline(tmp_path):
    p = _manifest(tmp_path)
    p.write_bytes(p.read_bytes().rstrip(b'\n'))
    wm.append_experiment(dict(task='Z'), p)
    rows = list(csv.reader(open(p, newline='')))
    assert [r[0] for r in rows] == ['task', 'D', 'E', 'Z']

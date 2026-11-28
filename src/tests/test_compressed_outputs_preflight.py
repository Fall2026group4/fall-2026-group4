"""Tests for scripts/shared/compressed_outputs_preflight.py (D55): gzipped result files cannot be skipped silently."""
import gzip
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'scripts/shared/compressed_outputs_preflight.py'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load('compressed_outputs_preflight', SCRIPT)
DATA = b'rank,layer,cond\n' + b''.join(b'%d,3,sae\n' % i for i in range(2000))


def _pair(folder, data=DATA, name='per_doc.csv', sidecar_hash=None, sidecar_name=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{name}.gz').write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    h = sidecar_hash or hashlib.sha256(data).hexdigest()
    (folder / f'{name}.sha256').write_text(f'{h}  {sidecar_name or name}\n')
    return folder / name


def test_verify_passes_without_decompressed_file(tmp_path):
    _pair(tmp_path / 'runs/gpt2_x_shared_main')
    pairs, findings = P.check([tmp_path], 'verify')
    assert len(pairs) == 1 and findings == []


def test_verify_fails_on_wrong_hash_or_sidecar_name(tmp_path):
    _pair(tmp_path / 'a', sidecar_hash='0' * 64)
    _pair(tmp_path / 'b', sidecar_name='other.csv')
    _, findings = P.check([tmp_path], 'verify')
    assert len(findings) == 2
    assert any('!= sidecar' in f for f in findings) and any("expected 'per_doc.csv'" in f for f in findings)


def test_verify_fails_on_corrupt_gzip(tmp_path):
    target = _pair(tmp_path / 'a')
    gz = target.with_name('per_doc.csv.gz')
    gz.write_bytes(gz.read_bytes()[:-20])
    _, findings = P.check([tmp_path], 'verify')
    assert len(findings) == 1


def test_require_flags_missing_and_stale_files(tmp_path):
    _pair(tmp_path / 'missing')
    stale = _pair(tmp_path / 'stale')
    stale.write_bytes(DATA + b'extra\n')
    ok = _pair(tmp_path / 'ok')
    ok.write_bytes(DATA)
    _, findings = P.check([tmp_path], 'require')
    assert len(findings) == 2
    assert any('missing/per_doc.csv: missing' in f for f in findings)
    assert any('stale/per_doc.csv: exists but' in f for f in findings)


def test_decompress_creates_verified_file_and_never_overwrites(tmp_path):
    target = _pair(tmp_path / 'a')
    _, findings = P.check([tmp_path], 'decompress')
    assert findings == [] and target.read_bytes() == DATA
    assert sorted(p.name for p in target.parent.iterdir()) == ['per_doc.csv', 'per_doc.csv.gz', 'per_doc.csv.sha256']
    target.write_bytes(b'edited\n')
    _, findings = P.check([tmp_path], 'decompress')
    assert len(findings) == 1 and target.read_bytes() == b'edited\n'


def test_decompress_refuses_when_gz_does_not_match_sidecar(tmp_path):
    target = _pair(tmp_path / 'a', sidecar_hash='f' * 64)
    _, findings = P.check([tmp_path], 'decompress')
    assert len(findings) == 1 and not target.exists()
    assert [p.name for p in target.parent.iterdir() if p.name.startswith('.')] == []


def test_command_runs_only_after_passing_preflight(tmp_path):
    _pair(tmp_path / 'a')
    marker = tmp_path / 'ran'
    cmd = [sys.executable, '-c', f'open({str(marker)!r}, "w").close()']
    assert P.main([str(tmp_path), '--', *cmd]) == 1 and not marker.exists()
    assert P.main([str(tmp_path), '--decompress', '--', *cmd]) == 0 and marker.exists()


def test_frozen_xb_loader_skips_gz_only_runs_and_preflight_catches_it(tmp_path):
    """The hazard this preflight exists for, shown on the unmodified frozen loader (03 load_runs, also used by 07)."""
    xb03 = _load('xb03_for_test', REPO / 'scripts/cross_model/03_matched_splicing_analysis.py')
    run = tmp_path / 'runs/gpt2_pile_shared_main'
    _pair(run)
    (run / 'DONE.json').write_text('{}')
    xb03.XB = tmp_path
    assert xb03.load_runs('runs') == []  # silently skipped
    _, findings = P.check([tmp_path], 'require')
    assert len(findings) == 1 and 'gpt2_pile_shared_main/per_doc.csv: missing' in findings[0]


def test_committed_compressed_outputs_are_intact():
    """Every committed X.gz with an X.sha256 sidecar under results/ decompresses to the recorded hash."""
    pairs, findings = P.check([REPO / 'results'], 'verify')
    assert findings == []
    tracked = subprocess.run(['git', 'ls-files', 'results'], cwd=REPO, capture_output=True, text=True).stdout.split()
    tracked_gz = [t for t in tracked if t.endswith('.gz')]
    found = {str(gz.relative_to(REPO)) for gz, _, _ in pairs}
    assert set(tracked_gz) <= found, 'every tracked .gz under results/ must have a .sha256 sidecar'


@pytest.mark.parametrize('args', [['--verify', '--', 'true'], ['--verify', '--decompress']])
def test_usage_errors(args):
    with pytest.raises(SystemExit) as e:
        P.main(args)
    assert e.value.code == 2

"""Release boundary tests: prevent unsafe paths and false finalization."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_validate', REPO / 'scripts/release_validate.py')
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
spec = importlib.util.spec_from_file_location('release_support', REPO / 'scripts/release_support.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


@pytest.mark.parametrize('name', ['../secret', '/etc/passwd', '.env', '.git/config', 'results/x', 'a/../../secret', 'a\\b', 'a//b'])
def test_unsafe_release_paths(tmp_path, name):
    with pytest.raises(ValueError):
        V.safe_path(tmp_path, name)


def test_symlink_cannot_escape_inventory(tmp_path):
    (tmp_path / 'real').write_text('value')
    (tmp_path / 'link').symlink_to(tmp_path / 'real')
    with pytest.raises(ValueError, match='symlink'):
        V.safe_path(tmp_path, 'link')


def test_content_hash_tampering_is_detected(tmp_path):
    rows = []
    for name in sorted(V.REQUIRED):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('reviewed fixture bytes')
        rows.append(dict(path=name, sha256=V.digest(path), bytes=path.stat().st_size))
    (tmp_path / V.CONTROL).write_text(json.dumps({
        'schema': 'lmm-release-contents-v1',
        'checksum_exclusions': [V.CONTROL], 'files': rows,
    }))
    assert V.load_inventory(tmp_path)
    (tmp_path / 'README.md').write_text('unreviewed modification')
    with pytest.raises(ValueError, match='checksum mismatch'):
        V.load_inventory(tmp_path)


def test_blockers_prevent_final_metadata(tmp_path):
    with pytest.raises(ValueError, match='blockers remain'):
        V.finalize_identity(tmp_path, 'a'*40, None, [{'id':'rights'}])


def test_dirty_or_wrong_commit_cannot_finalize(tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'git', lambda root,*args: 'b'*40 if args[:2] == ('rev-parse','HEAD') else ' M README.md')
    with pytest.raises(ValueError, match='exact candidate HEAD'):
        V.finalize_identity(tmp_path, 'a'*40, None, [])
    with pytest.raises(ValueError, match='clean working tree'):
        V.finalize_identity(tmp_path, 'b'*40, None, [])


def test_tag_must_match_candidate(tmp_path, monkeypatch):
    def git(root,*args):
        if args == ('rev-parse','HEAD'): return 'a'*40
        if args[0]=='status': return ''
        if args == ('rev-parse','--is-shallow-repository'): return 'false'
        return 'b'*40
    monkeypatch.setattr(V, 'git', git)
    with pytest.raises(ValueError, match='tag does not resolve'):
        V.finalize_identity(tmp_path, 'a'*40, 'owner-tag', [])


def test_fixture_cannot_be_substituted_for_paper(tmp_path):
    p=tmp_path/'fixture.csv';p.write_text('market,policy,seed,n_episodes\nSynthetic,dqn,42,3\n')
    with pytest.raises(ValueError, match='360'):
        S.generate(p)


def test_saved_support_table():
    source=REPO/'release/evidence/revision_v20/audit/economic_by_seed.csv'
    S.check_expected(S.generate(source), REPO/'release/evidence/benchmark_supplement.csv')

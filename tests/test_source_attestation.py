"""A writing-only attestation cannot authorize changed execution or artifacts."""
import json
from pathlib import Path
import subprocess

import pytest

from lmm.experiments import source_attestation as S


def put(repo, name, text):
    path = repo/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    put(repo, '.gitignore', 'results/\n__pycache__/\n')
    put(repo, 'src/lmm/agent.py', 'RATE = 1\n')
    put(repo, 'configs/base.yaml', 'rate: 1\n')
    put(repo, 'scripts/fit.py', 'RATE = 1\n')
    put(repo, 'requirements.txt', 'numpy==2.0\n')
    put(repo, S.FORECAST_MODULE, 'def fit_contract():\n    return 1\n\ndef validate_forecast_fit():\n    return 2\n')
    put(repo, 'src/lmm/experiments/make_report.py', 'TITLE = "report"\n')
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'training'], cwd=repo, check=True)
    revision = S.git(repo, 'rev-parse', 'HEAD').decode().strip()
    return repo, revision


def minimal_run(root, revision):
    rd = root/'synthetic/dqn_seed42'
    put(rd, 'config_resolved.yaml', 'test: true\n')
    put(rd, 'git_sha.txt', revision+'-dirty\n')
    put(rd, 'pipeline_complete.json', json.dumps({'files': {
        name: S.digest(rd/name) for name in ('config_resolved.yaml', 'git_sha.txt')}}))
    return rd


def attest(repository):
    repo, revision = repository
    root = repo/'results/campaign'
    rd = minimal_run(root, revision)
    put(repo, 'paper/main.tex', 'Manuscript only.\n')
    put(repo, 'src/lmm/experiments/make_report.py', 'TITLE = "audited report"\n')
    path = S.create(repo, root, revision, 'Only the paper was added during training.')
    return S.SourceAttestation.load(repo, root, path), rd


def test_explicit_attestation_preserves_dirty_records_and_binds_sources(repository):
    a, rd = attest(repository)
    before = (rd/'git_sha.txt').read_bytes(), (rd/'pipeline_complete.json').read_bytes()
    a.validate_run(rd)
    put(a.repo, 'paper/main.tex', 'More manuscript edits.\n')
    a.revalidate()
    assert before == ((rd/'git_sha.txt').read_bytes(), (rd/'pipeline_complete.json').read_bytes())
    with pytest.raises(FileExistsError):
        S.create(a.repo, a.root, a.training_revision, 'Do not replace previous evidence.')


@pytest.mark.parametrize('name', ['src/lmm/agent.py', 'configs/base.yaml',
                                  'scripts/fit.py', 'requirements.txt'])
def test_execution_changes_fail_even_with_author_statement(repository, name):
    repo, revision = repository
    root = repo/'results/campaign'
    minimal_run(root, revision)
    put(repo, name, 'changed\n')
    with pytest.raises(ValueError, match='execution source differs'):
        S.create(repo, root, revision, 'Only the paper was changed.')
    assert not (root/S.RELATIVE_PATH).exists()


@pytest.mark.parametrize('change', ['added', 'deleted', 'ignored'])
def test_source_inventory_is_closed(repository, change):
    repo, revision = repository
    if change == 'deleted':
        (repo/'src/lmm/agent.py').unlink()
    else:
        if change == 'ignored':
            with (repo/'.gitignore').open('a') as stream:
                stream.write('src/lmm/extra.py\n')
        put(repo, 'src/lmm/extra.py', 'RATE = 9\n')
    with pytest.raises(ValueError, match='inventory changed'):
        S.audit_sources(repo, revision)


def test_forecast_exception_only_covers_validation_function(repository):
    repo, revision = repository
    path = repo/S.FORECAST_MODULE
    path.write_text(path.read_text().replace('return 2', 'return 3'))
    S.audit_sources(repo, revision)
    path.write_text(path.read_text().replace('return 1', 'return 99'))
    with pytest.raises(ValueError, match='execution source differs'):
        S.audit_sources(repo, revision)


@pytest.mark.parametrize('change', ['reporter', 'revision', 'manifest', 'config', 'extra_run', 'attestation'])
def test_attestation_rejects_later_changes(repository, change):
    a, rd = attest(repository)
    if change == 'reporter':
        put(a.repo, 'src/lmm/experiments/make_report.py', 'TITLE = "changed again"\n')
    elif change == 'extra_run':
        put(a.root, 'another/run/config_resolved.yaml', '{}')
    elif change == 'attestation':
        payload = json.loads(a.path.read_text())
        payload['author_statement'] = 'changed during report generation'
        a.path.write_text(json.dumps(payload))
    else:
        filename = {'revision': 'git_sha.txt', 'manifest': 'pipeline_complete.json',
                    'config': 'config_resolved.yaml'}[change]
        with (rd/filename).open('a') as stream:
            stream.write(' ')
    with pytest.raises((ValueError, OSError)):
        a.revalidate()


def test_wrong_campaign_and_unrelated_revisions_rejected(repository):
    a, rd = attest(repository)
    with pytest.raises(ValueError, match='wrong-campaign'):
        S.SourceAttestation.load(a.repo, a.root.parent, a.path)
    with pytest.raises(ValueError, match='outside'):
        a.validate_run(a.root.parent/'other')
    root = a.root.parent/'unrelated'
    minimal_run(root, 'f'*40)
    with pytest.raises(ValueError, match='unrelated training'):
        S.create(a.repo, root, a.training_revision, 'Only paper changed.')


def test_author_statement_is_required(repository):
    repo, revision = repository
    with pytest.raises(ValueError, match='author statement'):
        S.create(repo, repo/'results', revision, ' ')


@pytest.mark.parametrize('statement', [None, '', ' ', 1])
def test_loaded_attestation_requires_a_nonempty_author_statement(repository, statement):
    a, _ = attest(repository)
    payload = json.loads(a.path.read_text())
    payload['author_statement'] = statement
    a.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='invalid'):
        S.SourceAttestation.load(a.repo, a.root, a.path)


def test_forecast_scope_and_completion_binding(repository):
    repo, revision = repository
    root = repo/'results/campaign'
    minimal_run(root, revision)
    fp = put(root, '_forecasts/synthetic/completion.json', '{}')
    path = S.create(repo, root, revision, 'Only the paper changed.')
    a = S.SourceAttestation.load(repo, root, path)
    assert a.forecast_revision(repo, root, 'synthetic', revision) == revision
    with pytest.raises(ValueError, match='outside'):
        a.forecast_revision(repo, root.parent, 'synthetic', revision)
    with pytest.raises(ValueError, match='not covered'):
        a.forecast_revision(repo, root, 'synthetic', 'f'*40)
    fp.write_text('{"modified": true}')
    with pytest.raises(ValueError, match='not covered'):
        a.forecast_revision(repo, root, 'synthetic', revision)


def test_publication_accepts_attested_dirty_runs_but_still_checks_artifacts(repository):
    from test_treatment_reporting import _publication_runs
    from lmm.experiments import publication as P
    repo, revision = repository
    root = repo/'results/campaign'
    runs = _publication_runs(root/'synthetic_rough_heston')
    for i, run in enumerate(runs):
        (run.run_dir/'git_sha.txt').write_text(revision+('-dirty' if i % 2 else '')+'\n')
        P.write_completion_manifest(run.run_dir)
    with pytest.raises(ValueError, match='git_sha.txt'):
        P.validate_publication_runs(runs, expected_git_sha=revision)
    path = S.create(repo, root, revision, 'Only the paper was added.')
    a = S.SourceAttestation.load(repo, root, path)
    P.validate_publication_runs(runs, source_attestation=a)
    (runs[0].run_dir/'eval/records.csv').write_text('modified evaluation\n')
    with pytest.raises(ValueError, match='changed/missing files'):
        P.validate_publication_runs(runs, source_attestation=a)

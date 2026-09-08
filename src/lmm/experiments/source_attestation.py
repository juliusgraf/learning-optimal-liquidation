"""Explicit post-run provenance for writing-only dirty training revisions.

Historical dirty contents were not captured by the original launcher. Acceptance
therefore requires an author's statement as well as present-day source equality;
it must never be described as a recovered historical source snapshot. Run and
forecast manifests are bound without rewriting their original Git identifiers.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

SCHEMA = 'lmm-writing-only-source-attestation-v1'
RELATIVE_PATH = '_provenance/source_attestation.json'
SOURCE_ROOTS = ('src', 'configs', 'scripts', 'legacy')
SOURCE_FILES = ('pyproject.toml', 'requirements.txt', 'MANIFEST.in')
# These implement reporting/validation, not simulation, fitting or learning.
# Their actual bytes are recorded and must remain identical after attestation.
REPORTING_FILES = frozenset({
    'src/lmm/experiments/source_attestation.py',
    'src/lmm/experiments/publication.py',
    'src/lmm/experiments/make_report.py',
})
FORECAST_MODULE = 'src/lmm/experiments/clearing_campaign.py'
DISCLOSURE = (
    'Post-run source attestation: the author confirms that the dirty working '
    'tree during training contained only manuscript/non-execution changes. '
    'Execution sources and configurations were checked against the recorded '
    'training commit; reporting/validation changes are recorded separately. '
    'Original Git identifiers, checkpoints, evaluations and completion manifests '
    'are preserved. Historical dirty contents were not captured by the launcher; '
    'their classification relies on the author attestation, not a reconstructed '
    'historical snapshot. This changes provenance acceptance only, not policy '
    'selection, inclusion criteria or reported economic estimates.'
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(repo, *args):
    try:
        return subprocess.check_output(['git', *args], cwd=repo)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f'cannot audit Git revision: {args!r}') from exc


def execution_bytes(name, data):
    if name == FORECAST_MODULE:
        # Only this validation function may differ. Fitter commands, contracts,
        # configs, runtime checks, preparation and campaign logic must match.
        tree = ast.parse(data)
        tree.body = [node for node in tree.body if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == 'validate_forecast_fit')]
        return ast.dump(tree, include_attributes=False).encode()
    return data


def audit_sources(repo: Path, revision: str) -> dict:
    """Reject changed, deleted or added execution sources, including ignored code."""
    scope = (*SOURCE_ROOTS, *SOURCE_FILES)
    names = set(git(repo, 'ls-tree', '-r', '--name-only', revision, '--', *scope)
                .decode().splitlines())
    current = {name for name in SOURCE_FILES if (repo/name).is_file()}
    for folder in SOURCE_ROOTS:
        current.update(p.relative_to(repo).as_posix() for p in (repo/folder).rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts
                       and not any(part.endswith('.egg-info') for part in p.parts)
                       and p.relative_to(repo).as_posix() != 'legacy/data.csv'
                       and p.suffix not in {'.pyc', '.pyo'} and p.name != '.DS_Store')
    if names - REPORTING_FILES != current - REPORTING_FILES:
        raise ValueError('execution source inventory changed: ' + repr(sorted(
            (names ^ current) - REPORTING_FILES)))
    executable = {}
    for name in sorted(names - REPORTING_FILES):
        old = execution_bytes(name, git(repo, 'show', f'{revision}:{name}'))
        new = execution_bytes(name, (repo/name).read_bytes())
        if old != new:
            raise ValueError(f'execution source differs from training revision: {name}')
        executable[name] = hashlib.sha256(new).hexdigest()
    reporting = {name: digest(repo/name) for name in sorted(
        (REPORTING_FILES | {FORECAST_MODULE}) & current)}
    return dict(execution=executable, reporting=reporting)


def run_inventory(root: Path) -> dict:
    result = {}
    for config in sorted(root.glob('*/*/config_resolved.yaml')):
        rd = config.parent
        if rd.parent.name.startswith('_'):
            continue
        cm = rd/'pipeline_complete.json'
        payload = json.loads(cm.read_text())
        for name in ('git_sha.txt', 'config_resolved.yaml'):
            if payload['files'].get(name) != digest(rd/name):
                raise ValueError(f'{rd}: {name} differs from completion manifest')
        result[rd.relative_to(root).as_posix()] = dict(
            git_revision=(rd/'git_sha.txt').read_text().strip(),
            completion_sha256=digest(cm))
    if not result:
        raise ValueError('source attestation requires completed saved runs')
    return result


def forecast_inventory(root: Path) -> dict:
    return {p.parent.name: digest(p) for p in sorted(
        (root/'_forecasts').glob('*/completion.json'))}


class SourceAttestation:
    def __init__(self, repo, root, path, payload):
        self.repo, self.root, self.path, self.payload = repo, root, path, payload
        self.training_revision = payload['training_revision']

    @classmethod
    def load(cls, repo, root, path):
        repo, root, path = Path(repo).resolve(), Path(root).resolve(), Path(path).resolve()
        payload = json.loads(path.read_text())
        if (payload.get('schema') != SCHEMA or payload.get('root') != str(root)
                or not isinstance(payload.get('author_statement'), str)
                or not payload['author_statement'].strip()
                or payload.get('disclosure') != DISCLOSURE):
            raise ValueError('invalid or wrong-campaign source attestation')
        value = cls(repo, root, path, payload)
        value.revalidate()
        return value

    def revalidate(self):
        p = self.payload
        if json.loads(self.path.read_text()) != p:
            raise ValueError('source attestation changed during reporting')
        revision = git(self.repo, 'rev-parse', f'{self.training_revision}^{{commit}}').decode().strip()
        if revision != self.training_revision:
            raise ValueError('attestation requires an exact training commit')
        if audit_sources(self.repo, revision) != p['sources']:
            raise ValueError('attested execution/reporting sources changed; review a new attestation')
        if run_inventory(self.root) != p['runs']:
            raise ValueError('attested run inventory, revision or completion manifest changed')
        if forecast_inventory(self.root) != p['forecasts']:
            raise ValueError('attested forecast completion manifest changed')
        if any(r['git_revision'] not in {revision, revision+'-dirty'} for r in p['runs'].values()):
            raise ValueError('attested run has an unrelated training revision')

    def validate_run(self, run_dir):
        try:
            name = Path(run_dir).resolve().relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError('run is outside the attested campaign') from exc
        entry = self.payload['runs'].get(name)
        if entry is None or entry != dict(
                git_revision=(Path(run_dir)/'git_sha.txt').read_text().strip(),
                completion_sha256=digest(Path(run_dir)/'pipeline_complete.json')):
            raise ValueError('run revision/completion is not covered by source attestation')

    def forecast_revision(self, repo, root, setting, recorded_revision):
        if Path(repo).resolve() != self.repo or Path(root).resolve() != self.root:
            raise ValueError('forecast is outside the attested repository/campaign')
        path = self.root/'_forecasts'/setting/'completion.json'
        if (self.payload['forecasts'].get(setting) != digest(path)
                or recorded_revision != self.training_revision):
            raise ValueError('forecast revision/completion is not covered by source attestation')
        return self.training_revision


def create(repo, root, revision, statement):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    if not statement.strip():
        raise ValueError('an explicit author statement about the historical dirty changes is required')
    revision = git(repo, 'rev-parse', f'{revision}^{{commit}}').decode().strip()
    sources = audit_sources(repo, revision)
    runs = run_inventory(root)
    if any(r['git_revision'] not in {revision, revision+'-dirty'} for r in runs.values()):
        raise ValueError('campaign contains unrelated training revisions')
    payload = dict(schema=SCHEMA, root=str(root), training_revision=revision,
        author_statement=statement.strip(), disclosure=DISCLOSURE,
        created_utc=datetime.now(timezone.utc).isoformat(),
        audit_head=git(repo, 'rev-parse', 'HEAD').decode().strip(),
        audit_worktree_status=git(repo, 'status', '--porcelain', '--untracked-files=all').decode(),
        sources=sources, runs=runs, forecasts=forecast_inventory(root))
    path = root/RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not silently replace an earlier attestation.
    with path.open('x') as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True)+'\n')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--training-revision', required=True)
    parser.add_argument('--author-statement', required=True)
    args = parser.parse_args(argv)
    try:
        print(create(Path(__file__).resolve().parents[3], args.root,
                     args.training_revision, args.author_statement))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Source attestation failed: {exc}\n')


if __name__ == '__main__':
    main()

"""Validate an explicit local release candidate. Never publishes, tags or commits.

`--finalize` requires a clean exact commit, cleared blockers and fresh checks.
Final metadata is written outside the checkout to avoid self-reference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'release/content-manifest.json'
SHA = re.compile(r'^[0-9a-f]{64}$')
FORBIDDEN = {'.git', '.env', '.envrc', '.venv', '.cache', '.claude', '.codex', '__pycache__', '.DS_Store'}
REQUIRED = {'README.md', 'REPRODUCING.md', 'LICENSE', 'CITATION.cff', 'THIRD_PARTY_NOTICES.md',
            'release/provenance.json', 'release/READINESS.md', 'release/RUNBOOK.md',
            'release/evidence/campaign.json', 'release/evidence/as_calibration.json',
            'release/evidence/data.json', 'release/evidence/historical_dates.json',
            'release/evidence/mesh_diagnostic.json', 'release/paper-environment.txt',
            'scripts/release_support.py', 'scripts/smoke_release.py'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    rel = PurePosixPath(name)
    if not name or rel.is_absolute() or '..' in rel.parts or '\\' in name or str(rel) != name:
        raise ValueError(f'unsafe release path: {name!r}')
    if any(part in FORBIDDEN for part in rel.parts) or rel.parts[0] in ('results', 'dist', 'build'):
        raise ValueError(f'forbidden release path: {name!r}')
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'symlink in release path: {name!r}')
    if not path.is_file():
        raise ValueError(f'missing release file: {name!r}')
    return path


def load_inventory(root):
    data = json.loads((root / CONTROL).read_text())
    if data.get('schema') != 'lmm-release-contents-v1' or data.get('checksum_exclusions') != [CONTROL]:
        raise ValueError('unsupported content manifest or checksum scope')
    files = data['files']
    if len({r['path'] for r in files}) != len(files):
        raise ValueError('duplicate content path')
    names = {r['path'] for r in files}
    if not REQUIRED <= names or CONTROL in names:
        raise ValueError('required content missing or checksum self-reference')
    for row in files:
        path = safe_path(root, row['path'])
        if not SHA.fullmatch(row['sha256']) or digest(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError(f'content checksum mismatch: {row["path"]}')
        if path.read_bytes()[:128].startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'LFS pointer without materialized content: {row["path"]}')
    return files


def validate_evidence(root):
    manifest = json.loads((root / 'release/provenance.json').read_text())
    for artifact in manifest['artifacts']:
        if artifact['availability'] == 'bundled':
            p = safe_path(root, artifact['path'])
            if digest(p) != artifact['sha256']:
                raise ValueError(f'provenance digest mismatch: {artifact["id"]}')
        # Excluded manuscript entries retain historical identities without
        # requiring the manuscript to be present in this software branch.
        elif artifact['availability'] not in ('restricted-local', 'local-only', 'missing', 'excluded'):
            raise ValueError('unknown artifact availability')
    campaign = json.loads((root / 'release/evidence/campaign.json').read_text())
    runs = campaign['runs']
    if len(runs) != 520 or len({r['run'] for r in runs}) != 520:
        raise ValueError('expected exactly 520 distinct paper runs')
    for run in runs:
        if not re.fullmatch('[0-9a-f]{40}', run['original_revision']):
            raise ValueError('invalid original source identity')
        if set(run['seeds']) != {'training', 'validation', 'test', 'normalizer'}:
            raise ValueError('incomplete seed purposes')
        if len(run['seeds']['test']) != 100 or len(run['seeds']['validation']) != 128:
            raise ValueError('incomplete evaluation seeds')
        for name, values in run['seeds'].items():
            if any(not isinstance(s, int) or s < 0 for s in values):
                raise ValueError('invalid recorded seed')
            for other, other_values in run['seeds'].items():
                if name != other and set(values) & set(other_values):
                    raise ValueError('overlapping split seeds')
        for name, value in run['files'].items():
            rel = PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or not SHA.fullmatch(value):
                raise ValueError('invalid original artifact binding')
    return manifest


def validate_citation_license(root):
    # Focused CFF 1.2 validation for this citation's supported fields. Full schema
    # validation is also documented; this intentionally does not invent metadata.
    cff = yaml.safe_load((root / 'CITATION.cff').read_text())
    if cff['cff-version'] != '1.2.0' or cff['type'] != 'software' or cff['license'] != 'MIT':
        raise ValueError('CFF format/license inconsistent')
    for key in ('title', 'message', 'repository-code'):
        if not isinstance(cff.get(key), str) or not cff[key].strip():
            raise ValueError(f'missing citation {key}')
    expected = [('Graf', 'Julius'), ('Mastrolia', 'Thibaut')]
    if [(a['family-names'], a['given-names']) for a in cff['authors']] != expected:
        raise ValueError('citation authors differ from confirmed metadata')
    text = (root / 'LICENSE').read_text()
    if not text.startswith('MIT License\n\nCopyright (c) 2026 Julius Graf and Thibaut Mastrolia\n'):
        raise ValueError('license differs from confirmed holders/year')
    # Bind the preserved standard text, not a loose keyword match.
    if digest(root / 'LICENSE') != '5bfd22401cd48c2dbba92999897b3f4700600c38283946ba605ae6836c57e8f3':
        raise ValueError('standard MIT text changed: review before release')
    if any(s in json.dumps(cff) for s in ('TODO', 'TBD', '[release', '[full commit')):
        raise ValueError('unresolved citation placeholder')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def run_checks(root):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / 'src'), MPLBACKEND='Agg', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', LMM_TORCH_INTRAOP_THREADS='1',
               LMM_TORCH_INTEROP_THREADS='1')
    commands = [[sys.executable, '-m', 'pytest', '-q', '-m', 'not network and not slow and not market_data'],
                [sys.executable, 'scripts/smoke_release.py'],
                [sys.executable, 'scripts/release_support.py', '--check'],
                [sys.executable, 'scripts/recover_historical_dates.py']]
    with tempfile.TemporaryDirectory(prefix='lmm-release-checks-') as tmp:
        env['MPLCONFIGDIR'] = tmp
        for command in commands:
            subprocess.run(command, cwd=root, env=env, check=True)
    return [{'command': c[1:], 'exit_code': 0} for c in commands]


def finalize_identity(root, commit, tag, blockers):
    if blockers:
        raise ValueError('release blockers remain: ' + ', '.join(b['id'] for b in blockers))
    if not re.fullmatch('[0-9a-f]{40}', commit or '') or git(root, 'rev-parse', 'HEAD') != commit:
        raise ValueError('--commit must equal the full exact candidate HEAD')
    if git(root, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError('finalization requires a clean working tree')
    if git(root, 'rev-parse', '--is-shallow-repository') != 'false':
        raise ValueError('shallow history is not final publication coverage')
    if tag:
        if tag.startswith('-') or git(root, 'rev-parse', '--verify', f'refs/tags/{tag}^{{commit}}') != commit:
            raise ValueError('owner-created tag does not resolve to candidate commit')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run-checks', action='store_true')
    parser.add_argument('--stage', type=Path, help='new directory outside checkout; explicitly DRAFT')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--commit')
    parser.add_argument('--tag')
    parser.add_argument('--metadata-out', type=Path, help='new external file; finalization only')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        files = load_inventory(root)
        manifest = validate_evidence(root)
        validate_citation_license(root)
        blockers = [b for b in manifest['blockers'] if b['status'] != 'resolved']
        if args.finalize:
            finalize_identity(root, args.commit, args.tag, blockers)
            tracked = set(git(root, 'ls-files').splitlines())
            included = {r['path'] for r in files} | {CONTROL}
            if tracked != included:
                raise ValueError('tracked repository contents differ from reviewed inclusion list; inspect omitted/untracked files')
            if any(line.startswith('160000 ') for line in git(root, 'ls-files', '-s').splitlines()):
                raise ValueError('submodules need a separately verified inclusion policy')
            if not args.run_checks or not args.metadata_out:
                raise ValueError('--finalize requires --run-checks and --metadata-out')
        elif args.commit or args.tag or args.metadata_out:
            raise ValueError('commit/tag/final metadata options require --finalize')
        checks = run_checks(root) if args.run_checks else []
        if args.stage:
            dest = args.stage.resolve()
            if dest.exists() or dest == root or root in dest.parents:
                raise ValueError('stage must be a new external directory')
            dest.mkdir(parents=True)
            for row in files:
                source = safe_path(root, row['path'])
                out = dest / row['path']; out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, out)
            shutil.copy2(root / CONTROL, dest / CONTROL)
            print(f'DRAFT staging copy: {dest}; not a publication or committed candidate')
        if args.finalize:
            # Recheck identity and all bytes AFTER commands; tests must not dirty the tree.
            finalize_identity(root, args.commit, args.tag, blockers)
            load_inventory(root)
            out = args.metadata_out.resolve()
            if out.exists() or out == root or root in out.parents:
                raise ValueError('final metadata must be a new external file')
            result = dict(schema='lmm-final-candidate-v1', commit=args.commit, tag=args.tag,
                          content_manifest_sha256=digest(root / CONTROL), checks=checks,
                          publication='NOT PERFORMED', public_access='NOT VERIFIED')
            out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'validation': 'PASS', 'scope': 'final candidate' if args.finalize else 'draft structure and supplied evidence',
                          'files': len(files), 'blockers': [b['id'] for b in blockers],
                          'publication': 'NOT PERFORMED'}, indent=2))
        return 0
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

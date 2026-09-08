"""Refresh checksums for an ALREADY REVIEWED explicit inclusion list.

This never discovers/adds files. Review content and provenance changes first.
"""
import argparse
import json
from pathlib import Path
from release_validate import ROOT, CONTROL, digest, safe_path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--refresh-existing', action='store_true', required=True)
    parser.parse_args()
    path = ROOT / CONTROL
    manifest = json.loads(path.read_text())
    for row in manifest['files']:
        file = safe_path(ROOT, row['path'])
        row.update(sha256=digest(file), bytes=file.stat().st_size)
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Refreshed {len(manifest["files"])} reviewed paths; no files added.')


if __name__ == '__main__':
    main()

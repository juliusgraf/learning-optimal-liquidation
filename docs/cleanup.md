# Repository cleanup — September 6, 2026

This is a historical cleanup record. The submission branch also removes
`CLAUDE.md`, top-level `audit/`, `legacy/`, their characterization tests, and
`docs/*_v19` files/directories. The preservation list below describes the earlier
cleanup, not the current distribution. Manuscript sources are maintained
separately. Git history, including the recovery commits below, is unchanged.

Removed **858 of the 1,116 previously tracked files**
(76.9%), plus 99 ignored development checkpoint/cache files.
The removed files occupied **530.7 MiB**:
23.9 MiB of tracked material and 506.8 MiB of local caches.
Small replacement documentation and recovery records are additional to those gross savings.
Git history was not rewritten, so historical Git-object storage is unchanged.

| Removed category | Tracked files |
|---|---:|
| Superseded verification output, reports, patches and duplicate narrative | 778 |
| Inactive diagnostic configuration overlays | 30 |
| One-off development and patch-generation scripts | 21 |
| Legacy plots | 20 |
| Obsolete manuscript PNGs, unreferenced by the current TeX | 8 |
| Obsolete audit generator | 1 |

The README, documentation index and reproduction guide now lead with the
completed study and reporting workflow. Active development guidance was corrected
to distinguish H information from anchoring and to point to the current pending
manuscript recommendations. All current model source, active configuration
values, tests and dependencies are unchanged.

## Preserved

- All result directories, including every completed v19 run, both follow-ups,
  selected checkpoints, reports and older experimental campaigns.
- Frozen historical data, raw archives, provenance sidecars and calibration inputs.
- Both protected TeX files, all current manuscript inputs and the existing PDF.
- The final analyses, writing plan, manuscript draft and pending v19 review patch,
  including the work that was untracked when cleanup began.
- All regression tests and the small legacy implementation they characterize.
- Compact v19 development ledgers covering accepted and rejected trials, both
  forecast-calibration protocols and the critic-calibration summary.

## Development history and recovery

The archive is the existing Git commit **`83c35645edfe2d15733f91ede4a54e1bd9660fde`**.
Every removed tracked file was checked byte-for-byte against that commit before
deletion. [cleanup_manifest.json](cleanup_manifest.json) records its original
path, byte size and SHA-256. Removal follows artifact age and purpose, not whether
a result was favorable. Older narrative links point to that exact revision.

Read an archived file locally, without restoring clutter:

```bash
git show 83c35645edfe2d15733f91ede4a54e1bd9660fde:docs/pathology_repair_v18.md
```

To extract selected historical material into a separate directory:

```bash
LMM_ARCHIVE_DIR="$(mktemp -d /tmp/lmm-development-history.XXXXXX)"
git archive 83c35645edfe2d15733f91ede4a54e1bd9660fde \
  docs/verification_v18 configs/diagnostic scripts/summarize_v18.py \
  | tar -x -C "$LMM_ARCHIVE_DIR"
```

For executable reproduction of an old investigation, use a separate checkout at
its original recorded revision so code and configurations remain matched.
The current tree's exclusions do not remove files from earlier commits.
The deleted ignored `.pt` files were development checkpoint caches and were never
in Git; the cache section of the manifest records their paths and sizes, not a
claim that those binary files can be recovered from Git. Final campaign weights
were not removed.

## Verification

- Full offline suite before and after: **607 passed, 23 deselected, 14 warnings**
  in each run. No tests were removed or weakened; network/slow tests were excluded.
- All **520 completed run inventories and content hashes** validated.
- Main report and both follow-ups regenerated into temporary directories, with
  all **13 numeric CSV outputs matching exactly** after resolving temporary
  provenance paths and aligning column order. Original reports were not overwritten.
- **172 source, test, active-config, data, manuscript and prior-work files**
  checked byte-identical, including both protected TeX files.
- Manuscript compiled successfully with all build output directed to `/tmp`.
- No full training was launched.

[cleanup_verification.json](cleanup_verification.json) records the commands,
protected hashes, campaign counts and compared report outputs.

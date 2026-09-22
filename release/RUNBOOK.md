# Public paper software release runbook

**Status updated 2026-09-21.** This repository accompanies the open/arXiv version
of the paper. Source code, configurations and compact saved numerical evidence
are distributed through GitHub. The old 2026-09-08 private-retention decisions
are historical snapshots and do not describe the current distribution scope.
Manuscript sources remain separate; do not restore `paper/` for this workflow.

The paper software version is **0.1.0**, identified by the annotated Git tag
`v0.1.0`. [READINESS.md](READINESS.md) records scope and remaining verification;
older test results retain their exact original revisions in
[verification.json](verification.json).

## Prepare and check the source snapshot

1. Review the current diff, [content-manifest.json](content-manifest.json) and
   [provenance.json](provenance.json). Confirm that only intended files are in the
   distribution. Keep restricted input prices, original weights, local results,
   caches and manuscript files outside the snapshot. Preserve original experiment
   identities and hashes; changed documentation must not relabel historical runs.
2. Update the explicit inclusion list when paths change. Refresh checksums only
   after reviewing the changes:

   ```sh
   python scripts/release_inventory.py --refresh-existing
   python scripts/release_validate.py
   ```

   Refreshing hashes never discovers new paths and is not an automatic CI step.
3. Use Python 3.11+ and the installation constraints in the root README. Stage in
   a new directory outside the checkout, then run the CI sequence there:

   ```sh
   python scripts/release_validate.py --stage /tmp/lmm-reviewed-candidate
   cd /tmp/lmm-reviewed-candidate
   python scripts/release_validate.py --run-checks
   python -m pip wheel . --no-deps --wheel-dir /tmp/lmm-reviewed-wheels
   ```

   A fresh dependency install is a separate check from reusing the existing local
   environment. Inspect wheel/sdist inventories and test the installed wheel;
   these software packages do not contain the complete research bundle.
4. Preserve the reviewed candidate with the annotated tag `v0.1.0`, matching
   the version in `CITATION.cff` and `pyproject.toml`. Resolve its full SHA with
   `git rev-parse 'v0.1.0^{commit}'`. Never move a published tag; later changes
   require a new version. The old successful hosted baseline does not verify
   later commits.

## Publish and verify the tagged source

Publish the reviewed commit and its tag to the public GitHub repository. Check
hosted CI for that exact commit, then inspect the tagged source archive as a
signed-out reader. Compare the downloaded bytes with `content-manifest.json`
and rerun the validator from the extracted snapshot. Record the actual tag,
commit, archive hash and verification date outside the tagged snapshot to avoid
a self-referential commit identity. A local annotated tag preserves a version
but does not by itself establish remote availability.

## Match the manuscript's claims

State that this bundle supplies source, resolved configurations, seeds,
provenance and saved numerical summaries. Full original run trees/checkpoints
and restricted historical inputs are not included. The default launchers do not
recreate all 520 refined paper runs by themselves; use the recorded source and
configuration recipe in [REPRODUCING.md](../REPRODUCING.md).

Disclose missing AS fit sample counts/records and the reused historical holdout.
Do not describe hashes as accessible checkpoints or a smoke test as independent
replication. Keep the public manuscript's availability statement consistent with those
limits. The separately maintained manuscript has not been inspected or changed
by this software release update.

## Optional full-release finalization

The public source repository's history contains excluded vendor and development
material. A curated snapshot does not make that history suitable for exposure.
The old history audit remains a separate record from this tagged compact
snapshot. Resolve the relevant blockers in `provenance.json` with recorded
evidence before using full-release finalization:

```sh
python scripts/release_validate.py --finalize --commit FULL_CANDIDATE_SHA \
  --run-checks --metadata-out /tmp/lmm-final-candidate.json
```

`FULL_CANDIDATE_SHA` is the full actual clean candidate commit. An optional
`--tag` must name a tag resolving to that commit. Finalization refuses unresolved
blockers, a dirty tree, mismatched inventory or missing content, and writes only
external metadata. It never commits, tags, pushes, publishes or changes visibility.
After any separately authorized publication, verify signed-out access and compare
the downloaded bytes with the final recorded checksums.

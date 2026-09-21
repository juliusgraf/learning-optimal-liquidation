# Submission candidate runbook

Use this workflow for the compact software/evidence branch. Manuscript sources
are maintained separately; do not restore `paper/` to complete this workflow.
The current intended review venue is an anonymous.4open.science mirror, which
has not yet been created. [READINESS.md](READINESS.md) records verified scope and
remaining steps; older validation results retain their exact tested revisions
in [verification.json](verification.json).

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
4. Commit the reviewed candidate and verify hosted CI for the exact new commit.
   Record its full SHA as the submission source identity. Do not treat the old
   successful hosted baseline as verification of later commits.

## Verify the anonymous review copy

Create the mirror from the intended candidate and inspect it as a signed-out
reader. Check authors/citations/copyright metadata, repository and Actions links,
package docstrings, validation literals, original diagnostic paths and any other
identifying content. Also check file availability, download access and the
README's relative file/section links. Do not assume the service has transformed
every type of file or external link correctly.

Hash validation applies to exact bytes. If the review service modifies files,
its downloaded copy is a different artifact from the validated source snapshot.
The content manifest, frozen-source fixture and MIT-text validator may then
reject it. Check that actual downloadable artifact; use a deliberately reviewed
and consistently rebound review copy if required, preserving the licensed
original and the relationship to its source. Do not blindly refresh frozen
executable reference code or remove checks to make a redacted copy pass.

Record the reviewer URL, verified source identity, verification date and the scope
of any transformations. Anonymous-site access is currently unverified. Neither
that access nor permission to publish follows from local draft validation.

## Match the manuscript's claims

State that this bundle supplies source, resolved configurations, seeds,
provenance and saved numerical summaries. Full original run trees/checkpoints
and restricted historical inputs are not included. The default launchers do not
recreate all 520 refined paper runs by themselves; use the recorded source and
configuration recipe in [REPRODUCING.md](../REPRODUCING.md).

Disclose missing AS fit sample counts/records and the reused historical holdout.
Do not describe hashes as accessible checkpoints or a smoke test as independent
replication. Evaluate any non-public-data exemption request against the journal's
[code and data policy](https://pubsonline.informs.org/page/opre/code-and-data-disclosure-policy).
The separately maintained manuscript and cover letter have not been inspected
or changed in this review.

## Optional later public-release finalization

The public source repository's history contains excluded vendor and development
material. A curated snapshot does not make that history suitable for exposure.
Review the distribution strategy before any visibility change or history rewrite.
Resolve the relevant blockers in `provenance.json` with recorded evidence and
choose an immutable release identity before using finalization:

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

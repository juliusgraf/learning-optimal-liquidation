# Owner publication runbook — draft, not executed

The owner has committed the preparation and CI repair in baseline
`02a8f00524b63b288e7e6ae813b55b9d195fb368`. Tests run #14 passed for that commit.
This documentation update performs no commit, tag, push, release, upload,
visibility change or history rewrite. The final paper release tag/identifier is
still unassigned, and the baseline is not automatically its eventual identity.
The owner approved the current readiness-documentation update and confirmed private
retention because history-distribution and artifact-availability remain unresolved.
Availability upon request remains the intended scope. The publication steps below
apply only if that decision changes with
separate authorization and the remaining gates are resolved.

1. Review `READINESS.md`, `provenance.json` and the private audit notes retained
   outside this repository. Resolve the vendor-data/history exposure first.
   Confirm the scope of every reachable branch/tag and any hosted content that
   will become public. A curated release asset does not hide Git history or
   GitHub's automatic source archives. Do not simply change visibility.
   Choose an authorized preservation/history strategy; any destructive rewrite,
   branch removal, force-push or migration requires separate approval. Preserve
   original experiment objects privately and maintain the mapping to their SHAs.
2. Keep hosted verification tied to the tested commit. The revised regression,
   constrained CPU dependency installation and wheel build passed on Ubuntu 24.04 /
   Python 3.11.16 at the baseline above; see `verification.json`. Check hosted CI
   separately for each later candidate commit. The regression compares current
   and frozen original source exactly within the same runtime; differing serialized
   hashes across runtimes are a documented limitation, not an unresolved Linux
   release gate. Do not weaken the comparison or automatically refresh expected
   results. Preserve original dependency records and distinguish them from each
   newly tested environment.
3. Decide the public bundle's advertised scope. The draft supplies configurations,
   seed summaries, fitted metadata, original hashes and partial result regeneration.
   Original checkpoints, full evaluation/training/forecast logs and raw inputs
   are not in the staging package. Obtain stable, versioned, authorized locators
   with SHA-256 for any promised external artifacts; distinguish restricted vendor
   access from missing public artifacts. If retaining a compact bundle, qualify
   the manuscript claim and document which replication steps cannot run from it.
   Accepted/excluded AS calibration counts and original fit samples are still
   absent, while the actual coefficients are preserved. This is a disclosed
   provenance limitation, not recovered evidence. A fresh fit is a new check.
   The retained manuscript still needs the owner's later approved update described
   below; it is not a prerequisite for this documentation-only status correction.
4. Review the explicit `content-manifest.json` inclusion list, all current tracked
   files omitted from that list, and the sanitized copies of research metadata.
   The draft excludes manuscript build logs, local caches, raw inputs, original
   weights and local run directories. Several excluded files remain tracked;
   decide their archival/removal without deleting unique evidence. MIT software
   scope and separate research-asset distribution permission are recorded in
   `THIRD_PARTY_NOTICES.md`; no new asset license was assigned.
5. Resolve each blocker in `provenance.json` with concrete evidence. Update
   provenance hashes only for reviewed changes; never rewrite historical output
   hashes to match altered files. Review and refresh the explicit content inventory
   with `python scripts/release_inventory.py --refresh-existing` after reviewing
   the diff. Adding paths requires editing the inclusion list deliberately.
6. Test a new staging directory outside the checkout:

   ```sh
   python scripts/release_validate.py --stage /tmp/lmm-reviewed-candidate
   cd /tmp/lmm-reviewed-candidate
   python scripts/release_validate.py --run-checks
   ```

   Use a new virtual environment and the recorded constraints where package access
   is available. Test the installed wheel as well as `PYTHONPATH=src`. Review the
   wheel/sdist contents: a software wheel is not the paper replication bundle.
   A draft validator exit 0 means structural checks passed, not that blockers are cleared.
7. Review and commit the candidate yourself after authorization. No file needs to
   contain the SHA of its own containing commit. From the resulting clean checkout,
   take `git rev-parse HEAD` as the full candidate SHA and run:

   ```sh
   python scripts/release_validate.py --finalize --commit FULL_CANDIDATE_SHA \
     --run-checks --metadata-out /tmp/lmm-final-candidate.json
   ```

   `FULL_CANDIDATE_SHA` is an unfinished template value, not a suggested revision.
   Finalization refuses unresolved blockers, dirty/untracked work, missing content,
   submodules without an inclusion policy, LFS pointers, checksum mismatches and
   tracked content outside the inclusion list. Test against the committed candidate
   again; passing tests on an earlier commit or a modified working tree do not
   replace this step. Check hosted CI for the new commit separately.
8. Select a version identifier, create its tag at that exact commit yourself, then
   rerun finalization with `--tag OWNER_SELECTED_TAG` and a new metadata output file.
   This verifies tag-to-commit agreement. Never move an existing published tag.
   Create local release assets from the reviewed inclusion list or verified staged
   copy, check their inventories, and hash the resulting archives. Attach the external
   final metadata and checksums to the release. `content-manifest.json` hashes every
   listed source file except itself; external archive SHA-256 covers the whole archive
   including that manifest. No checksum claims to hash itself.
9. Only with separate publication authorization, push the reviewed history/commit
   and tag, prepare the versioned release and upload the reviewed assets. Consider
   GitHub immutable releases: create a draft, attach every asset, then publish.
   GitHub locks the tag and assets once immutable publication occurs, while release
   notes remain editable. No setting has been enabled here. See the official
   [immutable release documentation](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases).
   An archival DOI is optional; do not invent one or an acceptance/publication date.
10. Verify **unauthenticated** access in a signed-out session: repository URL,
    release page, exact tag/commit, automatic source archive and every attached
    asset. Download the public bytes and compare SHA-256 to final metadata and
    the release manifest. Record URLs, UTC verification time, full commit,
    release/tag identifier, asset sizes/hashes and the access result in a separate
    publication verification record. Do not reuse authenticated API success as
    evidence of public access. Fill the manuscript template only after this passes.

GitHub automatically offers source archives for the tagged repository tree;
[its release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
explains why a curated attachment is not the only publication surface.

## Later author-approved manuscript update

The retained `paper/main.tex` lacks the author-reported AS missing-samples/counts
disclosure and the Code and Data Availability statement for source code,
experiment configurations and saved numerical summaries upon request, with
Alpaca historical inputs obtained separately. The manuscript is deliberately
untouched in this readiness-documentation update; its replacement is a separate
owner task, not a prerequisite for reviewing and committing this update.

1. Import the author-approved manuscript sources and required assets into `paper/`.
   Inspect the applicable build process before executing it, rebuild the corresponding
   PDF, and inspect the resulting PDF to confirm that it actually contains the
   intended changes. Do not reconstruct the author's wording from this checklist.
2. Review the manuscript diff and check affected claims, tables, figures and
   references against the supporting outputs. Refreshed checksums establish byte
   identity, not scientific correctness. Do not silently regenerate expected results
   or alter numerical evidence to match revised prose.
3. Update the current manuscript artifact hashes and descriptions in
   `release/provenance.json` (including the `paper/main.tex` and `paper/main.pdf`
   entries), and other current-artifact bindings only where they genuinely depend
   on changed files. Explicitly record that the manuscript was updated **after
   release preparation**. Preserve original experiment revisions, input hashes,
   calibration records, saved numerical results and historical verification records;
   do not rewrite historical evidence to make changed files appear original. A new
   current binding or verification record must remain distinct from an old record.
4. Deliberately review any added or removed release paths and amend the explicit
   inclusion list only for approved path changes. Then refresh its existing entries:

   ```sh
   python scripts/release_inventory.py --refresh-existing
   ```

5. Validate the reviewed contents and run the offline checks:

   ```sh
   python scripts/release_validate.py --run-checks
   ```

   Do not disable checksum checks or add automatic checksum updating to CI.
6. Leave the changes for author review, commit and push. Check the hosted CI result
   for that new commit separately. Preserve prior successful CI records with their
   exact tested commits; run #14 verifies `02a8f00524b63b288e7e6ae813b55b9d195fb368`,
   not every future HEAD. Do not assign an immutable paper-release identity until
   its separate finalization and publication checks are complete.

## Unfinished manuscript availability template

“The replication bundle accompanying this paper is available in the GitHub
repository [verified repository citation], release [published release identifier],
commit [full finalized release commit SHA].”

When the compact/restricted-data scope is retained, accompany it with:

“The bundle includes source, configurations, provenance records and saved
seed-level summaries. Historical quote inputs require separately authorized
access and are not redistributed. Original checkpoints and full run records
[insert verified versioned location, or state explicitly that they are not
included]; the supplied bundle supports [state the verified regeneration scope].”

These are unfinished templates, not statements ready for manuscript use. The
repository remains private in the baseline run metadata; its releases list was
empty in the original audit. No release identifier has been selected. The
preparation is committed at the baseline above, but final release approval and
public availability have not been established. The owner's intended upon-request
manuscript wording must come from the author-approved source, not this future
public-release template.

## Draft release notes

This candidate adds a compact evidence bundle for the current 520-run v20 study,
traces original synthetic and historical source revisions separately, provides
recorded configurations/seeds/calibration/selection metadata, and regenerates
paired benchmark support from unrounded saved summaries. First-party software
uses MIT with confirmed holders and year; non-code asset scope is explicit.

No simulator, reward, selection, RNG, evaluation or reported-result definition
changed during preparation. No full experiment or mesh/calibration rerun was
performed. Hosted Linux validation is passing at the recorded baseline, with
cross-runtime bitwise equality unclaimed. Distribution-history review, original
artifact availability and final candidate/public verification remain open. AS
sample accounting remains a documented provenance limitation. Use the limitations
in the readiness report as the release notes' scope;
do not announce a completed public replication until the corresponding gates close.

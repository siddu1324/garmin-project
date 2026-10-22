# Repository publication through Notebook 21

This maintenance checkpoint publishes source, configurations, tests, Notebooks 00–21 and reviewed
aggregate reports to the existing `siddu1324/garmin-project` repository on `main`. It does not alter
scientific results, frozen assignments, raw inputs or the sealed final test.

- `maintenance.json` pins the before/after hashes of the main README, ignore rules and Git privacy guard.
  The exact old hashes remain in earlier checkpoint snapshots; only these named maintenance files
  receive exceptions. Unexpected changes to any research artifact still fail verification.
- `history/README_before_publication.md` preserves the earlier project overview.
- `notebook_verification.json` records the new execution/integrity check without replacing Notebook 21's
  original verification report. The original executed notebook is also preserved privately.
- `verification.json` records tests, lint, publication privacy checks and staged-file review.
- `config/publication_allowlist.json` lists reviewed shareable paths. CSV exceptions are individually
  enumerated in `.gitignore`; no blanket data-file exception is introduced.

The main README and report index provide navigation; version directories and supersession records
remain unchanged. Redundant source ZIP/unpacked copies, caches and private outputs are excluded
rather than deleted. Canonical implementations are under `src/`.

Excluded: raw archives, participant records, normalized/feature Parquets, participant mappings and
split assignments, private manifests, individual predictions, models, local configuration, credentials,
`.env` files and temporary execution files. Aggregate provenance manifests are included only when their
contents describe shareable files/configuration and contain no participant keys or source filenames.

Notebook 21 checks saved development artifacts and never runs a new model search. Its publication
verification explicitly retains old checksums and records maintenance exceptions; it does not silently
relabel old seals as matching new code. Earlier sealed notebooks remain historical snapshots; reproduce
them with their recorded source/bundle versions. Viewing public results and running synthetic tests is
possible without private participant data; data-dependent re-execution requires the authorized archive.

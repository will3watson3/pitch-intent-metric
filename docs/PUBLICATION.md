# Public repository scope

The repository is intentionally limited to the pipeline, review interfaces, current comparable-pitch model, portable regression tests, synthetic example, and documentation. A root `.gitignore` allowlist controls which scripts are published.

The following remain local: real video and frames, downloaded Statcast exports, reviewed labels, generated reports, binary models, external logo assets, credentials, caches, virtual environments, one-off experiment scripts, old notes, and the separate `scoutdeck/` application. Ignoring these files does not delete or relocate them.

The README includes an aggregate evaluation snapshot so readers can assess the current limitations. It does not include the underlying private research sample. The public demo uses invented records.

When adding a supported script, add its direct imports and subprocess dependencies to the allowlist, document its entry point, and check it from a clean checkout. Before committing, inspect `git status --short`, `git diff --cached --stat`, and the staged contents. Keep real credentials out of source code and never use force-add to bypass exclusions without reviewing the specific file.

Broadcast footage, logos, hosted detector models, and downloaded data are external resources and are not redistributed here. No open-source license has been selected for this initial publication; public visibility alone does not grant a general reuse license.

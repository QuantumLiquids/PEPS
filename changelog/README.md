# Changelog Layout

Use the changelog directory with two clear tracks:

- `changelog/details/`: incremental change entries (feature, fix, refactor, breaking-change notes) that feed upcoming releases.
- `changelog/releases/`: finalized version release notes (`release-vX.Y.Z`).

## Naming Convention

- Detail entry: `YYYY-MM-DD-<topic>.md`
- Release entry: `YYYY-MM-DD-release-vX.Y.Z.md`

## Authoring Rules

- Add day-to-day change notes to `details/`.
- Add version summary notes only to `releases/`.
- Link release notes to prior release notes under `changelog/releases/`.
- Keep release notes referencing detailed notes under `changelog/details/` as needed.

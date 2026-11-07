# Agents - Agent Guidance

This file holds the repository's own facts on top of the global contract.
`CLAUDE.md` imports it.

## State

The repository holds documents and CI only. Code arrives with its first slice
and brings its own toolchain, commands and tests; this file then names them.

## Where things live

- `docs/VISION.md` is the operator-blessed vision text. Change its words only
  with his yes.
- The persona, memory and runner shape is not ruled yet. `docs/ARCHITECTURE.md`
  states only what exists.

## Checks

- `Documents` (`.github/workflows/ci.yml`) runs lychee offline over every
  tracked Markdown file and fails on a relative link or anchor that points
  nowhere. Its arguments live in the workflow, so the root holds no lychee
  config.
- `quality/config.toml` configures the shared gate: `mandatory_jobs` names
  `Documents`, and there is no `legacy` key. The required check is
  `quality / verdict`; overnightworks/quality's README owns the gate's
  semantics. The checks that come with code join `mandatory_jobs` in that slice.
- The root holds only what the gate's `root-layout` rule allows; a new root
  entry needs a `[[root.allow]]` entry with its reason.
- `sonar.sources` names `.github` whole; add each directory of code with the
  slice that brings it.

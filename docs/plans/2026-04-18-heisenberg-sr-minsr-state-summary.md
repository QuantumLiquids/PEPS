# Heisenberg SR/MinSR State Summary

**Date:** 2026-04-18

## Scope

This note records the current stopping point for the Heisenberg SR/MinSR investigation and defines what should stay in the `PEPS` repository versus what belongs downstream in `HeisenbergVMCPEPS` or only in local storage.

The MinSR result summary is already curated in Obsidian:

- `/Users/wanghaoxin/Documents/Obsidian Vault/SR MinSR Convergence Rate and Spectral Gap.md`

Curated backups of the analysis scripts and summary artifacts were copied into `HeisenbergVMCPEPS` here:

- `../HeisenbergVMCPEPS/run/4x4J2=0D8/20260308_sr_minsr_tuning/`
- `../HeisenbergVMCPEPS/run/8x8J2=0D8/20260308_20260317_sr_minsr_tuning/`
- `../HeisenbergVMCPEPS/run/8x8J2=0D8/20260312_mc10000_focus/`

## Repository Boundary Decision

### Keep in `PEPS`

- Stable reference material that directly supports a future upstream integration test.
- The 4x4 square Heisenberg OBC ED reference under `tests/test_data/ed_reference/`.
- High-level design/state documents describing the intended integration-test workflow.

### Move to `HeisenbergVMCPEPS`

- Experiment orchestration scripts for SR/MinSR sweeps.
- Curated run summaries, selected CSV tables, final plots, and parameter files used to justify future thresholds.
- Any simple-update TPS or checkpoint package required to reproduce the future 4x4 cluster test.

### Keep local-only or delete after migration

- Raw rsync mirrors downloaded from the cluster.
- Large run directories containing chunk logs, repeated checkpoints, intermediate trajectories, and heartbeat state.
- Ad hoc analysis workspaces that are useful during tuning but do not belong in the upstream library repository.

## Current Test Decision

The current 4x4 integration-test draft should **not** be treated as a finished upstream test yet.

Reasons:

- The draft uses placeholder optimizer parameters and placeholder tolerances.
- The test file currently relies on filesystem handoff between separate `TEST_F` cases, so correctness depends on test execution order instead of a self-contained fixture.
- The intended final assertion has changed: the more meaningful target is a 4x4 cluster test where the energy error shows exponential decay and the fitted exponent falls inside a deliberately broad acceptance window.

## Future Restart Point

When this work resumes, the recommended sequence is:

1. Download the simple-update TPS/checkpoint package needed for reproducible 4x4 restarts from the cluster.
2. Curate the SR/MinSR tuning artifacts in `HeisenbergVMCPEPS`.
3. Lock down the final 4x4 test criterion around exponential-decay behavior and fitted exponent range.
4. Upstream only the minimal, stable PEPS-side assets: ED reference, final test parameters, and the final integration test implementation.

## Practical Cleanup Guidance

- `PEPS` should not absorb the large `4x4J2=0OBCD8`, `8x8J2=0OBCD8_*`, or raw run-output directories as a normal commit.
- If those directories need to be preserved, preserve them downstream or outside git as an experiment archive.
- If a later upstream commit is needed, prefer a reduced package: README, selected plots, selected CSV summaries, tuned parameter JSON, and provenance notes, not raw run trees.

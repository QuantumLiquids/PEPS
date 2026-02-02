# History: `VMCPEPSExecutor` → `VMCPEPSOptimizer`

This page is **historical documentation** for the executor → optimizer refactor series.

If you are writing new code today, start from the current user docs entry points:

- Tutorials: `../../tutorials/index.md`
- How-to: `../../howto/index.md`
- Explanation: `../../explanation/index.md`
- Reference: `../index.md`

## Scope and versions (tags + anchor commits)

The refactor series started before the v0.0.1 release tag and continued across multiple follow-up commits.

- Refactor start (first large step): `fd401d0` (Major VMC PEPS refactoring: extract optimizer and create independent executor)
- Release tags in this repository:
  - v0.0.1: `24936cb`
  - v0.0.2: `7f49e27`
  - v0.0.3: `4cf92e5`

## What changed (conceptually)

This was not a single commit; it gradually moved the codebase from a monolithic “executor does everything” design to a more explicit composition:

- **Configuration became a first-class input** (instead of “occupancy arrays + (ly,lx)” scattered across constructors).
- **Parameters were split by responsibility**: optimizer params, Monte Carlo params, and contraction params.
- **High-level APIs were clarified** (explicit state conversion helpers; clearer construction patterns; unified measurement registry interface).

## Commit lists (refactor series)

### Foundational refactor

```
fd401d0 Major VMC PEPS refactoring: Extract optimizer and create independent executor
```

### Follow-ups up to v0.0.1

```
8ffec1e Add test data for heisenberg complex number
f275ac3 Fix several unit test bugs
cc9161b Fix bug
0314f5f fix header messy
b215d15 Make simple update and NDB MC tests work
0c2ca59 Major test suite reorganization and API improvements
829313a Complete test suite reorganization - remove old directories
f7f317b Update according Tensor API udpate
465fcfe feat: Complete test suite reorganization and establish new test architecture
b0bb83d Update test_exact_sum_optimization.cpp
c20766e setup Doc
c375116 feat: add documentation infrastructure and build system improvements
f848d5b Proposal for params refactor and harmless change
24936cb Add .cursor/ to .gitignore to ignore Cursor IDE files across all branches (tag: v0.0.1)
```

### API/optimizer work between v0.0.1 and v0.0.2

```
852b2d4 refactor: Update Optimizer to use new parameter structures with variant dispatch
f5210a6 feat: complete optimizer params refactor implementation
a4765ac feat: Complete legacy VMC PEPS executor cleanup and modernization
a0daad3 refactor: unify Monte Carlo PEPS API parameters
5eadbca Refactor executors constructor to factory function and add Chinese documentation
e0f48cd refactor: complete optimizer params refactor with MPI-aware exact sum evaluator
cb798c3 refactor: inline UpdateTPSByGradient function
49fdc2e chore(headers, docs): cleanup imports; sync guides; add MC updater tutorial
845a98e docs,test: restructure docs; move test_data; update code/tests
dd6ee66 Rename TOP LEVEL APIs
044ee31 refactor: switch MonteCarloEngine (originally MonteCarloPEPSBaseExecutor) from inherit to composition
0c437f1 update docs; rename header; separate declare and impl
da40bc8 docs: migrate tutorials; optimizer: add lr schedulers and SGD exact-sum tests; stop tracking .cursor; ignore Testing/
69d9dd5 optimizer: gradient clipping + tests; migrate to qlten::hp_numeric::kMPIMasterRank; docs: coding standards placeholder
748eb40 update documents
44e6452 add API/tests; prune optimizer; add RFCs; WIP: extract MC energy evaluator
8e0ed20 vmc: use MCEnergyGradEvaluator; remove legacy accumulators/paths; update Doxygen
bf88b9f Update std err estimation in energy evaluator; Real->std::real();
6db53d9 minor update
2e68cf8 Fix Intel icpx compiler crash
85597a6 feat: add MC update instantiation params; improve state conversion API; unify Transverse Field Ising naming; add VMC Chinese tutorial and examples
ccbbb57 2 rfc; remove normalization in optimization; clean up
881debc chore: checkpoint messy edits before redesign
2ec6d76 Registry refactor WIP: metadata overhaul; SquaretJ energy still zero
892ff49 Complete tJ integration test and helper; eliminate the warning for legacy api.
d38abc8 fix bugs
a970c67 Improve DescribeObservables
c644615 More type gymnastics; extend TPS to PBC
2d76900 feat: add TFIM PBC TRG solver, rename OBC solver, and add PBC simple-update test data
7f49e27 docs: release v0.0.2 (tag: v0.0.2)
```

### Changes after v0.0.2 (v0.0.3 line)

```
24de948 Unify psi-consistency warnings; Fix UpdateSingleSite_ Bug; Add Hubbard simple update test
fc7eaa6 feat: Add WaveFunctionSum and configurable configuration rescue
24b178c feat(optimizer): add Adam/AdamW optimizer
2a55994 vmc: add TRG/PBC support and explicit trial API
3dcf6ee feat: add TRG read-only replacement evaluation and Heisenberg PBC test
bf81c49 api: unify OBC/PBC dispatch; rebaseline XXZ structure-factor regression
648bf4c bmps/vmc/tests: update heisenberg baseline; unify file layout; fix PBC code paths
8863be8 fix: use ElementWiseSquaredNorm for Adam/AdaGrad complex correctness
4cf92e5 Release v0.0.3 (tag: v0.0.3)
```

## How to reproduce / inspect (commands)

```
git show fd401d0
git show v0.0.1 v0.0.2 v0.0.3
git log --oneline fd401d0..v0.0.1
git log --oneline v0.0.1..v0.0.2 -- include/qlpeps/api/vmc_api.h include/qlpeps/algorithm/vmc_update include/qlpeps/optimizer
```


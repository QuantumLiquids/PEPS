---
title: Test Suite Modularization & Coverage Roadmap
last_updated: 2026-03-02
status: active
---

# Test Suite Modularization & Coverage Roadmap

## CTest Label Taxonomy

All fast-suite tests carry two orthogonal label dimensions: module and speed.
Multiple `-L` flags AND together: `ctest -L optimizer -L fast` runs only fast optimizer tests.

### Module Labels (13)

| Label | Description |
|-------|-------------|
| `tn-core` | PEPS, TPS, SplitIndexTPS, DuoMatrix, TenMatrix, Configuration, TensorNetwork2D, Arnoldi |
| `contractor` | BMPSContractor, TRGContractor, TRG PBC benchmark |
| `mc-tools` | Statistics, Suwa-Todo, MC configuration updaters |
| `model-solver` | t-J, XXZ, Hubbard model solvers |
| `simple-update` | Boson/fermion simple update, advanced stop, observability |
| `loop-update` | Loop update core + observability |
| `vmc-pipeline` | VMC PEPS optimizer executor, API smoke test |
| `mc-evaluator` | MC energy+gradient evaluator (smoke + full), exact summation evaluator |
| `mc-measure` | MC PEPS measurer, MC updater TRG smoke, MC Ising TRG PBC |
| `mc-engine` | MonteCarloEngine, MCPEPSMeasurer death test |
| `golden` | Boson/fermion MC SR golden (single + MPI), exact optimizer golden, SR-vs-MinSR equivalence |
| `optimizer` | Optimizer core (SR, MinSR, Adam, AdaGrad, SGD, LBFGS), exact-sum tests, LR schedulers, gradient preprocessing, spike detection, recovery policies |
| `utility` | CG solver, observable matrix, CG MPI solver |

Tests may carry multiple module labels where appropriate (e.g., SR-vs-MinSR equivalence carries both `golden` and `optimizer`).

### Speed Labels (2)

| Label | Criteria | Purpose |
|-------|----------|---------|
| `fast` | <5 seconds | Safe for every compile cycle |
| `medium` | 5-120 seconds | Run during focused module development |

Every fast-suite test has an explicit speed label. The existing `slow` label on `BUILD_SLOW_TESTS` targets is separate.

### Developer Workflow

| Scenario | Command | Estimated Time |
|----------|---------|---------------|
| Changed optimizer code (quick) | `ctest -L optimizer -L fast` | ~2 min |
| Changed optimizer code (thorough) | `ctest -L optimizer` | ~8 min |
| Changed model solver | `ctest -L model-solver` | ~1 min |
| Changed TN core | `ctest -L tn-core` | ~3 min |
| Quick smoke after any change | `ctest -L fast` | ~3 min |
| Pre-merge full fast suite | `ctest` | ~26 min |
| Slow integration tests | `ctest -L mpi-integration` | cluster only |

### Implementation

Labels are `set_tests_properties(... PROPERTIES LABELS "module;speed")` calls in `tests/CMakeLists.txt`, placed immediately after each test registration. No changes to test macros, no file moves, fully backward-compatible.

---

## Coverage Improvement Roadmap

### Model Solver Testing

**Layer 1A — Exact-sum energy + gradient golden regression** (implemented)

`test_exact_summation_evaluator.cpp` covers 4 models with both energy and gradient golden values:
- Spinless fermion (3 t2 values)
- Heisenberg OBC
- Transverse Ising
- t-J

Golden pattern: `kPrintGolden = true` → capture → hardcode as `constexpr` → set `false`. Gradient characterized via `WeightedProbeInnerProduct` + `RandomProbeInnerProduct` with fixed seeds.

**Models to add (future):**
- J1-J2 XXZ OBC/PBC
- Heisenberg PBC
- Spinless fermion PBC (TRG path)
- t-J NNN variant

**Layer 1B — ExactSummationMeasurer** (implemented)

`ExactSumMeasurerMPI` free function in `exact_summation_measurer.h`, iterating over
all configurations to produce deterministic measurement results. Golden regression
for `EvaluateObservables()` across 4 OBC models (Spinless Fermion, Heisenberg, TFIM, t-J)
with QuSpin ED benchmarks for physical validation.

### MC Configuration Updater Testing

**Strategy 1 — Conservation & smoke tests** (implemented)

`test_mc_updater_conservation.cpp` verifies on synthetic 2x2 OBC SplitIndexTPS:
- `MCUpdateSquareNNExchangeOBC` preserves state counts (particle number)
- `MCUpdateSquareNNFullSpaceUpdateOBC` keeps amplitude finite

**Strategy 2 — Transition probability ratio verification** (future)

Independently compute `|psi(sigma')|^2 / |psi(sigma)|^2` via full tensor contraction and verify updater's computed ratio matches.

**Strategy 3 — Observable vs. exact on small system** (future)

MC sampling with fixed RNG seed, sufficient statistics, compare energy to exact within `3 * binning_error`.

### PBC/TRG Path Coverage (future)

- PBC exact-summation golden test
- PBC MC evaluator test parallel to OBC `test_mc_energy_grad_evaluator`

---

## Integration Test Redesign (Future)

### Tier 1 — Algorithm pipeline tests (label: `integration-local`)

- 2x2 systems with ED reference energies
- Full SU -> VMC(SR) -> Measure pipeline per algorithm variant
- Verify energy convergence AND physical observable assertions
- Fast enough for local dev (each <2 min with 4 MPI ranks)

### Tier 2 — Model coverage matrix (label: `integration-cluster`)

- One test per model on 4x4+ systems
- Energy convergence to within tolerance of ED/DMRG references
- Requires cluster resources (many MPI ranks, long runtime)
- AI agent-driven: documented via skill/prompt templates for automated parameter tuning

---

## Constraints

- Existing test behavior must not change
- All current passing tests continue to pass
- Tests can only be removed if clearly redundant (requires explicit justification)
- No new external dependencies for test infrastructure
- Golden values use the `kPrintGolden` + probe inner product pattern

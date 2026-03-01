# MinSR (Minimum-Step Stochastic Reconfiguration) Design

## References

- Chen & Heyl, "Empowering deep neural quantum states through efficient optimization," Nature Physics 20, 1476 (2024). [arXiv:2302.01941]
- Wu & Nys, "Real-Time Dynamics in Two Dimensions with Tensor Network States via Time-Dependent Variational Monte Carlo," arXiv:2512.06768v4 (2026).

## Motivation

Traditional SR solves `S * delta_theta = g` where `S = O_bar^dagger O_bar` is `Np x Np`.
The current implementation uses an implicit S-matrix (`SRSMatrix`) with CG iteration,
avoiding explicit construction of S. Each CG iteration costs `O(Ns_local * Np)` per rank.

MinSR reformulates as `T * y = epsilon_bar` where `T = O_bar O_bar^dagger` is `Ns x Ns`,
then `delta_theta = O_bar^dagger y`. When `Np >> Ns`, this system is far smaller and
can be solved via direct eigendecomposition rather than iterative CG.

The two formulations are mathematically equivalent (both yield the pseudo-inverse
solution `delta_theta = O_bar^{+} epsilon_bar`).

## Module layout

| File | Purpose |
|------|---------|
| `optimizer_params.h` | `MinSRParams`, `MinSRSolverMode` enum, `AlgorithmParams` variant entry, builder `WithMinSR()` |
| `optimizer.h` | `MinSRUpdate()` declaration |
| `optimizer_impl.h` | `MinSRUpdate()` four-stage pipeline; `IterativeOptimize` dispatch branch |
| `minsr_tmatrix.h` | `MinSRTMatrix` class: ring exchange, four-term centering, row-block storage |
| `minsr_eigensolve.h` | Path B (replicated LAPACK eigensolve), pseudo-inverse cutoff, `MinSREigenSolveDispatch()` |
| `minsr_scalapack.h` | Path A (ScaLAPACK distributed eigensolve), guarded by `QLPEPS_HAS_SCALAPACK` |
| `cmake/Modules/FindScaLAPACK.cmake` | CMake finder for ScaLAPACK (MKL, AOCL, OpenBLAS backends) |

## Algorithm

### Notation

| Symbol | Meaning |
|--------|---------|
| Ns | Total MC samples across all ranks |
| Ns_local | Samples per rank (`Ns / P`) |
| Np | Number of variational parameters |
| P | Number of MPI ranks |
| O_i | i-th O* sample (a `SplitIndexTPS`) |
| O_mean | Mean O* across all samples (on master only) |
| T | Centered Gram matrix, `Ns x Ns` |
| epsilon_bar | Centered energy vector, length Ns |

### MinSRUpdate four-stage pipeline

**Stage 1 — Build epsilon_bar** (`optimizer_impl.h`):

Each rank computes local entries following the Wirtinger convention:
```
epsilon_bar_i = conj(E_loc,i - <E_loc>) / Ns
```
Then `MPI_Allgather` to form the full vector on all ranks.

**Stage 2 — Construct T via ring exchange** (`MinSRTMatrix::Construct`):

Ring exchange of O* samples computes raw Gram entries, then applies
four-term centering without storing explicit centered samples:

```
raw_ij = <O_i | O_j>                          (SITPS inner product)
m_i = (1/Ns) * sum_j raw_ij                   (row mean, from row sums)
c = (1/Ns) * sum_i m_i                        (grand mean)
T_ij = (raw_ij - m_i - conj(m_j) + c) / Ns   (centered + normalized)
```

The centering terms derive from matrix row averages, so O_mean is **not** needed
during T construction. Each rank holds its `Ns_local` contiguous rows (row-major dense).

The ring uses a pipeline-safe pattern: rank `P-1` sends first in each round to
break the circular blocking-send deadlock.

**Stage 3 — Eigensolve with pseudo-inverse cutoff** (`MinSREigenSolveDispatch`):

Dispatches to Path A or Path B based on `MinSRParams::solver_mode`:

| Mode | Behavior |
|------|----------|
| `kReplicated` | Always Path B |
| `kDistributed` | Always Path A; throws if ScaLAPACK unavailable |
| `kAuto` (default) | Path A if ScaLAPACK available AND `Ns > 5000`; else Path B |

Pseudo-inverse cutoff (Chen & Heyl Eq. 22-23) applied to eigenvalues:
```
cutoff = r_pinv * |lambda_max| + a_pinv

Soft (default):  lambda_plus = lambda^5 / (lambda^6 + cutoff^6)
Hard:            lambda_plus = 1/lambda if |lambda| > cutoff, else 0
```

Then `y = Z * diag(lambda_plus) * Z^H * epsilon_bar`.

**Stage 4 — Back-substitution** (`optimizer_impl.h`):

Distributed computation, no ring exchange needed. O_mean is required on master only:
```
delta_theta = O_bar^dagger * y = sum_i y_i * (O_i - O_mean)
            = sum_i y_i * O_i  -  (sum_i y_i) * O_mean

Each rank:    local_sum = sum_{i in local} y_i * O_i
              local_y_sum = sum_{i in local} y_i
MPI_Reduce -> master
Master:       delta_theta = global_sum - global_y_sum * O_mean
```

## Eigensolve paths

### Path B: Replicated LAPACK (default)

1. `MPI_Allgather` row-blocks → full `T` on all ranks.
2. `LAPACKE_dsyev` (real) / `LAPACKE_zheev` (complex). For complex, explicit
   row-to-column-major transpose before calling LAPACK (row-major H read as
   col-major gives H^T = conj(H) != H).
3. All ranks compute identical `y` — no broadcast needed.
4. Per-rank memory: `O(Ns^2)` for the full replica.

### Path A: Distributed ScaLAPACK

1. Create BLACS contexts: Px1 (source row-block) and near-square 2D grid (target).
2. Local row-to-column-major transpose (required for both real and complex — the
   local `Ns_local x Ns` block is rectangular, so row-major != col-major even
   for a globally symmetric matrix).
3. `pdgemr2d_` / `pzgemr2d_`: redistribute to block-cyclic (NB=64).
4. `pdsyev_` / `pzheev_`: distributed eigendecomposition.
5. Distributed matvec via local block iteration + `MPI_Allreduce` (not pdgemv —
   avoids the complexity of describing epsilon_bar as a block-cyclic vector).
6. Per-rank memory: `O(Ns^2 / P)`.

BLACS context and grid are created per call (RAII via `ScaLAPACKContext`). The
cost of `Cblacs_gridinit` is negligible vs the `O(Ns^3/P)` eigensolve.

## Memory budget

Per-rank peak memory during MinSRUpdate (additional to existing allocations):

| Data | Size | Path |
|------|------|------|
| Ring receive buffer | Ns_local SITPS objects | both |
| T row-block (dense) | `Ns_local * Ns * sizeof(TenElemT)` | both |
| m vector (centering) | `Ns * sizeof(TenElemT)` | both |
| Full T replica | `Ns^2 * sizeof(TenElemT)` | B only |
| ScaLAPACK local block | `~(Ns/sqrt(P))^2 * sizeof(TenElemT)` | A only |
| y vector | `Ns * sizeof(TenElemT)` | both |

Example: Ns=10K, P=56, complex double (16 bytes):
- T row-block: 180 * 10K * 16B = 29 MB
- Path A local: (10K/7)^2 * 16B = 33 MB
- Path B full replica: 10K^2 * 16B = **1.6 GB**

Path B is comfortable for `Ns <= ~5000` (400 MB); above that, Path A is strongly preferred.

## MPI communication summary

Per MinSRUpdate step:

| Phase | Communication | Volume per rank |
|-------|--------------|-----------------|
| Ring exchange | (P-1) x MPI_Sendrecv (SITPS batch) | Ns_local * serialized SITPS |
| Allgather m | 1x MPI_Allgather (Ns scalars) | `Ns * sizeof(TenElemT)` |
| **Path B:** Allgather T | 1x MPI_Allgather (row-blocks) | `Ns_local * Ns * sizeof(TenElemT)` |
| **Path A:** Redistribution | pdgemr2d_ | `~Ns_local * Ns * sizeof(TenElemT)` |
| **Path A:** Eigensolve | ScaLAPACK internal | library-managed |
| **Path A:** Allreduce y | 2x MPI_Allreduce (Ns-vector) | `Ns * sizeof(TenElemT)` |
| Back-substitute | MPI_Reduce (SITPS + scalar) | serialized SITPS |

## Spike detection (S3) adaptation

MinSR has no CG loop, so the "suspiciously few CG iterations" trigger is
inapplicable. The dispatch passes `std::numeric_limits<size_t>::max()` as
the iteration count sentinel, bypassing the CG-specific guard while keeping
the natural gradient norm anomaly check active.

## ScaLAPACK build configuration

ScaLAPACK is **off by default** (`-DQLPEPS_USE_SCALAPACK=OFF`). Path B always works.

```bash
# Enable Path A (requires ScaLAPACK installed)
cmake .. -DQLPEPS_USE_SCALAPACK=ON ...
```

When enabled, CMake searches for the library matching the active BLAS backend:
- **MKL**: `mkl_scalapack_lp64` + `mkl_blacs_intelmpi_lp64` in `$MKLROOT/lib`
- **AOCL**: `scalapack` in `$AOCL_ROOT/lib`
- **OpenBLAS**: `libscalapack` in Homebrew / system paths

The `FindScaLAPACK.cmake` module creates the `ScaLAPACK::ScaLAPACK` imported
target and verifies via `check_function_exists(pdpotrf_)`. When found,
`-DQLPEPS_HAS_SCALAPACK` is defined and `link_libraries(ScaLAPACK::ScaLAPACK)`
propagates to all project targets. Downstream users of this header-only library
must link ScaLAPACK themselves.

## Parameters

```cpp
enum class MinSRSolverMode {
  kAuto,        ///< Path B if Ns <= 5000; Path A if ScaLAPACK available, else Path B
  kReplicated,  ///< Force Path B (LAPACK, all ranks replicate T)
  kDistributed  ///< Force Path A (ScaLAPACK); error if not available
};

struct MinSRParams {
  double r_pinv = 1e-12;       ///< Relative pseudo-inverse cutoff
  double a_pinv = 0.0;         ///< Absolute pseudo-inverse cutoff
  bool soft_cutoff = true;     ///< Soft cutoff (Eq. 23) vs hard
  MinSRSolverMode solver_mode = MinSRSolverMode::kAuto;
};
```

Builder: `OptimizerParamsBuilder::WithMinSR(const MinSRParams& params)`.

## Future work

- **PEPS gauge removal** (Wu & Nys Sec. III.A-B): QR projection to remove null vectors from O before constructing T. Improves conditioning; critical for tVMC (real-time dynamics).
- **Small-o trick** (Wu & Nys Sec. III.C): Exploit `O_{s,alpha} = 0` when `p != s(x)` to reduce memory by factor d (local Hilbert space dimension).
- **Cholesky solver** (`pdpotrf` + `pdpotrs`): `O(Ns^3/6)` vs `O(Ns^3)` for eigendecomposition. Needs diagonal shift for positive definiteness.
- **Communication-computation overlap**: Pipeline ring exchange so round r+1 communication overlaps with round r inner product computation.

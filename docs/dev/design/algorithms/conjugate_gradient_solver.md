# Conjugate Gradient Solver Design Document

## Overview

The conjugate gradient (CG) solver (`include/qlpeps/utility/conjugate_gradient_solver.h`) solves
self-adjoint positive-definite linear systems `A x = b` using the CG method, with both serial and
MPI-parallel variants. It is a generic, concept-constrained C++20 template that works with any
user-provided matrix and vector types.

The solver is used in two contexts within the library:

1. **Stochastic Reconfiguration (SR)** — solves `(S + eps I) x = g` where `S` is the sample
   covariance matrix (Quantum Geometric Tensor) and `g` is the energy gradient. This is the
   primary and most demanding use case: `S` has rank ~10^4 from Monte Carlo samples but the
   parameter space is 10^5-10^7, making `S` severely rank-deficient. The diagonal shift makes
   it full-rank but extremely ill-conditioned.

2. **Loop update full-environment truncation** — solves `B x = p` where `B` is a well-conditioned
   positive definite tensor. Uses the serial solver only; no MPI.

## File Layout

| File | Role |
|------|------|
| `include/qlpeps/utility/conjugate_gradient_solver.h` | CG algorithm (serial + MPI) |
| `include/qlpeps/optimizer/optimizer_params.h` | `ConjugateGradientParams`, `StochasticReconfigurationParams` |
| `include/qlpeps/optimizer/stochastic_reconfiguration_smatrix.h` | `SRSMatrix` operator (matrix-vector product for SR) |
| `include/qlpeps/optimizer/optimizer_impl.h` | SR optimizer calling MPI CG |
| `include/qlpeps/two_dim_tn/peps/square_lattice_peps_projection4_impl.h` | Loop update calling serial CG |
| `tests/test_utility/test_conjugate_gradient_solver.cpp` | Serial CG tests (13 cases) |
| `tests/test_utility/test_conjugate_gradient_mpi_solver.cpp` | MPI CG tests (7 cases, 3 ranks) |
| `tests/test_utility/my_vector_matrix.h` | Test-only `MyVector`/`MySquareMatrix` satisfying CG concepts |

## C++20 Concepts

The solver uses four concepts to constrain template arguments at compile time:

**`CGInnerProductType`** — constrains inner product results to `double` or `std::complex<double>`.

**`CGVectorType`** — requires:
- `NormSquare()` returning the squared 2-norm (double)
- `operator*` as inner product between two vectors, returning a `CGInnerProductType`
- `operator+`, `operator-`, `operator+=` for vector arithmetic
- Scalar-vector multiplication with `double`

**`CGMatrixType<MatrixType, VectorType>`** — requires `matrix * vector` returning `VectorType`.

**`CGMPICommunicationVectorType`** — extends `CGVectorType` with:
- `std::default_initializable`
- ADL-found `MPI_Bcast(v, comm)`, `MPI_Send(v, dest, comm, tag)`, `MPI_Recv(v, src, comm, tag)`

These concepts are the CG solver's only interface contract. Any type satisfying them can be used;
the library's primary concrete type is `SplitIndexTPS` (for SR) and tensor types (for loop update).

## API

### Signatures

```cpp
// Serial
template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType>
CGResult<VectorType> ConjugateGradientSolver(
    const MatrixType &matrix_a,
    const VectorType &b,
    const VectorType &x0,
    const ConjugateGradientParams &params);

// MPI
template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType> && CGMPICommunicationVectorType<VectorType>
CGResult<VectorType> ConjugateGradientSolver(
    const MatrixType &matrix_a,
    const VectorType &b,
    const VectorType &x0,
    const ConjugateGradientParams &params,
    const MPI_Comm &comm);
```

### Result Type

```cpp
template<typename VectorType>
struct CGResult {
  VectorType x;                    // Solution (best iterate when not converged)
  double residual_norm;            // Final residual 2-norm ||r||
  size_t iterations;               // Number of CG iterations performed
  CGTerminationReason reason;      // Why CG stopped

  bool converged() const { return reason == CGTerminationReason::kConverged; }
};
```

`converged` is a method, not a field. This is intentional: the old boolean field was replaced by the
`reason` enum, and making it a method ensures all old `cg_result.converged` accesses become compile
errors (field access vs. function call).

### Termination Reasons

```cpp
enum class CGTerminationReason {
  kConverged,           // ||r|| <= tolerance
  kMaxIterations,       // Hit max_iter without convergence
  kIndefiniteMatrix,    // p^T A p <= 0 detected
  kNumericalBreakdown,  // NaN or Inf detected in residual, alpha, or beta
  kStagnated            // Step norm < eps * ||x|| for 3 consecutive iterations
};
```

`to_string(CGTerminationReason)` provides a human-readable name for diagnostics.

### Parameter Structs

```cpp
struct ConjugateGradientParams {
  size_t max_iter = 100;
  double relative_tolerance = 1e-4;
  double absolute_tolerance = 0.0;
  int residual_recompute_interval = 20;
  double orthogonality_threshold = 0.5;
  // No constructors — pure C++20 aggregate.
};

struct StochasticReconfigurationParams {
  ConjugateGradientParams cg_params;
  double diag_shift = 0.001;
  bool normalize_update = false;
  double adaptive_diagonal_shift = 0.0;
  // No constructors — pure aggregate.
};
```

Both are pure C++20 aggregates (no user-declared constructors). This enables designated initializers
and ensures stale positional constructor calls `Type(a, b, c)` are compile errors. Brace positional
init `Type{a, b, c}` still compiles; designated init is enforced by convention.

`diag_shift` lives in `StochasticReconfigurationParams`, not in `ConjugateGradientParams`, because
it is a property of the SR covariance matrix (applied inside `SRSMatrix::operator*`), not a CG
algorithm parameter.

## Algorithm

Standard textbook CG with six robustness extensions. The serial and MPI variants are
algorithmically identical; the MPI variant distributes matrix-vector products.

### Notation

| Symbol | Code / Parameter | Meaning |
|--------|-----------------|---------|
| $A$ | `matrix_a` | Self-adjoint positive-definite operator |
| $b$ | `b` | Right-hand side vector |
| $x_k$ | `x` | Solution at iteration $k$ |
| $r_k = b - A x_k$ | `r` | Residual at iteration $k$ |
| $p_k$ | `p` | Search direction |
| $\alpha_k = \|r_k\|^2 / p_k^\dagger A p_k$ | `alpha` | Step length |
| $\beta_k = \|r_{k+1}\|^2 / \|r_k\|^2$ | `beta` | Conjugacy coefficient |
| $\varepsilon_{\text{rel}}$ | `relative_tolerance` | Relative stopping tolerance |
| $\varepsilon_{\text{abs}}$ | `absolute_tolerance` | Absolute stopping tolerance |
| $\eta$ | `orthogonality_threshold` | Restart threshold |
| $\varepsilon_m$ | `std::numeric_limits<double>::epsilon()` | Machine epsilon (~1e-16) |

### Stopping Criterion

$$\|r_k\| \le \max\!\bigl(\varepsilon_{\text{rel}}\,\|b\|,\;\varepsilon_{\text{abs}}\bigr)$$

Internally compared as squared quantities to avoid `sqrt`:

$$\|r_k\|^2 \le \max\!\bigl(\varepsilon_{\text{rel}}^2\,\|b\|^2,\;\varepsilon_{\text{abs}}^2\bigr)$$

The dual tolerance design handles two edge cases:
- When $b = 0$ or $\|b\|$ is tiny, relative-only tolerance sets $\text{tol} \to 0$, making
  convergence impossible. A nonzero $\varepsilon_{\text{abs}}$ provides a floor.
- Default $\varepsilon_{\text{abs}} = 0$ preserves the legacy relative-only behavior for existing
  code that does not set it.

### Best-Iterate Tracking

Maintains the $x_k$ with smallest $\|r_k\|^2$ across all iterations. All non-converged returns
(`kMaxIterations`, `kStagnated`, `kNumericalBreakdown`, `kIndefiniteMatrix`) use the best iterate,
not the last iterate. This avoids returning a solution that diverged away from a near-optimal point.

Memory cost: one extra `VectorType` copy.

### NaN/Inf Detection

After computing $\|r_{k+1}\|^2$, checks `std::isfinite()`. Also checks `std::isfinite(beta)` to
catch the $0/0 = \text{NaN}$ case when $\|r_k\|^2 = 0$. Returns `kNumericalBreakdown` immediately
on detection.

Design decision: no `kIndefinitePrecond` reason. In unpreconditioned CG,
$\beta_k = \|r_{k+1}\|^2 / \|r_k\|^2$ is a ratio of non-negative values; a sign flip cannot
occur. The NaN from $0/0$ is caught here instead.

### Stagnation Detection

Counts consecutive iterations where the step norm is below machine epsilon relative to the solution:

$$|\alpha_k|^2 \,\|p_k\|^2 < \varepsilon_m^2 \,\|x_k\|^2$$

After 3 such iterations, returns `kStagnated`.

The $\varepsilon_m^2$ threshold (not $\varepsilon_m$) is deliberate: the comparison is on squared
norms, so $\varepsilon_m^2$ corresponds to a relative step size of $\varepsilon_m$ (~1e-16). Using
$\varepsilon_m$ directly would trigger at $\sqrt{\varepsilon_m}$ (~1e-8), which is premature.

No extra `VectorType` allocated; uses $|\alpha_k|^2 \|p_k\|^2$ instead of materializing
$\alpha_k p_k$.

### Orthogonality-Based Restart

CG residuals are theoretically mutually orthogonal. In finite precision they lose orthogonality,
degrading convergence. When loss of orthogonality is detected:

$$|\operatorname{Re}(r_{k-1}^\dagger\, r_k)| > \eta \,\|r_k\|^2$$

the search direction is reset to the steepest descent direction ($p_k = r_k$), discarding
accumulated conjugacy information. This does not terminate the solver; it continues the loop.

Default $\eta = 0.5$ (Sierra/SM convention). Configurable via `ConjugateGradientParams`.

Memory cost: one extra `VectorType` copy (previous residual).

### Periodic Residual Recomputation

Every `residual_recompute_interval` iterations, recomputes $r_k = b - A x_k$ from scratch instead
of the recurrence $r_{k+1} = r_k - \alpha_k A p_k$. This counteracts floating-point drift in the
residual. Set to 0 to disable.

### Indefinite Matrix Detection

Checks $p_k^\dagger A p_k > 0$ each iteration (`pap_is_valid()`). For complex types, also verifies
$|\operatorname{Im}(p_k^\dagger A p_k)| < 10^{-10}$ (Hermitian matrices have real quadratic forms).
Returns `kIndefiniteMatrix` if violated.

## MPI Architecture

The MPI variant uses a master/slave architecture:

```
Master                              Slaves
  |                                   |
  +-- broadcast(start) ------------->  |
  |                                   |
  +-- [CG loop]                       +-- [wait loop]
  |     |                             |     |
  |     +-- broadcast(multiplication) |     +-- receive instruction
  |     +-- broadcast(v)              |     +-- receive v
  |     +-- local: mat*v              |     +-- local: mat*v
  |     +-- gather partial results    |     +-- send partial result
  |     +-- sum partials              |
  |     +-- ... CG iteration ...      |
  |                                   |
  +-- broadcast(finish) ----------->  +-- exit loop
```

### Instruction Protocol

```cpp
enum ConjugateGradientSolverInstruction { start, multiplication, finish };
```

Master broadcasts instructions; slaves loop receiving instructions until `finish`.

**Critical invariant**: every early-return path in `ConjugateGradientSolverMaster` broadcasts
`finish` before returning. Omitting this deadlocks slave ranks blocked in
`SlaveReceiveBroadcastInstruction()`.

### Distributed Matrix-Vector Product

1. Master broadcasts the vector `v` to all ranks (`MPI_Bcast`)
2. Each rank computes local `mat * v` (partial contribution)
3. Slaves send their partial results to master (`MPI_Send`)
4. Master collects via `MPI_Recv` (accepts `MPI_ANY_SOURCE`) and sums all partials

### Result Distribution

Only master gets the real `CGResult`. Slaves return a placeholder `{x0, 0.0, 0, kMaxIterations}`.
Callers that need the solution on all ranks must broadcast it separately. The SR optimizer
(`CalculateNaturalGradient`) broadcasts the termination reason as an `int` and throws
`std::runtime_error` on non-convergence, ensuring all ranks observe the same failure.

## SRSMatrix

The `SRSMatrix` implements the SR covariance matrix-vector product using centered scalar
projection (avoiding catastrophic cancellation):

```
S v = (1/N) Sigma_i O*_i * delta_i + diag_shift * v
```

where `delta_i = (O_i . v) - (O_mean . v)` subtracts the mean at the scalar level, not the
TPS level. This avoids `large - large = small` cancellation that would occur if subtracting
TPS-level means.

MPI contract:
- `Ostar_mean_` is non-null only on master. Master computes `mean_dot_v` and broadcasts the
  scalar to all ranks.
- `diag_shift` is applied only on master (guarded by `Ostar_mean_ != nullptr`).
- In debug mode, verifies `<delta> ~ 0` (single-rank only).

## Integration with the SR Optimizer

In `optimizer_impl.h`, the `CalculateNaturalGradient` method:

1. Creates `SRSMatrix` with `Ostar_samples`, `Ostar_mean`, `mpi_size_`, `comm_`
2. Sets `s_matrix.diag_shift = sr_params.diag_shift`
3. Calls `ConjugateGradientSolver(s_matrix, gradient, init_guess, sr_params.cg_params, comm_)`
4. Broadcasts `CGTerminationReason` (as int) from master to all ranks
5. On non-convergence: throws `std::runtime_error` with reason, iteration count, and residual norm
6. On success: returns `{cg_result.x, cg_result.iterations}`

## Memory Overhead

Per CG solve (MPI master only), beyond the standard CG working set:

| Feature | Extra VectorType copies | Extra scalars |
|---------|------------------------|---------------|
| Orthogonality restart | 1 (`r_prev`) | 0 |
| Best-iterate tracking | 1 (`best_x`) | 1 (`best_residual_norm_sq`) |
| Stagnation detection | 0 | 1 (`stagnation_count`) |
| NaN/Inf detection | 0 | 0 |
| **Total** | **2** | **2** |

For SR with `SplitIndexTPS` vectors, this means 2 extra copies of the variational parameter vector.
For loop update with tensor blocks, the overhead is negligible.

## Test Coverage

### Serial Tests (13 cases)

- Real and complex SPD systems (3x3)
- Relative tolerance scale independence
- Residual recompute interval
- Zero/tiny RHS with relative-only tolerance (does not converge — by design)
- Zero/tiny RHS with absolute tolerance (converges)
- Breakdown detection (singular matrix)
- Non-convergence reported (`max_iter = 1`; checks `converged() == false`, does not assert `kMaxIterations` by reason)
- `kIndefiniteMatrix`: matrix with negative eigenvalue — `EXPECT_EQ` on reason
- `kNumericalBreakdown`: injected Inf after 1st matvec — `EXPECT_EQ` on reason
- `kStagnated`: 10x10 diagonal with 15 orders of magnitude eigenvalue spread — `EXPECT_EQ` on reason
- Orthogonality restart convergence (low threshold)

### MPI Tests (7 cases, 3 ranks)

- Real and complex distributed systems
- Zero/tiny RHS behavior mirroring serial tests

Serial tests assert termination reasons via `EXPECT_EQ` for `kIndefiniteMatrix`,
`kNumericalBreakdown`, and `kStagnated`. `kMaxIterations` and `kConverged` are tested
indirectly via `converged()`. MPI tests check `converged()` only; they do not assert
specific termination reasons.

## Design Decisions

### Why `converged()` is a method, not a field

Replacing the old `bool converged` field with a method `bool converged() const` turns all old field
accesses into compile errors (`cg_result.converged` vs `cg_result.converged()`). This prevents
silent breakage when migrating call sites.

### Why no preconditioner

The diagonal shift in `SRSMatrix` serves as the only regularization. A full preconditioner was
considered unnecessary for the current use cases and would significantly complicate the concept
requirements.

### Why no `pAp` sign-change detection

`pap_is_valid()` already gates on `p^T A p > 0` and returns immediately. A second iteration with
different-sign `pAp` is unreachable. Adding a sign-change check would be dead code.

### Why beta > 1 is not flagged

CG minimizes the A-norm of the error, not the residual 2-norm. The residual can and does increase
between iterations, making `beta > 1` normal. An earlier debug warning for this case was removed.

## Related Files

- `include/qlpeps/optimizer/optimizer.h` — Optimizer class declaration
- `include/qlpeps/optimizer/optimizer_impl.h` — SR optimizer calling MPI CG
- `include/qlpeps/two_dim_tn/peps/square_lattice_peps.h` — `FullEnvironmentTruncationParams` holding CG params
- `include/qlpeps/two_dim_tn/peps/square_lattice_peps_projection4_impl.h` — Loop update calling serial CG
- `docs/plans/2026-02-27-cg-solver-redesign.md` — Implementation plan for the CG redesign

---

*Created: 2026-02-28*

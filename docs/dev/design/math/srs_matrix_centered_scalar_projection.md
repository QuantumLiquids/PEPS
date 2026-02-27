# ADR: Centered Scalar Projection for SRSMatrix

**Date:** 2026-02-27
**Status:** Accepted
**Supersedes:** Original difference-of-moments implementation

## Context

The SR (stochastic reconfiguration) optimizer requires repeated matrix-vector products $Sv$ with the sample covariance matrix

$$S = \langle O^\dagger O \rangle - \langle O^\dagger \rangle \langle O \rangle = \left\langle (O - \bar{O})^\dagger (O - \bar{O}) \right\rangle$$

where $O_i \in \mathbb{C}^P$ are variational log-derivative samples and $\bar{O} = \langle O \rangle$. This product is computed matrix-free (never forming $S$ explicitly) inside a conjugate gradient solver. Three algebraically equivalent formulations exist for this matrix-free product, differing only in floating-point behavior.

### Why numerical stability matters here

$S$ is a sample covariance with $\operatorname{rank}(S) \le N_{\text{samples}} - 1$. In typical PEPS-VMC runs, $N \sim 10^4$ samples and $P \sim 10^5\text{--}10^7$ parameters, so $S$ is massively rank-deficient. For any direction $v$ aligned with a low-variance (or null) mode, the true $Sv$ is tiny compared to the individual moment terms. The formulation determines whether this small result is obtained by subtracting two large quantities (losing digits) or by accumulating already-small quantities.

## Three Formulations

### Option 1: Difference of moments (original implementation)

$$Sv = \langle O^\dagger (O \cdot v) \rangle - \bar{O}^\dagger (\bar{O} \cdot v)$$

Each sample contributes $O_i^\dagger \cdot (O_i \cdot v)$ to a TPS-level accumulation. After averaging, the mean outer product $\bar{O}^\dagger (\bar{O} \cdot v)$ is subtracted. Both terms live at the *moment* scale. When $v$ is a low-variance direction, they are nearly equal TPS objects, and their difference suffers catastrophic cancellation — losing relative precision element-wise across the entire TPS.

This formulation also slightly breaks positive semi-definiteness in finite precision, since the result is a difference of two PSD-like contributions rather than a sum.

### Option 2: Centered scalar projection

$$\delta_i = (O_i \cdot v) - (\bar{O} \cdot v) \qquad \text{(scalar subtraction)}$$

$$Sv = \frac{1}{N} \sum_i O_i^\dagger \cdot \delta_i \qquad \text{(TPS accumulation at fluctuation scale)}$$

The mean subtraction is moved to the *scalar* inner product $O_i \cdot v$. Each TPS contribution $O_i^\dagger \cdot \delta_i$ is already at the fluctuation scale: if $(O_i \cdot v)$ has a large mean component, $\delta_i$ removes it before multiplying into the TPS. The TPS accumulation never involves moment-scale quantities.

The scalar subtraction $(O_i \cdot v) - (\bar{O} \cdot v)$ can itself lose digits when the mean dominates, but this is unavoidable — the covariance genuinely lives in those small fluctuations. The key improvement is that this loss of digits happens in a scalar, not across an entire TPS object.

### Option 3: Fully centered covariance form

$$Sv = \left\langle (O - \bar{O})^\dagger \cdot \big((O - \bar{O}) \cdot v\big) \right\rangle$$

The best formulation for numerical stability in general: both sides of the outer product are centered, the accumulation is manifestly PSD (a sum of rank-1 PSD terms), and no moment-scale quantities ever appear.

## Decision: Option 2

### Why not Option 3

The $O^\ast$ samples in this codebase are **sparse**: each $O_i^\ast$ is a `SplitIndexTPS` where only one physical index (the sample-space index) is populated per site. This uses $\sim 1/p$ of the memory of a dense `SplitIndexTPS`, where $p$ is the physical dimension.

**Explicit centering destroys sparsity.** Computing $O_i - \bar{O}$ produces a dense `SplitIndexTPS` (the mean $\bar{O}$ is dense), increasing both memory and computation by a factor of $\sim p$ per sample. For a spin-1/2 system ($p = 2$) this is modest, but for multi-orbital or bosonic models ($p$ can be large) the cost is significant.

**Avoiding explicit centering.** Option 3 can be rewritten without per-sample TPS subtraction:

$$Sv = \langle O^\dagger \cdot \delta \rangle - \bar{O}^\dagger \cdot \langle \delta \rangle, \qquad \delta_i = (O_i \cdot v) - (\bar{O} \cdot v)$$

This is Option 2 plus a correction term $-\bar{O}^\dagger \cdot \langle \delta \rangle$. In exact arithmetic, $\langle \delta \rangle = \langle O \cdot v \rangle - \bar{O} \cdot v = 0$ by definition of the mean, so the correction vanishes and Options 2 and 3 are identical. **The only difference between the two is whether $\langle \delta \rangle = 0$ holds in finite precision.**

In practice, $\langle \delta \rangle$ is computed from the same samples and the same $\bar{O}$ used to define `mean_dot_v`, so it is zero to machine precision in single-rank mode. In multi-rank mode, the master's $\bar{O} \cdot v$ is broadcast to all ranks, ensuring a consistent reference — any per-rank deviation in $\langle \delta \rangle$ reflects the expected difference between local and global sample means, not a numerical inconsistency.

A debug-mode assertion (`#ifndef NDEBUG`, single-rank only) verifies $|\langle \delta \rangle| < 10^{-10}$, confirming this equivalence empirically.

### Why Option 2 over Option 1

Option 2 eliminates TPS-level catastrophic cancellation. The improvement is most significant when $v$ aligns with low-variance modes — exactly the directions where the CG solver spends most of its iterations in an ill-conditioned SR system.

### Summary

| | Stability | Memory | Compute | Sparsity preserved |
|---|---|---|---|---|
| Option 1 | Worst (TPS-level cancellation) | Baseline | Baseline | Yes |
| Option 2 | Good (scalar-level cancellation) | Baseline | Baseline + 1 broadcast | Yes |
| Option 3 (explicit) | Best | $\sim p \times$ baseline | $\sim p \times$ baseline | No |
| Option 3 (rewritten) | $=$ Option 2 when $\langle \delta \rangle = 0$ | Baseline | Baseline + 1 broadcast | Yes |

Option 2 captures the essential stability gain of Option 3 at no additional memory or compute cost.

## Implementation

`SRSMatrix::operator*` in `stochastic_reconfiguration_smatrix.h`:

1. Master computes $\texttt{mean\_dot\_v} = \bar{O} \cdot v$ and broadcasts the scalar to all ranks via `MPI_Bcast`.
2. Each rank computes $\delta_i = O_i \cdot v - \texttt{mean\_dot\_v}$ per sample and accumulates $O_i^\dagger \cdot \delta_i$.
3. Diagonal shift is applied only on master (gated by `Ostar_mean_ != nullptr`).

The constructor takes an `MPI_Comm` argument. `Ostar_mean_` remains nullable (nullptr on non-master ranks), preserving the existing MPI contract from `MPIMeanTensor`.

## Related Files

- `include/qlpeps/optimizer/stochastic_reconfiguration_smatrix.h` — SRSMatrix class
- `include/qlpeps/optimizer/optimizer_impl.h` — `CalculateNaturalGradient`, sole call site
- `docs/dev/design/math/sr-bosonic-peps.md` — SR mathematical foundations

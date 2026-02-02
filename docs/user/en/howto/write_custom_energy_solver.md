# Write a custom energy solver (How-to)

This page shows how to implement a custom model energy solver using the square-lattice base classes provided by this repository. It targets NN / NNN Hamiltonians on a square lattice (and “stuff you can squeeze into NN/NNN on a square lattice”).

## Choose a base class (square lattice)

Bases live under:

- `include/qlpeps/algorithm/vmc_update/model_solvers/base/`

Pick the simplest base that matches your Hamiltonian:

1. NN-only: `SquareNNModelEnergySolver<YourModel>`
2. NN + NNN: `SquareNNNModelEnergySolver<YourModel>`

Performance note:

- If your model has no NNN terms, do not use the NNN base with “J2 = 0”; it can add extra contraction work.

Applicability note:

- This framework is intended for square-lattice NN/NNN Hamiltonians, but it also works for “exotic” models as long as you can express them within the same locality pattern and your contraction logic is consistent.

## Part 1: NN model (`SquareNNModelEnergySolver`)

### Inherit the base

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"

class MyNNModel : public SquareNNModelEnergySolver<MyNNModel> {
  // ...
};
```

### Required hook: `EvaluateBondEnergy`

You must implement **exactly one** of the two signatures (spin/boson vs fermion). Do not mix them.

#### Spin / boson interface

```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    const TenElemT inv_psi);
```

Key point:

- `inv_psi` is provided by the base (`1 / Psi(S)`) using a consistent contraction path.

#### Fermionic interface

```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi);
```

Key point:

- For fermions, the amplitude sign can depend on index ordering. You typically must recompute `psi = Psi(S)` locally using the same contraction conventions as `psi_ex = Psi(S')`, then use `ratio = conj(psi_ex / psi)`.
- See: `docs/dev/design/math/fermion-sign-in-bmps-contraction.md`.

### Optional hook: on-site diagonal terms

If your model has purely diagonal on-site terms (chemical potential, fields, etc.), implement:

```cpp
double EvaluateTotalOnsiteEnergy(const Configuration &config);
```

Why keep this separate:

- It keeps `EvaluateBondEnergy` clean and uniform when the bond term is uniform.
- It avoids sprinkling on-site special cases into every bond evaluation.

### Math definition (what you are computing)

For a NN bond \(\langle i,j\rangle\) and a Monte Carlo configuration \(S\):

\[
E_{\text{bond}}(S;i,j)=\sum_{\sigma'_i,\sigma'_j}
\langle \sigma'_i\sigma'_j|\hat{H}^{\text{bond}}_{ij}|\sigma_i\sigma_j\rangle
\cdot \frac{\Psi^*(S')}{\Psi^*(S)} ,
\]

where \(S'\) differs from \(S\) only on sites \(i,j\).

Implementation rule of thumb:

- diagonal contribution is a constant based on \((\sigma_i,\sigma_j)\)
- off-diagonal contribution uses “matrix element × conjugated amplitude ratio”

This matches the solver-level local energy convention documented in:

- `../explanation/model_energy_solver_math.md`

### Example (spin): Heisenberg + staggered field

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"
#include "qlpeps/utility/helpers.h" // ComplexConjugate

namespace qlpeps {

class StaggeredFieldHeisenbergModel
    : public SquareNNModelEnergySolver<StaggeredFieldHeisenbergModel> {
 public:
  StaggeredFieldHeisenbergModel(double J, double h) : J_(J), h_(h) {}

  static constexpr bool requires_density_measurement = false;
  static constexpr bool requires_spin_sz_measurement = true;

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      const TenElemT inv_psi) {
    if (config1 == config2) {
      // Diagonal: J * <Sz_i Sz_j> = +J/4 for parallel spins (using 0/1 encoding).
      return TenElemT(0.25 * J_);
    }
    // Off-diagonal: spin flip terms, weighted by conj(psi_ex / psi).
    const TenElemT psi_ex = contractor.ReplaceNNSiteTrace(
        tn, site1, site2, orient,
        split_index_tps_on_site1[config2],
        split_index_tps_on_site2[config1]);
    const TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
    return TenElemT(-0.25 * J_) + ratio * TenElemT(0.5 * J_);
  }

  double EvaluateTotalOnsiteEnergy(const Configuration &config) {
    double e = 0.0;
    for (size_t row = 0; row < config.rows(); ++row) {
      for (size_t col = 0; col < config.cols(); ++col) {
        const double sz = static_cast<double>(config({row, col})) - 0.5; // 0->-0.5, 1->+0.5
        const double sign = ((row + col) % 2 == 0) ? 1.0 : -1.0;
        e += h_ * sign * sz;
      }
    }
    return e;
  }

 private:
  double J_;
  double h_;
};

} // namespace qlpeps
```

### Example (fermion): t–J-style hopping + exchange (pattern)

This is a minimal sketch showing the **critical sign-consistency rule**: compute `psi` locally using the same conventions as `psi_ex`.

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"
#include "qlpeps/utility/helpers.h" // ComplexConjugate

namespace qlpeps {

class SimpleTJModel : public SquareNNModelEnergySolver<SimpleTJModel> {
 public:
  SimpleTJModel(double t, double J) : t_(t), J_(J) {}

  static constexpr bool requires_density_measurement = true;
  static constexpr bool requires_spin_sz_measurement = true;

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi) {
    if (config1 == config2) {
      psi.reset(); // no need to compute amplitude for diagonal-only contribution
      return TenElemT(0);
    }

    // Critical: recompute Psi(S) locally using the same contraction path/order.
    psi = contractor.Trace(tn, site1, site2, orient);
    const TenElemT psi_ex = contractor.ReplaceNNSiteTrace(
        tn, site1, site2, orient,
        split_index_tps_on_site1[config2],
        split_index_tps_on_site2[config1]);
    const TenElemT ratio = ComplexConjugate(psi_ex / psi.value());

    // Your model logic chooses the matrix element (hopping vs exchange vs ...)
    return -t_ * ratio;
  }

  double EvaluateTotalOnsiteEnergy(const Configuration &) { return 0.0; }

 private:
  double t_;
  double J_;
};

} // namespace qlpeps
```

## Part 2: NNN model (`SquareNNNModelEnergySolver`)

### Inherit the base

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_energy_solver.h"

class MyNNNModel : public SquareNNNModelEnergySolver<MyNNNModel> {
  // ...
};
```

### Additional required hook: `EvaluateNNNEnergy`

Besides all NN hooks, you must implement NNN evaluation:

#### Spin / boson interface

```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateNNNEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    const TenElemT inv_psi);
```

#### Fermionic interface

```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateNNNEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi);
```

New parameter:

- `diagonal_dir`: diagonal direction (`LEFTUP_TO_RIGHTDOWN` or `LEFTDOWN_TO_RIGHTUP`).

### Fermionic NNN hopping: easy-to-miss sign issues

NNN terms often require you to manipulate local physical-leg ordering explicitly. A robust approach for a 2×2 plaquette (sites 1–4):

1. Contract the local environment down to the plaquette physical legs.
2. Reorder physical legs so the two acted-on sites are adjacent, and the ordering matches how you define `psi = Psi(S)`.
3. Extract `psi` for \(S\) and `psi_ex` for \(S'\) using the same external-leg ordering.
4. Use `ratio = conj(psi_ex / psi)`.

NN terms are often protected by helper APIs (`Trace` / `ReplaceNNSiteTrace`) that already enforce ordering consistency; NNN terms usually are not.

Mathematically, the NNN evaluator has the same structure as NN, but on diagonal bonds \(\langle\!\langle i,j\rangle\!\rangle\):

\[
E_{\text{off}}^{\text{(NNN)}}(S;i,j)
= \sum_{\sigma'_i,\sigma'_j}
H_{\text{off}}^{\text{(NNN)}}\big((\sigma_i,\sigma_j)\to(\sigma'_i,\sigma'_j)\big)\,
\frac{\Psi^*(S')}{\Psi^*(S)}.
\]

Implementation rule remains:

- spin/boson: `ratio = ComplexConjugate(psi_ex * inv_psi)`
- fermion: compute `psi` locally and use `ratio = ComplexConjugate(psi_ex / psi)`

## Related

- Math and conventions: `../explanation/model_energy_solver_math.md`
- Built-in models (for reference implementations): `../reference/model_observables_registry.md`

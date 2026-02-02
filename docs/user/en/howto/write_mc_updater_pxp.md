# Write a custom updater (PXP) — basics

This page shows how to implement a custom Monte Carlo updater for a **hard-constrained model**, using the square-lattice PXP model as a concrete example.

## Backend note (OBC/BMPS vs PBC/TRG)

This example is written for the square-lattice NN update framework and is most commonly used with **OBC + BMPS contraction**.

If you run **PBC + TRG**, make sure:

- your `SplitIndexTPS` has periodic boundary condition, and
- your `PEPSParams` carries TRG truncation params,

otherwise the public wrappers (`VmcOptimize` / `MonteCarloMeasure`) will throw.

## Goal

We want a sweep updater that:

- projects out illegal local configurations (“two adjacent excitations”), and
- still samples correctly (preserves the target distribution via balance / a valid selection kernel).

PXP constraint (binary local Hilbert space):

- excitation is `1`
- constraint: for any nearest-neighbor pair \((i,j)\), the local state \((1,1)\) is forbidden.

## Two equivalent ways to enforce the constraint

- Option A (recommended): remove forbidden candidates at candidate-generation time, while keeping the candidate ordering fixed.
- Option B: keep a fixed candidate list, but assign weight 0 to forbidden candidates.

The core requirement is the same: the candidate set and its **ordering** must be fixed and must not depend on the current state when using non-detailed-balance multi-candidate selection.

## Tooling: non-detailed-balance multi-candidate selection

This repo provides a helper for non-detailed-balance selection:

- `SuwaTodoStateUpdate(init_state, weights, rng)`
  - header: `qlpeps/vmc_basic/monte_carlo_tools/suwa_todo_update.h`

Important constraint:

- the candidate ordering must be fixed throughout the simulation; do not reorder based on `init_state`.

## Where to integrate (square lattice NN updater CRTP)

For the square-lattice NN update framework, implement the local hook used by the CRTP base:

```cpp
template<typename TenElemT, typename QNT>
bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1,
                              const SiteIdx &site2,
                              BondOrientation bond_dir,
                              const SplitIndexTPS<TenElemT, QNT> &sitps,
                              TPSWaveFunctionComponent<TenElemT, QNT> &tps_component);
```

This hook is called repeatedly by `MCUpdateSquareNNUpdateBaseOBC<Derived>` during the horizontal-then-vertical scan.

Note:

- For OBC/BMPS, use `MCUpdateSquareNNUpdateBaseOBC<Derived>`.
- For PBC/TRG, there is a corresponding base `MCUpdateSquareNNUpdateBasePBC<Derived>`, which follows a different sweep style (random-bond selection).

Base class location:

- `include/qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h`

## Practical checklist

- Fix a candidate ordering (e.g. lexicographic over `(c1,c2)`), and keep it unchanged.
- For forbidden candidates, either drop them or set their weight to 0 (weight=0 is often easier to audit).
- After selecting a new local state, perform an atomic update via `tps_component.UpdateLocal(...)`.

## Complete example: `MCUpdateSquareNNFullSpacePXP`

This example extends “full-space NN update” with PXP projection, then uses non-DB selection.

```cpp
#include "qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h"
#include "qlpeps/vmc_basic/monte_carlo_tools/suwa_todo_update.h"

namespace qlpeps {

class MCUpdateSquareNNFullSpacePXP : public MCUpdateSquareNNUpdateBaseOBC<MCUpdateSquareNNFullSpacePXP> {
 public:
  template<typename TenElemT, typename QNT>
  bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                TPSWaveFunctionComponent<TenElemT, QNT> &tps_component) {
    auto &tn = tps_component.tn;
    const size_t dim = sitps.PhysicalDim();

    const size_t init_state = tps_component.config(site1) * dim + tps_component.config(site2);
    std::vector<TenElemT> alternative_psi(dim * dim);
    alternative_psi[init_state] = tps_component.amplitude;

    // Fixed candidate ordering: (c1,c2) lexicographic.
    for (size_t c1 = 0; c1 < dim; ++c1) {
      for (size_t c2 = 0; c2 < dim; ++c2) {
        const size_t id = c1 * dim + c2;
        if (id == init_state) continue;
        const bool forbidden = (c1 == 1 && c2 == 1);
        if (forbidden) {
          alternative_psi[id] = TenElemT(0);
          continue;
        }
        alternative_psi[id] = tn.ReplaceNNSiteTrace(site1, site2, bond_dir,
                                                    sitps(site1)[c1],
                                                    sitps(site2)[c2]);
      }
    }

    // Weights for selection: |psi_new / psi_old|^2 (with forbidden -> 0).
    const TenElemT &psi_old = tps_component.amplitude;
    std::vector<double> weights(dim * dim, 0.0);
    for (size_t i = 0; i < alternative_psi.size(); ++i) {
      if (i == init_state) { weights[i] = 1.0; continue; }
      const double r = std::abs(alternative_psi[i] / psi_old);
      weights[i] = r * r;
    }

    const size_t final_state = SuwaTodoStateUpdate(init_state, weights, random_engine_);
    if (final_state == init_state) return false;

    tps_component.UpdateLocal(sitps, alternative_psi[final_state],
                              std::make_pair(site1, final_state / dim),
                              std::make_pair(site2, final_state % dim));
    return true;
  }
};

} // namespace qlpeps
```

## Using your updater in an executor

```cpp
using Updater = qlpeps::MCUpdateSquareNNFullSpacePXP;
auto result = VmcOptimize(params, sitps, MPI_COMM_WORLD, solver, Updater{});
```

## Related

- Updater selection: `choose_mc_updater.md`

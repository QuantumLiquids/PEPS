# Built-in Model Observable Support

This guide summarizes which registry keys each Monte Carlo measurement solver currently emits.
It also highlights gaps against the former "one-point/two-point" pipeline so that we can track
regressions while the registry API stabilises.

> **Note**
> Every key listed below is the authoritative interface for downstream tooling.
> If a solver returns additional vectors without declaring them via `DescribeObservables()`,
> CSV dumps and visualisers will treat them as anonymous flat arrays. Please extend the metadata
> instead of relying on implicit knowledge.

## Summary Table

| Model class | Registry keys today | Shape / index notes | Legacy coverage gaps |
|-------------|---------------------|----------------------|-----------------------|
| `TransverseFieldIsingSquare` | `energy`, `spin_z`, `sigma_x`, `SzSz_row` | `energy`: scalar; site observables `{Ly,Lx}` with `{y,x}`; `SzSz_row` flat | Gap resolved—metadata now matches returned keys. |
| `SquareSpinOneHalfXXZModelOBC` | `energy`, `spin_z`, `SzSz_all2all`, `bond_energy_h`, `bond_energy_v` | Site fields `{Ly,Lx}` (`{y,x}`); bond energies labelled `{bond_id}`; `SzSz_all2all` packed upper tri | None – legacy two-point correlators mapped directly to `SzSz_all2all`. |
| `SquareSpinOneHalfJ1J2XXZModelOBC` | `energy`, `spin_z`, `bond_energy_h`, `bond_energy_v`, `bond_energy_dr`, `bond_energy_ur` | Site fields `{Ly,Lx}`; bond energies `{bond_id}` | Matches legacy outputs for NN/NNN bond energies. |
| `SquareHubbardModel` | `energy`, `spin_z`, `charge`, `double_occupancy`, `bond_energy_h`, `bond_energy_v` | Site fields `{Ly,Lx}`; bond energies `{bond_id}` | Gap resolved—`double_occupancy` now explicit. |
| `SquareSpinlessFermion` | `energy`, `charge`, `bond_energy_h`, `bond_energy_v`, `bond_energy_dr`, `bond_energy_ur` | Charge per site `{Ly,Lx}`; bond energies `{bond_id}` | Legacy code exposed fermionic kinetic energy split into hopping directions; regrouped here as bond energies. |
| `SquaretJNNModel`, `SquaretJNNNModel`, `SquaretJVModel` | `energy`, `spin_z`, `charge`, `SC_bond_singlet_h`, `SC_bond_singlet_v`, `bond_energy_h`, `bond_energy_v`, (`bond_energy_dr/ur` when NNN enabled) | Site fields `{Ly,Lx}`; bond energies `{bond_id}`; SC keys `{Ly,Lx}` | Legacy stored singlet correlators under two-point bins; mapping preserved with explicit keys. |
| `SpinOneHalfTriHeisenbergSqrPEPS` | `energy`, `spin_z`, `bond_energy_h`, `bond_energy_v`, `bond_energy_ur`, `SzSz_row`, `SmSp_row`, `SpSm_row`, `SzSz_all2all` | Triangular bonds flattened; row correlators length `Lx/2`. | Same channels as legacy, but now keyed explicitly. |
| `SpinOneHalfTriJ1J2HeisenbergSqrPEPS` | Same as triangular model above | Same as above | Same as above |

Notes:

- The generic base `SquareNNNModelMeasurementSolver` still emits `energy`, `bond_energy_*`, plus
  `spin_z` and/or `charge` depending on the static capability flags of the concrete model.
- Any solver that needs additional channels should advertise them via `DescribeObservables()`;
  otherwise the executor cannot attach shape metadata and downstream tooling falls back to treating
  the data as 1D arrays.
- `psi_list` remains an internal working buffer. Global summaries (`psi_mean`, `psi_rel_err`) are
  written by the executor and intentionally absent from per-model metadata.

## Differences relative to main branch

The refactor removed the monolithic `Result` struct (`energy`, `one_point_functions`,
`two_point_functions`, …). When comparing with `main` today we observe:

1. Energy and bond energies are still present under `energy`, `bond_energy_h/v/(dr/ur)`.
2. Site-local observables (`spin_z`, `charge`) survive with explicit keys.
3. Some specialised correlators are still computed but lack metadata (e.g. `sigma_x`, `SzSz_row`
   in the transverse-field Ising solver). These keys were previously exported to CSV; we should
   add them back to `DescribeObservables()` or adjust dumping code accordingly.
4. Legacy "psi consistency" fields are no longer part of the public struct; instead
   `MCPEPSMeasurer::GetEnergyEstimate()` provides a narrow compatibility layer and the registry
   contains everything else.
5. Double-occupancy histograms (Hubbard) and raw `psi_list` dumps disappeared by design; list them
   in RFC follow-ups if the data should return.

If additional gaps appear during migration, record them in
`docs/dev/rfc/2025-09-11-observable-registry-and-results-organization.md` so we can track
required follow-up tasks.


# Model energy solver: math and conventions

This page documents the conventions used by model energy solvers in this repository, especially for **complex-valued wavefunctions and gradients**.

## What a “model energy solver” does

In VMC optimization, a solver provides (per Monte Carlo sample):

- the **local energy** \(E_{\mathrm{loc}}(S)\), and
- optionally, gradient-related “hole” tensors used by the optimizer (e.g. SR).

The goal is to keep all model-specific Hamiltonian logic inside the solver and keep the executor generic.

In many built-in models, the same class also implements the measurement-solver interface (registry-based observables) via `ModelMeasurementSolver<...>`. This keeps “what the model is” in one place while allowing VMC optimization and measurement to share the same Hamiltonian logic.

## Local energy (complex wavefunction)

The VMC energy is computed as an expectation over configurations \(S\):

\[
E = \langle E_{\mathrm{loc}}(S) \rangle.
\]

The local energy definition used by the code is:

\[
E_{\mathrm{loc}}(S) = \sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)} \langle S'| H | S\rangle .
\]

Important detail:

- The amplitude ratio uses **complex conjugation**. In practice, the solver helpers take responsibility for this convention so call sites do not need special cases.

## Gradients: Wirtinger convention (and the “factor 2” trap)

For complex parameters \(\theta\), the steepest-descent direction is governed by derivatives w.r.t. \(\theta^*\), not \(\theta\).

A common source of confusion is that the “real case derivative” and the “complex Wirtinger derivative” differ by a factor of 2 in simple examples (e.g. \(x^2\) vs \(|z|^2 = z^*z\)). This codebase follows a **single unified convention**: treat real parameters as a special case of complex parameters and do not introduce extra factor-2 branches.

The gradient formula used by the implementation can be written as:

\[
\frac{\partial E}{\partial \theta_i^*} =
\langle E_{\mathrm{loc}}^* \, O_i^* \rangle
- \langle E_{\mathrm{loc}}^* \rangle \langle O_i^* \rangle ,
\]

where \(O_i = \partial \ln \Psi / \partial \theta_i\) are logarithmic derivatives.

In practice, the choice of whether to use \(\langle E_{\mathrm{loc}} \rangle\), \(\langle E_{\mathrm{loc}}^* \rangle\), or \(\langle \Re(E_{\mathrm{loc}}) \rangle\) can affect SR numerical stability; this repository uses the conjugated form consistently.

### Fermionic convention (current implementation)

For fermionic (Z2-graded) tensors, the code uses a per-sample/per-configuration mapping:

\[
R_i^*(S)=\frac{(\partial_{\theta_i^*}\Psi^*(S))\,\Psi(S)}{|\Psi(S)|^2},
\qquad
O_i^*(S)=\Pi\!\left(R_i^*(S)\right).
\]

So the flow is:

1. build graded-safe \(R^*\),
2. map once to physical \(O^*=\Pi(R^*)\),
3. accumulate gradients and SR buffers in this physical \(O^*\) representation.

This replaces the old convention of applying `gradient.ActFermionPOps()` only at the end.

## Solver interface (what you implement)

The CRTP base is:

- `include/qlpeps/algorithm/vmc_update/model_energy_solver.h`

The executor calls:

```cpp
template<typename TenElemT, typename QNT, bool calchols, typename WaveFunctionComponentT>
TenElemT CalEnergyAndHoles(const SplitIndexTPS<TenElemT, QNT>* sitps,
                           WaveFunctionComponentT* tps_sample,
                           TensorNetwork2D<TenElemT, QNT>& hole_res);
```

Your concrete solver implements:

```cpp
template<typename TenElemT, typename QNT, bool calchols, typename WaveFunctionComponentT>
TenElemT CalEnergyAndHolesImpl(const SplitIndexTPS<TenElemT, QNT>* sitps,
                               WaveFunctionComponentT* tps_sample,
                               TensorNetwork2D<TenElemT, QNT>& hole_res,
                               std::vector<TenElemT>& psi_list);
```

Meaning of the arguments:

- `sitps`: the wavefunction state \(|\Psi\rangle\) in `SplitIndexTPS` form.
- `tps_sample`: the current Monte Carlo sample (configuration + cached TN + amplitude + contractor).
- `hole_res`: output “holes” (valid only if `calchols == true`).
- `psi_list`: a list of amplitudes collected while contracting the TN at different positions.

### `calchols` (performance-critical)

- If `calchols == false`, the solver should skip all gradient/hole work and compute energy only.
- Because this is a template parameter, the compiler can optimize away the dead branch.

## Psi-consistency warnings (BMPS/TRG truncation effects)

In 2D tensor-network contraction with truncation, the computed wavefunction amplitude can depend slightly on contraction path/window position. The solver records a `psi_list` for a sample; the framework computes a summary \((\psi_{\text{mean}}, \psi_{\text{rel\_err}})\) and can emit warnings when the mismatch exceeds a threshold.

Interpretation:

- large `psi_rel_err` often indicates contraction truncation is too aggressive (increase BMPS/TRG accuracy settings).

## Built-in model solvers (energy + measurement)

Built-in solvers live in:

- `include/qlpeps/algorithm/vmc_update/model_solvers/`

Many model classes implement both energy and measurement interfaces.

| Model | OBC header | PBC header |
|---|---|---|
| TFIM (square) | `transverse_field_ising_square_obc.h` (`TransverseFieldIsingSquareOBC`) | `transverse_field_ising_square_pbc.h` (`TransverseFieldIsingSquarePBC`) |
| XXZ (square) | `square_spin_onehalf_xxz_obc.h` (`SquareSpinOneHalfXXZModelOBC`) | — |
| J1–J2 XXZ (square) | `square_spin_onehalf_j1j2_xxz_obc.h` (`SquareSpinOneHalfJ1J2XXZModelOBC`) | `square_spin_onehalf_j1j2_xxz_pbc.h` |
| Hubbard (square) | `square_hubbard_model.h` | — |
| t–J (square) | `square_tJ_model.h` | — |
| Spinless fermion (square) | `square_spinless_fermion.h` | — |
| Triangle Heisenberg (mapped to square PEPS) | `spin_onehalf_triangle_heisenberg_sqrpeps.h` | — |
| Triangle J1–J2 Heisenberg (mapped to square PEPS) | `spin_onehalf_triangle_heisenbergJ1J2_sqrpeps.h` | — |
| Heisenberg (square, PBC) | — | `heisenberg_square_pbc.h` |

### Built-in solver details (constructors and Hamiltonians)

The table above is the “where to include”; this section records the most common constructor parameters.

#### TFIM (square)

- OBC: `TransverseFieldIsingSquareOBC(double h)` in `transverse_field_ising_square_obc.h`
- PBC: `TransverseFieldIsingSquarePBC(double h)` in `transverse_field_ising_square_pbc.h`

\[
H = - \sum_{\langle i,j \rangle} \sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i
\]

Parameter meaning:

- `h`: transverse field strength (controls quantum fluctuations).

#### Square spin-1/2 XXZ (OBC)

- `SquareSpinOneHalfXXZModelOBC()` (default Heisenberg, \(J_z=J_{xy}=1\))
- `SquareSpinOneHalfXXZModelOBC(double jz, double jxy, double pinning00)`

\[
H = \sum_{\langle i,j \rangle}\left(J_z S^z_i S^z_j + J_{xy}(S^x_i S^x_j + S^y_i S^y_j)\right) - h_{00} S^z_{00}
\]

#### Square spin-1/2 J1–J2 XXZ (OBC)

- `SquareSpinOneHalfJ1J2XXZModelOBC(double j2)` (J1 fixed to 1 in the implementation mixin)
- `SquareSpinOneHalfJ1J2XXZModelOBC(double jz, double jxy, double jz2, double jxy2, double pinning_field00)`

\[
H = \sum_{\langle i,j \rangle}\left(J_{z1} S^z_i S^z_j + J_{xy1}(S^x_i S^x_j + S^y_i S^y_j)\right)
+ \sum_{\langle\!\langle i,j \rangle\!\rangle}\left(J_{z2} S^z_i S^z_j + J_{xy2}(S^x_i S^x_j + S^y_i S^y_j)\right)
\]

Note:

- When `j2 = 0`, this model is mathematically equivalent to the NN XXZ model, but the implementation and cost profile can differ.

#### Triangle Heisenberg / J1–J2 (mapped to square PEPS)

- `SpinOneHalfTriHeisenbergSqrPEPS()` in `spin_onehalf_triangle_heisenberg_sqrpeps.h`
- `SpinOneHalfTriJ1J2HeisenbergSqrPEPS(double j2)` in `spin_onehalf_triangle_heisenbergJ1J2_sqrpeps.h`

#### t–J family (square, OBC)

In `square_tJ_model.h`:

- `SquaretJNNModel(double t, double J, double mu)`
- `SquaretJNNNModel(double t, double t2, double J, double mu)`
- `SquaretJVModel(double t, double t2, double J, double V, double mu)`

Hamiltonian summary (schematic):

- hopping + spin exchange + chemical potential, with optional NNN hopping and NN density interaction.

More explicit forms:

- NN t–J:

\[
H = -t\sum_{\langle i,j\rangle,\sigma} (c^\dagger_{i,\sigma} c_{j,\sigma} + h.c.)
+ J \sum_{\langle i,j\rangle} \left(\vec{S}_i \cdot \vec{S}_j - \frac{1}{4} n_i n_j\right)
- \mu \sum_{i,\sigma} n_{i,\sigma}
\]

- NNN t–J (adds NNN hopping term with \(t_2\)):

\[
H_{\text{NNN hop}} = -t_2\sum_{\langle\!\langle i,j\rangle\!\rangle,\sigma} (c^\dagger_{i,\sigma} c_{j,\sigma} + h.c.)
\]

- t–J–V (adds NN density interaction \(V\sum_{\langle i,j\rangle} n_i n_j\)). A useful special point is \(V=J/4\), where the \(V\) term cancels the \(-\frac{1}{4}n_in_j\) piece in the exchange.

Parameter meaning:

- `t`: NN hopping, `t2`: NNN hopping (if enabled).
- `J`: exchange coupling.
- `V`: NN density interaction (t–J–V only).
- `mu`: chemical potential.

#### Hubbard (square, OBC)

- `SquareHubbardModel(double t, double U, double mu)` in `square_hubbard_model.h`

\[
H = -t \sum_{\langle i,j \rangle,\sigma}(c^\dagger_{i\sigma}c_{j\sigma} + h.c.)
+ U \sum_i n_{i\uparrow}n_{i\downarrow} - \mu\sum_{i,\sigma} n_{i\sigma}
\]

Parameter meaning:

- `t`: NN hopping strength.
- `U`: on-site repulsion (large \(U/t\) corresponds to the Mott regime).
- `mu`: chemical potential (controls filling).

#### Spinless fermion (square, OBC)

In `square_spinless_fermion.h`:

- `SquareSpinlessFermion(double t, double V)` (NN hopping)
- `SquareSpinlessFermion(double t, double t2, double V)` (NN + NNN hopping)

\[
H = -t \sum_{\langle i,j \rangle} (c^\dagger_i c_j + h.c.)
 -t_2 \sum_{\langle\!\langle i,j \rangle\!\rangle} (c^\dagger_i c_j + h.c.)
 +V \sum_{\langle i,j \rangle} n_i n_j
\]

Parameter meaning:

- `t`: NN hopping.
- `t2`: NNN hopping (optional).
- `V`: NN density-density interaction.

## Header naming note

Some OBC model headers were renamed to use an `_obc` suffix. In general, prefer the current `_obc` / `_pbc` headers rather than relying on older filenames.

## Developing new solvers

For NN/NNN square-lattice models, start from the base classes and follow the hook contracts:

- `../howto/write_custom_energy_solver.md`

## Related

- Energy and measurement solver overview: `energy_measurement_solver_overview.md`
- Write a custom solver (NN/NNN bases): `../howto/write_custom_energy_solver.md`

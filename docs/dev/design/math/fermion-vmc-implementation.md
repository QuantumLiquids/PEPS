---
title: Fermionic VMC/SR Implementation Conventions
last_updated: 2026-02-11
---

## 1. Scope

This note is the implementation-facing companion to:

1. `docs/dev/design/math/fermion-vmc-math.md` (math definition),
2. `docs/dev/design/math/exact-summation.md` (exact-summation derivation).

It records how the current code realizes those definitions.

## 2. Current Convention (One-Sentence Version)

For fermions, build graded-safe pre-parity \(R^*\), convert once to physical
\(O^*=\Pi(R^*)\), and use this physical \(O^*\) representation consistently in:

1. MC gradient accumulation,
2. exact-summation gradient accumulation,
3. SR buffers and SR linear solve.

## 3. Gradient and SR Objects

Per configuration/sample \(S\):

$$
R_i^*(S)
:=
\frac{\mathcal{C}\left[\Psi(S)\,\partial_{\theta_i^*}\Psi^\dagger(S)\right]}
     {\mathcal{C}\left[\Psi^\dagger(S)\,\Psi(S)\right]},
\qquad
O_i^*(S):=\Pi\!\left(R_i^*(S)\right).
$$

With bra local-energy convention:

$$
g_i=\left\langle E_{\mathrm{loc}}^*(S)\,O_i^*(S)\right\rangle
-E^*\left\langle O_i^*(S)\right\rangle.
$$

SR solves

$$
(S+\lambda I)\,x=g,\qquad
S_{ij}=\langle O_i^* O_j\rangle-\langle O_i^*\rangle\langle O_j\rangle.
$$

Because \(g\), `Ostar_samples`, and `Ostar_mean` are already in physical \(O^*\),
SR no longer needs extra fermion-parity pre/post transforms around CG.

## 4. Code Mapping

1. Pre-parity helper \(R^*\):
   `include/qlpeps/utility/helpers.h` (`CalGTenForFermionicTensors`)
2. MC path \(R^*\to O^*\) per sample:
   `include/qlpeps/algorithm/vmc_update/mc_energy_grad_evaluator.h`
3. Exact-summation path \(R^*\to O^*\) per configuration increment:
   `include/qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h`
4. SR matrix consumes physical \(O^*\):
   `include/qlpeps/optimizer/stochastic_reconfiguration_smatrix.h`
5. Natural gradient solve without extra parity wrapping:
   `include/qlpeps/optimizer/optimizer_impl.h`

## 5. Performance Note: Per-Sample \(\Pi\) Cost

Applying \(\Pi\) (`ActFermionPOps()`) once per sample/configuration increment adds extra local tensor work, but in practice this cost is typically small compared with dominant tensor-network contractions (energy + holes).

The important trade-off is:

1. old style (Before Feb 11st, 2026): parity transforms could be revisited around SR solve logic,
2. current style: parity is paid once when constructing \(O^*\), then SR iterations reuse already-physical buffers.

For typical runs, this improves behavior traceability and keeps overhead negligible relative to contraction cost.

## 6. Regression Guards

1. Exact evaluator baseline:
   `tests/test_algorithm/test_exact_summation_evaluator.cpp`
2. Exact + optimizer integration golden:
   `tests/test_optimizer/test_fermion_exact_optimizer_integration_golden.cpp`
3. MC + SR golden:
   `tests/test_algorithm/test_fermion_mc_sr_golden.cpp`

The two golden tests lock energy plus multiple gradient/natural-gradient signatures.

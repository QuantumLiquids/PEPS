/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-06-16
*
* Description: QuantumLiquids/PEPS project.
*/

#ifndef QLPEPS_UTILITY_HELPERS_H
#define QLPEPS_UTILITY_HELPERS_H

#include <complex>
#include "qlten/qlten.h"

namespace qlpeps {

// Real functions removed - use std::real directly

double ComplexConjugate(const double &x) { return x; }
std::complex<double> ComplexConjugate(const std::complex<double> &z) { return std::conj(z); }

/**
 * @brief Build the per-sample fermionic pre-parity ratio tensor \f$R^*(S)\f$ for one site.
 *
 * Math definition (see `docs/dev/design/math/fermion-vmc-math.md`)
 * - For complex parameters we use the Wirtinger convention and define
 *   \f[
 *     R^*(S) \equiv \frac{(\partial_{\theta^*}\Psi^*(S))\,\Psi(S)}{|\Psi(S)|^2}.
 *   \f]
 * - The physical log-derivative is then \f$O^*(S)=\Pi(R^*(S))\f$, where \f$\Pi\f$ is
 *   implemented by `ActFermionPOps()`.
 *
 * Why fermions need a special construction? see `docs/dev/design/math/fermion-vmc-math.md`
 *
 * This helper computes the **graded-safe pre-parity ratio** \f$R^*(S)\f$ above, where
 *   \f$\Psi(S)\f$ is reconstructed locally from the same hole/environment tensor.
 *
 * Code ↔ math variable mapping (anchors)
 * - `hole_ten_dag`:
 *   - In code: `holes(site)` in `mc_energy_grad_evaluator.h`, produced by
 *     `ModelEnergySolver::CalEnergyAndHoles(...)`.
 *   - In square-lattice BMPS solvers it is constructed as
 *     `Dag(contractor.PunchHole(tn, site, ...))` (see `square_nnn_energy_solver.h`).
 *   - Math: \f$\partial_{\theta^*}\Psi^*(S)\f$ (a tensor with the same index structure as the
 *     active projected site tensor component).
 * - `split_index_tps_ten`:
 *   - In code: `engine_.WavefuncComp().tn(site)` (projected site tensor for the current configuration).
 *   - Math: the ket tensor \f$T_i(S)\f$ for this site, such that \f$\Psi(S)\f$ is obtained by contracting
 *     the hole/environment with \f$T_i(S)\f$ (up to 1D parity legs).
 *
 * @note This function does NOT apply `ActFermionPOps()`. Callers choose where to map
 *       \f$R^*\to O^*=\Pi(R^*)\f$ depending on the accumulation path.
 *
 * @param hole_ten_dag  Hole tensor \f$\partial_{\theta^*}\Psi^*(S)\f$ (see mapping above).
 * @param split_index_tps_ten  Projected ket tensor \f$T_i(S)\f$ at this site (not conjugated).
 * @return Tensor encoding \f$R^*(S)\f$ for this site (same indices as `split_index_tps_ten`).
 */
template<typename TenElemT, typename QNT>
qlten::QLTensor<TenElemT, QNT> CalGTenForFermionicTensors(
    const qlten::QLTensor<TenElemT, QNT> &hole_ten_dag,
    const qlten::QLTensor<TenElemT, QNT> &split_index_tps_ten
) {
  auto hole_ten = qlten::Dag(hole_ten_dag);
  qlten::QLTensor<TenElemT, QNT> psi_ten, hole_dag_psi;
  qlten::Contract(&hole_ten, {1, 2, 3, 4}, &split_index_tps_ten, {0, 1, 2, 3}, &psi_ten);
  qlten::Contract(&hole_ten_dag, {0}, &psi_ten, {0}, &hole_dag_psi);
  return hole_dag_psi * (1.0 / std::norm(psi_ten.GetElem({0, 0})));
}

}//qlpeps
#endif //QLPEPS_UTILITY_HELPERS_H

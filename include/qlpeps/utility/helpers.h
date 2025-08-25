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

// Extract real part for double/complex without depending on external Real overloads
inline double Real(const double &x) { return x; }
inline double Real(const std::complex<double> &z) { return std::real(z); }

double ComplexConjugate(const double &x) { return x; }
std::complex<double> ComplexConjugate(const std::complex<double> &z) { return std::conj(z); }

/**
 * @brief Calculate gradient tensor for fermionic systems
 * @param hole_ten_dag  Hole tensor (complex conjugated) from CalEnergyAndHoles function
 * @param split_index_tps_ten  Tensor in split index TPS (not complex conjugated)
 * @return Gradient tensor for fermionic systems
 * 
 * This function handles the special case of fermionic tensors where the standard
 * gradient calculation needs to account for fermionic statistics.
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

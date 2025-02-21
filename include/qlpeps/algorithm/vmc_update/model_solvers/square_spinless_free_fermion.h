/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-18
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for the spinless free-fermion in square lattice
*/
#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_SPINLESS_FREE_FERMION
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_SPINLESS_FREE_FERMION

#include "qlpeps/algorithm/vmc_update/model_solvers/square_nn_fermion_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_nn_fermion_measure_solver.h"
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate

namespace qlpeps {
using namespace qlten;

/**
 * only include the nearest-neighbor hopping
 *  H = -t \sum_{<i,j>} c_i^dag c_j + h.c.
 *  with t = 1
 *
 * 0 for filled, 1 for empty
 */

class SquareSpinlessFreeFermion : public SquareNNFermionModelEnergySolver<SquareSpinlessFreeFermion>,
                                  public SquareNNFermionMeasureSolver<SquareSpinlessFreeFermion> {

 public:
  SquareSpinlessFreeFermion(void) = default;

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi // return value, used for check the accuracy
  );

  // do not consider the chemical potential
  constexpr double EvaluateTotalOnsiteEnergy(const Configuration &config) { return 0; }

  double CalDensityImpl(const size_t config) const {
    return double(1 - config);
  }
};


template<typename TenElemT, typename QNT>
TenElemT SquareSpinlessFreeFermion::EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi
) {
  if (config1 == config2) {
    psi.reset();
    return 0.0;
  } else {// one site empty, the other site filled
    psi = tn.Trace(site1, site2, orient);
    TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                            split_index_tps_on_site1[config2],
                                            split_index_tps_on_site2[config1]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
    return -ratio;
  }
}

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_SPINLESS_FREE_FERMION

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-02-20
*
* Description: QuantumLiquids/PEPS project. Measurement Solver Base for nearest-neighbor models on square lattice
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_SQUARE_NN_MODEL_MEASUREMENT_SOLVERS_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_SQUARE_NN_MODEL_MEASUREMENT_SOLVERS_H

#include "square_nnn_model_measurement_solver.h"

namespace qlpeps {

/**
 * SquareNNModelMeasurementSolver is the base class to define general nearest-neighbor
 * model measurement solver on the square lattices,
 * work for the energy and order parameter measurements.
 * The CRTP technique is used to realize the Polymorphism.
 *
 * To defined the concrete model which inherit from SquareNNModelMeasurementSolver,
 * for boson model the following member function with specific signature must to be defined:
 * template<typename TenElemT, typename QNT>
  [[nodiscard]] TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      const TenElemT inv_psi
  )
  the following member functions can be defined if you want to perform the measurement, or relevant measure will be ignored if not define:
   - CalDensityImpl (work for, like fermion models)
   - CalSpinSz (work for, like t-J, Hubbard, and spin models)
   - EvaluateOffDiagOrderInRow (to evaluate the off-diagonal orders in specific row)
   - EvaluateBondSC (optional)
 *
 */

template<class ExplicitlyModel>
using SquareNNModelMeasurementSolver = SquareNNNModelMeasurementSolver<ExplicitlyModel, false>;
} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_SQUARE_NN_MODEL_MEASUREMENT_SOLVERS_H
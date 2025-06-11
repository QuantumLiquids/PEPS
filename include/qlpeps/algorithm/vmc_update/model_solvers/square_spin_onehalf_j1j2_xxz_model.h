/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-20
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 J1-J2 Heisenberg model in square lattice
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_H

#include "square_spin_onehalf_xxz_model.h"          // SquareSpinOneHalfXXZModelMixIn
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_model_measurement_solver.h"

namespace qlpeps {
using namespace qlten;

/**
 * J_1-J_2 XXZ Model
 * For NN and NNN interaction terms。
 * For Jz = 0 reduce to planer limit
 */
class SquareSpinOneHalfJ1J2XXZModel : public SquareNNNModelEnergySolver<SquareSpinOneHalfJ1J2XXZModel>,
                                      public SquareNNNModelMeasurementSolver<SquareSpinOneHalfJ1J2XXZModel>,
                                      public SquareSpinOneHalfXXZModelMixIn {
 public:

  SquareSpinOneHalfJ1J2XXZModel(void) = delete;

  ///< J1-J2 Heisenberg model
  SquareSpinOneHalfJ1J2XXZModel(double j2) :
      SquareSpinOneHalfXXZModelMixIn(1, 1, j2, j2, 0) {}
  ///< Generic construction
  SquareSpinOneHalfJ1J2XXZModel(double jz, double jxy, double jz2, double jxy2, double pinning_field00) :
      SquareSpinOneHalfXXZModelMixIn(jz, jxy, jz2, jxy2, pinning_field00) {}
};//SquareSpinOneHalfJ1J2XXZModel

}//qlpeps


#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_H

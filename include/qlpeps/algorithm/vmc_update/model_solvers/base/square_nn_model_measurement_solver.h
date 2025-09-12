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
 * SquareNNModelMeasurementSolver: NN-only specialization of the NNN base, using the
 * registry-based observable API. See SquareNNNModelMeasurementSolver for required hooks.
 */

template<class ExplicitlyModel>
using SquareNNModelMeasurementSolver = SquareNNNModelMeasurementSolver<ExplicitlyModel, false>;
} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_SQUARE_NN_MODEL_MEASUREMENT_SOLVERS_H
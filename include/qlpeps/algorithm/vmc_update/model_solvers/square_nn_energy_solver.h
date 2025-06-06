/*
 * Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
 * Creation Date: 2025-02-20
 *
 * Description: QuantumLiquids/PEPS project.
 * Model Energy Solver for in square lattice models, with only NN bond energy contributions.
 * Like the XXZ model, the spinless free fermion, the t-J and the Hubbard models.
 * The on-site energy only allow the diagonal terms like the Hubbard repulsion or chemical potential.
 *
 * To use the class, one should inherit the class and define the member function EvaluateBondEnergy
 * and EvaluateTotalOnsiteEnergy.
*/


#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_ENERGY_SOLVER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_ENERGY_SOLVER_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "qlpeps/algorithm/vmc_update/model_solvers/square_nnn_energy_solver.h"

namespace qlpeps {

template<class ExplicitlyModel>
using SquareNNModelEnergySolver = SquareNNNModelEnergySolver<ExplicitlyModel, false>;

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_ENERGY_SOLVER_H

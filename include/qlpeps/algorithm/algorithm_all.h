// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-25
*
* Description: QuantumLiquids/PEPS project. The main header file for algorithms
*/


#ifndef QLPEPS_ALGORITHM_ALGORITHM_ALL_H
#define QLPEPS_ALGORITHM_ALGORITHM_ALL_H

//simple update headers
#include "qlpeps/algorithm/simple_update/simple_update.h"
#include "qlpeps/algorithm/simple_update/simple_update_model_all.h"

//loop update
#include "qlpeps/algorithm/loop_update/loop_update.h"

//Monte-Carlo method
#include "qlpeps/algorithm/vmc_update/vmc_peps.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_measurement.h"
#include "qlpeps/algorithm/vmc_update/wave_function_component_classes/wave_function_component_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"

#endif //QLPEPS_ALGORITHM_ALGORITHM_ALL_H

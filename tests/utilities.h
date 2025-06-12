// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-06-11
*
* Description: QuantumLiquids/PEPS project. Utilities for tests.
*/


#ifndef PEPS_TESTS_UTILITIES_H
#define PEPS_TESTS_UTILITIES_H

#include <string>

std::string GenTPSPath(std::string model_name, size_t Dmax, size_t Lx, size_t Ly) {
#if TEN_ELEM_TYPE == QLTEN_Double
  return "dtps_" + model_name + "_D" + std::to_string(Dmax) + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#elif TEN_ELEM_TYPE == QLTEN_Complex
  return "ztps_" + model_name + "_D" + std::to_string(Dmax)  + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#else
#error "Unexpected TEN_ELEM_TYPE"
#endif
}

#endif //PEPS_TESTS_UTILITIES_H

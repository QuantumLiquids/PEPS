// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-19
*
* Description: QuantumLiquids/PEPS project. Constant declarations.
*/

/**
@file consts.h
@brief Constant declarations.
*/



#ifndef QLPEPS_CONSTS_H
#define QLPEPS_CONSTS_H

#include <string>     // string
#include <vector>     // vector
#include <random>     // uniform distribution
namespace qlpeps {

const std::string kTpsPath = "tps";
const std::string kTpsPathBase = "tps";  // Base name for TPS dumps (will append "final"/"lowest")
const std::string kPepsPath = "peps";
const std::string kRuntimeTempPath = ".temp";
const std::string kEnvFileBaseName = "env";
const std::string kTpsTenBaseName = "tps_ten";
const std::string kBoundaryMpsTenBaseName = "bmps_ten";
const std::string pm_sign = "\u00b1";         // for output standard error

const size_t kMaxTaylorExpansionOrder = 1000;
const int kEnergyOutputPrecision = 8;

std::uniform_real_distribution<double> unit_even_distribution(0, 1);
} /* qlpeps */


#endif //QLPEPS_CONSTS_H

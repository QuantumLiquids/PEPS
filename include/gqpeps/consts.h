// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-19
*
* Description: GraceQ/VMC-PEPS project. Constant declarations.
*/

/**
@file consts.h
@brief Constant declarations.
*/



#ifndef GQPEPS_CONSTS_H
#define GQPEPS_CONSTS_H

#include <string>     // string
#include <vector>     // vector

namespace gqpeps {

const std::string kTpsPath = "tps";
const std::string kPepsPath = "peps";
const std::string kRuntimeTempPath = ".temp";
const std::string kEnvFileBaseName = "env";
const std::string kTpsTenBaseName = "tps_ten";
const std::string kBoundaryMpsTenBaseName = "bmps_ten";

const int kEnergyOutputPrecision = 8;

} /* gqpeps */


#endif //GQPEPS_CONSTS_H

// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: GraceQ/VMC-PEPS project. Implementation for the variational Monte-Carlo PEPS
*/

#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H

namespace gqpeps {
using namespace gqten;

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::Execute(void) {

}
template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::MCSweep_(void) {

}

}//gqpeps

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H

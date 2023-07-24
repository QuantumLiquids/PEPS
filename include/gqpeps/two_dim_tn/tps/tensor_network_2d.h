// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-24
*
* Description: GraceQ/VMC-PEPS project. The 2-dimensional tensor network class.
*/

#ifndef VMC_PEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H
#define VMC_PEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/framework/ten_matrix.h"

namespace gqpeps {
using namespace gqten;

/**
 *         3
 *         |
 *      0--t--2
 *         |
 *         1
 * @tparam TenElemT
 * @tparam QNT
 */
template<typename TenElemT, typename QNT>
class TensorNetwork2D : public TenMatrix<GQTensor<TenElemT, QNT>> {

};

}//gqpeps

#endif //VMC_PEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H

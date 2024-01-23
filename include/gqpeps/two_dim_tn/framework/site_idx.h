// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-23
*
* Description: GraceQ/VMC-PEPS project. Site Index (row, col)
*/

#ifndef GRACEQ_GQPEPS_TWO_DIM_TN_FRAMEWORK_SITEIDX_H
#define GRACEQ_GQPEPS_TWO_DIM_TN_FRAMEWORK_SITEIDX_H

#include <array>
#include <stddef.h>   //size_t

namespace gqpeps {

///< Site Index
class SiteIdx : public std::array<size_t, 2> {
 public:
  size_t row(void) const { return (*this)[0]; }
  size_t col(void) const { return (*this)[1]; }
};

}//gqpeps
#endif //GRACEQ_GQPEPS_TWO_DIM_TN_FRAMEWORK_SITEIDX_H

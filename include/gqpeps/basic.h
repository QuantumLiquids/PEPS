// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-03
*
* Description: GraceQ/VMC-PEPS project. Basic structures and classes.
*/

#ifndef GQPEPS_BASIC_H
#define GQPEPS_BASIC_H

#include <string>     // string
#include <vector>     // vector

namespace gqpeps {

enum BondDirection {
  HORIZONTAL,
  VERTICAL
};

struct TruncatePara {
  size_t D_min;
  size_t D_max;
  double trunc_err;

  TruncatePara(size_t d_min, size_t d_max, double trunc_error)
      : D_min(d_min), D_max(d_max), trunc_err(trunc_error) {}
};



}

#endif //GQPEPS_BASIC_H

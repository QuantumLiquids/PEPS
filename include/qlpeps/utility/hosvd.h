// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-10
*
* Description: QuantumLiquids/PEPS project. High order SVD
*/


#ifndef GRACEQ_UTILITY_HOSVD_H
#define GRACEQ_UTILITY_HOSVD_H

#include "qlten/qlten.h"

namespace qlpeps {
using namespace qlten;

template<typename TenElemT, typename QNT>
struct HOSVDRes {
  using Tensor = QLTensor<TenElemT, QNT>;
  using DTensor = QLTensor<QLTEN_Double, QNT>;
  std::vector<Tensor> u_tens;
  std::vector<DTensor> lambda_tens; // rank-2 tensors
  Tensor middle_ten;
  std::vector<size_t> actual_D;
  std::vector<double> actual_trunc_err;

  HOSVDRes(const size_t n) : u_tens(n), lambda_tens(n), middle_ten(), actual_D(n), actual_trunc_err(n) {}

  double MaxTruncErr(void) const {
    return *std::max_element(actual_trunc_err.cbegin(), actual_trunc_err.cend());
  }

  size_t MaxD(void) const {
    return *std::max_element(actual_D.cbegin(), actual_D.cend());
  }

  size_t MinD(void) const {
    return *std::min_element(actual_D.cbegin(), actual_D.cend());
  }
};

/**
 *
 * t = u_tens * lambda_tens * middle_ten
 *
 * Reference : Phys. Rev. X 4, 011025, Eq 31.
 * Note that the definition used here is different from that in the above reference
 * The S tensor in the reference is corresponding to lambda_tens * middle_ten
 *
 * Index order :
 *   u_tens : original external legs first, last the new generated legs
 *   lambda_tens : 2-leg, from u_tens side to middle_ten side
 *   middle_ten : firstly the new generated legs, then the original external legs.
 *                the new generated legs in order consistent with u_tens order.
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param t
 * @param ldims
 * @param lqns
 * @param Dmin
 * @param Dmax
 * @param trunc_err
 * @return
 */
template<typename TenElemT, typename QNT>
HOSVDRes<TenElemT, QNT> HOSVD(
    const QLTensor<TenElemT, QNT> &t,
    const std::vector<size_t> &ldims,
    const std::vector<QNT> &lqns,
    const size_t Dmin, const size_t Dmax, const double trunc_err
) {
  size_t n = ldims.size();
  assert(n == lqns.size());
  HOSVDRes<TenElemT, QNT> res(n);
  using Tensor = QLTensor<TenElemT, QNT>;

  Tensor iterative_core_ten = t;
  for (size_t i = 0; i < n; i++) {
    Tensor vt;
    SVD(iterative_core_ten, ldims[i], lqns[i],
        trunc_err, Dmin, Dmax,
        &res.u_tens[i], &res.lambda_tens[i], &vt,
        &res.actual_trunc_err, &res.actual_D);
    iterative_core_ten = Tensor();
    Contract<TenElemT, QNT, false, false>(vt, res.lambda_tens[i], 0, 1, 1, iterative_core_ten);
  }
  //Split out lambdas
  const size_t rank_core = iterative_core_ten.Rank();
  for (int i = n - 1; i >= 0; i--) {
    QLTensor<QLTEN_Double, QNT> inv_lambda = ElementWiseInv(res.lambda_tens[i], trunc_err);
    Tensor tmp;
    Contract<TenElemT, QNT, true, false>(inv_lambda, iterative_core_ten, 1, rank_core - 1, 1, tmp);
    iterative_core_ten = tmp;
  }
  res.middle_ten = iterative_core_ten;
  return res;
}

}//qlpeps
#endif //GRACEQ_UTILITY_HOSVD_H

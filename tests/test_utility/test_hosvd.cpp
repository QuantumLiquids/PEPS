/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-10-04
*
* Description: QuantumLiquids/PEPS project. Unittests for high-order SVD function
*/


#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "qlten/tensor_manipulation/basic_operations.h"     // Dag
#include "qlten/utility/utils_inl.h"
#include "qlten/framework/hp_numeric/lapack.h"
#include "qlten/utility/timer.h"
#include "qlpeps/utility/hosvd.h"                           // test target

using namespace qlten;
using U1QN = special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

struct TestSvd : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal(1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal(2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  U1QN qnm2 = U1QN({QNCard(qn_nm, U1QNVal(-2))});

  size_t d_s = 3;
  QNSctT qnsct0_s = QNSctT(qn0, d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  IndexT idx_in_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DQLTensor dten_2d_s = DQLTensor({idx_in_s, idx_out_s});
  DQLTensor dten_3d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s});
  DQLTensor dten_4d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  ZQLTensor zten_2d_s = ZQLTensor({idx_in_s, idx_out_s});
  ZQLTensor zten_3d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s});
  ZQLTensor zten_4d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});
};

inline size_t IntDot(const size_t &size, const size_t *x, const size_t *y) {
  size_t res = 0;
  for (size_t i = 0; i < size; ++i) { res += x[i] * y[i]; }
  return res;
}

inline double ToDouble(const double d) {
  return d;
}

inline double ToDouble(const QLTEN_Complex z) {
  return z.real();
}

inline DQLTensor SVDTensRestore(
    const std::vector<DQLTensor> u_tens,
    const std::vector<DQLTensor> lambda_tens,
    const std::vector<size_t> ldims) {
  DQLTensor res;


  DQLTensor t_restored_tmp;
  Contract(pu, ps, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, &res);
  return res;
}
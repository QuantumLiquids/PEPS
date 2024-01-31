// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-08
*
* Description: QuantumLiquids/PEPS project. Unittests for TensorNetwork2D
*/

#include "gtest/gtest.h"
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

struct TestMPOMultiplyBMPS : public testing::Test {
  const size_t L = 6;
  IndexT vb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 4)},
                         TenIndexDirType::OUT
  ); //virtual bond dimension = 4
  IndexT vb_in = InverseIndex(vb_out);

  std::vector<DQLTensor *> dtransfer_mpo;
  BMPS<QLTEN_Double, U1QN> dbmps = BMPS<QLTEN_Double, U1QN>(BMPSPOSITION::DOWN, 6);
  std::vector<ZQLTensor *> ztransfer_mpo;
  BMPS<QLTEN_Complex, U1QN> zbmps = BMPS<QLTEN_Complex, U1QN>(BMPSPOSITION::DOWN, 6);

  void SetUp(void) {
    dtransfer_mpo = std::vector<DQLTensor *>(6);
    for (auto &pten : dtransfer_mpo) {
      pten = new DQLTensor({vb_in, vb_out, vb_out, vb_in});
    }

  }
  BMPSTruncatePara trunc_para = BMPSTruncatePara(4, 8, 1e-12, VARIATION2Site);

};
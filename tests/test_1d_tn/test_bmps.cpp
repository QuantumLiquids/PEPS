// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-08
*
* Description: GraceQ/VMC-PEPS project. Unittests for TensorNetwork2D
*/

#include "gtest/gtest.h"
#include "gqpeps/ond_dim_tn/boundary_mps/bmps.h"

using namespace gqten;
using namespace gqpeps;

using gqten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

struct TestMPOMultiplyBMPS : public testing::Test {
  const size_t L = 6;
  IndexT vb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 4)},
                         GQTenIndexDirType::OUT
  ); //virtual bond dimension = 4
  IndexT vb_in = InverseIndex(vb_out);

  std::vector<DGQTensor *> dtransfer_mpo;
  BMPS<GQTEN_Double, U1QN> dbmps = BMPS<GQTEN_Double, U1QN>(BMPSPOSITION::DOWN, 6);
  std::vector<ZGQTensor *> ztransfer_mpo;
  BMPS<GQTEN_Complex, U1QN> zbmps = BMPS<GQTEN_Complex, U1QN>(BMPSPOSITION::DOWN, 6);

  void SetUp(void) {
    dtransfer_mpo = std::vector<DGQTensor *>(6);
    for (auto &pten : dtransfer_mpo) {
      pten = new DGQTensor({vb_in, vb_out, vb_out, vb_in});
    }

  }
  BMPSTruncatePara trunc_para = BMPSTruncatePara(4, 8, 1e-12, VARIATION2Site);

};
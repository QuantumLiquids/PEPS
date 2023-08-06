// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-03
*
* Description: GraceQ/VMC-PEPS project. Unittests for TensorNetwork2D
*/

#include "gqten/gqten.h"
#include "gtest/gtest.h"
#include "gqpeps/two_dim_tn/tps/tps.h"          // TPS
#include "gqpeps/two_dim_tn/tps/tensor_network_2d.h"

using namespace gqten;
using namespace gqpeps;

using gqten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;


struct TestSpin2DTensorNetwork : public testing::Test {

  const size_t Lx = 4; //cols
  const size_t Ly = 4;

  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         GQTenIndexDirType::OUT
  );
//  IndexT pb_out = IndexT({
//                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
//                         GQTenIndexDirType::OUT
//  );

  IndexT pb_in = InverseIndex(pb_out);


  void SetUp(void) {
    TPS<GQTEN_Double, U1QN> tps(Ly, Lx);
    gqten::hp_numeric::SetTensorManipulationThreads(1);
    gqten::hp_numeric::SetTensorTransposeNumThreads(1);
    tps.Load("tps_heisenberg_D4");

  }

};
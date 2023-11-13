// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-03
*
* Description: GraceQ/VMC-PEPS project. Unittests for TensorNetwork2D
*/

#include "gqten/gqten.h"
#include "gtest/gtest.h"
#include "gqpeps/algorithm/vmc_update/tensor_network_2d.h"
#include "gqpeps/two_dim_tn/tps/split_index_tps.h"    //TPS, SplitIndexTPS

using namespace gqten;
using namespace gqpeps;

using gqten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;


struct TestSpin2DTensorNetwork : public testing::Test {

  const size_t Lx = 4;  // cols
  const size_t Ly = 4;  // rows

#ifdef U1SYM
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         GQTenIndexDirType::OUT
  );
#else
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         GQTenIndexDirType::OUT
  );
#endif

  IndexT pb_in = InverseIndex(pb_out);

  Configuration config = Configuration(Ly, Lx);

  TensorNetwork2D<GQTEN_Double, U1QN> tn2d = TensorNetwork2D<GQTEN_Double, U1QN>(Ly, Lx);

  TruncatePara trunc_para = TruncatePara(4, 8, 1e-12);

  void SetUp(void) {
    TPS<GQTEN_Double, U1QN> tps(Ly, Lx);
    gqten::hp_numeric::SetTensorManipulationThreads(1);
    gqten::hp_numeric::SetTensorTransposeNumThreads(1);
    tps.Load("tps_heisenberg_D4");

    SplitIndexTPS<GQTEN_Double, U1QN> split_index_tps(tps);
    for (size_t i = 0; i < Lx; i++) { //col index
      for (size_t j = 0; j < Ly; j++) { //row index
        config({j, i}) = (i + j) % 2;
      }
    }
    tn2d = TensorNetwork2D<GQTEN_Double, U1QN>(split_index_tps, config);
  }
};

TEST_F(TestSpin2DTensorNetwork, HeisenbergD4NNTrace) {
  tn2d.GrowBMPSForRow(2, trunc_para);
  tn2d.InitBTen(BTenPOSITION::LEFT, 2);
  tn2d.GrowFullBTen(BTenPOSITION::RIGHT, 2, 2, true);
  double psi_a = tn2d.Trace({2, 0}, HORIZONTAL);
  std::cout << "Amplitude by horizontal BMPS = " << psi_a << std::endl;

  tn2d.BTenMoveStep(BTenPOSITION::RIGHT);
  double psi_b = tn2d.Trace({2, 1}, HORIZONTAL);
  EXPECT_NEAR(psi_a, psi_b, 1e-14);

  tn2d.GrowBMPSForCol(1, trunc_para);
  tn2d.InitBTen(BTenPOSITION::DOWN, 1);
  tn2d.GrowFullBTen(BTenPOSITION::UP, 1, 2, true);
  double psi_c = tn2d.Trace({Ly - 2, 1}, VERTICAL);
  std::cout << "Amplitude by vertical BMPS = " << psi_c << std::endl;
  tn2d.BTenMoveStep(BTenPOSITION::UP);
  double psi_d = tn2d.Trace({Ly - 3, 1}, VERTICAL);
  EXPECT_NEAR(psi_c, psi_d, 1e-14);
}


TEST_F(TestSpin2DTensorNetwork, HeisenbergD4NNTraceBMPSSingleSiteUpdate) {
  tn2d.GrowBMPSForRow(2, trunc_para, gqpeps::VARIATION1Site);
  tn2d.InitBTen(BTenPOSITION::LEFT, 2);
  tn2d.GrowFullBTen(BTenPOSITION::RIGHT, 2, 2, true);
  double psi_a = tn2d.Trace({2, 0}, HORIZONTAL);
  std::cout << "Amplitude by horizontal BMPS = " << psi_a << std::endl;

  tn2d.BTenMoveStep(BTenPOSITION::RIGHT);
  double psi_b = tn2d.Trace({2, 1}, HORIZONTAL);
  EXPECT_NEAR(psi_a, psi_b, 1e-14);

  tn2d.GrowBMPSForCol(1, trunc_para);
  tn2d.InitBTen(BTenPOSITION::DOWN, 1);
  tn2d.GrowFullBTen(BTenPOSITION::UP, 1, 2, true);
  double psi_c = tn2d.Trace({Ly - 2, 1}, VERTICAL);
  std::cout << "Amplitude by vertical BMPS = " << psi_c << std::endl;
  tn2d.BTenMoveStep(BTenPOSITION::UP);
  double psi_d = tn2d.Trace({Ly - 3, 1}, VERTICAL);
  EXPECT_NEAR(psi_c, psi_d, 1e-14);
}

TEST_F(TestSpin2DTensorNetwork, HeisenbergD4BTen2) {
  tn2d.GrowBMPSForRow(2, trunc_para);
  tn2d.InitBTen2(BTenPOSITION::LEFT, 2);
  tn2d.GrowFullBTen2(BTenPOSITION::RIGHT, 2, 2, true);
}
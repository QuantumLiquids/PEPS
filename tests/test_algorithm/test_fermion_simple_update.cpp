// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-08-16
*
* Description: QuantumLiquids/PEPS project. Unittests for PEPS Simple Update in Fermionic model.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h"
#include "qlpeps/algorithm/simple_update/triangle_nn_on_sqr_peps_simple_update.h"

using namespace qlten;
using namespace qlpeps;

using qlmps::CaseParamsParserBasic;
char *params_file;

using qlten::special_qn::fZ2QN;

struct SimpleUpdateTestParams : public CaseParamsParserBasic {
  SimpleUpdateTestParams(const char *f) : CaseParamsParserBasic(f) {
    Lx = ParseInt("Lx");
    Ly = ParseInt("Ly");
    D = ParseInt("D");
    Tau0 = ParseDouble("Tau0");
    Steps = ParseInt("Steps");
  }

  size_t Ly;
  size_t Lx;
  size_t D;
  double Tau0;
  size_t Steps;
};

struct Z2tJModelTools : public testing::Test {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using DTensor = QLTensor<QLTEN_Double, fZ2QN>;
  using ZTensor = QLTensor<QLTEN_Complex, fZ2QN>;

  SimpleUpdateTestParams params = SimpleUpdateTestParams(params_file);
  size_t Lx = params.Lx; //cols
  size_t Ly = params.Ly;
  double t = 1.0;
  double J = 0.3;
  fZ2QN qn0 = fZ2QN(0);
  IndexT pb_out = IndexT({QNSctT(fZ2QN(1), 2), // |up>, |down>
                          QNSctT(fZ2QN(0), 1)}, // |0> empty state
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);

  DTensor df = DTensor({pb_in, pb_out});
  DTensor dsz = DTensor({pb_in, pb_out});
  DTensor dsp = DTensor({pb_in, pb_out});
  DTensor dsm = DTensor({pb_in, pb_out});
  DTensor dcup = DTensor({pb_in, pb_out});
  DTensor dcdagup = DTensor({pb_in, pb_out});
  DTensor dcdn = DTensor({pb_in, pb_out});
  DTensor dcdagdn = DTensor({pb_in, pb_out});

  ZTensor zf = ZTensor({pb_in, pb_out});
  ZTensor zsz = ZTensor({pb_in, pb_out});
  ZTensor zsp = ZTensor({pb_in, pb_out});
  ZTensor zsm = ZTensor({pb_in, pb_out});
  ZTensor zcup = ZTensor({pb_in, pb_out});
  ZTensor zcdagup = ZTensor({pb_in, pb_out});
  ZTensor zcdn = ZTensor({pb_in, pb_out});
  ZTensor zcdagdn = ZTensor({pb_in, pb_out});

  // nearest-neighbor Hamiltonian term, for the construction of evolve gates
  DTensor dham_tj_nn = DTensor({pb_in, pb_out, pb_in, pb_out});
  ZTensor zham_tj_nn = ZTensor({pb_in, pb_out, pb_in, pb_out});

  void SetUp(void) {
    //set up basic operators
    df({0, 0}) = -1;
    df({1, 1}) = -1;
    df({2, 2}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({1, 0}) = 1;
    dsm({0, 1}) = 1;
    dcup({0, 2}) = 1;
    dcdagup({2, 0}) = 1;
    dcdn({1, 2}) = 1;
    dcdagdn({2, 1}) = 1;

    zf({0, 0}) = -1;
    zf({1, 1}) = -1;
    zf({2, 2}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({1, 0}) = 1;
    zsm({0, 1}) = 1;
    zcup({0, 2}) = 1;
    zcdagup({2, 0}) = 1;
    zcdn({1, 2}) = 1;
    zcdagdn({2, 1}) = 1;

    //set up nearest-neighbor Hamiltonian terms
    //-t (c^dag_{i, s} c_{j,s} + c^dag_{j,s} c_{i,s}) + J S_i \cdot S_j
    dham_tj_nn({2, 0, 0, 2}) = t; //extra sign here
    dham_tj_nn({2, 1, 1, 2}) = t; //extra sign here
    dham_tj_nn({0, 2, 2, 0}) = -t;
    dham_tj_nn({1, 2, 2, 1}) = -t;

    dham_tj_nn({0, 0, 0, 0}) = 0.25 * J;
    dham_tj_nn({1, 1, 1, 1}) = 0.25 * J;
    dham_tj_nn({1, 1, 0, 0}) = -0.25 * J;
    dham_tj_nn({0, 0, 1, 1}) = -0.25 * J;
    dham_tj_nn({0, 1, 1, 0}) = 0.5 * J;
    dham_tj_nn({1, 0, 0, 1}) = 0.5 * J;

    zham_tj_nn = ToComplex(dham_tj_nn);
  }
};

TEST_F(Z2tJModelTools, tJModelHalfFilling) {
  // ED ground state energy in 4x4 = -9.189207065192949 * J
  qlten::hp_numeric::SetTensorManipulationThreads(1);
  SquareLatticePEPS<QLTEN_Double, fZ2QN> peps0(pb_out, Ly, Lx);

  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  //half-filling
  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(params.Steps, params.Tau0, 1, params.D, 1e-10);
  SimpleUpdateExecutor<QLTEN_Double, fZ2QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Double, fZ2QN>(update_para, peps0,
                                                                             dham_tj_nn);
  su_exe->Execute();
  auto peps = su_exe->GetPEPS();
  delete su_exe;
  for (auto gamma : peps.Gamma) {
    EXPECT_EQ(gamma.GetQNBlkNum(), 1);
  }
  for (auto lam : peps.lambda_horiz) {
    EXPECT_EQ(lam.GetQNBlkNum(), 1);
  }
  for (auto lam : peps.lambda_vert) {
    EXPECT_EQ(lam.GetQNBlkNum(), 1);
  }
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  params_file = argv[1];
  return RUN_ALL_TESTS();
}

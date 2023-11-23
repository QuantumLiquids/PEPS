// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: GraceQ/VMC-PEPS project. Unittests for Simple Update in PEPS optimization.
*/

#define PLAIN_TRANSPOSE 1

#include "gtest/gtest.h"
#include "gqten/gqten.h"
#include "gqmps2/case_params_parser.h"
#include "gqpeps/algorithm/simple_update/square_lattice_nn_simple_update.h"
#include "gqpeps/algorithm/simple_update/triangle_nn_on_sqr_peps_simple_update.h"

using namespace gqten;
using namespace gqpeps;

using gqten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

using gqmps2::CaseParamsParserBasic;

char *params_file;

struct SimpleUpdateParams : public CaseParamsParserBasic {
  SimpleUpdateParams(const char *f) : CaseParamsParserBasic(f) {
    Lx = ParseInt("Lx");
    Ly = ParseInt("Ly");
  }

  size_t Ly;
  size_t Lx;
};


// Test spin systems
struct TestSimpleUpdateSpinSystem : public testing::Test {
  size_t Lx; //cols
  size_t Ly;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
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

  DGQTensor did = DGQTensor({pb_in, pb_out});
  DGQTensor dsz = DGQTensor({pb_in, pb_out});
  DGQTensor dsp = DGQTensor({pb_in, pb_out});
  DGQTensor dsm = DGQTensor({pb_in, pb_out});

  ZGQTensor zid = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsz = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsp = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsm = ZGQTensor({pb_in, pb_out});

  DGQTensor ham_hei_nn = DGQTensor({pb_in, pb_out, pb_in, pb_out});
  DGQTensor ham_hei_tri;  // three-site hamiltonian in triangle lattice

  void SetUp(void) {
    SimpleUpdateParams params = SimpleUpdateParams(params_file);
    Lx = params.Lx;
    Ly = params.Ly;

    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    ham_hei_nn({0, 0, 0, 0}) = 0.25;
    ham_hei_nn({1, 1, 1, 1}) = 0.25;
    ham_hei_nn({1, 1, 0, 0}) = -0.25;
    ham_hei_nn({0, 0, 1, 1}) = -0.25;
    ham_hei_nn({0, 1, 1, 0}) = 0.5;
    ham_hei_nn({1, 0, 0, 1}) = 0.5;

    DGQTensor ham_hei_tri_terms[3];
    for (size_t i = 0; i < 3; i++) {
      ham_hei_tri_terms[i] = DGQTensor({pb_in, pb_out, pb_in, pb_out, pb_in, pb_out});
    }

    for (size_t i = 0; i < 2; i++) {
      ham_hei_tri_terms[0]({0, 0, 0, 0, i, i}) = 0.25;
      ham_hei_tri_terms[0]({1, 1, 1, 1, i, i}) = 0.25;
      ham_hei_tri_terms[0]({1, 1, 0, 0, i, i}) = -0.25;
      ham_hei_tri_terms[0]({0, 0, 1, 1, i, i}) = -0.25;
      ham_hei_tri_terms[0]({0, 1, 1, 0, i, i}) = 0.5;
      ham_hei_tri_terms[0]({1, 0, 0, 1, i, i}) = 0.5;
    }

    for (size_t i = 0; i < 2; i++) {
      ham_hei_tri_terms[1]({0, 0, i, i, 0, 0}) = 0.25;
      ham_hei_tri_terms[1]({1, 1, i, i, 1, 1}) = 0.25;
      ham_hei_tri_terms[1]({1, 1, i, i, 0, 0}) = -0.25;
      ham_hei_tri_terms[1]({0, 0, i, i, 1, 1}) = -0.25;
      ham_hei_tri_terms[1]({0, 1, i, i, 1, 0}) = 0.5;
      ham_hei_tri_terms[1]({1, 0, i, i, 0, 1}) = 0.5;
    }

    for (size_t i = 0; i < 2; i++) {
      ham_hei_tri_terms[2]({i, i, 0, 0, 0, 0}) = 0.25;
      ham_hei_tri_terms[2]({i, i, 1, 1, 1, 1}) = 0.25;
      ham_hei_tri_terms[2]({i, i, 1, 1, 0, 0}) = -0.25;
      ham_hei_tri_terms[2]({i, i, 0, 0, 1, 1}) = -0.25;
      ham_hei_tri_terms[2]({i, i, 0, 1, 1, 0}) = 0.5;
      ham_hei_tri_terms[2]({i, i, 1, 0, 0, 1}) = 0.5;
    }
    ham_hei_tri = ham_hei_tri_terms[0] + ham_hei_tri_terms[1] + ham_hei_tri_terms[2];

    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;
  }
};

TEST_F(TestSimpleUpdateSpinSystem, NNIsing) {
  gqten::hp_numeric::SetTensorManipulationThreads(1);
  gqten::hp_numeric::SetTensorTransposeNumThreads(1);

  SquareLatticePEPS<GQTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));

  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(5, 0.01, 1, 4, 1e-5);
  DGQTensor ham_nn;
  Contract(&dsz, {}, &dsz, {}, &ham_nn);
  SimpleUpdateExecutor<GQTEN_Double, U1QN> *su_exe = new SquareLatticeNNSimpleUpdateExecutor(update_para, peps0,
                                                                                             ham_nn);
  su_exe->Execute();
  delete su_exe;
}

TEST_F(TestSimpleUpdateSpinSystem, NNHeisenberg) {
  SquareLatticePEPS<GQTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  size_t sz_int = 0;
  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(10, 0.1, 1, 2, 1e-5);

  SimpleUpdateExecutor<GQTEN_Double, U1QN> *su_exe = new SquareLatticeNNSimpleUpdateExecutor(update_para, peps0,
                                                                                             ham_hei_nn);
  su_exe->Execute();

  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->SetStepLenth(0.05);
  su_exe->Execute();

  auto tps4 = TPS<GQTEN_Double, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult("su_update_resultD4", true);
  tps4.Dump("tps_heisenberg_D4");
  delete su_exe;
}

TEST_F(TestSimpleUpdateSpinSystem, NNHeisenbergD8) {
  SquareLatticePEPS<GQTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  peps0.Load("su_update_resultD4");

  SimpleUpdatePara update_para(10, 0.1, 1, 8, 1e-10);

  auto su_exe = new SquareLatticeNNSimpleUpdateExecutor(update_para, peps0, ham_hei_nn);
  su_exe->Execute();

  auto tps8 = TPS<GQTEN_Double, U1QN>(su_exe->GetPEPS());

  su_exe->DumpResult("su_update_resultD8", true);
  tps8.Dump("tps_heisenberg_D8");
  delete su_exe;
}

TEST_F(TestSimpleUpdateSpinSystem, TriangleNNHeisenberg) {
  SquareLatticePEPS<GQTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  size_t sz_int = 0;
  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(20, 0.1, 1, 2, 1e-5);

  SimpleUpdateExecutor<GQTEN_Double, U1QN> *su_exe
      = new TriangleNNModelSquarePEPSSimpleUpdateExecutor(update_para, peps0,
                                                          ham_hei_nn,
                                                          ham_hei_tri);
  su_exe->Execute();

  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->SetStepLenth(0.05);
  su_exe->Execute();

  auto tps4 = TPS<GQTEN_Double, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult("su_update_tri_heisenberg_D4", true);
  tps4.Dump("tps_tri_heisenberg_D4");
  delete su_exe;
}


//TEST_F(TestSimpleUpdateSpinSystem, NNHeisenbergD16) {
//  SquareLatticePEPS<GQTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
//  peps0.Load("su_update_resultD8");
//
//  SimpleUpdatePara update_para(10, 0.05, 1, 16, 1e-10);
//
//  auto su_exe = SimpleUpdateExecutor(update_para, ham_hei_nn, peps0);
//  su_exe.Execute();
//  auto tps16 = TPS<GQTEN_Double, U1QN>(su_exe.GetPEPS());
//
//  su_exe.DumpResult("su_update_resultD16", true);
//  tps16.Dump("tps_heisenberg_D16");
//}


int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  params_file = argv[1];
  return RUN_ALL_TESTS();
}

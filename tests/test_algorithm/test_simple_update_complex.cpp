// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: QuantumLiquids/PEPS project. Unittests for Simple Update in PEPS optimization.
*/

#define PLAIN_TRANSPOSE 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/algorithm/simple_update/simple_update_model_all.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

using qlmps::CaseParamsParserBasic;

char *params_file;

struct SystemSizeParams : public CaseParamsParserBasic {
  SystemSizeParams(const char *f) : CaseParamsParserBasic(f) {
    Lx = ParseInt("Lx");
    Ly = ParseInt("Ly");
  }

  size_t Ly;
  size_t Lx;
};

// Test spin systems
struct SpinSystemSimpleUpdate : public testing::Test {
  size_t Lx; //cols
  size_t Ly;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
#ifdef U1SYM
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         TenIndexDirType::OUT
  );
#else
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT
  );
#endif
  IndexT pb_in = InverseIndex(pb_out);

  DQLTensor did = DQLTensor({pb_in, pb_out});
  DQLTensor dsz = DQLTensor({pb_in, pb_out});
  DQLTensor dsp = DQLTensor({pb_in, pb_out});
  DQLTensor dsm = DQLTensor({pb_in, pb_out});

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});

  DQLTensor dham_hei_nn = DQLTensor({pb_in, pb_out, pb_in, pb_out});
  DQLTensor dham_hei_tri;  // three-site hamiltonian in triangle lattice

  ZQLTensor zham_hei_nn = ZQLTensor({pb_in, pb_out, pb_in, pb_out});
  ZQLTensor zham_hei_tri;
  void SetUp(void) {
    SystemSizeParams params = SystemSizeParams(params_file);
    Lx = params.Lx;
    Ly = params.Ly;

    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    dham_hei_nn({0, 0, 0, 0}) = 0.25;
    dham_hei_nn({1, 1, 1, 1}) = 0.25;
    dham_hei_nn({1, 1, 0, 0}) = -0.25;
    dham_hei_nn({0, 0, 1, 1}) = -0.25;
    dham_hei_nn({0, 1, 1, 0}) = 0.5;
    dham_hei_nn({1, 0, 0, 1}) = 0.5;

    DQLTensor ham_hei_tri_terms[3];
    for (size_t i = 0; i < 3; i++) {
      ham_hei_tri_terms[i] = DQLTensor({pb_in, pb_out, pb_in, pb_out, pb_in, pb_out});
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
    dham_hei_tri = ham_hei_tri_terms[0] + ham_hei_tri_terms[1] + ham_hei_tri_terms[2];

    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;

    zham_hei_nn({0, 0, 0, 0}) = 0.25;
    zham_hei_nn({1, 1, 1, 1}) = 0.25;
    zham_hei_nn({1, 1, 0, 0}) = -0.25;
    zham_hei_nn({0, 0, 1, 1}) = -0.25;
    zham_hei_nn({0, 1, 1, 0}) = 0.5;
    zham_hei_nn({1, 0, 0, 1}) = 0.5;

    zham_hei_tri = ToComplex(dham_hei_tri);
  }
};

TEST_F(SpinSystemSimpleUpdate, NNIsing) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  SquareLatticePEPS<QLTEN_Complex, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));

  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(5, 0.01, 1, 4, 1e-5);
  ZQLTensor ham_nn;
  Contract(&zsz, {}, &zsz, {}, &ham_nn);
  SimpleUpdateExecutor<QLTEN_Complex, U1QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Complex, U1QN>(update_para, peps0,
                                                                             ham_nn);
  su_exe->Execute();
  delete su_exe;
}

TEST_F(SpinSystemSimpleUpdate, NNHeisenberg) {
  SquareLatticePEPS<QLTEN_Complex, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(10, 0.1, 1, 2, 1e-5);

  SimpleUpdateExecutor<QLTEN_Complex, U1QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Complex, U1QN>(update_para, peps0,
                                                                             zham_hei_nn);
  su_exe->Execute();

  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->SetStepLenth(0.05);
  su_exe->Execute();

  auto tps4 = TPS<QLTEN_Complex, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult("zsu_update_resultD4", true);
  tps4.Dump("ztps_heisenberg_D4");
  delete su_exe;
}

TEST_F(SpinSystemSimpleUpdate, NNHeisenbergD8) {
  SquareLatticePEPS<QLTEN_Complex, U1QN> peps0(pb_out, Ly, Lx);
  peps0.Load("zsu_update_resultD4");

  SimpleUpdatePara update_para(10, 0.1, 1, 8, 1e-10);

  auto su_exe = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Complex, U1QN>(update_para, peps0, zham_hei_nn);
  su_exe->Execute();

  auto tps8 = TPS<QLTEN_Complex, U1QN>(su_exe->GetPEPS());

  su_exe->DumpResult("zsu_update_resultD8", true);
  tps8.Dump("ztps_heisenberg_D8");
  delete su_exe;
}

TEST_F(SpinSystemSimpleUpdate, TriangleNNHeisenberg) {
  SquareLatticePEPS<QLTEN_Complex, U1QN> peps0(pb_out, Ly, Lx);
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

  SimpleUpdateExecutor<QLTEN_Complex, U1QN> *su_exe
      = new TriangleNNModelSquarePEPSSimpleUpdateExecutor<QLTEN_Complex, U1QN>(update_para, peps0,
                                                                               zham_hei_nn,
                                                                               zham_hei_tri);
  su_exe->Execute();

  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->SetStepLenth(0.05);
  su_exe->Execute();

  auto tps4 = TPS<QLTEN_Complex, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult("zsu_update_tri_heisenberg_D4", true);
  tps4.Dump("ztps_tri_heisenberg_D4");
  delete su_exe;
}

struct TestSimpleUpdateSpinSystemSquareJ1J2 : public testing::Test {
  size_t Lx; //cols
  size_t Ly;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
#ifdef U1SYM
  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                          QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         TenIndexDirType::OUT
  );
#else
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT
  );
#endif
  IndexT pb_in = InverseIndex(pb_out);

  DQLTensor did = DQLTensor({pb_in, pb_out});
  DQLTensor dsz = DQLTensor({pb_in, pb_out});
  DQLTensor dsp = DQLTensor({pb_in, pb_out});
  DQLTensor dsm = DQLTensor({pb_in, pb_out});

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});

  DQLTensor dham_hei_nn = DQLTensor({pb_in, pb_out, pb_in, pb_out});
  DQLTensor dham_hei_tri;  // three-site hamiltonian in triangle lattice

  ZQLTensor zham_hei_nn = ZQLTensor({pb_in, pb_out, pb_in, pb_out});
  ZQLTensor zham_hei_tri;

  void SetUp(void) {
    SystemSizeParams params = SystemSizeParams(params_file);
    Lx = params.Lx;
    Ly = params.Ly;
    double j1 = 1.0;
    double j2 = 0.52;

    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    dham_hei_nn({0, 0, 0, 0}) = 0.25;
    dham_hei_nn({1, 1, 1, 1}) = 0.25;
    dham_hei_nn({1, 1, 0, 0}) = -0.25;
    dham_hei_nn({0, 0, 1, 1}) = -0.25;
    dham_hei_nn({0, 1, 1, 0}) = 0.5;
    dham_hei_nn({1, 0, 0, 1}) = 0.5;

    zham_hei_nn = ToComplex(dham_hei_nn);
    DQLTensor ham_hei_tri_terms[3];
    for (size_t i = 0; i < 3; i++) {
      ham_hei_tri_terms[i] = DQLTensor({pb_in, pb_out, pb_in, pb_out, pb_in, pb_out});
    }

    for (size_t i = 0; i < 2; i++) {// A-B site
      ham_hei_tri_terms[0]({0, 0, 0, 0, i, i}) = 0.25;
      ham_hei_tri_terms[0]({1, 1, 1, 1, i, i}) = 0.25;
      ham_hei_tri_terms[0]({1, 1, 0, 0, i, i}) = -0.25;
      ham_hei_tri_terms[0]({0, 0, 1, 1, i, i}) = -0.25;
      ham_hei_tri_terms[0]({0, 1, 1, 0, i, i}) = 0.5;
      ham_hei_tri_terms[0]({1, 0, 0, 1, i, i}) = 0.5;
    }

    for (size_t i = 0; i < 2; i++) {// A-C site
      ham_hei_tri_terms[1]({0, 0, i, i, 0, 0}) = 0.25;
      ham_hei_tri_terms[1]({1, 1, i, i, 1, 1}) = 0.25;
      ham_hei_tri_terms[1]({1, 1, i, i, 0, 0}) = -0.25;
      ham_hei_tri_terms[1]({0, 0, i, i, 1, 1}) = -0.25;
      ham_hei_tri_terms[1]({0, 1, i, i, 1, 0}) = 0.5;
      ham_hei_tri_terms[1]({1, 0, i, i, 0, 1}) = 0.5;
    }

    for (size_t i = 0; i < 2; i++) {//B-C site
      ham_hei_tri_terms[2]({i, i, 0, 0, 0, 0}) = 0.25;
      ham_hei_tri_terms[2]({i, i, 1, 1, 1, 1}) = 0.25;
      ham_hei_tri_terms[2]({i, i, 1, 1, 0, 0}) = -0.25;
      ham_hei_tri_terms[2]({i, i, 0, 0, 1, 1}) = -0.25;
      ham_hei_tri_terms[2]({i, i, 0, 1, 1, 0}) = 0.5;
      ham_hei_tri_terms[2]({i, i, 1, 0, 0, 1}) = 0.5;
    }
    dham_hei_tri = 0.5 * j1 * ham_hei_tri_terms[0]
                   + j2 * ham_hei_tri_terms[1]
                   + 0.5 * j1 * ham_hei_tri_terms[2];

    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;
    zham_hei_tri = ToComplex(dham_hei_tri);
  }
};

TEST_F(TestSimpleUpdateSpinSystemSquareJ1J2, J1J2Heisenberg) {
  SquareLatticePEPS<QLTEN_Complex, U1QN> peps0(pb_out, Ly, Lx);
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

  SimpleUpdateExecutor<QLTEN_Complex, U1QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Complex, U1QN>(update_para, peps0,
                                                                            zham_hei_nn);
  su_exe->Execute();

  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->SetStepLenth(0.05);
  su_exe->Execute();

  auto tps4 = TPS<QLTEN_Complex, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult("zsu_update_resultj1j2D4", true);
  tps4.Dump("ztps_heisenbergj1j2_D4");
  delete su_exe;
}

//TEST_F(SpinSystemSimpleUpdate, NNHeisenbergD16) {
//  SquareLatticePEPS<QLTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
//  peps0.Load("su_update_resultD8");
//
//  SimpleUpdatePara update_para(10, 0.05, 1, 16, 1e-10);
//
//  auto su_exe = SimpleUpdateExecutor(update_para, dham_hei_nn, peps0);
//  su_exe.Execute();
//  auto tps16 = TPS<QLTEN_Double, U1QN>(su_exe.GetPEPS());
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

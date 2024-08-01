/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-07-30
*
* Description: QuantumLiquids/PEPS project. Unittests for Loop Update in PEPS optimization.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/algorithm/simple_update/loop_update.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

using qlmps::CaseParamsParserBasic;

using LoopGateT = std::array<DQLTensor, 4>;
char *params_file;

struct SystemSizeParams : public CaseParamsParserBasic {
  SystemSizeParams(const char *f) : CaseParamsParserBasic(f) {
    Lx = ParseInt("Lx");
    Ly = ParseInt("Ly");
  }

  size_t Ly;
  size_t Lx;
};

struct TransverseIsingLoopUpdate : public testing::Test {
  SystemSizeParams params = SystemSizeParams(params_file);
  size_t Lx = params.Lx; //cols
  size_t Ly = params.Ly;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);

  DQLTensor did = DQLTensor({pb_in, pb_out});
  DQLTensor dsz = DQLTensor({pb_in, pb_out});
  DQLTensor dsp = DQLTensor({pb_in, pb_out});
  DQLTensor dsm = DQLTensor({pb_in, pb_out});

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});

  double tau = 0.1;
  double h = 2.0; //h_c \simeq 3.04
  DuoMatrix<LoopGateT> evolve_gates = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);

  void SetUp(void) {
    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;

    //corner
    evolve_gates({0, 0}) = GenerateLoopGates(tau, h,
                                             2, 3, 4, 3,
                                             1, 2, 2, 1);
    evolve_gates({0, Lx - 2}) = GenerateLoopGates(tau, h,
                                                  3, 2, 3, 4,
                                                  1, 1, 2, 2);
    evolve_gates({Ly - 2, 0}) = GenerateLoopGates(tau, h,
                                                  3, 4, 3, 2,
                                                  2, 2, 1, 1);
    evolve_gates({Ly - 2, Lx - 2}) = GenerateLoopGates(tau, h,
                                                       4, 3, 2, 3,
                                                       2, 1, 1, 2);

    auto gates_upper = GenerateLoopGates(tau, h,
                                         3, 3, 4, 4,
                                         1, 2, 2, 2);
    auto gates_lower = GenerateLoopGates(tau, h,
                                         4, 4, 3, 3,
                                         2, 2, 1, 2);
    for (size_t col = 1; col < Lx - 2; col++) {
      evolve_gates({0, col}) = gates_upper;
      evolve_gates({Ly - 2, col}) = gates_lower;
    }

    auto gates_left = GenerateLoopGates(tau, h,
                                        3, 4, 4, 3,
                                        2, 2, 2, 1);
    auto gates_middle = GenerateLoopGates(tau, h,
                                          4, 4, 4, 4,
                                          2, 2, 2, 2);
    auto gates_right = GenerateLoopGates(tau, h,
                                         4, 3, 3, 4,
                                         2, 1, 2, 2);
    for (size_t row = 1; row < Ly - 2; row++) {
      evolve_gates({row, 0}) = gates_left;
      evolve_gates({row, Lx - 2}) = gates_right;
    }
    for (size_t col = 1; col < Lx - 2; col++) {
      for (size_t row = 1; row < Ly - 2; row++) {
        evolve_gates({row, col}) = gates_middle;
      }
    }
  }

  LoopGateT GenerateLoopGates(
      const double tau, // imaginary time
      const double h, //magnetic field,
      const size_t z0, const size_t z1, const size_t z2, const size_t z3, //coordination number
      const size_t n0, const size_t n1, const size_t n2, const size_t n3  //bond share number
      //n0 bond is between z0 an z1 points
  ) {
    const std::vector<size_t> zs = {z0, z1, z2, z3};
    const std::vector<size_t> ns = {n0, n1, n2, n3};
    LoopGateT gates;
    gates[0] = DQLTensor({pb_out, pb_in, pb_out, pb_out});
    gates[1] = DQLTensor({pb_in, pb_in, pb_out, pb_out});
    gates[2] = DQLTensor({pb_in, pb_in, pb_out, pb_in});
    gates[3] = DQLTensor({pb_out, pb_in, pb_out, pb_in});
    for (size_t i = 0; i < 4; i++) {
      DQLTensor &gate = gates[i];
      //Id
      gate({0, 0, 0, 0}) = 1.0;
      gate({0, 1, 1, 0}) = 1.0;
      //h * tau * sigma_x
      gate({0, 0, 1, 0}) = h * tau / double(zs[i]);
      gate({0, 1, 0, 0}) = h * tau / double(zs[i]);
      //sigma_z * tau
      gate({0, 0, 0, 1}) = 1.0 * tau / double(ns[i]);
      gate({0, 1, 1, 1}) = -1.0 * tau / double(ns[i]);
      //sigma_z
      gate({1, 0, 0, 0}) = 1.0;
      gate({1, 1, 1, 0}) = -1.0;
    }
    return gates;
  }
};

TEST_F(TransverseIsingLoopUpdate, TransverseIsing) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);
  //initial peps as direct product state
  SquareLatticePEPS<QLTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));

  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara simple_update_para(100, tau, 1, 4, 1e-5);
  LanczosParams lanczos_params(1e-10, 30);
  double fet_tol = 1e-12;
  double fet_max_iter = 30;
  ConjugateGradientParams cg_params(100, 1e-10, 20, 0.0);

  SimpleUpdateExecutor<QLTEN_Double, U1QN>
      *su_exe = new LoopUpdate<QLTEN_Double, U1QN>(simple_update_para,
                                                   lanczos_params,
                                                   fet_tol, fet_max_iter,
                                                   cg_params,
                                                   peps0,
                                                   evolve_gates);
  su_exe->Execute();
  delete su_exe;
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  params_file = argv[1];
  return RUN_ALL_TESTS();
}

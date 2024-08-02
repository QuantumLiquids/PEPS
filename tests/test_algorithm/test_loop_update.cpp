/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-07-30
*
* Description: QuantumLiquids/PEPS project. Unittests for Loop Update in PEPS optimization.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h"
#include "qlpeps/algorithm/simple_update/loop_update.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square.h"
#include "qlpeps/algorithm/vmc_update/wave_function_component_classes/square_tps_sample_full_space_nn_flip.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_measurement.h"

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
  DQLTensor dsigmaz = DQLTensor({pb_in, pb_out});
  DQLTensor dsigmax = DQLTensor({pb_in, pb_out});
  DQLTensor dsp = DQLTensor({pb_in, pb_out});
  DQLTensor dsm = DQLTensor({pb_in, pb_out});
  DQLTensor dham_nn;

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});

  double tau0 = 0.1;
  double tau1 = 0.01;
  double tau2 = 0.001;
  double h = 3.0; //h_c \simeq 3.04
  // ED ground state energy = -50.186623882777752
  DuoMatrix<LoopGateT> evolve_gates0 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);
  DuoMatrix<LoopGateT> evolve_gates1 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);
  DuoMatrix<LoopGateT> evolve_gates2 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);

  boost::mpi::communicator world;//mpi support for measurement
  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsigmaz({0, 0}) = 1;
    dsigmaz({1, 1}) = -1;
    dsigmax({0, 1}) = 1;
    dsigmax({1, 0}) = 1;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;

    DQLTensor ham_nn1, ham_nn2, ham_nn3;
    Contract(&(dsigmaz), {}, &dsigmaz, {}, &ham_nn1);
    Contract(&(dsigmax), {}, &did, {}, &ham_nn2);
    Contract(&(did), {}, &dsigmax, {}, &ham_nn3);
    dham_nn = -ham_nn1 + (-0.25) * h * ham_nn2 + (-0.25) * h * ham_nn3;

    GenerateTransIsingAllEvolveGates(tau0, h, evolve_gates0);
    GenerateTransIsingAllEvolveGates(tau1, h, evolve_gates1);
    GenerateTransIsingAllEvolveGates(tau2, h, evolve_gates2);
  }

  void GenerateTransIsingAllEvolveGates(
      const double tau, // imaginary time
      const double h,
      DuoMatrix<LoopGateT> &evolve_gates //output
  ) {
    //corner
    evolve_gates({0, 0}) = GenerateTransIsingLoopGates(tau, h,
                                                       1, 2, 4, 2,
                                                       1, 2, 2, 1);
    evolve_gates({0, Lx - 2}) = GenerateTransIsingLoopGates(tau, h,
                                                            2, 1, 2, 4,
                                                            1, 1, 2, 2);
    evolve_gates({Ly - 2, 0}) = GenerateTransIsingLoopGates(tau, h,
                                                            2, 4, 2, 1,
                                                            2, 2, 1, 1);
    evolve_gates({Ly - 2, Lx - 2}) = GenerateTransIsingLoopGates(tau, h,
                                                                 4, 2, 1, 2,
                                                                 2, 1, 1, 2);

    auto gates_upper = GenerateTransIsingLoopGates(tau, h,
                                                   2, 2, 4, 4,
                                                   1, 2, 2, 2);
    auto gates_lower = GenerateTransIsingLoopGates(tau, h,
                                                   4, 4, 2, 2,
                                                   2, 2, 1, 2);
    for (size_t col = 1; col < Lx - 2; col++) {
      evolve_gates({0, col}) = gates_upper;
      evolve_gates({Ly - 2, col}) = gates_lower;
    }

    auto gates_left = GenerateTransIsingLoopGates(tau, h,
                                                  2, 4, 4, 2,
                                                  2, 2, 2, 1);
    auto gates_middle = GenerateTransIsingLoopGates(tau, h,
                                                    4, 4, 4, 4,
                                                    2, 2, 2, 2);
    auto gates_right = GenerateTransIsingLoopGates(tau, h,
                                                   4, 2, 2, 4,
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

  LoopGateT GenerateTransIsingLoopGates(
      const double tau, // imaginary time
      const double h, //magnetic field,
      const size_t m0, const size_t m1, const size_t m2, const size_t m3, //how many loops share the site
      const size_t n0, const size_t n1, const size_t n2, const size_t n3  //bond share number
      //n0 bond is between z0 an z1 points
  ) {
    const std::vector<size_t> ms = {m0, m1, m2, m3};
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
      gate({0, 0, 1, 0}) = h * tau / double(ms[i]);
      gate({0, 1, 0, 0}) = h * tau / double(ms[i]);
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

  //simple update, in boundary the setting of hamiltonian has edge errors
  SimpleUpdateExecutor<QLTEN_Double, U1QN>
      *su_exe1 = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Double, U1QN>(SimpleUpdatePara(100, 0.1, 1, 4, 1e-5),
                                                                             peps0,
                                                                             dham_nn);
  su_exe1->Execute();
  auto peps1 = su_exe1->GetPEPS();
  delete su_exe1;
  peps1.NormalizeAllTensor();
  //loop update
  SimpleUpdatePara simple_update_para(100, tau0, 1, 4, 1e-10);
  LanczosParams lanczos_params(1e-10, 30);
  double fet_tol = 1e-12;
  double fet_max_iter = 30;
  ConjugateGradientParams cg_params(100, 1e-10, 20, 0.0);

  SimpleUpdateExecutor<QLTEN_Double, U1QN>
      *loop_exe = new LoopUpdate<QLTEN_Double, U1QN>(simple_update_para,
                                                     lanczos_params,
                                                     fet_tol, fet_max_iter,
                                                     cg_params,
                                                     peps1,
                                                     evolve_gates0);
  loop_exe->Execute();

  auto peps2 = loop_exe->GetPEPS();
  delete loop_exe;

  simple_update_para.tau = tau1;
  loop_exe = new LoopUpdate<QLTEN_Double, U1QN>(simple_update_para,
                                                lanczos_params,
                                                fet_tol, fet_max_iter,
                                                cg_params,
                                                peps2,
                                                evolve_gates1);

  loop_exe->Execute();
  auto peps3 = loop_exe->GetPEPS();
  delete loop_exe;

  simple_update_para.tau = tau2;
  simple_update_para.steps = 300;
  loop_exe = new LoopUpdate<QLTEN_Double, U1QN>(simple_update_para,
                                                lanczos_params,
                                                fet_tol, fet_max_iter,
                                                cg_params,
                                                peps3,
                                                evolve_gates2);

  loop_exe->Execute();
  auto peps4 = loop_exe->GetPEPS();
  delete loop_exe;




  //measure the energy
  auto tps = TPS<QLTEN_Double, U1QN>(peps4);
  auto sitps = SplitIndexTPS<QLTEN_Double, U1QN>(tps);
  using Model = TransverseIsingSquare<QLTEN_Double, U1QN>;
  using SquareTPSSampleFullSpaceNNFlipT = SquareTPSSampleFullSpaceNNFlip<QLTEN_Double, U1QN>;
  size_t mc_samples = 1000;
  size_t mc_warmup = 100;
  VMCOptimizePara mc_measure_para = VMCOptimizePara(
      BMPSTruncatePara(4, 8, 1e-10,
                       CompressMPSScheme::VARIATION2Site,
                       std::make_optional<double>(1e-14),
                       std::make_optional<size_t>(10)),
      mc_samples, mc_warmup, 1,
      std::vector<size_t>(2, Lx * Ly / 2),
      Ly, Lx,
      {0.1},
      StochasticGradient);
  auto measure_executor =
      new MonteCarloMeasurementExecutor<QLTEN_Double, U1QN, SquareTPSSampleFullSpaceNNFlipT, Model>(mc_measure_para,
                                                                                                    sitps,
                                                                                                    world,
                                                                                                    Model(h));
  measure_executor->Execute();
  measure_executor->OutputEnergy();
  delete measure_executor;
}

int main(int argc, char *argv[]) {
  boost::mpi::environment env;
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  params_file = argv[1];
  return RUN_ALL_TESTS();
}

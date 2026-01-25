/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-07-30
*
* Description: QuantumLiquids/PEPS project. Unittests for Loop Update in PEPS optimization.
* 
* DEPRECATED: This test file has been abandoned and is no longer maintained.
* The loop update functionality may be removed in future versions.
* Use at your own risk.
*/

#define PLAIN_TRANSPOSE 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/algorithm/simple_update/simple_update_model_all.h"
#include "qlpeps/algorithm/loop_update/loop_update.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_xxz_obc.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_triangle_heisenberg_sqrpeps.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;
using TenElemT = TEN_ELEM_TYPE;
using Tensor = QLTensor<TenElemT, U1QN>;

using LoopGateT = std::array<Tensor, 4>;

// Hardcoded parameters instead of reading from file
struct SystemSizeParams {
  SystemSizeParams() {
    Lx = 4;  // From test_params.json
    Ly = 4;  // From test_params.json
  }

  size_t Ly;
  size_t Lx;
};

struct TransverseIsingLoopUpdate : public testing::Test {
  SystemSizeParams params;  // Remove file dependency
  size_t Lx = params.Lx; //cols
  size_t Ly = params.Ly;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);

  Tensor did = Tensor({pb_in, pb_out});
  Tensor dsz = Tensor({pb_in, pb_out});
  Tensor dsigmaz = Tensor({pb_in, pb_out});
  Tensor dsigmax = Tensor({pb_in, pb_out});
  Tensor dsp = Tensor({pb_in, pb_out});
  Tensor dsm = Tensor({pb_in, pb_out});
  Tensor dham_nn;

  double tau0 = 0.1;
  double tau1 = 0.01;
  double tau2 = 0.001;
  double h = 3.0; //h_c \simeq 3.04
  // ED ground state energy = -50.186623882777752
  DuoMatrix<LoopGateT> evolve_gates0 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);
  DuoMatrix<LoopGateT> evolve_gates1 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);
  DuoMatrix<LoopGateT> evolve_gates2 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);

  const MPI_Comm comm = MPI_COMM_WORLD;//mpi support for measurement
  int rank;
  int mpi_size;
  void SetUp(void) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
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

    Tensor ham_nn1, ham_nn2, ham_nn3;
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
    gates[0] = Tensor({pb_out, pb_in, pb_out, pb_out});
    gates[1] = Tensor({pb_in, pb_in, pb_out, pb_out});
    gates[2] = Tensor({pb_in, pb_in, pb_out, pb_in});
    gates[3] = Tensor({pb_out, pb_in, pb_out, pb_in});
    for (size_t i = 0; i < 4; i++) {
      Tensor &gate = gates[i];
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
  SquareLatticePEPS<TenElemT, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));

  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  //simple update, in boundary the setting of hamiltonian has edge errors
  SimpleUpdateExecutor<TenElemT, U1QN>
      *su_exe1 = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, U1QN>(SimpleUpdatePara(100, 0.1, 1, 4, 1e-5),
                                                                         peps0,
                                                                         dham_nn);
  su_exe1->Execute();
  auto peps1 = su_exe1->GetPEPS();
  delete su_exe1;
  peps1.NormalizeAllTensor();
  //loop update
  ArnoldiParams arnoldi_params(1e-13, 30);
  double fet_tol = 1e-12;
  double fet_max_iter = 30;
  ConjugateGradientParams cg_params(100, 1e-10, 20, 0.0);
  FullEnvironmentTruncateParams fet_params(1, 4, 1e-10,
                                           fet_tol, fet_max_iter,
                                           cg_params);

  auto *loop_exe = new LoopUpdateExecutor<TenElemT, U1QN>(LoopUpdateTruncatePara(
                                                              arnoldi_params,
                                                              1e-7,
                                                              fet_params),
                                                          100,
                                                          tau0,
                                                          evolve_gates0,
                                                          peps1);
  loop_exe->Execute();

  auto peps2 = loop_exe->GetPEPS();
  delete loop_exe;

  loop_exe = new LoopUpdateExecutor<TenElemT, U1QN>(LoopUpdateTruncatePara(
                                                        arnoldi_params,
                                                        1e-7,
                                                        fet_params),
                                                    100,
                                                    tau1,
                                                    evolve_gates1,
                                                    peps2);

  loop_exe->Execute();
  auto peps3 = loop_exe->GetPEPS();
  delete loop_exe;

  loop_exe = new LoopUpdateExecutor<TenElemT, U1QN>(LoopUpdateTruncatePara(
                                                        arnoldi_params,
                                                        1e-7,
                                                        fet_params),
                                                    200,
                                                    tau2,
                                                    evolve_gates2,
                                                    peps3);

  loop_exe->Execute();
  auto peps4 = loop_exe->GetPEPS();
  delete loop_exe;


  //measure the energy
  auto tps = TPS<TenElemT, U1QN>(peps4);
  auto sitps = SplitIndexTPS<TenElemT, U1QN>(tps);
  using Model = TransverseFieldIsingSquare;
  using MCUpdateSquareNNFullSpaceUpdateT = MCUpdateSquareNNFullSpaceUpdate;
  size_t mc_samples = 1000;
  size_t mc_warmup = 100;
  std::string tps_path = "TPS_TransverseIsing" + std::to_string(Lx) + "x" + std::to_string(Ly);
  Configuration measure_config(Ly, Lx, OccupancyNum(std::vector<size_t>(2, Lx * Ly / 2)));
  MonteCarloParams measure_mc_params(mc_samples, mc_warmup, 1, measure_config, false); // not warmed up initially
  PEPSParams measure_peps_params(BMPSTruncateParams<qlten::QLTEN_Double>::Variational2Site(4, 8, 1e-10, 1e-14, 10));
  MCMeasurementParams mc_measure_para(measure_mc_params, measure_peps_params);
  auto measure_executor =
      new MCPEPSMeasurer<TenElemT, U1QN, MCUpdateSquareNNFullSpaceUpdateT, Model>(mc_measure_para,
                                                                                                 sitps,
                                                                                                 comm,
                                                                                                 Model(h));
  measure_executor->Execute();
  measure_executor->OutputEnergy();
  delete measure_executor;
}

//Square Heisenberg
struct HeisenbergLoopUpdate : public testing::Test {
  SystemSizeParams params;  // Remove file dependency
  size_t Lx = params.Lx; //cols
  size_t Ly = params.Ly;

#ifdef U1SYM
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1)},
                         TenIndexDirType::OUT
  );
  IndexT vb_out = IndexT({QNSctT(U1QN(0), 2),
                          QNSctT(U1QN(-1), 1),
                          QNSctT(U1QN(1), 1)},
                         TenIndexDirType::OUT
  );
#else
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT
  );
   IndexT vb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 4)},
                         TenIndexDirType::OUT
  );
#endif
  IndexT pb_in = InverseIndex(pb_out);

  IndexT vb_in = InverseIndex(vb_out);

  Tensor did = Tensor({pb_in, pb_out});
  Tensor dsz = Tensor({pb_in, pb_out});
  Tensor dsp = Tensor({pb_in, pb_out});
  Tensor dsm = Tensor({pb_in, pb_out});

  Tensor dham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});

  double tau0 = 0.01;
  double tau1 = 0.001;
  // ED ground state energy = -9.189207065192949
  DuoMatrix<LoopGateT> evolve_gates0 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);
  DuoMatrix<LoopGateT> evolve_gates1 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);

  const MPI_Comm comm = MPI_COMM_WORLD;//mpi support for measurement
  int rank, mpi_size;
  void GenerateSquareHeisenbergAllEvolveGates(
      const double tau, // imaginary time
      DuoMatrix<LoopGateT> &evolve_gates //output
  ) {
    //corner
    evolve_gates({0, 0}) = GenerateSquareHeisenbergLoopGates(tau,
                                                             1, 2, 2, 1);
    evolve_gates({0, Lx - 2}) = GenerateSquareHeisenbergLoopGates(tau,
                                                                  1, 1, 2, 2);
    evolve_gates({Ly - 2, 0}) = GenerateSquareHeisenbergLoopGates(tau,
                                                                  2, 2, 1, 1);
    evolve_gates({Ly - 2, Lx - 2}) = GenerateSquareHeisenbergLoopGates(tau,
                                                                       2, 1, 1, 2);

    auto gates_upper = GenerateSquareHeisenbergLoopGates(tau,
                                                         1, 2, 2, 2);
    auto gates_lower = GenerateSquareHeisenbergLoopGates(tau,
                                                         2, 2, 1, 2);
    for (size_t col = 1; col < Lx - 2; col++) {
      evolve_gates({0, col}) = gates_upper;
      evolve_gates({Ly - 2, col}) = gates_lower;
    }

    auto gates_left = GenerateSquareHeisenbergLoopGates(tau,
                                                        2, 2, 2, 1);
    auto gates_middle = GenerateSquareHeisenbergLoopGates(tau,
                                                          2, 2, 2, 2);
    auto gates_right = GenerateSquareHeisenbergLoopGates(tau,
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

  LoopGateT GenerateSquareHeisenbergLoopGates(
      const double tau, // imaginary time
      const size_t n0, const size_t n1, const size_t n2, const size_t n3  //bond share number
      //n0 bond is the upper horizontal bond in the loop
  ) {
    const std::vector<size_t> ns = {n0, n1, n2, n3};
    LoopGateT gates;
    for (size_t i = 0; i < 4; i++) {
      gates[i] = Tensor({vb_in, pb_in, pb_out, vb_out});
      Tensor &gate = gates[i];
      //Id
      gate({0, 0, 0, 0}) = 1.0;
      gate({0, 1, 1, 0}) = 1.0;
      //-s_z * tau
      gate({0, 0, 0, 1}) = -0.5 * tau / double(ns[i]);
      gate({0, 1, 1, 1}) = 0.5 * tau / double(ns[i]);
      //s_z
      gate({1, 0, 0, 0}) = 0.5;
      gate({1, 1, 1, 0}) = -0.5;

      //-s^+ * tau/2
      gate({0, 0, 1, 2}) = -1.0 * tau / double(ns[i]) / 2.0;
      //s^-
      gate({2, 1, 0, 0}) = 1.0;

      //-s^- * tau/2
      gate({0, 1, 0, 3}) = -1.0 * tau / double(ns[i]) / 2.0;
      //s^+
      gate({3, 0, 1, 0}) = 1.0;
    }
    return gates;
  }

  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
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

    GenerateSquareHeisenbergAllEvolveGates(tau0, evolve_gates0);
    GenerateSquareHeisenbergAllEvolveGates(tau1, evolve_gates1);
  }
};

TEST_F(HeisenbergLoopUpdate, Heisenberg) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);
  //initial peps as direct product state
  SquareLatticePEPS<TenElemT, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));

  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  //simple update, in boundary the setting of hamiltonian has edge errors
  SimpleUpdateExecutor<TenElemT, U1QN>
      *su_exe1 = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, U1QN>(SimpleUpdatePara(100, 0.1, 1, 4, 1e-10),
                                                                         peps0,
                                                                         dham_hei_nn);
  su_exe1->Execute();
  auto peps1 = su_exe1->GetPEPS();
  delete su_exe1;
  peps1.NormalizeAllTensor();
  //loop update

  ArnoldiParams arnoldi_params(1e-10, 100);
  double fet_tol = 1e-13;
  double fet_max_iter = 30;
  ConjugateGradientParams cg_params(100, 1e-10, 20, 0.0);

  FullEnvironmentTruncateParams fet_params(1, 4, 1e-10,
                                           fet_tol, fet_max_iter,
                                           cg_params);

  auto *loop_exe = new LoopUpdateExecutor<TenElemT, U1QN>(LoopUpdateTruncatePara(
                                                              arnoldi_params,
                                                              1e-7,
                                                              fet_params),
                                                          150,
                                                          tau0,
                                                          evolve_gates0,
                                                          peps1);

  loop_exe->Execute();

  auto peps2 = loop_exe->GetPEPS();
  delete loop_exe;

  //measure simple update state energy
  using Model = SquareSpinOneHalfXXZModelOBC;
  using WaveFunctionT = MCUpdateSquareNNExchange;
  size_t mc_samples = 1000;
  size_t mc_warmup = 100;
  std::string tps_path = "TPS_Heisenberg" + std::to_string(Lx) + "x" + std::to_string(Ly);
  Configuration measure_config2(Ly, Lx, OccupancyNum(std::vector<size_t>(2, Lx * Ly / 2)));
  MonteCarloParams measure_mc_params2(mc_samples, mc_warmup, 1, measure_config2, false); // not warmed up initially
  PEPSParams measure_peps_params2(BMPSTruncateParams<qlten::QLTEN_Double>(4, 8, 1e-10,
                                                   CompressMPSScheme::VARIATION2Site,
                                                   std::make_optional<double>(1e-14),
                                                   std::make_optional<size_t>(10)));
  MCMeasurementParams mc_measure_para(measure_mc_params2, measure_peps_params2);
  auto tps1 = TPS<TenElemT, U1QN>(peps1);
  auto sitps1 = SplitIndexTPS<TenElemT, U1QN>(tps1);
  sitps1.NormalizeAllSite();

  auto measure_executor1 =
      new MCPEPSMeasurer<TenElemT, U1QN, WaveFunctionT, Model>(mc_measure_para,
                                                                              sitps1,
                                                                              comm,
                                                                              Model());
  measure_executor1->Execute();
  measure_executor1->OutputEnergy();
  delete measure_executor1;

  //measure the loop update energy
  auto tps = TPS<TenElemT, U1QN>(peps2);
  auto sitps = SplitIndexTPS<TenElemT, U1QN>(tps);
  auto measure_executor =
      new MCPEPSMeasurer<TenElemT, U1QN, WaveFunctionT, Model>(mc_measure_para,
                                                                              sitps,
                                                                              comm,
                                                                              Model());
  measure_executor->Execute();
  measure_executor->OutputEnergy();
  delete measure_executor;

}

//Triangle Heisenberg
struct TriangleHeisenbergLoopUpdate : public testing::Test {
  SystemSizeParams params;  // Remove file dependency
  size_t Lx = params.Lx; //cols
  size_t Ly = params.Ly;

//#ifdef U1SYM
//  IndexT pb_out = IndexT({
//                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
//                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
//                         TenIndexDirType::OUT
//  );
//#else
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT
  );
//#endif
  IndexT pb_in = InverseIndex(pb_out);
  IndexT vb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 4)},
                         TenIndexDirType::OUT
  );
  IndexT vb_in = InverseIndex(vb_out);

  Tensor did = Tensor({pb_in, pb_out});
  Tensor dsz = Tensor({pb_in, pb_out});
  Tensor dsp = Tensor({pb_in, pb_out});
  Tensor dsm = Tensor({pb_in, pb_out});

  Tensor dham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
  Tensor dham_hei_tri;

  double tau0 = 0.1;
  double tau1 = 0.01;
  double tau2 = 0.001;
  // ED ground state energy = -7.709643309360509
  DuoMatrix<LoopGateT> evolve_gates0 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);
  DuoMatrix<LoopGateT> evolve_gates1 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);
  DuoMatrix<LoopGateT> evolve_gates2 = DuoMatrix<LoopGateT>(Ly - 1, Lx - 1);

  const MPI_Comm comm = MPI_COMM_WORLD;//mpi support for measurement
  void GenerateTriangleHeisenbergAllEvolveGates(
      const double tau, // imaginary time
      DuoMatrix<LoopGateT> &evolve_gates //output
  ) {
    //corner
    evolve_gates({0, 0}) = GenerateTriangleHeisenbergLoopGates(tau,
                                                               1, 2, 2, 1);
    evolve_gates({0, Lx - 2}) = GenerateTriangleHeisenbergLoopGates(tau,
                                                                    1, 1, 2, 2);
    evolve_gates({Ly - 2, 0}) = GenerateTriangleHeisenbergLoopGates(tau,
                                                                    2, 2, 1, 1);
    evolve_gates({Ly - 2, Lx - 2}) = GenerateTriangleHeisenbergLoopGates(tau,
                                                                         2, 1, 1, 2);

    auto gates_upper = GenerateTriangleHeisenbergLoopGates(tau,
                                                           1, 2, 2, 2);
    auto gates_lower = GenerateTriangleHeisenbergLoopGates(tau,
                                                           2, 2, 1, 2);
    for (size_t col = 1; col < Lx - 2; col++) {
      evolve_gates({0, col}) = gates_upper;
      evolve_gates({Ly - 2, col}) = gates_lower;
    }

    auto gates_left = GenerateTriangleHeisenbergLoopGates(tau,
                                                          2, 2, 2, 1);
    auto gates_middle = GenerateTriangleHeisenbergLoopGates(tau,
                                                            2, 2, 2, 2);
    auto gates_right = GenerateTriangleHeisenbergLoopGates(tau,
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

  LoopGateT GenerateTriangleHeisenbergLoopGates(
      const double tau, // imaginary time
      const size_t n0, const size_t n1, const size_t n2, const size_t n3  //bond share number
      //n0 bond is the upper horizontal bond in the loop
  ) {
    const std::vector<size_t> ns = {n0, n1, n2, n3};
    LoopGateT gates;
    gates[0] = Tensor({vb_out, pb_in, pb_out, vb_out});
    gates[1] = Tensor({vb_in, pb_in, pb_out, vb_out});
    gates[2] = Tensor({vb_in, pb_in, pb_out, vb_in});
    gates[3] = Tensor({vb_out, pb_in, pb_out, vb_in});
    for (size_t i = 0; i < 4; i++) {
      Tensor &gate = gates[i];
      //Id
      gate({0, 0, 0, 0}) = 1.0;
      gate({0, 1, 1, 0}) = 1.0;
      //-s_z * tau
      gate({0, 0, 0, 1}) = -0.5 * tau / double(ns[i]);
      gate({0, 1, 1, 1}) = 0.5 * tau / double(ns[i]);
      //s_z
      gate({1, 0, 0, 0}) = 0.5;
      gate({1, 1, 1, 0}) = -0.5;

      //-s^+ * tau/2
      gate({0, 0, 1, 2}) = -1.0 * tau / double(ns[i]) / 2.0;
      //s^-
      gate({2, 1, 0, 0}) = 1.0;

      //-s^- * tau/2
      gate({0, 1, 0, 3}) = -1.0 * tau / double(ns[i]) / 2.0;
      //s^+
      gate({3, 0, 1, 0}) = 1.0;
    }

    for (auto i : {0}) {
      Tensor &gate = gates[i];
      gate({1, 0, 0, 1}) = double(ns[3]);
      gate({1, 1, 1, 1}) = double(ns[3]);

      gate({2, 0, 0, 2}) = double(ns[3]);
      gate({2, 1, 1, 2}) = double(ns[3]);

      gate({3, 0, 0, 3}) = double(ns[3]);
      gate({3, 1, 1, 3}) = double(ns[3]);
    }
    return gates;
  }
  int mpi_size;
  int rank;
  void SetUp(void) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
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

    Tensor ham_hei_tri_terms[3];
    for (size_t i = 0; i < 3; i++) {
      ham_hei_tri_terms[i] = Tensor({pb_in, pb_out, pb_in, pb_out, pb_in, pb_out});
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

    GenerateTriangleHeisenbergAllEvolveGates(tau0, evolve_gates0);
    GenerateTriangleHeisenbergAllEvolveGates(tau1, evolve_gates1);
    GenerateTriangleHeisenbergAllEvolveGates(tau2, evolve_gates2);
  }
};

TEST_F(TriangleHeisenbergLoopUpdate, MultiThread) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);
  omp_set_num_threads(4);
  //initial peps as direct product state
  SquareLatticePEPS<TenElemT, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));

  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  //simple update, in boundary the setting of hamiltonian has edge errors
  SimpleUpdateExecutor<TenElemT, U1QN>
      *su_exe1 =
      new TriangleNNModelSquarePEPSSimpleUpdateExecutor<TenElemT, U1QN>(SimpleUpdatePara(100, 0.1, 1, 4, 1e-10),
                                                                        peps0,
                                                                        dham_hei_nn,
                                                                        dham_hei_tri);
  su_exe1->Execute();
  auto peps1 = su_exe1->GetPEPS();
  delete su_exe1;
  peps1.NormalizeAllTensor();
  //loop update
  ArnoldiParams arnoldi_params(1e-9 * tau0, 300);
  double fet_tol = 1e-12;
  double fet_max_iter = 30;
  ConjugateGradientParams cg_params(100, 1e-8, 20, 0.0);

  FullEnvironmentTruncateParams fet_params(1, 4, 1e-10,
                                           fet_tol, fet_max_iter,
                                           cg_params);

  auto *loop_exe = new LoopUpdateExecutor<TenElemT, U1QN>(LoopUpdateTruncatePara(
                                                              arnoldi_params,
                                                              1e-7,
                                                              fet_params),
                                                          100,
                                                          tau0,
                                                          evolve_gates0,
                                                          peps1);

  loop_exe->Execute();

  auto peps2 = loop_exe->GetPEPS();
  delete loop_exe;

  loop_exe = new LoopUpdateExecutor<TenElemT, U1QN>(LoopUpdateTruncatePara(
                                                        arnoldi_params,
                                                        1e-7,
                                                        fet_params),
                                                    100,
                                                    tau1,
                                                    evolve_gates1,
                                                    peps2);

  loop_exe->Execute();
  auto peps3 = loop_exe->GetPEPS();
  delete loop_exe;

  loop_exe = new LoopUpdateExecutor<TenElemT, U1QN>(LoopUpdateTruncatePara(
                                                        arnoldi_params,
                                                        1e-7,
                                                        fet_params),
                                                    100,
                                                    tau2,
                                                    evolve_gates2,
                                                    peps3);

  loop_exe->Execute();
  auto peps4 = loop_exe->GetPEPS();
  delete loop_exe;

  //measure simple update state energy

  using Model = SpinOneHalfTriHeisenbergSqrPEPS;
  using WaveFunctionT = MCUpdateSquareTNN3SiteExchange;
  size_t mc_samples = 1000;
  size_t mc_warmup = 100;
  Configuration measure_config3(Ly, Lx, OccupancyNum(std::vector<size_t>(2, Lx * Ly / 2)));
  MonteCarloParams measure_mc_params3(mc_samples, mc_warmup, 1, measure_config3, false); // not warmed up initially
  PEPSParams measure_peps_params3(BMPSTruncateParams<qlten::QLTEN_Double>(4, 8, 1e-10,
                                                   CompressMPSScheme::VARIATION2Site,
                                                   std::make_optional<double>(1e-14),
                                                   std::make_optional<size_t>(10)));
  MCMeasurementParams mc_measure_para(measure_mc_params3, measure_peps_params3);
  auto tps1 = TPS<TenElemT, U1QN>(peps1);
  auto sitps1 = SplitIndexTPS<TenElemT, U1QN>(tps1);

  auto measure_executor1 =
      new MCPEPSMeasurer<TenElemT, U1QN, WaveFunctionT, Model>(mc_measure_para,
                                                                              sitps1,
                                                                              comm,
                                                                              Model());
  measure_executor1->Execute();
  measure_executor1->OutputEnergy();
  delete measure_executor1;

  //measure the loop update energy
  auto tps = TPS<TenElemT, U1QN>(peps4);
  auto sitps = SplitIndexTPS<TenElemT, U1QN>(tps);
  auto measure_executor =
      new MCPEPSMeasurer<TenElemT, U1QN, WaveFunctionT, Model>(mc_measure_para,
                                                                              sitps,
                                                                              comm,
                                                                              Model());
  measure_executor->Execute();
  measure_executor->OutputEnergy();
  delete measure_executor;

}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  // params_file = argv[1]; // Removed file dependency
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}

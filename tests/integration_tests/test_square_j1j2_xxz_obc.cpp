// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-12-19
*
* Description: QuantumLiquids/PEPS project. Integration testing for J1-J2 XXZ Model with modern VMC optimization API.
*/

#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/api/vmc_api.h"
#include "../test_mpi_env.h"
#include "../utilities.h"

using namespace qlten;
using namespace qlpeps;

using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = TEN_ELEM_TYPE;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;

class J1J2XXZSystem : public MPITest {
protected:
  size_t Lx = 4;
  size_t Ly = 3; 
  size_t Dpeps = 6;
  
  double jz1 = 0.5;
  double jxy1 = 1;
  double jz2 = -0.2; // no frustration
  double jxy2 = -0.3;
  double energy_ed = -6.523925897312232;
  
  QNT qn0 = QNT();
  IndexT pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);
  
  Tensor ham_hei_nn;
  Tensor ham_hei_nnn;
  
  std::string model_name = "square_j1j2_xxz";
  std::string tps_path = GenTPSPath(model_name, Dpeps, Lx, Ly);
  
  // Modern VMC parameters (keep only SGD here)
  VMCPEPSOptimizerParams vmc_sgd_params = VMCPEPSOptimizerParams(
      OptimizerParamsBuilder()
          .SetMaxIterations(40)
          .SetLearningRate(0.1)
          .WithSGD()
          .Build(),
      MonteCarloParams(100, 100, 1,
                       Configuration(Ly, Lx,
                                     OccupancyNum({Lx * Ly / 2, Lx * Ly / 2})), // Sz = 0
                       false), // not warmed up initially
      PEPSParams(BMPSTruncateParams<qlten::QLTEN_Double>::SVD(6, 12, 1e-15)));

  MonteCarloParams measure_mc_params{1000, 1000, 1,
                                     Configuration(Ly, Lx,
                                                   OccupancyNum({Lx * Ly / 2, Lx * Ly / 2})), // Sz = 0
                                     false}; // not warmed up initially
  PEPSParams measure_peps_params{BMPSTruncateParams<qlten::QLTEN_Double>(Dpeps, 2 * Dpeps, 1e-15,
                                                  CompressMPSScheme::SVD_COMPRESS,
                                                  std::make_optional<double>(1e-14),
                                                  std::make_optional<size_t>(10))};
  MCMeasurementParams measure_para{measure_mc_params, measure_peps_params};
      
  void SetUp() override {
    MPITest::SetUp();
    
    ham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
    ham_hei_nn({0, 0, 0, 0}) = 0.25 * jz1;
    ham_hei_nn({1, 1, 1, 1}) = 0.25 * jz1;
    ham_hei_nn({1, 1, 0, 0}) = -0.25 * jz1;
    ham_hei_nn({0, 0, 1, 1}) = -0.25 * jz1;
    ham_hei_nn({0, 1, 1, 0}) = 0.5 * jxy1;
    ham_hei_nn({1, 0, 0, 1}) = 0.5 * jxy1;

    ham_hei_nnn = Tensor({pb_in, pb_out, pb_in, pb_out});
    ham_hei_nnn({0, 0, 0, 0}) = 0.25 * jz2;
    ham_hei_nnn({1, 1, 1, 1}) = 0.25 * jz2;
    ham_hei_nnn({1, 1, 0, 0}) = -0.25 * jz2;
    ham_hei_nnn({0, 0, 1, 1}) = -0.25 * jz2;
    ham_hei_nnn({0, 1, 1, 0}) = 0.5 * jxy2;
    ham_hei_nnn({1, 0, 0, 1}) = 0.5 * jxy2;
  }
  
  void RunSimpleUpdate() {
    if (rank == hp_numeric::kMPIMasterRank) {
      SquareLatticePEPS<TenElemT, QNT> peps0(pb_out, Ly, Lx);
      std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
      for (size_t y = 0; y < Ly; y++) {
        for (size_t x = 0; x < Lx; x++) {
          size_t sz_int = x + y;
          activates[y][x] = sz_int % 2;
        }
      }
      peps0.Initial(activates);

      SimpleUpdatePara update_para(1000, 0.1, 1, 4, 1e-15);
      auto su_exe = new SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>(
          update_para, peps0, ham_hei_nn, ham_hei_nnn);
      
      su_exe->Execute();
      
      // Save optimized TPS
      auto tps = qlpeps::ToTPS<TenElemT, QNT>(su_exe->GetPEPS());
      for (auto &ten : tps) {
        ten *= (1.0 / ten.GetMaxAbs());
      }
      SplitIndexTPS<TenElemT, QNT> sitps = ToSplitIndexTPS<TenElemT, QNT>(tps);
      sitps.Dump(tps_path);
      
      delete su_exe;
    }
  }
  
  template<typename ModelT, typename MCUpdaterT>
  void RunVMCOptimization(const ModelT& model, const VMCPEPSOptimizerParams& params) {
    MPI_Barrier(comm);
    
    SplitIndexTPS<TenElemT, QNT> tps(Ly, Lx);
    tps.Load(tps_path);

    size_t start_flop = flop;
    Timer vmc_timer("vmc");

    auto result = VmcOptimize<TenElemT, QNT, MCUpdaterT, ModelT>(
        params, tps, comm, model, MCUpdaterT{});

    size_t end_flop = flop;
    double elapsed_time = vmc_timer.PrintElapsed();
    double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
    std::cout << "VMC Gflops = " << Gflops / elapsed_time << std::endl;

    tps = result.state;
    
    // Save optimized TPS
    tps.Dump(tps_path);
  }
  
  template<typename ModelT, typename MCUpdaterT>
  void RunMCMeasurement(const ModelT& model) {
    MPI_Barrier(comm);
    
    SplitIndexTPS<TenElemT, QNT> tps(Ly, Lx);
    tps.Load(tps_path);

    size_t start_flop = flop;
    Timer mc_timer("mc");

    auto measure_result = MonteCarloMeasure<TenElemT, QNT, MCUpdaterT, ModelT>(
        tps, measure_para, comm, model, MCUpdaterT{});

    size_t end_flop = flop;
    double elapsed_time = mc_timer.PrintElapsed();
    double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
    std::cout << "MC Gflops = " << Gflops / elapsed_time << std::endl;

    auto [energy, en_err] = measure_result.energy;
    std::cout << "Measured energy: " << std::real(energy) << " ± " << en_err << std::endl;
    
    // Verify energy is close to expected
    EXPECT_NEAR(std::real(energy), energy_ed, 0.01);
  }
};

TEST_F(J1J2XXZSystem, SimpleUpdate) {
  RunSimpleUpdate();
}

TEST_F(J1J2XXZSystem, StochasticGradientOpt) {
  using Model = SquareSpinOneHalfJ1J2XXZModelOBC;
  Model j1j2_model(jz1, jxy1, jz2, jxy2, 0);
  
  // VMC optimization with Stochastic Gradient Descent
  RunVMCOptimization<Model, MCUpdateSquareNNExchange>(j1j2_model, vmc_sgd_params);
  
  // Monte Carlo measurement
  RunMCMeasurement<Model, MCUpdateSquareNNExchange>(j1j2_model);
}

#if 0
TEST_F(J1J2XXZSystem, LBFGSLineSearch) {
  using Model = SquareSpinOneHalfJ1J2XXZModelOBC;
  Model j1j2_model(jz1, jxy1, jz2, jxy2, 0);
  
  // VMC optimization with L-BFGS (replaces NaturalGradientLineSearch)
  RunVMCOptimization<Model, MCUpdateSquareTNN3SiteExchange>(j1j2_model, vmc_lbfgs_params);
  
  // Monte Carlo measurement  
  RunMCMeasurement<Model, MCUpdateSquareTNN3SiteExchange>(j1j2_model);
}
#endif

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}

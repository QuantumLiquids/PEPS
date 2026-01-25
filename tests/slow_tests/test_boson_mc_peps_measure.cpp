// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-06
*
* Description: QuantumLiquids/PEPS project. Unittests for Monte-Carlo Measurement for finite-size PEPS.
*
* Note: This test is work for 5 mins
*/
/**
 * @brief Monte Carlo measurement test for square Heisenberg model
 * 
 * This is a slow integration test that typically takes 5 minutes to complete.
 * It performs full Monte Carlo sampling with 1000 samples and 1000 warmup steps
 * on a 4×4 lattice with bond dimension D=8.
 * 
 * Expected energy: -9.18912 ± tolerance
 * 
 * @note This test is disabled by default. Use --gtest_also_run_disabled_tests
 *       or set RUN_SLOW_TESTS=ON in cmake to run it.
 */
#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_xxz_obc.h"
#include "qlmps/case_params_parser.h"
#include "../test_mpi_env.h"
using namespace qlten;
using namespace qlpeps;

#if TEN_ELEM_TYPE_NUM == 1
std::string data_type_in_file_name = "Double";
#elif TEN_ELEM_TYPE_NUM == 2
std::string data_type_in_file_name = "Complex";
#else
#error "Unexpected TEN_ELEM_TYPE_NUM"
#endif

#include <filesystem>

struct SqrHeiMCPEPS : MPITest {
  using QNT = qlten::special_qn::TrivialRepQN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using QNSctVecT = QNSectorVec<QNT>;
  using TenElemT = TEN_ELEM_TYPE;
  using TPSSampleFlipT = MCUpdateSquareNNExchange;

  size_t Lx = 4;
  size_t Ly = 4;
  size_t N = Lx * Ly;
  size_t Dpeps = 8;
  double E0_ED = -9.189207065192933;
  double e0_state = -9.18912;
  Configuration config{Ly, Lx, OccupancyNum(std::vector<size_t>(2, N / 2))};
  MonteCarloParams mc_params{1000, 1000, 1, config, false}; // not warmed up initially
  PEPSParams peps_params{BMPSTruncateParams<qlten::QLTEN_Double>(Dpeps, 2 * Dpeps, 1e-15,
                                          CompressMPSScheme::SVD_COMPRESS,
                                          std::make_optional<double>(1e-14),
                                          std::make_optional<size_t>(10))};
  MCMeasurementParams para{mc_params, peps_params};

  void SetUp(void) {
    MPITest::SetUp();
    qlten::hp_numeric::SetTensorManipulationThreads(1);

    // New API: TPS path is handled by the caller, not stored in parameters
  }
};

TEST_F(SqrHeiMCPEPS, MeasureHeisenberg) {
  using Model = SquareSpinOneHalfXXZModelOBC;

  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "Starting Monte Carlo measurement test on 4 by 4 Heisenberg Model (expected time: ~5 minutes)..." << std::endl;
  }
  // Load TPS explicitly - this is the new API pattern
  std::string tps_path = (std::filesystem::path(TEST_SOURCE_DIR) / "slow_tests/test_data" / ("tps_square_heisenberg4x4D8" + data_type_in_file_name)).string();
  auto sitps = SplitIndexTPS<TenElemT, QNT>(Ly, Lx);
  sitps.Load(tps_path);
  
  auto executor = new MCPEPSMeasurer<TenElemT, QNT, TPSSampleFlipT, Model>(sitps,
                                                                                          para,
                                                                                          comm);
  executor->Execute();

  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "Measurement completed, analyzing results..." << std::endl;
  }

  auto [energy, en_err] = executor->OutputEnergy();
  auto energy_estimate = executor->GetEnergyEstimate();

  if (rank == hp_numeric::kMPIMasterRank && mpi_size > 1) {
    //Justify whether as expected
    EXPECT_NEAR(std::real(energy), e0_state, 1.5 * en_err);
  } else {
    EXPECT_NEAR(std::real(energy), e0_state, 0.1); //TODO: bin statistic even for ONE Markov chain
  }
  delete executor;
}

/*

struct FileParams : public CaseParamsParserBasic {
  FileParams(const char *f) : CaseParamsParserBasic(f) {
    Lx = ParseInt("Lx");
    Ly = ParseInt("Ly");
    D = ParseInt("D");
    Db_min = ParseInt("Dbmps_min");
    Db_max = ParseInt("Dbmps_max");
    MC_samples = ParseInt("MC_samples");
    WarmUp = ParseInt("WarmUp");
    size_t update_times = ParseInt("UpdateNum");
  }

  size_t Ly;
  size_t Lx;
  size_t D;
  size_t Db_min;
  size_t Db_max;
  size_t MC_samples;
  size_t WarmUp;
};

// Test spin systems
struct SpinSystemMCPEPS : public testing::Test {
  using U1QN = qlten::special_qn::U1QN;
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using QNSctVecT = QNSectorVec<U1QN>;
  using TenElemT = TEN_ELEM_TYPE;
  using TPSSampleNNFlipT = MCUpdateSquareNNExchange;

  FileParams params = FileParams(params_file);
  size_t Lx = params.Lx; //cols
  size_t Ly = params.Ly;
  size_t N = Lx * Ly;

  Configuration measurement_config{Ly, Lx, OccupancyNum(std::vector<size_t>(2, N / 2))};
  MonteCarloParams measurement_mc_params{params.MC_samples, params.WarmUp, 1, measurement_config, false}; // not warmed up initially
  PEPSParams measurement_peps_params{BMPSTruncateParams<qlten::QLTEN_Double>::Variational2Site(params.Db_min, params.Db_max, 1e-10, 1e-14, 10)};
  MCMeasurementParams mc_measurement_para{measurement_mc_params, measurement_peps_params};

  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }

    qlten::hp_numeric::SetTensorManipulationThreads(1);
    mc_measurement_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(params.D);
  }
};

TEST_F(SpinSystemMCPEPS, TriHeisenbergD4) {
  using Model = SpinOneHalfTriHeisenbergSqrPEPS;
  MCPEPSMeasurer<TenElemT, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  Model triangle_hei_solver;
  // Load TPS explicitly - this is the new API pattern
  std::string tps_path = "vmc_tps_tri_heisenbergD" + std::to_string(params.D);
  auto sitps = SplitIndexTPS<TenElemT, U1QN>(Ly, Lx);
  sitps.Load(tps_path);

  executor = new MCPEPSMeasurer<TenElemT, U1QN, TPSSampleNNFlipT, Model>(sitps,
                                                                                        mc_measurement_para,
                                                                                        comm, triangle_hei_solver);

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemMCPEPS, TriJ1J2HeisenbergD4) {
  using Model = SpinOneHalfTriJ1J2HeisenbergSqrPEPS<TenElemT, U1QN>;
  MCPEPSMeasurer<TenElemT, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  Model trianglej1j2_hei_solver(0.2);
  mc_measurement_para.wavefunction_path = "vmc_tps_tri_heisenbergD" + std::to_string(params.D);
  executor = new MCPEPSMeasurer<TenElemT, U1QN, TPSSampleNNFlipT, Model>(mc_measurement_para,
                                                                                        Ly,
                                                                                        Lx,
                                                                                        comm,
                                                                                        trianglej1j2_hei_solver);
  executor->Execute();
  delete executor;
}
 */

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}

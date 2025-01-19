// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-06
*
* Description: QuantumLiquids/PEPS project. Unittests for Monte-Carlo Measurement for finite-size PEPS.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_measurement.h"
#include "qlpeps/algorithm/vmc_update/wave_function_component_classes/wave_function_component_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlmps/case_params_parser.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;
using TenElemT = TEN_ELEM_TYPE;
using TPSSampleNNFlipT = SquareTPSSampleNNExchange<TenElemT, U1QN>;

using qlmps::CaseParamsParserBasic;

char *params_file;

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
  FileParams params = FileParams(params_file);
  size_t Lx = params.Lx; //cols
  size_t Ly = params.Ly;
  size_t N = Lx * Ly;

  MCMeasurementPara mc_measurement_para = MCMeasurementPara(
      BMPSTruncatePara(params.Db_min, params.Db_max, 1e-10,
                       CompressMPSScheme::VARIATION2Site,
                       std::make_optional<double>(1e-14),
                       std::make_optional<size_t>(10)),
      params.MC_samples, params.WarmUp, 1,
      std::vector<size_t>(2, N / 2),
      Ly, Lx);

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
  using Model = SpinOneHalfTriHeisenbergSqrPEPS<TenElemT, U1QN>;
  MonteCarloMeasurementExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  Model triangle_hei_solver;
  mc_measurement_para.wavefunction_path = "vmc_tps_tri_heisenbergD" + std::to_string(params.D);

  executor = new MonteCarloMeasurementExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(mc_measurement_para,
                                                                                        Ly, Lx,
                                                                                        comm, triangle_hei_solver);

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemMCPEPS, TriJ1J2HeisenbergD4) {
  using Model = SpinOneHalfTriJ1J2HeisenbergSqrPEPS<TenElemT, U1QN>;
  MonteCarloMeasurementExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  Model trianglej1j2_hei_solver(0.2);
  mc_measurement_para.wavefunction_path = "vmc_tps_tri_heisenbergD" + std::to_string(params.D);
  executor = new MonteCarloMeasurementExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(mc_measurement_para,
                                                                                        Ly,
                                                                                        Lx,
                                                                                        comm,
                                                                                        trianglej1j2_hei_solver);
  executor->Execute();
  delete executor;
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  params_file = argv[1];
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}


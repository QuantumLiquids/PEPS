/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-19
*
* Description: QuantumLiquids/PEPS project. Unittests for PEPS Monte-Carlo measurement in Fermion model.
*/


#include "gtest/gtest.h"
#include "qlten/qlten.h"

#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"

using namespace qlten;
using namespace qlpeps;


using qlten::special_qn::fZ2QN;
using TenElemT = TEN_ELEM_TYPE;

struct FileParams {
  // Parameters embedded directly in code (from test_params.json)
  size_t Ly = 4;
  size_t Lx = 4;
  size_t D = 4;
  size_t Db_min = 4;
  size_t Db_max = 4;
  size_t MC_samples = 100;
  size_t WarmUp = 100;
  double Tau0 = 0.1;
  size_t Steps = 100;
  bool Continue_from_VMC = false;
  double StepLengthFirst = 0.001;
  double StepLengthDecrease = 0.0;
  size_t UpdateNum = 10;
};

struct Z2SpinlessFreeFermionTools : public testing::Test {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using DTensor = QLTensor<TenElemT, fZ2QN>;

  FileParams file_params;
  size_t Lx = file_params.Lx; //cols
  size_t Ly = file_params.Ly;
  size_t N = Lx * Ly;

  double t = 1.0;
  fZ2QN qn0 = fZ2QN(0);
  IndexT loc_phy_ket = IndexT({QNSctT(fZ2QN(1), 1),  // |1> occupied
                               QNSctT(fZ2QN(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  Configuration mc_config{Ly, Lx, OccupancyNum({N / 2, N / 2})};
  MonteCarloParams mc_params{file_params.MC_samples, file_params.WarmUp, 1, mc_config, false}; // not warmed up initially
  PEPSParams peps_params{BMPSTruncatePara(file_params.Db_min, file_params.Db_max, 1e-10,
                                          CompressMPSScheme::SVD_COMPRESS,
                                          std::make_optional<double>(1e-14),
                                          std::make_optional<size_t>(10))};
  MCMeasurementParams mc_measurement_para{mc_params, peps_params};

  std::string simple_update_peps_path = "peps_spinless_free_fermion_half_filling";
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
  }
};

TEST_F(Z2SpinlessFreeFermionTools, MonteCarloMeasureNNUpdate) {
  SquareSpinlessFermion spinless_fermion_solver(1, 0, 0);

  SquareLatticePEPS<TenElemT, fZ2QN> peps(loc_phy_ket, Ly, Lx);
  peps.Load(simple_update_peps_path);
  auto tps = TPS<TenElemT, fZ2QN>(peps);
  auto sitps = SplitIndexTPS<TenElemT, fZ2QN>(tps);
  auto measure_executor =
      new MonteCarloMeasurementExecutor<TenElemT, fZ2QN, MCUpdateSquareNNExchange, SquareSpinlessFermion>(
          sitps,
          mc_measurement_para,
          comm,
          spinless_fermion_solver);

  measure_executor->Execute();
  measure_executor->OutputEnergy();
  delete measure_executor;
}

TEST_F(Z2SpinlessFreeFermionTools, MonteCarloMeasure3SiteUpdate) {
  using Model = SquareSpinlessFermion;
  Model spinless_fermion_solver(1, 0, 0);

  SquareLatticePEPS<TenElemT, fZ2QN> peps(loc_phy_ket, Ly, Lx);
  peps.Load(simple_update_peps_path);
  auto tps = TPS<TenElemT, fZ2QN>(peps);
  auto sitps = SplitIndexTPS<TenElemT, fZ2QN>(tps);
  auto measure_executor =
      new MonteCarloMeasurementExecutor<TenElemT, fZ2QN, MCUpdateSquareTNN3SiteExchange, Model>(sitps,
                                                                                                mc_measurement_para,
                                                                                                comm,
                                                                                                spinless_fermion_solver);

  measure_executor->Execute();
  measure_executor->OutputEnergy();
  delete measure_executor;
}

const bool SquaretJModelMixIn::enable_sc_measurement = false;
struct Z2tJModelTools : public testing::Test {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using DTensor = QLTensor<TenElemT, fZ2QN>;

  FileParams file_params;
  size_t Lx = file_params.Lx; //cols
  size_t Ly = file_params.Ly;
  size_t N = Lx * Ly;
  double t = 1.0;
  double J = 0.3;
  double doping = 0.125;
  size_t hole_num = size_t(double(N) * doping);

  IndexT loc_phy_ket = IndexT({QNSctT(fZ2QN(1), 2), // |up>, |down>
                               QNSctT(fZ2QN(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  Configuration mc_config2{Ly, Lx, OccupancyNum({(N - hole_num) / 2, (N - hole_num) / 2, hole_num})};
  MonteCarloParams mc_params2{file_params.MC_samples, file_params.WarmUp, 1, mc_config2, false}; // not warmed up initially
  PEPSParams peps_params2{BMPSTruncatePara(file_params.Db_min, file_params.Db_max, 1e-10,
                                           CompressMPSScheme::SVD_COMPRESS,
                                           std::make_optional<double>(1e-14),
                                           std::make_optional<size_t>(10))};
  MCMeasurementParams mc_measurement_para{mc_params2, peps_params2};

  std::string simple_update_peps_path = "peps_tj_doping0.125";
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
    // New API: TPS path is handled by the caller, not stored in parameters
  }
};

TEST_F(Z2tJModelTools, MonteCarloMeasureNNUpdate) {
  SquaretJNNModel tj_solver(t, J, 0);

  SquareLatticePEPS<TenElemT, fZ2QN> peps(loc_phy_ket, Ly, Lx);
  peps.Load("peps_tj_doping0.125");
  auto tps = TPS<TenElemT, fZ2QN>(peps);
  auto sitps = SplitIndexTPS<TenElemT, fZ2QN>(tps);
  auto measure_executor =
      new MonteCarloMeasurementExecutor<TenElemT, fZ2QN, MCUpdateSquareNNExchange, SquaretJNNModel>(sitps,
                                                                                                    mc_measurement_para,
                                                                                                    comm,
                                                                                                    tj_solver);

  measure_executor->Execute();
  measure_executor->OutputEnergy();
  delete measure_executor;
}

TEST_F(Z2tJModelTools, MonteCarloMeasure3SiteUpdate) {
  using Model = SquaretJNNModel;
  Model tj_solver(t, J, 0);

  SquareLatticePEPS<TenElemT, fZ2QN> peps(loc_phy_ket, Ly, Lx);
  peps.Load("peps_tj_doping0.125");
  auto tps = TPS<TenElemT, fZ2QN>(peps);
  auto sitps = SplitIndexTPS<TenElemT, fZ2QN>(tps);
  auto measure_executor =
      new MonteCarloMeasurementExecutor<TenElemT, fZ2QN, MCUpdateSquareTNN3SiteExchange, Model>(sitps,
                                                                                                mc_measurement_para,
                                                                                                comm,
                                                                                                tj_solver);

  measure_executor->Execute();
  measure_executor->OutputEnergy();
  delete measure_executor;
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
}

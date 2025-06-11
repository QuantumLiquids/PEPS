/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-19
*
* Description: QuantumLiquids/PEPS project. Unittests for PEPS Monte-Carlo measurement in Fermion model.
*/


#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/algorithm/vmc_update/configuration_update_strategies/monte_carlo_sweep_updater_all.h"

using namespace qlten;
using namespace qlpeps;

using qlmps::CaseParamsParserBasic;
char *params_file;

using qlten::special_qn::fZ2QN;
using TenElemT = TEN_ELEM_TYPE;

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

struct Z2SpinlessFreeFermionTools : public testing::Test {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using DTensor = QLTensor<TenElemT, fZ2QN>;

  FileParams file_params = FileParams(params_file);
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

  MCMeasurementPara mc_measurement_para = MCMeasurementPara(
      BMPSTruncatePara(file_params.Db_min, file_params.Db_max, 1e-10,
                       CompressMPSScheme::SVD_COMPRESS,
                       std::make_optional<double>(1e-14),
                       std::make_optional<size_t>(10)),
      file_params.MC_samples, file_params.WarmUp, 1,
      {N / 2, N / 2},
      Ly, Lx);

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
    mc_measurement_para.wavefunction_path =
        "tps_spinless_fermion_half_filling_D" + std::to_string(file_params.D);
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
          mc_measurement_para,
          sitps,
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
      new MonteCarloMeasurementExecutor<TenElemT, fZ2QN, MCUpdateSquareTNN3SiteExchange, Model>(mc_measurement_para,
                                                                                                sitps,
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

  FileParams file_params = FileParams(params_file);
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
  MCMeasurementPara mc_measurement_para = MCMeasurementPara(
      BMPSTruncatePara(file_params.Db_min, file_params.Db_max, 1e-10,
                       CompressMPSScheme::SVD_COMPRESS,
                       std::make_optional<double>(1e-14),
                       std::make_optional<size_t>(10)),
      file_params.MC_samples, file_params.WarmUp, 1,
      {(N - hole_num) / 2, (N - hole_num) / 2, hole_num},
      Ly, Lx);

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
    mc_measurement_para.wavefunction_path =
        "tps_tJ_doping" + std::to_string(doping) + "_D" + std::to_string(file_params.D);
  }
};

TEST_F(Z2tJModelTools, MonteCarloMeasureNNUpdate) {
  SquaretJNNModel tj_solver(t, J, false, 0);

  SquareLatticePEPS<TenElemT, fZ2QN> peps(loc_phy_ket, Ly, Lx);
  peps.Load("peps_tj_doping0.125");
  auto tps = TPS<TenElemT, fZ2QN>(peps);
  auto sitps = SplitIndexTPS<TenElemT, fZ2QN>(tps);
  auto measure_executor =
      new MonteCarloMeasurementExecutor<TenElemT, fZ2QN, MCUpdateSquareNNExchange, SquaretJNNModel>(mc_measurement_para,
                                                                                                    sitps,
                                                                                                    comm,
                                                                                                    tj_solver);

  measure_executor->Execute();
  measure_executor->OutputEnergy();
  delete measure_executor;
}

TEST_F(Z2tJModelTools, MonteCarloMeasure3SiteUpdate) {
  using Model = SquaretJNNModel;
  Model tj_solver(t, J, false, 0);

  SquareLatticePEPS<TenElemT, fZ2QN> peps(loc_phy_ket, Ly, Lx);
  peps.Load("peps_tj_doping0.125");
  auto tps = TPS<TenElemT, fZ2QN>(peps);
  auto sitps = SplitIndexTPS<TenElemT, fZ2QN>(tps);
  auto measure_executor =
      new MonteCarloMeasurementExecutor<TenElemT, fZ2QN, MCUpdateSquareTNN3SiteExchange, Model>(mc_measurement_para,
                                                                                                sitps,
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
  params_file = argv[1];
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
}

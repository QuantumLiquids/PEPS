/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-19
*
* Description: QuantumLiquids/PEPS project. Unittests for PEPS Monte-Carlo measurement in Fermion model.
*/


#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/qlpeps.h"
#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

std::string GenTPSPath(std::string model_name, size_t Dmax, size_t Lx, size_t Ly) {
#if TEN_ELEM_TYPE == QLTEN_Double
  return "dtps_" + model_name + "_D" + std::to_string(Dmax) + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#elif TEN_ELEM_TYPE == QLTEN_Double
  return "ztps_" + model_name + "_D" + std::to_string(Dmax)  + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#else
#error "Unexpected TEN_ELEM_TYPE"
#endif
}

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
    step_len = std::vector<double>(update_times);
    Continue_from_VMC = ParseBool("Continue_from_VMC");
    if (update_times > 0) {
      step_len[0] = ParseDouble("StepLengthFirst");
      double step_len_change = ParseDouble("StepLengthDecrease");
      for (size_t i = 1; i < update_times; i++) {
        step_len[i] = step_len[0] - i * step_len_change;
      }
    }
  }

  size_t Ly;
  size_t Lx;
  size_t D;
  size_t Db_min;
  size_t Db_max;
  size_t MC_samples;
  size_t WarmUp;
  bool Continue_from_VMC;
  std::vector<double> step_len;
};

struct Z2SpinlessFreeFermionTools : public MPITest {
  using QNT = qlten::special_qn::fZ2QN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using QNSctVecT = QNSectorVec<QNT>;
  using TenElemT = QLTEN_Double;
  using Tensor = QLTensor<TenElemT, QNT>;
  using TPSSampleNNFlipT = MCUpdateSquareNNExchange;

  size_t Lx = 4; //cols
  size_t Ly = 3;
  size_t Dpeps = 4;  // hope it can be easy
  size_t ele_num = 4;

  size_t N = Lx * Ly;
  double t = 1.0;
  double mu = -0.707107;
  QNT qn0 = QNT(0);
  IndexT loc_phy_ket = IndexT({QNSctT(QNT(1), 1),  // |1> occupied
                               QNSctT(QNT(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  VMCOptimizePara optimize_para =
      VMCOptimizePara(BMPSTruncatePara(Dpeps, Dpeps * 2,
                                       1e-15, CompressMPSScheme::SVD_COMPRESS,
                                       std::make_optional<double>(1e-14),
                                       std::make_optional<size_t>(10)),
                      1000, 1000, 1,
                      std::vector<size_t>{8, 4},
                      Ly, Lx,
                      std::vector<double>(40, 0.2),
                      StochasticReconfiguration,
                      ConjugateGradientParams(100, 1e-4, 20, 0.01));

  std::string model_name = "spinless_free_fermion";
  std::string tps_path = GenTPSPath(model_name, Dpeps, Lx, Ly);

  void SetUp(void) {
    MPITest::SetUp();
    qlten::hp_numeric::SetTensorManipulationThreads(1);
    optimize_para.wavefunction_path = tps_path;
  }
};

TEST_F(Z2SpinlessFreeFermionTools, VariationalMonteCarloUpdate) {
  using Model = SquareSpinlessFreeFermion;
  Model spinless_fermion_solver;

  SplitIndexTPS<TenElemT, QNT> tps(Ly, Lx);
  tps.Load(tps_path);

  VMCPEPSExecutor<QLTEN_Double, QNT, TPSSampleNNFlipT, Model> *executor(nullptr);

  executor =
      new VMCPEPSExecutor<QLTEN_Double, QNT, TPSSampleNNFlipT, Model>(optimize_para,
                                                                      tps,
                                                                      comm,
                                                                      spinless_fermion_solver);

  executor->Execute();



  delete executor;
}
/*
struct Z2tJModelTools : public testing::Test {
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using Tensor = QLTensor<QLTEN_Double, fZ2QN>;
  using TPSSampleNNFlipT = MCUpdateSquareNNExchange;

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
  VMCOptimizePara optimize_para =
      VMCOptimizePara(BMPSTruncatePara(file_params.Db_min, file_params.Db_max,
                                       1e-15, CompressMPSScheme::SVD_COMPRESS,
                                       std::make_optional<double>(1e-14),
                                       std::make_optional<size_t>(10)),
                      file_params.MC_samples, file_params.WarmUp, 1,
                      {(N - hole_num) / 2, (N - hole_num) / 2, hole_num},
                      Ly, Lx,
                      file_params.step_len,
                      StochasticReconfiguration,
                      ConjugateGradientParams(100, 1e-4, 20, 0.01));

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
    optimize_para.wavefunction_path =
        "tps_tJ_doping" + std::to_string(doping) + "_D" + std::to_string(file_params.D);
  }
};

TEST_F(Z2tJModelTools, VariationalMonteCarloUpdate) {
  using Model = SquaretJModel;
  Model tj_solver(t, J, false, 0);

  VMCPEPSExecutor<QLTEN_Double, fZ2QN, TPSSampleNNFlipT, Model> *executor(nullptr);

  if (file_params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<QLTEN_Double, fZ2QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                                 Ly, Lx,
                                                                                 comm,
                                                                                 tj_solver);

  } else {
    SquareLatticePEPS<QLTEN_Double, fZ2QN> peps(loc_phy_ket, Ly, Lx);
    peps.Load(simple_update_peps_path);
    auto tps = TPS<QLTEN_Double, fZ2QN>(peps);
    auto sitps = SplitIndexTPS<QLTEN_Double, fZ2QN>(tps);

    executor =
        new VMCPEPSExecutor<QLTEN_Double, fZ2QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                          sitps,
                                                                          comm,
                                                                          tj_solver);

  }
  executor->Execute();
  delete executor;
}
*/
int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}

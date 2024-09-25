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

using namespace qlten;
using namespace qlpeps;

using qlmps::CaseParamsParserBasic;
char *params_file;

using qlten::special_qn::fZ2QN;

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

struct Z2tJModelTools : public testing::Test {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using DTensor = QLTensor<QLTEN_Double, fZ2QN>;
  using ZTensor = QLTensor<QLTEN_Complex, fZ2QN>;
  using TPSSampleNNFlipT = SquareTPSSampleNNExchange<QLTEN_Double, fZ2QN>;

  FileParams file_params = FileParams(params_file);
  size_t Lx = file_params.Lx; //cols
  size_t Ly = file_params.Ly;
  size_t N = Lx * Ly;
  double t = 1.0;
  double J = 0.3;
  double doping = 0.125;
  size_t hole_num = size_t(double(N) * doping);

  IndexT pb_out = IndexT({QNSctT(fZ2QN(1), 2), // |up>, |down>
                          QNSctT(fZ2QN(0), 1)}, // |0> empty state
                         TenIndexDirType::OUT
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
  boost::mpi::communicator world;
  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }

    qlten::hp_numeric::SetTensorManipulationThreads(1);
    optimize_para.wavefunction_path =
        "tps_tJ_doping" + std::to_string(doping) + "_D" + std::to_string(file_params.D);
  }
};

TEST_F(Z2tJModelTools, MonteCarloMeasure) {
  using Model = SquaretJModel<QLTEN_Double, fZ2QN>;
  Model tj_solver(t, J);

  VMCPEPSExecutor<QLTEN_Double, fZ2QN, TPSSampleNNFlipT, Model> *executor(nullptr);

  if (file_params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<QLTEN_Double, fZ2QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                                 Ly, Lx,
                                                                                 world,
                                                                                 tj_solver);

  } else {
    SquareLatticePEPS<QLTEN_Double, fZ2QN> peps(pb_out, Ly, Lx);
    peps.Load(simple_update_peps_path);
    auto tps = TPS<QLTEN_Double, fZ2QN>(peps);
    auto sitps = SplitIndexTPS<QLTEN_Double, fZ2QN>(tps);

    executor =
        new VMCPEPSExecutor<QLTEN_Double, fZ2QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                          sitps,
                                                                          world,
                                                                          tj_solver);

  }
  executor->Execute();
  delete executor;
}

int main(int argc, char *argv[]) {
  boost::mpi::environment env;
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  params_file = argv[1];
  return RUN_ALL_TESTS();
}

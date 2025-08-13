// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-12-19
*
* Description: QuantumLiquids/PEPS project. Integration testing for J1-J2 XXZ Model with VMC optimization.
*/

#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "integration_test_framework.h"

using namespace qlten;
using namespace qlpeps;

using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = TEN_ELEM_TYPE;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;

class J1J2XXZSystem : public IntegrationTestFramework<QNT> {
protected:
  double jz1 = 0.5;
  double jxy1 = 1;
  double jz2 = -0.2; // no frustration
  double jxy2 = -0.3;
  
  Tensor ham_hei_nn;
  Tensor ham_hei_nnn;
  
  void SetUpIndices() override {
    pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
    pb_in = InverseIndex(pb_out);
  }
  
  void SetUpHamiltonians() override {
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
  
  void SetUpParameters() override {
    model_name = "square_j1j2_xxz";
    energy_ed = -6.523925897312232;
    
    optimize_para = VMCOptimizePara(
        BMPSTruncatePara(6, 12, 1e-15,
                         CompressMPSScheme::SVD_COMPRESS,
                         std::make_optional<double>(1e-14),
                         std::make_optional<size_t>(10)),
        100, 100, 1,
        std::vector<size_t>(2, Lx * Ly / 2),
        Ly, Lx,
        std::vector<double>(40, 0.3),
        StochasticReconfiguration,
        ConjugateGradientParams(100, 1e-5, 20, 0.001));
    
    measure_para = MCMeasurementPara(
        BMPSTruncatePara(Dpeps, 2 * Dpeps, 1e-15,
                         CompressMPSScheme::SVD_COMPRESS,
                         std::make_optional<double>(1e-14),
                         std::make_optional<size_t>(10)),
        1000, 1000, 1,
        std::vector<size_t>(2, Lx * Ly / 2),
        Ly, Lx);
  }
};

TEST_F(J1J2XXZSystem, SimpleUpdate) {
  if (rank == kMPIMasterRank) {
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
    
    RunSimpleUpdate(su_exe);
    delete su_exe;
  }
}

TEST_F(J1J2XXZSystem, ZeroUpdate) {
  using Model = SquareSpinOneHalfJ1J2XXZModel;
  Model j1j2_model(jz1, jxy1, jz2, jxy2, 0);
  RunZeroUpdateTest<Model, MCUpdateSquareTNN3SiteExchange>(j1j2_model);
}

TEST_F(J1J2XXZSystem, StochasticReconfigurationOpt) {
  using Model = SquareSpinOneHalfJ1J2XXZModel;
  Model j1j2_model(jz1, jxy1, jz2, jxy2, 0);
  
  // VMC optimization
  RunVMCOptimization<Model, MCUpdateSquareTNN3SiteExchange>(j1j2_model);
  
  // Monte Carlo measurement
  RunMCMeasurement<Model, MCUpdateSquareTNN3SiteExchange>(j1j2_model);
}

TEST_F(J1J2XXZSystem, StochasticGradientOpt) {
  using Model = SquareSpinOneHalfJ1J2XXZModel;
  Model j1j2_model(jz1, jxy1, jz2, jxy2, 0);
  
  // Change to stochastic gradient
  optimize_para.update_scheme = StochasticGradient;
  optimize_para.step_lens = std::vector<double>(40, 0.1);
  
  // VMC optimization
  RunVMCOptimization<Model, MCUpdateSquareNNExchange>(j1j2_model);
  
  // Monte Carlo measurement
  RunMCMeasurement<Model, MCUpdateSquareNNExchange>(j1j2_model);
}

TEST_F(J1J2XXZSystem, NaturalGradientLineSearch) {
  using Model = SquareSpinOneHalfJ1J2XXZModel;
  Model j1j2_model(jz1, jxy1, jz2, jxy2, 0);
  
  // Change to natural gradient line search
  optimize_para.update_scheme = NaturalGradientLineSearch;
  optimize_para.cg_params = ConjugateGradientParams(100, 1e-4, 20, 0.01);
  
  // VMC optimization
  RunVMCOptimization<Model, MCUpdateSquareTNN3SiteExchange>(j1j2_model);
  
  // Monte Carlo measurement
  RunMCMeasurement<Model, MCUpdateSquareTNN3SiteExchange>(j1j2_model);
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
} 
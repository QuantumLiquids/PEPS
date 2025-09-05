// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-02-27
*
* Description: QuantumLiquids/PEPS project. API smoke tests using wrappers
*              VmcOptimize and MonteCarloMeasure on 2x2 OBC transverse Ising
*              at h=0 (reduces to classical Ising).
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/api/vmc_api.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"

#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

static inline double ClassicalIsing2x2OBC_GroundEnergy() { return -4.0; }

class APISmoke2x2 : public MPITest {
protected:
  using QNT = qlten::special_qn::TrivialRepQN;
  using TenElemT = TEN_ELEM_TYPE;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using Tensor = QLTensor<TenElemT, QNT>;

  size_t Lx = 2;
  size_t Ly = 2;

  IndexT pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  void SetUp() override {
    MPITest::SetUp();
    qlten::hp_numeric::SetTensorManipulationThreads(1);
  }
};

TEST_F(APISmoke2x2, OptimizeThenMeasure_h0) {
  SquareLatticePEPS<TenElemT, QNT> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx, 0));
  peps0.Initial(activates);
  SplitIndexTPS<TenElemT, QNT> sitps = TPS<TenElemT, QNT>(peps0);

  // Add small random noise to SplitIndexTPS tensors to break symmetry
  {
    QNT qn0 = QNT();
    const double noise_eps = 1e-3;
    for (size_t r = 0; r < Ly; ++r) {
      for (size_t c = 0; c < Lx; ++c) {
        auto &vec_comp = sitps({r, c});
        for (auto &ten_comp : vec_comp) {
          if (!ten_comp.IsDefault()) {
            auto noise = ten_comp; // same indices/shape
            noise.Random(qn0);
            noise *= TenElemT(noise_eps);
            ten_comp += noise;
          }
        }
      }
    }
  }

  Configuration init_config(Ly, Lx);
  for (size_t r = 0; r < Ly; ++r) {
    for (size_t c = 0; c < Lx; ++c) { init_config({r, c}) = 0; }
  }
  auto opt_params = OptimizerParamsBuilder()
      .SetMaxIterations(2)
      .SetLearningRate(0.1)
      .WithSGD()
      .Build();
  MonteCarloParams mc_params(10, 10, 1, init_config, false);
  PEPSParams peps_params(BMPSTruncatePara(2, 4, 1e-15,
                                          CompressMPSScheme::SVD_COMPRESS,
                                          std::make_optional<double>(1e-14),
                                          std::make_optional<size_t>(10)));
  VMCPEPSOptimizerParams vmc_params(opt_params, mc_params, peps_params);

  TransverseFieldIsingSquare model(/*h=*/0.0);

  auto opt = VmcOptimize<TenElemT, QNT,
                         MCUpdateSquareNNFullSpaceUpdate,
                         TransverseFieldIsingSquare>(
      vmc_params, sitps, comm, model, MCUpdateSquareNNFullSpaceUpdate{});

  if (rank == hp_numeric::kMPIMasterRank) {
    const auto &traj = opt->GetEnergyTrajectory();
    ASSERT_FALSE(traj.empty());
    EXPECT_NEAR(std::real(traj.back()), ClassicalIsing2x2OBC_GroundEnergy(), 0.2);
  }

  auto meas = MonteCarloMeasure<TenElemT, QNT,
                                MCUpdateSquareNNFullSpaceUpdate,
                                TransverseFieldIsingSquare>(
      opt->GetState(), MCMeasurementParams(mc_params, peps_params), comm, model, MCUpdateSquareNNFullSpaceUpdate{});

  auto [energy, en_err] = meas->OutputEnergy();
  if (rank == hp_numeric::kMPIMasterRank) {
    EXPECT_NEAR(std::real(energy), ClassicalIsing2x2OBC_GroundEnergy(), 0.1);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}



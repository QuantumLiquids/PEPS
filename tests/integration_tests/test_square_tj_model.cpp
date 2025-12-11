// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-02-25
*
* Description: QuantumLiquids/PEPS project. Integration test for t-J model on square lattice.
* For ED helper, find the `tests/tools/tJ_OBC.py` script.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "integration_test_framework.h"
#include "../test_mpi_env.h"
#include <array>

using namespace qlten;
using namespace qlpeps;

class SquaretJModelSystem
    : public IntegrationTestFramework<qlten::special_qn::fZ2QN, SquaretJModelSystem> {
protected:
  using QNT = qlten::special_qn::fZ2QN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using Tensor = QLTensor<TenElemT, QNT>;
  
  // Model parameters
  double t = 1.0;
  double J = 0.3;
  size_t hole_num = 4;
  size_t num_up = 4;
  size_t num_down = 4;
  inline static const double kChemicalPotential = 0.776927653748;
  static constexpr double kEDEnergy = -8.93157918694544;
  static constexpr double kEnergyTolerance = 5e-2;
  static constexpr std::array<double, 12> kEDChargeMap = {
      0.7212004230746919, 0.6236483846273850, 0.7212004230746910,
      0.6623151920824453, 0.6093203850583544, 0.6623151920824456,
      0.6623151920824445, 0.6093203850583540, 0.6623151920824455,
      0.7212004230746913, 0.6236483846273855, 0.7212004230746909};
  static constexpr std::array<double, 12> kEDSpinZMap = {
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0};
   
  // Physical indices for t-J model (3 states: up, down, empty)
  IndexT loc_phy_ket;
  IndexT loc_phy_bra;
  Tensor ham_nn;
  Tensor ham_onsite;
  
  void SetUpIndices() override {
    // Physical indices: |up>, |down>, |empty>
    loc_phy_ket = IndexT({QNSctT(QNT(1), 2),  // |up>, |down> (occupied)
                          QNSctT(QNT(0), 1)},  // |empty> state
                         TenIndexDirType::IN);
    loc_phy_bra = InverseIndex(loc_phy_ket);
  
  }
  
  void SetUpHamiltonians() override {
    const double V = 0.0;

    ham_nn = Tensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});
    ham_nn({2, 0, 2, 0}) = -t;
    ham_nn({2, 1, 2, 1}) = -t;
    ham_nn({0, 2, 0, 2}) = -t;
    ham_nn({1, 2, 1, 2}) = -t;
    ham_nn({0, 0, 0, 0}) = V;
    ham_nn({1, 1, 1, 1}) = V;
    ham_nn({0, 1, 1, 0}) = -0.5 * J + V;
    ham_nn({1, 0, 0, 1}) = -0.5 * J + V;
    ham_nn({0, 1, 0, 1}) = 0.5 * J;
    ham_nn({1, 0, 1, 0}) = 0.5 * J;
    ham_nn.Transpose({3, 0, 2, 1});

    ham_onsite = Tensor({loc_phy_ket, loc_phy_bra});
    ham_onsite({0, 0}) = -kChemicalPotential;
    ham_onsite({1, 1}) = -kChemicalPotential;
  }
  
  void SetUpParameters() override {
    model_name = "square_tj_model";
    energy_ed = kEDEnergy;
    
    // VMC optimization parameters - Modern API
    BMPSTruncateParams<qlten::QLTEN_Double> truncate_para = BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, Dpeps * 2, 1e-15);
    
    Configuration initial_config(Ly, Lx, OccupancyNum({num_up, num_down, hole_num}));
    MonteCarloParams mc_params(100, 100, 1, initial_config, false);
    PEPSParams peps_params(truncate_para);
    
    ConjugateGradientParams cg_params(100, 1e-5, 20, 0.001);
    auto opt_params = OptimizerFactory::CreateStochasticReconfiguration(100, cg_params, 0.1);
    
    optimize_para = VMCPEPSOptimizerParams(opt_params, mc_params, peps_params);
    
    // Monte Carlo measurement parameters
    Configuration measure_config(Ly, Lx, OccupancyNum({num_up, num_down, hole_num}));
    MonteCarloParams measure_mc_params(1000, 1000, 1, measure_config, false);
    PEPSParams measure_peps_params(BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, Dpeps * 2, 1e-15));
    measure_para = MCMeasurementParams(measure_mc_params, measure_peps_params);
  }

 public:
  template<typename ModelT, typename MCUpdaterT>
  void ValidateMeasurementResults(
      const MCPEPSMeasurer<TenElemT, QNT, MCUpdaterT, ModelT> &measurer) const {
    if (rank != hp_numeric::kMPIMasterRank) {
      return;
    }

    auto energy_stats = measurer.GetEnergyEstimate();
    ASSERT_TRUE(energy_stats.has_value());
    EXPECT_NEAR(std::real(energy_stats->energy), kEDEnergy, kEnergyTolerance);
    EXPECT_LT(std::abs(std::imag(energy_stats->energy)), 1e-10);

    const auto &observables = measurer.ObservableRegistry();
    auto charge_it = observables.find("charge");
    ASSERT_TRUE(charge_it != observables.end());
    const auto &charge_mean = charge_it->second.first;
    ASSERT_EQ(charge_mean.size(), kEDChargeMap.size());
    for (size_t idx = 0; idx < charge_mean.size(); ++idx) {
      const double actual = static_cast<double>(std::real(charge_mean[idx]));
      const double imag_part = static_cast<double>(std::imag(charge_mean[idx]));
      EXPECT_NEAR(actual, kEDChargeMap[idx], 2e-2); //TODO: run and relax the tolerance
      EXPECT_LT(std::abs(imag_part), 1e-8);
    }

    auto spin_it = observables.find("spin_z");
    ASSERT_TRUE(spin_it != observables.end());
    const auto &spin_mean = spin_it->second.first;
    ASSERT_EQ(spin_mean.size(), kEDSpinZMap.size());
    for (size_t idx = 0; idx < spin_mean.size(); ++idx) {
      const double actual = static_cast<double>(std::real(spin_mean[idx]));
      const double imag_part = static_cast<double>(std::imag(spin_mean[idx]));
      EXPECT_NEAR(actual, kEDSpinZMap[idx], 2e-2); //TODO: run and relax the tolerance
      EXPECT_LT(std::abs(imag_part), 1e-8);
    }
  }
  bool EnableFrameworkEnergyCheck() const override { return false; }

  double FrameworkEnergyTolerance() const override { return kEnergyTolerance; }
};

// Test simple update optimization
TEST_F(SquaretJModelSystem, SimpleUpdate) {
  if (rank == hp_numeric::kMPIMasterRank) {
    SquareLatticePEPS<TenElemT, QNT> peps0(loc_phy_ket, Ly, Lx);
    
    // Initialize with checkerboard pattern
    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
    size_t n_int = 0;
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        if (n_int < Lx * Ly - hole_num) {
          activates[y][x] = (n_int % 2) ;  // 0 for up, 1 for down
        } else {
          activates[y][x] = 2;  // Empty sites for holes
        }
        n_int++;
      }
    }
    peps0.Initial(activates);
    
    SimpleUpdatePara update_para(50, 0.1, 1, Dpeps, 1e-6);
    SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT> su_exe(update_para, peps0, ham_nn, ham_onsite);
    su_exe.Execute();

    auto tps = ToTPS<TenElemT, QNT>(su_exe.GetPEPS());
    auto sitps = ToSplitIndexTPS<TenElemT, QNT>(tps);
    sitps.Dump(tps_path);
  }
}

// Test VMC optimization with Stochastic Reconfiguration
TEST_F(SquaretJModelSystem, StochasticReconfigurationOpt) {
  using Model = SquaretJNNModel;
  using MCUpdater = MCUpdateSquareNNExchange;
  
  Model model(t, J, 0);
  RunVMCOptimization<Model, MCUpdater>(model);
}

TEST_F(SquaretJModelSystem, Measure) {
  using Model = SquaretJNNModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model(t, J, 0);
  RunMCMeasurement<Model, MCUpdater>(model);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}

// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-02-25
*
* Description: QuantumLiquids/PEPS project. Integration test for t-J model on square lattice.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "integration_test_framework.h"
#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

class SquaretJModelSystem : public IntegrationTestFramework<qlten::special_qn::fZ2QN> {
protected:
  using QNT = qlten::special_qn::fZ2QN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using Tensor = QLTensor<TenElemT, QNT>;
  
  // Model parameters
  double t = 1.0;
  double J = 0.3;
  double doping = 0.125;
  size_t hole_num;
  
  // Physical indices for t-J model (3 states: up, down, empty)
  IndexT loc_phy_ket;
  IndexT loc_phy_bra;
  
  void SetUpIndices() override {
    // Physical indices: |up>, |down>, |empty>
    loc_phy_ket = IndexT({QNSctT(QNT(1), 2),  // |up>, |down> (occupied)
                          QNSctT(QNT(0), 1)},  // |empty> state
                         TenIndexDirType::IN);
    loc_phy_bra = InverseIndex(loc_phy_ket);
    
    // Calculate hole number based on doping
    hole_num = size_t(double(Lx * Ly) * doping);
  }
  
  void SetUpHamiltonians() override {
    // Placeholder for Hamiltonian setup
    // In practice, you'd need the full t-J Hamiltonian construction
  }
  
  void SetUpParameters() override {
    model_name = "square_tj_model";
    energy_ed = -0.5;  // Placeholder - should be updated with actual ED energy
    
    // VMC optimization parameters
    optimize_para = VMCOptimizePara(
        BMPSTruncatePara(Dpeps, Dpeps * 2, 1e-15,
                         CompressMPSScheme::SVD_COMPRESS,
                         std::make_optional<double>(1e-14),
                         std::make_optional<size_t>(10)),
        100, 100, 1,
        {(Lx * Ly - hole_num) / 2, (Lx * Ly - hole_num) / 2, hole_num},
        Ly, Lx,
        std::vector<double>(40, 0.2),
        StochasticReconfiguration,
        ConjugateGradientParams(100, 1e-4, 20, 0.01));
    
    // Monte Carlo measurement parameters
    measure_para = MCMeasurementPara(
        BMPSTruncatePara(Dpeps, Dpeps * 2, 1e-15,
                         CompressMPSScheme::SVD_COMPRESS,
                         std::make_optional<double>(1e-14),
                         std::make_optional<size_t>(10)),
        1000, 1000, 1,
        {(Lx * Ly - hole_num) / 2, (Lx * Ly - hole_num) / 2, hole_num},
        Ly, Lx);
  }
};

// Test simple update optimization
TEST_F(SquaretJModelSystem, SimpleUpdate) {
  if (rank == kMPIMasterRank) {
    SquareLatticePEPS<TenElemT, QNT> peps0(loc_phy_ket, Ly, Lx);
    
    // Initialize with checkerboard pattern
    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
    size_t n_int = 0;
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        if (n_int < Lx * Ly - hole_num) {
          activates[y][x] = (n_int % 2) + 1;  // 1 for up, 2 for down
        } else {
          activates[y][x] = 0;  // Empty sites for holes
        }
        n_int++;
      }
    }
    peps0.Initial(activates);
    
    // Save initial TPS for now (Hamiltonian setup would be needed for full simple update)
    auto tps = TPS<TenElemT, QNT>(peps0);
    SplitIndexTPS<TenElemT, QNT> sitps = tps;
    sitps.Dump(tps_path);
  }
}

// Test VMC optimization with Stochastic Reconfiguration
TEST_F(SquaretJModelSystem, StochasticReconfigurationOpt) {
  using Model = SquaretJModel;
  using MCUpdater = MCUpdateSquareNNExchange;
  
  Model model(t, J, false, 0);
  RunVMCOptimization<Model, MCUpdater>(model);
}

// Test Monte Carlo measurement after VMC optimization
TEST_F(SquaretJModelSystem, StochasticReconfigurationMeasure) {
  using Model = SquaretJModel;
  using MCUpdater = MCUpdateSquareNNExchange;
  
  Model model(t, J, false, 0);
  RunMCMeasurement<Model, MCUpdater>(model);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}

// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-02-25
*
* Description: QuantumLiquids/PEPS project. Unittests for PEPS Simple Update in fermion model.
*/


#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

std::string GenTPSPath(std::string model_name, size_t Dmax, size_t Lx, size_t Ly) {
#if TEN_ELEM_TYPE == QLTEN_Double
  return "dtps_" + model_name + "_D" + std::to_string(Dmax) + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#elif TEN_ELEM_TYPE == QLTEN_Complex
  return "ztps_" + model_name + "_D" + std::to_string(Dmax)  + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#else
#error "Unexpected TEN_ELEM_TYPE"
#endif
}

///< Without chemical potential contribution
double CalGroundStateEnergyForSpinlessNNFreeFermionOBC(
    const size_t Lx,
    const size_t Ly,
    const size_t particle_num
) {
  const size_t num_sites = Lx * Ly;
  std::vector<double> energy_levels;

  // Calculate the energy levels
  for (size_t kx = 0; kx < Lx; ++kx) {
    for (size_t ky = 0; ky < Ly; ++ky) {
      double theta_x = M_PI * (kx + 1) / (Lx + 1);
      double theta_y = M_PI * (ky + 1) / (Ly + 1);
      double energy = -2 * (std::cos(theta_x) + std::cos(theta_y));
      energy_levels.push_back(energy);
    }
  }

  // Sort energy levels in ascending order
  std::sort(energy_levels.begin(), energy_levels.end());

  // Sum the lowest `particle_num` energy levels
  double ground_state_energy = 0.0;
  for (size_t i = 0; i < particle_num; ++i) {
    ground_state_energy += energy_levels[i];
  }

  return ground_state_energy;
}

struct Z2SpinlessFreeFermionSystem : public MPITest {
  using QNT = qlten::special_qn::fZ2QN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using QNSctVecT = QNSectorVec<QNT>;
  using TenElemT = QLTEN_Double;
  using Tensor = QLTensor<TenElemT, QNT>;
  size_t Lx = 4; //cols
  size_t Ly = 3;
  size_t Dpeps = 4;
  double energy_ED = -7.478708665;
  size_t N = Lx * Ly;
  size_t ele_num = 4;
  double t = 1.0;
  double mu = -0.707107; //chemical potential

  QNT qn0 = QNT(0);
  // |ket>
  IndexT loc_phy_ket = IndexT({QNSctT(QNT(1), 1),  // |1> occupied
                               QNSctT(QNT(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  // <bra|
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  Tensor c = Tensor({loc_phy_ket, loc_phy_bra});   // annihilation operator
  Tensor cdag = Tensor({loc_phy_ket, loc_phy_bra});// creation operator
  Tensor n = Tensor({loc_phy_ket, loc_phy_bra});   // density operator

  Tensor ham_nn = Tensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});//site: i-j-j-i (i<j)
  std::string model_name = "spinless_free_fermion";
  std::string tps_path = GenTPSPath(model_name, Dpeps, Lx, Ly);

  VMCOptimizePara optimize_para =
      VMCOptimizePara(BMPSTruncatePara(Dpeps, Dpeps * 3,
                                       1e-15, CompressMPSScheme::SVD_COMPRESS,
                                       std::make_optional<double>(1e-14),
                                       std::make_optional<size_t>(10)),
                      100, 100, 1,
                      std::vector<size_t>{4, 8},
                      Ly, Lx,
                      std::vector<double>(60, 0.2),
                      StochasticReconfiguration,
                      ConjugateGradientParams(100, 1e-4, 20, 0.01));

  MCMeasurementPara measure_para = MCMeasurementPara(
      BMPSTruncatePara(Dpeps, 3 * Dpeps, 1e-15,
                       CompressMPSScheme::SVD_COMPRESS,
                       std::make_optional<double>(1e-14),
                       std::make_optional<size_t>(10)),
      1000, 1000, 1,
      std::vector<size_t>(2, Lx * Ly / 2),
      Ly, Lx);

  void SetUp(void) {
    MPITest::SetUp();
    n({0, 0}) = 1.0;
    n.Transpose({1, 0});
    c({1, 0}) = 1;
    cdag({0, 1}) = 1;

    ham_nn({1, 0, 1, 0}) = -t;
    ham_nn({0, 1, 0, 1}) = -t;
    ham_nn.Transpose({3, 0, 2, 1}); // transpose indices order for consistent with simple update convention
    qlten::hp_numeric::SetTensorManipulationThreads(1);
    optimize_para.wavefunction_path = tps_path;
    measure_para.wavefunction_path = tps_path;
  }
};

TEST_F(Z2SpinlessFreeFermionSystem, SimpleUpdate) {
  if (rank == kMPIMasterRank) {
    SquareLatticePEPS<TenElemT, QNT> peps0(loc_phy_ket, Ly, Lx);

    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
    //half-filling
    size_t n_int = 0;
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        activates[y][x] = n_int % 2;
        n_int++;
      }
    }
    peps0.Initial(activates);

    SimpleUpdatePara update_para(1000, 0.1, 1, Dpeps, 1e-10);
    SimpleUpdateExecutor<TenElemT, QNT>
        *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>(update_para, peps0,
                                                                         ham_nn,
                                                                         -mu * n);
    su_exe->Execute();
    su_exe->ResetStepLenth(0.01);
    su_exe->Execute();
    su_exe->ResetStepLenth(0.001);
    su_exe->Execute();
    auto peps = su_exe->GetPEPS();
    auto tps = TPS<TenElemT, QNT>(su_exe->GetPEPS());
    SplitIndexTPS<TenElemT, QNT> sitps = tps;
    sitps.Dump(tps_path);
    delete su_exe;
  }
}

TEST_F(Z2SpinlessFreeFermionSystem, StochasticReconfigurationOptAndMeasure) {
  MPI_Barrier(comm);

  SplitIndexTPS<TenElemT, QNT> tps(Ly, Lx);
  if (!tps.Load(tps_path)) {
    std::cerr << "Error in load the TPS data." << std::endl;
    exit(1);
  };

  //VMC
  auto executor =
      new VMCPEPSExecutor<TenElemT, QNT, MCUpdateSquareTNN3SiteExchange, SquareSpinlessFreeFermion>(optimize_para, tps,
                                                                                                    comm);
  size_t start_flop = flop;
  Timer vmc_timer("vmc");

  executor->Execute();

  size_t end_flop = flop;
  double elapsed_time = vmc_timer.PrintElapsed();
  double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
  std::cout << "Gflops = " << Gflops / elapsed_time << std::endl;

  tps = executor->GetState();
  delete executor;

  //Measure
  auto measure_exe =
      new MonteCarloMeasurementExecutor<TenElemT, QNT, MCUpdateSquareTNN3SiteExchange, SquareSpinlessFreeFermion>(
          measure_para,
          tps,
          comm);
  start_flop = flop;

  measure_exe->Execute();

  end_flop = flop;
  elapsed_time = vmc_timer.PrintElapsed();
  Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
  std::cout << "Gflops = " << Gflops / elapsed_time << std::endl;

  auto [energy, en_err] = measure_exe->OutputEnergy();
  EXPECT_NEAR(Real(energy), energy_ED, 1E-3);

  //Measure2

  auto measure_exe2 =
      new MonteCarloMeasurementExecutor<TenElemT, QNT, MCUpdateSquareNNExchange, SquareSpinlessFreeFermion>(
          measure_para,
          tps,
          comm);
  measure_exe2->Execute();
  auto [energy2, en_err2] = measure_exe2->OutputEnergy();
  EXPECT_NEAR(Real(energy), Real(energy2), en_err + en_err2);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}


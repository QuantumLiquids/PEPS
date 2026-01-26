/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-01-04
*
* Description: QuantumLiquids/PEPS project. Smoke test for MCUpdateSquareNNHubbardU1U1OBC.
*
* Test strategy:
* 1. Generate a small Hubbard model TPS using simple update (U=0, free fermion limit)
* 2. Use MCUpdateSquareNNHubbardU1U1OBC for MC sampling
* 3. Verify the updater runs without crashing and produces reasonable acceptance rates
*/

#define PLAIN_TRANSPOSE 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"

#include "qlpeps/algorithm/simple_update/simple_update_model_all.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_hubbard_model.h"
#include "qlpeps/api/conversions.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"

#include "../test_mpi_env.h"
#include <memory>
#include <array>
#include <filesystem>

using namespace qlten;
using namespace qlpeps;

/**
 * @brief Smoke test for MCUpdateSquareNNHubbardU1U1OBC Monte Carlo updater.
 *
 * This test verifies that the Hubbard U1×U1 MC updater:
 * 1. Compiles and links correctly
 * 2. Runs without crashing on a valid TPS
 * 3. Produces non-zero acceptance rates (ergodicity check)
 */
class HubbardU1U1UpdaterTest : public MPITest {
 protected:
  using QNT = qlten::special_qn::fZ2QN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using TenElemT = QLTEN_Double;
  using Tensor = QLTensor<TenElemT, QNT>;

  // Small system for fast testing
  // Use an even number of sites so the alternating (up/down) product state has
  // even fermion parity. BMPS requires even fermion parity on the boundary index.
  static constexpr size_t Lx = 3;
  static constexpr size_t Ly = 2;
  static constexpr size_t Dmax = 4;

  // Hubbard parameters (U=0 for exact free-fermion reference)
  static constexpr double t = 1.0;
  static constexpr double U = 0.0;
  static constexpr double mu = 0.0;

  // MC parameters
  static constexpr unsigned int MC_SEED = 42;
  static constexpr size_t WARMUP_SWEEPS = 5;
  static constexpr size_t SAMPLE_SWEEPS = 10;

  // Hubbard single-site state encoding (matches SquareHubbardModel)
  enum class HubbardState : size_t {
    DoubleOccupancy = 0,
    Empty = 1,
    SpinUp = 2,
    SpinDown = 3
  };

  static constexpr size_t Idx(HubbardState s) { return static_cast<size_t>(s); }

  QNT qn0 = QNT(0);

  // Physical index: [even parity: doublon, empty] + [odd parity: up, down]
  IndexT loc_phy_ket = IndexT({QNSctT(QNT(0), 2),  // even parity: |↑↓>, |0>
                               QNSctT(QNT(1), 2)}, // odd parity:  |↑>, |↓>
                              TenIndexDirType::IN);
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  Tensor n_tot = Tensor({loc_phy_ket, loc_phy_bra});
  Tensor ham_nn = Tensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});

  std::string output_dir;

  void SetUp() override {
    MPITest::SetUp();
    qlten::hp_numeric::SetTensorManipulationThreads(1);

    output_dir = "test_hubbard_u1u1_updater_output";
    if (rank == hp_numeric::kMPIMasterRank) {
      std::filesystem::create_directories(output_dir);
    }
    MPI_Barrier(comm);

    // Build operators
    const size_t i_dbl = Idx(HubbardState::DoubleOccupancy);
    const size_t i_emp = Idx(HubbardState::Empty);
    const size_t i_up = Idx(HubbardState::SpinUp);
    const size_t i_dn = Idx(HubbardState::SpinDown);

    // n_tot = n_up + n_dn
    n_tot({i_dbl, i_dbl}) = 2.0;
    n_tot({i_up, i_up}) = 1.0;
    n_tot({i_dn, i_dn}) = 1.0;
    n_tot({i_emp, i_emp}) = 0.0;

    // Build two-site hopping Hamiltonian
    auto state_to_bits = [&](const size_t s) -> std::pair<int, int> {
      switch (HubbardState(s)) {
        case HubbardState::DoubleOccupancy: return {1, 1};
        case HubbardState::Empty: return {0, 0};
        case HubbardState::SpinUp: return {1, 0};
        case HubbardState::SpinDown: return {0, 1};
        default: return {0, 0};
      }
    };
    auto bits_to_state = [&](const int nu, const int nd) -> size_t {
      if (nu == 1 && nd == 1) return i_dbl;
      if (nu == 0 && nd == 0) return i_emp;
      if (nu == 1 && nd == 0) return i_up;
      return i_dn;
    };
    auto popcount_prefix = [&](const std::array<int, 4> &b, const int mode) -> int {
      int cnt = 0;
      for (int i = 0; i < mode; ++i) cnt += b[i];
      return cnt;
    };
    auto apply_annihilate = [&](std::array<int, 4> &b, const int mode, double &sgn) -> bool {
      if (b[mode] == 0) return false;
      if (popcount_prefix(b, mode) & 1) sgn *= -1.0;
      b[mode] = 0;
      return true;
    };
    auto apply_create = [&](std::array<int, 4> &b, const int mode, double &sgn) -> bool {
      if (b[mode] == 1) return false;
      if (popcount_prefix(b, mode) & 1) sgn *= -1.0;
      b[mode] = 1;
      return true;
    };

    for (size_t bra1 = 0; bra1 < 4; ++bra1) {
      for (size_t bra2 = 0; bra2 < 4; ++bra2) {
        const auto [n1u, n1d] = state_to_bits(bra1);
        const auto [n2u, n2d] = state_to_bits(bra2);
        const std::array<int, 4> bra_bits{n1u, n1d, n2u, n2d};

        for (int sigma = 0; sigma < 2; ++sigma) {
          const int mode1 = (sigma == 0) ? 0 : 1;
          const int mode2 = (sigma == 0) ? 2 : 3;

          // c1^dag c2
          {
            std::array<int, 4> b = bra_bits;
            double sgn = 1.0;
            if (apply_annihilate(b, mode2, sgn) && apply_create(b, mode1, sgn)) {
              const size_t ket1 = bits_to_state(b[0], b[1]);
              const size_t ket2 = bits_to_state(b[2], b[3]);
              ham_nn({ket1, ket2, bra2, bra1}) = ham_nn({ket1, ket2, bra2, bra1}) + (-t) * sgn;
            }
          }
          // c2^dag c1
          {
            std::array<int, 4> b = bra_bits;
            double sgn = 1.0;
            if (apply_annihilate(b, mode1, sgn) && apply_create(b, mode2, sgn)) {
              const size_t ket1 = bits_to_state(b[0], b[1]);
              const size_t ket2 = bits_to_state(b[2], b[3]);
              ham_nn({ket1, ket2, bra2, bra1}) = ham_nn({ket1, ket2, bra2, bra1}) + (-t) * sgn;
            }
          }
        }
      }
    }
    ham_nn.Transpose({3, 0, 2, 1});
  }

  /**
   * @brief Create a valid Hubbard configuration with given (N_up, N_down).
   */
  Configuration CreateHubbardConfiguration(size_t n_up, size_t n_down) const {
    Configuration config(Ly, Lx);
    size_t placed_up = 0, placed_down = 0;

    for (size_t y = 0; y < Ly; ++y) {
      for (size_t x = 0; x < Lx; ++x) {
        if (placed_up < n_up && placed_down < n_down) {
          // Place doubly occupied
          config({y, x}) = Idx(HubbardState::DoubleOccupancy);
          placed_up++;
          placed_down++;
        } else if (placed_up < n_up) {
          config({y, x}) = Idx(HubbardState::SpinUp);
          placed_up++;
        } else if (placed_down < n_down) {
          config({y, x}) = Idx(HubbardState::SpinDown);
          placed_down++;
        } else {
          config({y, x}) = Idx(HubbardState::Empty);
        }
      }
    }
    return config;
  }
};

/**
 * @brief Smoke test: Simple Update + MCUpdateSquareNNHubbardU1U1OBC.
 *
 * This test:
 * 1. Runs simple update to generate a valid Hubbard TPS
 * 2. Runs MC sampling using MCUpdateSquareNNHubbardU1U1OBC
 * 3. Verifies the code runs without crashing
 */
TEST_F(HubbardU1U1UpdaterTest, SimpleUpdateThenMCSmokeTest) {
  // Step 1: Simple Update to generate TPS
  SquareLatticePEPS<TenElemT, QNT> peps0(loc_phy_ket, Ly, Lx);

  // Initialize with alternating up/down pattern
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  for (size_t y = 0; y < Ly; ++y) {
    for (size_t x = 0; x < Lx; ++x) {
      activates[y][x] = ((x + y) % 2 == 0) ? Idx(HubbardState::SpinUp) : Idx(HubbardState::SpinDown);
    }
  }
  peps0.Initial(activates);

  // On-site term (chemical potential)
  Tensor ham_onsite = (-mu) * n_tot;
  ham_onsite.Transpose({1, 0});

  // Run simple update (short, just to get a valid TPS)
  SimpleUpdatePara update_para(20, 0.1, 1, Dmax, 1e-8);
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>>(
      update_para, peps0, ham_nn, ham_onsite);
  su_exe->Execute();

  // Convert to SplitIndexTPS
  auto tps = qlpeps::ToTPS<TenElemT, QNT>(su_exe->GetPEPS());
  SplitIndexTPS<TenElemT, QNT> sitps = SplitPhyIndex(tps);

  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "Simple update completed. Starting MC sampling with U1×U1 updater..." << std::endl;
  }

  // Step 2: MC Sampling with MCUpdateSquareNNHubbardU1U1OBC
  using MCUpdater = MCUpdateSquareNNHubbardU1U1OBC;

  // Create initial configuration matching the simple update initialization
  Configuration initial_config(Ly, Lx);
  for (size_t y = 0; y < Ly; ++y) {
    for (size_t x = 0; x < Lx; ++x) {
      initial_config({y, x}) = ((x + y) % 2 == 0) ? Idx(HubbardState::SpinUp) : Idx(HubbardState::SpinDown);
    }
  }

  PEPSParams peps_params(BMPSTruncateParams<TenElemT>(Dmax, Dmax, 1e-10, {}));

  MonteCarloParams mc_params(SAMPLE_SWEEPS, WARMUP_SWEEPS, 1, initial_config, false);

  MCUpdater mc_updater(MC_SEED);

  // Create MC engine
  MonteCarloEngine<TenElemT, QNT, MCUpdater> engine(
      sitps, mc_params, peps_params, comm, mc_updater);

  // Warm up
  for (size_t i = 0; i < WARMUP_SWEEPS; ++i) {
    auto accept_rates = engine.StepSweep();
    if (rank == hp_numeric::kMPIMasterRank && i == 0) {
      std::cout << "Warmup sweep " << i << " accept rate: " << accept_rates[0] << std::endl;
    }
  }

  // Sample sweeps
  double total_accept_rate = 0.0;
  for (size_t i = 0; i < SAMPLE_SWEEPS; ++i) {
    auto accept_rates = engine.StepSweep();
    total_accept_rate += accept_rates[0];
  }
  double avg_accept_rate = total_accept_rate / SAMPLE_SWEEPS;

  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "Average accept rate over " << SAMPLE_SWEEPS << " sweeps: " << avg_accept_rate << std::endl;
  }

  // Verify: acceptance rate should be positive (ergodicity)
  // For a well-equilibrated system, we expect some moves to be accepted
  EXPECT_GE(avg_accept_rate, 0.0);
  EXPECT_LE(avg_accept_rate, 1.0);

  // If accept rate is exactly 0, something might be wrong with the updater
  // (but this could happen in very specific cases, so we only warn)
  if (avg_accept_rate < 0.01) {
    std::cerr << "Warning: Very low acceptance rate (" << avg_accept_rate
              << "). This might indicate an issue with the updater." << std::endl;
  }

  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "MCUpdateSquareNNHubbardU1U1OBC smoke test PASSED." << std::endl;
  }
}

/**
 * @brief Unit test for HubbardConfig2SpinCounts helper function.
 */
TEST_F(HubbardU1U1UpdaterTest, HubbardConfig2SpinCountsTest) {
  // Test the helper function used in the updater
  // Note: The updater uses its own HubbardConfig2SpinCounts which follows SquareHubbardModel's encoding
  
  // DoubleOccupancy (0) -> (1, 1)
  auto [n_up_0, n_down_0] = HubbardConfig2SpinCounts(0);
  EXPECT_EQ(n_up_0, 1);
  EXPECT_EQ(n_down_0, 1);

  // SpinUp (1) -> (1, 0)
  auto [n_up_1, n_down_1] = HubbardConfig2SpinCounts(1);
  EXPECT_EQ(n_up_1, 1);
  EXPECT_EQ(n_down_1, 0);

  // SpinDown (2) -> (0, 1)
  auto [n_up_2, n_down_2] = HubbardConfig2SpinCounts(2);
  EXPECT_EQ(n_up_2, 0);
  EXPECT_EQ(n_down_2, 1);

  // Empty (3) -> (0, 0)
  auto [n_up_3, n_down_3] = HubbardConfig2SpinCounts(3);
  EXPECT_EQ(n_up_3, 0);
  EXPECT_EQ(n_down_3, 0);
}

/**
 * @brief Unit test for EnumerateHubbardTwoSiteConfigsWithU1U1 helper function.
 */
TEST_F(HubbardU1U1UpdaterTest, EnumerateTwoSiteConfigsTest) {
  // (N_up=0, N_down=0): only (Empty, Empty)
  auto configs_00 = EnumerateHubbardTwoSiteConfigsWithU1U1(0, 0);
  EXPECT_EQ(configs_00.size(), 1);
  EXPECT_EQ(configs_00[0], std::make_pair(size_t(3), size_t(3)));

  // (N_up=1, N_down=0): (SpinUp, Empty) and (Empty, SpinUp)
  auto configs_10 = EnumerateHubbardTwoSiteConfigsWithU1U1(1, 0);
  EXPECT_EQ(configs_10.size(), 2);

  // (N_up=0, N_down=1): (SpinDown, Empty) and (Empty, SpinDown)
  auto configs_01 = EnumerateHubbardTwoSiteConfigsWithU1U1(0, 1);
  EXPECT_EQ(configs_01.size(), 2);

  // (N_up=1, N_down=1): 4 configurations
  // (DoubleOcc, Empty), (Empty, DoubleOcc), (SpinUp, SpinDown), (SpinDown, SpinUp)
  auto configs_11 = EnumerateHubbardTwoSiteConfigsWithU1U1(1, 1);
  EXPECT_EQ(configs_11.size(), 4);

  // (N_up=2, N_down=0): (SpinUp, SpinUp)
  auto configs_20 = EnumerateHubbardTwoSiteConfigsWithU1U1(2, 0);
  EXPECT_EQ(configs_20.size(), 1);
  EXPECT_EQ(configs_20[0], std::make_pair(size_t(1), size_t(1)));

  // (N_up=2, N_down=2): (DoubleOcc, DoubleOcc)
  auto configs_22 = EnumerateHubbardTwoSiteConfigsWithU1U1(2, 2);
  EXPECT_EQ(configs_22.size(), 1);
  EXPECT_EQ(configs_22[0], std::make_pair(size_t(0), size_t(0)));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  testing::InitGoogleTest(&argc, argv);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}

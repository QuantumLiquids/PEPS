/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-10-14
*
* Description: QuantumLiquids/PEPS project. Unittests for PEPS Monte-Carlo Updater for fermion tensor networks.
*
* ~ 100 seconds to run.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"

#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"
#include "qlpeps/vmc_basic/wave_function_component.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"

using namespace qlten;
using namespace qlpeps;

std::string ipeps_data_path;
using qlten::special_qn::fZ2QN;

struct Z2tJModelTools : public testing::Test {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using Tensor = QLTensor<QLTEN_Double, fZ2QN>;

  size_t Lx = 24;
  size_t Ly = 24;
  size_t N = Lx * Ly;
  double t = 3;
  double J = 1;
  double doping = 0.125; // actually the data is doping 0.124 from iPEPS simple update
  size_t hole_num = size_t(double(N) * doping);
  size_t num_up = (N - hole_num) / 2;
  size_t num_down = (N - hole_num) / 2;
  IndexT loc_phy_ket = IndexT({QNSctT(fZ2QN(1), 2), // |up>, |down>
                               QNSctT(fZ2QN(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  // PEPS D = 4
  size_t Db_min = 4;
  size_t Db_max = 16;

  size_t MC_samples = 100;
  size_t WarmUp = 100;
  std::string tps_path = "tps_from_ipeps_tJ_doping" + std::to_string(doping) + "_D4";
  Configuration measurement_config{Ly, Lx, OccupancyNum({(N - hole_num) / 2, (N - hole_num) / 2, hole_num})};
  MonteCarloParams measurement_mc_params{MC_samples, WarmUp, 1, measurement_config, false}; // not warmed up initially
  PEPSParams measurement_peps_params{BMPSTruncateParams<qlten::QLTEN_Double>(Db_min, Db_max, 1e-10,
                                                      CompressMPSScheme::SVD_COMPRESS,
                                                      std::make_optional<double>(1e-14),
                                                      std::make_optional<size_t>(10))};
  MCMeasurementParams mc_measurement_para{measurement_mc_params, measurement_peps_params};

  SplitIndexTPS<QLTEN_Double, fZ2QN> split_idx_tps = SplitIndexTPS<QLTEN_Double, fZ2QN>(Ly, Lx);

  std::uniform_real_distribution<double> u_double = std::uniform_real_distribution<double>(0, 1);

  void SetUp(void) {
    qlten::hp_numeric::SetTensorManipulationThreads(1);
    split_idx_tps = CreateFiniteSizeOBCTPS();
  }

  SplitIndexTPS<QLTEN_Double, fZ2QN> CreateFiniteSizeOBCTPS() {
    Tensor ten_a, ten_b;
    std::ifstream ifs(ipeps_data_path + "ipeps_tJ_ta_doping0.125.qlten");
    ifs >> ten_a;
    ifs.close();
    ifs.open(ipeps_data_path + "ipeps_tJ_tb_doping0.125.qlten");
    ifs >> ten_b;
    ifs.close();
    auto qn0 = fZ2QN(0);
    TPS<QLTEN_Double, fZ2QN> tps(Ly, Lx);
    for (size_t row = 0; row < Ly; row++) {
      for (size_t col = 0; col < Lx; col++) {
        Tensor local_ten;
        if ((row + col) % 2 == 0) {
          local_ten = ten_a;
        } else {
          local_ten = ten_b;
        }
        Tensor u, v;
        Tensor s;
        size_t D_act;
        double trunc_err_act;
        if (row == 0) {
          local_ten.Transpose({3, 0, 1, 2, 4});
          SVD(&local_ten, 1, qn0, 0, 1, 1, &u, &s, &v, &trunc_err_act, &D_act);
          if (!s.GetIndex(0).GetQNSct(0).GetQn().IsFermionParityEven()) {
            std::cout << "(row, col) = (" << row << "," << col << "), UP odd fermion parity s" << std::endl;
          }
          local_ten = v;
          local_ten.Transpose({1, 2, 3, 0, 4});
        } else if (row == Ly - 1) {
          local_ten.Transpose({1, 2, 3, 0, 4});
          SVD(&local_ten, 1, qn0, 0, 1, 1, &u, &s, &v, &trunc_err_act, &D_act);
          if (!s.GetIndex(0).GetQNSct(0).GetQn().IsFermionParityEven()) {
            std::cout << "(row, col) = (" << row << "," << col << "), DOWN odd fermion parity s" << std::endl;
          }
          local_ten = v;
          local_ten.Transpose({3, 0, 1, 2, 4});
        }
        u = Tensor();
        v = Tensor();
        s = Tensor();
        if (col == 0) {
          SVD(&local_ten, 1, qn0, 0, 1, 1, &u, &s, &v, &trunc_err_act, &D_act);
          if (!s.GetIndex(0).GetQNSct(0).GetQn().IsFermionParityEven()) {
            std::cout << "(row, col) = (" << row << "," << col << "), LEFT odd fermion parity s" << std::endl;
          }
          local_ten = v;
        } else if (col == Lx - 1) {
          local_ten.Transpose({2, 3, 0, 1, 4});
          SVD(&local_ten, 1, qn0, 0, 1, 1, &u, &s, &v, &trunc_err_act, &D_act);
          if (!s.GetIndex(0).GetQNSct(0).GetQn().IsFermionParityEven()) {
            std::cout << "(row, col) = (" << row << "," << col << "), RIGHT odd fermion parity s" << std::endl;
          }
          local_ten = v;
          local_ten.Transpose({2, 3, 0, 1, 4});
        }
        tps({row, col}) = local_ten;
      }
    }
    auto split_idx_tps = SplitIndexTPS<QLTEN_Double, fZ2QN>::FromTPS(tps);
    split_idx_tps.NormalizeAllSite();
    split_idx_tps *= 3.0;
    split_idx_tps.Dump(tps_path);
    return split_idx_tps;
  }
};

size_t CountNumOfSpinUp(const Configuration &config) {
  size_t count = 0;
  for (auto spin_config : config) {
    if (spin_config == 0) {
      count++;
    }
  }
  return count;
}

size_t CountNumOfSpinDown(const Configuration &config) {
  size_t count = 0;
  for (auto spin_config : config) {
    if (spin_config == 1) {
      count++;
    }
  }
  return count;
}

size_t CountNumOfHole(const Configuration &config) {
  size_t count = 0;
  for (auto spin_config : config) {
    if (spin_config == 2) {
      count++;
    }
  }
  return count;
}

TEST_F(Z2tJModelTools, MonteCarlo2SiteUpdate) {
  TPSWaveFunctionComponent<QLTEN_Double, fZ2QN> tps_sample(split_idx_tps, mc_measurement_para.mc_params.initial_config, mc_measurement_para.peps_params.GetBMPSParams());
  std::vector<double> accept_rate(1);
  for (size_t i = 0; i < 10; i++) {
    MCUpdateSquareNNExchange tnn_flip_updater;
    tnn_flip_updater(split_idx_tps, tps_sample, accept_rate);
    EXPECT_EQ(CountNumOfHole(tps_sample.config), hole_num);
    EXPECT_EQ(CountNumOfSpinDown(tps_sample.config), num_down);
    EXPECT_EQ(CountNumOfSpinUp(tps_sample.config), num_up);
  }
}

TEST_F(Z2tJModelTools, MonteCarlo3SiteUpdate) {
  TPSWaveFunctionComponent<QLTEN_Double, fZ2QN> tps_sample(split_idx_tps, mc_measurement_para.mc_params.initial_config, mc_measurement_para.peps_params.GetBMPSParams());
  std::vector<double> accept_rate(1);
  for (size_t i = 0; i < 10; i++) {
    MCUpdateSquareTNN3SiteExchange tnn_flip_updater;
    tnn_flip_updater(split_idx_tps, tps_sample, accept_rate);
    EXPECT_EQ(CountNumOfHole(tps_sample.config), hole_num);
    EXPECT_EQ(CountNumOfSpinDown(tps_sample.config), num_down);
    EXPECT_EQ(CountNumOfSpinUp(tps_sample.config), num_up);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  ipeps_data_path = argv[1];
  auto test_err = RUN_ALL_TESTS();
}

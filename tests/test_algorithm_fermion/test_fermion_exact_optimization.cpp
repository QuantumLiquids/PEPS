/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-10-23
*
* Description: QuantumLiquids/PEPS project. Unittests for the gradient,
*   by optimizing the fermion PEPS by exact summation.
*/


#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::fZ2QN;

struct SimpleUpdateTestParams {
  size_t Ly;
  size_t Lx;
  size_t D;
  double Tau0;
  size_t Steps;
};

Configuration Vec2Config(std::vector<size_t> &config_vec,
                         const size_t Lx, const size_t Ly) {
  Configuration config(Ly, Lx);
  for (size_t i = 0; i < config_vec.size(); i++) {
    const size_t row = i / Lx;
    const size_t col = i % Lx;
    config({row, col}) = config_vec[i];
  }
  return config;
}

struct Z2SpinlessFreeFermionTools : public testing::Test {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using DTensor = QLTensor<QLTEN_Double, fZ2QN>;
  size_t Lx = 2;
  size_t Ly = 2;

  SplitIndexTPS<QLTEN_Double, fZ2QN> split_index_tps = SplitIndexTPS<QLTEN_Double, fZ2QN>(Ly, Lx);
  double t = 1.0;
  fZ2QN qn0 = fZ2QN(0);
  // |ket>
  IndexT loc_phy_ket = IndexT({QNSctT(fZ2QN(1), 1),  // |1> occupied
                               QNSctT(fZ2QN(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  // <bra|
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  std::vector<Configuration> all_configs;
  void SetUp(void) {
    split_index_tps = SimpleUpdate();
    GenerateAllConfigs();
  }

  SplitIndexTPS<QLTEN_Double, fZ2QN> SimpleUpdate(void) {
    SquareLatticePEPS<QLTEN_Double, fZ2QN> peps0(loc_phy_ket, Ly, Lx);

    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
    //half-filling
    size_t n_int = 1;
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        activates[y][x] = n_int % 2;
        n_int++;
      }
    }
    peps0.Initial(activates);

    DTensor ham_nn = DTensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});//site: i-j-j-i (i<j)
    ham_nn({1, 0, 1, 0}) = -t;
    ham_nn({0, 1, 0, 1}) = -t;
    ham_nn.Transpose({3, 0, 2, 1}); // transpose indices order for consistent with simple update convention


    SimpleUpdatePara update_para(3, 0.1, 1, 4, 1e-10);
    SimpleUpdateExecutor<QLTEN_Double, fZ2QN>
        *su_exe = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Double, fZ2QN>(update_para, peps0,
                                                                               ham_nn);
    su_exe->Execute();
    auto peps = su_exe->GetPEPS();
    delete su_exe;
    auto tps = TPS<QLTEN_Double, fZ2QN>(peps);
    SplitIndexTPS<QLTEN_Double, fZ2QN> sitps = SplitIndexTPS<QLTEN_Double, fZ2QN>(tps);
    sitps.ScaleMaxAbsForAllSite(1.0);
    return sitps;
  }

  //half-filling
  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {0, 0, 1, 1};
    do {
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    } while (std::next_permutation(config_vec.begin(),
                                   config_vec.end()));
    // Generates the next lexicographical permutation
  }
};

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> EvaluateLocalPsiPartialPsiDag(
    const QLTensor<TenElemT, QNT> &hole_dag,
    const QLTensor<TenElemT, QNT> &local_psi_ten
) {
  const QLTensor<TenElemT, QNT> hole = Dag(hole_dag);
  QLTensor<TenElemT, QNT> psi_scale_ten, psi_partial_psi_dag;
  Contract(&hole, {1, 2, 3, 4}, &local_psi_ten, {0, 1, 2, 3}, &psi_scale_ten);
  Contract(&hole_dag, {0}, &psi_scale_ten, {0}, &psi_partial_psi_dag);
  return psi_partial_psi_dag;
}

TEST_F(Z2SpinlessFreeFermionTools, ExactSummation) {
  std::vector<double> weights;
  std::vector<double> e_loc_set;
  SplitIndexTPS<QLTEN_Double, fZ2QN> g_weighted_sum(Ly, Lx, 2);
  SplitIndexTPS<QLTEN_Double, fZ2QN> g_times_e_weighted_sum(Ly, Lx, 2);
  using WaveFunctionComponentT = SquareTPSSampleNNExchange<QLTEN_Double, fZ2QN>;
  WaveFunctionComponentT::trun_para =
      BMPSTruncatePara(1, 4, 1e-16, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  using Model = SquareSpinlessFreeFermion<QLTEN_Double, fZ2QN>;
  Model spin_less_fermion;
  for (auto &config : all_configs) {
    WaveFunctionComponentT tps_sample(split_index_tps, config);
    weights.push_back(std::norm(tps_sample.amplitude));
    TensorNetwork2D<QLTEN_Double, fZ2QN> holes_dag(Ly, Lx);
    double e_loc = spin_less_fermion.template CalEnergyAndHoles<WaveFunctionComponentT, true>(&split_index_tps,
                                                                                              &tps_sample,
                                                                                              holes_dag);
    e_loc_set.push_back(e_loc);

    SplitIndexTPS<QLTEN_Double, fZ2QN> gradient_sample(Ly, Lx, 2);
    for (size_t row = 0; row < Ly; row++) {
      for (size_t col = 0; col < Lx; col++) {
        size_t basis = tps_sample.config({row, col});
        auto psi_partial_psi_dag = EvaluateLocalPsiPartialPsiDag(holes_dag({row, col}), tps_sample.tn({row, col}));
        gradient_sample({row, col})[basis] = psi_partial_psi_dag;
      }
    }
    g_weighted_sum += gradient_sample;
    g_times_e_weighted_sum += e_loc * gradient_sample;
  }

  double weight_sum = 0.0;  // wave-function overlap
  double e_loc_sum = 0.0;
  for (size_t i = 0; i < e_loc_set.size(); i++) {
    e_loc_sum += e_loc_set[i] * weights[i];
    weight_sum += weights[i];
  }
  double energy = e_loc_sum / weight_sum;
  std::cout << "simple update E0 = " << std::setw(14) << std::fixed
            << std::setprecision(kEnergyOutputPrecision) << energy << std::endl;
  SplitIndexTPS<QLTEN_Double, fZ2QN> gradient = (g_times_e_weighted_sum - energy * g_weighted_sum) * (1.0 / weight_sum);
  gradient.ActFermionPOps();
  std::cout << "grad norm : " << gradient.NormSquare() << std::endl;
  for (size_t i = 0; i < 20; i++) {
    weights = {};
    e_loc_set = {};
    double step_len = (i + 1) * 0.0001;
    SplitIndexTPS<QLTEN_Double, fZ2QN> updated_split_index_tps = split_index_tps - step_len * gradient;

    for (auto &config : all_configs) {
      WaveFunctionComponentT tps_sample(updated_split_index_tps, config);
      weights.push_back(std::norm(tps_sample.amplitude));
      TensorNetwork2D<QLTEN_Double, fZ2QN> holes_dag(Ly, Lx);
      double
          e_loc = spin_less_fermion.template CalEnergyAndHoles<WaveFunctionComponentT, true>(&updated_split_index_tps,
                                                                                             &tps_sample,
                                                                                             holes_dag);
      e_loc_set.push_back(e_loc);
    }
    weight_sum = 0.0;  // wave-function overlap
    e_loc_sum = 0.0;
    for (size_t j = 0; j < e_loc_set.size(); j++) {
      e_loc_sum += e_loc_set[j] * weights[j];
      weight_sum += weights[j];
    }
    energy = e_loc_sum / weight_sum;
    std::cout << "step length : " << step_len << ", E0 = " << std::setw(14) << std::fixed
              << std::setprecision(kEnergyOutputPrecision) << energy << std::endl;
  }//i, step lengh of udate
}

struct Z2tJTools : public testing::Test {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using DTensor = QLTensor<QLTEN_Double, fZ2QN>;
  size_t Lx = 2;
  size_t Ly = 2;

  SplitIndexTPS<QLTEN_Double, fZ2QN> split_index_tps = SplitIndexTPS<QLTEN_Double, fZ2QN>(Ly, Lx);
  double t = 1.0;
  double J = 0.3;
  double energy_exact = -2.94316357;
  size_t Db = 4;
  fZ2QN qn0 = fZ2QN(0);
  // |ket>
  IndexT loc_phy_ket = IndexT({QNSctT(fZ2QN(1), 2), // |up>, |down>
                               QNSctT(fZ2QN(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  // <bra|
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  std::vector<Configuration> all_configs;
  void SetUp(void) {
    split_index_tps = SimpleUpdate();
    GenerateAllConfigs();
  }

  SplitIndexTPS<QLTEN_Double, fZ2QN> SimpleUpdate(void) {
    SquareLatticePEPS<QLTEN_Double, fZ2QN> peps0(loc_phy_ket, Ly, Lx);

    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
    activates = {{2, 2}, {0, 1}};
    peps0.Initial(activates);

    DTensor dham_tj_nn = DTensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});//site: i-j-j-i (i<j)
    dham_tj_nn({2, 0, 2, 0}) = -t;
    dham_tj_nn({2, 1, 2, 1}) = -t;
    dham_tj_nn({0, 2, 0, 2}) = -t;
    dham_tj_nn({1, 2, 1, 2}) = -t;

    dham_tj_nn({0, 0, 0, 0}) = 0.25 * J;    //FM, diagonal element
    dham_tj_nn({1, 1, 1, 1}) = 0.25 * J;    //FM, diagonal element
    dham_tj_nn({0, 1, 1, 0}) = -0.25 * J;   //AFM,diagonal element
    dham_tj_nn({1, 0, 0, 1}) = -0.25 * J;   //AFM,diagonal element
    dham_tj_nn({0, 1, 0, 1}) = 0.5 * J;     //off diagonal element
    dham_tj_nn({1, 0, 1, 0}) = 0.5 * J;     //off diagonal element

    dham_tj_nn.Transpose({3, 0, 2, 1}); // transpose indices order for consistent with simple update convention


    SimpleUpdatePara update_para(100, 0.1, 1, Db, 1e-10);
    SimpleUpdateExecutor<QLTEN_Double, fZ2QN>
        *su_exe = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Double, fZ2QN>(update_para, peps0,
                                                                               dham_tj_nn);
    su_exe->Execute();
    auto peps = su_exe->GetPEPS();
    delete su_exe;
    auto tps = TPS<QLTEN_Double, fZ2QN>(peps);
    SplitIndexTPS<QLTEN_Double, fZ2QN> sitps = SplitIndexTPS<QLTEN_Double, fZ2QN>(tps);
    return sitps;
  }

  //half-filling
  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {2, 2, 0, 1};
    std::sort(config_vec.begin(), config_vec.end());
    do {
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    } while (std::next_permutation(config_vec.begin(),
                                   config_vec.end()));
    // Generates the next lexicographical permutation
  }
};

TEST_F(Z2tJTools, ExactSummation) {
  using WaveFunctionComponentT = SquareTPSSampleNNExchange<QLTEN_Double, fZ2QN>;
  WaveFunctionComponentT::trun_para =
      BMPSTruncatePara(1, Db, 1e-16, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  double energy;
  for (size_t i = 0; i < 100; i++) {
    std::vector<double> weights;
    std::vector<double> e_loc_set;
    SplitIndexTPS<QLTEN_Double, fZ2QN> g_weighted_sum(Ly, Lx, 3);
    SplitIndexTPS<QLTEN_Double, fZ2QN> g_times_e_weighted_sum(Ly, Lx, 3);
    using Model = SquaretJModel<QLTEN_Double, fZ2QN>;
    Model tj_model(t, J, false);
    for (auto &config : all_configs) {
      WaveFunctionComponentT tps_sample(split_index_tps, config);
      weights.push_back(std::norm(tps_sample.amplitude));
      TensorNetwork2D<QLTEN_Double, fZ2QN> holes_dag(Ly, Lx);
      double e_loc = tj_model.template CalEnergyAndHoles<WaveFunctionComponentT, true>(&split_index_tps,
                                                                                       &tps_sample,
                                                                                       holes_dag);
      e_loc_set.push_back(e_loc);

      SplitIndexTPS<QLTEN_Double, fZ2QN> gradient_sample(Ly, Lx, 3);
      for (size_t row = 0; row < Ly; row++) {
        for (size_t col = 0; col < Lx; col++) {
          size_t basis = tps_sample.config({row, col});
          auto psi_partial_psi_dag = EvaluateLocalPsiPartialPsiDag(holes_dag({row, col}), tps_sample.tn({row, col}));
          gradient_sample({row, col})[basis] = psi_partial_psi_dag;
        }
      }
      g_weighted_sum += gradient_sample;
      g_times_e_weighted_sum += e_loc * gradient_sample;
    }

    double weight_sum = 0.0;  // wave-function overlap
    double e_loc_sum = 0.0;
    for (size_t j = 0; j < e_loc_set.size(); j++) {
      e_loc_sum += e_loc_set[j] * weights[j];
      weight_sum += weights[j];
    }
    energy = e_loc_sum / weight_sum;

    SplitIndexTPS<QLTEN_Double, fZ2QN>
        gradient = (g_times_e_weighted_sum - energy * g_weighted_sum) * (1.0 / weight_sum);
    gradient.ActFermionPOps();
    std::cout << "E0 = " << std::setw(14) << std::fixed
              << std::setprecision(kEnergyOutputPrecision) << energy
              << "Grad Norm :" << std::setw(8) << std::fixed
              << std::setprecision(kEnergyOutputPrecision) << gradient.NormSquare()
              << std::endl;
    split_index_tps += (-0.1) * gradient;
  }
  EXPECT_GE(energy, energy_exact);
  EXPECT_NEAR(energy, energy_exact, 1e-5);
}

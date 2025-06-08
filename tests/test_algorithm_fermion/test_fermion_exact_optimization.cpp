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

template<typename ModelT, typename TenElemT, typename QNT>
double RunExactSumGradientOpt(
    ModelT &model,
    SplitIndexTPS<TenElemT, QNT> &split_index_tps,
    const std::vector<Configuration> &all_configs,
    const BMPSTruncatePara &trun_para,
    size_t Ly, size_t Lx,
    const double energy_exact,
    const size_t n_steps,
    const double step_length
);

/**
 * @brief Computes the ground state energy of spinless free fermions
 *        on a 2x2 OBC square lattice with NN and NNN hopping, equivalent
 *        to a 4-site PBC chain.
 *
 * @param t     Nearest-neighbor hopping amplitude
 * @param t2    Next-nearest-neighbor hopping amplitude
 * @return double Ground state energy
 */
double Calculate2x2OBCSpinlessFreeFermionEnergy(double t, double t2) {
  std::vector<double> k_values = {0.0, M_PI / 2.0, M_PI, 3.0 * M_PI / 2.0}; //momentum in 4-site PBC chain
  std::vector<double> single_particle_energies;
  for (auto k : k_values) {
    double energy = -2.0 * t * std::cos(k) - t2 * std::cos(2.0 * k);
    single_particle_energies.push_back(energy);
  }
  double ground_state_energy = 0.0;
  for (auto energy : single_particle_energies) {
    ground_state_energy += (double) (energy < 0) * energy;
  }
  return ground_state_energy;
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
  std::vector<double> t2_list = {2.1, 0, -2.5};
  // available t2: (-inf, -2] U {0} U [2, inf), these value make sure the ground state particle number is even (=2).
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

TEST_F(Z2SpinlessFreeFermionTools, ExactSumGradientOpt) {
  auto trun_para =
      BMPSTruncatePara(1, 8, 1e-16, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  using Model = SquareSpinlessFermion;
  for (auto t2 : t2_list) {
    auto energy_exact = Calculate2x2OBCSpinlessFreeFermionEnergy(t, t2);
    Model spinless_fermion_model(t, t2, 0);
    RunExactSumGradientOpt(spinless_fermion_model,
                           split_index_tps,
                           all_configs,
                           trun_para,
                           Ly,
                           Lx,
                           energy_exact,
                           200,
                           0.5);
  }
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
  double mu = 0.0;
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

TEST_F(Z2tJTools, ExactSumGradientOpt) {
  auto trun_para =
      BMPSTruncatePara(1, Db, 1e-16, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  using Model = SquaretJModel;
  Model tj_model(t, J, false, mu);
  RunExactSumGradientOpt(tj_model, split_index_tps,
                         all_configs, trun_para, Ly, Lx,
                         energy_exact, 100, 0.5);
}

template<typename ModelT, typename TenElemT, typename QNT>
double RunExactSumGradientOpt(
    ModelT &model,
    SplitIndexTPS<TenElemT, QNT> &split_index_tps,
    const std::vector<Configuration> &all_configs,
    const BMPSTruncatePara &trun_para,
    size_t Ly, size_t Lx,
    const double energy_exact,
    const size_t n_steps,
    const double step_length
) {
  double energy = 0.0;
  size_t phy_dim = split_index_tps.PhysicalDim();
  using SplitIndexTPSType = SplitIndexTPS<TenElemT, QNT>;
  for (size_t i = 0; i < n_steps; i++) {
    std::vector<double> weights;
    std::vector<double> e_loc_set;
    SplitIndexTPSType g_weighted_sum(Ly, Lx, phy_dim);
    SplitIndexTPSType g_times_e_weighted_sum(Ly, Lx, phy_dim);

    for (auto &config : all_configs) {
      TPSWaveFunctionComponent<TenElemT, QNT>
          tps_sample(split_index_tps, config, trun_para);
      weights.push_back(std::norm(tps_sample.amplitude));
      TensorNetwork2D<TenElemT, QNT> holes_dag(Ly, Lx);
      double e_loc =
          model.template CalEnergyAndHoles<TenElemT, QNT, true>(
              &split_index_tps, &tps_sample, holes_dag);
      e_loc_set.push_back(e_loc);

      SplitIndexTPSType gradient_sample(Ly, Lx, phy_dim);
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

    double weight_sum = 0.0;
    double e_loc_sum = 0.0;
    for (size_t j = 0; j < e_loc_set.size(); j++) {
      e_loc_sum += e_loc_set[j] * weights[j];
      weight_sum += weights[j];
    }
    energy = e_loc_sum / weight_sum;

    SplitIndexTPSType gradient = (g_times_e_weighted_sum - energy * g_weighted_sum) * (1.0 / weight_sum);
    gradient.ActFermionPOps();
    std::cout << "step : " << i
              << " E0 = " << std::setw(14) << std::fixed
              << std::setprecision(kEnergyOutputPrecision) << energy
              << " Grad Norm :" << std::setw(8) << std::fixed
              << std::setprecision(kEnergyOutputPrecision) << gradient.NormSquare()
              << std::endl;
    split_index_tps += (-step_length) * gradient;
  }
  EXPECT_GE(energy, energy_exact);
  EXPECT_NEAR(energy, energy_exact, 1e-5);
  return energy;
}

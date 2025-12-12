// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-12-11
*
* Description: QuantumLiquids/PEPS project. Unittests for BMPSContractor
*/

#include <bitset>
#include <fstream>
#include <complex>
#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/api/conversions.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps_contractor.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using qlten::special_qn::ZnQN;
using qlten::special_qn::fZ2QN;

///< Exact solution for Finite-size OBC Square Ising model
class SquareIsingModel {
  public:
    SquareIsingModel(size_t lx, size_t ly, double temperature)
      : lx_(lx), ly_(ly),
        N_(lx * ly),
        temperature_(temperature) {
      if (lx_ < ly_) {
        std::swap(lx_, ly_);
      }
      transfer_mat_dim_ = (1 << ly_);
      transfer_matrix_ = std::vector<std::vector<double> >(transfer_mat_dim_,
                                                           std::vector<double>(transfer_mat_dim_, 0));
      boundary_vec_ = std::vector<double>(transfer_mat_dim_, 0);
    }

    double CalculateExactFreeEnergy() {
      // Calculate the transfer matrix
      CalculateBoundaryVec_();
      CalculateTransferMatrix_();

      // Calculate the partition function using the transfer matrix
      double partition_function = CalculatePartitionFunction();
      // Calculate the free energy
      double free_energy = -log(partition_function) / N_ * temperature_;

      return free_energy;
    }

    double CalculatePartitionFunction() {
      std::vector<double> current_state(boundary_vec_);
      std::vector<double> next_state(transfer_matrix_.size(), 0.0);

      for (size_t i = 0; i < lx_ - 1; ++i) {
        for (size_t j = 0; j < transfer_matrix_.size(); ++j) {
          for (size_t k = 0; k < transfer_matrix_.size(); ++k) {
            next_state[k] += current_state[j] * transfer_matrix_[j][k];
          }
        }
        std::swap(current_state, next_state);
        std::fill(next_state.begin(), next_state.end(), 0.0);
      }

      double partition_function = 0.0;
      for (size_t i = 0; i < boundary_vec_.size(); ++i) {
        partition_function += current_state[i] * boundary_vec_[i];
      }
      return partition_function;
    }

  private:
    void CalculateTransferMatrix_() {
      for (size_t row = 0; row < transfer_mat_dim_; ++row) {
        std::bitset<64> config(row);
        double e_row = CalHalfEnergyChain_(config);
        for (size_t j = row; j < transfer_mat_dim_; ++j) {
          std::bitset<64> next_config(j);
          double e = e_row + CalHalfEnergyChain_(next_config) + CalLadderEnergy_(config, next_config);
          transfer_matrix_[row][j] = exp(-e / temperature_);
          if (row != j) {
            transfer_matrix_[j][row] = transfer_matrix_[row][j];
          }
        }
      }
    }

    void CalculateBoundaryVec_() {
      for (size_t idx = 0; idx < transfer_mat_dim_; ++idx) {
        std::bitset<64> config(idx);
        boundary_vec_[idx] = exp(-CalHalfEnergyChain_(config) / temperature_);
      }
    }

    template<size_t N>
    [[nodiscard]] double CalHalfEnergyChain_(const std::bitset<N> &config) const {
      std::bitset<N> shift_config = (config >> 1);
      size_t different_bond_num = (config ^ shift_config).count() - config[ly_ - 1];
      size_t bond_num = ly_ - 1;
      return (double) different_bond_num - (double) bond_num / 2.0; //FM
    }

    template<size_t N>
    [[nodiscard]] double CalLadderEnergy_(const std::bitset<N> &config, const std::bitset<N> &next_config) const {
      size_t different_bond_num = (config ^ next_config).count();
      size_t bond_num = ly_;
      return 2.0 * different_bond_num - (double) bond_num; //FM
    }

    template<size_t N>
    double CalculateTransferMatrixEffEnergy_(const std::bitset<N> &config, const std::bitset<N> &next_config) const {
      return CalHalfEnergyChain_(config) + CalHalfEnergyChain_(next_config) + CalLadderEnergy_(config, next_config);
    }

    size_t lx_; // linear size
    size_t ly_; // linear size
    const size_t N_; // Site number
    const double temperature_; // Temperature
    size_t transfer_mat_dim_;
    std::vector<std::vector<double> > transfer_matrix_;
    std::vector<double> boundary_vec_;
};

struct OBCIsing2DTenNetWithoutZ2 : public testing::Test {
  using QNT = U1QN;
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<QNT>;
  using DQLTensor = QLTensor<QLTEN_Double, QNT>;
  using ZQLTensor = QLTensor<QLTEN_Complex, QNT>;

  const size_t Lx = 12;
  const size_t Ly = 12;
  const double beta = std::log(1 + std::sqrt(2.0)) / 2.0; // critical point
  IndexT vb_out = IndexT({QNSctT(QNT(0), 2)},
                         TenIndexDirType::OUT
  );
  IndexT vb_in = InverseIndex(vb_out);
  IndexT trivial_out = IndexT({QNSctT(QNT(0), 1)},
                              TenIndexDirType::OUT
  );
  IndexT trivial_in = InverseIndex(trivial_out);

  TensorNetwork2D<QLTEN_Double, QNT> dtn2d = TensorNetwork2D<QLTEN_Double, QNT>(Ly, Lx);
  TensorNetwork2D<QLTEN_Complex, QNT> ztn2d = TensorNetwork2D<QLTEN_Complex, QNT>(Ly, Lx);

  double F_ex;
  double Z_ex; //partition function, exact value
  double tn_free_en_norm_factor = 0.0;
  void SetUp() {
    DQLTensor boltzmann_weight = DQLTensor({vb_in, vb_out});
    double e = -1.0; // FM Ising
    boltzmann_weight({0, 0}) = std::exp(-1.0 * beta * e);
    boltzmann_weight({0, 1}) = std::exp(1.0 * beta * e);
    boltzmann_weight({1, 0}) = std::exp(1.0 * beta * e);
    boltzmann_weight({1, 1}) = std::exp(-1.0 * beta * e);

    DQLTensor core_ten_m = DQLTensor({vb_in, vb_out, vb_out, vb_in});
    for (size_t i = 0; i < 2; i++) {
      core_ten_m({i, i, i, i}) = 1.0;
    }
    DQLTensor t_m;
    {
      DQLTensor temp[3];
      Contract(&boltzmann_weight, {1}, &core_ten_m, {3}, temp);
      Contract(&boltzmann_weight, {0}, temp, {3}, &t_m);
      t_m.Transpose({2, 3, 0, 1});
    }

    for (size_t row = 1; row < Ly - 1; row++) {
      for (size_t col = 1; col < Lx - 1; col++) {
        dtn2d({row, col}) = t_m;
      }
    }

    DQLTensor core_ten_up = DQLTensor({vb_in, vb_out, vb_out, trivial_in});
    DQLTensor core_ten_left = DQLTensor({trivial_in, vb_out, vb_out, vb_in});
    DQLTensor core_ten_down = DQLTensor({vb_in, trivial_out, vb_out, vb_in});
    DQLTensor core_ten_right = DQLTensor({vb_in, vb_out, trivial_out, vb_in});
    for (size_t i = 0; i < 2; i++) {
      core_ten_left({0, i, i, i}) = 1.0;
      core_ten_up({i, i, i, 0}) = 1.0;
      core_ten_down({i, 0, i, i}) = 1.0;
      core_ten_right({i, i, 0, i}) = 1.0;
    }

    DQLTensor t_up, t_left, t_down, t_right; {
      DQLTensor temp[3];
      temp[0] = core_ten_up;
      temp[0].Transpose({3, 0, 1, 2});
      Contract(&boltzmann_weight, {0}, temp, {3}, &t_up);
      t_up.Transpose({2, 3, 0, 1});
    } {
      DQLTensor temp[3];
      Contract(&boltzmann_weight, {1}, &core_ten_left, {3}, temp);
      Contract(&boltzmann_weight, {0}, temp, {3}, &t_left);
      t_left.Transpose({2, 3, 0, 1});
    } {
      Contract(&boltzmann_weight, {1}, &core_ten_right, {3}, &t_right);
      t_right.Transpose({1, 2, 3, 0});
    } {
      DQLTensor temp[3];
      Contract(&boltzmann_weight, {1}, &core_ten_down, {3}, temp);
      Contract(&boltzmann_weight, {0}, temp, {3}, &t_down);
      t_down.Transpose({2, 3, 0, 1});
    }

    for (size_t row = 1; row < Ly - 1; row++) {
      dtn2d({row, 0}) = t_left;
      dtn2d({row, Lx - 1}) = t_right;
    }
    for (size_t col = 1; col < Lx - 1; col++) {
      dtn2d({0, col}) = t_up;
      dtn2d({Ly - 1, col}) = t_down;
    }

    DQLTensor core_ten_left_upper = DQLTensor({trivial_in, vb_out, vb_out, trivial_in});
    DQLTensor core_ten_left_lower = DQLTensor({trivial_in, trivial_out, vb_out, vb_in});
    DQLTensor core_ten_right_lower = DQLTensor({vb_in, trivial_out, trivial_out, vb_in});
    DQLTensor core_ten_right_upper = DQLTensor({vb_in, vb_out, trivial_out, trivial_in});
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 2; j++) {
        double elem = boltzmann_weight({i, j});
        core_ten_left_upper({0, i, j, 0}) = elem;
        core_ten_right_lower({i, 0, 0, j}) = elem;
      }
    }

    for (size_t i = 0; i < 2; i++) {
      double ten_elem = 1.0;
      core_ten_left_lower({0, 0, i, i}) = ten_elem;
      core_ten_right_upper({i, i, 0, 0}) = ten_elem;
    } {
      DQLTensor temp[3];
      Contract(&boltzmann_weight, {1}, &core_ten_left_lower, {3}, temp);
      core_ten_left_lower = DQLTensor();
      Contract(&boltzmann_weight, {0}, temp, {3}, &core_ten_left_lower);
      core_ten_left_lower.Transpose({2, 3, 0, 1});
    }

    dtn2d({0, 0}) = core_ten_left_upper;
    dtn2d({Ly - 1, 0}) = core_ten_left_lower;
    dtn2d({Ly - 1, Lx - 1}) = core_ten_right_lower;
    dtn2d({0, Lx - 1}) = core_ten_right_upper;

    std::default_random_engine random_engine;
    random_engine.seed(std::random_device{}());
    double rand_phase_sum(0.0);
    for (size_t row = 0; row < Ly; row++) {
      for (size_t col = 0; col < Lx; col++) {
        tn_free_en_norm_factor += std::log(dtn2d({row, col}).Normalize());
        double rand_phase = unit_even_distribution(random_engine); // 0<= rand_phase< 1
        ztn2d({row, col}) = ToComplex(dtn2d({row, col})) * exp(std::complex<double>(0, 2 * M_PI * rand_phase));
        rand_phase_sum += rand_phase;
      }
    }
    ztn2d({0, 0}) *= exp(std::complex<double>(0, -2 * M_PI * rand_phase_sum));
    
    // Note: TensorNetwork2D::InitBMPS removed, replaced by BMPSContractor::Init in tests
    // But TensorNetwork2D constructor might have done something if constructed from TPS.
    // Here we constructed from (rows, cols), so it's empty. We need to manually call InitBMPS in contractor.

    // calculate exact partition function
    SquareIsingModel model(Lx, Ly, 1.0 / beta);
    F_ex = model.CalculateExactFreeEnergy();
    Z_ex = std::exp(-F_ex * beta * Lx * Ly);
  } //SetUp
};

template<typename TenElemT, typename QNT>
std::vector<TenElemT> Contract2DTNUsingBMPSContractor(
  const TensorNetwork2D<TenElemT, QNT>& tn2d,
  BMPSTruncateParams<typename qlten::RealTypeTrait<TenElemT>::type> trunc_para
) {
  BMPSContractor<TenElemT, QNT> contractor(tn2d.rows(), tn2d.cols());
  contractor.Init(tn2d);

  std::vector<TenElemT> amplitudes;
  amplitudes.reserve(26);
  contractor.GrowBMPSForRow(tn2d, 2, trunc_para);
  contractor.InitBTen(tn2d, BTenPOSITION::LEFT, 2);
  contractor.GrowFullBTen(tn2d, BTenPOSITION::RIGHT, 2, 2, true);
  amplitudes.push_back(contractor.Trace(tn2d, {2, 0}, HORIZONTAL));
  amplitudes.push_back(contractor.ReplaceTNNSiteTrace(tn2d, {2, 0},
                                                HORIZONTAL,
                                                tn2d({2, 0}),
                                                tn2d({2, 1}),
                                                tn2d({2, 2})));
  contractor.BTenMoveStep(tn2d, BTenPOSITION::RIGHT);
  amplitudes.push_back(contractor.Trace(tn2d, {2, 1}, HORIZONTAL));
  amplitudes.push_back(contractor.ReplaceTNNSiteTrace(tn2d, {2, 1},
                                                HORIZONTAL,
                                                tn2d({2, 1}),
                                                tn2d({2, 2}),
                                                tn2d({2, 3})));

  contractor.GrowBMPSForCol(tn2d, 1, trunc_para);
  contractor.InitBTen(tn2d, BTenPOSITION::DOWN, 1);
  contractor.GrowFullBTen(tn2d, BTenPOSITION::UP, 1, 2, true);
  amplitudes.push_back(contractor.Trace(tn2d, {tn2d.rows() - 2, 1}, VERTICAL));
  contractor.BTenMoveStep(tn2d, BTenPOSITION::UP);
  amplitudes.push_back(contractor.Trace(tn2d, {tn2d.rows() - 3, 1}, VERTICAL));
  amplitudes.push_back(contractor.ReplaceTNNSiteTrace(tn2d, {tn2d.rows() - 3, 1},
                                                VERTICAL,
                                                tn2d({tn2d.rows() - 3, 1}),
                                                tn2d({tn2d.rows() - 2, 1}),
                                                tn2d({tn2d.rows() - 1, 1})));

  /***** HORIZONTAL MPS *****/
  contractor.GrowBMPSForRow(tn2d, 1, trunc_para);
  contractor.InitBTen2(tn2d, BTenPOSITION::LEFT, 1);
  contractor.GrowFullBTen2(tn2d, BTenPOSITION::RIGHT, 1, 2, true);

  amplitudes.push_back(contractor.ReplaceNNNSiteTrace(tn2d, {1, 0},
                                                LEFTDOWN_TO_RIGHTUP,
                                                HORIZONTAL,
                                                tn2d({2, 0}),
                                                tn2d({1, 1}))); // trace original tn

  amplitudes.push_back(contractor.ReplaceNNNSiteTrace(tn2d, {1, 0},
                                                LEFTUP_TO_RIGHTDOWN,
                                                HORIZONTAL,
                                                tn2d({1, 0}),
                                                tn2d({2, 1}))); // trace original tn

  contractor.BTen2MoveStep(tn2d, BTenPOSITION::RIGHT, 1);
  amplitudes.push_back(contractor.ReplaceNNNSiteTrace(tn2d, {1, 1},
                                                LEFTDOWN_TO_RIGHTUP,
                                                HORIZONTAL,
                                                tn2d({2, 1}),
                                                tn2d({1, 2}))); // trace original tn

  amplitudes.push_back(contractor.ReplaceNNNSiteTrace(tn2d, {1, 1},
                                                LEFTUP_TO_RIGHTDOWN,
                                                HORIZONTAL,
                                                tn2d({1, 1}),
                                                tn2d({2, 2}))); // trace original tn
  if constexpr (QLTensor<TenElemT, QNT>::IsFermionic()) {
    //since the code for VERTICAL NNN Trace and Sqrt5 Trace are not implemented.
    return amplitudes;
  }
  amplitudes.push_back(contractor.ReplaceSqrt5DistTwoSiteTrace(tn2d, {1, 0},
                                                         LEFTDOWN_TO_RIGHTUP,
                                                         HORIZONTAL,
                                                         tn2d({2, 0}),
                                                         tn2d({1, 2}))); // trace original tn
  amplitudes.push_back(contractor.ReplaceSqrt5DistTwoSiteTrace(tn2d, {1, 1},
                                                         LEFTDOWN_TO_RIGHTUP,
                                                         HORIZONTAL,
                                                         tn2d({2, 1}),
                                                         tn2d({1, 3}))); // trace original tn
  amplitudes.push_back(contractor.ReplaceSqrt5DistTwoSiteTrace(tn2d, {1, 0},
                                                         LEFTUP_TO_RIGHTDOWN,
                                                         HORIZONTAL,
                                                         tn2d({1, 0}),
                                                         tn2d({2, 2}))); // trace original tn
  amplitudes.push_back(contractor.ReplaceSqrt5DistTwoSiteTrace(tn2d, {1, 1},
                                                         LEFTUP_TO_RIGHTDOWN,
                                                         HORIZONTAL,
                                                         tn2d({1, 1}),
                                                         tn2d({2, 3}))); // trace original tn

  /***** VERTICAL MPS *****/
  contractor.GrowBMPSForCol(tn2d, 1, trunc_para);
  contractor.GrowFullBTen2(tn2d, BTenPOSITION::DOWN, 1, 2, true);
  contractor.GrowFullBTen2(tn2d, BTenPOSITION::UP, 1, 2, true);
  amplitudes.push_back(contractor.ReplaceNNNSiteTrace(tn2d, {2, 1},
                                                LEFTDOWN_TO_RIGHTUP,
                                                VERTICAL,
                                                tn2d({3, 1}),
                                                tn2d({2, 2}))); // trace original tn
  amplitudes.push_back(contractor.ReplaceNNNSiteTrace(tn2d, {2, 1},
                                                LEFTUP_TO_RIGHTDOWN,
                                                VERTICAL,
                                                tn2d({2, 1}),
                                                tn2d({3, 2}))); // trace original tn

  contractor.BTen2MoveStep(tn2d, BTenPOSITION::UP, 1);
  amplitudes.push_back(contractor.ReplaceNNNSiteTrace(tn2d, {1, 1},
                                                LEFTDOWN_TO_RIGHTUP,
                                                VERTICAL,
                                                tn2d({2, 1}),
                                                tn2d({1, 2}))); // trace original tn

  amplitudes.push_back(contractor.ReplaceNNNSiteTrace(tn2d, {1, 1},
                                                LEFTUP_TO_RIGHTDOWN,
                                                VERTICAL,
                                                tn2d({1, 1}),
                                                tn2d({2, 2}))); // trace original tn
  amplitudes.push_back(contractor.ReplaceSqrt5DistTwoSiteTrace(tn2d, {1, 1},
                                                         LEFTDOWN_TO_RIGHTUP,
                                                         VERTICAL,
                                                         tn2d({3, 1}),
                                                         tn2d({1, 2}))); // trace original tn
  amplitudes.push_back(contractor.ReplaceSqrt5DistTwoSiteTrace(tn2d, {1, 1},
                                                         LEFTUP_TO_RIGHTDOWN,
                                                         VERTICAL,
                                                         tn2d({1, 1}),
                                                         tn2d({3, 2}))); // trace original tn
  return amplitudes;
}

TEST_F(OBCIsing2DTenNetWithoutZ2, TestDynamicUpdateAndPunchHole) {
  BMPSContractor<QLTEN_Double, QNT> contractor(Ly, Lx);
  contractor.Init(dtn2d);
  
  // 1. Initial calculation
  // Use a small bond dimension for speed
  auto trunc_para = BMPSTruncateParams<qlten::QLTEN_Double>::SVD(4, 10, 1e-10);
  
  contractor.GrowBMPSForRow(dtn2d, 2, trunc_para);
  // Build BOTH left/right boundary tensors for this row slice.
  // PunchHole(HORIZONTAL) needs bten_set_[LEFT][col] and bten_set_[RIGHT][cols-col-1].
  contractor.GrowFullBTen(dtn2d, BTenPOSITION::LEFT, 2, 2, true);
  contractor.GrowFullBTen(dtn2d, BTenPOSITION::RIGHT, 2, 2, true);
  
  double val1 = contractor.Trace(dtn2d, {2, 0}, HORIZONTAL);
  
  // 2. Test PunchHole
  // Verify dimensions and consistency with Trace
  auto hole_ten = contractor.PunchHole(dtn2d, {2, 1}, HORIZONTAL);
  EXPECT_FALSE(hole_ten.IsDefault());
  EXPECT_GE(hole_ten.Rank(), 4);

  // Consistency check: Contracting Hole with Site Tensor should equal Trace
  // Note: Trace(site, HORIZONTAL) contracts the bond between site and site+(0,1).
  // Ideally, both represent the full network contraction (Z).
  double trace_val_at_site = contractor.Trace(dtn2d, {2, 1}, HORIZONTAL);
  
  DQLTensor site_tensor = dtn2d({2, 1});
  DQLTensor res;
  // Bosonic contraction: Hole indices (0,1,2,3) match Site indices (0,1,2,3)
  // Directions are opposite/compatible by definition of Hole.
  Contract(&hole_ten, {0, 1, 2, 3}, &site_tensor, {0, 1, 2, 3}, &res);
  double hole_contraction_val = res();
  
  EXPECT_NEAR(trace_val_at_site, hole_contraction_val, 1e-10) << "PunchHole contraction should match Trace result";

  // 3. Simulate VMC update
  // Change tensor at {2, 1}
  auto old_ten = dtn2d({2, 1});
  auto new_ten = old_ten * 0.5; // Simple scaling
  dtn2d({2, 1}) = new_ten;
  
  contractor.InvalidateEnvs({2, 1});
  
  // 4. Re-calculate
  // We need to rebuild the environment that was invalidated
  // Since InvalidateEnvs chops off the BMPS, we can just call Grow again.
  // However, GrowBMPSForRow/GrowFullBTen logic usually appends.
  // But InitBTen clears BTen.
  
  // Re-grow row BMPS (it will pick up from where it was cut)
  contractor.GrowBMPSForRow(dtn2d, 2, trunc_para);
  
  // Rebuild BOTH left/right boundary tensors for this row slice.
  contractor.GrowFullBTen(dtn2d, BTenPOSITION::LEFT, 2, 2, true);
  contractor.GrowFullBTen(dtn2d, BTenPOSITION::RIGHT, 2, 2, true);
  
  double val2 = contractor.Trace(dtn2d, {2, 0}, HORIZONTAL);
  
  // Since we scaled one tensor by 0.5, the trace should scale by 0.5
  // (Assuming Trace includes this tensor, which it does)
  EXPECT_NEAR(val2, val1 * 0.5, 1e-10);
}

TEST_F(OBCIsing2DTenNetWithoutZ2, TestIsingTenNetRealNumberContraction) {
  // Test with Variational2Site compression
  auto trunc_para_2site = BMPSTruncateParams<qlten::QLTEN_Double>::Variational2Site(10, 30, 1e-15, 1e-14, 10);
  auto Z_set = Contract2DTNUsingBMPSContractor(dtn2d, trunc_para_2site);
  for (size_t i = 1; i < Z_set.size(); i++) {
    EXPECT_NEAR(-(std::log(Z_set[i]) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
  }

  // Test with Variational1Site compression
  auto trunc_para_1site = BMPSTruncateParams<qlten::QLTEN_Double>::Variational1Site(10, 30, 1e-15, 1e-14, 10);
  Z_set = Contract2DTNUsingBMPSContractor(dtn2d, trunc_para_1site);
  for (size_t i = 1; i < Z_set.size(); i++) {
    EXPECT_NEAR(-(std::log(Z_set[i]) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
  }

  // Test with SVD compression
  auto trunc_para_svd = BMPSTruncateParams<qlten::QLTEN_Double>::SVD(10, 30, 1e-15);
  Z_set = Contract2DTNUsingBMPSContractor(dtn2d, trunc_para_svd);
  for (size_t i = 1; i < Z_set.size(); i++) {
    EXPECT_NEAR(-(std::log(Z_set[i]) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
  }
}


/**
 * Open Boundary Condition two-dimensional Ising model's Tensor network, with imposing Z2 symmetry.
 */
struct OBCIsing2DZ2TenNet : public testing::Test {
  using Z2QN = ZnQN<2>;
  using QNT = Z2QN;
  using IndexT = Index<Z2QN>;
  using QNSctT = QNSector<Z2QN>;
  using QNSctVecT = QNSectorVec<Z2QN>;
  using DQLTensor = QLTensor<QLTEN_Double, Z2QN>;
  using ZQLTensor = QLTensor<QLTEN_Complex, Z2QN>;

  const size_t Lx = 10;
  const size_t Ly = 24;
  const double beta = std::log(1 + std::sqrt(2.0)) / 2.0; // critical point
  IndexT vb_out = IndexT({
                           QNSctT(Z2QN(0), 1),
                           QNSctT(Z2QN(1), 1)
                         },
                         TenIndexDirType::OUT
  );
  IndexT vb_in = InverseIndex(vb_out);
  IndexT trivial_out = IndexT({QNSctT(Z2QN(0), 1)},
                              TenIndexDirType::OUT
  );
  IndexT trivial_in = InverseIndex(trivial_out);

  TensorNetwork2D<QLTEN_Double, QNT> dtn2d = TensorNetwork2D<QLTEN_Double, QNT>(Ly, Lx);
  TensorNetwork2D<QLTEN_Complex, QNT> ztn2d = TensorNetwork2D<QLTEN_Complex, QNT>(Ly, Lx);

  double F_ex;
  double Z_ex; //partition function, exact value
  double tn_free_en_norm_factor = 0.0;
  void SetUp() {
    DQLTensor boltzmann_weight = DQLTensor({vb_in, vb_out});
    double e = -1.0; // FM Ising
    boltzmann_weight({0, 0}) = std::exp(-1.0 * beta * e) + std::exp(1.0 * beta * e);
    boltzmann_weight({1, 1}) = std::exp(-1.0 * beta * e) - std::exp(1.0 * beta * e);
    auto boltzmann_weight_sqrt = boltzmann_weight;
    boltzmann_weight_sqrt({0, 0}) = std::sqrt(boltzmann_weight_sqrt({0, 0}));
    boltzmann_weight_sqrt({1, 1}) = std::sqrt(boltzmann_weight_sqrt({1, 1}));
    DQLTensor core_ten_m = DQLTensor({vb_in, vb_out, vb_out, vb_in});
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 2; j++) {
        for (size_t k = 0; k < 2; k++) {
          size_t l = (j + k + 2 - i) % 2;
          core_ten_m({i, j, k, l}) = 0.5;
        }
      }
    }
    DQLTensor t_m; // = DQLTensor({vb_in, vb_out, vb_out, vb_in});
    {
      DQLTensor temp[3];
      Contract(&boltzmann_weight_sqrt, {1}, &core_ten_m, {3}, temp);
      Contract(&boltzmann_weight_sqrt, {0}, temp, {3}, temp + 1);
      Contract(&boltzmann_weight_sqrt, {0}, temp + 1, {3}, temp + 2);
      Contract(&boltzmann_weight_sqrt, {1}, temp + 2, {3}, &t_m);
    }

    for (size_t row = 1; row < Ly - 1; row++) {
      for (size_t col = 1; col < Lx - 1; col++) {
        dtn2d({row, col}) = t_m;
      }
    }

    DQLTensor core_ten_up = DQLTensor({vb_in, vb_out, vb_out, trivial_in});
    DQLTensor core_ten_left = DQLTensor({trivial_in, vb_out, vb_out, vb_in});
    DQLTensor core_ten_down = DQLTensor({vb_in, trivial_out, vb_out, vb_in});
    DQLTensor core_ten_right = DQLTensor({vb_in, vb_out, trivial_out, vb_in});
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 2; j++) {
        size_t k = (i + j) % 2;
        core_ten_left({0, i, j, k}) = 1.0 / std::sqrt(2.0);
        core_ten_up({i, j, k, 0}) = 1.0 / std::sqrt(2.0);
        core_ten_down({i, 0, j, k}) = 1.0 / std::sqrt(2.0);
        core_ten_right({i, j, 0, k}) = 1.0 / std::sqrt(2.0);
      }
    }
    DQLTensor t_up, t_left, t_down, t_right; {
      DQLTensor temp[3];
      temp[0] = core_ten_up;
      temp[0].Transpose({3, 0, 1, 2});
      Contract(&boltzmann_weight_sqrt, {0}, temp, {3}, temp + 1);
      Contract(&boltzmann_weight_sqrt, {0}, temp + 1, {3}, temp + 2);
      Contract(&boltzmann_weight_sqrt, {1}, temp + 2, {3}, &t_up);
    } {
      DQLTensor temp[3];
      Contract(&boltzmann_weight_sqrt, {1}, &core_ten_left, {3}, temp);
      Contract(&boltzmann_weight_sqrt, {0}, temp, {3}, temp + 1);
      Contract(&boltzmann_weight_sqrt, {0}, temp + 1, {3}, temp + 2);
      (temp + 2)->Transpose({3, 0, 1, 2});
      t_left = temp[2];
    } {
      DQLTensor temp[3];
      Contract(&boltzmann_weight_sqrt, {1}, &core_ten_right, {3}, temp);
      temp->Transpose({3, 0, 1, 2});
      Contract(&boltzmann_weight_sqrt, {0}, temp, {3}, temp + 2);
      Contract(&boltzmann_weight_sqrt, {1}, temp + 2, {3}, &t_right);
    } {
      DQLTensor temp[3];
      Contract(&boltzmann_weight_sqrt, {1}, &core_ten_down, {3}, temp);
      Contract(&boltzmann_weight_sqrt, {0}, temp, {3}, temp + 1);
      (temp + 1)->Transpose({3, 0, 1, 2});
      Contract(&boltzmann_weight_sqrt, {1}, temp + 1, {3}, &t_down);
    }

    for (size_t row = 1; row < Ly - 1; row++) {
      dtn2d({row, 0}) = t_left;
      dtn2d({row, Lx - 1}) = t_right;
    }
    for (size_t col = 1; col < Lx - 1; col++) {
      dtn2d({0, col}) = t_up;
      dtn2d({Ly - 1, col}) = t_down;
    }

    DQLTensor core_ten_left_upper = DQLTensor({trivial_in, vb_out, vb_out, trivial_in});
    DQLTensor core_ten_left_lower = DQLTensor({trivial_in, trivial_out, vb_out, vb_in});
    DQLTensor core_ten_right_lower = DQLTensor({vb_in, trivial_out, trivial_out, vb_in});
    DQLTensor core_ten_right_upper = DQLTensor({vb_in, vb_out, trivial_out, trivial_in});
    for (size_t i = 0; i < 2; i++) {
      double ten_elem = std::exp(-1.0 * beta * e) + (i == 0 ? 1.0 : -1.0) * std::exp(1.0 * beta * e);
      core_ten_left_upper({0, i, i, 0}) = ten_elem;
      core_ten_left_lower({0, 0, i, i}) = ten_elem;
      core_ten_right_lower({i, 0, 0, i}) = ten_elem;
      core_ten_right_upper({i, i, 0, 0}) = ten_elem;
    }

    dtn2d({0, 0}) = core_ten_left_upper;
    dtn2d({Ly - 1, 0}) = core_ten_left_lower;
    dtn2d({Ly - 1, Lx - 1}) = core_ten_right_lower;
    dtn2d({0, Lx - 1}) = core_ten_right_upper;

    std::default_random_engine random_engine;
    random_engine.seed(std::random_device{}());
    double rand_phase_sum(0.0);
    for (size_t row = 0; row < Ly; row++) {
      for (size_t col = 0; col < Lx; col++) {
        tn_free_en_norm_factor += std::log(dtn2d({row, col}).Normalize());
        double rand_phase = unit_even_distribution(random_engine); // 0<= rand_phase< 1
        ztn2d({row, col}) = ToComplex(dtn2d({row, col})) * exp(std::complex<double>(0, 2 * M_PI * rand_phase));
        rand_phase_sum += rand_phase;
      }
    }
    ztn2d({0, 0}) *= exp(std::complex<double>(0, -2 * M_PI * rand_phase_sum));
    
    // Note: InitBMPS removed from here, as BMPSContractor handles it.

    // calculate exact partition function
    SquareIsingModel model(Lx, Ly, 1.0 / beta);
    F_ex = model.CalculateExactFreeEnergy();
    Z_ex = std::exp(-F_ex * beta * Lx * Ly);
  } //SetUp
};

TEST_F(OBCIsing2DZ2TenNet, TestTrace) {
  // Test with SVD compression
  auto trunc_para_svd = BMPSTruncateParams<qlten::QLTEN_Double>::SVD(1, 10, 1e-15);
  auto dZ_set = Contract2DTNUsingBMPSContractor(dtn2d, trunc_para_svd);
  for (size_t i = 0; i < dZ_set.size(); i++) {
    EXPECT_NEAR(-(std::log(dZ_set[i]) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
  }
  auto zZ_set = Contract2DTNUsingBMPSContractor(ztn2d, trunc_para_svd);
  for (size_t i = 0; i < zZ_set.size(); i++) {
    EXPECT_NEAR(-(std::log(zZ_set[i].real()) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
    EXPECT_NEAR(zZ_set[i].imag(), 0.0, 1e-15);
  }

  // Test with Variational1Site compression
  auto trunc_para_var1 = BMPSTruncateParams<qlten::QLTEN_Double>::Variational1Site(1, 10, 1e-15, 1e-14, 10);
  dZ_set = Contract2DTNUsingBMPSContractor(dtn2d, trunc_para_var1);
  for (size_t i = 0; i < dZ_set.size(); i++) {
    EXPECT_NEAR(-(std::log(dZ_set[i]) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
  }
  zZ_set = Contract2DTNUsingBMPSContractor(ztn2d, trunc_para_var1);
  for (size_t i = 0; i < zZ_set.size(); i++) {
    EXPECT_NEAR(-(std::log(zZ_set[i].real()) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
    EXPECT_NEAR(zZ_set[i].imag(), 0.0, 1e-15);
  }

  // Test with SVD compression (again)
  auto trunc_para_svd2 = BMPSTruncateParams<qlten::QLTEN_Double>::SVD(1, 10, 1e-15);
  dZ_set = Contract2DTNUsingBMPSContractor(dtn2d, trunc_para_svd2);
  for (size_t i = 0; i < dZ_set.size(); i++) {
    EXPECT_NEAR(-(std::log(dZ_set[i]) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
  }
  zZ_set = Contract2DTNUsingBMPSContractor(ztn2d, trunc_para_svd2);
  for (size_t i = 0; i < zZ_set.size(); i++) {
    EXPECT_NEAR(-(std::log(zZ_set[i].real()) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
    EXPECT_NEAR(zZ_set[i].imag(), 0.0, 1e-15);
  }
}

struct ProjectedtJTensorNetwork : public testing::Test {
  using QNT = fZ2QN;
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using Tensor = QLTensor<QLTEN_Double, fZ2QN>;
  using TPSSampleNNFlipT = MCUpdateSquareNNExchange;
  using TPSSampleTNNFlipT = MCUpdateSquareTNN3SiteExchange;

  size_t Lx = 24;
  size_t Ly = 20;
  size_t N = Lx * Ly;
  double t = 3;
  double J = 1;
  double doping = 0.125; // actually the data is doping 0.124 from iPEPS simple update
  size_t hole_num = size_t(double(N) * doping);
  size_t num_up = (N - hole_num) / 2;
  size_t num_down = (N - hole_num) / 2;
  IndexT loc_phy_ket = IndexT({
                                QNSctT(fZ2QN(1), 2), // |up>, |down>
                                QNSctT(fZ2QN(0), 1)
                              },
                              // |0> empty state
                              TenIndexDirType::IN
  );

  size_t Db_min = 16;
  size_t Db_max = 50;

  size_t MC_samples = 100;
  size_t WarmUp = 100;

  TensorNetwork2D<QLTEN_Double, QNT> dtn2d = TensorNetwork2D<QLTEN_Double, QNT>(Ly, Lx);
  TensorNetwork2D<QLTEN_Complex, QNT> ztn2d = TensorNetwork2D<QLTEN_Complex, QNT>(Ly, Lx);

  void SetUp() {
    qlten::hp_numeric::SetTensorManipulationThreads(1);
    SplitIndexTPS<QLTEN_Double, fZ2QN> split_idx_tps = CreateFiniteSizeOBCtJTPS();

    auto trun_para = BMPSTruncateParams<qlten::QLTEN_Double>(Db_min,
                                      Db_max,
                                      1e-10,
                                      CompressMPSScheme::SVD_COMPRESS,
                                      std::make_optional<double>(1e-14),
                                      std::make_optional<size_t>(10));
    Configuration config({
      {1, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
      {0, 1, 1, 0, 0, 2, 1, 2, 0, 2, 0, 0, 2, 1, 0, 2, 0, 1, 0, 2, 0, 0, 2, 1},
      {1, 2, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 0, 2, 1, 1, 0, 1, 0, 1, 0, 1, 0},
      {1, 0, 0, 1, 1, 2, 0, 1, 0, 2, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 2, 0, 1},
      {1, 1, 2, 0, 1, 1, 1, 0, 2, 1, 2, 2, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0},
      {2, 0, 0, 1, 2, 0, 0, 1, 1, 0, 2, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1},
      {1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 0},
      {0, 1, 0, 1, 1, 1, 0, 1, 0, 2, 2, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
      {1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 0},
      {0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1},
      {1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2, 1, 0, 1, 0, 2, 1, 0, 1, 2, 1, 2, 0, 2},
      {0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 2, 1, 2, 0, 0, 0, 1, 2, 1},
      {1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 2, 2, 1},
      {0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0},
      {1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 1, 0, 2},
      {0, 1, 0, 1, 0, 2, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0, 1, 2, 1},
      {1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 1, 1, 2, 1, 0, 0, 0},
      {0, 1, 1, 2, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1},
      {1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 2, 1, 0, 0, 0},
      {0, 1, 0, 1, 2, 0, 0, 2, 1, 2, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2}
    });
    //    config.Random({N * 7 / 16, N * 7 / 16, N / 8});
    TPSWaveFunctionComponent<QLTEN_Double, fZ2QN> tps_sample(split_idx_tps, config, trun_para);
    dtn2d = tps_sample.tn;
    
    // Note: No need to delete inner BMPS, dtn2d is treated as raw data.
    
    for (size_t row = 0; row < Ly; row++) {
      for (size_t col = 0; col < Lx; col++) {
        ztn2d({row, col}) = ToComplex(dtn2d({row, col}));
      }
    }
    // ztn2d.InitBMPS(); // Handled by Contractor Init
  } //SetUp

  SplitIndexTPS<QLTEN_Double, fZ2QN> CreateFiniteSizeOBCtJTPS() {
    std::string test_data_dir = std::string(TEST_SOURCE_DIR) + "/test_data/";
    Tensor ten_a, ten_b;
    std::ifstream ifs;

    std::string path_a = test_data_dir + "ipeps_tJ_ta_doping0.125.qlten";
    ifs.open(path_a);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open test data file: " + path_a);
    }
    ifs >> ten_a;
    ifs.close();

    std::string path_b = test_data_dir + "ipeps_tJ_tb_doping0.125.qlten";
    ifs.open(path_b);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open test data file: " + path_b);
    }
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
    SplitIndexTPS<QLTEN_Double, fZ2QN> split_idx_tps = SplitIndexTPS<QLTEN_Double, fZ2QN>::FromTPS(tps);
    split_idx_tps.NormalizeAllSite();
    split_idx_tps *= 3.0;
    return split_idx_tps;
  }
};

TEST_F(ProjectedtJTensorNetwork, TestTrace) {
  BMPSTruncateParams<qlten::QLTEN_Double> trunc_para = BMPSTruncateParams<qlten::QLTEN_Double>(Db_min,
                                                 Db_max,
                                                 1e-15,
                                                 CompressMPSScheme::SVD_COMPRESS,
                                                 std::make_optional<double>(1e-14),
                                                 std::make_optional<size_t>(10));
  auto dpsi_set = Contract2DTNUsingBMPSContractor(dtn2d, trunc_para);
  for (size_t i = 1; i < dpsi_set.size(); i++) {
    EXPECT_NEAR(std::abs(dpsi_set[i]) / std::abs(dpsi_set[0]), 1.0, 1e-7);
  }
  auto zpsi_set = Contract2DTNUsingBMPSContractor(ztn2d, trunc_para);
  for (size_t i = 1; i < zpsi_set.size(); i++) {
    EXPECT_NEAR(std::abs(zpsi_set[i].real()) / std::abs(dpsi_set[0]), 1.0, 1e-7);
    EXPECT_NEAR(zpsi_set[i].imag(), 0.0, 1e-15);
  }
}

/**
 * @note Tests based on this class should be run after simple update.
 */
struct ProjectedSpinTenNet : public testing::Test {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;

  const size_t Lx = 4; // cols
  const size_t Ly = 4; // rows

#ifdef U1SYM
  IndexT pb_out = IndexT({
                           QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                           QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)
                         },
                         TenIndexDirType::OUT
  );
#else
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT
  );
#endif

  IndexT pb_in = InverseIndex(pb_out);

  Configuration config = Configuration(Ly, Lx);

  TensorNetwork2D<QLTEN_Double, U1QN> tn2d = TensorNetwork2D<QLTEN_Double, U1QN>(Ly, Lx);

  BMPSTruncateParams<qlten::QLTEN_Double> trunc_para = BMPSTruncateParams<qlten::QLTEN_Double>::Variational2Site(4, 8, 1e-12, 1e-14, 10);

  using Tensor = QLTensor<QLTEN_Double, U1QN>;
  void SetUp() {
    SquareLatticePEPS<QLTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
    std::vector<std::vector<size_t> > activates(Ly, std::vector<size_t>(Lx));
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        size_t sz_int = x + y;
        activates[y][x] = sz_int % 2;
      }
    }
    peps0.Initial(activates);

    Tensor ham_hei_nn({pb_in, pb_out, pb_in, pb_out});
    ham_hei_nn({0, 0, 0, 0}) = 0.25;
    ham_hei_nn({1, 1, 1, 1}) = 0.25;
    ham_hei_nn({1, 1, 0, 0}) = -0.25;
    ham_hei_nn({0, 0, 1, 1}) = -0.25;
    ham_hei_nn({0, 1, 1, 0}) = 0.5;
    ham_hei_nn({1, 0, 0, 1}) = 0.5;

    SimpleUpdatePara update_para(10, 0.1, 1, 4, 1e-15);
    SquareLatticeNNSimpleUpdateExecutor<QLTEN_Double, U1QN> su_exe(update_para, peps0, ham_hei_nn);
    su_exe.Execute();

    TPS<QLTEN_Double, U1QN> tps = qlpeps::ToTPS<QLTEN_Double, U1QN>(su_exe.GetPEPS());
    SplitIndexTPS<QLTEN_Double, U1QN> split_index_tps = SplitIndexTPS<QLTEN_Double, U1QN>::FromTPS(tps);
    for (size_t i = 0; i < Lx; i++) {
      //col index
      for (size_t j = 0; j < Ly; j++) {
        //row index
        config({j, i}) = (i + j) % 2;
      }
    }
    tn2d = TensorNetwork2D<QLTEN_Double, U1QN>(split_index_tps, config);
  }
};

TEST_F(ProjectedSpinTenNet, HeisenbergD4WaveFunctionComponnet) {
  auto psi = Contract2DTNUsingBMPSContractor(tn2d, trunc_para);
  for (size_t i = 1; i < psi.size(); i++) {
    EXPECT_NEAR(1, psi[i] / psi[0], 1e-10);
  }
}

int main(int argc, char *argv[]) {

  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}

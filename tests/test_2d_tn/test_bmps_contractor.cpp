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
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h"

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
  contractor.ShiftBTenWindow(tn2d, BTenPOSITION::RIGHT);
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
  contractor.ShiftBTenWindow(tn2d, BTenPOSITION::UP);
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

  contractor.ShiftBTen2Window(tn2d, BTenPOSITION::RIGHT, 1);
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

  contractor.ShiftBTen2Window(tn2d, BTenPOSITION::UP, 1);
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
  
  contractor.EraseEnvsAfterUpdate({2, 1});
  
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
 * @brief Test BMPSWalker BTen cache interface for fermionic tensor networks.
 *
 * This test verifies that the fermionic BTen contraction patterns work correctly
 * by comparing TraceWithBTen results against BMPSContractor::Trace reference values.
 *
 * Key differences from bosonic case:
 * - Fermionic tensors use FuseIndex for correct fermion sign handling
 * - Vacuum BTen has 4 indices (vs 3 for bosonic)
 * - Different contraction patterns with Transpose operations
 */
TEST_F(ProjectedtJTensorNetwork, BMPSWalkerFermionicBTenTest) {
  using TenElemT = QLTEN_Double;
  using ContractorT = BMPSContractor<TenElemT, QNT>;
  using WalkerT = ContractorT::BMPSWalker;
  using TransferMPO = std::vector<Tensor *>;

  BMPSTruncateParams<qlten::QLTEN_Double> trunc_para = BMPSTruncateParams<qlten::QLTEN_Double>(Db_min,
                                                 Db_max,
                                                 1e-15,
                                                 CompressMPSScheme::SVD_COMPRESS,
                                                 std::make_optional<double>(1e-14),
                                                 std::make_optional<size_t>(10));

  // Initialize contractor
  ContractorT contractor(dtn2d.rows(), dtn2d.cols());
  contractor.Init(dtn2d);

  // Grow BMPS environments for a middle row
  const size_t test_row = Ly / 2;
  contractor.GrowBMPSForRow(dtn2d, test_row, trunc_para);

  // Get walker from UP direction
  WalkerT walker = contractor.GetWalker(dtn2d, UP);
  EXPECT_EQ(walker.GetStackSize(), test_row + 1) << "Walker should have absorbed rows [0, test_row)";

  // Build MPO for test_row (needed for BTen operations)
  TransferMPO row_mpo;
  row_mpo.reserve(dtn2d.cols());
  for (size_t col = 0; col < dtn2d.cols(); ++col) {
    row_mpo.push_back(const_cast<Tensor*>(&dtn2d({test_row, col})));
  }

  // Get DOWN boundary
  const auto& down_stack = contractor.GetBMPS(DOWN);
  size_t needed_down_idx = dtn2d.rows() - 1 - test_row;
  ASSERT_LT(needed_down_idx, down_stack.size()) << "DOWN stack should cover rows below test_row";
  const auto& bottom_env = down_stack[needed_down_idx];

  // Use BMPSContractor::Trace as reference (it already supports fermionic tensors)
  // Need to initialize BOTH left and right BTen for Trace to work
  const size_t mid_col = Lx / 2;
  contractor.InitBTen(dtn2d, BTenPOSITION::LEFT, test_row);
  contractor.GrowFullBTen(dtn2d, BTenPOSITION::LEFT, test_row, mid_col, true);
  contractor.InitBTen(dtn2d, BTenPOSITION::RIGHT, test_row);
  contractor.GrowFullBTen(dtn2d, BTenPOSITION::RIGHT, test_row, mid_col + 1, true);
  TenElemT reference_val = contractor.Trace(dtn2d, {test_row, mid_col}, HORIZONTAL);
  EXPECT_NE(reference_val, 0.0) << "Reference Trace should return non-zero for valid fermionic contraction";

  // Test 1: Initialize LEFT BTen from col 0 to mid_col
  walker.InitBTenLeft(row_mpo, bottom_env, mid_col);
  EXPECT_EQ(walker.GetBTenLeftCol(), mid_col)
      << "InitBTenLeft should grow left BTen to target_col";

  // Test 2: Initialize RIGHT BTen from col Lx-1 to mid_col+1
  walker.InitBTenRight(row_mpo, bottom_env, mid_col);
  EXPECT_EQ(walker.GetBTenRightCol(), mid_col + 1)
      << "InitBTenRight should set right edge to target_col + 1";

  // Test 3: TraceWithBTen should match reference_val
  const Tensor& mid_site = dtn2d({test_row, mid_col});
  TenElemT bten_val = walker.TraceWithBTen(mid_site, mid_col, bottom_env);
  EXPECT_NEAR(std::abs(bten_val), std::abs(reference_val), 1e-8 * std::abs(reference_val))
      << "TraceWithBTen should match BMPSContractor::Trace result for fermionic tensors";

  // Test 4: Step-by-step growth and trace at different columns
  walker.ClearBTen();

  // Grow LEFT BTen step by step
  walker.InitBTenLeft(row_mpo, bottom_env, 0);
  for (size_t col = 0; col < mid_col; ++col) {
    walker.GrowBTenLeftStep(row_mpo, bottom_env);
    EXPECT_EQ(walker.GetBTenLeftCol(), col + 1)
        << "GrowBTenLeftStep should advance left edge by 1";
  }

  // Initialize RIGHT BTen
  walker.InitBTenRight(row_mpo, bottom_env, mid_col);

  // Trace again - should still match
  TenElemT bten_val2 = walker.TraceWithBTen(mid_site, mid_col, bottom_env);
  EXPECT_NEAR(std::abs(bten_val2), std::abs(reference_val), 1e-8 * std::abs(reference_val))
      << "Step-by-step grown BTen should produce same result";

  // Test 5: Test GrowBTenRightStep
  walker.ClearBTen();

  // Initialize RIGHT BTen at rightmost column
  walker.InitBTenRight(row_mpo, bottom_env, Lx - 1);
  EXPECT_EQ(walker.GetBTenRightCol(), Lx) << "InitBTenRight at Lx-1 should set right col to Lx";

  // Grow RIGHT BTen leftward step by step
  for (size_t step = 0; step < Lx - 1 - mid_col; ++step) {
    walker.GrowBTenRightStep(row_mpo, bottom_env);
    EXPECT_EQ(walker.GetBTenRightCol(), Lx - 1 - step)
        << "GrowBTenRightStep should decrease right edge by 1";
  }
  EXPECT_EQ(walker.GetBTenRightCol(), mid_col + 1);

  // Now initialize LEFT BTen
  walker.InitBTenLeft(row_mpo, bottom_env, mid_col);

  // Trace - should match reference
  TenElemT bten_val3 = walker.TraceWithBTen(mid_site, mid_col, bottom_env);
  EXPECT_NEAR(std::abs(bten_val3), std::abs(reference_val), 1e-8 * std::abs(reference_val))
      << "Right-to-left grown BTen should produce same result";
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

/**
 * @brief Test BMPSWalker basic functionality.
 *
 * Verifies that:
 * 1. GetWalker correctly forks a walker from the contractor's internal state.
 * 2. Walker::EvolveStep correctly absorbs lattice layers.
 * 3. Walker's BMPS state is independent of the contractor's internal state.
 */
TEST_F(ProjectedSpinTenNet, BMPSWalkerBasicTest) {
  using TenElemT = QLTEN_Double;
  using QNT = U1QN;
  using ContractorT = BMPSContractor<TenElemT, QNT>;
  using WalkerT = typename ContractorT::BMPSWalker;

  // Initialize contractor with row environments
  ContractorT contractor(tn2d.rows(), tn2d.cols());
  contractor.Init(tn2d);

  // Grow BMPS from top (UP direction) to row 1
  // After this, UP stack has absorbed row 0
  contractor.GrowBMPSForRow(tn2d, 1, trunc_para);

  // Get a walker forked from UP direction
  WalkerT walker = contractor.GetWalker(tn2d, UP);

  // Verify walker's initial state matches contractor's UP stack
  const auto& contractor_up_stack = contractor.GetBMPS(UP);
  EXPECT_GT(contractor_up_stack.size(), 0);
  EXPECT_EQ(walker.GetStackSize(), contractor_up_stack.size());
  EXPECT_EQ(walker.GetPosition(), UP);

  // Save original stack size
  size_t original_stack_size = walker.GetStackSize();

  // Evolve the walker by one step (should absorb the next row)
  walker.EvolveStep(trunc_para);
  EXPECT_EQ(walker.GetStackSize(), original_stack_size + 1);

  // Verify that contractor's stack is unchanged (walker is independent)
  EXPECT_EQ(contractor.GetBMPS(UP).size(), original_stack_size);

  // Evolve walker one more step
  walker.EvolveStep(trunc_para);
  EXPECT_EQ(walker.GetStackSize(), original_stack_size + 2);

  // Contractor still unchanged
  EXPECT_EQ(contractor.GetBMPS(UP).size(), original_stack_size);

  // Test that we can create multiple independent walkers
  WalkerT walker2 = contractor.GetWalker(tn2d, UP);
  EXPECT_EQ(walker2.GetStackSize(), original_stack_size);  // Fresh copy from contractor

  // Evolve walker2 - should not affect walker1
  walker2.EvolveStep(trunc_para);
  EXPECT_EQ(walker2.GetStackSize(), original_stack_size + 1);
  EXPECT_EQ(walker.GetStackSize(), original_stack_size + 2);  // walker1 unchanged
}

/**
 * @brief Test BMPSWalker::ContractRow functionality.
 *
 * Verifies that ContractRow produces correct overlap values by comparing
 * with the standard Trace method from BMPSContractor.
 */
TEST_F(OBCIsing2DTenNetWithoutZ2, BMPSWalkerContractRowTest) {
  using TenElemT = QLTEN_Double;
  using ContractorT = BMPSContractor<TenElemT, QNT>;
  using WalkerT = typename ContractorT::BMPSWalker;
  using TransferMPO = std::vector<DQLTensor *>;
  
  auto trunc_para = BMPSTruncateParams<QLTEN_Double>::SVD(10, 30, 1e-15);

  ContractorT contractor(dtn2d.rows(), dtn2d.cols());
  contractor.Init(dtn2d);

  // Build environments for a middle row (e.g., row 2)
  // UP environment: absorb rows 0, 1
  contractor.GrowBMPSForRow(dtn2d, 2, trunc_para);
  
  // Get UP walker (has absorbed rows 0, 1)
  // GrowBMPSForRow(row=2) means UP stack contains: vacuum, row0, row1
  // So stack size = 3 (or row + 1 for general case)
  WalkerT walker = contractor.GetWalker(dtn2d, UP);
  EXPECT_EQ(walker.GetStackSize(), 3); // vacuum + row0 + row1 absorbed
  
  // DOWN environment should have absorbed rows Ly-1, Ly-2, ..., 3
  const auto& down_stack = contractor.GetBMPS(DOWN);
  
  // The DOWN stack should cover rows from bottom up to just below row 2
  // GrowBMPSForRow(row=2) means: UP covers [0, row), DOWN covers (row, Ly)
  // So DOWN should have Ly - 1 - 2 = Ly - 3 BMPS entries (plus vacuum = Ly-2)
  
  // DOWN environment: GrowBMPSForRow(row=2) creates DOWN stack with rows [3, Ly-1] absorbed
  // For 12x12 lattice: DOWN stack size should be Ly - 2 = 10 (vacuum + 9 rows)
  EXPECT_GE(down_stack.size(), dtn2d.rows() - 2);
  
  // Build MPO for row 2
  TransferMPO row2_mpo;
  row2_mpo.reserve(dtn2d.cols());
  for (size_t col = 0; col < dtn2d.cols(); ++col) {
    row2_mpo.push_back(const_cast<DQLTensor*>(&dtn2d({2, col})));
  }
  
  // Get the correct DOWN boundary covering rows [3, Ly-1]
  // down_stack[k] has absorbed k rows from bottom (row Ly-1 down to row Ly-k)
  // We need down_stack that covers rows 3 to Ly-1, that's Ly - 3 rows
  size_t needed_down_idx = dtn2d.rows() - 1 - 2; // = Ly - 3 = 9 for 12x12
  ASSERT_LT(needed_down_idx, down_stack.size()) << "DOWN stack should cover row 3 to Ly-1";
  
  const auto& bottom_env = down_stack[needed_down_idx];
  
  // ContractRow: <walker_bmps | row2_mpo | bottom_env>
  TenElemT walker_val = walker.ContractRow(row2_mpo, bottom_env);
  
  // Compare with contractor's Trace method
  contractor.InitBTen(dtn2d, BTenPOSITION::LEFT, 2);
  contractor.GrowFullBTen(dtn2d, BTenPOSITION::RIGHT, 2, 2, true);
  TenElemT contractor_val = contractor.Trace(dtn2d, {2, 0}, HORIZONTAL);
  
  // Both should give the partition function (same value)
  EXPECT_NE(contractor_val, 0.0);
  EXPECT_NEAR(walker_val / contractor_val, 1.0, 1e-8);
}

/**
 * @brief Test BMPSWalker BTen cache interface functionality.
 * 
 * Note: BTen caching is a placeholder for future optimization.
 * This test verifies the interface works correctly (column tracking).
 * Actual BTen caching with tensor contraction is deferred to future work.
 * 
 * Tests:
 * 1. InitBTenLeft / InitBTenRight correctly track column positions
 * 2. GrowBTenLeftStep / GrowBTenRightStep correctly advance counters
 * 3. ClearBTen resets state
 */
TEST_F(OBCIsing2DTenNetWithoutZ2, BMPSWalkerBTenCacheTest) {
  using TenElemT = double;
  using DQLTensor = qlten::QLTensor<TenElemT, U1QN>;
  using TransferMPO = std::vector<DQLTensor *>;
  using ContractorT = BMPSContractor<TenElemT, U1QN>;
  using WalkerT = ContractorT::BMPSWalker;
  
  // Create contractor and grow boundaries
  ContractorT contractor(dtn2d.rows(), dtn2d.cols());
  contractor.Init(dtn2d);
  
  BMPSTruncateParams<double> trunc_para(1, 20, 1e-14, CompressMPSScheme::SVD_COMPRESS);
  contractor.GrowBMPSForRow(dtn2d, 2, trunc_para);
  
  // Get walker
  WalkerT walker = contractor.GetWalker(dtn2d, UP);
  
  // Get DOWN boundary for row 2
  const auto& down_stack = contractor.GetBMPS(DOWN);
  size_t needed_down_idx = dtn2d.rows() - 1 - 2;
  ASSERT_LT(needed_down_idx, down_stack.size());
  const auto& bottom_env = down_stack[needed_down_idx];
  
  // Build MPO for row 2
  TransferMPO row2_mpo;
  row2_mpo.reserve(dtn2d.cols());
  for (size_t col = 0; col < dtn2d.cols(); ++col) {
    row2_mpo.push_back(const_cast<DQLTensor*>(&dtn2d({2, col})));
  }
  
  // Test 1: ContractRow as reference (BTen optimization falls back to this)
  TenElemT reference_val = walker.ContractRow(row2_mpo, bottom_env);
  EXPECT_NE(reference_val, 0.0) << "ContractRow should return non-zero for valid contraction";
  
  // Test 2: Initialize BTen column tracking
  const size_t Lx = dtn2d.cols();
  const size_t mid_col = Lx / 2;
  
  walker.InitBTenLeft(row2_mpo, bottom_env, mid_col);
  EXPECT_EQ(walker.GetBTenLeftCol(), mid_col) 
      << "InitBTenLeft should set left edge to target_col";
  
  walker.InitBTenRight(row2_mpo, bottom_env, mid_col);
  EXPECT_EQ(walker.GetBTenRightCol(), mid_col + 1) 
      << "InitBTenRight should set right edge to target_col + 1";
  
  // Test 3: GrowBTenLeftStep functionality
  // To test manual step-by-step growth, we reset and grow manually
  walker.ClearBTen();
  EXPECT_EQ(walker.GetBTenLeftCol(), 0);
  EXPECT_EQ(walker.GetBTenRightCol(), 0);
  
  // Grow LEFT BTen step by step up to mid_col
  // Note: walker needs initial vacuum state. 
  // Since we cleared it, we need to re-init. 
  // But InitBTenLeft does full growth loop. 
  // Let's use InitBTenLeft with target=0 to just setup vacuum.
  walker.InitBTenLeft(row2_mpo, bottom_env, 0); 
  
  for (size_t col = 0; col < mid_col; ++col) {
    walker.GrowBTenLeftStep(row2_mpo, bottom_env);
    EXPECT_EQ(walker.GetBTenLeftCol(), col + 1) 
        << "GrowBTenLeftStep should advance left edge by 1";
  }

  // Now we need to prepare RIGHT BTen for Test 4
  walker.InitBTenRight(row2_mpo, bottom_env, mid_col);

  // Test 4: TraceWithBTen should match reference_val
  // Now both Left (manually grown) and Right (Init func) are ready at mid_col.
  const DQLTensor& mid_site = dtn2d({2, mid_col});
  TenElemT bten_val = walker.TraceWithBTen(mid_site, mid_col, bottom_env);
  EXPECT_NEAR(bten_val, reference_val, 1e-8) << "TraceWithBTen should match ContractRow result";
  
  // Test 5: Verify ContractRow still works as fallback for structure factor
  // This is the intended usage pattern until BTen caching is fully implemented
  // Note: We use a fresh walker to ensure no interference from BTen cache state
  // (though ContractRow should be independent, this isolates the test).
  WalkerT walker_fallback = contractor.GetWalker(dtn2d, UP);
  TenElemT fallback_val = walker_fallback.ContractRow(row2_mpo, bottom_env);
  EXPECT_NEAR(fallback_val, reference_val, 1e-8) 
      << "ContractRow should remain functional as fallback";
}

/**
 * @brief Test BMPSWalker::ShiftBTenWindow functionality.
 *
 * Verifies:
 * 1. ShiftBTenWindow(RIGHT) advances left BTen and pops right BTen
 * 2. ShiftBTenWindow(LEFT) advances right BTen and pops left BTen
 * 3. After shifting, TraceWithBTen still gives correct results
 */
TEST_F(OBCIsing2DTenNetWithoutZ2, BMPSWalkerShiftBTenWindowTest) {
  using TenElemT = double;
  using DQLTensor = qlten::QLTensor<TenElemT, U1QN>;
  using TransferMPO = std::vector<DQLTensor *>;
  using ContractorT = BMPSContractor<TenElemT, U1QN>;
  using WalkerT = ContractorT::BMPSWalker;
  
  // Create contractor and grow boundaries
  ContractorT contractor(dtn2d.rows(), dtn2d.cols());
  contractor.Init(dtn2d);
  
  BMPSTruncateParams<double> trunc_para(1, 20, 1e-14, CompressMPSScheme::SVD_COMPRESS);
  contractor.GrowBMPSForRow(dtn2d, 2, trunc_para);
  
  // Get walker
  WalkerT walker = contractor.GetWalker(dtn2d, UP);
  
  // Get DOWN boundary for row 2
  const auto& down_stack = contractor.GetBMPS(DOWN);
  size_t needed_down_idx = dtn2d.rows() - 1 - 2;
  ASSERT_LT(needed_down_idx, down_stack.size());
  const auto& bottom_env = down_stack[needed_down_idx];
  
  // Build MPO for row 2
  TransferMPO row2_mpo;
  row2_mpo.reserve(dtn2d.cols());
  for (size_t col = 0; col < dtn2d.cols(); ++col) {
    row2_mpo.push_back(const_cast<DQLTensor*>(&dtn2d({2, col})));
  }
  
  // Reference value from ContractRow
  TenElemT reference_val = walker.ContractRow(row2_mpo, bottom_env);
  
  const size_t Lx = dtn2d.cols();
  
  // Test 1: Initialize BTen window at column 1 (left edge=1, right edge=2)
  walker.InitBTenLeft(row2_mpo, bottom_env, 1);
  walker.InitBTenRight(row2_mpo, bottom_env, 1);
  
  EXPECT_EQ(walker.GetBTenLeftCol(), 1);
  EXPECT_EQ(walker.GetBTenRightCol(), 2);
  
  // Test 2: ShiftBTenWindow(RIGHT) - shift to the right
  // This should: grow left (col 1->2), pop right (col 2->3)
  walker.ShiftBTenWindow(row2_mpo, bottom_env, RIGHT);
  EXPECT_EQ(walker.GetBTenLeftCol(), 2);
  EXPECT_EQ(walker.GetBTenRightCol(), 3);
  
  // Test 3: Verify TraceWithBTen still works after shift
  const DQLTensor& mid_site = dtn2d({2, 2});
  TenElemT trace_val = walker.TraceWithBTen(mid_site, 2, bottom_env);
  EXPECT_NEAR(trace_val, reference_val, 1e-8);
  
  // Test 4: ShiftBTenWindow(LEFT) - shift back to the left
  // This should: grow right (col 3->2), pop left (col 2->1)
  walker.ShiftBTenWindow(row2_mpo, bottom_env, LEFT);
  EXPECT_EQ(walker.GetBTenLeftCol(), 1);
  EXPECT_EQ(walker.GetBTenRightCol(), 2);
  
  // Test 5: Verify TraceWithBTen at column 1
  const DQLTensor& site1 = dtn2d({2, 1});
  TenElemT trace_val2 = walker.TraceWithBTen(site1, 1, bottom_env);
  EXPECT_NEAR(trace_val2, reference_val, 1e-8);
}

/**
 * @brief Test BMPSWalker::TraceWithTwoSiteBTen functionality.
 *
 * Verifies:
 * 1. TraceWithTwoSiteBTen correctly computes trace for adjacent site pairs
 * 2. Results match ContractRow reference when using identity-like operators
 * 3. Window can be shifted and recomputed correctly
 */
TEST_F(OBCIsing2DTenNetWithoutZ2, BMPSWalkerTraceWithTwoSiteBTenTest) {
  using TenElemT = double;
  using DQLTensor = qlten::QLTensor<TenElemT, U1QN>;
  using TransferMPO = std::vector<DQLTensor *>;
  using ContractorT = BMPSContractor<TenElemT, U1QN>;
  using WalkerT = ContractorT::BMPSWalker;
  
  // Create contractor and grow boundaries
  ContractorT contractor(dtn2d.rows(), dtn2d.cols());
  contractor.Init(dtn2d);
  
  BMPSTruncateParams<double> trunc_para(1, 20, 1e-14, CompressMPSScheme::SVD_COMPRESS);
  contractor.GrowBMPSForRow(dtn2d, 2, trunc_para);
  
  // Get walker
  WalkerT walker = contractor.GetWalker(dtn2d, UP);
  
  // Get DOWN boundary for row 2
  const auto& down_stack = contractor.GetBMPS(DOWN);
  size_t needed_down_idx = dtn2d.rows() - 1 - 2;
  ASSERT_LT(needed_down_idx, down_stack.size());
  const auto& bottom_env = down_stack[needed_down_idx];
  
  // Build MPO for row 2
  TransferMPO row2_mpo;
  row2_mpo.reserve(dtn2d.cols());
  for (size_t col = 0; col < dtn2d.cols(); ++col) {
    row2_mpo.push_back(const_cast<DQLTensor*>(&dtn2d({2, col})));
  }
  
  // Reference value from ContractRow
  TenElemT reference_val = walker.ContractRow(row2_mpo, bottom_env);
  EXPECT_NE(reference_val, 0.0);
  
  const size_t Lx = dtn2d.cols();
  
  // Test 1: Initialize BTen window for two-site trace at columns 1 and 2
  // Left BTen should cover [0, 1), Right BTen should cover (2, Lx-1]
  walker.InitBTenLeft(row2_mpo, bottom_env, 1);
  walker.InitBTenRight(row2_mpo, bottom_env, 2);
  
  EXPECT_EQ(walker.GetBTenLeftCol(), 1);
  EXPECT_EQ(walker.GetBTenRightCol(), 3);  // Right edge is site_col+1 after the last covered site
  
  // Test 2: TraceWithTwoSiteBTen at columns 1-2 using original tensors
  // This should give the same result as the full ContractRow
  const DQLTensor& site1 = dtn2d({2, 1});
  const DQLTensor& site2 = dtn2d({2, 2});
  TenElemT two_site_val = walker.TraceWithTwoSiteBTen(site1, site2, 1, row2_mpo, bottom_env);
  EXPECT_NEAR(two_site_val, reference_val, 1e-8) 
      << "TraceWithTwoSiteBTen with original tensors should match ContractRow";
  
  // Test 3: Shift window to the right and compute at columns 2-3
  walker.ShiftBTenWindow(row2_mpo, bottom_env, RIGHT);
  EXPECT_EQ(walker.GetBTenLeftCol(), 2);
  EXPECT_EQ(walker.GetBTenRightCol(), 4);
  
  const DQLTensor& site2b = dtn2d({2, 2});
  const DQLTensor& site3 = dtn2d({2, 3});
  TenElemT two_site_val2 = walker.TraceWithTwoSiteBTen(site2b, site3, 2, row2_mpo, bottom_env);
  EXPECT_NEAR(two_site_val2, reference_val, 1e-8)
      << "TraceWithTwoSiteBTen after shift should still match ContractRow";
}

int main(int argc, char *argv[]) {

  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}

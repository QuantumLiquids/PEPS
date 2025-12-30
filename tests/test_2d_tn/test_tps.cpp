// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-12-30
*
* Description: QuantumLiquids/PEPS project. Unittests for TPS class.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/tps.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using DTPS = TPS<QLTEN_Double, U1QN>;
using ZTPS = TPS<QLTEN_Complex, U1QN>;
using DTensor = QLTensor<QLTEN_Double, U1QN>;
using ZTensor = QLTensor<QLTEN_Complex, U1QN>;

// =============================================================================
// Test Fixture
// =============================================================================

struct TPSTest : public testing::Test {
  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});

  IndexT pb_out = IndexT({
                           QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                           QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)
                         },
                         TenIndexDirType::OUT
  );

  size_t Lx = 3;
  size_t Ly = 3;
  size_t D = 4;

  /**
   * @brief Create a random TPS for testing
   */
  DTPS CreateRandTPS(BoundaryCondition bc, size_t bond_dim) {
    DTPS tps(Ly, Lx, bc);
    size_t d_sec = bond_dim / 2;
    IndexT virt_out_idx = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), d_sec),
                                  QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), bond_dim - d_sec)},
                                 TenIndexDirType::OUT);
    IndexT virt_in_idx = InverseIndex(virt_out_idx);

    IndexT trivial_idx_out = IndexT({QNSctT(qn0, 1)}, TenIndexDirType::OUT);
    IndexT trivial_idx_in = InverseIndex(trivial_idx_out);

    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        IndexT left_idx, lower_idx, right_idx, upper_idx;
        
        if (bc == BoundaryCondition::Periodic) {
          left_idx = virt_in_idx;
          lower_idx = virt_out_idx;
          right_idx = virt_out_idx;
          upper_idx = virt_in_idx;
        } else {
          // OBC: boundary indices are trivial (dim=1)
          left_idx = (col == 0) ? trivial_idx_in : virt_in_idx;
          lower_idx = (row == Ly - 1) ? trivial_idx_out : virt_out_idx;
          right_idx = (col == Lx - 1) ? trivial_idx_out : virt_out_idx;
          upper_idx = (row == 0) ? trivial_idx_in : virt_in_idx;
        }
        
        tps({row, col}) = DTensor({left_idx, lower_idx, right_idx, upper_idx, pb_out});
        tps({row, col}).Random(qn0);
      }
    }
    return tps;
  }

  ZTPS CreateRandComplexTPS(BoundaryCondition bc) {
    ZTPS tps(Ly, Lx, bc);
    IndexT virt_out_idx = IndexT({QNSctT(qn0, D)}, TenIndexDirType::OUT);
    IndexT virt_in_idx = InverseIndex(virt_out_idx);
    IndexT trivial_idx_out = IndexT({QNSctT(qn0, 1)}, TenIndexDirType::OUT);
    IndexT trivial_idx_in = InverseIndex(trivial_idx_out);

    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        IndexT left_idx, lower_idx, right_idx, upper_idx;
        
        if (bc == BoundaryCondition::Periodic) {
          left_idx = virt_in_idx;
          lower_idx = virt_out_idx;
          right_idx = virt_out_idx;
          upper_idx = virt_in_idx;
        } else {
          left_idx = (col == 0) ? trivial_idx_in : virt_in_idx;
          lower_idx = (row == Ly - 1) ? trivial_idx_out : virt_out_idx;
          right_idx = (col == Lx - 1) ? trivial_idx_out : virt_out_idx;
          upper_idx = (row == 0) ? trivial_idx_in : virt_in_idx;
        }
        
        tps({row, col}) = ZTensor({left_idx, lower_idx, right_idx, upper_idx, pb_out});
        tps({row, col}).Random(qn0);
      }
    }
    return tps;
  }
};

// =============================================================================
// WaveFunctionSum Tests (OBC)
// =============================================================================

TEST_F(TPSTest, WaveFunctionSumOBC_Basic) {
  DTPS tps1 = CreateRandTPS(BoundaryCondition::Open, D);
  DTPS tps2 = CreateRandTPS(BoundaryCondition::Open, D);

  DTPS result = WaveFunctionSum(tps1, tps2);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
  EXPECT_EQ(result.GetBoundaryCondition(), BoundaryCondition::Open);

  // Bulk bonds should be 2*D
  auto bulk_tensor = result({1, 1});
  EXPECT_EQ(bulk_tensor.GetShape()[0], 2 * D);
  EXPECT_EQ(bulk_tensor.GetShape()[1], 2 * D);
  EXPECT_EQ(bulk_tensor.GetShape()[2], 2 * D);
  EXPECT_EQ(bulk_tensor.GetShape()[3], 2 * D);
  EXPECT_EQ(bulk_tensor.GetShape()[4], 2);

  // Boundary indices should remain dim=1
  auto corner_tensor = result({0, 0});
  EXPECT_EQ(corner_tensor.GetShape()[0], 1);
  EXPECT_EQ(corner_tensor.GetShape()[3], 1);
  EXPECT_EQ(corner_tensor.GetShape()[1], 2 * D);
  EXPECT_EQ(corner_tensor.GetShape()[2], 2 * D);
}

TEST_F(TPSTest, WaveFunctionSumOBC_SelfSum) {
  DTPS tps = CreateRandTPS(BoundaryCondition::Open, D);
  DTPS result = WaveFunctionSum(tps, tps);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
  EXPECT_EQ(result({1, 1}).GetShape()[0], 2 * D);
}

// =============================================================================
// WaveFunctionSum Tests (PBC)
// =============================================================================

TEST_F(TPSTest, WaveFunctionSumPBC_Basic) {
  DTPS tps1 = CreateRandTPS(BoundaryCondition::Periodic, D);
  DTPS tps2 = CreateRandTPS(BoundaryCondition::Periodic, D);

  DTPS result = WaveFunctionSum(tps1, tps2);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
  EXPECT_EQ(result.GetBoundaryCondition(), BoundaryCondition::Periodic);

  // For PBC, ALL bonds should be expanded
  auto corner_tensor = result({0, 0});
  EXPECT_EQ(corner_tensor.GetShape()[0], 2 * D);
  EXPECT_EQ(corner_tensor.GetShape()[1], 2 * D);
  EXPECT_EQ(corner_tensor.GetShape()[2], 2 * D);
  EXPECT_EQ(corner_tensor.GetShape()[3], 2 * D);
}

// =============================================================================
// WaveFunctionSum with Coefficients
// =============================================================================

TEST_F(TPSTest, WaveFunctionSumWithCoefficients) {
  DTPS tps1 = CreateRandTPS(BoundaryCondition::Open, D);
  DTPS tps2 = CreateRandTPS(BoundaryCondition::Open, D);

  DTPS result = WaveFunctionSum(2.0, tps1, 3.0, tps2);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
  EXPECT_EQ(result({1, 1}).GetShape()[0], 2 * D);
}

TEST_F(TPSTest, WaveFunctionSumWithZeroCoeff) {
  DTPS tps1 = CreateRandTPS(BoundaryCondition::Open, D);
  DTPS tps2 = CreateRandTPS(BoundaryCondition::Open, D);

  // Should not crash with zero coefficient
  DTPS result = WaveFunctionSum(1.0, tps1, 0.0, tps2);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
}

// =============================================================================
// WaveFunctionSum of N TPS
// =============================================================================

TEST_F(TPSTest, WaveFunctionSumNTPS) {
  DTPS tps1 = CreateRandTPS(BoundaryCondition::Open, D);
  DTPS tps2 = CreateRandTPS(BoundaryCondition::Open, D);
  DTPS tps3 = CreateRandTPS(BoundaryCondition::Open, D);

  std::vector<DTPS> tps_list = {tps1, tps2, tps3};
  DTPS result = WaveFunctionSum(tps_list);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
  EXPECT_EQ(result({1, 1}).GetShape()[0], 3 * D);
}

TEST_F(TPSTest, WaveFunctionSumNTPSWithCoefficients) {
  DTPS tps1 = CreateRandTPS(BoundaryCondition::Open, D);
  DTPS tps2 = CreateRandTPS(BoundaryCondition::Open, D);
  DTPS tps3 = CreateRandTPS(BoundaryCondition::Open, D);

  std::vector<DTPS> tps_list = {tps1, tps2, tps3};
  std::vector<QLTEN_Double> coeffs = {1.0, 2.0, 3.0};
  DTPS result = WaveFunctionSum(tps_list, coeffs);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
  EXPECT_EQ(result({1, 1}).GetShape()[0], 3 * D);
}

TEST_F(TPSTest, WaveFunctionSumSingleTPS) {
  DTPS tps = CreateRandTPS(BoundaryCondition::Open, D);

  std::vector<DTPS> tps_list = {tps};
  DTPS result = WaveFunctionSum(tps_list);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
  EXPECT_EQ(result({1, 1}).GetShape()[0], D);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(TPSTest, WaveFunctionSum_1x1Lattice_OBC) {
  DTPS tps1(1, 1, BoundaryCondition::Open);
  DTPS tps2(1, 1, BoundaryCondition::Open);

  IndexT trivial_idx_out = IndexT({QNSctT(qn0, 1)}, TenIndexDirType::OUT);
  IndexT trivial_idx_in = InverseIndex(trivial_idx_out);

  tps1({0, 0}) = DTensor({trivial_idx_in, trivial_idx_out, trivial_idx_out, trivial_idx_in, pb_out});
  tps1({0, 0}).Random(qn0);

  tps2({0, 0}) = DTensor({trivial_idx_in, trivial_idx_out, trivial_idx_out, trivial_idx_in, pb_out});
  tps2({0, 0}).Random(qn0);

  // For 1x1 OBC, no indices should be expanded
  DTPS result = WaveFunctionSum(tps1, tps2);

  EXPECT_EQ(result.rows(), 1);
  EXPECT_EQ(result.cols(), 1);
  EXPECT_EQ(result({0, 0}).GetShape()[0], 1);
  EXPECT_EQ(result({0, 0}).GetShape()[1], 1);
  EXPECT_EQ(result({0, 0}).GetShape()[2], 1);
  EXPECT_EQ(result({0, 0}).GetShape()[3], 1);
}

TEST_F(TPSTest, WaveFunctionSum_1x1Lattice_PBC) {
  DTPS tps1(1, 1, BoundaryCondition::Periodic);
  DTPS tps2(1, 1, BoundaryCondition::Periodic);

  size_t d = 2;
  IndexT virt_out_idx = IndexT({QNSctT(qn0, d)}, TenIndexDirType::OUT);
  IndexT virt_in_idx = InverseIndex(virt_out_idx);

  tps1({0, 0}) = DTensor({virt_in_idx, virt_out_idx, virt_out_idx, virt_in_idx, pb_out});
  tps1({0, 0}).Random(qn0);

  tps2({0, 0}) = DTensor({virt_in_idx, virt_out_idx, virt_out_idx, virt_in_idx, pb_out});
  tps2({0, 0}).Random(qn0);

  DTPS result = WaveFunctionSum(tps1, tps2);

  EXPECT_EQ(result.rows(), 1);
  EXPECT_EQ(result.cols(), 1);
  EXPECT_EQ(result({0, 0}).GetShape()[0], 2 * d);
  EXPECT_EQ(result({0, 0}).GetShape()[1], 2 * d);
  EXPECT_EQ(result({0, 0}).GetShape()[2], 2 * d);
  EXPECT_EQ(result({0, 0}).GetShape()[3], 2 * d);
}

// =============================================================================
// Complex Tensor Tests
// =============================================================================

TEST_F(TPSTest, WaveFunctionSumComplex_OBC) {
  ZTPS tps1 = CreateRandComplexTPS(BoundaryCondition::Open);
  ZTPS tps2 = CreateRandComplexTPS(BoundaryCondition::Open);

  ZTPS result = WaveFunctionSum(tps1, tps2);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
  EXPECT_EQ(result.GetBoundaryCondition(), BoundaryCondition::Open);
}

TEST_F(TPSTest, WaveFunctionSumComplex_PBC) {
  ZTPS tps1 = CreateRandComplexTPS(BoundaryCondition::Periodic);
  ZTPS tps2 = CreateRandComplexTPS(BoundaryCondition::Periodic);

  ZTPS result = WaveFunctionSum(tps1, tps2);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
  EXPECT_EQ(result.GetBoundaryCondition(), BoundaryCondition::Periodic);
  EXPECT_EQ(result({0, 0}).GetShape()[0], 2 * D);
}

TEST_F(TPSTest, WaveFunctionSumComplex_WithCoefficients) {
  ZTPS tps1 = CreateRandComplexTPS(BoundaryCondition::Open);
  ZTPS tps2 = CreateRandComplexTPS(BoundaryCondition::Open);

  QLTEN_Complex alpha(1.0, 2.0);
  QLTEN_Complex beta(3.0, -1.0);

  ZTPS result = WaveFunctionSum(alpha, tps1, beta, tps2);

  EXPECT_EQ(result.rows(), Ly);
  EXPECT_EQ(result.cols(), Lx);
}


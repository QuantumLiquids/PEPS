// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-22
*
* Description: QuantumLiquids/PEPS project. Unittests for SplitIndexTPS
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps_impl.h"
#include "qlpeps/consts.h"
#include <filesystem>

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using DTPS = TPS<QLTEN_Double, U1QN>;
using DSITPS = SplitIndexTPS<QLTEN_Double, U1QN>;
using DTensor = QLTensor<QLTEN_Double, U1QN>;

struct SplitIdxTPSData : public testing::Test {
  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});

  IndexT pb_out = IndexT({
                           QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                           QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)
                         },
                         TenIndexDirType::OUT
  );

  IndexT pb_in = InverseIndex(pb_out);
  size_t Lx = 4;
  size_t Ly = 4;
  size_t D = 6;
  size_t N = Lx * Ly;

  std::string tps_path = "test_s_tps";
  DTPS dtps = DTPS(Ly, Lx);
  DSITPS dsitps = DSITPS(Ly, Lx);

  void SetUp() override {
    dtps = CreateRandTestTPS();
    dsitps = SplitIndexTPS(dtps);
  }

  void TearDown() override {
    // Clean up the created directory and its contents
    if (std::filesystem::exists(tps_path)) {
      std::filesystem::remove_all(tps_path);
    }
  }

  DTPS CreateRandTestTPS() {
    DTPS tps(Ly, Lx);
    size_t d_sec = D / 3;
    IndexT virt_out_idx = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), d_sec),
                                  QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), d_sec),
                                  QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), D - 2 * d_sec)},
                                 TenIndexDirType::OUT);
    IndexT virt_in_idx = InverseIndex(virt_out_idx);

    IndexT left_idx = virt_in_idx;
    IndexT lower_idx = virt_out_idx;
    IndexT right_idx = virt_out_idx;
    IndexT upper_idx = virt_in_idx;

    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        tps({row, col}) = DTensor({left_idx, lower_idx, right_idx, upper_idx, pb_out});
        tps({row, col}).Random(qn0);
      }
    }
    return tps;
  }
};

TEST_F(SplitIdxTPSData, DumpAndLoad) {
  dsitps.Dump(tps_path);
  DSITPS loaded_dsitps;
  loaded_dsitps.Load(tps_path);

  // Verify that the loaded TPS is the same as the original
  EXPECT_EQ(loaded_dsitps.rows(), dsitps.rows());
  EXPECT_EQ(loaded_dsitps.cols(), dsitps.cols());
  for (size_t row = 0; row < dsitps.rows(); ++row) {
    for (size_t col = 0; col < dsitps.cols(); ++col) {
      for (size_t i = 0; i < dsitps.PhysicalDim(); ++i) {
        auto diff_ten = dsitps({row, col})[i] + (-loaded_dsitps({row, col})[i]);
        EXPECT_NEAR(diff_ten.Get2Norm(), 0.0, 1e-12);
      }
    }
  }
}

TEST_F(SplitIdxTPSData, TestTransfer2TPS) {
  DTPS dtps2 = dsitps.GroupIndices(pb_out);
  EXPECT_EQ(dtps2, dtps);
}

TEST_F(SplitIdxTPSData, TestBasicOperation) {
  DSITPS zeros_sitps = 0.0 * dsitps;
  EXPECT_EQ(zeros_sitps.rows(), Ly);
  EXPECT_EQ(zeros_sitps.cols(), Lx);
  EXPECT_EQ(zeros_sitps.PhysicalDim(), dsitps.PhysicalDim());
  for (std::vector<DTensor> &zero_tens : zeros_sitps) {
    for (DTensor &zero_ten : zero_tens) {
      EXPECT_DOUBLE_EQ(zero_ten.Get2Norm(), 0.0);
    }
  }
  auto dsitps2 = dsitps;
  dsitps2 += dsitps;
  dsitps2 -= dsitps;
  for (size_t row = 0; row < Ly; row++) {
    for (size_t col = 0; col < Lx; col++) {
      for (size_t i = 0; i < 2; i++) {
        auto init_ten = dsitps({row, col})[i];
        auto res_ten = dsitps2({row, col})[i];
        auto diff_ten = init_ten + (-res_ten);
        EXPECT_NEAR(diff_ten.Get2Norm(), 0.0, 1e-13);
      }
    }
  }
}

TEST_F(SplitIdxTPSData, TestNormalization) {
  dsitps.NormalizeAllSite();
  for (const auto &split_ten_vec : dsitps) {
    double total_norm_sq = 0;
    for (const auto &ten : split_ten_vec) {
      total_norm_sq += std::pow(ten.GetQuasi2Norm(), 2);
    }
    EXPECT_NEAR(total_norm_sq, 1.0, 1e-12);
  }

  // Verify that dsitps2 created from a normalized dtps is also normalized
  DSITPS dsitps2(dtps);
  dsitps2.NormalizeAllSite();
  for (const auto &split_ten_vec : dsitps2) {
    double total_norm_sq = 0;
    for (const auto &ten : split_ten_vec) {
      total_norm_sq += std::pow(ten.GetQuasi2Norm(), 2);
    }
    EXPECT_NEAR(total_norm_sq, 1.0, 1e-12);
  }
  EXPECT_NEAR(dsitps2.NormSquare(), (double) Lx * Ly, 1e-12);
}

TEST_F(SplitIdxTPSData, ArithmeticOperators) {
  DSITPS a = dsitps;
  DSITPS b = dsitps;
  DSITPS c = a + b;
  DSITPS d = c - a;

  EXPECT_NEAR((d - b).NormSquare(), 0.0, 1e-12);

  DSITPS e = a * 2.0;
  DSITPS f = e * 0.5;
  EXPECT_NEAR((f - a).NormSquare(), 0.0, 1e-12);

  a += b;
  a -= b;
  EXPECT_NEAR((a - b).NormSquare(), 0.0, 1e-12);

  a *= 2.0;
  a *= 0.5;
  EXPECT_NEAR((a - b).NormSquare(), 0.0, 1e-12);
}

TEST_F(SplitIdxTPSData, DotProduct) {
  DSITPS a = dsitps;
  a.NormalizeAllSite();
  double dot_product = a * a;
  EXPECT_NEAR(dot_product, a.NormSquare(), 1e-12);
}

TEST_F(SplitIdxTPSData, MoveSemantics) {
  DSITPS a = dsitps;
  DSITPS b = a;

  // Test move constructor
  DSITPS c = std::move(a);
  EXPECT_EQ(c.rows(), b.rows());
  EXPECT_EQ(c.cols(), b.cols());
  EXPECT_NEAR((c - b).NormSquare(), 0.0, 1e-12);

  // a should be in a valid but unspecified (likely empty) state
  // Depending on implementation, it might be empty or hold default-constructed tensors
  EXPECT_TRUE(a.empty() || a.rows() == 0);

  // Test move assignment
  DSITPS d(Ly, Lx);
  d = std::move(c);
  EXPECT_EQ(d.rows(), b.rows());
  EXPECT_EQ(d.cols(), b.cols());
  EXPECT_NEAR((d - b).NormSquare(), 0.0, 1e-12);
  EXPECT_TRUE(c.empty() || c.rows() == 0);
}

// =============================
// Clipping API tests (merged)
// =============================

static DSITPS CreateSimpleSITPS_Clip(size_t Ly, size_t Lx) {
  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1), QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);
  IndexT v = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)}, TenIndexDirType::OUT);
  IndexT vin = InverseIndex(v);
  DSITPS t(Ly, Lx, 2);
  for (size_t r=0;r<Ly;++r) {
    for (size_t c=0;c<Lx;++c) {
      for (size_t i=0;i<2;++i) {
        t({r,c})[i] = DTensor({vin, v, v, vin});
        t({r,c})[i].Fill(U1QN({QNCard("Sz", U1QNVal(0))}), (i==0) ? 10.0 : -0.5);
      }
    }
  }
  return t;
}

TEST_F(SplitIdxTPSData, ElementWiseClipToInPlace) {
  DSITPS t = CreateSimpleSITPS_Clip(1,1);
  t.ElementWiseClipTo(1.0);
  // All elements must be within [-1,1]
  for (auto &vec : t) {
    for (auto &ten : vec) {
      EXPECT_LE(ten.GetMaxAbs(), 1.0 + 1e-12);
    }
  }
}

TEST_F(SplitIdxTPSData, ElementWiseClipToOutOfPlace) {
  DSITPS t = CreateSimpleSITPS_Clip(1,1);
  DSITPS t_orig = t;
  DSITPS clipped = ElementWiseClipTo(t, 0.2);
  // Original unchanged
  for (size_t r = 0; r < t.rows(); ++r) {
    for (size_t c = 0; c < t.cols(); ++c) {
      for (size_t i = 0; i < t.PhysicalDim(); ++i) {
        auto diff = t({r, c})[i] + (-t_orig({r, c})[i]);
        EXPECT_LE(diff.GetMaxAbs(), 1e-12);
      }
    }
  }
  for (auto &vec : clipped) {
    for (auto &ten : vec) {
      EXPECT_LE(ten.GetMaxAbs(), 0.2 + 1e-12);
    }
  }
}

TEST_F(SplitIdxTPSData, ClipByGlobalNormInPlace) {
  DSITPS t = CreateSimpleSITPS_Clip(1,1);
  // Compute current quasi-norm
  double nsq = 0.0;
  for (auto &vec : t) {
    for (auto &ten : vec) {
      double q = ten.GetQuasi2Norm();
      nsq += q*q;
    }
  }
  double r = std::sqrt(nsq);
  double target = r * 0.1;
  t.ClipByGlobalNorm(target);
  // New global quasi-norm should be ~ target (within tolerance)
  double nsq2 = 0.0;
  for (auto &vec : t) {
    for (auto &ten : vec) {
      double q = ten.GetQuasi2Norm();
      nsq2 += q*q;
    }
  }
  EXPECT_NEAR(std::sqrt(nsq2), target, 1e-9);
}

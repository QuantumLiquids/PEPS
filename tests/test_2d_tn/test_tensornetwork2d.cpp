// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-03
*
* Description: GraceQ/VMC-PEPS project. Unittests for TensorNetwork2D
*/

#include "gtest/gtest.h"
#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "gqpeps/two_dim_tn/tps/split_index_tps.h"    //TPS, SplitIndexTPS

using namespace gqten;
using namespace gqpeps;

using gqten::special_qn::U1QN;
using gqten::special_qn::U1U1ZnQN;

struct Test2DIsingTensorNetworkNoQN : public testing::Test {
  using QNT = U1QN;
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<QNT>;
  using QNSctVecT = QNSectorVec<QNT>;
  using DGQTensor = GQTensor<GQTEN_Double, QNT>;
  using ZGQTensor = GQTensor<GQTEN_Complex, QNT>;

  const size_t Lx = 20;
  const size_t Ly = 20;
  const double beta = std::log(1 + std::sqrt(2.0)) / 2.0; // critical point
  IndexT vb_out = IndexT({QNSctT(QNT(0), 2)},
                         GQTenIndexDirType::OUT
  );
  IndexT vb_in = InverseIndex(vb_out);
  IndexT trivial_out = IndexT({QNSctT(QNT(0), 1)},
                              GQTenIndexDirType::OUT
  );
  IndexT trivial_in = InverseIndex(trivial_out);

  TensorNetwork2D<GQTEN_Double, QNT> dtn2d = TensorNetwork2D<GQTEN_Double, QNT>(Ly, Lx);
  TensorNetwork2D<GQTEN_Complex, QNT> ztn2d = TensorNetwork2D<GQTEN_Complex, QNT>(Ly, Lx);

  double F_ex = -2.0709079359461788;
//  double F_ex = -std::log(2.0) / beta; //high temperature approx
//  double F_ex = -(2.0 - (Lx+Ly)/(Lx*Ly)); //low temperature approx
  double Z_ex = std::exp(-F_ex * beta * Lx * Ly); //partition function, exact value
  double tn_free_en_norm_factor = 0.0;
  void SetUp(void) {
    DGQTensor boltzmann_weight = DGQTensor({vb_in, vb_out});
    double e = -1.0; // FM Ising
    boltzmann_weight({0, 0}) = std::exp(-1.0 * beta * e);
    boltzmann_weight({0, 1}) = std::exp(1.0 * beta * e);
    boltzmann_weight({1, 0}) = std::exp(1.0 * beta * e);
    boltzmann_weight({1, 1}) = std::exp(-1.0 * beta * e);

    DGQTensor core_ten_m = DGQTensor({vb_in, vb_out, vb_out, vb_in});
    for (size_t i = 0; i < 2; i++) {
      core_ten_m({i, i, i, i}) = 1.0;
    }
    DGQTensor t_m;// = DGQTensor({vb_in, vb_out, vb_out, vb_in});
    {
      DGQTensor temp[3];
      Contract(&boltzmann_weight, {1}, &core_ten_m, {3}, temp);
      Contract(&boltzmann_weight, {0}, temp, {3}, &t_m);
      t_m.Transpose({2, 3, 0, 1});
    }

    for (size_t row = 1; row < Ly - 1; row++) {
      for (size_t col = 1; col < Lx - 1; col++) {
        dtn2d({row, col}) = t_m;
      }
    }

    DGQTensor core_ten_up = DGQTensor({vb_in, vb_out, vb_out, trivial_in});
    DGQTensor core_ten_left = DGQTensor({trivial_in, vb_out, vb_out, vb_in});
    DGQTensor core_ten_down = DGQTensor({vb_in, trivial_out, vb_out, vb_in});
    DGQTensor core_ten_right = DGQTensor({vb_in, vb_out, trivial_out, vb_in});
    for (size_t i = 0; i < 2; i++) {
      core_ten_left({0, i, i, i}) = 1.0;
      core_ten_up({i, i, i, 0}) = 1.0;
      core_ten_down({i, 0, i, i}) = 1.0;
      core_ten_right({i, i, 0, i}) = 1.0;
    }

    DGQTensor t_up, t_left, t_down, t_right;
    {
      DGQTensor temp[3];
      temp[0] = core_ten_up;
      temp[0].Transpose({3, 0, 1, 2});
      Contract(&boltzmann_weight, {0}, temp, {3}, &t_up);
      t_up.Transpose({2, 3, 0, 1});
    }
    {
      DGQTensor temp[3];
      Contract(&boltzmann_weight, {1}, &core_ten_left, {3}, temp);
      Contract(&boltzmann_weight, {0}, temp, {3}, &t_left);
      t_left.Transpose({2, 3, 0, 1});
    }
    {

      Contract(&boltzmann_weight, {1}, &core_ten_right, {3}, &t_right);
      t_right.Transpose({1, 2, 3, 0});
    }
    {
      DGQTensor temp[3];
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

    DGQTensor core_ten_left_upper = DGQTensor({trivial_in, vb_out, vb_out, trivial_in});
    DGQTensor core_ten_left_lower = DGQTensor({trivial_in, trivial_out, vb_out, vb_in});
    DGQTensor core_ten_right_lower = DGQTensor({vb_in, trivial_out, trivial_out, vb_in});
    DGQTensor core_ten_right_upper = DGQTensor({vb_in, vb_out, trivial_out, trivial_in});
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
    }
    {
      DGQTensor temp[3];
      Contract(&boltzmann_weight, {1}, &core_ten_left_lower, {3}, temp);
      core_ten_left_lower = DGQTensor();
      Contract(&boltzmann_weight, {0}, temp, {3}, &core_ten_left_lower);
      core_ten_left_lower.Transpose({2, 3, 0, 1});
    }

    dtn2d({0, 0}) = core_ten_left_upper;
    dtn2d({Ly - 1, 0}) = core_ten_left_lower;
    dtn2d({Ly - 1, Lx - 1}) = core_ten_right_lower;
    dtn2d({0, Lx - 1}) = core_ten_right_upper;

    for (size_t row = 0; row < Ly; row++) {
      for (size_t col = 0; col < Lx; col++) {
        tn_free_en_norm_factor += std::log(dtn2d({row, col}).Normalize());
        ztn2d({row, col}) = ToComplex(dtn2d({row, col}));
      }
    }
    dtn2d.InitBMPS();
    ztn2d.InitBMPS();

    // calculate exact partition function

  }//SetUp
};

TEST_F(Test2DIsingTensorNetworkNoQN, Test2DIsingOBCTensorNetworkRealNumber) {
  double psi[22];
  auto dtn2d_copy = dtn2d;
  BMPSTruncatePara trunc_para = BMPSTruncatePara(10, 30, 1e-15, CompressMPSScheme::VARIATION2Site);
  dtn2d.GrowBMPSForRow(2, trunc_para);
  dtn2d.InitBTen(BTenPOSITION::LEFT, 2);
  dtn2d.GrowFullBTen(BTenPOSITION::RIGHT, 2, 2, true);
  psi[0] = dtn2d.Trace({2, 0}, HORIZONTAL);

  dtn2d.BTenMoveStep(BTenPOSITION::RIGHT);
  psi[1] = dtn2d.Trace({2, 1}, HORIZONTAL);

  dtn2d.GrowBMPSForCol(1, trunc_para);
  dtn2d.InitBTen(BTenPOSITION::DOWN, 1);
  dtn2d.GrowFullBTen(BTenPOSITION::UP, 1, 2, true);
  psi[2] = dtn2d.Trace({Ly - 2, 1}, VERTICAL);
  dtn2d.BTenMoveStep(BTenPOSITION::UP);
  psi[3] = dtn2d.Trace({Ly - 3, 1}, VERTICAL);

  trunc_para.compress_scheme = gqpeps::CompressMPSScheme::VARIATION1Site;
  dtn2d.GrowBMPSForRow(2, trunc_para);
  dtn2d.InitBTen(BTenPOSITION::LEFT, 2);
  dtn2d.GrowFullBTen(BTenPOSITION::RIGHT, 2, 2, true);
  psi[4] = dtn2d.Trace({2, 0}, HORIZONTAL);

  dtn2d.BTenMoveStep(BTenPOSITION::RIGHT);
  psi[5] = dtn2d.Trace({2, 1}, HORIZONTAL);

  dtn2d.GrowBMPSForCol(1, trunc_para);
  dtn2d.InitBTen(BTenPOSITION::DOWN, 1);
  dtn2d.GrowFullBTen(BTenPOSITION::UP, 1, 2, true);
  psi[6] = dtn2d.Trace({Ly - 2, 1}, VERTICAL);
  dtn2d.BTenMoveStep(BTenPOSITION::UP);
  psi[7] = dtn2d.Trace({Ly - 3, 1}, VERTICAL);

  /***** HORIZONTAL MPS *****/
  dtn2d.GrowBMPSForRow(1, trunc_para);
  dtn2d.InitBTen2(BTenPOSITION::LEFT, 1);
  dtn2d.GrowFullBTen2(BTenPOSITION::RIGHT, 1, 2, true);

  psi[8] = dtn2d.ReplaceNNNSiteTrace({1, 0}, LEFTDOWN_TO_RIGHTUP,
                                     HORIZONTAL,
                                     dtn2d({2, 0}), dtn2d({1, 1})); // trace original tn
  psi[9] = dtn2d.ReplaceNNNSiteTrace({1, 0}, LEFTUP_TO_RIGHTDOWN,
                                     HORIZONTAL,
                                     dtn2d({1, 0}), dtn2d({2, 1})); // trace original tn

  dtn2d.BTen2MoveStep(BTenPOSITION::RIGHT, 1);
  psi[10] = dtn2d.ReplaceNNNSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                      HORIZONTAL,
                                      dtn2d({2, 1}), dtn2d({1, 2})); // trace original tn

  psi[11] = dtn2d.ReplaceNNNSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                      HORIZONTAL,
                                      dtn2d({1, 1}), dtn2d({2, 2})); // trace original tn
  psi[12] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 0}, LEFTDOWN_TO_RIGHTUP,
                                               HORIZONTAL,
                                               dtn2d({2, 0}), dtn2d({1, 2})); // trace original tn
  psi[13] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                               HORIZONTAL,
                                               dtn2d({2, 1}), dtn2d({1, 3})); // trace original tn
  psi[14] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 0}, LEFTUP_TO_RIGHTDOWN,
                                               HORIZONTAL,
                                               dtn2d({1, 0}), dtn2d({2, 2})); // trace original tn
  psi[15] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                               HORIZONTAL,
                                               dtn2d({1, 1}), dtn2d({2, 3})); // trace original tn


  /***** VERTICAL MPS *****/
  dtn2d.GrowBMPSForCol(1, trunc_para);
  dtn2d.GrowFullBTen2(BTenPOSITION::DOWN, 1, 2, true);
  dtn2d.GrowFullBTen2(BTenPOSITION::UP, 1, 2, true);
  psi[16] = dtn2d.ReplaceNNNSiteTrace({2, 1}, LEFTDOWN_TO_RIGHTUP,
                                      VERTICAL,
                                      dtn2d({3, 1}), dtn2d({2, 2})); // trace original tn
  psi[17] = dtn2d.ReplaceNNNSiteTrace({2, 1}, LEFTUP_TO_RIGHTDOWN,
                                      VERTICAL,
                                      dtn2d({2, 1}), dtn2d({3, 2})); // trace original tn

  dtn2d.BTen2MoveStep(BTenPOSITION::UP, 1);
  psi[18] = dtn2d.ReplaceNNNSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                      VERTICAL,
                                      dtn2d({2, 1}), dtn2d({1, 2})); // trace original tn

  psi[19] = dtn2d.ReplaceNNNSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                      VERTICAL,
                                      dtn2d({1, 1}), dtn2d({2, 2})); // trace original tn
  psi[20] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                               VERTICAL,
                                               dtn2d({3, 1}), dtn2d({1, 2})); // trace original tn
  psi[21] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                               VERTICAL,
                                               dtn2d({1, 1}), dtn2d({3, 2})); // trace original tn

  for (size_t i = 1; i < 22; i++) {
    EXPECT_NEAR(-(std::log(psi[i]) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
  }
  dtn2d.BTen2MoveStep(BTenPOSITION::DOWN, 1);
  for (size_t i = 0; i < dtn2d_copy.rows(); i++) {
    for (size_t j = 0; j < dtn2d_copy.cols(); j++) {
      EXPECT_NE(dtn2d_copy(i, j), dtn2d(i, j));
      EXPECT_EQ(dtn2d_copy({i, j}), dtn2d({i, j}));
    }
  }
}

struct Test2DIsingTensorNetwork : public testing::Test {
  using QNT = U1U1ZnQN<2>;
  using IndexT = Index<U1U1ZnQN<2>>;
  using QNSctT = QNSector<U1U1ZnQN<2>>;
  using QNSctVecT = QNSectorVec<U1U1ZnQN<2>>;
  using DGQTensor = GQTensor<GQTEN_Double, U1U1ZnQN<2>>;
  using ZGQTensor = GQTensor<GQTEN_Complex, U1U1ZnQN<2>>;

  const size_t Lx = 20;
  const size_t Ly = 20;
  const double beta = std::log(1 + std::sqrt(2.0)) / 2.0; // critical point
  IndexT vb_out = IndexT({QNSctT(U1U1ZnQN<2>(0, 0, 0), 1),
                          QNSctT(U1U1ZnQN<2>(0, 0, 1), 1)},
                         GQTenIndexDirType::OUT
  );
  IndexT vb_in = InverseIndex(vb_out);
  IndexT trivial_out = IndexT({QNSctT(U1U1ZnQN<2>(0, 0, 0), 1)},
                              GQTenIndexDirType::OUT
  );
  IndexT trivial_in = InverseIndex(trivial_out);

  TensorNetwork2D<GQTEN_Double, QNT> dtn2d = TensorNetwork2D<GQTEN_Double, QNT>(Ly, Lx);
  TensorNetwork2D<GQTEN_Complex, QNT> ztn2d = TensorNetwork2D<GQTEN_Complex, QNT>(Ly, Lx);

  double F_ex = -2.0709079359461788;
  double Z_ex = std::exp(-F_ex * beta * Lx * Ly); //partition function, exact value
  double tn_free_en_norm_factor = 0.0;
  void SetUp(void) {
    DGQTensor boltzmann_weight = DGQTensor({vb_in, vb_out});
    double e = -1.0; // FM Ising
    boltzmann_weight({0, 0}) = std::exp(-1.0 * beta * e) + std::exp(1.0 * beta * e);
    boltzmann_weight({1, 1}) = std::exp(-1.0 * beta * e) - std::exp(1.0 * beta * e);
    auto boltzmann_weight_sqrt = boltzmann_weight;
    boltzmann_weight_sqrt({0, 0}) = std::sqrt(boltzmann_weight_sqrt({0, 0}));
    boltzmann_weight_sqrt({1, 1}) = std::sqrt(boltzmann_weight_sqrt({1, 1}));
    DGQTensor core_ten_m = DGQTensor({vb_in, vb_out, vb_out, vb_in});
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 2; j++) {
        for (size_t k = 0; k < 2; k++) {
          size_t l = (j + k + 2 - i) % 2;
          core_ten_m({i, j, k, l}) = 0.5;
        }
      }
    }
    DGQTensor t_m;// = DGQTensor({vb_in, vb_out, vb_out, vb_in});
    {
      DGQTensor temp[3];
      Contract(&boltzmann_weight_sqrt, {1}, &core_ten_m, {3}, temp);
      Contract(&boltzmann_weight_sqrt, {0}, temp, {3}, temp + 1);
      Contract(&boltzmann_weight_sqrt, {0}, temp + 1, {3}, temp + 2);
      Contract(&boltzmann_weight_sqrt, {1}, temp + 2, {3}, &t_m);
    }

//    t_m.Show();
    for (size_t row = 1; row < Ly - 1; row++) {
      for (size_t col = 1; col < Lx - 1; col++) {
        dtn2d({row, col}) = t_m;
      }
    }

    DGQTensor core_ten_up = DGQTensor({vb_in, vb_out, vb_out, trivial_in});
    DGQTensor core_ten_left = DGQTensor({trivial_in, vb_out, vb_out, vb_in});
    DGQTensor core_ten_down = DGQTensor({vb_in, trivial_out, vb_out, vb_in});
    DGQTensor core_ten_right = DGQTensor({vb_in, vb_out, trivial_out, vb_in});
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 2; j++) {
        size_t k = (i + j) % 2;
        core_ten_left({0, i, j, k}) = 1.0 / std::sqrt(2.0);
        core_ten_up({i, j, k, 0}) = 1.0 / std::sqrt(2.0);
        core_ten_down({i, 0, j, k}) = 1.0 / std::sqrt(2.0);
        core_ten_right({i, j, 0, k}) = 1.0 / std::sqrt(2.0);
      }
    }
    DGQTensor t_up, t_left, t_down, t_right;
    {
      DGQTensor temp[3];
      temp[0] = core_ten_up;
      temp[0].Transpose({3, 0, 1, 2});
      Contract(&boltzmann_weight_sqrt, {0}, temp, {3}, temp + 1);
      Contract(&boltzmann_weight_sqrt, {0}, temp + 1, {3}, temp + 2);
      Contract(&boltzmann_weight_sqrt, {1}, temp + 2, {3}, &t_up);
    }
    {
      DGQTensor temp[3];
      Contract(&boltzmann_weight_sqrt, {1}, &core_ten_left, {3}, temp);
      Contract(&boltzmann_weight_sqrt, {0}, temp, {3}, temp + 1);
      Contract(&boltzmann_weight_sqrt, {0}, temp + 1, {3}, temp + 2);
      (temp + 2)->Transpose({3, 0, 1, 2});
      t_left = temp[2];
    }
    {
      DGQTensor temp[3];
      Contract(&boltzmann_weight_sqrt, {1}, &core_ten_right, {3}, temp);
      temp->Transpose({3, 0, 1, 2});
      Contract(&boltzmann_weight_sqrt, {0}, temp, {3}, temp + 2);
      Contract(&boltzmann_weight_sqrt, {1}, temp + 2, {3}, &t_right);
    }
    {
      DGQTensor temp[3];
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

    DGQTensor core_ten_left_upper = DGQTensor({trivial_in, vb_out, vb_out, trivial_in});
    DGQTensor core_ten_left_lower = DGQTensor({trivial_in, trivial_out, vb_out, vb_in});
    DGQTensor core_ten_right_lower = DGQTensor({vb_in, trivial_out, trivial_out, vb_in});
    DGQTensor core_ten_right_upper = DGQTensor({vb_in, vb_out, trivial_out, trivial_in});
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

    for (size_t row = 0; row < Ly; row++) {
      for (size_t col = 0; col < Lx; col++) {
        tn_free_en_norm_factor += std::log(dtn2d({row, col}).Normalize());
        ztn2d({row, col}) = ToComplex(dtn2d({row, col}));
      }
    }
    dtn2d.InitBMPS();
    ztn2d.InitBMPS();

    // calculate exact partition function

  }//SetUp
};

TEST_F(Test2DIsingTensorNetwork, Test2DIsingOBCTensorNetworkRealNumber) {
  double psi[22];
  BMPSTruncatePara trunc_para = BMPSTruncatePara(10, 30, 1e-15, CompressMPSScheme::VARIATION2Site);
  dtn2d.GrowBMPSForRow(2, trunc_para);
  dtn2d.InitBTen(BTenPOSITION::LEFT, 2);
  dtn2d.GrowFullBTen(BTenPOSITION::RIGHT, 2, 2, true);
  psi[0] = dtn2d.Trace({2, 0}, HORIZONTAL);

  dtn2d.BTenMoveStep(BTenPOSITION::RIGHT);
  psi[1] = dtn2d.Trace({2, 1}, HORIZONTAL);

  dtn2d.GrowBMPSForCol(1, trunc_para);
  dtn2d.InitBTen(BTenPOSITION::DOWN, 1);
  dtn2d.GrowFullBTen(BTenPOSITION::UP, 1, 2, true);
  psi[2] = dtn2d.Trace({Ly - 2, 1}, VERTICAL);
  dtn2d.BTenMoveStep(BTenPOSITION::UP);
  psi[3] = dtn2d.Trace({Ly - 3, 1}, VERTICAL);

  trunc_para.compress_scheme = gqpeps::CompressMPSScheme::VARIATION1Site;
  dtn2d.GrowBMPSForRow(2, trunc_para);
  dtn2d.InitBTen(BTenPOSITION::LEFT, 2);
  dtn2d.GrowFullBTen(BTenPOSITION::RIGHT, 2, 2, true);
  psi[4] = dtn2d.Trace({2, 0}, HORIZONTAL);

  dtn2d.BTenMoveStep(BTenPOSITION::RIGHT);
  psi[5] = dtn2d.Trace({2, 1}, HORIZONTAL);

  dtn2d.GrowBMPSForCol(1, trunc_para);
  dtn2d.InitBTen(BTenPOSITION::DOWN, 1);
  dtn2d.GrowFullBTen(BTenPOSITION::UP, 1, 2, true);
  psi[6] = dtn2d.Trace({Ly - 2, 1}, VERTICAL);
  dtn2d.BTenMoveStep(BTenPOSITION::UP);
  psi[7] = dtn2d.Trace({Ly - 3, 1}, VERTICAL);

  /***** HORIZONTAL MPS *****/
  dtn2d.GrowBMPSForRow(1, trunc_para);
  dtn2d.InitBTen2(BTenPOSITION::LEFT, 1);
  dtn2d.GrowFullBTen2(BTenPOSITION::RIGHT, 1, 2, true);

  psi[8] = dtn2d.ReplaceNNNSiteTrace({1, 0}, LEFTDOWN_TO_RIGHTUP,
                                     HORIZONTAL,
                                     dtn2d({2, 0}), dtn2d({1, 1})); // trace original tn
  psi[9] = dtn2d.ReplaceNNNSiteTrace({1, 0}, LEFTUP_TO_RIGHTDOWN,
                                     HORIZONTAL,
                                     dtn2d({1, 0}), dtn2d({2, 1})); // trace original tn

  dtn2d.BTen2MoveStep(BTenPOSITION::RIGHT, 1);
  psi[10] = dtn2d.ReplaceNNNSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                      HORIZONTAL,
                                      dtn2d({2, 1}), dtn2d({1, 2})); // trace original tn

  psi[11] = dtn2d.ReplaceNNNSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                      HORIZONTAL,
                                      dtn2d({1, 1}), dtn2d({2, 2})); // trace original tn
  psi[12] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 0}, LEFTDOWN_TO_RIGHTUP,
                                               HORIZONTAL,
                                               dtn2d({2, 0}), dtn2d({1, 2})); // trace original tn
  psi[13] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                               HORIZONTAL,
                                               dtn2d({2, 1}), dtn2d({1, 3})); // trace original tn
  psi[14] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 0}, LEFTUP_TO_RIGHTDOWN,
                                               HORIZONTAL,
                                               dtn2d({1, 0}), dtn2d({2, 2})); // trace original tn
  psi[15] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                               HORIZONTAL,
                                               dtn2d({1, 1}), dtn2d({2, 3})); // trace original tn


  /***** VERTICAL MPS *****/
  dtn2d.GrowBMPSForCol(1, trunc_para);
  dtn2d.GrowFullBTen2(BTenPOSITION::DOWN, 1, 2, true);
  dtn2d.GrowFullBTen2(BTenPOSITION::UP, 1, 2, true);
  psi[16] = dtn2d.ReplaceNNNSiteTrace({2, 1}, LEFTDOWN_TO_RIGHTUP,
                                      VERTICAL,
                                      dtn2d({3, 1}), dtn2d({2, 2})); // trace original tn
  psi[17] = dtn2d.ReplaceNNNSiteTrace({2, 1}, LEFTUP_TO_RIGHTDOWN,
                                      VERTICAL,
                                      dtn2d({2, 1}), dtn2d({3, 2})); // trace original tn

  dtn2d.BTen2MoveStep(BTenPOSITION::UP, 1);
  psi[18] = dtn2d.ReplaceNNNSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                      VERTICAL,
                                      dtn2d({2, 1}), dtn2d({1, 2})); // trace original tn

  psi[19] = dtn2d.ReplaceNNNSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                      VERTICAL,
                                      dtn2d({1, 1}), dtn2d({2, 2})); // trace original tn
  psi[20] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                               VERTICAL,
                                               dtn2d({3, 1}), dtn2d({1, 2})); // trace original tn
  psi[21] = dtn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                               VERTICAL,
                                               dtn2d({1, 1}), dtn2d({3, 2})); // trace original tn

  for (size_t i = 1; i < 22; i++) {
    EXPECT_NEAR(-(std::log(psi[i]) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
  }
  dtn2d.BTen2MoveStep(BTenPOSITION::DOWN, 1);
}

TEST_F(Test2DIsingTensorNetwork, Test2DIsingOBCTensorNetworkComplexNumber) {
  GQTEN_Complex psi[22];
  BMPSTruncatePara trunc_para = BMPSTruncatePara(10, 30, 1e-15, CompressMPSScheme::VARIATION2Site);
  ztn2d.GrowBMPSForRow(2, trunc_para);
  ztn2d.InitBTen(BTenPOSITION::LEFT, 2);
  ztn2d.GrowFullBTen(BTenPOSITION::RIGHT, 2, 2, true);
  psi[0] = ztn2d.Trace({2, 0}, HORIZONTAL);

  ztn2d.BTenMoveStep(BTenPOSITION::RIGHT);
  psi[1] = ztn2d.Trace({2, 1}, HORIZONTAL);

  ztn2d.GrowBMPSForCol(1, trunc_para);
  ztn2d.InitBTen(BTenPOSITION::DOWN, 1);
  ztn2d.GrowFullBTen(BTenPOSITION::UP, 1, 2, true);
  psi[2] = ztn2d.Trace({Ly - 2, 1}, VERTICAL);
  ztn2d.BTenMoveStep(BTenPOSITION::UP);
  psi[3] = ztn2d.Trace({Ly - 3, 1}, VERTICAL);

  trunc_para.compress_scheme = gqpeps::CompressMPSScheme::VARIATION1Site;
  ztn2d.GrowBMPSForRow(2, trunc_para);
  ztn2d.InitBTen(BTenPOSITION::LEFT, 2);
  ztn2d.GrowFullBTen(BTenPOSITION::RIGHT, 2, 2, true);
  psi[4] = ztn2d.Trace({2, 0}, HORIZONTAL);

  ztn2d.BTenMoveStep(BTenPOSITION::RIGHT);
  psi[5] = ztn2d.Trace({2, 1}, HORIZONTAL);

  ztn2d.GrowBMPSForCol(1, trunc_para);
  ztn2d.InitBTen(BTenPOSITION::DOWN, 1);
  ztn2d.GrowFullBTen(BTenPOSITION::UP, 1, 2, true);
  psi[6] = ztn2d.Trace({Ly - 2, 1}, VERTICAL);
  ztn2d.BTenMoveStep(BTenPOSITION::UP);
  psi[7] = ztn2d.Trace({Ly - 3, 1}, VERTICAL);

  /***** HORIZONTAL MPS *****/
  ztn2d.GrowBMPSForRow(1, trunc_para);
  ztn2d.InitBTen2(BTenPOSITION::LEFT, 1);
  ztn2d.GrowFullBTen2(BTenPOSITION::RIGHT, 1, 2, true);

  psi[8] = ztn2d.ReplaceNNNSiteTrace({1, 0}, LEFTDOWN_TO_RIGHTUP,
                                     HORIZONTAL,
                                     ztn2d({2, 0}), ztn2d({1, 1})); // trace original tn
  psi[9] = ztn2d.ReplaceNNNSiteTrace({1, 0}, LEFTUP_TO_RIGHTDOWN,
                                     HORIZONTAL,
                                     ztn2d({1, 0}), ztn2d({2, 1})); // trace original tn

  ztn2d.BTen2MoveStep(BTenPOSITION::RIGHT, 1);
  psi[10] = ztn2d.ReplaceNNNSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                      HORIZONTAL,
                                      ztn2d({2, 1}), ztn2d({1, 2})); // trace original tn

  psi[11] = ztn2d.ReplaceNNNSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                      HORIZONTAL,
                                      ztn2d({1, 1}), ztn2d({2, 2})); // trace original tn
  psi[12] = ztn2d.ReplaceSqrt5DistTwoSiteTrace({1, 0}, LEFTDOWN_TO_RIGHTUP,
                                               HORIZONTAL,
                                               ztn2d({2, 0}), ztn2d({1, 2})); // trace original tn
  psi[13] = ztn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                               HORIZONTAL,
                                               ztn2d({2, 1}), ztn2d({1, 3})); // trace original tn
  psi[14] = ztn2d.ReplaceSqrt5DistTwoSiteTrace({1, 0}, LEFTUP_TO_RIGHTDOWN,
                                               HORIZONTAL,
                                               ztn2d({1, 0}), ztn2d({2, 2})); // trace original tn
  psi[15] = ztn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                               HORIZONTAL,
                                               ztn2d({1, 1}), ztn2d({2, 3})); // trace original tn




  /***** VERTICAL MPS *****/
  ztn2d.GrowBMPSForCol(1, trunc_para);
  ztn2d.GrowFullBTen2(BTenPOSITION::DOWN, 1, 2, true);
  ztn2d.GrowFullBTen2(BTenPOSITION::UP, 1, 2, true);
  psi[16] = ztn2d.ReplaceNNNSiteTrace({2, 1}, LEFTDOWN_TO_RIGHTUP,
                                      VERTICAL,
                                      ztn2d({3, 1}), ztn2d({2, 2})); // trace original tn
  psi[17] = ztn2d.ReplaceNNNSiteTrace({2, 1}, LEFTUP_TO_RIGHTDOWN,
                                      VERTICAL,
                                      ztn2d({2, 1}), ztn2d({3, 2})); // trace original tn

  ztn2d.BTen2MoveStep(BTenPOSITION::UP, 1);
  psi[18] = ztn2d.ReplaceNNNSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                      VERTICAL,
                                      ztn2d({2, 1}), ztn2d({1, 2})); // trace original tn

  psi[19] = ztn2d.ReplaceNNNSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                      VERTICAL,
                                      ztn2d({1, 1}), ztn2d({2, 2})); // trace original tn
  psi[20] = ztn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                               VERTICAL,
                                               ztn2d({3, 1}), ztn2d({1, 2})); // trace original tn
  psi[21] = ztn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                               VERTICAL,
                                               ztn2d({1, 1}), ztn2d({3, 2})); // trace original tn

  for (size_t i = 1; i < 22; i++) {
    EXPECT_NEAR(-(std::log(psi[i].real()) + tn_free_en_norm_factor) / Lx / Ly / beta, F_ex, 1e-8);
    EXPECT_DOUBLE_EQ(psi[i].imag(), 0.0);
  }
  ztn2d.BTen2MoveStep(BTenPOSITION::DOWN, 1);
}

struct TestSpin2DTensorNetwork : public testing::Test {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using QNSctVecT = QNSectorVec<U1QN>;
  using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
  using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

  const size_t Lx = 4;  // cols
  const size_t Ly = 4;  // rows

#ifdef U1SYM
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         GQTenIndexDirType::OUT
  );
#else
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         GQTenIndexDirType::OUT
  );
#endif

  IndexT pb_in = InverseIndex(pb_out);

  Configuration config = Configuration(Ly, Lx);

  TensorNetwork2D<GQTEN_Double, U1QN> tn2d = TensorNetwork2D<GQTEN_Double, U1QN>(Ly, Lx);

  BMPSTruncatePara trunc_para = BMPSTruncatePara(4, 8, 1e-12, CompressMPSScheme::VARIATION2Site);

  void SetUp(void) {
    TPS<GQTEN_Double, U1QN> tps(Ly, Lx);
//    gqten::hp_numeric::SetTensorManipulationThreads(1);
//    gqten::hp_numeric::SetTensorTransposeNumThreads(1);
    tps.Load("tps_heisenberg_D4");

    SplitIndexTPS<GQTEN_Double, U1QN> split_index_tps(tps);
    for (size_t i = 0; i < Lx; i++) { //col index
      for (size_t j = 0; j < Ly; j++) { //row index
        config({j, i}) = (i + j) % 2;
      }
    }
    tn2d = TensorNetwork2D<GQTEN_Double, U1QN>(split_index_tps, config);
  }
};

TEST_F(TestSpin2DTensorNetwork, HeisenbergD4NNTraceBMPS2SiteVariationUpdate) {
  tn2d.GrowBMPSForRow(2, trunc_para);
  tn2d.InitBTen(BTenPOSITION::LEFT, 2);
  tn2d.GrowFullBTen(BTenPOSITION::RIGHT, 2, 2, true);
  double psi_a = tn2d.Trace({2, 0}, HORIZONTAL);
  std::cout << "Amplitude by horizontal BMPS = " << psi_a << std::endl;

  tn2d.BTenMoveStep(BTenPOSITION::RIGHT);
  double psi_b = tn2d.Trace({2, 1}, HORIZONTAL);
  EXPECT_NEAR(psi_a, psi_b, 1e-14);

  tn2d.GrowBMPSForCol(1, trunc_para);
  tn2d.InitBTen(BTenPOSITION::DOWN, 1);
  tn2d.GrowFullBTen(BTenPOSITION::UP, 1, 2, true);
  double psi_c = tn2d.Trace({Ly - 2, 1}, VERTICAL);
  std::cout << "Amplitude by vertical BMPS = " << psi_c << std::endl;
  tn2d.BTenMoveStep(BTenPOSITION::UP);
  double psi_d = tn2d.Trace({Ly - 3, 1}, VERTICAL);
  EXPECT_NEAR(psi_c, psi_d, 1e-14);
}

TEST_F(TestSpin2DTensorNetwork, HeisenbergD4NNTraceBMPSSingleSiteVariationUpdate) {
  trunc_para.compress_scheme = gqpeps::CompressMPSScheme::VARIATION1Site;
  tn2d.GrowBMPSForRow(2, trunc_para);
  tn2d.InitBTen(BTenPOSITION::LEFT, 2);
  tn2d.GrowFullBTen(BTenPOSITION::RIGHT, 2, 2, true);
  double psi_a = tn2d.Trace({2, 0}, HORIZONTAL);
  std::cout << "Amplitude by horizontal BMPS = " << psi_a << std::endl;

  tn2d.BTenMoveStep(BTenPOSITION::RIGHT);
  double psi_b = tn2d.Trace({2, 1}, HORIZONTAL);
  EXPECT_NEAR(psi_a, psi_b, 1e-15);

  tn2d.GrowBMPSForCol(1, trunc_para);
  tn2d.InitBTen(BTenPOSITION::DOWN, 1);
  tn2d.GrowFullBTen(BTenPOSITION::UP, 1, 2, true);
  double psi_c = tn2d.Trace({Ly - 2, 1}, VERTICAL);
  std::cout << "Amplitude by vertical BMPS = " << psi_c << std::endl;
  tn2d.BTenMoveStep(BTenPOSITION::UP);
  double psi_d = tn2d.Trace({Ly - 3, 1}, VERTICAL);
  EXPECT_NEAR(psi_c, psi_d, 1e-15);
}

TEST_F(TestSpin2DTensorNetwork, HeisenbergD4BTen2Trace) {
  /***** HORIZONTAL MPS *****/
  tn2d.GrowBMPSForRow(1, trunc_para);
  tn2d.InitBTen2(BTenPOSITION::LEFT, 1);
  tn2d.GrowFullBTen2(BTenPOSITION::RIGHT, 1, 2, true);
  double psi[8];
  psi[0] = tn2d.ReplaceNNNSiteTrace({1, 0}, LEFTDOWN_TO_RIGHTUP,
                                    HORIZONTAL,
                                    tn2d({2, 0}), tn2d({1, 1})); // trace original tn
  psi[1] = tn2d.ReplaceNNNSiteTrace({1, 0}, LEFTUP_TO_RIGHTDOWN,
                                    HORIZONTAL,
                                    tn2d({1, 0}), tn2d({2, 1})); // trace original tn
  std::cout << "Amplitude by horizontal BMPS = " << psi[0] << std::endl;

  tn2d.BTen2MoveStep(BTenPOSITION::RIGHT, 1);
  psi[2] = tn2d.ReplaceNNNSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                    HORIZONTAL,
                                    tn2d({2, 1}), tn2d({1, 2})); // trace original tn

  psi[3] = tn2d.ReplaceNNNSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                    HORIZONTAL,
                                    tn2d({1, 1}), tn2d({2, 2})); // trace original tn
  psi[4] = tn2d.ReplaceSqrt5DistTwoSiteTrace({1, 0}, LEFTDOWN_TO_RIGHTUP,
                                             HORIZONTAL,
                                             tn2d({2, 0}), tn2d({1, 2})); // trace original tn
  psi[5] = tn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                             HORIZONTAL,
                                             tn2d({2, 1}), tn2d({1, 3})); // trace original tn
  psi[6] = tn2d.ReplaceSqrt5DistTwoSiteTrace({1, 0}, LEFTUP_TO_RIGHTDOWN,
                                             HORIZONTAL,
                                             tn2d({1, 0}), tn2d({2, 2})); // trace original tn
  psi[7] = tn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                             HORIZONTAL,
                                             tn2d({1, 1}), tn2d({2, 3})); // trace original tn

  for (size_t i = 1; i < 8; i++) {
    EXPECT_NEAR(psi[0], psi[i], 1e-15);
  }


  /***** VERTICAL MPS *****/
  tn2d.GrowBMPSForCol(1, trunc_para);
  tn2d.InitBTen2(BTenPOSITION::DOWN, 1);
  tn2d.GrowFullBTen2(BTenPOSITION::UP, 1, 2, true);
  psi[0] = tn2d.ReplaceNNNSiteTrace({2, 1}, LEFTDOWN_TO_RIGHTUP,
                                    VERTICAL,
                                    tn2d({3, 1}), tn2d({2, 2})); // trace original tn
  psi[1] = tn2d.ReplaceNNNSiteTrace({2, 1}, LEFTUP_TO_RIGHTDOWN,
                                    VERTICAL,
                                    tn2d({2, 1}), tn2d({3, 2})); // trace original tn
  std::cout << "Amplitude by horizontal BMPS = " << psi[0] << std::endl;

  tn2d.BTen2MoveStep(BTenPOSITION::UP, 1);
  psi[2] = tn2d.ReplaceNNNSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                    VERTICAL,
                                    tn2d({2, 1}), tn2d({1, 2})); // trace original tn

  psi[3] = tn2d.ReplaceNNNSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                    VERTICAL,
                                    tn2d({1, 1}), tn2d({2, 2})); // trace original tn
  psi[5] = tn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTDOWN_TO_RIGHTUP,
                                             VERTICAL,
                                             tn2d({3, 1}), tn2d({1, 2})); // trace original tn
  psi[7] = tn2d.ReplaceSqrt5DistTwoSiteTrace({1, 1}, LEFTUP_TO_RIGHTDOWN,
                                             VERTICAL,
                                             tn2d({1, 1}), tn2d({3, 2})); // trace original tn
  for (size_t i = 1; i < 8; i++) {
    EXPECT_NEAR(psi[0], psi[i], 1e-15);
  }
  tn2d.BTen2MoveStep(BTenPOSITION::DOWN, 1);
}
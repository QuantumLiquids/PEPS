/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-08-09
*
* Description: QuantumLiquids/PEPS project. Unittests for Arnoldi Solver.
*/
#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/peps/arnoldi_solver.h"

using namespace qlten;
using namespace qlpeps;
using qlten::special_qn::U1QN;

template<typename TenElemT, typename QNT>
void RunTestArnoldiSolver(
    const QLTensor<QLTEN_Double, QNT> &sigma,
    const QLTensor<TenElemT, QNT> &Upsilon,
    const ArnoldiParams &params
) {
  using TenT = QLTensor<TenElemT, QNT>;
  auto sigma_dag = Dag(sigma);
  const Index<QNT> index0 = InverseIndex(sigma.GetIndex(0));
  const Index<QNT> index1 = sigma.GetIndex(0);
  TenT left_vec0 = TenT({index0, index1});
  TenT right_vec0 = TenT({index1, index0});
  for (size_t j = 0; j < index0.dim(); j++) {
    left_vec0({j, j}) = 1.0;
    right_vec0({j, j}) = 1.0;
  }

  auto eig_res = ArnoldiSolver(Upsilon,
                               sigma,
                               sigma_dag,
                               left_vec0,
                               params,
                               TransfTenMultiVec<TenElemT, QNT>(left_vec_multiple_transfer_tens<TenElemT, QNT>));

  auto multipled_left_vec = left_vec_multiple_transfer_tens(eig_res.eig_vec, sigma, sigma_dag, Upsilon);
  auto multipled_left_vec2 = eig_res.eigenvalue * eig_res.eig_vec;
  TenT diff = multipled_left_vec + (-multipled_left_vec2);
  EXPECT_TRUE(diff.Get2Norm() / multipled_left_vec.Get2Norm() < 1e-10);

  eig_res = ArnoldiSolver(Upsilon,
                          sigma,
                          sigma_dag,
                          right_vec0,
                          params,
                          TransfTenMultiVec<TenElemT, QNT>(right_vec_multiple_transfer_tens<TenElemT, QNT>));

  auto multipled_right_vec = right_vec_multiple_transfer_tens(eig_res.eig_vec, sigma, sigma_dag, Upsilon);
  auto multipled_right_vec2 = eig_res.eigenvalue * eig_res.eig_vec;
  diff = multipled_right_vec + (-multipled_right_vec2);
  EXPECT_TRUE(diff.Get2Norm() / multipled_right_vec.Get2Norm() < 1e-10);
}

struct TensorSet : public testing::Test {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using QNSctVecT = QNSectorVec<U1QN>;

  using DTensor = QLTensor<QLTEN_Double, U1QN>;
  using ZTensor = QLTensor<QLTEN_Complex, U1QN>;

  std::vector<size_t> d = {4, 10, 20};
  U1QN qn0 = U1QN(0);
  IndexT pb_out = IndexT({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_in);
  std::vector<IndexT> idx_in, idx_out;
  void SetUp(void) {
    idx_in.resize(d.size());
    idx_out.resize(d.size());
    for (size_t i = 0; i < d.size(); i++) {
      idx_in[i] = IndexT({QNSctT(qn0, d[i])}, TenIndexDirType::IN);;
      idx_out[i] = InverseIndex(idx_in[i]);
    }
  }
};

TEST_F(TensorSet, RunTestArnoldiSolver) {
  ArnoldiParams params(1e-8, 500);
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<> dist(0, 1);

  for (size_t i = 0; i < d.size(); i++) {
    DTensor sigma({idx_in[i], idx_out[i]});
    for (size_t j = 0; j < idx_in[i].dim(); j++) {
      sigma({j, j}) = dist(rng);
    }
    DTensor dgamma({idx_in[i], pb_out, idx_out[i]});
    dgamma.Random(qn0);
    DTensor dgamma_dag = Dag(dgamma);
    DTensor dUpsilon;
    Contract(&dgamma, {1}, &dgamma_dag, {1}, &dUpsilon);
    dUpsilon.Transpose({0, 2, 1, 3});
    RunTestArnoldiSolver(sigma, dUpsilon, params);
    ZTensor zgamma({idx_in[i], pb_out, idx_out[i]});
    zgamma.Random(qn0);
    ZTensor zgamma_dag = Dag(zgamma);
    ZTensor zUpsilon;
    Contract(&zgamma, {1}, &zgamma_dag, {1}, &zUpsilon);
    zUpsilon.Transpose({0, 2, 1, 3});
    RunTestArnoldiSolver(sigma, zUpsilon, params);
  }
}
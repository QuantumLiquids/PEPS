/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-07-23
*
* Description: QuantumLiquids/PEPS project. The square PEPS class, project 4-site projectors implementation.
* Reference: [1] PRB 102, 075147 (2020), "Loop update for iPEPS in 2D".
*            [2] arXiv: 1801.05390v2, Glen Evenbly
*/


#ifndef VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_PROJECTION4_IMPL_H
#define VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_PROJECTION4_IMPL_H

#include "arnoldi_solver.h"

namespace qlpeps {
using namespace qlten;
using qlmps::mock_qlten::SVD;

//helper

/**
 * SVD for 2-leg tensors with one index pointing in and the other pointing out.
 *
 * The output tensor will keep the same index direction forms with the input tensor.
 */
template<typename TenElemT, typename QNT>
void MatSVD(
    const QLTensor<TenElemT, QNT> &t,
    const QNT &lqndiv,
    const QLTEN_Double trunc_err, const size_t Dmin, const size_t Dmax,
    QLTensor<TenElemT, QNT> &u,
    QLTensor<QLTEN_Double, QNT> &s,
    QLTensor<TenElemT, QNT> &vt,
    QLTEN_Double *pactual_trunc_err, size_t *pD
) {
  assert(t.Rank() == 2);
  assert(t.GetIndex(0).GetDir() != t.GetIndex(1).GetDir());
  if (t.GetIndex(0).GetDir() == IN) {
    SVD(&t, 1, lqndiv, trunc_err, Dmin, Dmax, &u, &s, &vt, pactual_trunc_err, pD);
    return;
  } else {
    QLTensor<TenElemT, QNT> transpose_t = t;
    transpose_t.Transpose({1, 0});
    SVD(&transpose_t, 1, t.Div() + (-lqndiv), trunc_err, Dmin, Dmax, &vt, &s, &u, pactual_trunc_err, pD);
    u.Transpose({1, 0});
    s.Transpose({1, 0});
    vt.Transpose({1, 0});
  }
}

///< no truncation version
template<typename TenElemT, typename QNT>
void MatSVD(
    const QLTensor<TenElemT, QNT> &t,
    const QNT &lqndiv,
    QLTensor<TenElemT, QNT> &u,
    QLTensor<QLTEN_Double, QNT> &s,
    QLTensor<TenElemT, QNT> &vt
) {
  assert(t.Rank() == 2);
  assert(t.GetIndex(0).GetDir() != t.GetIndex(1).GetDir());
  if (t.GetIndex(0).GetDir() == IN) {
    SVD(&t, 1, lqndiv, &u, &s, &vt);
    return;
  } else {
    QLTensor<TenElemT, QNT> transpose_t = t;
    transpose_t.Transpose({1, 0});
    SVD(&transpose_t, 1, t.Div() + (-lqndiv), &vt, &s, &u);
    u.Transpose({1, 0});
    s.Transpose({1, 0});
    vt.Transpose({1, 0});
  }
}

/**
 * MPS order :
 *     2    3(PEPS virtual leg but part of physical leg in MPS context)
 *      \  /
 *  0---Gamma---4(one of PEPS virtual leg, also virtual leg in MPS context)
 *        |
 *        1 (PEPS physical leg, physical leg also in MPS context)
 *
 *  See Ref[1] Figure 2 (c)
 */
template<typename TenElemT, typename QNT>
void TransposeGammaTensorIndicesIntoMPSOrder(std::array<QLTensor<TenElemT, QNT>, 4> &gammas) {
  gammas[0].Transpose({1, 4, 0, 3, 2});
  gammas[1].Transpose({0, 4, 2, 3, 1});
  gammas[2].Transpose({3, 4, 1, 2, 0});
  gammas[3].Transpose({2, 4, 0, 1, 3});
}

///< indices order of gamma are not changed
template<typename TenElemT, typename QNT>
void Eat2EnvLambdasInMPSOrderGamma(QLTensor<TenElemT, QNT> &gamma,
                                   const QLTensor<TenElemT, QNT> &env_lambda_l,
                                   const QLTensor<TenElemT, QNT> &env_lambda_r) {
  QLTensor<TenElemT, QNT> tmp, res;
  Contract(&gamma, {2}, &env_lambda_l, {1}, &tmp);
  Contract(&tmp, {2}, &env_lambda_r, {1}, &res);
  res.Transpose({0, 1, 3, 4, 2});
  gamma = res;
}

template<typename TenElemT, typename QNT>
void TransposeBackGammaTensorIndicesFromMPSOrder(std::array<QLTensor<TenElemT, QNT>, 4> &gammas) {
  gammas[0].Transpose({2, 0, 4, 3, 1});
  gammas[1].Transpose({0, 4, 2, 3, 1});
  gammas[2].Transpose({4, 2, 3, 0, 1});
  gammas[3].Transpose({2, 3, 0, 4, 1});
}

template<typename TenElemT, typename QNT>
void SplitOut2EnvLambdasInMPSOrderGammas(QLTensor<TenElemT, QNT> &gamma,
                                         const QLTensor<TenElemT, QNT> &env_lambda_l,
                                         const QLTensor<TenElemT, QNT> &env_lambda_r
) {
  QLTensor<TenElemT, QNT> tmp, res, inv_lambda_l, inv_lambda_r;
  inv_lambda_l = ElementWiseInv(env_lambda_l, 1e-200);
  inv_lambda_r = ElementWiseInv(env_lambda_r, 1e-200);
  Contract(&gamma, {2}, &inv_lambda_l, {1}, &tmp);
  Contract(&tmp, {2}, &inv_lambda_r, {1}, &res);
  res.Transpose({0, 1, 3, 4, 2});
  gamma = res;
}

//forward declaration
template<typename TenElemT, typename QNT>
void WeightedTraceGaugeFixingInSquareLocalLoop(
    const ArnoldiParams &arnoldi_params,
    std::array<QLTensor<TenElemT, QNT>, 4> &gammas,  //input & output
    std::array<QLTensor<QLTEN_Double, QNT>, 4> &lambdas, //input & output
    std::array<QLTensor<TenElemT, QNT>, 4> &Upsilons //output
);

template<typename TenElemT, typename QNT>
double SquareLatticePEPS<TenElemT, QNT>::LocalSquareLoopProject(
    const qlpeps::SquareLatticePEPS<TenElemT, QNT>::LocalSquareLoopGateT &gate_tens,
    const qlpeps::SiteIdx &upper_left_site,
    const qlpeps::LoopUpdateTruncatePara &params,
    const bool print_time) {
  const size_t row = upper_left_site.row();
  const size_t col = upper_left_site.col();
  Timer pat_loop_projector_timer("pat_loop_projector");
  PatSquareLocalLoopProjector_(gate_tens, upper_left_site);
  if (print_time) {
    pat_loop_projector_timer.PrintElapsed();
  }
  Timer loop_projection_pre_procedure_timer("loop_projection_pre_procedure");
  std::array<QLTensor<TenElemT, QNT>, 4> gammas, lambdas, Upsilons, env_lambda_ls, env_lambda_rs;
  gammas[0] = Gamma(upper_left_site);
  gammas[1] = Gamma({row, col + 1});
  gammas[2] = Gamma({row + 1, col + 1});
  gammas[3] = Gamma({row + 1, col});
  lambdas[0] = lambda_horiz({row, col + 1});
  lambdas[1] = lambda_vert({row + 1, col + 1});
  lambdas[2] = lambda_horiz({row + 1, col + 1});
  lambdas[3] = lambda_vert({row + 1, col});
  lambdas[2].Transpose({1, 0});
  lambdas[3].Transpose({1, 0});

  TransposeGammaTensorIndicesIntoMPSOrder(gammas);
#ifndef NDEBUG
  //check the indices
  for (size_t i = 0; i < 4; i++) {
    auto idx_a = gammas[i].GetIndex(4);
    auto idx_b = lambdas[i].GetIndex(0);
    assert(idx_a == InverseIndex(idx_b));
  }
#endif

  // eat lambdas of envs
  env_lambda_ls[0] = lambda_horiz({row, col});
  env_lambda_rs[0] = lambda_vert({row, col});
  env_lambda_ls[1] = lambda_horiz({row, col + 2});
  env_lambda_ls[1].Transpose({1, 0});
  env_lambda_rs[1] = lambda_vert({row, col + 1});
  env_lambda_ls[2] = lambda_vert({row + 2, col + 1});
  env_lambda_ls[2].Transpose({1, 0});
  env_lambda_rs[2] = lambda_horiz({row + 1, col + 2});
  env_lambda_rs[2].Transpose({1, 0});
  env_lambda_ls[3] = lambda_horiz({row + 1, col});
  env_lambda_rs[3] = lambda_vert({row + 2, col});
  env_lambda_rs[3].Transpose({1, 0});
  for (size_t i = 0; i < 4; i++) {
    Eat2EnvLambdasInMPSOrderGamma(gammas[i], env_lambda_ls[i], env_lambda_rs[i]);
  }
  if (print_time) {
    loop_projection_pre_procedure_timer.PrintElapsed();
  }
  Timer weighted_trace_gauge_fixing_timer("weighted_trace_gauge_fixing");
  WeightedTraceGaugeFixingInSquareLocalLoop(params.arnoldi_params, gammas, lambdas, Upsilons);
  if (print_time) {
    weighted_trace_gauge_fixing_timer.PrintElapsed();
  }
  Timer full_env_truncate_timer("full_env_truncate");
  FullEnvironmentTruncateInSquareLocalLoop_(params.fet_params, gammas, lambdas, Upsilons);
  if (print_time) {
    full_env_truncate_timer.PrintElapsed();
  }
  Timer loop_projection_post_procedure_timer("loop_projection_post_procedure");
  //split out the lambdas of envs
  for (size_t i = 0; i < 4; i++) {
    SplitOut2EnvLambdasInMPSOrderGammas(gammas[i], env_lambda_ls[i], env_lambda_rs[i]);
  }

  TransposeBackGammaTensorIndicesFromMPSOrder(gammas);
  // normalize Lambda and return the normalization factor
  double norm = 1.0;
  for (auto &lambda : lambdas) {
    norm *= lambda.Normalize();
  }
  for (auto &gamma : gammas) {
    norm *= gamma.Normalize();
  }
#ifndef NDEBUG
  auto phy_idx = Gamma({0, 0}).GetIndex(4);
#endif
  Gamma(upper_left_site) = gammas[0];
  Gamma({row, col + 1}) = gammas[1];
  Gamma({row + 1, col + 1}) = gammas[2];
  Gamma({row + 1, col}) = gammas[3];
  lambda_horiz({row, col + 1}) = lambdas[0];
  lambda_vert({row + 1, col + 1}) = lambdas[1];
  lambdas[2].Transpose({1, 0});
  lambdas[3].Transpose({1, 0});
  lambda_horiz({row + 1, col + 1}) = lambdas[2];
  lambda_vert({row + 1, col}) = lambdas[3];
#ifndef NDEBUG
  for (const auto &gamma : {Gamma(upper_left_site), Gamma({row, col + 1}),
                            Gamma({row + 1, col + 1}), Gamma({row + 1, col})}) {
    assert(phy_idx == gamma.GetIndex(4));
  }
#endif
  if (print_time) {
    loop_projection_post_procedure_timer.PrintElapsed();
  }
  return norm;
}

/** Each evolve  gate form a loop and includes 4 tensors, and
   *  represents the local imaginary evolve gate exp(-\tau * h).
   *
   *  The order of the 4 tensors in one loop is accord to Ref. [1] Fig. 2 (a)
   *
   *  And the orders of the legs in MPO tensors are
   *
   *        2                  2
   *        |                  |
   *        |                  |
   *  0---[gate 0]---3  0---[gate 1]---3
   *        |                  |
   *        |                  |
   *        1                  1
   *  so on so forth for gate 2 and 3. Note the leg 3 of gate 0 are connected to
   *  the leg 0 of gate 1, so on so forth.
   *  And the legs 1 of the gates are connected to PEPS physical legs.
   *  (The diagrams here are upside down from the Fig. in the Ref. [1].)
   *
   *
   *  Output: The Gamma and Lambda tensors are updated,
   *          while the legs order of Gamma/Lambda tensors are not changed.
   */
template<typename TenElemT, typename QNT>
void SquareLatticePEPS<TenElemT, QNT>::PatSquareLocalLoopProjector_(
    const qlpeps::SquareLatticePEPS<TenElemT, QNT>::LocalSquareLoopGateT &gate_tens,
    const qlpeps::SiteIdx &upper_left_site) {
  using DTenT = QLTensor<QLTEN_Double, QNT>;
  const size_t row = upper_left_site.row();
  const size_t col = upper_left_site.col();
#ifndef NDEBUG
  auto phy_idx = Gamma({0, 0}).GetIndex(4);
#endif
  std::vector<TenT *> lambdas(4, nullptr);
  lambdas[0] = lambda_horiz(row, col + 1);
  lambdas[1] = lambda_vert(row + 1, col + 1);
  lambdas[2] = lambda_horiz(row + 1, col + 1);
  lambdas[3] = lambda_vert(row + 1, col);
  lambdas[2]->Transpose({1, 0});
  lambdas[3]->Transpose({1, 0});

  //Update Gamma Tensors
  TenT &Gamma0 = Gamma(upper_left_site);
  TenT tmp[12], q, r, u, vt;
  DTenT s;
  Contract(&Gamma0, {4}, &gate_tens[0], {1}, tmp);
  tmp->Transpose({1, 4, 0, 3, 5, 2, 6});
  QR(tmp, 5, qn0_, &q, &r);
  Contract(&r, {1}, lambdas[0], {0}, tmp + 1);
  mock_qlten::SVD(tmp + 1, 1, qn0_, &u, &s, &vt);
  Contract(&q, {5}, &u, {0}, tmp + 2);  // tmp + 2 is gamma0
  *lambdas[0] = std::move(s);

  TenT &Gamma1 = Gamma({row, col + 1});
  Contract(&Gamma1, {4}, &gate_tens[1], {1}, tmp + 3);
  Contract(&vt, {1, 2}, tmp + 3, {4, 0}, tmp + 4);
  tmp[4].Transpose({0, 2, 3, 4, 1, 5});
  q = TenT();
  r = TenT();
  QR(tmp + 4, 4, qn0_, &q, &r);
  Contract(&r, {1}, lambdas[1], {0}, tmp + 5);
  u = TenT();
  vt = TenT();
  s = DTenT();
  mock_qlten::SVD(tmp + 5, 1, qn0_, &u, &s, &vt);
  Gamma1 = TenT();
  Contract(&q, {4}, &u, {0}, &Gamma1);
  Gamma1.Transpose({0, 4, 1, 2, 3});
  *lambdas[1] = std::move(s);

  TenT &Gamma2 = Gamma({row + 1, col + 1});
  Contract(&Gamma2, {4}, &gate_tens[2], {1}, tmp + 6);
  Contract(&vt, {1, 2}, tmp + 6, {4, 3}, tmp + 7);
  tmp[7].Transpose({0, 2, 3, 4, 1, 5});
  q = TenT();
  r = TenT();
  s = DTenT();
  QR(tmp + 7, 4, qn0_, &q, &r);
  Contract(&r, {1}, lambdas[2], {0}, tmp + 8);
  u = TenT();
  vt = TenT();
  mock_qlten::SVD(tmp + 8, 1, qn0_, &u, &s, &vt);
  Gamma2 = TenT();
  Contract(&q, {4}, &u, {0}, &Gamma2);
  Gamma2.Transpose({4, 1, 2, 0, 3});
  s.Transpose({1, 0});
  *lambdas[2] = std::move(s);

  TenT &Gamma3 = Gamma({row + 1, col});
  Contract(&Gamma3, {4}, &gate_tens[3], {1}, tmp + 9);
  Contract(&vt, {1, 2}, tmp + 9, {4, 2}, tmp + 10);
  tmp[10].Transpose({0, 1, 2, 4, 3, 5});
  q = TenT();
  r = TenT();
  QR(tmp + 10, 4, qn0_, &q, &r);
  Contract(&r, {1}, lambdas[3], {0}, tmp + 11);
  u = TenT();
  vt = TenT();
  s = DTenT();
  mock_qlten::SVD(tmp + 11, 1, qn0_, &u, &s, &vt);
  Gamma3 = TenT();
  Contract(&q, {4}, &u, {0}, &Gamma3);
  Gamma3.Transpose({1, 2, 0, 4, 3});
  s.Transpose({1, 0});
  *lambdas[3] = std::move(s);

  Gamma0 = TenT();
  Contract(&vt, {2, 1}, tmp + 2, {0, 1}, &Gamma0);
  Gamma0.Transpose({1, 0, 4, 2, 3});
#ifndef NDEBUG
  for (const auto &gamma : {Gamma(upper_left_site), Gamma({row, col + 1}),
                            Gamma({row + 1, col + 1}), Gamma({row + 1, col})}) {
    assert(phy_idx == gamma.GetIndex(4));
  }
#endif
}

/// < qusi-positive means some diagonal elements may be negative but with very small abosultly value
/// < induced from the numeric errors.
template<typename QNT>
QLTensor<QLTEN_Double, QNT> QuasiSquareRootDiagMat(
    const QLTensor<QLTEN_Double, QNT> &quasi_positive_mat,
    const double tolerance = 1e-15
) {
  QLTensor<QLTEN_Double, QNT> sqrt = quasi_positive_mat;
  for (size_t i = 0; i < sqrt.GetShape()[0]; i++) {
    double elem = sqrt({i, i});
    if (elem >= 0) {
      sqrt({i, i}) = std::sqrt(elem);
    } else {
      if (elem < -tolerance)
        std::cout << "warning: trying to find square root of " << std::scientific << elem << std::endl;
      sqrt({i, i}) = 0.0;
    }
  }
  return sqrt;
}

template<typename QNT>
void FixSignForDiagMat(
    QLTensor<QLTEN_Double, QNT> &diag_mat
) {
  double diag_sum = 0.0;
  for (size_t i = 0; i < diag_mat.GetShape()[0]; i++) {
    diag_sum += diag_mat({i, i});
  }
  if (diag_sum < 0) {
    diag_mat *= -1;
  }
}

template<typename TenElemT, typename QNT>
void WeightedTraceGaugeFixing(
    const ArnoldiParams &arnoldi_params,
    QLTensor<TenElemT, QNT> &Upsilon,
    QLTensor<QLTEN_Double, QNT> &sigma,
    QLTensor<TenElemT, QNT> &gamma_head,
    QLTensor<TenElemT, QNT> &gamma_tail
) {
  const auto qn0 = sigma.Div();
  using TenT = QLTensor<TenElemT, QNT>;
  using DTenT = QLTensor<QLTEN_Double, QNT>;

  DTenT sigma_dag = Dag(sigma);
  //calculate the left/right eigen vectors
  /**
   *
   * |-1-0-sigma^dag-1--1---------3
   * |                      |
   * L                   Upsilon
   * |                      |
   * |-0-0-sigma-1-----0----------2
   *
   * 1----------3----0-sigma^dag-1---1---
   *       |                            |
   *    Upsilon                         R
   *       |                            |
   * 0----------2----0-sigma-1-------0--|
   */
  const Index<QNT> index0 = InverseIndex(sigma.GetIndex(0));
  const Index<QNT> index1 = sigma.GetIndex(0);
  TenT left_vec0 = TenT({index0, index1});
  TenT right_vec0 = TenT({index1, index0});
  for (size_t j = 0; j < index0.dim(); j++) {
    left_vec0({j, j}) = 1.0;
    right_vec0({j, j}) = 1.0;
  }
  double upsilon_norm = Upsilon.Normalize();
  auto left_eigen_sys = ArnoldiSolver(Upsilon,
                                      sigma,
                                      sigma_dag,
                                      left_vec0,
                                      arnoldi_params,
                                      left_vec_multiple_transfer_tens<TenElemT, QNT>);
  auto righ_eigen_sys = ArnoldiSolver(Upsilon,
                                      sigma,
                                      sigma_dag,
                                      right_vec0,
                                      arnoldi_params,
                                      right_vec_multiple_transfer_tens<TenElemT, QNT>);
  //EVD for eigenvectors, and update the Upsilon_i, Gammas, and Lambdas
  TenT u_l, d_l, u_r, d_r;
  DTenT sqrt_dl, sqrt_dr, inv_sqrt_dl, inv_sqrt_dr;
  SymMatEVD(&left_eigen_sys.right_eig_vec, &u_l, &d_l);
  SymMatEVD(&righ_eigen_sys.right_eig_vec, &u_r, &d_r);
  FixSignForDiagMat(d_l);
  FixSignForDiagMat(d_r);
  d_l.Normalize();
  d_r.Normalize();
  sqrt_dl = QuasiSquareRootDiagMat(d_l);
  sqrt_dr = QuasiSquareRootDiagMat(d_r);

  inv_sqrt_dl = ElementWiseInv(sqrt_dl, 1e-200);
  inv_sqrt_dr = ElementWiseInv(sqrt_dr, 1e-200);
  TenT ul_dag = Dag(u_l);
  TenT ur_dag = Dag(u_r);


  //calculate sigma_prime
  DTenT sigma_prime;
  TenT tmp0, tmp1, tmp2;
  Contract(&sqrt_dl, {0}, &u_l, {1}, &tmp0);
  Contract(&tmp0, {1}, &sigma, {0}, &tmp1);
  Contract(&tmp1, {1}, &u_r, {0}, &tmp2);
  Contract(&tmp2, {1}, &sqrt_dr, {0}, &sigma_prime);

  sigma = DTenT();
  TenT v_l, v_r_dag;
  MatSVD(sigma_prime, qn0, v_l, sigma, v_r_dag);
  //Update Upsilon, and corresponding 2 gammas
  //The original data of lambdas and Gammas in PEPS are not changed.

  TenT x_inv;
  tmp0 = TenT();
  tmp1 = TenT();
  tmp2 = TenT();
  Contract(&ul_dag, {1}, &inv_sqrt_dl, {1}, &tmp0);
  Contract(&tmp0, {1}, &v_l, {0}, &x_inv);
  TenT x_inv_dag = Dag(x_inv);
  Contract(&Upsilon, {2}, &x_inv, {0}, &tmp1);
  Contract(&tmp1, {2}, &x_inv_dag, {0}, &tmp2);

  TenT y_inv, tmp3;
  tmp0 = TenT();
  tmp1 = TenT();
  Contract(&v_r_dag, {1}, &inv_sqrt_dr, {0}, &tmp0);
  Contract(&tmp0, {1}, &ur_dag, {1}, &y_inv);
  TenT y_inv_dag = Dag(y_inv);
  Contract(&y_inv, {1}, &tmp2, {0}, &tmp3);
  Upsilon = TenT();
  Contract(&y_inv_dag, {1}, &tmp3, {1}, &Upsilon);
  Upsilon.Transpose({1, 0, 2, 3});

  tmp0 = TenT();
  tmp1 = TenT();
  Contract(&gamma_tail, {4}, &x_inv, {0}, &tmp0);
  gamma_tail = std::move(tmp0);
  Contract(&y_inv, {1}, &gamma_head, {0}, &tmp1);
  gamma_head = std::move(tmp1);
}

/**  Fix weighted trace gauge for the loop tensors
 *
 *  Input: the gammas should contain the environment lambdas, and be in the MPS indices orders.
 *         The lambdas should be in the MPS indices orders.
 *
 */
template<typename TenElemT, typename QNT>
void WeightedTraceGaugeFixingInSquareLocalLoop(
    const ArnoldiParams &arnoldi_params,
    std::array<QLTensor<TenElemT, QNT>, 4> &gammas,  //input & output
    std::array<QLTensor<QLTEN_Double, QNT>, 4> &lambdas, //input & output
    std::array<QLTensor<TenElemT, QNT>, 4> &Upsilons //output
) {
  using TenT = QLTensor<TenElemT, QNT>;
  using DTenT = QLTensor<QLTEN_Double, QNT>;

  // Construct the Upsilon_i tensor
  std::array<QLTensor<TenElemT, QNT>, 4> gamma_gamma_dags;//transfer matrix, tmp data
  for (size_t i = 0; i < 4; i++) {
    TenT gamma_dag = Dag(gammas[i]);
    Contract(&gammas[i], {1, 2, 3}, &gamma_dag, {1, 2, 3}, &gamma_gamma_dags[i]); //O(D^6 d)
    gamma_gamma_dags[i].Transpose({0, 2, 1, 3});
  }

  for (size_t i = 0; i < 4; i++) {
    TenT tmp = gamma_gamma_dags[i];
    for (size_t j = 1; j < 4; j++) {
      const size_t eating_gamma_idx = (i + j) % 4;
      const size_t eating_lambda_idx = (i + j - 1) % 4;
      TenT &eating_gamma_gamma_dag = gamma_gamma_dags[eating_gamma_idx];
      TenT &eating_lambda = lambdas[eating_lambda_idx];
      TenT lambda_dag = Dag(eating_lambda);
      TenT tmp2, tmp3;
      Contract(&tmp, {2}, &eating_lambda, {0}, &tmp2);
      Contract(&tmp2, {2}, &lambda_dag, {0}, &tmp3);
      tmp = TenT();
      Contract(&tmp3, {2, 3}, &eating_gamma_gamma_dag, {0, 1}, &tmp);
    }
    Upsilons[i] = tmp;
  }

  for (size_t i = 0; i < 4; i++) {
    WeightedTraceGaugeFixing(arnoldi_params, Upsilons[i], lambdas[(i + 3) % 4], gammas[i], gammas[(i + 3) % 4]);
  }

#ifndef NDEBUG
  assert(gammas[0].GetIndex(4).GetDir() == OUT);
#endif
}

template<typename ElemT, typename QNT>
struct PtenVec {
  using TenT = QLTensor<ElemT, QNT>;
  PtenVec(const TenT &p_ten) : p_ten(p_ten) {}

  PtenVec operator=(const PtenVec &rhs) {
    p_ten = rhs.p_ten;
    return *this;
  }

  PtenVec operator+=(const PtenVec &rhs) {
    p_ten += rhs;
    return *this;
  }

  PtenVec operator-(void) const {
    return PtenVec(-p_ten);
  }

  PtenVec operator+(const PtenVec &rhs) const {
    return PtenVec(p_ten + rhs.p_ten);
  }

  PtenVec operator-(const PtenVec &rhs) const {
    return PtenVec(p_ten + (-rhs.p_ten));
  }

  PtenVec operator*(const ElemT &scalar) const {
    return PtenVec(p_ten * scalar);
  }

  ///< Inner product, return Dag(*this) * rhs
  ElemT operator*(const PtenVec &rhs) const {
    TenT scalar, p_ten_dag = Dag(this->p_ten);
    Contract(&p_ten_dag, {0, 1}, &rhs.p_ten, {0, 1}, &scalar);
    return scalar();
  }
  double NormSquare(void) const {
    double norm = p_ten.Get2Norm();
    return norm * norm;
  }
  TenT p_ten;
};

template<typename ElemT, typename QNT>
PtenVec<ElemT, QNT> operator*(const ElemT &scalar, const PtenVec<ElemT, QNT> &p_ten_vec) {
  return p_ten_vec * scalar;
}

template<typename QNT>
PtenVec<QLTEN_Complex, QNT> operator*(const double &scalar, const PtenVec<QLTEN_Complex, QNT> &p_ten_vec) {
  return p_ten_vec * QLTEN_Complex(scalar, 0.0);
}

template<typename ElemT, typename QNT>
struct BtenMat {
  using TenT = QLTensor<ElemT, QNT>;
  BtenMat(const TenT &b_ten) : b_ten(b_ten) {}
  PtenVec<ElemT, QNT> operator*(const PtenVec<ElemT, QNT> &p_ten_vec) const {
    TenT res_ten;
    Contract(&p_ten_vec.p_ten, {0, 1}, &(this->b_ten), {0, 1}, &res_ten);
    return PtenVec<ElemT, QNT>(res_ten);
  }
  TenT b_ten;
};

template<typename QNT>
double diag_mat_diff(const QLTensor<QLTEN_Double, QNT> &sigma1,
                     const QLTensor<QLTEN_Double, QNT> &sigma2) {
  if (sigma1.GetShape()[0] != sigma2.GetShape()[0]) {
    return std::numeric_limits<double>::infinity();
  }
  QLTensor<QLTEN_Double, QNT> diff_ten = sigma1 + (-sigma2);
  double sigma_norm = sigma1.Get2Norm();
  if (sigma_norm > 0.0) {
    return diff_ten.Get2Norm() / sigma_norm;
  } else {
    sigma_norm = sigma2.Get2Norm();
    if (sigma_norm > 0.0) {
      return diff_ten.Get2Norm() / sigma_norm;
    } else {
      return 0.0;
    }
  }

}
/** truncation, FET
 *
 *   0-----2           3
 *   |     |           |
 *   si    |           |
 *   g     |--Upsilon--|
 *   ma    |           |
 *   |     |           |
 *   1-----0           1
 *
 *  @return u, v^dag, and sigma
 */
template<typename TenElemT, typename QNT>
std::pair<QLTensor<TenElemT, QNT>, QLTensor<TenElemT, QNT>> FullEnvironmentTruncate(
    const QLTensor<TenElemT, QNT> &Upsilon,
    QLTensor<QLTEN_Double, QNT> &sigma, //input & output
    const FullEnvironmentTruncateParams &trunc_params
) {
  using TenT = QLTensor<TenElemT, QNT>;
  using DTenT = QLTensor<QLTEN_Double, QNT>;
  using PtenVecT = PtenVec<TenElemT, QNT>;
  using BtenMatT = BtenMat<TenElemT, QNT>;
  QNT qn0 = sigma.Div();
  TenT sigma_Upsilon_tmp;
  Contract(&Upsilon, {0, 2}, &sigma, {1, 0}, &sigma_Upsilon_tmp);
  DTenT sigma_orig = sigma;
  // then data of sigma can be thrown.
  sigma = DTenT();
  TenT u, vdag;
  //initialize u, vdag, and sigma_tilde
  double actual_trunc_err;
  size_t actual_D;
  MatSVD(sigma_orig, qn0, trunc_params.trunc_err, trunc_params.Dmin, trunc_params.Dmax,
         vdag, sigma, u, &actual_trunc_err, &actual_D);

  DTenT sigma_old = sigma;
  for (size_t iter = 0; iter < trunc_params.max_iter; iter++) {
    //fix u, solve sigma, vdag
    TenT u_dag = Dag(u);
    TenT P_ten, B_ten, R_ten_init, L_ten_init;
    Contract(&u_dag, {1}, &sigma_Upsilon_tmp, {0}, &P_ten);
    Contract(&vdag, {1}, &sigma, {0}, &R_ten_init);
    R_ten_init.Transpose({1, 0});
    TenT tmp[4];
    Contract(&u, {1}, &Upsilon, {0}, tmp);
    Contract(tmp, {1}, &u_dag, {1}, &B_ten);
    B_ten.Transpose({0, 1, 3, 2});
    // solve R by R*B = P, (conjugate gradient?)
    PtenVecT p_ten_vec(P_ten);
    PtenVecT r_init(R_ten_init);
    BtenMatT b_ten_mat(B_ten);
    size_t cg_iter;

    PtenVecT update_r_vec =
        ConjugateGradientSolver(b_ten_mat, p_ten_vec, r_init,
                                trunc_params.cg_params.max_iter,
                                trunc_params.cg_params.tolerance, cg_iter);

    Contract(&update_r_vec.p_ten, {0}, &u, {0}, tmp + 1);
    u = TenT();
    vdag = TenT();
    sigma = DTenT();
    // update v, and sigma according to u*R = u'*sigma*v
    MatSVD(tmp[1], qn0, trunc_params.trunc_err, trunc_params.Dmin, trunc_params.Dmax,
           vdag, sigma, u, &actual_trunc_err, &actual_D);
    //fix v, solve u, sigma
    TenT v = Dag(vdag);
    P_ten = TenT();
    B_ten = TenT();
    Contract(&sigma_Upsilon_tmp, {1}, &v, {0}, &P_ten);
    p_ten_vec = PtenVecT(P_ten);
    Contract(&Upsilon, {2}, &vdag, {0}, tmp + 2);
    Contract(tmp + 2, {2}, &v, {0}, &B_ten);
    B_ten.Transpose({0, 2, 1, 3});
    b_ten_mat = BtenMatT(B_ten);

    Contract(&sigma, {1}, &u, {0}, &L_ten_init);
    L_ten_init.Transpose({1, 0});
    PtenVecT l_init(L_ten_init);
    PtenVecT update_l_vec =
        ConjugateGradientSolver(b_ten_mat, p_ten_vec, l_init,
                                trunc_params.cg_params.max_iter,
                                trunc_params.cg_params.tolerance, cg_iter);
    Contract(&vdag, {1}, &update_l_vec.p_ten, {1}, tmp + 3);

    u = TenT();
    vdag = TenT();
    sigma = DTenT();
    MatSVD(tmp[3], qn0, trunc_params.trunc_err, trunc_params.Dmin, trunc_params.Dmax,
           vdag, sigma, u, &actual_trunc_err, &actual_D);
    if (diag_mat_diff(sigma, sigma_old) < trunc_params.tolerance) {
      break;
    } else {
      sigma_old = sigma;
    }
  }
  return std::make_pair(u, vdag);
}

template<typename TenElemT, typename QNT>
void SquareLatticePEPS<TenElemT, QNT>::
FullEnvironmentTruncateInSquareLocalLoop_(
    const FullEnvironmentTruncateParams &trunc_params,
    std::array<QLTensor<TenElemT, QNT>, 4> &gammas,  //input & output
    std::array<QLTensor<TenElemT, QNT>, 4> &lambdas, //input & output
    const std::array<QLTensor<TenElemT, QNT>, 4> &Upsilons //input & output
) const {
  for (size_t i = 0; i < 4; i++) {
    auto [u, vdag] = FullEnvironmentTruncate(Upsilons[i], lambdas[(i + 3) % 4], trunc_params);
    // truncate Gamma tensors accordingly
    TenT &gamma_tail = gammas[(i + 3) % 4];
    TenT &gamma_head = gammas[i];
    TenT tmp0, tmp1;
    Contract(&gamma_tail, {4}, &vdag, {0}, &tmp0);
    gamma_tail = std::move(tmp0);
    Contract(&u, {1}, &gamma_head, {0}, &tmp1);
    gamma_head = std::move(tmp1);
  }
}
}//qlpeps

#endif //VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_PROJECTION4_IMPL_H

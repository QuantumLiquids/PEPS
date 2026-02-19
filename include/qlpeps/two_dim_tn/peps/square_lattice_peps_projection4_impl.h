/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-07-23
*
* Description: QuantumLiquids/PEPS project. The square PEPS class, project 4-site projectors implementation.
*              Note: Current implementation only supports Open Boundary Condition (OBC).
* Reference: [1] PRB 102, 075147 (2020), "Loop update for iPEPS in 2D".
*            [2] arXiv: 1801.05390v2, Glen Evenbly
*/


#ifndef QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_PROJECTION4_IMPL_H
#define QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_PROJECTION4_IMPL_H

#include "arnoldi_solver.h"

namespace qlpeps {
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
    const typename qlten::RealTypeTrait<TenElemT>::type trunc_err, const size_t Dmin, const size_t Dmax,
    QLTensor<TenElemT, QNT> &u,
    QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &s,
    QLTensor<TenElemT, QNT> &vt,
    typename qlten::RealTypeTrait<TenElemT>::type *pactual_trunc_err, size_t *pD
) {
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
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
    QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &s,
    QLTensor<TenElemT, QNT> &vt
) {
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
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
                                   const QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &env_lambda_l,
                                   const QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &env_lambda_r) {
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
                                         const QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &env_lambda_l,
                                         const QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &env_lambda_r,
                                         const double inv_tol
) {
  QLTensor<TenElemT, QNT> tmp, res;
  auto inv_lambda_l = DiagMatInv(env_lambda_l, inv_tol);
  auto inv_lambda_r = DiagMatInv(env_lambda_r, inv_tol);
  Contract(&gamma, {2}, &inv_lambda_l, {1}, &tmp);
  Contract(&tmp, {2}, &inv_lambda_r, {1}, &res);
  res.Transpose({0, 1, 3, 4, 2});
  gamma = res;
}

//forward declaration
template<typename TenElemT, typename QNT>
void WeightedTraceGaugeFixingInSquareLocalLoop(
    const ArnoldiParams &arnoldi_params,
    const typename qlten::RealTypeTrait<TenElemT>::type,
    std::array<QLTensor<TenElemT, QNT>, 4> &gammas,  //input & output
    std::array<QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT>, 4> &lambdas, //input & output
    std::array<QLTensor<TenElemT, QNT>, 4> &Upsilons, //output
    const double hermiticity_correction_tol,
    const double hermiticity_warning_tol,
    const size_t power_method_max_iter,
    const double power_method_tolerance,
    const size_t power_method_burn_in
);

template<typename TenElemT, typename QNT>
std::array<QLTensor<TenElemT, QNT>, 4>
SquareLatticePEPS<TenElemT, QNT>::GetLoopGammas_(const qlpeps::SiteIdx upper_left_site) const {
  const size_t row = upper_left_site.row();
  const size_t col = upper_left_site.col();
  std::array<QLTensor<TenElemT, QNT>, 4> gammas;
  gammas[0] = Gamma(upper_left_site);
  gammas[1] = Gamma({row, col + 1});
  gammas[2] = Gamma({row + 1, col + 1});
  gammas[3] = Gamma({row + 1, col});
  return gammas;
}

template<typename TenElemT, typename QNT>
std::array<QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT>, 4>
SquareLatticePEPS<TenElemT, QNT>::GetLoopInternalLambdas_(const qlpeps::SiteIdx upper_left_site) const {
  const size_t row = upper_left_site.row();
  const size_t col = upper_left_site.col();
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  std::array<QLTensor<RealT, QNT>, 4> lambdas;
  lambdas[0] = lambda_horiz({row, col + 1});
  lambdas[1] = lambda_vert({row + 1, col + 1});
  lambdas[2] = lambda_horiz({row + 1, col + 1});
  lambdas[3] = lambda_vert({row + 1, col});
  lambdas[2].Transpose({1, 0});
  lambdas[3].Transpose({1, 0});
  return lambdas;
}

template<typename TenElemT, typename QNT>
std::pair<std::array<QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT>, 4>, std::array<QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT>, 4>>
SquareLatticePEPS<TenElemT, QNT>::GetLoopEnvLambdas_(const qlpeps::SiteIdx upper_left_site) const {
  const size_t row = upper_left_site.row();
  const size_t col = upper_left_site.col();
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  std::array<QLTensor<RealT, QNT>, 4> env_lambda_ls, env_lambda_rs;
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
  return {env_lambda_ls, env_lambda_rs};
}

template<typename TenElemT, typename QNT>
TenElemT LoopTrace(
    const QLTensor<TenElemT, QNT> &Upsilon,
    const QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &lambda
) {
  QLTensor<TenElemT, QNT> temp0, scale_ten;
  auto lambda_dag = Dag(lambda);
  Contract(&Upsilon, {2, 0}, &lambda, {0, 1}, &temp0);
  Contract(&temp0, {1, 0}, &lambda_dag, {0, 1}, &scale_ten);
  return scale_ten();
}

/**
 * MPS order :
 *     2    3(PEPS virtual leg but part of physical leg in MPS context)
 *      \  /
 *  0---gamma---4(one of PEPS virtual leg, also virtual leg in MPS context)
 *        |
 *        1 (PEPS physical leg, physical leg also in MPS context)
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> CalTransferMatOfGamma(
    const QLTensor<TenElemT, QNT> &gamma
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT gamma_dag = Dag(gamma);
  TenT gamma_gamma_dag;
  if constexpr (!TenT::IsFermionic()) {
    Contract(&gamma, {1, 2, 3}, &gamma_dag, {1, 2, 3}, &gamma_gamma_dag); //O(D^6 d)
  } else {
    auto id2 = Eye<TenElemT>(gamma.GetIndex(2));
    auto id3 = Eye<TenElemT>(gamma.GetIndex(3));
    TenT temp1, temp2;
    Contract(&gamma, {2}, &id2, {1}, &temp1);
    Contract(&temp1, {2}, &id3, {1}, &temp2);
    Contract(&temp2, {1, 3, 4}, &gamma_dag, {1, 2, 3}, &gamma_gamma_dag);
  }
  gamma_gamma_dag.Transpose({0, 2, 1, 3});
  return gamma_gamma_dag;
}

template<typename TenElemT, typename QNT>
std::array<QLTensor<TenElemT, QNT>, 4> ConstructUpsilons(
    const std::array<QLTensor<TenElemT, QNT>, 4> &gammas,
    const std::array<QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT>, 4> &lambdas
) {
  using TenT = QLTensor<TenElemT, QNT>;
  std::array<QLTensor<TenElemT, QNT>, 4> gamma_gamma_dags, Upsilons;
  for (size_t i = 0; i < 4; i++) {
    gamma_gamma_dags[i] = CalTransferMatOfGamma(gammas[i]);
  }

  for (size_t i = 0; i < 4; i++) {
    TenT tmp = gamma_gamma_dags[i];
    for (size_t j = 1; j < 4; j++) {
      const size_t eating_gamma_idx = (i + j) % 4;
      const size_t eating_lambda_idx = (i + j - 1) % 4;
      TenT &eating_gamma_gamma_dag = gamma_gamma_dags[eating_gamma_idx];
      auto &eating_lambda = lambdas[eating_lambda_idx];
      auto lambda_dag = Dag(eating_lambda);
      TenT tmp2, tmp3;
      Contract(&tmp, {2}, &eating_lambda, {0}, &tmp2);
      Contract(&tmp2, {2}, &lambda_dag, {0}, &tmp3);
      tmp = TenT();
      Contract(&tmp3, {2, 3}, &eating_gamma_gamma_dag, {0, 1}, &tmp);
    }
    Upsilons[i] = tmp;
  }
  return Upsilons;
}

// <1|0>
template<typename TenElemT, typename QNT>
TenElemT GammaLambdaMPSLoopOverlap(
    const std::array<QLTensor<TenElemT, QNT>, 4> &gammas0,
    const std::array<QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT>, 4> &lambdas0,
    const std::array<QLTensor<TenElemT, QNT>, 4> &gammas1,
    const std::array<QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT>, 4> &lambdas1
) {
  std::array<QLTensor<TenElemT, QNT>, 4> eaten_gammas0, eaten_gammas1;
  for (size_t i = 0; i < 4; i++) {
    Contract(&gammas0[i], {4}, &lambdas0[i], {0}, &eaten_gammas0[i]);
    Contract(&gammas1[i], {4}, &lambdas1[i], {0}, &eaten_gammas1[i]);
  }
  std::array<QLTensor<TenElemT, QNT>, 4> gamma_gamma_dags;
  using TenT = QLTensor<TenElemT, QNT>;
  for (size_t i = 0; i < 4; i++) {
    eaten_gammas1[i].Dag();
    if constexpr (!TenT::IsFermionic()) {
      Contract(&eaten_gammas0[i], {1, 2, 3}, &eaten_gammas1[i], {1, 2, 3}, &gamma_gamma_dags[i]); //O(D^6 d)
    } else {
      TenT id2 = Eye<TenElemT>(eaten_gammas0[i].GetIndex(2));
      TenT id3 = Eye<TenElemT>(eaten_gammas0[i].GetIndex(3));

      TenT temp1, temp2;
      Contract(&eaten_gammas0[i], {2}, &id2, {1}, &temp1);
      Contract(&temp1, {2}, &id3, {1}, &temp2);
      Contract(&temp2, {1, 3, 4}, &eaten_gammas1[i], {1, 2, 3}, &gamma_gamma_dags[i]);
    }
    gamma_gamma_dags[i].Transpose({0, 2, 1, 3});
  }
  TenT temp[3];
  Contract(&gamma_gamma_dags[0], {2, 3}, &gamma_gamma_dags[1], {0, 1}, temp);
  Contract(temp, {2, 3}, &gamma_gamma_dags[2], {0, 1}, temp + 1);
  Contract(temp + 1, {2, 3, 0, 1}, &gamma_gamma_dags[3], {0, 1, 2, 3}, temp + 2);
  TenElemT overlap = temp[2]();
  return overlap;
}

template<typename TenElemT, typename QNT>
std::pair<typename qlten::RealTypeTrait<TenElemT>::type, typename qlten::RealTypeTrait<TenElemT>::type> SquareLatticePEPS<TenElemT, QNT>::LocalSquareLoopProject(
    const qlpeps::SquareLatticePEPS<TenElemT, QNT>::LocalSquareLoopGateT &gate_tens,
    const qlpeps::SiteIdx &upper_left_site,
    const qlpeps::LoopUpdateTruncatePara &params,
    const bool print_time) {
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  const size_t row = upper_left_site.row();
  const size_t col = upper_left_site.col();
  auto gammas_original = GetLoopGammas_(upper_left_site);
  auto lambdas_original = GetLoopInternalLambdas_(upper_left_site);
  TransposeGammaTensorIndicesIntoMPSOrder(gammas_original);
  auto [env_lambda_ls, env_lambda_rs] = GetLoopEnvLambdas_(upper_left_site);
  for (size_t i = 0; i < 4; i++) {
    Eat2EnvLambdasInMPSOrderGamma(gammas_original[i], env_lambda_ls[i], env_lambda_rs[i]);
  }
  auto Upsilons_original = ConstructUpsilons(gammas_original, lambdas_original);
  auto wave_function_norm = LoopTrace(Upsilons_original[0], lambdas_original[3]);

  Timer pat_loop_projector_timer("pat_loop_projector");
  PatSquareLocalLoopProjector_(gate_tens, upper_left_site);
  if (print_time) {
    pat_loop_projector_timer.PrintElapsed();
  }
  Timer loop_projection_pre_procedure_timer("loop_projection_pre_procedure");
  std::array<QLTensor<TenElemT, QNT>, 4> Upsilons;
  auto gammas = GetLoopGammas_(upper_left_site);
  auto lambdas = GetLoopInternalLambdas_(upper_left_site);

  TransposeGammaTensorIndicesIntoMPSOrder(gammas);
#ifndef NDEBUG
  //check the indices
  for (size_t i = 0; i < 4; i++) {
    auto idx_a = gammas[i].GetIndex(4);
    auto idx_b = lambdas[i].GetIndex(0);
    assert(idx_a == InverseIndex(idx_b));
  }
#endif

  for (size_t i = 0; i < 4; i++) {
    Eat2EnvLambdasInMPSOrderGamma(gammas[i], env_lambda_ls[i], env_lambda_rs[i]);
  }
  auto overlap = GammaLambdaMPSLoopOverlap(gammas, lambdas,
                                           gammas_original, lambdas_original);
  if (print_time) {
    loop_projection_pre_procedure_timer.PrintElapsed();
  }
  Timer weighted_trace_gauge_fixing_timer("weighted_trace_gauge_fixing");
  WeightedTraceGaugeFixingInSquareLocalLoop(params.arnoldi_params, params.inv_tol, gammas, lambdas, Upsilons,
                                             params.hermiticity_correction_tol, params.hermiticity_warning_tol,
                                             params.power_method_max_iter, params.power_method_tolerance,
                                             params.power_method_burn_in);
  if (print_time) {
    weighted_trace_gauge_fixing_timer.PrintElapsed();
  }
  Timer full_env_truncate_timer("full_env_truncate");
  FullEnvironmentTruncateInSquareLocalLoop(params.fet_params, gammas, lambdas, Upsilons);
  if (print_time) {
    full_env_truncate_timer.PrintElapsed();
  }
  Timer loop_projection_post_procedure_timer("loop_projection_post_procedure");
  //split out the lambdas of envs
  for (size_t i = 0; i < 4; i++) {
    SplitOut2EnvLambdasInMPSOrderGammas(gammas[i], env_lambda_ls[i], env_lambda_rs[i], params.env_lambda_inv_tol);
  }

  TransposeBackGammaTensorIndicesFromMPSOrder(gammas);
  // normalize Lambda and return the normalization factor
  RealT norm = 1.0;
  for (auto &lambda : lambdas) {
    norm *= lambda.QuasiNormalize();
  }
  for (auto &gamma : gammas) {
    norm *= gamma.QuasiNormalize();
  }

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
  auto phy_idx = Gamma({0, 0}).GetIndex(4);
  for (const auto &gamma : {Gamma(upper_left_site), Gamma({row, col + 1}),
                            Gamma({row + 1, col + 1}), Gamma({row + 1, col})}) {
    assert(phy_idx == gamma.GetIndex(4));
  }
#endif
  if (print_time) {
    loop_projection_post_procedure_timer.PrintElapsed();
  }
  return {norm, std::real(overlap / wave_function_norm)};
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
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  using DTenT = QLTensor<RealT, QNT>;
  const size_t row = upper_left_site.row();
  const size_t col = upper_left_site.col();
#ifndef NDEBUG
  auto phy_idx = Gamma({0, 0}).GetIndex(4);
#endif
  std::vector<DTenT *> lambdas(4, nullptr);
  lambdas[0] = lambda_horiz(row, col + 1);
  lambdas[1] = lambda_vert(row + 1, col + 1);
  lambdas[2] = lambda_horiz(row + 1, col + 1);
  lambdas[3] = lambda_vert(row + 1, col);
  lambdas[2]->Transpose({1, 0});
#ifndef NDEBUG // for omp parallel + HTTP transpose make catastrophic
  assert((*lambdas[2])({0, 0}) != 0.0);
#endif
  lambdas[3]->Transpose({1, 0});


  //Update Gamma Tensors
  TenT & Gamma0 = Gamma(upper_left_site);
  TenT tmp[12], q[4], r[4], u[4], vt[4];
  DTenT s[4];
  Contract(&Gamma0, {4}, &gate_tens[0], {1}, tmp);
  tmp->Transpose({1, 4, 0, 3, 5, 2, 6});
  QR(tmp, 5, qn0_, q, r);
  Contract(r, {1}, lambdas[0], {0}, tmp + 1);
  qlmps::mock_qlten::SVD(tmp + 1, 1, qn0_, u, s, vt);
  Contract(q, {5}, u, {0}, tmp + 2);  // tmp + 2 is gamma0
  *lambdas[0] = std::move(s[0]);

  TenT & Gamma1 = Gamma({row, col + 1});
  Contract(&Gamma1, {4}, &gate_tens[1], {1}, tmp + 3);
  Contract(vt, {1, 2}, tmp + 3, {4, 0}, tmp + 4);
  tmp[4].Transpose({0, 2, 3, 4, 1, 5});
  QR(tmp + 4, 4, qn0_, q + 1, r + 1);
  Contract(r + 1, {1}, lambdas[1], {0}, tmp + 5);
  qlmps::mock_qlten::SVD(tmp + 5, 1, qn0_, u + 1, s + 1, vt + 1);
  Gamma1 = TenT();
  Contract(q + 1, {4}, u + 1, {0}, &Gamma1);
  Gamma1.Transpose({0, 4, 1, 2, 3});
  *lambdas[1] = std::move(s[1]);

  TenT & Gamma2 = Gamma({row + 1, col + 1});
  Contract(&Gamma2, {4}, &gate_tens[2], {1}, tmp + 6);
  Contract(vt + 1, {1, 2}, tmp + 6, {4, 3}, tmp + 7);
  tmp[7].Transpose({0, 2, 3, 4, 1, 5});
  QR(tmp + 7, 4, qn0_, q + 2, r + 2);
  Contract(r + 2, {1}, lambdas[2], {0}, tmp + 8);
  qlmps::mock_qlten::SVD(tmp + 8, 1, qn0_, u + 2, s + 2, vt + 2);
  Gamma2 = TenT();
  Contract(q + 2, {4}, u + 2, {0}, &Gamma2);
  Gamma2.Transpose({4, 1, 2, 0, 3});
  s[2].Transpose({1, 0});
  *lambdas[2] = std::move(s[2]);

  TenT & Gamma3 = Gamma({row + 1, col});
  Contract(&Gamma3, {4}, &gate_tens[3], {1}, tmp + 9);
  Contract(vt + 2, {1, 2}, tmp + 9, {4, 2}, tmp + 10);
  tmp[10].Transpose({0, 1, 2, 4, 3, 5});
  QR(tmp + 10, 4, qn0_, q + 3, r + 3);
  Contract(r + 3, {1}, lambdas[3], {0}, tmp + 11);
  qlmps::mock_qlten::SVD(tmp + 11, 1, qn0_, u + 3, s + 3, vt + 3);
  Gamma3 = TenT();
  Contract(q + 3, {4}, u + 3, {0}, &Gamma3);
  Gamma3.Transpose({1, 2, 0, 4, 3});
  s[3].Transpose({1, 0});
  *lambdas[3] = std::move(s[3]);

  Gamma0 = TenT();
  Contract(vt + 3, {2, 1}, tmp + 2, {0, 1}, &Gamma0);
  Gamma0.Transpose({1, 0, 4, 2, 3});
#ifndef NDEBUG
  for (const auto &gamma : {Gamma(upper_left_site), Gamma({row, col + 1}),
                            Gamma({row + 1, col + 1}), Gamma({row + 1, col})}) {
    assert(phy_idx == gamma.GetIndex(4));
  }
#endif
}

/// < quasi-positive means some diagonal elements may be negative but with very small absolutely value
/// < induced from the numeric errors.
template<typename RealT, typename QNT>
QLTensor<RealT, QNT> QuasiSquareRootDiagMat(
    const QLTensor<RealT, QNT> &quasi_positive_mat,
    const RealT tolerance = static_cast<RealT>(1e-15)
) {
  if constexpr (QLTensor<RealT, QNT>::IsFermionic()) {
    assert(quasi_positive_mat.GetIndex(0).GetDir() == IN);
  }
  QLTensor<RealT, QNT> sqrt = quasi_positive_mat;
  for (size_t i = 0; i < sqrt.GetShape()[0]; i++) {
    RealT elem = sqrt({i, i});
    if (elem >= static_cast<RealT>(0)) {
      sqrt({i, i}) = static_cast<RealT>(std::sqrt(static_cast<double>(elem)));
    } else {
      if (elem < -tolerance)
        std::cout << "warning: trying to find square root of " << std::scientific << elem << std::endl;
      sqrt({i, i}) = static_cast<RealT>(0.0);
    }
  }
  return sqrt;
}

template<typename RealT, typename QNT>
void FixSignForDiagMat(
    QLTensor<RealT, QNT> &diag_mat
) {
  RealT diag_sum = 0.0;
  if constexpr (QLTensor<RealT, QNT>::IsFermionic()) {
    auto index = diag_mat.GetIndex(0);
    for (size_t i = 0; i < diag_mat.GetShape()[0]; i++) {
      if (index.GetQNSctFromActualCoor(i).IsFermionParityEven()) {
        diag_sum += diag_mat({i, i});
      }
    }
  } else {
    for (size_t i = 0; i < diag_mat.GetShape()[0]; i++) {
      diag_sum += diag_mat({i, i});
    }
  }
  if (diag_sum < static_cast<RealT>(0)) {
    diag_mat *= static_cast<RealT>(-1);
  }
}

template<typename TenElemT, typename QNT>
double EvaluateHermiticity(
    const QLTensor<TenElemT, QNT> &mat,
    const std::vector<size_t> trans_axes = {1, 0}
) {
  auto mat_dag = Dag(mat);
  mat_dag.Transpose(trans_axes);
  auto diff_ten = mat_dag + (-mat);
  double diff = diff_ten.GetQuasi2Norm() / mat.GetQuasi2Norm();
  return diff;
}

template<typename TenElemT, typename QNT>
bool CheckHermiticity(
    const QLTensor<TenElemT, QNT> &mat,
    const double tolerance,
    const std::vector<size_t> trans_axes = {1, 0}
) {
  auto mat_dag = Dag(mat);
  mat_dag.Transpose(trans_axes);
  auto diff_ten = mat_dag + (-mat);
  double diff = diff_ten.GetQuasi2Norm() / mat.GetQuasi2Norm();
//  assert((diff < tolerance));
  return (diff < tolerance);
}

template<typename TenElemT, typename QNT>
void SymmetrizeMat(
    QLTensor<TenElemT, QNT> &mat,
    const std::vector<size_t> trans_axes = {1, 0}
) {
  auto mat_dag = Dag(mat);
  mat_dag.Transpose(trans_axes);
  mat = (mat_dag + mat) * 0.5;
}

template<typename TenElemT, typename QNT>
ArnoldiRes<TenElemT, QNT> PowerMethod(
    const QLTensor<TenElemT, QNT> &Upsilon,
    const QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &sigma,
    const QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &sigma_dag,
    const QLTensor<TenElemT, QNT> &vec0,
    TransfTenMultiVec<TenElemT, QNT> transfer_tens_multiple_vec,
    const size_t max_iter,
    const double tolerance,
    const size_t burn_in
) {
  using TenT = QLTensor<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  const RealT iter_tol = static_cast<RealT>(tolerance);
  RealT eigen_value_last = 0;
  size_t iter;
  TenT vec = vec0;
  vec.QuasiNormalize();
  TenT vec_last_dag = Dag(vec);
  std::vector<TenT> fermion_parity_ops(2);
  if constexpr (QLTensor<TenElemT, QNT>::IsFermionic()) {
    vec_last_dag.ActFermionPOps();
  }
  for (iter = 0; iter < max_iter; iter++) {
    vec = transfer_tens_multiple_vec(vec, sigma, sigma_dag, Upsilon);
    RealT eigen_value = vec.QuasiNormalize();
    if (iter > burn_in && std::abs((eigen_value - eigen_value_last) / eigen_value) < iter_tol) {
      QLTensor<TenElemT, QNT> overlap_ten;
      Contract(&vec_last_dag, {0, 1}, &vec, {0, 1}, &overlap_ten);
      if (std::abs(std::abs(TenElemT(overlap_ten())) - 1.0) < iter_tol)
        return {eigen_value, vec};
    }
    eigen_value_last = eigen_value;
    vec_last_dag = Dag(vec);
    if constexpr (QLTensor<TenElemT, QNT>::IsFermionic()) {
      vec_last_dag.ActFermionPOps();
    }
  }
  std::cout << "dominant eigenvector solver in power method doesn't converge" << std::endl;
  return {eigen_value_last, vec};
}

/**
 * Gauge fixing in weighted trace gauge
 *
 * The order of the tensor indices and connection way :
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
 *
 */
template<typename TenElemT, typename QNT>
void WeightedTraceGaugeFixing(
    const ArnoldiParams &arnoldi_params,
    const typename qlten::RealTypeTrait<TenElemT>::type inv_tol,
    QLTensor<TenElemT, QNT> &Upsilon,
    QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &sigma,
    QLTensor<TenElemT, QNT> &gamma_head,
    QLTensor<TenElemT, QNT> &gamma_tail,
    const double hermiticity_correction_tol,
    const double hermiticity_warning_tol,
    const size_t power_method_max_iter,
    const double power_method_tolerance,
    const size_t power_method_burn_in
) {
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  RealT diff;
#ifndef NDEBUG
  diff = EvaluateHermiticity(Upsilon, {1, 0, 3, 2});
  assert(diff < 1e-12);
#endif

  const auto qn0 = sigma.Div();
  using TenT = QLTensor<TenElemT, QNT>;
  using DTenT = QLTensor<RealT, QNT>;

  DTenT sigma_dag = Dag(sigma);
  //calculate the left/right eigen vectors
  const Index<QNT> index0 = InverseIndex(sigma.GetIndex(0));
  const Index<QNT> index1 = sigma.GetIndex(0);
  TenT left_vec0 = TenT({index0, index1});
  TenT right_vec0 = TenT({index1, index0});
  for (size_t j = 0; j < index0.dim(); j++) {
    left_vec0({j, j}) = 1.0;
    right_vec0({j, j}) = 1.0;
  }
//  double upsilon_norm = Upsilon.Normalize();
// Normalizing Upsilon induce incorrect result. I don't know why.
  ArnoldiRes<TenElemT, QNT> left_eigen_sys = ArnoldiSolver(Upsilon,
                                                           sigma,
                                                           sigma_dag,
                                                           left_vec0,
                                                           arnoldi_params,
                                                           TransfTenMultiVec<TenElemT, QNT>(
                                                               left_vec_multiple_transfer_tens<TenElemT, QNT>));

  SymmetrizeMat(left_eigen_sys.eig_vec);
  left_eigen_sys = PowerMethod(Upsilon,
                               sigma,
                               sigma_dag,
                               left_eigen_sys.eig_vec,
                               TransfTenMultiVec<TenElemT, QNT>(left_vec_multiple_transfer_tens<TenElemT, QNT>),
                               power_method_max_iter,
                               power_method_tolerance,
                               power_method_burn_in);

  ArnoldiRes<TenElemT, QNT> right_eigen_sys = ArnoldiSolver(Upsilon,
                                                            sigma,
                                                            sigma_dag,
                                                            right_vec0,
                                                            arnoldi_params,
                                                            TransfTenMultiVec<TenElemT, QNT>(
                                                                right_vec_multiple_transfer_tens<TenElemT,
                                                                                                 QNT>));

  SymmetrizeMat(right_eigen_sys.eig_vec);
  right_eigen_sys = PowerMethod(Upsilon,
                                sigma,
                                sigma_dag,
                                right_eigen_sys.eig_vec,
                                TransfTenMultiVec<TenElemT, QNT>(right_vec_multiple_transfer_tens<TenElemT, QNT>),
                                power_method_max_iter,
                                power_method_tolerance,
                                power_method_burn_in);

  //EVD for eigenvectors, and update the Upsilon_i, Gammas, and Lambdas
  TenT u_l, u_r;
  DTenT d_l, d_r, sqrt_dl, sqrt_dr, inv_sqrt_dl, inv_sqrt_dr;
  SymMatEVD(&left_eigen_sys.eig_vec, &u_l, &d_l);
  SymMatEVD(&right_eigen_sys.eig_vec, &u_r, &d_r);
  FixSignForDiagMat(d_l);
  FixSignForDiagMat(d_r);
  d_l.QuasiNormalize();
  d_r.QuasiNormalize();
  sqrt_dl = QuasiSquareRootDiagMat(d_l);
  sqrt_dr = QuasiSquareRootDiagMat(d_r);

  inv_sqrt_dl = DiagMatInv(sqrt_dl, inv_tol);
  inv_sqrt_dr = DiagMatInv(sqrt_dr, inv_tol);
  TenT ul_dag = Dag(u_l);
  TenT ur_dag = Dag(u_r);

  // Convert real diagonal tensors to working element type for typed Contract
  TenT sqrt_dl_elem, inv_sqrt_dl_elem;
  if constexpr (std::is_same_v<TenElemT, RealT>) {
    sqrt_dl_elem = sqrt_dl;
    inv_sqrt_dl_elem = inv_sqrt_dl;
  } else {
    sqrt_dl_elem = ToComplex(sqrt_dl);
    inv_sqrt_dl_elem = ToComplex(inv_sqrt_dl);
  }

  //calculate sigma_prime
  TenT temp[11];
  Contract<TenElemT, QNT, false, false>(sqrt_dl_elem, u_l, 0, 1, 1, temp[0]);
  Contract(temp, {1}, &sigma, {0}, temp + 1);
  Contract(temp + 1, {1}, &u_r, {0}, temp + 2);
  Contract(temp + 2, {1}, &sqrt_dr, {0}, temp + 3);

  sigma = DTenT();
  TenT v_l, v_r_dag;
  MatSVD(temp[3], qn0, v_l, sigma, v_r_dag);
  //Update Upsilon, and corresponding 2 gammas
  //The original data of lambdas and Gammas in PEPS are not changed.

  TenT x_inv;
  Contract<TenElemT, QNT, true, false>(ul_dag, inv_sqrt_dl_elem, 1, 1, 1, temp[4]);
  Contract(temp + 4, {1}, &v_l, {0}, &x_inv);
  TenT x_inv_dag = Dag(x_inv);
  Contract(&Upsilon, {3}, &x_inv_dag, {0}, temp + 5);
  Contract<TenElemT, QNT, false, true>(temp[5], x_inv, 2, 0, 1, temp[6]);
  diff = EvaluateHermiticity(temp[6], {3, 2, 1, 0});
  if (diff < hermiticity_correction_tol) {
    SymmetrizeMat(temp[6], {3, 2, 1, 0});
  } else {
    std::cout << "temp[6] is extremely asymmetric. diff = " << diff << "consider truncate d_R more." << std::endl;
    exit(-1);
  }
  TenT y_inv;
  Contract(&v_r_dag, {1}, &inv_sqrt_dr, {0}, temp + 7);
  Contract(temp + 7, {1}, &ur_dag, {1}, &y_inv);
  TenT y_inv_dag = Dag(y_inv);
  Upsilon = TenT();
  Contract(&y_inv, {}, &y_inv_dag, {}, temp + 8);
  diff = EvaluateHermiticity(temp[8], {2, 3, 0, 1});
  assert(diff < 1e-6);
  Contract(temp + 8, {1, 3}, temp + 6, {1, 2}, &Upsilon);
//  Contract<TenElemT, QNT, true, false>(y_inv_dag, temp[6], 1, 2, 1, temp[8]);
//  Contract<TenElemT, QNT, true, false>(y_inv, temp[8], 1, 3, 1, Upsilon);
//  Contract(&y_inv_dag, {1}, temp + 6, {2}, temp + 8);
//  Contract(&y_inv, {1}, temp + 8, {2}, &Upsilon);
  Upsilon.Transpose({0, 1, 3, 2});
  diff = EvaluateHermiticity(Upsilon, {1, 0, 3, 2});
  if (diff > hermiticity_warning_tol) {
    if (diff < hermiticity_correction_tol) {
      std::cout << "Brute-force symmetrize the Upsilon tensor." << std::endl;
      SymmetrizeMat(Upsilon, {1, 0, 3, 2});
    } else {
      std::cout << "Upsilon is extremely asymmetric. diff = " << diff << "consider truncate d_R more." << std::endl;
      exit(-1);
    }
  }
  Contract(&gamma_tail, {4}, &x_inv, {0}, temp + 9);
  gamma_tail = std::move(temp[9]);
  Contract(&y_inv, {1}, &gamma_head, {0}, temp + 10);
  gamma_head = std::move(temp[10]);
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
    const typename qlten::RealTypeTrait<TenElemT>::type inv_tol,
    std::array<QLTensor<TenElemT, QNT>, 4> &gammas,  //input & output
    std::array<QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT>, 4> &lambdas, //input & output
    std::array<QLTensor<TenElemT, QNT>, 4> &Upsilons, //output
    const double hermiticity_correction_tol,
    const double hermiticity_warning_tol,
    const size_t power_method_max_iter,
    const double power_method_tolerance,
    const size_t power_method_burn_in
) {
  // Construct the Upsilon_i tensor
  Upsilons = ConstructUpsilons(gammas, lambdas);
  for (size_t i = 0; i < 4; i++) {
    WeightedTraceGaugeFixing(arnoldi_params,
                             inv_tol, Upsilons[i], lambdas[(i + 3) % 4], gammas[i], gammas[(i + 3) % 4],
                             hermiticity_correction_tol, hermiticity_warning_tol,
                             power_method_max_iter, power_method_tolerance, power_method_burn_in);
  }
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
    if constexpr (TenT::IsFermionic()) {
      p_ten_dag.ActFermionPOps();
#ifndef NDEBUG
      std::vector<TenT> fermion_parity_ops(2);
      for (size_t i = 0; i < 2; i++) {
        Index<QNT> idx = p_ten.GetIndex(i);
        fermion_parity_ops[i] = Eye<ElemT, QNT>(InverseIndex(idx));
      }
      TenT temp1, temp2;
      Contract(&fermion_parity_ops[1], {1}, &p_ten_dag, {1}, &temp1);
      Contract(&fermion_parity_ops[0], {1}, &temp1, {1}, &temp2);

      TenT diff = p_ten_dag + (-temp2);
      assert(diff.GetQuasi2Norm() < 1e-12);
#endif
    }
    Contract(&p_ten_dag, {0, 1}, &rhs.p_ten, {0, 1}, &scalar);
    return scalar();
  }
  double NormSquare(void) const {
    double norm = p_ten.GetQuasi2Norm();
    assert(std::abs(norm * norm - (*this) * (*this)) < 1e-14);
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

template<typename RealT, typename QNT>
double diag_mat_diff(const QLTensor<RealT, QNT> &sigma1,
                     const QLTensor<RealT, QNT> &sigma2) {
  if (sigma1.GetShape()[0] != sigma2.GetShape()[0]) {
    return std::numeric_limits<double>::infinity();
  }
  QLTensor<RealT, QNT> diff_ten = sigma1 + (-sigma2);
  double sigma_norm = sigma1.GetQuasi2Norm();
  if (sigma_norm > 0.0) {
    return diff_ten.GetQuasi2Norm() / sigma_norm;
  } else {
    sigma_norm = sigma2.GetQuasi2Norm();
    if (sigma_norm > 0.0) {
      return diff_ten.GetQuasi2Norm() / sigma_norm;
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
std::pair<QLTensor<TenElemT, QNT>, QLTensor<TenElemT, QNT>>
FullEnvironmentTruncate(
    const QLTensor<TenElemT, QNT> &Upsilon,
    QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT> &sigma, //input & output
    const FullEnvironmentTruncateParams &trunc_params
) {
  using TenT = QLTensor<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  using DTenT = QLTensor<RealT, QNT>;
  using PtenVecT = PtenVec<TenElemT, QNT>;
  using BtenMatT = BtenMat<TenElemT, QNT>;
  QNT qn0 = sigma.Div();
  TenT sigma_Upsilon_tmp;
  Contract(&Upsilon, {0, 2}, &sigma, {1, 0}, &sigma_Upsilon_tmp);
  TenT sigma_orig;
  if constexpr (std::is_same<TenElemT, RealT>::value) {
    sigma_orig = sigma;
  } else {
    sigma_orig = ToComplex(sigma);
  }

  // then data of sigma can be thrown.
  sigma = DTenT();
  TenT u, vdag;
  //initialize u, vdag, and sigma_tilde
  RealT actual_trunc_err;
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
#ifndef NDEBUG
    auto diff = EvaluateHermiticity(Upsilon, {1, 0, 3, 2});
    assert(diff < 1e-8);

//    auto B_ten_dag = Dag(B_ten);
//    auto B_ten_trans = B_ten;
//    B_ten_trans.Transpose({2, 3, 0, 1});
//    auto diff_ten = B_ten_dag + (-B_ten_trans);
//    auto diff_norm = diff_ten.GetQuasi2Norm();
//    assert(diff_norm < 1e-3);
#endif

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
void FullEnvironmentTruncateInSquareLocalLoop(
    const FullEnvironmentTruncateParams &trunc_params,
    std::array<QLTensor<TenElemT, QNT>, 4> &gammas,  //input & output
    std::array<QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, QNT>, 4> &lambdas, //input & output
    const std::array<QLTensor<TenElemT, QNT>, 4> &Upsilons //input & output
) {
  using TenT = QLTensor<TenElemT, QNT>;
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

#endif //QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_PROJECTION4_IMPL_H

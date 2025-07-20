// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-21
*
* Description: QuantumLiquids/PEPS project. The generic PEPS class, implementation.
*/


#ifndef VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_PROJECTION_IMPL_H
#define VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_PROJECTION_IMPL_H

#include "qlmps/utilities.h"       //mock_qlten::SVD

namespace qlpeps {
using namespace qlten;
using qlmps::mock_qlten::SVD;

template<typename TenElemT, typename QNT>
TenElemT EvaluateTwoSiteEnergy(const QLTensor<TenElemT, QNT> &ham,
                               const QLTensor<TenElemT, QNT> &state) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT temp, temp_scale;
  Contract(&state, {2, 3}, &ham, {0, 2}, &temp);
  /*
   *       2        3
   *       |        |
   *  0------state------1
   */
  if (TenT::IsFermionic()) {
    TenT state_dag = Dag(state);
    state_dag.ActFermionPOps();
    Contract(&temp, {0, 1, 2, 3}, &state_dag, {0, 1, 2, 3}, &temp_scale);
    return temp_scale();
  } else {// bosonic case
    temp.Dag();
    Contract(&temp, {0, 1, 2, 3}, &state, {0, 1, 2, 3}, &temp_scale);
    return temp_scale();
  }
}

template<typename TenElemT, typename QNT>
ProjectionRes<TenElemT> SquareLatticePEPS<TenElemT, QNT>::NearestNeighborSiteProject(const TenT &gate_ten,
                                                                                     const SiteIdx &site,
                                                                                     const BondOrientation &orientation,
                                                                                     const SimpleUpdateTruncatePara &trunc_para,
                                                                                     const TenT &ham) {
  double norm;
  std::optional<TenElemT> e_loc;
  const size_t row = site[0], col = site[1];
  TenT tmp_ten[7];
  TenT q0, r0, q1, r1;
  TenT u, vt;
  double actual_trunc_err;
  size_t actual_D;
  TenT inv_lambda;
#ifndef NDEBUG
  auto physical_index = Gamma(row, col)->GetIndex(4);
#endif
  switch (orientation) {
    case HORIZONTAL: {
      /*                          0                                         0
       *                          |                                         |
       *                    Lam_v[rows_][cols_]                          Lam_v[rows_][cols_+1]
       *                          |                                         |
       *                          1                                         1
       *                          3                                         3
       *                          |                                         |
       *0-Lam_h[rows_][cols_]-1 0-Gamma[rows_][cols_]-2 0-Lam_h[rows_][cols_+1]-1 0-Gamma[rows_][cols_+1]-2 0-Lam_h[rows_][cols_+2]-1
       *                          |                                         |
       *                          1                                         1
       *                          0                                         0
       *                          |                                         |
       *                  Lam_v[rows_+1][cols_]                          Lam_v[rows_+1][cols_+1]
       *                          |                                         |
       *                          1                                         1
       */
      const size_t lcol = col;
      const size_t rcol = lcol + 1;
      assert(rcol < Cols());
      const SiteIdx r_site = {row, rcol};

      //Contract left site 3 lambdas
      tmp_ten[0] = Eat3SurroundLambdas_(site, RIGHT);
      QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);

      //Contract right site 3 lambdas
      tmp_ten[1] = Eat3SurroundLambdas_(r_site, LEFT);
      QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);

      Contract(&r0, {2}, lambda_horiz(row, rcol), {0}, tmp_ten + 2);
      Contract<TenElemT, QNT, true, false>(tmp_ten[2], r1, 2, 2, 1, tmp_ten[3]);
      Contract(tmp_ten + 3, {1, 3}, &gate_ten, {0, 1}, tmp_ten + 4);

      norm = tmp_ten[4].QuasiNormalize();
      if (!ham.IsDefault()) { //estimate the local energy by local environment
        const TenT *state = tmp_ten + 4;
        e_loc = EvaluateTwoSiteEnergy(ham, *state);
      }
      tmp_ten[4].Transpose({0, 2, 1, 3});
      lambda_horiz({row, rcol}) = DTenT();
      SVD(tmp_ten + 4, 2, qn0_,
          trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
          &u, lambda_horiz(row, rcol), &vt,
          &actual_trunc_err, &actual_D);

      // hand over lambdas from q0, q1, contract u or vt, setting Gammas
      //left site
      tmp_ten[5] = QTenSplitOutLambdas_(q0, site, RIGHT, trunc_para.trunc_err);
      Gamma(site) = TenT();
      Contract<TenElemT, QNT, false, true>(tmp_ten[5], u, 0, 0, 1, Gamma(site));
      Gamma(site).Transpose({1, 2, 4, 0, 3});

      tmp_ten[6] = QTenSplitOutLambdas_(q1, r_site, LEFT, trunc_para.trunc_err);
      Gamma(r_site) = TenT();
      Contract<TenElemT, QNT, false, false>(tmp_ten[6], vt, 0, 1, 1, Gamma(r_site));
      Gamma(r_site).Transpose({4, 0, 1, 2, 3});
      break;
    }
    case VERTICAL: {
      /*                           0
      *                            |
      *                     Lam_v[rows_][cols_]
      *                            |
      *                            1
      *                            3
      *                            |
      *  0-Lam_h[rows_][cols_]-1 0-Gamma[rows_][cols_]-2 0-Lam_h[rows_][cols_+1]-1
      *                            |
      *                            1
      *                            0
      *                            |
      *                    Lam_v[rows_+1][cols_]
      *                            |
      *                            1
      *                            3
      *                            |
      *0-Lam_h[rows_+1][cols_]-1 0-Gamma[rows_+1][cols_]-2 0-Lam_h[rows_+1][cols_+1]-1
      *                            |
      *                            1
      *                            0
      *                            |
      *                    Lam_v[rows_+2][cols_]
      *                            |
      *                            1
      */
      assert(row + 1 < this->Rows());
      tmp_ten[0] = Eat3SurroundLambdas_(site, DOWN);
      QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);

      tmp_ten[1] = Eat3SurroundLambdas_({row + 1, col}, UP);
      QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);

      Contract<TenElemT, QNT, true, true>(r0, lambda_vert({row + 1, col}), 2, 0, 1, tmp_ten[2]);
      Contract<TenElemT, QNT, true, false>(tmp_ten[2], r1, 2, 2, 1, tmp_ten[3]);
      Contract(tmp_ten + 3, {1, 3}, &gate_ten, {0, 1}, tmp_ten + 4);

      norm = tmp_ten[4].QuasiNormalize();
      if (!ham.IsDefault()) { //estimate the local energy by local environment
        const TenT *state = tmp_ten + 4;
        e_loc = EvaluateTwoSiteEnergy(ham, *state);
      }
      tmp_ten[4].Transpose({0, 2, 1, 3});
      lambda_vert({row + 1, col}) = DTenT();
      SVD(tmp_ten + 4, 2, qn0_,
          trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
          &u, lambda_vert(row + 1, col), &vt,
          &actual_trunc_err, &actual_D);

      // hand over lambdas from q0, q1, contract u or vt, setting Gammas
      tmp_ten[5] = QTenSplitOutLambdas_(q0, site, DOWN, trunc_para.trunc_err);
      Gamma(site) = TenT();
      Contract<TenElemT, QNT, true, true>(tmp_ten[5], u, 3, 0, 1, Gamma(site));
      Gamma(site).Transpose({2, 4, 0, 1, 3});

      tmp_ten[6] = QTenSplitOutLambdas_(q1, {row + 1, col}, UP, trunc_para.trunc_err);
      Gamma({row + 1, col}) = TenT();
      Contract<TenElemT, QNT, true, true>(tmp_ten[6], vt, 1, 1, 1, Gamma({row + 1, col}));
      Gamma({row + 1, col}).Transpose({2, 1, 0, 4, 3});
      break;
    }
    default: {
      std::cout << "We suppose square lattice now." << std::endl;
    }
  }
#ifndef NDEBUG
  assert(physical_index == Gamma(row, col)->GetIndex(4));
  assert(Gamma(row, col)->GetIndex(1) == lambda_vert(row + 1, col)->GetIndex(1));
  for (size_t i = 0; i < 7; i++) {
    assert(!tmp_ten[i].HasNan());
  }
#endif
  return {norm, actual_trunc_err, actual_D, e_loc};
}

template<typename TenElemT, typename QNT>
double SquareLatticePEPS<TenElemT, QNT>::NextNearestNeighborSiteProject(const TenT &gate_ten,
                                                                        const qlpeps::SiteIdx &first_site,
                                                                        const qlpeps::BondOrientation &orientation,
                                                                        const qlpeps::SimpleUpdateTruncatePara &trunc_para) {
  double norm;
  const size_t row = first_site[0], col = first_site[1];
  TenT tmp_ten[11];
  TenT q0, r0, q1, r1;
  TenT u1, vt1, u2, vt2;
  QLTensor<QLTEN_Double, QNT> s1, s2;
  double actual_trunc_err;
  size_t actual_D;
  TenT inv_lambda;
  switch (orientation) {
    case HORIZONTAL: {
      const size_t lcol = col;
      const size_t mcol = col + 1;
      const size_t rcol = col + 2;
      const SiteIdx lsite = {row, lcol};
      const SiteIdx msite = {row, mcol};
      const SiteIdx rsite = {row, rcol};

      tmp_ten[0] = Eat3SurroundLambdas_(lsite, RIGHT);
      QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);
      tmp_ten[1] = Eat3SurroundLambdas_(rsite, LEFT);
      QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);
      tmp_ten[2] = EatSurroundLambdas_(msite);

      break;
    }
    case VERTICAL: {

      break;
    }
    default: {
      std::cerr << std::endl;
    }
  }
}

/**
 *
 *      |             |
 *      |             |
 * -----B-------------A--------
 *      |             |
 *      |             |
 *      |             |
 * -----C----------------------
 *      |             |
 *      |             |
 *
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param gate_ten  order of the indexes: upper-right site; upper-left site; lower-left site.
 * @param upper_site
 * @param trunc_para
 * @return
 */
template<typename TenElemT, typename QNT>
ProjectionRes<TenElemT> SquareLatticePEPS<TenElemT,
                                          QNT>::UpperLeftTriangleProject(const SquareLatticePEPS::TenT &gate_ten,
                                                                         const SiteIdx &left_upper_site,
                                                                         const SimpleUpdateTruncatePara &trunc_para) {
#ifndef NDEBUG
  auto physical_index = Gamma(left_upper_site).GetIndex(4);
#endif
  double norm = 1;
  size_t row = left_upper_site[0], col = left_upper_site[1];
  SiteIdx right_site = {row, col + 1};
  SiteIdx lower_site = {row + 1, col};
  TenT tmp_ten[11], q0, r0, q1, r1;
  tmp_ten[0] = Eat3SurroundLambdas_(right_site, LEFT);
  QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);
  tmp_ten[1] = Eat3SurroundLambdas_(lower_site, UP);
  QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);
  tmp_ten[2] = EatSurroundLambdas_(left_upper_site);
  Contract<TenElemT, QNT, false, false>(tmp_ten[2], r1, 2, 2, 1, tmp_ten[3]);
  Contract<TenElemT, QNT, true, true>(r0, tmp_ten[3], 2, 0, 1, tmp_ten[4]);
  Contract(tmp_ten + 4, {1, 3, 6}, &gate_ten, {0, 1, 2}, tmp_ten + 5);
  /**
   *  tmp_ten[5] is a rank-7 tensor
   *         1
   *         |
   *  2--tmp_ten[5]--0, physical index = 4,5,6, with order upper-right site->upper-left site->lower-left site.
   *        |
   *        3
   */

  tmp_ten[5].Transpose({0, 4, 5, 1, 2, 6, 3});
  TenT u1, vt1, u2, vt2;
  DTenT s1, s2;
  double trunc_err1, trunc_err2;
  size_t D1, D2;
  qlten::SVD(tmp_ten + 5, 5, tmp_ten[5].Div(),
             trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u1, &s1, &vt1, &trunc_err1, &D1);
  norm *= s1.QuasiNormalize();
  lambda_vert({lower_site}) = s1;
  tmp_ten[6] = QTenSplitOutLambdas_(q1, lower_site, UP, trunc_para.trunc_err);

  Gamma(lower_site) = TenT();
  Contract<TenElemT, QNT, false, false>(vt1, tmp_ten[6], 2, 1, 1, Gamma(lower_site));
  Gamma(lower_site).Transpose({4, 3, 2, 0, 1});
  Contract(&u1, {5}, &s1, {0}, &tmp_ten[7]);
  qlten::SVD(tmp_ten + 7, 2, qn0_, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u2, &s2, &vt2, &trunc_err2, &D2);

  /**
 *       2
 *       |
 *  3---vt2--0, physical index = 1
 *      |
 *      4
 */
  norm *= s2.QuasiNormalize();
  lambda_horiz({right_site}) = s2;
  lambda_horiz({right_site}).Transpose({1, 0});
  tmp_ten[8] = QTenSplitOutLambdas_(q0, right_site, LEFT, trunc_para.trunc_err);
  Gamma(right_site) = TenT();
  Contract<TenElemT, QNT, false, true>(tmp_ten[8], u2, 0, 0, 1, Gamma(right_site));
  Gamma(right_site).Transpose({4, 0, 1, 2, 3});

  auto inv_lam = DiagMatInv(s1, trunc_para.trunc_err);
  Contract(&vt2, {4}, &inv_lam, {0}, &tmp_ten[9]);
  inv_lam = DiagMatInv(lambda_vert({row, col}), trunc_para.trunc_err);
  Contract<TenElemT, QNT, false, true>(tmp_ten[9], inv_lam, 2, 1, 1, tmp_ten[10]);
  inv_lam = DiagMatInv(lambda_horiz({row, col}), trunc_para.trunc_err);
  Gamma({left_upper_site}) = TenT();
  Contract<TenElemT, QNT, false, true>(tmp_ten[10], inv_lam, 0, 1, 1, Gamma({left_upper_site}));
  Gamma({left_upper_site}).Transpose({4, 0, 1, 3, 2});
#ifndef NDEBUG
  assert(physical_index == Gamma(left_upper_site).GetIndex(4));
  assert(physical_index == Gamma(right_site).GetIndex(4));
  assert(physical_index == Gamma(lower_site).GetIndex(4));
#endif
  return {norm, trunc_err1 + trunc_err2, std::max(D1, D2)};
}

/**
 *
 *      |             |
 *      |             |
 * -------------------A--------
 *      |             |
 *      |             |
 *      |             |
 * -----C-------------B--------
 *      |             |
 *      |             |
 * @tparam TenElemT
 * @tparam QNT
 * @param gate_ten  order of the indexes: upper-right site; lower-right site; lower-left site.
 * @param upper_site
 * @param trunc_para
 * @return
 */
template<typename TenElemT, typename QNT>
ProjectionRes<TenElemT> SquareLatticePEPS<TenElemT,
                                          QNT>::LowerRightTriangleProject(const SquareLatticePEPS::TenT &gate_ten,
                                                                          const SiteIdx &upper_site,
                                                                          const SimpleUpdateTruncatePara &trunc_para) {
#ifndef NDEBUG
  auto physical_index = Gamma(upper_site).GetIndex(4);
#endif
  double norm = 1;
  size_t row = upper_site[0], col = upper_site[1];
  SiteIdx left_site = {row + 1, col - 1};
  SiteIdx right_down_site = {row + 1, col};
  TenT tmp_ten[11], q0, r0, q1, r1;
  tmp_ten[0] = Eat3SurroundLambdas_(left_site, RIGHT);
  QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);
  tmp_ten[1] = Eat3SurroundLambdas_(upper_site, DOWN);
  QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);
  tmp_ten[2] = EatSurroundLambdas_(right_down_site);
  Contract<TenElemT, QNT, true, false>(r1, tmp_ten[2], 2, 4, 1, tmp_ten[3]);
  Contract<TenElemT, QNT, false, false>(tmp_ten[3], r0, 3, 2, 1, tmp_ten[4]);
  Contract(tmp_ten + 4, {3, 4, 6}, &gate_ten, {0, 1, 2}, tmp_ten + 5);
  /**
   *  tmp_ten[5] is a rank-7 tensor
   *         2
   *         |
   *  3--tmp_ten[5]--1, physical index = 4,5,6
   *        |
   *        0
   */
  tmp_ten[5].Transpose({6, 3, 0, 1, 5, 4, 2});
  TenT u1, vt1, u2, vt2;
  DTenT s1, s2;
  double trunc_err1, trunc_err2;
  size_t D1, D2;
  qlten::SVD(tmp_ten + 5, 5, tmp_ten[5].Div(),
             trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u1, &s1, &vt1, &trunc_err1, &D1);
  norm *= s1.QuasiNormalize();
  lambda_vert({right_down_site}) = s1;
  lambda_vert({right_down_site}).Transpose({1, 0});
  tmp_ten[6] = QTenSplitOutLambdas_(q1, upper_site, DOWN, trunc_para.trunc_err);

  Gamma(upper_site) = TenT();
  Contract<TenElemT, QNT, true, true>(tmp_ten[6], vt1, 3, 2, 1, Gamma(upper_site));
  Gamma(upper_site).Transpose({2, 3, 0, 1, 4});
  Contract(&u1, {5}, &s1, {0}, &tmp_ten[7]);
  qlten::SVD(tmp_ten + 7, 2, qn0_, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u2, &s2, &vt2, &trunc_err2, &D2);
  /**
   *       4
   *       |
   *  0---vt2--2, physical index = 3
   *       |
   *       1
   */
  norm *= s2.QuasiNormalize();
  lambda_horiz({right_down_site}) = s2;
  tmp_ten[8] = QTenSplitOutLambdas_(q0, left_site, RIGHT, trunc_para.trunc_err);
  Gamma(left_site) = TenT();
  Contract<TenElemT, QNT, false, false>(tmp_ten[8], u2, 0, 1, 1, Gamma(left_site));
  Gamma(left_site).Transpose({1, 2, 3, 0, 4});
  auto inv_lam = DiagMatInv(s1, trunc_para.trunc_err);
  Contract(&vt2, {4}, &inv_lam, {0}, &tmp_ten[9]);
  inv_lam = DiagMatInv(lambda_vert({row + 2, col}), trunc_para.trunc_err);
  Contract<TenElemT, QNT, false, true>(tmp_ten[9], inv_lam, 1, 0, 1, tmp_ten[10]);
  inv_lam = DiagMatInv(lambda_horiz({row + 1, col + 1}), trunc_para.trunc_err);
  Gamma({right_down_site}) = TenT();
  Contract<TenElemT, QNT, false, true>(tmp_ten[10], inv_lam, 0, 0, 1, Gamma({right_down_site}));
  Gamma({right_down_site}).Transpose({2, 3, 4, 1, 0});
#ifndef NDEBUG
  assert(physical_index == Gamma(upper_site).GetIndex(4));
  assert(physical_index == Gamma(left_site).GetIndex(4));
  assert(physical_index == Gamma(right_down_site).GetIndex(4));
#endif
  return {norm, trunc_err1 + trunc_err2, std::max(D1, D2)};
}

/**
 *
 *      |             |
 *      |             |
 * -----A----------------------
 *      |             |
 *      |             |
 *      |             |
 * -----B-------------C--------
 *      |             |
 *      |             |
 * @param gate_ten  order of the indexes: upper-left site; lower-left site; lower-right site.
 * @param upper_site
 * @param trunc_para
 * @return
 */
template<typename TenElemT, typename QNT>
double SquareLatticePEPS<TenElemT, QNT>::LowerLeftTriangleProject(const QLTensor<TenElemT, QNT> &gate_ten,
                                                                  const qlpeps::SiteIdx &upper_left_site,
                                                                  const qlpeps::SimpleUpdateTruncatePara &trunc_para) {
  double norm = 1;
  size_t row = upper_left_site[0], col = upper_left_site[1];
  SiteIdx lower_left_site = {row + 1, col};
  SiteIdx lower_right_site = {row + 1, col + 1};
#ifndef NDEBUG
  auto index_1 = Gamma(upper_left_site).GetIndexes();
  auto index_2 = Gamma(lower_left_site).GetIndexes();
  auto index_3 = Gamma(lower_right_site).GetIndexes();
#endif
  TenT tmp_ten[11], q0, r0, q1, r1;
  tmp_ten[0] = Eat3SurroundLambdas_(upper_left_site, DOWN);
  QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);
  tmp_ten[1] = Eat3SurroundLambdas_(lower_right_site, LEFT);
  QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);
  /**
   *       2
   *       |
   *  3---q1---1
   *      |
   *      0
   *
   */
  tmp_ten[2] = EatSurroundLambdas_(lower_left_site);
  Contract<TenElemT, QNT, true, false>(r0, tmp_ten[2], 2, 4, 1, tmp_ten[3]);
  Contract<TenElemT, QNT, true, false>(tmp_ten[3], r1, 5, 2, 1, tmp_ten[4]);
  Contract(tmp_ten + 4, {1, 2, 6}, &gate_ten, {0, 1, 2}, tmp_ten + 5);

  /**
   *  tmp_ten[5] is a rank-7 tensor
   *         0
   *         |
   *  1--tmp_ten[5]--3, physical index = 4,5,6, with order upper-left site->lower-left site->lower-right site.
   *        |
   *        2
   */
  tmp_ten[5].Transpose({0, 4, 5, 1, 2, 6, 3}); //(0,4) for upper site, (5,1,2) for left-lower site, (6,3) for right site
  TenT u1, vt1, u2, vt2;
  DTenT s1, s2;
  double trunc_err;
  size_t D;
  qlten::SVD(tmp_ten + 5, 5, tmp_ten[5].Div(),
             trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u1, &s1, &vt1, &trunc_err, &D);
  lambda_horiz({lower_right_site}) = s1;
  tmp_ten[6] = QTenSplitOutLambdas_(q1, lower_right_site, LEFT, trunc_para.trunc_err);
  /**
   *       3
   *       |
   *  0--tmp6--2    no phy idx
   *      |
   *      1
   */
  Gamma(lower_right_site) = TenT();
  Contract<TenElemT, QNT, true, true>(vt1, tmp_ten[6], 2, 0, 1, Gamma(lower_right_site));
  Gamma(lower_right_site).Transpose({0, 2, 3, 4, 1});

  Contract(&u1, {5}, &s1, {0}, &tmp_ten[7]);
  qlten::SVD(tmp_ten + 7, 2, qn0_, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u2, &s2, &vt2, &trunc_err, &D);
  /**
   *       0
   *       |
   *  2---vt2--4, physical index = 1
   *       |
   *       3
   */
#ifndef NDEBUG
  assert(vt2.GetIndex(1) == index_2.back());
#endif
  norm *= s2.QuasiNormalize();
  lambda_vert({lower_left_site}) = s2;
  tmp_ten[8] = QTenSplitOutLambdas_(q0, upper_left_site, DOWN, trunc_para.trunc_err);
/*
 *        1
 *        |
 *  2--tmp_ten[8]--0
 *        |
 *        3
 */
  Gamma(upper_left_site) = TenT();
  Contract<TenElemT, QNT, true, true>(tmp_ten[8], u2, 3, 0, 1, Gamma(upper_left_site));
  Gamma(upper_left_site).Transpose({2, 4, 0, 1, 3});

  auto inv_lam = DiagMatInv(s1, trunc_para.trunc_err);
  Contract(&vt2, {4}, &inv_lam, {0}, &tmp_ten[9]);
  inv_lam = DiagMatInv(lambda_vert({row + 2, col}), trunc_para.trunc_err);
  Contract<TenElemT, QNT, false, true>(tmp_ten[9], inv_lam, 3, 0, 1, tmp_ten[10]);
  inv_lam = DiagMatInv(lambda_horiz({row + 1, col}), trunc_para.trunc_err);
  Gamma({lower_left_site}) = TenT();
  Contract<TenElemT, QNT, false, false>(tmp_ten[10], inv_lam, 3, 1, 1, Gamma({lower_left_site}));
  Gamma({lower_left_site}).Transpose({4, 0, 1, 2, 3});
#ifndef NDEBUG
  auto index_1p = Gamma(upper_left_site).GetIndexes();
  auto index_2p = Gamma(lower_left_site).GetIndexes();
  auto index_3p = Gamma(lower_right_site).GetIndexes();
  assert(index_1.back() == index_1p.back());
  assert(index_2.back() == index_2p.back());
  assert(index_3.back() == index_3p.back());
#endif
  return norm;
}

}//qlpeps

#endif //VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_BASIC_IMPL_H

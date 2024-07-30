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

namespace qlpeps {
using namespace qlten;
using qlmps::mock_qlten::SVD;


//helper

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
  Contract(&gamma, {2}, &env_lambda_r, {1}, &res);
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
  inv_lambda_l = ElementWiseInv(env_lambda_l, 1e-15);
  inv_lambda_r = ElementWiseInv(env_lambda_r, 1e-15);
  Contract(&gamma, {2}, &inv_lambda_l, {1}, &tmp);
  Contract(&gamma, {2}, &inv_lambda_r, {1}, &res);
  res.Transpose({0, 1, 3, 4, 2});
  gamma = res;
}

template<typename TenElemT, typename QNT>
double SquareLatticePEPS<TenElemT, QNT>::LocalSquareLoopProject(
    const qlpeps::SquareLatticePEPS<TenElemT, QNT>::LocalSquareLoopGateT &gate_tens,
    const qlpeps::SiteIdx &upper_left_site,
    const qlpeps::LoopUpdateTruncatePara &params) {
  const size_t row = upper_left_site.row();
  const size_t col = upper_left_site.col();

  PatSquareLocalLoopProjector_(gate_tens, upper_left_site);

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

  WeightedTraceGaugeFixingInSquareLocalLoop_(params, gammas, lambdas, Upsilons);
  FullEnvironmentTruncateInSquareLocalLoop_(params, gammas, lambdas, Upsilons);

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
  const size_t row = upper_left_site.row();
  const size_t col = upper_left_site.col();

  //Update Gamma Tensors
  const TenT &Gamma0 = Gamma(upper_left_site);
  TenT tmp;
  Contract(&Gamma0, {4}, &gate_tens[0], {1}, &tmp);
  tmp.FuseIndex(1, 4);
  tmp.FuseIndex(2, 5);
  tmp.Transpose({2, 1, 0, 3});
  Gamma(upper_left_site) = tmp;

  const TenT &Gamma1 = Gamma({row, col + 1});
  tmp = TenT();
  Contract(&Gamma1, {4}, &gate_tens[1], {1}, &tmp);
  tmp.FuseIndex(0, 4);
  tmp.FuseIndex(1, 5);
  tmp.Transpose({1, 0, 2, 3, 4});
  Gamma({row, col + 1}) = tmp;

  const TenT &Gamma2 = Gamma({row + 1, col + 1});
  tmp = TenT();
  Contract(&Gamma2, {4}, &gate_tens[2], {1}, &tmp);
  tmp.FuseIndex(0, 6);
  tmp.FuseIndex(3, 4);
  tmp.Transpose({1, 2, 3, 0, 4});
  Gamma({row + 1, col + 1}) = tmp;

  const TenT &Gamma3 = Gamma({row + 1, col});
  tmp = TenT();
  Contract(&Gamma3, {4}, &gate_tens[3], {1}, &tmp);
  tmp.FuseIndex(2, 4);
  tmp.FuseIndex(3, 5);
  tmp.Transpose({2, 3, 1, 0, 4});
  Gamma({row + 1, col}) = tmp;

  //Update Lambda Tensors
  std::vector<TenT *> lambdas(4, nullptr);
  lambdas[0] = lambda_horiz(row, col + 1);
  lambdas[1] = lambda_vert(row + 1, col + 1);
  lambdas[2] = lambda_horiz(row + 1, col + 1);
  lambdas[3] = lambda_vert(row + 1, col);

  for (size_t i = 0; i < 4; i++) {
    Index<QNT> idx0 = gate_tens[i + 1].GetIndex(0);
    Index<QNT> idx1 = gate_tens[i].GetIndex(3);
    if (i > 1) {
      std::swap(idx0, idx1);
    }
    TenT eye = TenT({idx0, idx1});
    for (size_t i = 0; i < idx0.dim(); i++) {
      eye({i, i}) = 1.0;
    }
    tmp = TenT();
    Contract(lambdas[i], {}, &eye, {}, &tmp);
    tmp.FuseIndex(1, 3);
    tmp.FuseIndex(1, 2);
    *lambdas[i] = tmp;
  }
}



//helper
/**
 *   * |-1-0-sigma^dag-1--1---------3
     * |                      |
     * L                   Upsilon
     * |                      |
     * |-0-0-sigma-1-----0----------2
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> left_vec_multiple_transfer_tens(
    const QLTensor<TenElemT, QNT> &left_vec,
    const QLTensor<TenElemT, QNT> &sigma,
    const QLTensor<TenElemT, QNT> &sigma_dag,
    const QLTensor<TenElemT, QNT> &Upsilon
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT tmp, tmp1, res;
  Contract(&left_vec, {1}, &sigma_dag, {0}, &tmp);
  Contract(&sigma, {0}, &tmp, {0}, &tmp1);
  Contract(&tmp1, {0, 1}, &Upsilon, {0, 1}, &res);
  return res;
}

/**
     * 1----------3----0-sigma^dag-1---1---
     *       |                            |
     *    Upsilon                         R
     *       |                            |
     * 0----------2----0-sigma-1-------0--|
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> right_vec_multiple_transfer_tens(
    const QLTensor<TenElemT, QNT> &right_vec,
    const QLTensor<TenElemT, QNT> &sigma,
    const QLTensor<TenElemT, QNT> &sigma_dag,
    const QLTensor<TenElemT, QNT> &Upsilon
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT tmp, tmp1, res;
  Contract(&sigma, {1}, &right_vec, {0}, &tmp);
  Contract(&tmp, {1}, &sigma_dag, {1}, &tmp1);
  Contract(&Upsilon, {2, 3}, &tmp1, {0, 1}, &res);
  return res;
}

/**
 * Fix weighted trace gauge for the loop tensors
 *
 */
template<typename TenElemT, typename QNT>
void SquareLatticePEPS<TenElemT, QNT>::
WeightedTraceGaugeFixingInSquareLocalLoop_(
    const qlpeps::LoopUpdateTruncatePara &params,
    std::array<QLTensor<TenElemT, QNT>, 4> &gammas,  //input & output
    std::array<QLTensor<TenElemT, QNT>, 4> &lambdas, //input & output
    std::array<QLTensor<TenElemT, QNT>, 4> &Upsilons //input & output
) {
  // Contruct the Upsilon_i tensor
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
    TenT &Upsilon = Upsilons[i];
    TenT &sigma = lambdas[(i + 3) % 4];
    TenT sigma_dag = Dag(sigma);
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
    TenT left_vec = TenT({index0, index1});
    TenT right_vec = TenT({index1, index0});
    for (size_t i = 0; i < index0.dim(); i++) {
      left_vec({i, i}) = 1.0;
      right_vec({i, i}) = 1.0;
    }
    left_vec.Normalize();
    right_vec.Normalize();
    //Naive Realization: power method. TODO:Lanczos
    double norm = 0;
    for (size_t i = 0; i < 100; i++) {
      left_vec = left_vec_multiple_transfer_tens(left_vec, sigma, sigma_dag, Upsilon);
      double update_norm = left_vec.Normalize();
      if (std::abs((update_norm - norm) / update_norm) < 1e-10) {
        break;
      } else {
        norm = update_norm;
      }
    }
    norm = 0;
    for (size_t i = 0; i < 100; i++) {
      right_vec = right_vec_multiple_transfer_tens(right_vec, sigma, sigma_dag, Upsilon);
      double update_norm = right_vec.Normalize();
      if (std::abs((update_norm - norm) / update_norm) < 1e-10) {
        break;
      } else {
        norm = update_norm;
      }
    }

    //EVD for eigenvectors, and update the Upsilon_i, Gammas, and Lambdas
    TenT u_l, d_l, u_r, d_r, sqrt_dl, sqrt_dr, inv_sqrt_dl, inv_sqrt_dr;
    SymMatEVD(&left_vec, &u_l, &d_l);
    SymMatEVD(&right_vec, &u_r, &d_r);
    sqrt_dl = ElementWiseSqrt(d_l);
    sqrt_dr = ElementWiseSqrt(d_r);
    inv_sqrt_dl = ElementWiseInv(sqrt_dl);
    inv_sqrt_dr = ElementWiseInv(sqrt_dr);
    TenT ul_dag = Dag(u_l);
    TenT ur_dag = Dag(u_r);


    //calculate sigma_prime
    TenT sigma_prime, tmp0, tmp1, tmp2;
    Contract(&sqrt_dl, {0}, &u_l, {1}, &tmp0);
    Contract(&tmp0, {1}, &sigma, {0}, &tmp1);
    Contract(&tmp1, {1}, &u_r, {0}, &tmp2);
    Contract(&tmp2, {1}, &sqrt_dr, {0}, &sigma_prime);

    sigma = TenT();
    TenT v_l, v_r_dag;
    qlmps::mock_qlten::SVD(&sigma_prime, 1, qn0_, &v_l, &sigma, &v_r_dag);
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
    Contract(&v_r_dag, {1}, &inv_sqrt_dr, {1}, &tmp0);
    Contract(&tmp0, {1}, &ur_dag, {1}, &y_inv);
    TenT y_inv_dag = Dag(y_inv);
    Contract(&y_inv, {1}, &tmp2, {0}, &tmp3);
    Upsilon = TenT();
    Contract(&y_inv_dag, {1}, &tmp3, {1}, &Upsilon);
    Upsilon.Transpose({1, 0, 2, 3});

    TenT &gamma_tail = gammas[(i + 3) % 4];
    TenT &gamma_head = gammas[i];
    tmp0 = TenT();
    tmp1 = TenT();
    Contract(&gamma_tail, {4}, &x_inv, {0}, &tmp0);
    gamma_tail = std::move(tmp0);
    Contract(&y_inv, {1}, &gamma_head, {0}, &tmp1);
    gamma_head = std::move(tmp1);
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
    const LoopUpdateTruncatePara &trunc_para
) {
  using TenT = QLTensor<TenElemT, QNT>;
  using DTenT = QLTensor<QLTEN_Double, QNT>;
  using PtenVecT = PtenVec<TenElemT, QNT>;
  using BtenMatT = BtenMat<TenElemT, QNT>;
  QNT qn0 = sigma.Div();
  TenT sigma_Upsilon_tmp;
  Contract(&Upsilon, {0, 2}, &sigma, {1, 0}, &sigma_Upsilon_tmp);
  // then data of sigma can be thrown.
  sigma = DTenT();
  TenT u, vdag;
  //initialize u, vdag, and sigma_tilde
  double actual_trunc_err;
  size_t actual_D;
  SVD(&sigma, 1, qn0, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
      &vdag, &sigma, &u, &actual_trunc_err, &actual_D);

  TenT sigma_old = sigma;
  for (size_t iter = 0; iter < trunc_para.fet_max_iter; iter++) {
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
                                trunc_para.cg_params.max_iter,
                                trunc_para.cg_params.tolerance, cg_iter);

    Contract(&update_r_vec.p_ten, {0}, &u, {0}, tmp + 1);
    u = TenT();
    vdag = TenT();
    sigma = TenT();
    // update v, and sigma according to u*R = u'*sigma*v
    SVD(tmp + 1, 1, qn0, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
        &vdag, &sigma, &u, &actual_trunc_err, &actual_D);
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
                                trunc_para.cg_params.max_iter,
                                trunc_para.cg_params.tolerance, cg_iter);
    Contract(&vdag, {1}, &update_l_vec.p_ten, {1}, tmp + 3);

    u = TenT();
    vdag = TenT();
    sigma = TenT();
    SVD(tmp + 3, 1, qn0, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
        &vdag, &sigma, &u, &actual_trunc_err, &actual_D);
    if (diag_mat_diff(sigma, sigma_old) < trunc_para.fet_tol) {
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
    const qlpeps::LoopUpdateTruncatePara &params,
    std::array<QLTensor<TenElemT, QNT>, 4> &gammas,  //input & output
    std::array<QLTensor<TenElemT, QNT>, 4> &lambdas, //input & output
    std::array<QLTensor<TenElemT, QNT>, 4> &Upsilons //input & output
) {
  for (size_t i = 0; i < 4; i++) {
    auto [u, vdag] = FullEnvironmentTruncate(Upsilons[i], lambdas[i], params);
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

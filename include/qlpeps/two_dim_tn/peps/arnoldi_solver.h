/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-08-09
*
* Description: QuantumLiquids/PEPS project. Arnoldi algorithm to find the eigen vector
*/

#ifndef VMC_PEPS_TWO_DIM_TN_PEPS_ARNOLDI_SOLVER_H
#define VMC_PEPS_TWO_DIM_TN_PEPS_ARNOLDI_SOLVER_H


#include "qlten/framework/hp_numeric/backend_selector.h" //lapacke

#include <functional>
#include <optional>
#include <memory>
#include "qlmps/algorithm/lanczos_params.h"           //LanczosParams
namespace qlpeps {
using ArnoldiParams = qlmps::LanczosParams;
using qlten::QLTensor;
using qlten::QLTEN_Double;

template<typename TenElemT, typename QNT>
struct ArnoldiRes {
  TenElemT eigenvalue;
  QLTensor<TenElemT, QNT> eig_vec;
};

template<typename TenElemT, typename QNT>
using TransfTenMultiVec = std::function<QLTensor<TenElemT, QNT>(
    const QLTensor<TenElemT, QNT> &,
    const QLTensor<QLTEN_Double, QNT> &,
    const QLTensor<QLTEN_Double, QNT> &,
    const QLTensor<TenElemT, QNT> &)>;

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
    const QLTensor<QLTEN_Double, QNT> &sigma,
    const QLTensor<QLTEN_Double, QNT> &sigma_dag,
    const QLTensor<TenElemT, QNT> &Upsilon
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT tmp, tmp1, res;
  Contract(&left_vec, {1}, &sigma_dag, {0}, &tmp);
  Contract<TenElemT, QNT, false, true>(sigma, tmp, 0, 0, 1, tmp1);
  Contract<TenElemT, QNT, true, false>(tmp1, Upsilon, 0, 0, 2, res);
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
    const QLTensor<QLTEN_Double, QNT> &sigma,
    const QLTensor<QLTEN_Double, QNT> &sigma_dag,
    const QLTensor<TenElemT, QNT> &Upsilon
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT tmp, tmp1, res;
  Contract(&sigma, {1}, &right_vec, {0}, &tmp);
  Contract<TenElemT, QNT, true, false>(tmp, sigma_dag, 1, 1, 1, tmp1);
  Contract(&Upsilon, {2, 3}, &tmp1, {0, 1}, &res);
  return res;
}

template<typename ElemT>
struct MatDomiEigenSystem {
  bool valid;
  ElemT eigen_value;
  std::vector<ElemT> right_eigen_vec;
  std::vector<ElemT> left_eigen_vec;
};

template<typename ElemT>
MatDomiEigenSystem<ElemT> HeiMatDiag(
    const std::vector<std::vector<ElemT>> &h, const size_t n) {
  using complex_t = std::complex<double>;
  std::vector<ElemT> h_mat(n * n, 0.0);

  // Flatten the n-by-n matrix
  for (size_t i = 0; i < n; ++i) { // row index
    for (size_t j = std::max(i, (size_t) 1) - 1; j < n; ++j) {
      h_mat[i * n + j] = h[i][j];
    }
  }
  std::vector<ElemT> vr(n * n), vl(n * n);
  std::vector<double> w_abs(n);
  std::unique_ptr<double[]> wr, wi;
  std::unique_ptr<complex_t[]> w;
  ElemT dominant_w;
  size_t max_idx;

  int info;
  if constexpr (std::is_same<ElemT, double>::value) {
    wr = std::make_unique<double[]>(n);
    wi = std::make_unique<double[]>(n);
    info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,
                         'V', // left eigenvectors are computed
                         'V', // right eigenvectors are computed
                         lapack_int(n), // order of h matrix
                         h_mat.data(),  // h matrix
                         lapack_int(n), // leading dimension of the h matrix (question : in lapack what's this parameter, what's the different with the size of h)
                         wr.get(), // real part of the eigenvalues
                         wi.get(), // imaginary part of the eigenvalues
                         vl.data(), // the output eigenvectors
                         lapack_int(n),
                         vr.data(), // the output eigenvectors.
                         lapack_int(n)
    );
    // possible alternative choice : LAPACKE_dhsein

    for (size_t i = 0; i < n; i++) {
      w_abs[i] = std::abs(complex_t(wr[i], wi[i]));
    }
    auto max_iter = std::max_element(w_abs.cbegin(), w_abs.cend());
    max_idx = max_iter - w_abs.cbegin();

    if (std::abs(wi[max_idx] / wr[max_idx]) > 1e-14) {
      //degeneracy case
      MatDomiEigenSystem<ElemT> res;
      res.valid = false;
      return res;
    }
    dominant_w = wr[max_idx];
  } else if constexpr (std::is_same<ElemT, complex_t>::value) {
    w = std::make_unique<complex_t[]>(n);
    info = LAPACKE_zgeev(LAPACK_ROW_MAJOR,
                         'V', // left eigenvectors are computed
                         'V', // right eigenvectors are computed
                         lapack_int(n), // order of h matrix
                         reinterpret_cast<lapack_complex_double *>(h_mat.data()),  // h matrix
                         lapack_int(n), // leading dimension of the h matrix (question : in lapack what's this parameter, what's the different with the size of h)
                         reinterpret_cast<lapack_complex_double *>(w.get()),
                         reinterpret_cast<lapack_complex_double *>(vl.data()), // the output eigenvectors
                         lapack_int(n),
                         reinterpret_cast<lapack_complex_double *>(vr.data()), // the output eigenvectors.
                         lapack_int(n)
    );
    for (size_t i = 0; i < n; i++) {
      w_abs[i] = std::abs(w[i]);
    }
    auto max_iter = std::max_element(w_abs.cbegin(), w_abs.cend());
    max_idx = max_iter - w_abs.cbegin();
    dominant_w = w[max_idx];
  } else {
    std::cout << "Unexpected element type" << std::endl;
    exit(1);
  }

  if (info != 0) {
    throw std::runtime_error("LAPACKE_dgees failed.");
  }

  // Extract the corresponding eigenvector
  std::vector<ElemT> right_dominant_eigenvector(n);
  std::vector<ElemT> left_dominant_eigenvector(n);
  for (size_t i = 0; i < n; ++i) {
    right_dominant_eigenvector[i] = vr[i * n + max_idx];
    left_dominant_eigenvector[i] = vl[i * n + max_idx];
  }

  return {true, dominant_w, right_dominant_eigenvector, left_dominant_eigenvector};
}

/**
 * For the fermionic case, note that one should define a positive norm by replace
 * QLTensor.Norm() with QLTensor.QuasiNorm(), equivalent to apply additional identity operators when calculate overlap
 *
 */
template<typename TenElemT, typename QNT>
ArnoldiRes<TenElemT, QNT> ArnoldiSolver(
    const QLTensor<TenElemT, QNT> &Upsilon,
    const QLTensor<QLTEN_Double, QNT> &sigma,
    const QLTensor<QLTEN_Double, QNT> &sigma_dag,
    const QLTensor<TenElemT, QNT> &vec0,
    const ArnoldiParams &params,
    TransfTenMultiVec<TenElemT, QNT> transfer_tens_multiple_vec
) {
  using TenT = QLTensor<TenElemT, QNT>;
  const size_t mat_eff_dim = vec0.size();
  const size_t max_iter = std::min(mat_eff_dim, params.max_iterations);
  std::vector<TenT *> bases(max_iter);
  std::vector<TenT *> bases_dag(max_iter);
  std::vector<std::vector<TenElemT>>
      h(max_iter, std::vector<TenElemT>(max_iter, 0.0));
  bases[0] = new TenT(vec0);
  bases[0]->QuasiNormalize();
  bases_dag[0] = new TenT(Dag(*bases[0]));
  if constexpr (QLTensor<TenElemT, QNT>::IsFermionic()) {
    bases_dag[0]->ActFermionPOps();
  }
//  std::optional<TenElemT> eigenvalue_last;
  double max_abs_h_elem(0.0);
  for (size_t k = 1; k < max_iter; k++) {
    bases[k] = new TenT();
    *bases[k] = transfer_tens_multiple_vec(*bases[k - 1], sigma, sigma_dag, Upsilon);
    for (size_t j = 0; j < k; j++) {
      TenT overlap_ten;
      Contract(bases_dag[j], {0, 1}, bases[k], {0, 1}, &overlap_ten);
      h[j][k - 1] = overlap_ten();
      (*bases[k]) += (-h[j][k - 1]) * (*bases[j]);
      max_abs_h_elem = std::max(max_abs_h_elem, std::abs(h[j][k - 1]));
    }
    h[k][k - 1] = bases[k]->QuasiNormalize();
    max_abs_h_elem = std::max(max_abs_h_elem, std::abs(h[k][k - 1]));
    bases_dag[k] = new TenT(Dag(*bases[k]));
    if constexpr (QLTensor<TenElemT, QNT>::IsFermionic()) {
      bases_dag[k]->ActFermionPOps();
      //such definition can make sure the overlap between <base|base> >=0, so that it constitutes well-defined norm.
    }
    if (std::abs(h[k][k - 1]) / max_abs_h_elem > params.error && k < max_iter - 1) {
      continue;
    }

    MatDomiEigenSystem<TenElemT> eigen_solution = HeiMatDiag(h, k);
    if (eigen_solution.valid
//    &&(eigenvalue_last.has_value() &&
//            std::abs((eigen_solution.eigen_value - eigenvalue_last.value()) / eigenvalue_last.value()) < params.error
//            || k == max_iter - 1)
        ) {
      ArnoldiRes<TenElemT, QNT> res;
      res.eigenvalue = eigen_solution.eigen_value;
      LinearCombine(k, eigen_solution.right_eigen_vec.data(), bases, TenElemT(0.0), &res.eig_vec);
      for (size_t i = 0; i < k + 1; i++) {
        delete bases[i];
        delete bases_dag[i];
      }
      if (k == max_iter - 1) {
        std::cout << "warning: arnoldi may not converge." << std::endl;
      }
      return res;
    }

//    if (eigen_solution.valid) {
//      eigenvalue_last = eigen_solution.eigen_value;
//    } else {
//      eigenvalue_last.reset();
//    }
  }
  // double element type but the dominant eigenvalues are complex number
  std::cout << "double element type but the dominant eigenvalues are complex number!" << std::endl;
  std::cout << "suggestion: 1. try good initial vector 2. try complex number code. " << std::endl;
  exit(2);
}

}//qlpeps

#endif //VMC_PEPS_TWO_DIM_TN_PEPS_ARNOLDI_SOLVER_H

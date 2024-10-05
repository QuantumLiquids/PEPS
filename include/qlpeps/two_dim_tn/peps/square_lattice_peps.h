// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: QuantumLiquids/VMC-SquareLatticePEPS project. The SquareLatticePEPS class.
*/


#ifndef QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_H
#define QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_H

#include "qlten/qlten.h"
#include "qlmps/utilities.h"                          //CreatPath
#include "qlpeps/two_dim_tn/framework/ten_matrix.h"
#include "qlpeps/two_dim_tn/framework/site_idx.h"
#include "qlpeps/consts.h"                            //kPepsPath
#include "qlpeps/two_dim_tn/tps/tps.h"                //ToTPS()
#include "qlpeps/basic.h"                             //BondOrientation
#include "qlpeps/utility/conjugate_gradient_solver.h"
#include "qlpeps/algorithm/vmc_update/vmc_optimize_para.h"  //ConjugateGradientParams
#include "arnoldi_solver.h"
namespace qlpeps {
using namespace qlten;

template<typename QNT>
using HilbertSpaces = std::vector<std::vector<Index<QNT>>>;
//Inner vector indices correspond to column indices
//Direction out

struct SimpleUpdateTruncatePara {
  size_t D_min;
  size_t D_max;
  double trunc_err;

  SimpleUpdateTruncatePara(size_t d_min, size_t d_max, double trunc_error)
      : D_min(d_min), D_max(d_max), trunc_err(trunc_error) {}
};

struct FullEnvironmentTruncateParams {
  FullEnvironmentTruncateParams(
      const size_t Dmin, const size_t Dmax, const double trunc_err,
      const double fet_tolerance, const size_t fet_max_iter,
      const ConjugateGradientParams &conjugate_gradient_params
  ) : Dmin(Dmin), Dmax(Dmax), trunc_err(trunc_err),
      tolerance(fet_tolerance), max_iter(fet_max_iter), cg_params(conjugate_gradient_params) {}
  size_t Dmin;
  size_t Dmax;
  double trunc_err;

  double tolerance;
  size_t max_iter;
  ConjugateGradientParams cg_params;
};

struct LoopUpdateTruncatePara {
  LoopUpdateTruncatePara(const ArnoldiParams &arnoldi_params,
                         const double inv_tol,
                         const FullEnvironmentTruncateParams &fet_params) :
      arnoldi_params(arnoldi_params), inv_tol(inv_tol), fet_params(fet_params) {}
  //gauge fixing
  ArnoldiParams arnoldi_params;
  double inv_tol;
  //full environment truncation
  FullEnvironmentTruncateParams fet_params;
};

template<typename ElemT>
struct ProjectionRes {
  double norm;      // normalization factor
  double trunc_err; // actual truncation error
  size_t D;         // actual bond dimension
  std::optional<ElemT> e_loc;    // local energy
};

/**
 *           3
 *           |
 *   0--Gamma[rows_][cols_]--2,   also contain physical index 4
 *           |
 *           1
 *
 *
 *   0--Lambda[rows_][cols_]--1
 *
 *
 *       0
 *       |
 *   Lambda[rows_][cols_]
 *       |
 *       1
 * @tparam TenElemT
 * @tparam QNT
 */
template<typename TenElemT, typename QNT>
class SquareLatticePEPS {
 public:
  using TenT = QLTensor<TenElemT, QNT>;
  using DTensor = QLTensor<QLTEN_Double, QNT>;

//  // Constructor with size
//  SquareLatticePEPS(size_t rows, size_t cols) : rows_(rows), cols_(cols), Gamma(rows, cols), lambda_vert(rows + 1, cols),
//                                   lambda_horiz(rows, cols + 1) {}
  //Constructors
  SquareLatticePEPS(const HilbertSpaces<QNT> &hilbert_spaces);

  SquareLatticePEPS(const Index<QNT> &local_hilbert_space, size_t rows, size_t cols);

  // Copy constructor
  SquareLatticePEPS(const SquareLatticePEPS<TenElemT, QNT> &rhs)
      : rows_(rhs.rows_), cols_(rhs.cols_), Gamma(rhs.Gamma), lambda_vert(rhs.lambda_vert),
        lambda_horiz(rhs.lambda_horiz) {}

  // Move constructor
  SquareLatticePEPS(SquareLatticePEPS<TenElemT, QNT> &&rhs) noexcept
      : rows_(rhs.rows_), cols_(rhs.cols_), Gamma(std::move(rhs.Gamma)), lambda_vert(std::move(rhs.lambda_vert)),
        lambda_horiz(std::move(rhs.lambda_horiz)) {}

  // Assignment operator
  SquareLatticePEPS<TenElemT, QNT> &operator=(const SquareLatticePEPS<TenElemT, QNT> &rhs) {
    if (this != &rhs) {
      rows_ = rhs.rows_;
      cols_ = rhs.cols_;
      Gamma = rhs.Gamma;
      lambda_vert = rhs.lambda_vert;
      lambda_horiz = rhs.lambda_horiz;
    }
    return *this;
  }

  // Move assignment operator
  SquareLatticePEPS<TenElemT, QNT> &operator=(SquareLatticePEPS<TenElemT, QNT> &&rhs) noexcept {
    if (this != &rhs) {
      rows_ = rhs.rows_;
      cols_ = rhs.cols_;
      Gamma = std::move(rhs.Gamma);
      lambda_vert = std::move(rhs.lambda_vert);
      lambda_horiz = std::move(rhs.lambda_horiz);
    }
    return *this;
  }

  bool operator==(const SquareLatticePEPS<TenElemT, QNT> &rhs) const;

  bool operator!=(const SquareLatticePEPS<TenElemT, QNT> &rhs) const {
    return !(*this == rhs);
  }

  void Initial(std::vector<std::vector<size_t>> &activates);

  //Getters
  // Function to get the number of rows in the SquareLatticePEPS
  size_t Rows(void) const { return rows_; }

  // Function to get the number of columns in the SquareLatticePEPS
  size_t Cols(void) const { return cols_; }

  std::pair<size_t, size_t> GetMinMaxBondDim(void) const;

  size_t GetMaxBondDim(void) const;

  // if the bond dimensions of each lambda are the same, except boundary lambdas
  bool IsBondDimensionEven(void) const;

  double NormalizeAllTensor(void);

  // Projecting Gate Functions
  double SingleSiteProject(
      const TenT &gate_ten,
      const SiteIdx &site,
      const bool canonicalize
  );

  ///< first two indexes of gate_ten are connected to PEPS's physical indexes
  ProjectionRes<TenElemT> NearestNeighborSiteProject(
      const TenT &gate_ten,
      const SiteIdx &site,
      const BondOrientation &orientation,
      const SimpleUpdateTruncatePara &trunc_para,
      const TenT &ham_ten = TenT()
  );

  double NextNearestNeighborSiteProject(
      const TenT &gate_ten,
      const SiteIdx &first_site,
      const BondOrientation &orientation,
      const SimpleUpdateTruncatePara &trunc_para
  );

  ProjectionRes<TenElemT> UpperLeftTriangleProject(
      const TenT &,
      const SiteIdx &,
      const SimpleUpdateTruncatePara &
  );

  ProjectionRes<TenElemT> LowerRightTriangleProject(
      const TenT &,
      const SiteIdx &,
      const SimpleUpdateTruncatePara &tunc_para
  );

  double LowerLeftTriangleProject(
      const TenT &gate_ten,
      const SiteIdx &upper_left_site,
      const SimpleUpdateTruncatePara &trunc_para
  );
  using LocalSquareLoopGateT = std::array<TenT, 4>;
  std::pair<double, double> LocalSquareLoopProject(
      const LocalSquareLoopGateT &gate_tens,
      const SiteIdx &upper_left_site,
      const LoopUpdateTruncatePara &params,
      const bool print_time = false
  );

  bool Dump(const std::string path = kPepsPath) const;

  bool Dump(const std::string path = kPepsPath, bool release_mem = false);

  bool Load(const std::string path = kPepsPath);

  operator TPS<TenElemT, QNT>(void) const;

  TenMatrix<TenT> Gamma; // The rank-5 projection tensors;
  TenMatrix<DTensor> lambda_vert; // vertical singular value tensors;
  TenMatrix<DTensor> lambda_horiz; // horizontal singular value tensors;

 private:
  //Helper For Projecting Gate
  TenT EatSurroundLambdas_(const SiteIdx &site) const;

  TenT Eat3SurroundLambdas_(const SiteIdx &site, const BTenPOSITION leaving_post) const;

  TenT QTenSplitOutLambdas_(const TenT &q, const SiteIdx &site,
                            const BTenPOSITION leaving_post, double inv_tolerance) const;

  void PatSquareLocalLoopProjector_(
      const LocalSquareLoopGateT &gate_tens,
      const SiteIdx &upper_left_site
  );

  std::array<QLTensor<TenElemT, QNT>, 4> GetLoopGammas_(const SiteIdx upper_left_site) const;
  std::array<QLTensor<QLTEN_Double, QNT>, 4> GetLoopInternalLambdas_(const SiteIdx upper_left_site) const;
  std::pair<std::array<QLTensor<QLTEN_Double, QNT>, 4>, std::array<QLTensor<QLTEN_Double, QNT>, 4>>
  GetLoopEnvLambdas_(const SiteIdx upper_left_site) const;

  static const QNT qn0_;

  size_t rows_; // Number of rows in the SquareLatticePEPS
  size_t cols_; // Number of columns in the SquareLatticePEPS
};

template<typename TenElemT, typename QNT>
const QNT SquareLatticePEPS<TenElemT, QNT>::qn0_ = QNT::Zero();

}

#include "qlpeps/two_dim_tn/peps/square_lattice_peps_basic_impl.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps_projection_impl.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps_projection4_impl.h"

#endif //QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_H

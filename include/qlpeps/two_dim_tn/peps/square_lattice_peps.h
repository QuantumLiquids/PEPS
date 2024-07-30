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
#include "qlmps/algorithm/lanczos_params.h"           //LanczosParams
#include "qlpeps/two_dim_tn/framework/ten_matrix.h"
#include "qlpeps/two_dim_tn/framework/site_idx.h"
#include "qlpeps/consts.h"                            //kPepsPath
#include "qlpeps/two_dim_tn/tps/tps.h"                //ToTPS()
#include "qlpeps/basic.h"                             //BondOrientation
#include "qlpeps/utility/conjugate_gradient_solver.h"
#include "qlpeps/algorithm/vmc_update/vmc_optimize_para.h"  //ConjugateGradientParams

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

using qlmps::LanczosParams;
struct LoopUpdateTruncatePara {
  LoopUpdateTruncatePara(const LanczosParams &lanczos_params,
                         const size_t Dmin, const size_t Dmax, const double trunc_err,
                         const double fet_tolerance, const size_t fet_max_iter,
                         const ConjugateGradientParams &conjugate_gradient_params) :
      lanczos_params(lanczos_params), D_min(Dmin), D_max(Dmax), trunc_err(trunc_err),
      fet_tol(fet_tolerance), fet_max_iter(fet_max_iter), cg_params(conjugate_gradient_params) {}
  //gauge fixing
  LanczosParams lanczos_params;

  //full environment truncation
  size_t D_min;
  size_t D_max;
  double trunc_err;
  double fet_tol;
  size_t fet_max_iter;
  ConjugateGradientParams cg_params; //for FET
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

  // Projecting Gate Functions
  double SingleSiteProject(
      const TenT &gate_ten,
      const SiteIdx &site
  );

  ///< first two indexes of gate_ten are connected to PEPS's physical indexes
  double NearestNeighborSiteProject(
      const TenT &gate_ten,
      const SiteIdx &site,
      const BondOrientation &orientation,
      const SimpleUpdateTruncatePara &trunc_para
  );

  double NextNearestNeighborSiteProject(
      const TenT &gate_ten,
      const SiteIdx &first_site,
      const BondOrientation &orientation,
      const SimpleUpdateTruncatePara &trunc_para
  );

  double UpperLeftTriangleProject(
      const TenT &gate_ten,
      const SiteIdx &upper_left_site,
      const SimpleUpdateTruncatePara &trunc_para
  );

  double LowerRightTriangleProject(
      const TenT &gate_ten,
      const SiteIdx &upper_right_site,
      const SimpleUpdateTruncatePara &trunc_para
  );

  double LowerLeftTriangleProject(
      const TenT &gate_ten,
      const SiteIdx &upper_left_site,
      const SimpleUpdateTruncatePara &trunc_para
  );
  using LocalSquareLoopGateT = std::array<TenT, 4>;
  double LocalSquareLoopProject(
      const LocalSquareLoopGateT &gate_tens,
      const SiteIdx &upper_left_site,
      const LoopUpdateTruncatePara &params
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

  void WeightedTraceGaugeFixingInSquareLocalLoop_(
      const qlpeps::LoopUpdateTruncatePara &params,
      std::array<QLTensor<TenElemT, QNT>, 4> &gammas,
      std::array<QLTensor<TenElemT, QNT>, 4> &lambdas,
      std::array<QLTensor<TenElemT, QNT>, 4> &Upsilons
  );

  void FullEnvironmentTruncateInSquareLocalLoop_(
      const qlpeps::LoopUpdateTruncatePara &params,
      std::array<QLTensor<TenElemT, QNT>, 4> &gammas,
      std::array<QLTensor<TenElemT, QNT>, 4> &lambdas,
      std::array<QLTensor<TenElemT, QNT>, 4> &Upsilons
  );
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

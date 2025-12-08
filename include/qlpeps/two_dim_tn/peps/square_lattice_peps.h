// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: QuantumLiquids/PEPS project. Square Lattice PEPS class definition.
*/


#ifndef QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_H
#define QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_H

#include "qlten/qlten.h"
#include "qlmps/utilities.h"                          //CreatPath
#include "qlpeps/two_dim_tn/framework/ten_matrix.h"
#include "qlpeps/two_dim_tn/framework/site_idx.h"
#include "qlpeps/consts.h"                            //kPepsPath
#include "qlpeps/two_dim_tn/common/boundary_condition.h"
#include "qlpeps/two_dim_tn/tps/tps.h"                //ToTPS()
#include "qlpeps/basic.h"                             //BondOrientation
#include "qlpeps/utility/conjugate_gradient_solver.h"
#include "qlpeps/optimizer/optimizer_params.h"                   //ConjugateGradientParams
#include "arnoldi_solver.h"
namespace qlpeps {

template<typename QNT>
using HilbertSpaces = std::vector<std::vector<Index<QNT>>>;
//Inner vector indices correspond to column indices
//Direction out

struct SimpleUpdateTruncatePara {
  size_t D_min;
  size_t D_max;
  double trunc_err;
  double inv_tol;  ///< tolerance for diagonal matrix inversion

  /// Backward compatible constructor: inv_tol defaults to trunc_err
  SimpleUpdateTruncatePara(size_t d_min, size_t d_max, double trunc_error)
      : D_min(d_min), D_max(d_max), trunc_err(trunc_error), inv_tol(trunc_error) {}

  /// Full constructor with explicit inv_tol
  SimpleUpdateTruncatePara(size_t d_min, size_t d_max, double trunc_error, double inv_tolerance)
      : D_min(d_min), D_max(d_max), trunc_err(trunc_error), inv_tol(inv_tolerance) {}
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
  using RealT = typename qlten::RealTypeTrait<ElemT>::type;
  RealT norm;      // normalization factor
  RealT trunc_err; // actual truncation error
  size_t D;         // actual bond dimension
  std::optional<ElemT> e_loc;    // local energy
};

/**
 * PEPS Network Layout & Indexing
 * ==============================
 *
 * OBC Layout (Rows=2, Cols=2)
 * ---------------------------
 * The lambda indices follow the convention:
 * - lambda_vert[r][c] is the vertical bond ABOVE Gamma[r][c].
 * - lambda_vert[Rows][c] is the boundary bond BELOW Gamma[Rows-1][c].
 * - lambda_horiz[r][c] is the horizontal bond LEFT of Gamma[r][c].
 * - lambda_horiz[r][Cols] is the boundary bond RIGHT of Gamma[r][Cols-1].
 *
 *       lv(0,0)     lv(0,1)
 *          |           |
 * lh(0,0)--G(0,0)--lh(0,1)--G(0,1)--lh(0,2)
 *          |           |
 *       lv(1,0)     lv(1,1)
 *          |           |
 * lh(1,0)--G(1,0)--lh(1,1)--G(1,1)--lh(1,2)
 *          |           |
 *       lv(2,0)     lv(2,1)
 *
 * where G is short for Gamma, lv is short for lambda_vert, lh is short for lambda_horiz.
 *
 * PBC Layout (Rows=2, Cols=2)
 * ---------------------------
 * In Periodic Boundary Conditions, the grid wraps around.
 * - lambda_vert[r][c] is the vertical bond ABOVE Gamma[r][c].
 * - lambda_vert[0][c] connects Gamma[R-1][c] (bottom) to Gamma[0][c] (top).
 * - lambda_horiz[r][c] is the horizontal bond LEFT of Gamma[r][c].
 * - lambda_horiz[r][0] connects Gamma[r][C-1] (right) to Gamma[r][0] (left).
 *
 * Note:
 * - lv(0,0) is shared: it is the North bond of G(0,0) and South bond of G(1,0).
 * - lh(0,0) is shared: it is the West bond of G(0,0) and East bond of G(0,1).
 *
 *       lv(0,0)     lv(0,1)   <-- connected to bottom G(1,x)
 *          |           |
 * lh(0,0)--G(0,0)--lh(0,1)--G(0,1)--lh(0,0) (wraps)
 *          |           |
 *       lv(1,0)     lv(1,1)
 *          |           |
 * lh(1,0)--G(1,0)--lh(1,1)--G(1,1)--lh(1,0) (wraps)
 *          |           |
 *       lv(0,0)     lv(0,1) (wraps)
 *
 * Tensor Index Order:
 * -------------------
 * The order of indices for the tensors is as follows:
 *
 * Gamma[row][col]: (West, South, East, North, Physical) = (0, 1, 2, 3, 4)
 *          3 (North)
 *          |
 *          v
 *          |
 *  0 -->-- Gamma -->-- 2 (East)
 * (West)   |
 *          v
 *          |
 *          1 (South)
 * (The direction is obeyed in constructor, but not guaranteed in projection functions.)
 *
 * lambda_vert[row][col]: (Up, Down) = (0, 1)
 *          0 (Up, connects to Gamma South)
 *          |
 *       lambda
 *          |
 *          1 (Down, connects to Gamma North)
 *
 * lambda_horiz[row][col]: (Left, Right) = (0, 1)
 *  0 (Left) -- lambda -- 1 (Right)
 *
 * @tparam TenElemT
 * @tparam QNT
 */
template<typename TenElemT, typename QNT>
class SquareLatticePEPS {
 public:
  using TenT = QLTensor<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  using DTenT = QLTensor<RealT, QNT>;

//  // Constructor with size
//  SquareLatticePEPS(size_t rows, size_t cols) : rows_(rows), cols_(cols), Gamma(rows, cols), lambda_vert(rows + 1, cols),
//                                   lambda_horiz(rows, cols + 1) {}
  //Constructors
  SquareLatticePEPS(const HilbertSpaces<QNT> &hilbert_spaces, BoundaryCondition bc = BoundaryCondition::Open);

  SquareLatticePEPS(const Index<QNT> &local_hilbert_space, size_t rows, size_t cols, BoundaryCondition bc = BoundaryCondition::Open);

  // Copy constructor
  SquareLatticePEPS(const SquareLatticePEPS<TenElemT, QNT> &rhs) = default;

  // Move constructor
  SquareLatticePEPS(SquareLatticePEPS<TenElemT, QNT> &&rhs) noexcept
      : rows_(rhs.rows_), cols_(rhs.cols_), boundary_condition_(rhs.boundary_condition_), Gamma(std::move(rhs.Gamma)), lambda_vert(std::move(rhs.lambda_vert)),
        lambda_horiz(std::move(rhs.lambda_horiz)) {}

  // Assignment operator
  SquareLatticePEPS<TenElemT, QNT> &operator=(const SquareLatticePEPS<TenElemT, QNT> &rhs) = default;

  // Move assignment operator
  SquareLatticePEPS<TenElemT, QNT> &operator=(SquareLatticePEPS<TenElemT, QNT> &&rhs) noexcept {
    if (this != &rhs) {
      rows_ = rhs.rows_;
      cols_ = rhs.cols_;
      boundary_condition_ = rhs.boundary_condition_;
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
  BoundaryCondition GetBoundaryCondition() const { return boundary_condition_; }

  // Boundary-aware Lambda Accessors
  // -------------------------------

  // Get the vertical lambda tensor ABOVE the site (row, col)
  const DTenT& GetLambdaVertNorth(size_t row, size_t col) const {
    return lambda_vert({row, col});
  }
  DTenT& GetLambdaVertNorth(size_t row, size_t col) {
    return lambda_vert({row, col});
  }

  // Get the vertical lambda tensor BELOW the site (row, col)
  const DTenT& GetLambdaVertSouth(size_t row, size_t col) const {
    if (boundary_condition_ == BoundaryCondition::Open) {
      return lambda_vert({row + 1, col});
    } else {
      return lambda_vert({(row + 1) % rows_, col});
    }
  }
  DTenT& GetLambdaVertSouth(size_t row, size_t col) {
    if (boundary_condition_ == BoundaryCondition::Open) {
      return lambda_vert({row + 1, col});
    } else {
      return lambda_vert({(row + 1) % rows_, col});
    }
  }

  // Get the horizontal lambda tensor to the LEFT of the site (row, col)
  const DTenT& GetLambdaHorizWest(size_t row, size_t col) const {
    return lambda_horiz({row, col});
  }
  DTenT& GetLambdaHorizWest(size_t row, size_t col) {
    return lambda_horiz({row, col});
  }

  // Get the horizontal lambda tensor to the RIGHT of the site (row, col)
  const DTenT& GetLambdaHorizEast(size_t row, size_t col) const {
    if (boundary_condition_ == BoundaryCondition::Open) {
      return lambda_horiz({row, col + 1});
    } else {
      return lambda_horiz({row, (col + 1) % cols_});
    }
  }
  DTenT& GetLambdaHorizEast(size_t row, size_t col) {
    if (boundary_condition_ == BoundaryCondition::Open) {
      return lambda_horiz({row, col + 1});
    } else {
      return lambda_horiz({row, (col + 1) % cols_});
    }
  }

  // Function to get the number of rows in the SquareLatticePEPS
  size_t Rows(void) const { return rows_; }

  // Function to get the number of columns in the SquareLatticePEPS
  size_t Cols(void) const { return cols_; }

  std::pair<size_t, size_t> GetMinMaxBondDim(void) const;

  size_t GetMaxBondDim(void) const;

  // if the bond dimensions of each lambda are the same, except boundary lambdas
  bool IsBondDimensionUniform(void) const;

  RealT NormalizeAllTensor(void);

  /// useful when in debug.
  IndexVec<QNT> GatherAllIndices(void) const;

  ///< first two indexes of gate_ten are connected to PEPS's physical indexes
  ProjectionRes<TenElemT> NearestNeighborSiteProject(
      const TenT &gate_ten,
      const SiteIdx &site,
      const BondOrientation &orientation,
      const SimpleUpdateTruncatePara &trunc_para,
      const TenT &ham_ten = TenT()
  );

  RealT NextNearestNeighborSiteProject(
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
      const TenT &gate_ten,
      const SiteIdx &upper_site,
      const SimpleUpdateTruncatePara &tunc_para,
      const TenT &ham_ten = TenT()
  );

  ProjectionRes<TenElemT> LowerLeftTriangleProject(
      const TenT &gate_ten,
      const SiteIdx &upper_left_site,
      const SimpleUpdateTruncatePara &trunc_para,
      const TenT &ham_ten = TenT()
  );

  ///< fix the convenction of index direction: from left to right, from up to down. For the convention of index direction may be broken after loop update.
  void RegularizeIndexDir();

  using LocalSquareLoopGateT = std::array<TenT, 4>;
  std::pair<RealT, RealT> LocalSquareLoopProject(
      const LocalSquareLoopGateT &gate_tens,
      const SiteIdx &upper_left_site,
      const LoopUpdateTruncatePara &params,
      const bool print_time = false
  );

  bool Dump(const std::string path = kPepsPath) const;

  bool Dump(const std::string path = kPepsPath, bool release_mem = false);

  bool Load(const std::string path = kPepsPath);

  /**
   * @brief Convert this PEPS to a TPS defined on the same lattice.
   *
   * The returned TPS carries the physical index structure of the current PEPS
   * and the surrounding singular-value tensors folded into each local tensor.
   *
   * @return Tensor product state representing the current PEPS configuration.
   */
  TPS<TenElemT, QNT> ToTPS() const;

  [[deprecated("Use ToTPS(peps) in qlpeps::api::conversions instead")]]
  operator TPS<TenElemT, QNT>(void) const;

  TenMatrix<TenT> Gamma; // The rank-5 projection tensors;
  TenMatrix<DTenT> lambda_vert; // vertical singular value tensors;
  TenMatrix<DTenT> lambda_horiz; // horizontal singular value tensors;

 private:
  //Helper For Projecting Gate
  TenT EatSurroundLambdas_(const SiteIdx &site) const;

  /**
   * @brief Absorbs 3 surrounding Lambda tensors into the Gamma tensor.
   *
   * Contracts the Gamma tensor at `site` with the Lambda tensors on three sides,
   * leaving the bond in the `leaving_post` direction open (i.e., not contracted with its Lambda).
   * This is typically used to prepare the effective tensor for the QR decomposition step in Simple Update,
   * where the open bond connects to the bond being updated.
   *
   * @param site The coordinates of the site.
   * @param leaving_post The direction where the Lambda tensor is NOT absorbed.
   * @return A rank-5 tensor formed by contracting Gamma with 3 Lambdas.
   */
  TenT Eat3SurroundLambdas_(const SiteIdx &site, const BTenPOSITION leaving_post) const;

  TenT QTenSplitOutLambdas_(const TenT &q, const SiteIdx &site,
                            const BTenPOSITION leaving_post, RealT inv_tolerance) const;

  void PatSquareLocalLoopProjector_(
      const LocalSquareLoopGateT &gate_tens,
      const SiteIdx &upper_left_site
  );

  std::array<QLTensor<TenElemT, QNT>, 4> GetLoopGammas_(const SiteIdx upper_left_site) const;
  std::array<QLTensor<RealT, QNT>, 4> GetLoopInternalLambdas_(const SiteIdx upper_left_site) const;
  std::pair<std::array<QLTensor<RealT, QNT>, 4>, std::array<QLTensor<RealT, QNT>, 4>>
  GetLoopEnvLambdas_(const SiteIdx upper_left_site) const;

  static const QNT qn0_;

  size_t rows_; // Number of rows in the SquareLatticePEPS
  size_t cols_; // Number of columns in the SquareLatticePEPS
  BoundaryCondition boundary_condition_;
};

template<typename TenElemT, typename QNT>
const QNT SquareLatticePEPS<TenElemT, QNT>::qn0_ = QNT::Zero();

}

#include "qlpeps/two_dim_tn/peps/square_lattice_peps_basic_impl.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps_projection_impl.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps_projection4_impl.h"

#endif //QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_H

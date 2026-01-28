/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-01
*
* Description: QuantumLiquids/PEPS project. Boundary MPS
*/

#ifndef QLPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_H
#define QLPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_H

#include <limits>                               // std::numeric_limits
#include <optional>                             //std::optional<T>
#include "qlten/qlten.h"
#include "qlmps/one_dim_tn/framework/ten_vec.h"
#include "qlmps/one_dim_tn/mps/finite_mps/finite_mps.h"
#include "qlmps/one_dim_tn/mpo/mpo.h"
#include "qlpeps/basic.h"                       //BMPSPOSITION
//if above include doesn't work, include mps_all.h

namespace qlpeps {
using qlten::Index;
using qlten::QNSector;
using qlten::QLTensor;
using qlmps::TenVec;
using qlmps::MPSTenCanoType;
using qlmps::kUncentralizedCenterIdx;
using qlmps::IN;
using qlmps::OUT;
using qlmps::NONE;

enum class CompressMPSScheme {
  SVD_COMPRESS,
  VARIATION2Site,
  VARIATION1Site
};

// Convert enum class to descriptive string
std::string CompressMPSSchemeString(CompressMPSScheme scheme) {
  switch (scheme) {
    case CompressMPSScheme::SVD_COMPRESS:return "SVD Compression";
    case CompressMPSScheme::VARIATION2Site:return "Two-Site Variational Compression";
    case CompressMPSScheme::VARIATION1Site:return "Single-Site Variational Compression";
    default:return "Unknown compression scheme";
  }
}

template<typename TenElemT>
struct BMPSTruncateParams {
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  // Defaults mirror TRGTruncateParams: safe to default-construct and yields "no truncation".
  size_t D_min = 1;
  size_t D_max = std::numeric_limits<size_t>::max();
  RealT trunc_err = RealT(0);
  CompressMPSScheme compress_scheme = CompressMPSScheme::SVD_COMPRESS;
  std::optional<RealT> convergence_tol = std::nullopt;
  std::optional<size_t> iter_max = std::nullopt;

  BMPSTruncateParams(void) = default;

  BMPSTruncateParams(size_t d_min, size_t d_max, RealT trunc_error,
                     CompressMPSScheme compress_scheme,
                     std::optional<RealT> convergence_tol,
                     std::optional<size_t> iter_max)
      : D_min(d_min), D_max(d_max), trunc_err(trunc_error), compress_scheme(compress_scheme),
        convergence_tol(convergence_tol), iter_max(iter_max) {}

  // Convenience constructor for SVD compression (variational parameters unused)
  BMPSTruncateParams(size_t d_min, size_t d_max, RealT trunc_error,
                     std::enable_if_t<true, CompressMPSScheme> compress_scheme)
      : D_min(d_min), D_max(d_max), trunc_err(trunc_error),
        compress_scheme(compress_scheme), convergence_tol(std::nullopt), iter_max(std::nullopt) {}

  // Static factories to avoid misuse and clarify intent
  // SVD compression: only D_min, D_max, trunc_err are relevant
  static BMPSTruncateParams SVD(size_t d_min, size_t d_max, RealT trunc_error) {
    return BMPSTruncateParams(d_min, d_max, trunc_error,
                              CompressMPSScheme::SVD_COMPRESS,
                              std::nullopt, std::nullopt);
  }

  // Two-site variational compression: require convergence_tol and iter_max
  static BMPSTruncateParams Variational2Site(size_t d_min, size_t d_max, RealT trunc_error,
                                             RealT convergence_tol, size_t iter_max) {
    return BMPSTruncateParams(d_min, d_max, trunc_error,
                              CompressMPSScheme::VARIATION2Site,
                              std::make_optional<RealT>(convergence_tol),
                              std::make_optional<size_t>(iter_max));
  }

  // One-site variational compression: require convergence_tol and iter_max
  static BMPSTruncateParams Variational1Site(size_t d_min, size_t d_max, RealT trunc_error,
                                             RealT convergence_tol, size_t iter_max) {
    return BMPSTruncateParams(d_min, d_max, trunc_error,
                              CompressMPSScheme::VARIATION1Site,
                              std::make_optional<RealT>(convergence_tol),
                              std::make_optional<size_t>(iter_max));
  }
};

// Backward compatibility:
// - BMPSTruncatePara is deprecated alias of double-specialized BMPSTruncateParams
using BMPSTruncatePara [[deprecated("Use BMPSTruncateParams<> instead")]] = BMPSTruncateParams<qlten::QLTEN_Double>;

/**
 * Boundary Matrix Product State class which is used in the contraction of single-layer 2D tensor network.
 *
 * For bosonic single-layer 2D tensor network, each tensor has four-leg,
 * so that the corresponding bmps tensor has the following form with index order :
 *      1
 *      |
 *  0---T---2
 *
 *  For fermionic single-layer 2D tensor network, each tensor has 5 legs, with the last index a trivial index
 *  to match the evenness of the tensors. The corresponding bmps tensor has 4 legs and the following index order :
 *
 *      1
 *      |
 *  0---T---2
 *      |
 *      4
 *  where leg 4 the trivial index.
 *
 *
 *  boundary-MPS tensor order :
 *                          UP
 *             7----6---5---4---3---2---1---0
 *             |    |   |   |   |   |   |   |
 *
 *             |    |   |   |   |   |   |   |
 *      0--- ----------------------------------- ---4
 *      |      |    |   |   |   |   |   |   |       |
 *      1--- ----------------------------------- ---3
 *      |      |    |   |   |   |   |   |   |       |
 * LEFT 2--- ----------------------------------- ---2 RIGHT
 *      |      |    |   |   |   |   |   |   |       |
 *      3--- ----------------------------------- ---1
 *      |      |    |   |   |   |   |   |   |       |
 *      4--- ----------------------------------- ---0
 *             |    |   |   |   |   |   |   |
 *
 *             |    |   |   |   |   |   |   |
 *             0----1---2---3---4---5---6---7
 *                      DOWN
 *
 * @warning CRITICAL STORAGE CONVENTION:
 * The physical storage order varies by position:
 * - UP/RIGHT: Reversed order. bmps[0] = rightmost/bottommost column, bmps[i] = col N-1-i
 * - DOWN/LEFT: Natural order. bmps[0] = leftmost/topmost column, bmps[i] = col i
 *
 * This convention is internally handled by AlignTransferMPOTensorOrder_() during MultiplyMPO.
 * Use AtLogicalCol() for position-independent access by logical column index.
 */
template<typename TenElemT, typename QNT>
class BMPS : public TenVec<QLTensor<TenElemT, QNT>> {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  using DTenT = QLTensor<RealT, QNT>;
  using IndexT = Index<QNT>;
  using TransferMPO = std::vector<Tensor *>;

  BMPS(const BMPSPOSITION position, const size_t size) : TenVec<Tensor>(size), position_(position),
                                                         center_(kUncentralizedCenterIdx),
                                                         tens_cano_type_(size, NONE) {}

  /**
   * Initialize the MPS as direct product state
   *
   * @param position
   * @param size
   * @param local_hilbert_space local_hilbert_space.dim() == 1
   */
  BMPS(const BMPSPOSITION position, const size_t size, const IndexT &local_hilbert_space);

  BMPS(const BMPSPOSITION position, const std::vector<IndexT> &hilbert_spaces);

  BMPS(const BMPS<TenElemT, QNT> &rhs) : TenVec<QLTensor<TenElemT, QNT>>(rhs),
                                         position_(rhs.position_),
                                         center_(rhs.center_),
                                         tens_cano_type_(rhs.tens_cano_type_) {}

  BMPS &operator=(const BMPS<TenElemT, QNT> &rhs) {
    assert(position_ == rhs.position_);
    TenVec<QLTensor<TenElemT, QNT>>::operator=(rhs);
    center_ = rhs.center_;
    tens_cano_type_ = rhs.tens_cano_type_;
    return *this;
  }

  // MPS local tensor access, set function
  Tensor &operator[](const size_t idx);

  // get function
  const Tensor &operator[](const size_t idx) const;

  // set function
  Tensor *&operator()(const size_t idx);

  // get function
  const Tensor *operator()(const size_t idx) const;

  /**
   * @brief Access tensor by logical column index (position-independent).
   *
   * Handles the storage convention difference between UP/RIGHT (reversed)
   * and DOWN/LEFT (natural order).
   *
   * @param col Logical column index in the 2D grid (0 = leftmost/topmost)
   * @return Reference to the tensor at logical column col
   *
   * @note For UP/RIGHT: returns (*this)[size()-1-col]
   *       For DOWN/LEFT: returns (*this)[col]
   */
  const Tensor &AtLogicalCol(size_t col) const {
    return (position_ == UP || position_ == RIGHT)
           ? (*this)[this->size() - 1 - col]
           : (*this)[col];
  }

  /**
   * @brief Mutable version of AtLogicalCol.
   */
  Tensor &AtLogicalCol(size_t col) {
    return (position_ == UP || position_ == RIGHT)
           ? (*this)[this->size() - 1 - col]
           : (*this)[col];
  }

  BMPSPOSITION Direction() const { return position_; }
  // MPS global operations
  void Centralize(const int);

  // MPS partial global operations.
  void LeftCanonicalize(const size_t);

  void RightCanonicalize(const size_t);

  // MPS local operations. Only tensors near the target site are needed in memory.
  void LeftCanonicalizeTen(const size_t);

  qlten::QLTensor<RealT, QNT> RightCanonicalizeTen(const size_t);

  //return (D, trunc_err)
  std::pair<size_t, RealT> RightCanonicalizeTruncate(const size_t, const size_t, const size_t, const RealT);

  int GetCenter(void) const { return center_; }

  std::vector<MPSTenCanoType> GetTensCanoType(void) const {
    return tens_cano_type_;
  }

  MPSTenCanoType GetTenCanoType(const size_t idx) const {
    return tens_cano_type_[idx];
  }

  std::vector<RealT> GetEntanglementEntropy(size_t n);

  void Reverse();

  void InplaceMultiplyMPO(TransferMPO &, const size_t, const size_t, const RealT,
                          const size_t max_iter, //only valid for variational methods
                          const CompressMPSScheme &scheme);

  /**
   * @note mpo will be reversed if the position is RIGHT or UP.
   * @param max_iter
   * @param scheme
   * @return
   */
  BMPS MultiplyMPO(TransferMPO &, const CompressMPSScheme &,
                   const size_t, const size_t, const RealT,
                   const std::optional<RealT> variational_converge_tol,//only valid for variational methods
                   const std::optional<size_t> max_iter
  ) const;

  BMPS MultiplyMPOWithPhyIdx(TransferMPO &, const size_t, const size_t, const RealT,
                             const size_t max_iter, //only valid for variational methods
                             const CompressMPSScheme &) const;

 private:

  /**
 * reverse mpo tensor order if the position is RIGHT or UP, so that it is aligned with boundary-MPS tensor order.
 */
  void AlignTransferMPOTensorOrder_(TransferMPO &) const;

  BMPS MultiplyMPOSVDCompress_(const TransferMPO &,
                               const size_t, const size_t, const RealT,
                               size_t &actual_Dmax, RealT &actual_trunc_err_max) const;

  BMPS MultiplyMPO2SiteVariationalCompress_(const TransferMPO &, const size_t, const size_t, const RealT,
                                            const RealT variational_converge_tol, const size_t max_iter) const;

  // strictly, 1-site variational compress method is only suitable for the cases those tensors have no symmetry constrain.
  BMPS MultiplyMPO1SiteVariationalCompress_(const TransferMPO &, const size_t, const size_t, const RealT,
                                            const RealT variational_converge_tol, const size_t max_iter) const;
  BMPS InitGuessForVariationalMPOMultiplication_(const TransferMPO &, const size_t, const size_t, const RealT) const;

  // todo code.
  RealT RightCanonicalizeTruncateWithPhyIdx_(const size_t, const size_t, const size_t, const RealT);

  // todo code.
  BMPS
  InitGuessForVariationalMPOMultiplicationWithPhyIdx_(const TransferMPO &,
                                                      const size_t,
                                                      const size_t,
                                                      const RealT) const;

  const BMPSPOSITION
      position_; //possible to remove this member and replace it with function parameter if the function needs
  int center_;
  std::vector<MPSTenCanoType> tens_cano_type_;

  static const QNT qn0_;
  static const IndexT index0_in_;
  static const IndexT index0_out_;
};

template<typename TenElemT, typename QNT>
const QNT BMPS<TenElemT, QNT>::qn0_ = QNT::Zero();

template<typename TenElemT, typename QNT>
const Index<QNT> BMPS<TenElemT, QNT>::index0_in_ = Index<QNT>({QNSector(qn0_, 1)}, IN);

template<typename TenElemT, typename QNT>
const Index<QNT> BMPS<TenElemT, QNT>::index0_out_ = Index<QNT>({QNSector(qn0_, 1)}, OUT);

}//qlpeps

#include "qlpeps/ond_dim_tn/boundary_mps/bmps_impl.h"

#endif //QLPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_H

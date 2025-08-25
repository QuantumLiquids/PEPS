/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-01
*
* Description: QuantumLiquids/PEPS project. Boundary MPS
*/

#ifndef QLPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_H
#define QLPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_H

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
using qlten::QLTEN_Double;
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

struct BMPSTruncatePara {
  size_t D_min;
  size_t D_max;
  double trunc_err;
  CompressMPSScheme compress_scheme;
  std::optional<double> convergence_tol;
  std::optional<size_t> iter_max;

  BMPSTruncatePara(void) = default;

  BMPSTruncatePara(size_t d_min, size_t d_max, double trunc_error,
                   CompressMPSScheme compress_scheme,
                   std::optional<double> convergence_tol,
                   std::optional<size_t> iter_max)
      : D_min(d_min), D_max(d_max), trunc_err(trunc_error), compress_scheme(compress_scheme),
        convergence_tol(convergence_tol), iter_max(iter_max) {}
};

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
 */
template<typename TenElemT, typename QNT>
class BMPS : public TenVec<QLTensor<TenElemT, QNT>> {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using DTenT = QLTensor<QLTEN_Double, QNT>;
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

  BMPSPOSITION Direction() const { return position_; }
  // MPS global operations
  void Centralize(const int);

  // MPS partial global operations.
  void LeftCanonicalize(const size_t);

  void RightCanonicalize(const size_t);

  // MPS local operations. Only tensors near the target site are needed in memory.
  void LeftCanonicalizeTen(const size_t);

  qlten::QLTensor<qlten::QLTEN_Double, QNT> RightCanonicalizeTen(const size_t);

  //return (D, trunc_err)
  std::pair<size_t, double> RightCanonicalizeTruncate(const size_t, const size_t, const size_t, const double);

  int GetCenter(void) const { return center_; }

  std::vector<MPSTenCanoType> GetTensCanoType(void) const {
    return tens_cano_type_;
  }

  MPSTenCanoType GetTenCanoType(const size_t idx) const {
    return tens_cano_type_[idx];
  }

  std::vector<double> GetEntanglementEntropy(size_t n);

  void Reverse();

  void InplaceMultipleMPO(TransferMPO &, const size_t, const size_t, const double,
                          const size_t max_iter, //only valid for variational methods
                          const CompressMPSScheme &scheme);

  /**
   * @note mpo will be reversed if the position is RIGHT or UP.
   * @param max_iter
   * @param scheme
   * @return
   */
  BMPS MultipleMPO(TransferMPO &, const CompressMPSScheme &,
                   const size_t, const size_t, const double,
                   const std::optional<double> variational_converge_tol,//only valid for variational methods
                   const std::optional<size_t> max_iter
  ) const;

  BMPS MultipleMPOWithPhyIdx(TransferMPO &, const size_t, const size_t, const double,
                             const size_t max_iter, //only valid for variational methods
                             const CompressMPSScheme &) const;

 private:

  /**
 * reverse mpo tensor order if the position is RIGHT or UP, so that it is aligned with boundary-MPS tensor order.
 */
  void AlignTransferMPOTensorOrder_(TransferMPO &) const;

  BMPS MultipleMPOSVDCompress_(const TransferMPO &,
                               const size_t, const size_t, const double,
                               size_t &actual_Dmax, double &actual_trunc_err_max) const;

  BMPS MultipleMPO2SiteVariationalCompress_(const TransferMPO &, const size_t, const size_t, const double,
                                            const double variational_converge_tol, const size_t max_iter) const;

  // strictly, 1-site variational compress method is only suitable for the cases those tensors have no symmetry constrain.
  BMPS MultipleMPO1SiteVariationalCompress_(const TransferMPO &, const size_t, const size_t, const double,
                                            const double variational_converge_tol, const size_t max_iter) const;
  BMPS InitGuessForVariationalMPOMultiplication_(const TransferMPO &, const size_t, const size_t, const double) const;

  // todo code.
  double RightCanonicalizeTruncateWithPhyIdx_(const size_t, const size_t, const size_t, const double);

  // todo code.
  BMPS
  InitGuessForVariationalMPOMultiplicationWithPhyIdx_(const TransferMPO &,
                                                      const size_t,
                                                      const size_t,
                                                      const double) const;

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

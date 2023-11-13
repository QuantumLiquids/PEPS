/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-01
*
* Description: GraceQ/VMC-PEPS project. Boundary MPS
*/

#ifndef GQPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_H
#define GQPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_H

#include "gqten/gqten.h"
#include "gqmps2/one_dim_tn/framework/ten_vec.h"
#include "gqmps2/one_dim_tn/mps/finite_mps/finite_mps.h"
#include "gqmps2/one_dim_tn/mpo/mpo.h"
#include "gqpeps/basic.h"                       //BMPSPOSITION
//if above include doesn't work, include mps_all.h

namespace gqpeps {
using namespace gqten;
using namespace gqmps2;

enum CompressMPSScheme {
  SVD_COMPRESS,
  VARIATION2Site,
  VARIATION1Site
};

/**
 *      1
 *      |
 *  0---T---2
 * @tparam TenElemT
 * @tparam QNT
 */
template<typename TenElemT, typename QNT>
class BMPS : public TenVec<GQTensor<TenElemT, QNT>> {
 public:
  using Tensor = GQTensor<TenElemT, QNT>;
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

  BMPS(const BMPS<TenElemT, QNT> &rhs) : TenVec<GQTensor<TenElemT, QNT>>(rhs),
                                         position_(rhs.position_),
                                         center_(rhs.center_),
                                         tens_cano_type_(rhs.tens_cano_type_) {}

  BMPS &operator=(const BMPS<TenElemT, QNT> &rhs) {
    assert(position_ == rhs.position_);
    TenVec<GQTensor<TenElemT, QNT>>::operator=(rhs);
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

  // MPS global operations
  void Centralize(const int);

  // MPS partial global operations.
  void LeftCanonicalize(const size_t);

  void RightCanonicalize(const size_t);

  // MPS local operations. Only tensors near the target site are needed in memory.
  void LeftCanonicalizeTen(const size_t);

  GQTensor<GQTEN_Double, QNT> RightCanonicalizeTen(const size_t);

  double RightCanonicalizeTruncate(const size_t, const size_t, const size_t, const double);

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
                          const size_t max_iter = 5, //only valid for variational methods
                          const CompressMPSScheme &scheme = VARIATION2Site);
  /**
   * @note For SVD compress, mpo does not change after Multiplication
   *       For Variational methods, mpo may reverse.
   * @param max_iter
   * @param scheme
   * @return
   */
  BMPS MultipleMPO(TransferMPO &, const size_t, const size_t, const double,
                   const size_t max_iter = 5, //only valid for variational methods
                   const CompressMPSScheme &scheme = VARIATION2Site) const;

  BMPS MultipleMPOWithPhyIdx(TransferMPO &, const size_t, const size_t, const double,
                             const size_t max_iter = 5, //only valid for variational methods
                             const CompressMPSScheme &scheme = VARIATION2Site) const;

 private:
  BMPS InitGuessForVariationalMPOMultiplication_(TransferMPO &, const size_t, const size_t, const double) const;

  double RightCanonicalizeTruncateWithPhyIdx_(const size_t, const size_t, const size_t, const double);

  BMPS
  InitGuessForVariationalMPOMultiplicationWithPhyIdx_(TransferMPO &, const size_t, const size_t, const double) const;

  const BMPSPOSITION position_; //possible to remove this member and replace it with function parameter if the function needs
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

}//gqpeps

#include "gqpeps/ond_dim_tn/boundary_mps/bmps_impl.h"

#endif //GQPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_H

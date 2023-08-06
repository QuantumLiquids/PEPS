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
//if above include doesn't work, include mps_all.h

namespace gqpeps {
using namespace gqten;
using namespace gqmps2;

///<
/**from which direction, or the position
 *
 * UP:   MPS tensors are numbered from right to left
               2--t--0
                 |
                 1

 * DOWN: MPS tensors are numbered from left to right
                   1
                   |
                0--t--2

 * LEFT: MPS tensors are numbered from up to down;
 */
enum BMPSPOSITION {
  UP,
  DOWN,
  LEFT,
  RIGHT
};

enum CompressMPSScheme {
  SVD_COMPRESS,
  VARIATION
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
  using LocalTenT = GQTensor<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  using TransferMPO = std::vector<LocalTenT *>;
 public:

  BMPS(const BMPSPOSITION position, const size_t size) : TenVec<LocalTenT>(size), position_(position),
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

  // MPS local tensor access, set function
  LocalTenT &operator[](const size_t idx);
  // get function
  const LocalTenT &operator[](const size_t idx) const;
  // set function
  LocalTenT *&operator()(const size_t idx);
  // get function
  const LocalTenT *operator()(const size_t idx) const;

  // MPS global operations
  void Centralize(const int);

  // MPS partial global operations.
  void LeftCanonicalize(const size_t);
  void RightCanonicalize(const size_t);
  double RightCanonicalizeTrunctate(const size_t,
                                    const GQTEN_Double trunc_err,
                                    const size_t Dmin,
                                    const size_t Dmax);

  // MPS local operations. Only tensors near the target site are needed in memory.
  void LeftCanonicalizeTen(const size_t);
  GQTensor<GQTEN_Double, QNT> RightCanonicalizeTen(const size_t);

  int GetCenter(void) const { return center_; }

  std::vector<MPSTenCanoType> GetTensCanoType(void) const {
    return tens_cano_type_;
  }
  MPSTenCanoType GetTenCanoType(const size_t idx) const {
    return tens_cano_type_[idx];
  }
  std::vector<double> GetEntanglementEntropy(size_t n);
  void Reverse();

  void InplaceMultipleMPO(const TransferMPO &mpo, const size_t Dmin, const size_t Dmax, const CompressMPSScheme &scheme = SVD_COMPRESS);
  BMPS MultipleMPO(const TransferMPO &mpo, const size_t Dmin, const size_t Dmax, const CompressMPSScheme &scheme = SVD_COMPRESS) const;

 private:
  const BMPSPOSITION position_;
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
const Index<QNT> BMPS<TenElemT, QNT>::index0_out_ = Index<QNT>({QNSector(qn0_, 1)}, IN);

}//gqpeps

#include "gqpeps/ond_dim_tn/boundary_mps/bmps_impl.h"

#endif //GQPEPS_OND_DIM_TN_BOUNDARY_MPS_BMPS_H

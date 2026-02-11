/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-10-18
*
* Description: QuantumLiquids/PEPS project. SMatrix in Stochastic Reconfiguration. Especially define the multiplication on vector.
*/


#ifndef QLPEPS_VMC_PEPS_STOCHASTIC_RECONFIGURATION_SMATRIX_H
#define QLPEPS_VMC_PEPS_STOCHASTIC_RECONFIGURATION_SMATRIX_H

#include "qlpeps/two_dim_tn/tps/split_index_tps.h"

namespace qlpeps {

/*
 * Ostar_samples, Ostar_mean are expected to be in the physical O* representation:
 *   O^*(S) = Pi(R^*(S))
 * as prepared by the energy evaluators.
 *
 * SRSMatrix implements:
 *   S = < O^* O > - < O^* >< O >
 * through sample tensors directly, without applying extra fermion-parity transforms.
 */
template<typename TenElemT, typename QNT>
class SRSMatrix {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  SRSMatrix(std::vector<SITPS> *Ostar_samples, SITPS *Ostar_mean,
            size_t world_size) :
      Ostar_samples_(Ostar_samples), Ostar_mean_(Ostar_mean),
      world_size_(world_size) {}

  SITPS operator*(const SITPS &v0) const {
    SITPS res = (*Ostar_samples_)[0] * ((*Ostar_samples_)[0] * v0);
    for (size_t i = 1; i < Ostar_samples_->size(); i++) {
      res += (*Ostar_samples_)[i] * ((*Ostar_samples_)[i] * v0);
    }
    res *= 1.0 / double(Ostar_samples_->size() * world_size_);
    if (Ostar_mean_ != nullptr) { //qlten::hp_numeric::kMPIMasterRank
      res += (-((*Ostar_mean_) * v0)) * (*Ostar_mean_);
      if (diag_shift != 0.0) {
        res += (diag_shift * v0);
      }
    }
    return res;
  }

  TenElemT diag_shift = 0.0;
 private:
  std::vector<SITPS> *Ostar_samples_;
  SITPS *Ostar_mean_;
  size_t world_size_;
};

}//qlpeps

#endif //QLPEPS_VMC_PEPS_STOCHASTIC_RECONFIGURATION_SMATRIX_H

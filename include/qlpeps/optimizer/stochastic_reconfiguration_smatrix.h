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
using namespace qlten;

/*
 * For the complex number case,
 *
 * gten_samples, gten_ave_ \sim \partial_\theta^* \Psi
 * SRSMatrix \sim gten_samples /\ gten_samples^* \sim  \partial_\theta^* \Psi  \partial_\theta \Psi
 *
 */
template<typename TenElemT, typename QNT>
class SRSMatrix {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  SRSMatrix(std::vector<SITPS> *gten_samples, SITPS *gten_ave,
            size_t world_size) :
      gten_samples_(gten_samples), gten_ave_(gten_ave),
      world_size_(world_size) {}

  SITPS operator*(const SITPS &v0) const {
    SITPS res = (*gten_samples_)[0] * ((*gten_samples_)[0] * v0);
    for (size_t i = 1; i < gten_samples_->size(); i++) {
      res += (*gten_samples_)[i] * ((*gten_samples_)[i] * v0);
    }
    res *= 1.0 / double(gten_samples_->size() * world_size_);
    if (gten_ave_ != nullptr) { //kMPIMasterRank
      res += (-((*gten_ave_) * v0)) * (*gten_ave_);
      if (diag_shift != 0.0) {
        res += (diag_shift * v0);
      }
    }
    return res;
  }

  TenElemT diag_shift = 0.0;
 private:
  std::vector<SITPS> *gten_samples_;
  SITPS *gten_ave_;
  size_t world_size_;
};

}//qlpeps

#endif //QLPEPS_VMC_PEPS_STOCHASTIC_RECONFIGURATION_SMATRIX_H

// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: GraceQ/VMC-PEPS project. The generic PEPS class, implementation.
*/

#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H

#include <random>                               //default_random_engine
#include "gqpeps/two_dim_tn/tps/tps.h"
#include "gqpeps/two_dim_tn/tps/tensor_network_2d.h"

#include "gqpeps/algorithm/vmc_update/model_energy_solver.h"

namespace gqpeps {
using namespace gqten;

std::default_random_engine random_engine;

struct VMCOptimizePara {

  double trunc_err; // Truncation Error on compressing boundary MPS
  double D_bmps_min;
  size_t D_bmps_max;    // Boundary MPS dimension

  size_t mc_samples;
  size_t mc_warm_up_sweeps;

  // e.g. In spin model, how many spin up sites and how many spin down sites.
  std::vector<size_t> occupancy_num;

};

template<typename TenElemT, typename QNT, typename EnergySolver>
class VMCPEPSExecutor : public Executor {
 public:
  using TenT = GQTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                  const TPST &tps_initial) :
      optimize_para(optimize_para),
      tps_(tps_initial),
      lx_(tps_initial.cols()),
      ly_(tps_initial.rows()),
      proj_tn_(tps_initial.rows(), tps_initial.cols()),
      config_(tps_initial.rows(), tps_initial.cols()) {

    config_.Random(optimize_para.occupancy_num);

    proj_tn_ = tps_.ProjectToConfiguration(config_);

    bmps_down_set_.reserve(ly_);
    IndexT idx = InverseIndex(proj_tn_(ly_ - 1, 0)->GetIndex(1));
    bmps_down_set_.push_back(BMPS<TenElemT, QNT>(DOWN, lx_, idx));

    bmps_up_set_.reserve(ly_);
    idx = InverseIndex(proj_tn_(0, 0)->GetIndex(3));
    bmps_down_set_.push_back(BMPS<TenElemT, QNT>(UP, lx_, idx));

    bmps_left_set_.reserve(lx_);
    idx = InverseIndex(proj_tn_(0, 0)->GetIndex(0));
    bmps_down_set_.push_back(BMPS<TenElemT, QNT>(LEFT, ly_, idx));

    bmps_right_set_.reserve(lx_);
    idx = InverseIndex(proj_tn_(0, lx_ - 1)->GetIndex(2));
    bmps_down_set_.push_back(BMPS<TenElemT, QNT>(RIGHT, ly_, idx));

  }

  void Execute(void) override;

  VMCOptimizePara optimize_para;

 private:

  void MCSweep_(void);

  TPST tps_;
  TensorNetwork2D<TenElemT, QNT> proj_tn_;
  const size_t lx_;
  const size_t ly_;

  std::vector<BMPS<TenElemT, QNT>> bmps_down_set_;
  std::vector<BMPS<TenElemT, QNT>> bmps_up_set_;
  std::vector<BMPS<TenElemT, QNT>> bmps_left_set_;
  std::vector<BMPS<TenElemT, QNT>> bmps_right_set_;

  Configuration config_;

  EnergySolver energy_solver_;
};

}//gqpeps;

#include "gqpeps/algorithm/vmc_update/vmc_peps_impl.h"

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H

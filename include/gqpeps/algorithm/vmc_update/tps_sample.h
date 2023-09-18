//
// Created by haoxinwang on 07/09/23.
//

#ifndef GRACEQ_VMC_PEPS_TPS_SAMPLE_H
#define GRACEQ_VMC_PEPS_TPS_SAMPLE_H


#include "gqpeps/two_dim_tn/tps/configuration.h"    //Configuration
#include "gqpeps/algorithm/vmc_update/tensor_network_2d.h"

std::default_random_engine random_engine;

namespace gqpeps {
template<typename TenElemT, typename QNT>
struct TPSSample {
 public:
  Configuration config;
  TensorNetwork2D<TenElemT, QNT> tn;
  TenElemT amplitude;

  TPSSample(const size_t rows, const size_t cols) : config(rows, cols), tn(rows, cols), amplitude(0) {}

  TPSSample(const size_t rows, const size_t cols, const TruncatePara &trun_para) :
      config(rows, cols), tn(rows, cols, trun_para), amplitude(0) {}

  TPSSample(const SplitIndexTPS<TenElemT, QNT> &sitps, const Configuration &config, const TruncatePara &trun_para)
      : config(config),
        tn(config.rows(), config.cols(), trun_para) {
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config, tn.GetTruncatePara());
    tn.GrowBMPSForRow(0);
    tn.GrowFullBTen(RIGHT, 0, 2, true);
    tn.InitBTen(LEFT, 0);
    amplitude = tn.Trace({0, 0}, HORIZONTAL);
  }

  /**
   * @note the function doesn't change the truncation error data in tn
   * @param sitps
   * @param occupancy_num
   */
  void RandomInit(const SplitIndexTPS<TenElemT, QNT> &sitps,
                  const std::vector<size_t> &occupancy_num,
                  const size_t rand_seed) {
    config.Random(occupancy_num, rand_seed);
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config, tn.GetTruncatePara());
    tn.GrowBMPSForRow(0);
    tn.GrowFullBTen(RIGHT, 0, 2, true);
    tn.InitBTen(LEFT, 0);
    amplitude = tn.Trace({0, 0}, HORIZONTAL);
  }

  void SetTruncatePara(const TruncatePara &trun_para) {
    tn.SetTruncatePara(trun_para);
  }


  size_t MCSequentiallySweep(const SplitIndexTPS<TenElemT, QNT> &sitps,
                             std::uniform_real_distribution<double> &u_double) {
    size_t accept_num = 0;
    tn.GenerateBMPSApproach(UP);
    for (size_t row = 0; row < tn.rows(); row++) {
      tn.InitBTen(LEFT, row);
      tn.GrowFullBTen(RIGHT, row, 2, true);
      for (size_t col = 0; col < tn.cols() - 1; col++) {
        accept_num += ExchangeUpdate({row, col}, {row, col + 1}, HORIZONTAL, sitps, u_double);
        if (col < tn.cols() - 2) {
          tn.BTenMoveStep(RIGHT);
        }
      }
      if (row < tn.rows() - 1) {
        tn.BMPSMoveStep(DOWN);
      }
    }

    tn.DeleteInnerBMPS(LEFT);
    tn.DeleteInnerBMPS(RIGHT);

    tn.GenerateBMPSApproach(LEFT);
    for (size_t col = 0; col < tn.cols(); col++) {
      tn.InitBTen(UP, col);
      tn.GrowFullBTen(DOWN, col, 2, true);
      for (size_t row = 0; row < tn.rows() - 1; row++) {
        accept_num += ExchangeUpdate({row, col}, {row + 1, col}, VERTICAL, sitps, u_double);
        if (row < tn.rows() - 2) {
          tn.BTenMoveStep(DOWN);
        }
      }
      if (col < tn.cols() - 1) {
        tn.BMPSMoveStep(RIGHT);
      }
    }

    tn.DeleteInnerBMPS(UP);
    return accept_num;
  }

  bool ExchangeUpdate(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                      const SplitIndexTPS<TenElemT, QNT> &sitps,
                      std::uniform_real_distribution<double> &u_double) {
    if (config(site1) == config(site2)) {
      return true;
    }
    assert(sitps(site1)[config(site1)].GetIndexes() == sitps(site1)[config(site2)].GetIndexes());
    TenElemT psi_b = tn.ReplaceNNSiteTrace(site1, site2, bond_dir, sitps(site1)[config(site2)],
                                           sitps(site2)[config(site1)]);
    bool exchange;
    TenElemT &psi_a = amplitude;
    if (std::fabs(psi_b) >= std::fabs(psi_a)) {
      exchange = true;
    } else {
      double div = std::fabs(psi_b) / std::fabs(psi_a);
      double P = div * div;
      if (u_double(random_engine) < P) {
        exchange = true;
      } else {
        exchange = false;
        return exchange;
      }
    }

    std::swap(config(site1), config(site2));
//    const size_t config_tmp = config(site1);
//    config(site1) = config(site2);
//    config(site2) = config_tmp;
    tn.UpdateSiteConfig(site1, config(site1), sitps);
    tn.UpdateSiteConfig(site2, config(site2), sitps);
    amplitude = psi_b;
    return exchange;
  }

};
}//gqpeps

#endif //GRACEQ_VMC_PEPS_TPS_SAMPLE_H

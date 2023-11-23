/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-9
*
* Description: GraceQ/VMC-PEPS project. The generic PEPS class, implementation.
*/

#ifndef GRACEQ_VMC_PEPS_TPS_SAMPLE_H
#define GRACEQ_VMC_PEPS_TPS_SAMPLE_H


#include "gqpeps/two_dim_tn/tps/configuration.h"    //Configuration
#include "gqpeps/algorithm/vmc_update/tensor_network_2d.h"

namespace gqpeps {
template<typename TenElemT, typename QNT>
struct TPSSample {
 public:
  Configuration config;
  TensorNetwork2D<TenElemT, QNT> tn;
  TenElemT amplitude;

  static BMPSTruncatePara trun_para;

  TPSSample(const size_t rows, const size_t cols) : config(rows, cols), tn(rows, cols), amplitude(0) {}

  TPSSample(const SplitIndexTPS<TenElemT, QNT> &sitps, const Configuration &config)
      : config(config), tn(config.rows(), config.cols()) {
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config);
    tn.GrowBMPSForRow(0, trun_para);
    tn.GrowFullBTen(RIGHT, 0, 2, true);
    tn.InitBTen(LEFT, 0);
    amplitude = tn.Trace({0, 0}, HORIZONTAL);
  }

  /**
   * @param sitps
   * @param occupancy_num
   */
  void RandomInit(const SplitIndexTPS<TenElemT, QNT> &sitps,
                  const std::vector<size_t> &occupancy_num) {
    config.Random(occupancy_num);
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config);
    tn.GrowBMPSForRow(0, trun_para);
    tn.GrowFullBTen(RIGHT, 0, 2, true);
    tn.InitBTen(LEFT, 0);
    amplitude = tn.Trace({0, 0}, HORIZONTAL);
  }


//  SplitIndexTPS<TenElemT, QNT> operator*(const SplitIndexTPS<TenElemT, QNT> &sitps) const {
//
//
//  }
//
//  operator SplitIndexTPS<TenElemT, QNT>() const {
//
//  }

  void MCCompressedKagomeLatticeSequentiallyLocalUpdateSweepSmoothBoundary(
      const SplitIndexTPS<TenElemT, QNT> &sitps,
      std::uniform_real_distribution<double> &u_double,
      size_t &accept_num_tri,
      size_t &accept_num_bond
  ) {
    accept_num_tri = 0;
    accept_num_bond = 0;
    tn.GenerateBMPSApproach(UP, trun_para);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      tn.InitBTen(LEFT, row);
      tn.GrowFullBTen(RIGHT, row, 1, true);
      for (size_t col = 0; col < tn.cols() - 1; col++) {
        accept_num_tri += CompressedKagomeLatticeSingleSiteUpdate_({row, col}, sitps, u_double, HORIZONTAL);
        accept_num_bond += CompressedKagomeLatticeExchangeUpdate_({row, col}, {row, col + 1}, HORIZONTAL, sitps,
                                                                  u_double);
        tn.BTenMoveStep(RIGHT);
      }
      accept_num_tri += CompressedKagomeLatticeInnerSiteVerticalBondExchangeUpdate_({row, tn.cols() - 1}, sitps,
                                                                                    u_double, HORIZONTAL);
      tn.BMPSMoveStep(DOWN, trun_para);
    }
    //for the last row flip update
    size_t last_row_idx = tn.rows() - 1;
    tn.InitBTen(LEFT, last_row_idx);
    tn.GrowFullBTen(RIGHT, last_row_idx, 1, true);
    for (size_t col = 0; col < tn.cols() - 1; col++) {
      accept_num_tri += CompressedKagomeLatticeInnerSiteHorizontalBondExchangeUpdate_({last_row_idx, col}, sitps,
                                                                                      u_double, HORIZONTAL);
      accept_num_bond += CompressedKagomeLatticeExchangeUpdate_({last_row_idx, col}, {last_row_idx, col + 1},
                                                                HORIZONTAL, sitps,
                                                                u_double);
      tn.BTenMoveStep(RIGHT);
    }

    tn.DeleteInnerBMPS(LEFT);
    tn.DeleteInnerBMPS(RIGHT);

    tn.GenerateBMPSApproach(LEFT, trun_para);
    for (size_t col = 0; col < tn.cols() - 1; col++) {
      tn.InitBTen(UP, col);
      tn.GrowFullBTen(DOWN, col, 1, true);
      for (size_t row = 0; row < tn.rows() - 1; row++) {
        accept_num_tri += CompressedKagomeLatticeSingleSiteUpdate_({row, col}, sitps, u_double, VERTICAL);
        accept_num_bond += CompressedKagomeLatticeExchangeUpdate_({row, col}, {row + 1, col}, VERTICAL, sitps,
                                                                  u_double);
        tn.BTenMoveStep(DOWN);
      }
      accept_num_tri += CompressedKagomeLatticeInnerSiteHorizontalBondExchangeUpdate_({tn.rows() - 1, col}, sitps,
                                                                                      u_double, VERTICAL);
      tn.BMPSMoveStep(RIGHT, trun_para);
    }
    size_t last_col_idx = tn.cols() - 1;
    tn.InitBTen(UP, last_col_idx);
    tn.GrowFullBTen(DOWN, last_col_idx, 1, true);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      accept_num_tri += CompressedKagomeLatticeInnerSiteVerticalBondExchangeUpdate_({row, last_col_idx}, sitps,
                                                                                    u_double, VERTICAL);
      accept_num_bond += CompressedKagomeLatticeExchangeUpdate_({row, last_col_idx}, {row + 1, last_col_idx}, VERTICAL,
                                                                sitps,
                                                                u_double);
      tn.BTenMoveStep(DOWN);
    }
    tn.DeleteInnerBMPS(UP);
    return;
  }


  // has corner version
  void MCCompressedKagomeLatticeSequentiallyLocalUpdateSweep(const SplitIndexTPS<TenElemT, QNT> &sitps,
                                                             std::uniform_real_distribution<double> &u_double,
                                                             size_t &accept_num_tri,
                                                             size_t &accept_num_bond) {
    accept_num_tri = 0;
    accept_num_bond = 0;
    tn.GenerateBMPSApproach(UP, trun_para);
    for (size_t row = 0; row < tn.rows(); row++) {
      tn.InitBTen(LEFT, row);
      tn.GrowFullBTen(RIGHT, row, 1, true);
      for (size_t col = 0; col < tn.cols() - 1; col++) {
        accept_num_tri += CompressedKagomeLatticeSingleSiteUpdate_({row, col}, sitps, u_double, HORIZONTAL);
        accept_num_bond += CompressedKagomeLatticeExchangeUpdate_({row, col}, {row, col + 1}, HORIZONTAL, sitps,
                                                                  u_double);
        tn.BTenMoveStep(RIGHT);
      }
      accept_num_tri += CompressedKagomeLatticeSingleSiteUpdate_({row, tn.cols() - 1}, sitps, u_double, HORIZONTAL);
      if (row < tn.rows() - 1) {
        tn.BMPSMoveStep(DOWN, trun_para);
      }
    }

    tn.DeleteInnerBMPS(LEFT);
    tn.DeleteInnerBMPS(RIGHT);

    tn.GenerateBMPSApproach(LEFT, trun_para);
    for (size_t col = 0; col < tn.cols(); col++) {
      tn.InitBTen(UP, col);
      tn.GrowFullBTen(DOWN, col, 1, true);
      for (size_t row = 0; row < tn.rows(); row++) {
        accept_num_tri += CompressedKagomeLatticeSingleSiteUpdate_({row, col}, sitps, u_double, VERTICAL);
        if (row < tn.rows() - 1) {
          accept_num_bond += CompressedKagomeLatticeExchangeUpdate_({row, col}, {row + 1, col}, VERTICAL, sitps,
                                                                    u_double);
          tn.BTenMoveStep(DOWN);
        }
      }
      if (col < tn.cols() - 1) {
        tn.BMPSMoveStep(RIGHT, trun_para);
      }
    }

    tn.DeleteInnerBMPS(UP);
//    return (accept_num_bond + accept_num_site) / 2; // a roughly estimation
  }


  size_t MCSequentiallyNNFlipSweep(const SplitIndexTPS<TenElemT, QNT> &sitps,
                                   std::uniform_real_distribution<double> &u_double) {
    size_t accept_num = 0;
    tn.GenerateBMPSApproach(UP, trun_para);
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
        tn.BMPSMoveStep(DOWN, trun_para);
      }
    }

    tn.DeleteInnerBMPS(LEFT);
    tn.DeleteInnerBMPS(RIGHT);

    tn.GenerateBMPSApproach(LEFT, trun_para);
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
        tn.BMPSMoveStep(RIGHT, trun_para);
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
    tn.UpdateSiteConfig(site1, config(site1), sitps);
    tn.UpdateSiteConfig(site2, config(site2), sitps);
    amplitude = psi_b;
    return exchange;
  }

 private:

  bool CompressedKagomeLatticeExchangeUpdate_(const SiteIdx &site1,
                                              const SiteIdx &site2,
                                              BondOrientation bond_dir,
                                              const SplitIndexTPS<TenElemT, QNT> &sitps,
                                              std::uniform_real_distribution<double> &u_double) {
    size_t eff_config1, eff_config2;
    size_t ex_config1, ex_config2;
    size_t config1 = config(site1), config2 = config(site2);
    if (bond_dir == HORIZONTAL) {
      eff_config1 = (config1 >> 2) & 1;
      eff_config2 = config2 & 1;
      ex_config1 = config1 ^ (1 << 2);
      ex_config2 = config2 ^ 1;
    } else {
      eff_config1 = (config1 >> 1) & 1;
      eff_config2 = config2 & 1;
      ex_config1 = config1 ^ (1 << 1);
      ex_config2 = config2 ^ 1;
    }

    if (eff_config1 == eff_config2) {
      return false;
    }
    if (sitps(site1)[ex_config1].GetQNBlkNum() == 0) {
      std::cout << "warning: site (" << site1[0] << ", " << site1[1] << ") on config " << ex_config1
                << " lost tensor block." << std::endl;
      return false;
    }
    if (sitps(site2)[ex_config2].GetQNBlkNum() == 0) {
      std::cout << "warning: site (" << site1[0] << ", " << site1[1] << ") on config " << ex_config1
                << " lost tensor block." << std::endl;
      return false;
    }
    TenElemT psi_b = tn.ReplaceNNSiteTrace(site1, site2, bond_dir, sitps(site1)[ex_config1],
                                           sitps(site2)[ex_config2]);
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

    config(site1) = ex_config1;
    config(site2) = ex_config2;
    tn.UpdateSiteConfig(site1, ex_config1, sitps);
    tn.UpdateSiteConfig(site2, ex_config2, sitps);
    amplitude = psi_b;
    return exchange;
  }

  bool CompressedKagomeLatticeInnerSiteVerticalBondExchangeUpdate_(
      const SiteIdx &site,
      const SplitIndexTPS<TenElemT, QNT> &sitps,
      std::uniform_real_distribution<double> &u_double,
      const BondOrientation mps_orient
  ) {
    size_t config_site = config(site);
    bool config_a = config_site & 1;
    bool config_b = config_site >> 1 & 1;
    if (config_a == config_b) {
      return false;
    }

    size_t ex_config = 2 * config_a + config_b + (config_site & 4);

    TenElemT psi_ex = tn.ReplaceOneSiteTrace(site, sitps(site)[ex_config], mps_orient);

    bool exchange;
    TenElemT &psi_a = amplitude;
    if (std::fabs(psi_ex) >= std::fabs(psi_a)) {
      exchange = true;
    } else {
      double div = std::fabs(psi_ex) / std::fabs(psi_a);
      double P = div * div;
      if (u_double(random_engine) < P) {
        exchange = true;
      } else {
        exchange = false;
        return exchange;
      }
    }

    config(site) = ex_config;
    tn.UpdateSiteConfig(site, ex_config, sitps);
    amplitude = psi_ex;
    return exchange;
  }

  bool CompressedKagomeLatticeInnerSiteHorizontalBondExchangeUpdate_(
      const SiteIdx &site,
      const SplitIndexTPS<TenElemT, QNT> &sitps,
      std::uniform_real_distribution<double> &u_double,
      const BondOrientation mps_orient
  ) {
    size_t config_site = config(site);
    bool config_a = config_site & 1;
    bool config_b = config_site >> 2 & 1;
    if (config_a == config_b) {
      return false;
    }

    size_t ex_config = 4 * config_a + (config_site & 2) + config_b;

    TenElemT psi_ex = tn.ReplaceOneSiteTrace(site, sitps(site)[ex_config], mps_orient);

    bool exchange;
    TenElemT &psi_a = amplitude;
    if (std::fabs(psi_ex) >= std::fabs(psi_a)) {
      exchange = true;
    } else {
      double div = std::fabs(psi_ex) / std::fabs(psi_a);
      double P = div * div;
      if (u_double(random_engine) < P) {
        exchange = true;
      } else {
        exchange = false;
        return exchange;
      }
    }

    config(site) = ex_config;
    tn.UpdateSiteConfig(site, ex_config, sitps);
    amplitude = psi_ex;
    return exchange;
  }

  bool CompressedKagomeLatticeSingleSiteUpdate_(const SiteIdx &site,
                                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                                std::uniform_real_distribution<double> &u_double,
                                                const BondOrientation mps_orient) {
    size_t config_site = config(site);
    if (config_site == 0 || config_site == 7) {
      return false;//or true
    }
    size_t rotate_config1 = config_site / 4 + 2 * (config_site % 4);
    size_t rotate_config2 = rotate_config1 / 4 + 2 * (rotate_config1 % 4);
    TenElemT psi_rotate1 = tn.ReplaceOneSiteTrace(site, sitps(site)[rotate_config1], mps_orient);
    TenElemT psi_rotate2 = tn.ReplaceOneSiteTrace(site, sitps(site)[rotate_config2], mps_orient);

    //make sure rotate1 is smaller than rotate2
    if (std::fabs(psi_rotate1) > std::fabs(psi_rotate2)) {
      std::swap(rotate_config1, rotate_config2);
      std::swap(psi_rotate1, psi_rotate2);
    }

    TenElemT psi0 = amplitude;
    double p0 = amplitude * amplitude;
    double p1 = psi_rotate1 * psi_rotate1;
    double p2 = psi_rotate2 * psi_rotate2; //p1<=p2

    if (p0 + p1 + p2 <= 2 * std::max(p0, p2)) {
      if (std::fabs(psi_rotate2) >= std::fabs(psi0)) {
        //skip to rotate2
        config(site) = rotate_config2;
        tn.UpdateSiteConfig(site, rotate_config2, sitps);
        amplitude = psi_rotate2;
        return true;
      }
      // psi0 is the largest amplitude
      double rand_num = u_double(random_engine);
      if (rand_num < p2 / p0) {
        //skip to rotate2
        config(site) = rotate_config2;
        tn.UpdateSiteConfig(site, rotate_config2, sitps);
        amplitude = psi_rotate2;
        return true;

      } else if (rand_num < (p1 + p2) / p0) {
        //skip tp rotate1
        config(site) = rotate_config1;
        tn.UpdateSiteConfig(site, rotate_config1, sitps);
        amplitude = psi_rotate1;
        return true;
      } else {
        return false;
      }
    } else { //p_middle + p_small > p_large
      double rand_num = u_double(random_engine);
      if (p0 >= p2) { //p0 = p_large, p2 = p_middle, p1 = p_small
        if (rand_num < p2 / (p2 + p1)) {
          //skip to rotate2
          config(site) = rotate_config2;
          tn.UpdateSiteConfig(site, rotate_config2, sitps);
          amplitude = psi_rotate2;
          return true;
        } else {
          //skip tp rotate1
          config(site) = rotate_config1;
          tn.UpdateSiteConfig(site, rotate_config1, sitps);
          amplitude = psi_rotate1;
          return true;
        }
      } else if (p0 <= p1) { //p0,p1,p2: small, middle large
        if (rand_num < p2 / (p0 + p1)) {
          //skip to rotate2, large
          config(site) = rotate_config2;
          tn.UpdateSiteConfig(site, rotate_config2, sitps);
          amplitude = psi_rotate2;
          return true;
        } else {
          //skip tp rotate1, middle
          config(site) = rotate_config1;
          tn.UpdateSiteConfig(site, rotate_config1, sitps);
          amplitude = psi_rotate1;
          return true;
        }
      } else { //p1 < p0 < p2
        if (rand_num <= p2 / (p0 + p1)) {
          //skip to rotate2, largest amplitude configuration
          config(site) = rotate_config2;
          tn.UpdateSiteConfig(site, rotate_config2, sitps);
          amplitude = psi_rotate2;
          return true;
        } else if (rand_num <= p2 / (p0 + p1) + p1 / p0 * (1 - p2 / (p0 + p1))) {
          //skip tp rotate1, smallest
          config(site) = rotate_config1;
          tn.UpdateSiteConfig(site, rotate_config1, sitps);
          amplitude = psi_rotate1;
          return true;
        } else {
          //no jump
          return false;
        }
      }
    } //end of the case p_middle + p_small > p_large
  } //CompressedKagomeLatticeSingleSiteUpdate_

}; //TPSSample

template<typename TenElemT, typename QNT>
BMPSTruncatePara TPSSample<TenElemT, QNT>::trun_para = BMPSTruncatePara(0, 0, 0.0);
}//gqpeps

#endif //GRACEQ_VMC_PEPS_TPS_SAMPLE_H

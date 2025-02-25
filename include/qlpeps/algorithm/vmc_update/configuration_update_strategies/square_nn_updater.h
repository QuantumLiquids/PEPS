/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-09
*
* Description: QuantumLiquids/PEPS project.
*
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_TPS_SAMPLE_NN_EXCHANGE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_TPS_SAMPLE_NN_EXCHANGE_H

#include "qlpeps/algorithm/vmc_update/wave_function_component.h"    // TPSWaveFunctionComponent
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/monte_carlo_tools/non_detailed_balance_mcmc.h"     // NonDBMCMCStateUpdate
#include "monte_carlo_sweep_updater_all.h"                          // MonteCarloSweepUpdaterBase

namespace qlpeps {

///< base class for CRTP
template<typename MCUpdater>
class MCUpdateSquareNNUpdateBase : public MonteCarloSweepUpdaterBase {
  using MonteCarloSweepUpdaterBase::MonteCarloSweepUpdaterBase;
 public:
  template<typename TenElemT, typename QNT>
  void operator()(const SplitIndexTPS<TenElemT, QNT> &sitps,
                  TPSWaveFunctionComponent<TenElemT, QNT> &tps_component,
                  std::vector<double> &accept_rates) {
    size_t flip_accept_num = 0;
    auto &tn = tps_component.tn;
    tn.GenerateBMPSApproach(UP, tps_component.trun_para);
    for (size_t row = 0; row < tn.rows(); row++) {
      tn.InitBTen(LEFT, row);
      tn.GrowFullBTen(RIGHT, row, 2, true);
      for (size_t col = 0; col < tn.cols() - 1; col++) {
        flip_accept_num += static_cast<MCUpdater *>(this)->TwoSiteNNUpdateLocalImpl({row, col},
                                                                                    {row, col + 1},
                                                                                    HORIZONTAL,
                                                                                    sitps,
                                                                                    tps_component);
        if (col < tn.cols() - 2) {
          tn.BTenMoveStep(RIGHT);
        }
      }
      if (row < tn.rows() - 1) {
        tn.BMPSMoveStep(DOWN, tps_component.trun_para);
      }
    }

    tn.DeleteInnerBMPS(LEFT);
    tn.DeleteInnerBMPS(RIGHT);

    tn.GenerateBMPSApproach(LEFT, tps_component.trun_para);
    for (size_t col = 0; col < tn.cols(); col++) {
      tn.InitBTen(UP, col);
      tn.GrowFullBTen(DOWN, col, 2, true);
      for (size_t row = 0; row < tn.rows() - 1; row++) {
        flip_accept_num += static_cast<MCUpdater *>(this)->TwoSiteNNUpdateLocalImpl({row, col},
                                                                                    {row + 1, col},
                                                                                    VERTICAL,
                                                                                    sitps,
                                                                                    tps_component);
        if (row < tn.rows() - 2) {
          tn.BTenMoveStep(DOWN);
        }
      }
      if (col < tn.cols() - 1) {
        tn.BMPSMoveStep(RIGHT, tps_component.trun_para);
      }
    }

    tn.DeleteInnerBMPS(UP);
    double bond_num = tn.cols() * (tn.rows() - 1) + tn.rows() * (tn.cols() - 1);
    accept_rates = {double(flip_accept_num) / bond_num};
  }

}; //


/**
 * Explicit class define the Monte-Carlo update strategy.
 *
 * Monte Carlo sweep defined by NN bond exchange,
 * work for both fermion and boson since the MC weight is defined by abs square of the wave function amplitude.
 *
 * Suitable application example cases:
 *    - spin-1/2 Heisenberg model with U1 symmetry constrain;
 *    - t-J model.
 */
class MCUpdateSquareNNExchange : public MCUpdateSquareNNUpdateBase<MCUpdateSquareNNExchange> {
 public:
  using MCUpdateSquareNNUpdateBase<MCUpdateSquareNNExchange>::MCUpdateSquareNNUpdateBase;
  template<typename TenElemT, typename QNT>
  bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                TPSWaveFunctionComponent<TenElemT, QNT> &tps_component) {
    if (tps_component.config(site1) == tps_component.config(site2)) {
      return false;
    }
    if constexpr (Index<QNT>::IsFermionic()) {
      std::vector<Index<QNT>> index1 = sitps(site1)[tps_component.config(site1)].GetIndexes();
      std::vector<Index<QNT>> index2 = sitps(site1)[tps_component.config(site2)].GetIndexes();
      for (size_t i = 0; i < 4; i++) {
        assert(index1[i] == index2[i]);
      }
    } else {
      assert(sitps(site1)[tps_component.config(site1)].GetIndexes()
                 == sitps(site1)[tps_component.config(site2)].GetIndexes());
    }

    TenElemT psi_b = tps_component.tn.ReplaceNNSiteTrace(site1, site2, bond_dir,
                                                         sitps(site1)[tps_component.config(site2)],
                                                         sitps(site2)[tps_component.config(site1)]);
    bool exchange;
    TenElemT &psi_a = tps_component.amplitude;
    if (std::abs(psi_b) >= std::abs(psi_a)) {
      exchange = true;
    } else {
      double div = std::abs(psi_b) / std::abs(psi_a);
      double P = div * div;
      if (this->u_double_(random_engine_) < P) {
        exchange = true;
      } else {
        exchange = false;
        return exchange;
      }
    }

    size_t temp_config1 = tps_component.config(site2);
    size_t temp_config2 = tps_component.config(site1);
    tps_component.UpdateLocal(sitps, psi_b,
                              std::make_pair(site1, temp_config1),
                              std::make_pair(site2, temp_config2));
    return exchange;
  }
};

/**
 * Explicit class define the Monte-Carlo update strategy.
 *
 * Monte Carlo sweep defined by NN bond update on square lattice, update upon
 * all the possible configurations without limitation on any symmetry constrain.
 * work for both fermion and boson since the MC weight is defined by abs square of the wave function amplitude.
 */
class MCUpdateSquareNNFullSpaceUpdate : public MCUpdateSquareNNUpdateBase<MCUpdateSquareNNExchange> {
 public:
  template<typename TenElemT, typename QNT>
  bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                TPSWaveFunctionComponent<TenElemT, QNT> &tps_component) {
    auto &tn = tps_component.tn;
    size_t dim = sitps.PhysicalDim();
    std::vector<TenElemT> alternative_psi(dim * dim);
    size_t init_config = tps_component.config(site1) * dim + tps_component.config(site2);
    assert(sitps(site1)[tps_component.config(site1)].GetIndexes()
               == sitps(site1)[tps_component.config(site2)].GetIndexes());
    alternative_psi[init_config] = tps_component.amplitude;
    for (size_t config1 = 0; config1 < dim; config1++) {
      for (size_t config2 = 0; config2 < dim; config2++) {
        size_t config = config1 * dim + config2;
        if (config != init_config) {
          alternative_psi[config] =
              tn.ReplaceNNSiteTrace(site1, site2, bond_dir,
                                    sitps(site1)[config1],
                                    sitps(site2)[config2]);
        }
      }
    }
    std::vector<double> weights(dim * dim);
    for (size_t i = 0; i < dim * dim; i++) {
      weights[i] = std::norm(alternative_psi[i] / tps_component.amplitude);
    }
    size_t final_state = NonDBMCMCStateUpdate(init_config, weights, random_engine_);
    if (final_state == init_config) {
      return false;
    }

    tps_component.UpdateLocal(sitps, alternative_psi[final_state],
                              std::make_pair(site1, final_state / dim),
                              std::make_pair(site2, final_state % dim));
    return true;
  }
};

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_TPS_SAMPLE_NN_EXCHANGE_H

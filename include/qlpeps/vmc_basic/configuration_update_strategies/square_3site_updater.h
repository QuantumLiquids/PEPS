/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-23
*
* Description: QuantumLiquids/PEPS project. Explicit class of wave function component in square lattice. Monte Carlo sweep realized by NNN 3-site exchange.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_3SITE_UPDATER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_3SITE_UPDATER_H

#include "qlpeps/vmc_basic/wave_function_component.h"    // TPSWaveFunctionComponent
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/vmc_basic/monte_carlo_tools/suwa_todo_update.h"     // NonDBMCMCStateUpdate

namespace qlpeps {

/**
 * base class for 3-site update, inherit by CRTP
 * @tparam MCUpdater should define function TNN3SiteUpdateImpl
 * e.g. MCUpdateSquareTNN3SiteExchange
 */
template<typename MCUpdater, typename WaveFunctionDress = qlpeps::NoDress>
class MCUpdateSquareTNN3SiteUpdateBase : public MonteCarloSweepUpdaterBase<WaveFunctionDress> {
  using MonteCarloSweepUpdaterBase<WaveFunctionDress>::MonteCarloSweepUpdaterBase;
 public:
  template<typename TenElemT, typename QNT>
  void operator()(const SplitIndexTPS<TenElemT, QNT> &sitps,
                  TPSWaveFunctionComponent<TenElemT, QNT, WaveFunctionDress> &tps_component,
                  std::vector<double> &accept_rates) {
    size_t flip_accept_num = 0;
    auto &tn = tps_component.tn;
    auto &contractor = tps_component.contractor;
    contractor.GenerateBMPSApproach(tn, UP, tps_component.trun_para);
    for (size_t row = 0; row < tn.rows(); row++) {
      contractor.InitBTen(tn, LEFT, row);
      contractor.GrowFullBTen(tn, RIGHT, row, 3, true);
      tps_component.amplitude = contractor.ReplaceTNNSiteTrace(tn, {row, 0}, HORIZONTAL,
                                                       sitps({row, 0})[tps_component.config({row, 0})],
                                                       sitps({row, 1})[tps_component.config({row, 1})],
                                                       sitps({row, 2})[tps_component.config({row, 2})]);
      for (size_t col = 0; col < tn.cols() - 2; col++) {
        flip_accept_num +=
            static_cast<MCUpdater *>(this)->TNN3SiteUpdateImpl({row, col},
                                                               {row, col + 1},
                                                               {row, col + 2},
                                                               HORIZONTAL,
                                                               sitps,
                                                               tps_component);
        if (col < tn.cols() - 3) {
          contractor.ShiftBTenWindow(tn, RIGHT);
        }
      }
      if (row < tn.rows() - 1) {
        contractor.ShiftBMPSWindow(tn, DOWN, tps_component.trun_para);
      }
    }

    contractor.DeleteInnerBMPS(LEFT);
    contractor.DeleteInnerBMPS(RIGHT);

    contractor.GenerateBMPSApproach(tn, LEFT, tps_component.trun_para);
    for (size_t col = 0; col < tn.cols(); col++) {
      contractor.InitBTen(tn, UP, col);
      contractor.GrowFullBTen(tn, DOWN, col, 3, true);
      tps_component.amplitude = contractor.ReplaceTNNSiteTrace(tn, {0, col}, VERTICAL,
                                                       sitps({0, col})[tps_component.config({0, col})],
                                                       sitps({1, col})[tps_component.config({1, col})],
                                                       sitps({2, col})[tps_component.config({2, col})]);
      for (size_t row = 0; row < tn.rows() - 2; row++) {
        flip_accept_num +=
            static_cast<MCUpdater *>(this)->TNN3SiteUpdateImpl({row, col},
                                                               {row + 1, col},
                                                               {row + 2, col},
                                                               VERTICAL,
                                                               sitps,
                                                               tps_component);
        if (row < tn.rows() - 3) {
          contractor.ShiftBTenWindow(tn, DOWN);
        }
      }
      if (col < tn.cols() - 1) {
        contractor.ShiftBMPSWindow(tn, RIGHT, tps_component.trun_para);
      }
    }

    contractor.DeleteInnerBMPS(UP);
    double total_flip_num = tn.cols() * (tn.rows() - 2) + tn.rows() * (tn.cols() - 2);
    accept_rates = {double(flip_accept_num) / total_flip_num};
  }

};

/**
 * Explicit class define the Monte-Carlo update strategy.
 *
 * Monte Carlo sweep defined by exchanging the configurations on TNN 3 sites,
 * work for both fermion and boson since the MC weight is defined by abs square of the wave function amplitude.
 *
 * Suitable application example cases:
 *    - spin-1/2 Heisenberg model with U1 symmetry constrain;
 *    - t-J model.
 */
class MCUpdateSquareTNN3SiteExchange :
    public MCUpdateSquareTNN3SiteUpdateBase<MCUpdateSquareTNN3SiteExchange, qlpeps::NoDress> {
 public:
  template<typename TenElemT, typename QNT>
  bool TNN3SiteUpdateImpl(const SiteIdx &site1, const SiteIdx &site2, const SiteIdx &site3,
                          BondOrientation bond_dir,
                          const SplitIndexTPS<TenElemT, QNT> &sitps,
                          TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress> &tps_component) {
    auto &tn = tps_component.tn;
    auto &contractor = tps_component.contractor;
    size_t spin1 = tps_component.config(site1);
    size_t spin2 = tps_component.config(site2);
    size_t spin3 = tps_component.config(site3);
    if (spin1 == spin2 && spin2 == spin3) {
      return false;
    }
    std::vector<size_t> spins = {spin1, spin2, spin3};
    std::sort(spins.begin(), spins.end());

    std::vector<std::vector<size_t>> permutations;
    do {
      permutations.push_back(spins);
    } while (std::next_permutation(spins.begin(), spins.end()));
    std::vector<size_t> initial_spins = {spin1, spin2, spin3};
    size_t init_state =
        std::distance(permutations.begin(), std::find(permutations.begin(), permutations.end(), initial_spins));
    std::vector<TenElemT> psis(permutations.size());
    double psi_abs_max = 0;
    for (size_t i = 0; i < permutations.size(); ++i) {
      if (i != init_state) {
        psis[i] = contractor.ReplaceTNNSiteTrace(tn, site1, bond_dir,
                                         sitps(site1)[permutations[i][0]],
                                         sitps(site2)[permutations[i][1]],
                                         sitps(site3)[permutations[i][2]]);

      } else {
        psis[i] = tps_component.amplitude;
      }
      psi_abs_max = std::max(psi_abs_max, std::abs(psis[i]));
    }
    std::vector<double> weights(permutations.size());
    for (size_t i = 0; i < weights.size(); i++) {
      weights[i] = std::norm(psis[i] / psi_abs_max);
    }

    size_t final_state = SuwaTodoStateUpdate(init_state, weights, random_engine_);
    if (final_state == init_state) {
      return false;
    }
    tps_component.UpdateLocal(sitps, psis[final_state],
                              std::make_pair(site1, permutations[final_state][0]),
                              std::make_pair(site2, permutations[final_state][1]),
                              std::make_pair(site3, permutations[final_state][2]));
    return true;
  }
}; //MCUpdateSquareTNN3SiteExchange

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_3SITE_UPDATER_H

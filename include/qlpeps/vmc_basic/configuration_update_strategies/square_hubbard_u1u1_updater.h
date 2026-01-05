/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-01-04
*
* Description: QuantumLiquids/PEPS project. Monte Carlo updater for Hubbard model
*              with U(1)_up × U(1)_down symmetry (separate conservation of N_up and N_down).
*/

#ifndef QLPEPS_VMC_BASIC_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_HUBBARD_U1U1_UPDATER_H
#define QLPEPS_VMC_BASIC_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_HUBBARD_U1U1_UPDATER_H

#include "square_nn_updater.h"
#include "qlpeps/vmc_basic/monte_carlo_tools/suwa_todo_update.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_hubbard_model.h"

namespace qlpeps {

/**
 * @brief Calculate (N_up, N_down) for a single Hubbard site configuration.
 * 
 * Configuration encoding:
 *   0: DoubleOccupancy |↑↓⟩ -> (N_up=1, N_down=1)
 *   1: SpinUp         |↑⟩  -> (N_up=1, N_down=0)
 *   2: SpinDown       |↓⟩  -> (N_up=0, N_down=1)
 *   3: Empty          |0⟩  -> (N_up=0, N_down=0)
 * 
 * @param config Single-site configuration (0-3).
 * @return std::pair<size_t, size_t> (N_up, N_down)
 */
inline std::pair<size_t, size_t> HubbardConfig2SpinCounts(size_t config) {
  switch (HubbardSingleSiteState(config)) {
    case HubbardSingleSiteState::DoubleOccupancy: return {1, 1};
    case HubbardSingleSiteState::SpinUp:          return {1, 0};
    case HubbardSingleSiteState::SpinDown:        return {0, 1};
    case HubbardSingleSiteState::Empty:           return {0, 0};
    default:
      std::cerr << "Invalid Hubbard configuration: " << config << std::endl;
      std::abort();
  }
}

/**
 * @brief Enumerate all two-site configuration pairs (c1, c2) that yield the same (N_up_total, N_down_total).
 * 
 * For Hubbard model, each site can be in one of 4 states: {0,1,2,3}.
 * The total (N_up, N_down) for two sites is the sum of individual contributions.
 * 
 * Given a target (N_up_total, N_down_total), this function returns all pairs
 * (config1, config2) such that:
 *   N_up(config1) + N_up(config2) == N_up_total
 *   N_down(config1) + N_down(config2) == N_down_total
 * 
 * @param n_up_total Total number of spin-up electrons (0, 1, or 2 for two sites).
 * @param n_down_total Total number of spin-down electrons (0, 1, or 2 for two sites).
 * @return Vector of valid (config1, config2) pairs.
 */
inline std::vector<std::pair<size_t, size_t>> EnumerateHubbardTwoSiteConfigsWithU1U1(
    size_t n_up_total, size_t n_down_total) {
  std::vector<std::pair<size_t, size_t>> result;
  constexpr size_t kHubbardDim = 4;
  
  for (size_t c1 = 0; c1 < kHubbardDim; ++c1) {
    auto [n_up_1, n_down_1] = HubbardConfig2SpinCounts(c1);
    for (size_t c2 = 0; c2 < kHubbardDim; ++c2) {
      auto [n_up_2, n_down_2] = HubbardConfig2SpinCounts(c2);
      if (n_up_1 + n_up_2 == n_up_total && n_down_1 + n_down_2 == n_down_total) {
        result.emplace_back(c1, c2);
      }
    }
  }
  return result;
}

/**
 * @brief Monte Carlo updater for Hubbard model with U(1)_up × U(1)_down symmetry.
 * 
 * This updater conserves both N_up (total spin-up electrons) and N_down
 * (total spin-down electrons) separately during MC sampling.
 * 
 * Strategy:
 * For each NN bond, compute the total (N_up, N_down) of the two sites,
 * enumerate all valid configuration pairs with the same quantum numbers,
 * then apply Suwa-Todo update to sample among them.
 * 
 * Example allowed transitions for two sites with (N_up=1, N_down=1):
 *   |↑, ↓⟩ ↔ |↓, ↑⟩ ↔ |↑↓, 0⟩ ↔ |0, ↑↓⟩
 * 
 * This is more ergodic than simple exchange, which can only do |↑, ↓⟩ ↔ |↓, ↑⟩.
 */
class MCUpdateSquareNNHubbardU1U1OBC : public MCUpdateSquareNNUpdateBaseOBC<MCUpdateSquareNNHubbardU1U1OBC> {
 public:
  using MCUpdateSquareNNUpdateBaseOBC<MCUpdateSquareNNHubbardU1U1OBC>::MCUpdateSquareNNUpdateBaseOBC;

  template<typename TenElemT, typename QNT>
  bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                TPSWaveFunctionComponent<TenElemT, QNT> &tps_component) {
    auto &tn = tps_component.tn;
    auto &contractor = tps_component.contractor;
    
    size_t config1 = tps_component.config(site1);
    size_t config2 = tps_component.config(site2);
    
    // Calculate current (N_up, N_down) for the two sites
    auto [n_up_1, n_down_1] = HubbardConfig2SpinCounts(config1);
    auto [n_up_2, n_down_2] = HubbardConfig2SpinCounts(config2);
    size_t n_up_total = n_up_1 + n_up_2;
    size_t n_down_total = n_down_1 + n_down_2;
    
    // Enumerate all valid configurations with the same (N_up, N_down)
    auto valid_configs = EnumerateHubbardTwoSiteConfigsWithU1U1(n_up_total, n_down_total);
    
    // If only one configuration is valid, no update possible
    if (valid_configs.size() <= 1) {
      return false;
    }
    
    // Find index of current configuration
    size_t init_state = 0;
    for (size_t i = 0; i < valid_configs.size(); ++i) {
      if (valid_configs[i].first == config1 && valid_configs[i].second == config2) {
        init_state = i;
        break;
      }
    }
    
    // Compute amplitudes for all valid configurations
    std::vector<TenElemT> psis(valid_configs.size());
    psis[init_state] = tps_component.amplitude;
    
    for (size_t i = 0; i < valid_configs.size(); ++i) {
      if (i != init_state) {
        psis[i] = contractor.ReplaceNNSiteTrace(tn, site1, site2, bond_dir,
                                                sitps(site1)[valid_configs[i].first],
                                                sitps(site2)[valid_configs[i].second]);
      }
    }
    
    // Compute weights for Suwa-Todo update (normalized by current amplitude to avoid overflow)
    std::vector<double> weights(valid_configs.size());
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] = std::norm(psis[i] / tps_component.amplitude);
    }
    
    // Suwa-Todo update
    size_t final_state = SuwaTodoStateUpdate(init_state, weights, random_engine_);
    
    if (final_state == init_state) {
      return false;
    }
    
    // Accept the new configuration
    tps_component.UpdateLocal(sitps, psis[final_state],
                              std::make_pair(site1, valid_configs[final_state].first),
                              std::make_pair(site2, valid_configs[final_state].second));
    return true;
  }
};

// Backward-compatible alias
using MCUpdateSquareNNHubbardU1U1 = MCUpdateSquareNNHubbardU1U1OBC;

} // namespace qlpeps

#endif // QLPEPS_VMC_BASIC_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_HUBBARD_U1U1_UPDATER_H


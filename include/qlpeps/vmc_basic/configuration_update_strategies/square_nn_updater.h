/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-09
*
* Description: QuantumLiquids/PEPS project.
*
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_TPS_SAMPLE_NN_EXCHANGE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_TPS_SAMPLE_NN_EXCHANGE_H

#include "qlpeps/vmc_basic/wave_function_component.h"               // TPSWaveFunctionComponent
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/vmc_basic/monte_carlo_tools/suwa_todo_update.h"     // NonDBMCMCStateUpdate
#include "monte_carlo_sweep_updater_base.h"                         // MonteCarloSweepUpdaterBase
#include "qlpeps/vmc_basic/tj_single_site_state.h"

namespace qlpeps {

///< base class for CRTP (Open Boundary Condition)
template<typename MCUpdater, typename WaveFunctionDress = qlpeps::NoDress>
class MCUpdateSquareNNUpdateBaseOBC : public MonteCarloSweepUpdaterBase<WaveFunctionDress> {
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
      contractor.GrowFullBTen(tn, RIGHT, row, 2, true);
      for (size_t col = 0; col < tn.cols() - 1; col++) {
        flip_accept_num += static_cast<MCUpdater *>(this)->TwoSiteNNUpdateLocalImpl({row, col},
                                                                                    {row, col + 1},
                                                                                    HORIZONTAL,
                                                                                    sitps,
                                                                                    tps_component);
        if (col < tn.cols() - 2) {
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
      contractor.GrowFullBTen(tn, DOWN, col, 2, true);
      for (size_t row = 0; row < tn.rows() - 1; row++) {
        flip_accept_num += static_cast<MCUpdater *>(this)->TwoSiteNNUpdateLocalImpl({row, col},
                                                                                    {row + 1, col},
                                                                                    VERTICAL,
                                                                                    sitps,
                                                                                    tps_component);
        if (row < tn.rows() - 2) {
          contractor.ShiftBTenWindow(tn, DOWN);
        }
      }
      if (col < tn.cols() - 1) {
        contractor.ShiftBMPSWindow(tn, RIGHT, tps_component.trun_para);
      }
    }

    contractor.DeleteInnerBMPS(UP);
    double bond_num = tn.cols() * (tn.rows() - 1) + tn.rows() * (tn.cols() - 1);
    accept_rates = {double(flip_accept_num) / bond_num};
  }

}; //

template<typename MCUpdater, typename WaveFunctionDress = qlpeps::NoDress>
using MCUpdateSquareNNUpdateBase = MCUpdateSquareNNUpdateBaseOBC<MCUpdater, WaveFunctionDress>;

///< base class for CRTP (Periodic Boundary Condition)
template<typename MCUpdater, typename WaveFunctionDress = qlpeps::NoDress>
class MCUpdateSquareNNUpdateBasePBC : public MonteCarloSweepUpdaterBase<WaveFunctionDress> {
  using MonteCarloSweepUpdaterBase<WaveFunctionDress>::MonteCarloSweepUpdaterBase;
 public:
  template<typename TenElemT, typename QNT, template<typename, typename> class ContractorT>
  void operator()(const SplitIndexTPS<TenElemT, QNT> &sitps,
                  TPSWaveFunctionComponent<TenElemT, QNT, WaveFunctionDress, ContractorT> &tps_component,
                  std::vector<double> &accept_rates) {
    size_t flip_accept_num = 0;
    auto &tn = tps_component.tn;
    
    // Total bonds in PBC = rows * cols * 2 (Horizontal + Vertical)
    // We update roughly the same number of bonds as a full sweep.
    size_t rows = tn.rows();
    size_t cols = tn.cols();
    size_t bond_num = rows * cols * 2; 

    // Random bond selection
    std::uniform_int_distribution<size_t> dist_row(0, rows - 1);
    std::uniform_int_distribution<size_t> dist_col(0, cols - 1);
    std::uniform_int_distribution<size_t> dist_dir(0, 1); // 0: Horizontal, 1: Vertical

    for (size_t i = 0; i < bond_num; ++i) {
        size_t r = dist_row(this->random_engine_);
        size_t c = dist_col(this->random_engine_);
        BondOrientation dir = (dist_dir(this->random_engine_) == 0) ? HORIZONTAL : VERTICAL;
        
        SiteIdx s1{r, c};
        SiteIdx s2;
        if (dir == HORIZONTAL) {
            s2 = SiteIdx{r, (c + 1) % cols};
        } else {
            s2 = SiteIdx{(r + 1) % rows, c};
        }

        flip_accept_num += static_cast<MCUpdater *>(this)->TwoSiteNNUpdateLocalImpl(s1, s2, dir, sitps, tps_component);
    }
    
    accept_rates = {double(flip_accept_num) / double(bond_num)};
  }
};


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
class MCUpdateSquareNNExchangeOBC : public MCUpdateSquareNNUpdateBaseOBC<MCUpdateSquareNNExchangeOBC> {
 public:
  using MCUpdateSquareNNUpdateBaseOBC<MCUpdateSquareNNExchangeOBC>::MCUpdateSquareNNUpdateBaseOBC;
  template<typename TenElemT, typename QNT>
  bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                TPSWaveFunctionComponent<TenElemT, QNT> &tps_component) {
    if (tps_component.config(site1) == tps_component.config(site2)) {
      return false;
    }
#ifndef NDEBUG
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
#endif
    TenElemT psi_b = tps_component.contractor.ReplaceNNSiteTrace(tps_component.tn, site1, site2, bond_dir,
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
 * Explicit class define the Monte-Carlo update strategy (PBC version).
 */
using MCUpdateSquareNNExchange = MCUpdateSquareNNExchangeOBC;

class MCUpdateSquareNNExchangePBC : public MCUpdateSquareNNUpdateBasePBC<MCUpdateSquareNNExchangePBC> {
 public:
  using MCUpdateSquareNNUpdateBasePBC<MCUpdateSquareNNExchangePBC>::MCUpdateSquareNNUpdateBasePBC;
  template<typename TenElemT, typename QNT, typename WaveFunctionDress, template<typename, typename> class ContractorT>
  bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                TPSWaveFunctionComponent<TenElemT, QNT, WaveFunctionDress, ContractorT> &tps_component) {
    if (tps_component.config(site1) == tps_component.config(site2)) {
      return false;
    }
    
    size_t c1 = tps_component.config(site1);
    size_t c2 = tps_component.config(site2);
    
    TenElemT psi_a = tps_component.amplitude;

    // Trial: keep the contractor "shadow cache" alive until accept/reject.
    using TensorType = typename TPSWaveFunctionComponent<TenElemT, QNT, WaveFunctionDress, ContractorT>::Tensor;
    std::vector<std::pair<SiteIdx, TensorType>> replacements{
        {site1, sitps(site1)[c2]},
        {site2, sitps(site2)[c1]},
    };
    std::vector<std::pair<SiteIdx, size_t>> new_cfgs{
        {site1, c2},
        {site2, c1},
    };
    auto trial = tps_component.BeginTrial(replacements, new_cfgs);
    const TenElemT psi_b = trial.amplitude;
    
    bool exchange = false;
    if (std::abs(psi_b) >= std::abs(psi_a)) {
      exchange = true;
    } else {
      double div = std::abs(psi_b) / std::abs(psi_a);
      double P = div * div;
      if (this->u_double_(this->random_engine_) < P) {
        exchange = true;
      }
    }
    
    if (exchange) {
        tps_component.AcceptTrial(std::move(trial), sitps);
    } else {
        tps_component.RejectTrial(std::move(trial));
    }
    
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
class MCUpdateSquareNNFullSpaceUpdateOBC : public MCUpdateSquareNNUpdateBaseOBC<MCUpdateSquareNNFullSpaceUpdateOBC> {
 public:
  template<typename TenElemT, typename QNT>
  bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                TPSWaveFunctionComponent<TenElemT, QNT> &tps_component) {
    auto &tn = tps_component.tn;
    auto &contractor = tps_component.contractor;
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
              contractor.ReplaceNNSiteTrace(tn, site1, site2, bond_dir,
                                    sitps(site1)[config1],
                                    sitps(site2)[config2]);
        }
      }
    }
    std::vector<double> weights(dim * dim);
    for (size_t i = 0; i < dim * dim; i++) {
      weights[i] = std::norm(alternative_psi[i] / tps_component.amplitude);
    }
    size_t final_state = SuwaTodoStateUpdate(init_config, weights, random_engine_);
    if (final_state == init_config) {
      return false;
    }

    tps_component.UpdateLocal(sitps, alternative_psi[final_state],
                              std::make_pair(site1, final_state / dim),
                              std::make_pair(site2, final_state % dim));
    return true;
  }
};

// Backward-compatible alias: never break userspace.
using MCUpdateSquareNNFullSpaceUpdate = MCUpdateSquareNNFullSpaceUpdateOBC;

/**
 * Full-space NN update for PBC using TRG trial/commit.
 * This updater does NOT assume Sz conservation.
 */
class MCUpdateSquareNNFullSpaceUpdatePBC : public MCUpdateSquareNNUpdateBasePBC<MCUpdateSquareNNFullSpaceUpdatePBC> {
 public:
  using MCUpdateSquareNNUpdateBasePBC<MCUpdateSquareNNFullSpaceUpdatePBC>::MCUpdateSquareNNUpdateBasePBC;

  template<typename TenElemT, typename QNT, typename WaveFunctionDress, template<typename, typename> class ContractorT>
  bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1, const SiteIdx &site2, BondOrientation /*bond_dir*/,
                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                TPSWaveFunctionComponent<TenElemT, QNT, WaveFunctionDress, ContractorT> &tps_component) {
    static_assert(std::is_same_v<ContractorT<TenElemT, QNT>, TRGContractor<TenElemT, QNT>>,
                  "MCUpdateSquareNNFullSpaceUpdatePBC requires TRGContractor.");

    const size_t dim = sitps.PhysicalDim();
    const size_t init_c1 = tps_component.config(site1);
    const size_t init_c2 = tps_component.config(site2);
    const size_t init_state = init_c1 * dim + init_c2;

    using TrialT = typename TPSWaveFunctionComponent<TenElemT, QNT, WaveFunctionDress, ContractorT>::Trial;
    std::vector<TrialT> trials;
    trials.reserve(dim * dim);

    std::vector<TenElemT> alternative_psi(dim * dim);
    for (size_t c1 = 0; c1 < dim; ++c1) {
      for (size_t c2 = 0; c2 < dim; ++c2) {
        const size_t state = c1 * dim + c2;
        if (state == init_state) {
          alternative_psi[state] = tps_component.amplitude;
          trials.emplace_back();
          continue;
        }
        using TensorType = typename TPSWaveFunctionComponent<TenElemT, QNT, WaveFunctionDress, ContractorT>::Tensor;
        std::vector<std::pair<SiteIdx, TensorType>> replacements{
            {site1, sitps(site1)[c1]},
            {site2, sitps(site2)[c2]},
        };
        std::vector<std::pair<SiteIdx, size_t>> new_cfgs{
            {site1, c1},
            {site2, c2},
        };
        auto trial = tps_component.BeginTrial(replacements, new_cfgs);
        alternative_psi[state] = trial.amplitude;
        trials.emplace_back(std::move(trial));
      }
    }

    std::vector<double> weights(dim * dim);
    const TenElemT psi_a = tps_component.amplitude;
    for (size_t i = 0; i < weights.size(); ++i) {
      const double r = std::abs(alternative_psi[i] / psi_a);
      weights[i] = r * r;
    }

    const size_t final_state = SuwaTodoStateUpdate(init_state, weights, random_engine_);
    if (final_state == init_state) {
      // No change; discard trials (if any).
      for (size_t i = 0; i < trials.size(); ++i) {
        if (i != init_state) {
          tps_component.RejectTrial(std::move(trials[i]));
        }
      }
      return false;
    }

    // Commit chosen trial, discard others.
    for (size_t i = 0; i < trials.size(); ++i) {
      if (i == final_state) {
        tps_component.AcceptTrial(std::move(trials[i]), sitps);
      } else if (i != init_state) {
        tps_component.RejectTrial(std::move(trials[i]));
      }
    }
    return true;
  }
};

/**
 * Monte Carlo update strategy for t-J model with Jastrow factor dressed wave function.
 * Local configuration: 0 = spin up, 1 = spin down, 2 = empty (hole).
 */
class MCUpdateSquareNNExchangeJastrowDressedTJ : public MCUpdateSquareNNUpdateBaseOBC<MCUpdateSquareNNExchangeJastrowDressedTJ,
                                                                                   JastrowDress> {
 public:
  using MCUpdateSquareNNUpdateBaseOBC<MCUpdateSquareNNExchangeJastrowDressedTJ, JastrowDress>::MCUpdateSquareNNUpdateBaseOBC;

  template<typename TenElemT, typename QNT>
  bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                TPSWaveFunctionComponent<TenElemT, QNT, JastrowDress> &tps_component) {
    size_t config1 = tps_component.config(site1);
    size_t config2 = tps_component.config(site2);

    if (config1 == config2) {
      return false;
    }

    // Compute new amplitude (PEPS part)
    TenElemT psi_b = tps_component.contractor.ReplaceNNSiteTrace(tps_component.tn, site1, site2, bond_dir,
                                                         sitps(site1)[config2],
                                                         sitps(site2)[config1]);

    // Compute Jastrow factor change
    double jastrow_ratio(0); //new divide old.
    auto &jastrow = tps_component.dress.jastrow;
    auto &density_config = tps_component.dress.density_config;
    if (density_config(site1) == density_config(site2)) {
      jastrow_ratio = 1.0;
    } else {
      // For t-J model, we know here one empty, one filled.
      double field_site1 = jastrow.JastrowFieldAtSite(density_config, site1);
      double field_site2 = jastrow.JastrowFieldAtSite(density_config, site2);
      if (density_config(site1) == 0 && density_config(site2) == 1) {
        jastrow_ratio = std::exp(field_site1 - field_site2);
      } else if (density_config(site1) == 1 && density_config(site2) == 0) {
        jastrow_ratio = std::exp(field_site2 - field_site1);
      } else {
        std::cerr << "Error: Invalid density configuration for t-J model." << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    // Metropolis-Hastings acceptance
    TenElemT &psi_a = tps_component.amplitude;
    double abs_ratio = std::abs(psi_b * jastrow_ratio) / std::abs(psi_a);
    double P = abs_ratio * abs_ratio;
    bool exchange = false;
    if (abs_ratio >= 1.0 || this->u_double_(this->random_engine_) < P) {
      exchange = true;
      size_t temp_config1 = tps_component.config(site2);
      size_t temp_config2 = tps_component.config(site1);
      tps_component.UpdateLocal(sitps, psi_b,
                                std::make_pair(site1, temp_config1),
                                std::make_pair(site2, temp_config2));
      tps_component.dress.UpdateLocalDensity(std::make_pair(site1, density_config(site2)),
                                             std::make_pair(site2, density_config(site1)));
    }
    return exchange;
  }
};

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_CONFIGURATION_UPDATE_STRATEGIES_SQUARE_TPS_SAMPLE_NN_EXCHANGE_H

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-23
*
* Description: QuantumLiquids/PEPS project. Explicit class of wave function component in square lattice. Monte Carlo sweep realized by NNN 3-site exchange.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_CLASSES_SQUARE_TPS_SAMPLE_3SITE_EXCHANGE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_CLASSES_SQUARE_TPS_SAMPLE_3SITE_EXCHANGE_H

#include "qlpeps/algorithm/vmc_update/wave_function_component.h"    // WaveFunctionComponent
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/monte_carlo_tools/non_detailed_balance_mcmc.h"     // NonDBMCMCStateUpdate

namespace qlpeps {
template<typename TenElemT, typename QNT>
class SquareTPSSample3SiteExchange : public WaveFunctionComponent<TenElemT, QNT> {
  using WaveFunctionComponentT = WaveFunctionComponent<TenElemT, QNT>;
 public:
  TensorNetwork2D<TenElemT, QNT> tn;

  SquareTPSSample3SiteExchange(const size_t rows, const size_t cols) : WaveFunctionComponentT(rows, cols),
                                                                       tn(rows, cols) {}

  SquareTPSSample3SiteExchange(const SplitIndexTPS<TenElemT, QNT> &sitps, const Configuration &config)
      : WaveFunctionComponentT(config), tn(config.rows(), config.cols()) {
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config);
    tn.GrowBMPSForRow(0, this->trun_para);
    tn.GrowFullBTen(RIGHT, 0, 2, true);
    tn.InitBTen(LEFT, 0);
    this->amplitude = tn.Trace({0, 0}, HORIZONTAL);
    if (!IsAmplitudeSquareLegal(this->amplitude)) {
      std::cout << "warning : wavefunction amplitude = "
                << this->amplitude
                << ", square of amplitude will be illegal! "
                << std::endl;
    }
  }

  /**
   * @param sitps
   * @param occupancy_num
   */
  void RandomInit(const SplitIndexTPS<TenElemT, QNT> &sitps,
                  const std::vector<size_t> &occupancy_num) {
    this->config.Random(occupancy_num);
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, this->config);
    tn.GrowBMPSForRow(0, this->trun_para);
    tn.GrowFullBTen(RIGHT, 0, 2, true);
    tn.InitBTen(LEFT, 0);
    this->amplitude = tn.Trace({0, 0}, HORIZONTAL);
    if (!IsAmplitudeSquareLegal(this->amplitude)) {
      std::cout << "warning : wavefunction amplitude = "
                << this->amplitude
                << ", square of amplitude will be illegal! "
                << std::endl;
    }
  }

  void MonteCarloSweepUpdate(const SplitIndexTPS<TenElemT, QNT> &sitps,
                             std::uniform_real_distribution<double> &u_double,
                             std::vector<double> &accept_rates) {
    size_t flip_accept_num = 0;
    tn.GenerateBMPSApproach(UP, this->trun_para);
    for (size_t row = 0; row < tn.rows(); row++) {
      tn.InitBTen(LEFT, row);
      tn.GrowFullBTen(RIGHT, row, 3, true);
      this->amplitude = tn.ReplaceTNNSiteTrace({row, 0}, HORIZONTAL,
                                               sitps({row, 0})[this->config({row, 0})],
                                               sitps({row, 1})[this->config({row, 1})],
                                               sitps({row, 2})[this->config({row, 2})]);
      for (size_t col = 0; col < tn.cols() - 2; col++) {
        flip_accept_num +=
            Exchange3SiteUpdate_({row, col}, {row, col + 1}, {row, col + 2}, HORIZONTAL, sitps, u_double);
        if (col < tn.cols() - 3) {
          tn.BTenMoveStep(RIGHT);
        }
      }
      if (row < tn.rows() - 1) {
        tn.BMPSMoveStep(DOWN, this->trun_para);
      }
    }

    tn.DeleteInnerBMPS(LEFT);
    tn.DeleteInnerBMPS(RIGHT);

    tn.GenerateBMPSApproach(LEFT, this->trun_para);
    for (size_t col = 0; col < tn.cols(); col++) {
      tn.InitBTen(UP, col);
      tn.GrowFullBTen(DOWN, col, 3, true);
      this->amplitude = tn.ReplaceTNNSiteTrace({0, col}, VERTICAL,
                                               sitps({0, col})[this->config({0, col})],
                                               sitps({1, col})[this->config({1, col})],
                                               sitps({2, col})[this->config({2, col})]);
      for (size_t row = 0; row < tn.rows() - 2; row++) {
        flip_accept_num += Exchange3SiteUpdate_({row, col}, {row + 1, col}, {row + 2, col}, VERTICAL, sitps, u_double);
        if (row < tn.rows() - 3) {
          tn.BTenMoveStep(DOWN);
        }
      }
      if (col < tn.cols() - 1) {
        tn.BMPSMoveStep(RIGHT, this->trun_para);
      }
    }

    tn.DeleteInnerBMPS(UP);
    double total_flip_num = tn.cols() * (tn.rows() - 2) + tn.rows() * (tn.cols() - 2);
    accept_rates = {double(flip_accept_num) / total_flip_num};
  }

 private:
  bool Exchange3SiteUpdate_(const SiteIdx &site1, const SiteIdx &site2, const SiteIdx &site3,
                            BondOrientation bond_dir,
                            const SplitIndexTPS<TenElemT, QNT> &sitps,
                            std::uniform_real_distribution<double> &u_double) {
    size_t spin1 = this->config(site1);
    size_t spin2 = this->config(site2);
    size_t spin3 = this->config(site3);
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
        psis[i] = tn.ReplaceTNNSiteTrace(site1, bond_dir,
                                         sitps(site1)[permutations[i][0]],
                                         sitps(site2)[permutations[i][1]],
                                         sitps(site3)[permutations[i][2]]);

      } else {
        psis[i] = this->amplitude;
      }
      psi_abs_max = std::max(psi_abs_max, std::abs(psis[i]));
    }
    std::vector<double> weights(permutations.size());
    for (size_t i = 0; i < weights.size(); i++) {
      weights[i] = std::norm(psis[i] / psi_abs_max);
    }

    size_t final_state = NonDBMCMCStateUpdate(init_state, weights, random_engine);
    if (final_state == init_state) {
      return false;
    }
    this->config(site1) = permutations[final_state][0];
    this->config(site2) = permutations[final_state][1];
    this->config(site3) = permutations[final_state][2];
    this->amplitude = psis[final_state];
    tn.UpdateSiteConfig(site1, this->config(site1), sitps);
    tn.UpdateSiteConfig(site2, this->config(site2), sitps);
    tn.UpdateSiteConfig(site3, this->config(site3), sitps);
    return true;
  }
}; //SquareTPSSample3SiteExchange

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_CLASSES_SQUARE_TPS_SAMPLE_3SITE_EXCHANGE_H

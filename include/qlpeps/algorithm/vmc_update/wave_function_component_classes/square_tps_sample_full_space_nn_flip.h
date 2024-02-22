/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-10
*
* Description: QuantumLiquids/PEPS project. Explicit class of wave function component in square lattice.
*              Monte Carlo sweep realized by NN bond flip without U1 quantum number conservation.
*/

#ifndef QLPEPS_VMC_PEPS_SQUARE_TPS_SAMPLE_FULL_SPACE_NN_FLIP_H
#define QLPEPS_VMC_PEPS_SQUARE_TPS_SAMPLE_FULL_SPACE_NN_FLIP_H

#include "qlpeps/algorithm/vmc_update/wave_function_component.h"    //WaveFunctionComponent
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/monte_carlo_tools/non_detailed_balance_mcmc.h"     // NonDBMCMCStateUpdate

namespace qlpeps {
template<typename TenElemT, typename QNT>
class SquareTPSSampleFullSpaceNNFlip : public WaveFunctionComponent<TenElemT, QNT> {
  using WaveFunctionComponentT = WaveFunctionComponent<TenElemT, QNT>;
 public:
  TensorNetwork2D<TenElemT, QNT> tn;

  SquareTPSSampleFullSpaceNNFlip(const size_t rows, const size_t cols) : WaveFunctionComponentT(rows, cols),
                                                                         tn(rows, cols) {}

  SquareTPSSampleFullSpaceNNFlip(const SplitIndexTPS<TenElemT, QNT> &sitps, const Configuration &config)
      : WaveFunctionComponentT(config), tn(config.rows(), config.cols()) {
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config);
    tn.GrowBMPSForRow(0, this->trun_para);
    tn.GrowFullBTen(RIGHT, 0, 2, true);
    tn.InitBTen(LEFT, 0);
    this->amplitude = tn.Trace({0, 0}, HORIZONTAL);
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
  }

  void MonteCarloSweepUpdate(const SplitIndexTPS<TenElemT, QNT> &sitps,
                             std::uniform_real_distribution<double> &u_double,
                             std::vector<double> &accept_rates) {
    size_t flip_accept_num = 0;
    tn.GenerateBMPSApproach(UP, this->trun_para);
    for (size_t row = 0; row < tn.rows(); row++) {
      tn.InitBTen(LEFT, row);
      tn.GrowFullBTen(RIGHT, row, 2, true);
      for (size_t col = 0; col < tn.cols() - 1; col++) {
        flip_accept_num += NNFlipUpdate_({row, col}, {row, col + 1}, HORIZONTAL, sitps, u_double);
        if (col < tn.cols() - 2) {
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
      tn.GrowFullBTen(DOWN, col, 2, true);
      for (size_t row = 0; row < tn.rows() - 1; row++) {
        flip_accept_num += NNFlipUpdate_({row, col}, {row + 1, col}, VERTICAL, sitps, u_double);
        if (row < tn.rows() - 2) {
          tn.BTenMoveStep(DOWN);
        }
      }
      if (col < tn.cols() - 1) {
        tn.BMPSMoveStep(RIGHT, this->trun_para);
      }
    }

    tn.DeleteInnerBMPS(UP);
    double bond_num = tn.cols() * (tn.rows() - 1) + tn.rows() * (tn.cols() - 1);
    accept_rates = {double(flip_accept_num) / bond_num};
  }

 private:
  bool NNFlipUpdate_(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                     const SplitIndexTPS<TenElemT, QNT> &sitps,
                     std::uniform_real_distribution<double> &u_double) {
    size_t dim = sitps.PhysicalDim();
    std::vector<TenElemT> alternative_psi(dim * dim);
    size_t init_config = this->config(site1) * dim + this->config(site2);
    assert(sitps(site1)[this->config(site1)].GetIndexes() == sitps(site1)[this->config(site2)].GetIndexes());
    alternative_psi[init_config] = this->amplitude;
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
      weights[i] = std::norm(alternative_psi[i] / alternative_psi[0]);
    }
    size_t final_state = NonDBMCMCStateUpdate(init_config, weights, u_double(random_engine));
    if (final_state == init_config) {
      return false;
    }
    this->config(site1) = final_state / dim;
    this->config(site2) = final_state % dim;
    this->amplitude = alternative_psi[final_state];
    tn.UpdateSiteConfig(site1, this->config(site1), sitps);
    tn.UpdateSiteConfig(site2, this->config(site2), sitps);
    return true;
  }
}; //SquareTPSSampleFullSpaceNNFlip

}//qlpeps

#endif //QLPEPS_VMC_PEPS_SQUARE_TPS_SAMPLE_FULL_SPACE_NN_FLIP_H

// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: GraceQ/VMC-PEPS project. Implementation for the variational Monte-Carlo PEPS
*/

#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H

#include <iomanip>
#include "gqpeps/algorithm/vmc_update/stochastic_reconfiguration_smatrix.h" //SRSMatrix
#include "gqpeps/utility/conjugate_gradient_solver.h"

namespace gqpeps {
using namespace gqten;

// helpers
template<typename T>
T Mean(const std::vector<T> data) {
  if (data.empty()) {
    return T(0);
  }
  auto const count = static_cast<T>(data.size());
//  return std::reduce(data.begin(), data.end()) / count;
  return std::accumulate(data.begin(), data.end(), T(0)) / count;
}

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> Mean(const std::vector<GQTensor<TenElemT, QNT> *> &tensor_list,
                             const size_t length) {
  std::vector<TenElemT> coefs(tensor_list.size(), TenElemT(1.0));
  GQTensor<TenElemT, QNT> sum;
  LinearCombine(coefs, tensor_list, TenElemT(0.0), &sum);
  return sum * (1.0 / double(length));
}

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> MPIMeanTensor(const GQTensor<TenElemT, QNT> &tensor,
                                      boost::mpi::communicator &world) {
  using Tensor = GQTensor<TenElemT, QNT>;
  if (world.rank() == kMasterProc) {
    std::vector<Tensor *> ten_list(world.size(), nullptr);
    for (size_t proc = 0; proc < world.size(); proc++) {
      if (proc != kMasterProc) {
        ten_list[proc] = new Tensor();
        recv_gqten(world, proc, 2 * proc, *ten_list[proc]);
      } else {
        ten_list[proc] = new Tensor(tensor);
      }
    }
    return Mean(ten_list, world.size());
  } else {
    send_gqten(world, kMasterProc, 2 * world.rank(), tensor);
    return Tensor();
  }
}

/// Note the definition
template<typename T>
T Variance(const std::vector<T> data,
           const T &mean) {
  size_t data_size = data.size();
  std::vector<T> diff(data_size);
  std::transform(data.begin(), data.end(), diff.begin(), [mean](double x) { return x - mean; });
  T sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  auto const count = static_cast<T>(data_size);
  return sq_sum / count;
}

template<typename T>
T StandardError(const std::vector<T> data,
                const T &mean) {
  return std::sqrt(Variance(data, mean));
}

template<typename T>
T Variance(const std::vector<T> data) {
  return Variance(data, Mean(data));
}

template<typename TenElemT, typename QNT, typename EnergySolver>
VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                                                              const TPST &tps_init,
                                                              const boost::mpi::communicator &world,
                                                              const EnergySolver &solver) :
    VMCPEPSExecutor<TenElemT, QNT, EnergySolver>(optimize_para, SITPST(tps_init), world, solver) {}

template<typename TenElemT, typename QNT, typename EnergySolver>
VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                                                              const SITPST &sitpst_init,
                                                              const boost::mpi::communicator &world,
                                                              const EnergySolver &solver) :
    world_(world),
    optimize_para(optimize_para),
    lx_(sitpst_init.cols()),
    ly_(sitpst_init.rows()),
    split_index_tps_(sitpst_init),
    tps_sample_(ly_, lx_),
    u_double_(0, 1),
    grad_(ly_, lx_),
//    gten_samples_(ly_, lx_),
//    g_times_energy_samples_(ly_, lx_),
    gten_sum_(ly_, lx_),
    g_times_energy_sum_(ly_, lx_),
    energy_solver_(solver),
    warm_up_(false) {
  random_engine.seed((size_t)
                         std::time(nullptr) + 10086 * world.rank());
  TPSSample<TenElemT, QNT>::trun_para = TruncatePara(optimize_para);
  tps_sample_.RandomInit(split_index_tps_, optimize_para.occupancy_num, 10087 * world.rank() + std::time(nullptr));

  ReserveSamplesDataSpace_();
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename EnergySolver>
VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                                                              const size_t ly, const size_t lx,
                                                              const boost::mpi::communicator &world,
                                                              const EnergySolver &solver):
    world_(world), optimize_para(optimize_para), lx_(lx), ly_(ly),
    split_index_tps_(ly, lx), tps_sample_(ly, lx),
    u_double_(0, 1), grad_(ly_, lx_),
//    gten_samples_(ly_, lx_),
//    g_times_energy_samples_(ly_, lx_),
    gten_sum_(ly_, lx_), g_times_energy_sum_(ly_, lx_),
    energy_solver_(solver), warm_up_(false) {
  TPSSample<TenElemT, QNT>::trun_para = TruncatePara(optimize_para);
  random_engine.seed((size_t)
                         std::time(nullptr) + 10086 * world.rank());
  LoadTenData();
  ReserveSamplesDataSpace_();
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}


template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::ReserveSamplesDataSpace_(void) {
  energy_samples_.reserve(optimize_para.mc_samples);
  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      size_t dim = split_index_tps_({row, col}).size();

//      gten_samples_({row, col}) = std::vector(dim, std::vector<Tensor *>());
//      g_times_energy_samples_({row, col}) = std::vector(dim, std::vector<Tensor *>());
//      for (size_t i = 0; i < dim; i++) {
//        gten_samples_({row, col})[i].reserve(optimize_para.mc_samples);
//        g_times_energy_samples_({row, col})[i].reserve(optimize_para.mc_samples);
//      }

      gten_sum_({row, col}) = std::vector<Tensor>(dim, Tensor(split_index_tps_({row, col})[0].GetIndexes()));
      g_times_energy_sum_({row, col}) = gten_sum_({row, col});
    }
  }
  for (size_t row = 0; row < ly_; row++)
    for (size_t col = 0; col < lx_; col++) {
      size_t dim = split_index_tps_({row, col}).size();
      grad_({row, col}) = std::vector<Tensor>(dim);
    }

  energy_trajectory_.reserve(optimize_para.step_lens.size());
  energy_error_traj_.reserve(optimize_para.step_lens.size());
  if (world_.rank() == kMasterProc)
    grad_norm_.reserve(optimize_para.step_lens.size());

  if (optimize_para.update_scheme == StochasticReconfiguration) {
    gten_samples_.reserve(optimize_para.mc_samples);
  }
}


template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::PrintExecutorInfo_(void) {
  if (world_.rank() == kMasterProc) {
    std::cout << std::left;  // Set left alignment for the output
    std::cout << "\n";
    std::cout << "=====> VARIATIONAL MONTE-CARLO PROGRAM FOR PEPS <=====" << "\n";
    std::cout << std::setw(30) << "System size (lx, ly):" << "(" << lx_ << ", " << ly_ << ")\n";
    std::cout << std::setw(30) << "PEPS bond dimension:" << split_index_tps_.GetMaxBondDimension() << "\n";
    std::cout << std::setw(30) << "BMPS bond dimension:" << optimize_para.bmps_trunc_para.D_min << "/"
              << optimize_para.bmps_trunc_para.D_max << "\n";
    std::cout << std::setw(30) << "Sampling numbers:" << optimize_para.mc_samples << "\n";
    std::cout << std::setw(30) << "Gradient update times:" << optimize_para.step_lens.size() << "\n";

    std::cout << "=====> TECHNICAL PARAMETERS <=====" << "\n";
    std::cout << std::setw(40) << "The number of processors (including master):" << world_.size() << "\n";
    std::cout << std::setw(40) << "The number of threads per processor:" << hp_numeric::GetTensorManipulationThreads()
              << "\n";
  }
}


template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);
  WarmUp_();
  OptimizeTPS_();
  Measure_();
  DumpTenData();
  SetStatus(ExecutorStatus::FINISH);
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::WarmUp_(void) {
  if (!warm_up_) {
    Timer warm_up_timer("warm_up");
    for (size_t sweep = 0; sweep < optimize_para.mc_warm_up_sweeps; sweep++) {
      MCSweep_();
    }
    double elasp_time = warm_up_timer.Elapsed();
    std::cout << "Proc " << std::setw(4) << world_.rank() << " warm-up completes T = " << elasp_time << "s."
              << std::endl;
    warm_up_ = true;
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::OptimizeTPS_(void) {
  size_t bond_num = lx_ * (ly_ - 1) + ly_ * (lx_ - 1);
  for (size_t iter = 0; iter < optimize_para.step_lens.size(); iter++) {
    Timer grad_update_timer("gradient_update");
    double step_len = optimize_para.step_lens[iter];
    size_t accept_num = 0;
    for (size_t sweep = 0; sweep < optimize_para.mc_samples; sweep++) {
      accept_num += MCSweep_();
      SampleEnergyAndHols_();
    }
    double accept_rate = double(accept_num) / double(bond_num * optimize_para.mc_samples);
    GatherStatisticEnergyAndGrad_();

    Timer tps_update_timer("tps_update");
    size_t sr_iter;
    if (optimize_para.update_scheme == StochasticGradient) {
      StochGradUpdateTPS_(grad_, step_len);
    } else if (optimize_para.update_scheme == RandomStepStochasticGradient) {
      step_len *= u_double_(random_engine);
      StochGradUpdateTPS_(grad_, step_len);
    } else if (optimize_para.update_scheme == StochasticReconfiguration) {
      sr_iter = StochReconfigUpdateTPS_(grad_, step_len);
    } else {
      std::cout << "update method does not support!" << std::endl;
      exit(2);
    }
    double tps_update_time = tps_update_timer.Elapsed();

    ClearEnergyAndHoleSamples_();
    if (world_.rank() == kMasterProc) {
      double gradient_update_time = grad_update_timer.Elapsed();
      std::cout << "Iter " << std::setw(4) << iter
                << "Alpha = " << std::setw(9) << std::scientific << std::setprecision(1) << step_len
                << "E0 = " << std::setw(14) << std::fixed << std::setprecision(kEnergyOutputPrecision)
                << energy_trajectory_.back()
                << "+- " << std::setw(10) << std::scientific << std::setprecision(2) << energy_error_traj_.back()
                << "Grad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << grad_norm_.back()
                << "Accept rate = " << std::setw(5) << std::fixed << std::setprecision(2) << accept_rate;

      if (optimize_para.update_scheme == StochasticReconfiguration) {
        std::cout << "SRSolver Iter = " << std::setw(4) << sr_iter;
      }
      std::cout << "TPS UpdateT = " << std::setw(6) << std::fixed << std::setprecision(2) << tps_update_time << "s"
                << " TotT = " << std::setw(8) << std::fixed << std::setprecision(2) << gradient_update_time << "s"
                << "\n";
    }
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::ClearEnergyAndHoleSamples_(void) {
  energy_samples_.clear();
//  for (size_t row = 0; row < ly_; row++) {
//    for (size_t col = 0; col < lx_; col++) {
//      const size_t phy_dim = split_index_tps_({row, col}).size();
//      for (size_t basis = 0; basis < phy_dim; basis++) {
//        auto &g_sample = gten_samples_({row, col})[basis];
//        auto &ge_sample = g_times_energy_samples_({row, col})[basis];
//        for (size_t i = 0; i < g_sample.size(); i++) {
//          delete g_sample[i];
//          delete ge_sample[i];
//        }
//        g_sample.clear();
//        ge_sample.clear();
//      }
//    }
//  }
  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      size_t dim = split_index_tps_({row, col}).size();

      gten_sum_({row, col}) = std::vector<Tensor>(dim, Tensor(split_index_tps_({row, col})[0].GetIndexes()));
      g_times_energy_sum_({row, col}) = gten_sum_({row, col});
    }
  }
  if (optimize_para.update_scheme == StochasticReconfiguration) {
    gten_samples_.clear();
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::SampleEnergyAndHols_(void) {
  TensorNetwork2D<TenElemT, QNT> holes(ly_, lx_);
  TenElemT energy_loc = energy_solver_.CalEnergyAndHoles(&split_index_tps_, &tps_sample_, holes);
  TenElemT inv_psi = 1.0 / tps_sample_.amplitude;
  energy_samples_.push_back(energy_loc);
  SITPST gten_sample(ly_, lx_, split_index_tps_.PhysicalDim());// only useful for Stochastic Reconfiguration
  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      size_t basis = tps_sample_.config({row, col});
//      Tensor *g_ten = new Tensor(), *gten_times_energy = new Tensor();
//      *g_ten = inv_psi * holes({row, col});
//      *gten_times_energy = energy_loc * (*g_ten);
//      gten_samples_({row, col})[basis].push_back(g_ten);
//      g_times_energy_samples_({row, col})[basis].push_back(gten_times_energy);
      Tensor gten = inv_psi * holes({row, col});
      gten_sum_({row, col})[basis] += gten;
      g_times_energy_sum_({row, col})[basis] += energy_loc * gten;
      //? when samples become large, does the summation reliable as the small number are added to large number.
      if (optimize_para.update_scheme == StochasticReconfiguration) {
        gten_sample({row, col})[basis] = gten;
      }
    }
  }
  if (optimize_para.update_scheme == StochasticReconfiguration) {
    gten_samples_.emplace_back(gten_sample);
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
SplitIndexTPS<TenElemT, QNT>
VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::GatherStatisticEnergyAndGrad_(void) {
  TenElemT en_self = Mean(energy_samples_); //energy value in each processor
  std::vector<TenElemT> en_list;
  boost::mpi::gather(world_, en_self, en_list, kMasterProc);
  TenElemT energy, en_err;
  if (world_.rank() == 0) {
    energy = Mean(en_list);
    en_err = StandardError(en_list, energy);
  }
  boost::mpi::broadcast(world_, energy, kMasterProc);
  boost::mpi::broadcast(world_, en_err, kMasterProc);
  energy_trajectory_.push_back(energy);
  energy_error_traj_.push_back(en_err);


  //calculate grad in each processor
  const size_t sample_num = optimize_para.mc_samples;
  gten_ave_ = gten_sum_ * (1.0 / sample_num);
  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      const size_t phy_dim = grad_({row, col}).size();
      for (size_t compt = 0; compt < phy_dim; compt++) {
//        if (g_times_energy_samples_({row, col})[compt].size() == 0) {
//          grad_({row, col})[compt] = Tensor(split_index_tps_({row, col})[compt].GetIndexes());
//        } else {
//          grad_({row, col})[compt] =
//              Mean(g_times_energy_samples_({row, col})[compt], sample_num) +
//              (-energy) * Mean(gten_samples_({row, col})[compt], sample_num);
//        }
        grad_({row, col})[compt] = g_times_energy_sum_({row, col})[compt] * (1.0 / sample_num)
                                   + (-energy) * gten_ave_({row, col})[compt];
      }
    }
  }
  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      const size_t phy_dim = grad_({row, col}).size();
      for (size_t compt = 0; compt < phy_dim; compt++) {
        // gather and estimate grad in master (and maybe the error bar of grad)
        grad_({row, col})[compt] = MPIMeanTensor(grad_({row, col})[compt], world_);
        // note here the grad data except in master are clear
        if (optimize_para.update_scheme == StochasticReconfiguration) {
          gten_ave_({row, col})[compt] = MPIMeanTensor(gten_ave_({row, col})[compt], world_);
        }
      }
    }
  }
  if (world_.rank() == kMasterProc) {
    grad_norm_.push_back(grad_.Norm());
  }
  //do not broad cast because only broad cast the updated TPS
  return grad_;
}

/**
 * Stochastic gradient descent update peps
 *
 * @tparam TenElemT
 * @tparam QNT
 * @tparam EnergySolver
 * @param grad
 * @param step_len
 */
template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::StochGradUpdateTPS_(const VMCPEPSExecutor::SITPST &grad,
                                                                       double step_len) {
  if (world_.rank() == kMasterProc) {
    for (size_t row = 0; row < ly_; row++)
      for (size_t col = 0; col < lx_; col++) {
        const size_t phy_dim = grad_({row, col}).size();
        double norm = 0;
        for (size_t compt = 0; compt < phy_dim; compt++) {
          split_index_tps_({row, col})[compt] += (-step_len) * grad({row, col})[compt];
          norm += split_index_tps_({row, col})[compt].Get2Norm();
        }
        double inv_norm = 1.0 / norm;
        for (size_t compt = 0; compt < phy_dim; compt++) {
          split_index_tps_({row, col})[compt] *= inv_norm;
          SendBroadCastGQTensor(world_, split_index_tps_({row, col})[compt], kMasterProc);
        }
      }
  } else {
    for (size_t row = 0; row < ly_; row++)
      for (size_t col = 0; col < lx_; col++) {
        const size_t phy_dim = grad_({row, col}).size();
        for (size_t compt = 0; compt < phy_dim; compt++) {
          split_index_tps_({row, col})[compt] = Tensor();
          RecvBroadCastGQTensor(world_, split_index_tps_({row, col})[compt], kMasterProc);
        }
      }
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
size_t VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::StochReconfigUpdateTPS_(
    const VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::SITPST &grad, double step_len) {
  SITPST *pgten_ave_ = &gten_ave_;
  if (world_.rank() != kMasterProc) {
    pgten_ave_ = nullptr;
  }
  SRSMatrix s_matrix(&gten_samples_, pgten_ave_, world_.size());
  s_matrix.diag_shift = 0.01;
  size_t iter;
  auto natural_grad = ConjugateGradientSolver(s_matrix, grad, grad,
                                              cg_params.max_iter, cg_params.tolerance,
                                              cg_params.residue_restart_step, iter, world_);
  StochGradUpdateTPS_(natural_grad, step_len);
  return iter;
}

template<typename TenElemT, typename QNT, typename EnergySolver>
size_t VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::MCSweep_(void) {
  if (optimize_para.mc_sweep_sheme == SequentiallyNNSiteFlip)
    return tps_sample_.MCSequentiallyNNFlipSweep(split_index_tps_, u_double_);
  else if (optimize_para.mc_sweep_sheme == CompressedLatticeKagomeLocalUpdate)
    return tps_sample_.MCCompressedKagomeLatticeLocalUpdateSweep(split_index_tps_, u_double_);
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::MCUpdateNNSite_(const SiteIdx &site_a, BondOrientation dir) {
  SiteIdx site_b(site_a);
  switch (dir) {
    case HORIZONTAL: {
      site_b[1]++;
      break;
    }
    case VERTICAL: {
      site_b[0]++;
      break;
    }
  }
  tps_sample_.ExchangeUpdate(site_a, site_b, dir, split_index_tps_, u_double_);
}


template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::Measure_(void) {

}


template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::LoadTenData(void) {
  LoadTenData(optimize_para.wavefunction_path);
}


template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::LoadTenData(const std::string &tps_path) {
  if (!split_index_tps_.Load(tps_path)) {
    std::cout << "Loading TPS files fails." << std::endl;
    exit(-1);
  }
  Configuration config(ly_, lx_);
  bool load_config = config.Load(tps_path, world_.rank());
  if (load_config) {
    tps_sample_ = TPSSample<TenElemT, QNT>(split_index_tps_, config, TruncatePara(optimize_para));
  } else {
    std::cout << "Loading configuration in rank " << world_.rank()
              << " fails. Random generate it and warm up."
              << std::endl;
    tps_sample_.RandomInit(split_index_tps_, optimize_para.occupancy_num, 10089 * world_.rank() + std::time(nullptr));
    WarmUp_();
  }
  warm_up_ = true;
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::DumpTenData(const bool release_mem) {
  DumpTenData(optimize_para.wavefunction_path, release_mem);
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::DumpTenData(const std::string &tps_path, const bool release_mem) {
  if (world_.rank() == kMasterProc) {
    split_index_tps_.Dump(tps_path, release_mem);
  }
  world_.barrier(); // configurations dump will collapse when creating path if there is no barrier.
  tps_sample_.config.Dump(tps_path, world_.rank());
}

}//gqpeps

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H

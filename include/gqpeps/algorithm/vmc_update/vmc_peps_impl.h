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
                                                              const boost::mpi::communicator &world) :
    VMCPEPSExecutor<TenElemT, QNT, EnergySolver>(optimize_para, SITPST(tps_init), world) {}

template<typename TenElemT, typename QNT, typename EnergySolver>
VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                                                              const SITPST &sitpst_init,
                                                              const boost::mpi::communicator &world) :
    world_(world),
    optimize_para(optimize_para),
    lx_(sitpst_init.cols()),
    ly_(sitpst_init.rows()),
    split_index_tps_(sitpst_init),
    tps_sample_(ly_, lx_, TruncatePara(optimize_para)),
    u_double_(0, 1),
    energy_solver_(&split_index_tps_, &tps_sample_),
    gten_samples_(ly_, lx_),
    g_times_energy_samples_(ly_, lx_),
    grad_(ly_, lx_) {
  tps_sample_.RandomInit(split_index_tps_, optimize_para.occupancy_num);

  energy_samples_.reserve(optimize_para.mc_samples);
  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      size_t dim = split_index_tps_({row, col}).size();
      gten_samples_({row, col}) = std::vector(dim, std::vector<Tensor *>());
      g_times_energy_samples_({row, col}) = std::vector(dim, std::vector<Tensor *>());
      for (size_t i = 0; i < dim; i++) {
        gten_samples_({row, col})[i].reserve(optimize_para.mc_samples);
        g_times_energy_samples_({row, col})[i].reserve(optimize_para.mc_samples);
      }
    }
  }
  for (size_t row = 0; row < ly_; row++)
    for (size_t col = 0; col < lx_; col++) {
      size_t dim = split_index_tps_({row, col}).size();
      grad_({row, col}) = std::vector<Tensor>(dim);
    }

  energy_trajectory_.reserve(optimize_para.step_lens.size());
  energy_error_traj_.reserve(optimize_para.step_lens.size());
  if (world.rank() == kMasterProc)
    grad_norm_.reserve(optimize_para.step_lens.size());

  std::cout << std::left;  // Set left alignment for the output

  if (world_.rank() == kMasterProc) {
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

  this->SetStatus(ExecutorStatus::INITED);

}


template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);
  WarmUp_();
  OptimizeTPS_();
  Measure_();
  SetStatus(ExecutorStatus::FINISH);
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::WarmUp_(void) {
  Timer warm_up_timer("warm_up");
  for (size_t sweep = 0; sweep < optimize_para.mc_warm_up_sweeps; sweep++) {
    MCSweepSequentially_();
  }
  double elasp_time = warm_up_timer.Elapsed();
  std::cout << "Proc " << world_.rank() << " warm-up completes T = " << elasp_time << "." << std::endl;
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::OptimizeTPS_(void) {
  for (size_t iter = 0; iter < optimize_para.step_lens.size(); iter++) {
    Timer grad_update_timer("gradient_update");
    double step_len = optimize_para.step_lens[iter];
    for (size_t sweep = 0; sweep < optimize_para.mc_samples; sweep++) {
      MCSweepSequentially_();
      SampleEnergyAndHols_();
    }
    GatherStatisticEnergyAndGrad_();
    GradUpdateTPS_(grad_, step_len);
    ClearEnergyAndHoleSamples_();
    if (world_.rank() == kMasterProc) {
      double gradient_update_time = grad_update_timer.Elapsed();
      std::cout << "Iter " << std::setw(4) << iter
                << "  Step length = " << std::setw(7) << std::scientific << std::setprecision(1) << step_len
                << "  E0 = " << std::setw(14) << std::fixed << std::setprecision(kEnergyOutputPrecision)
                << energy_trajectory_.back()
                << " +- " << std::setw(10) << std::scientific << std::setprecision(4) << energy_error_traj_.back()
                << " Grad Norm = " << std::setw(7) << std::scientific << std::setprecision(1) << grad_norm_.back()
                << "  TotT = " << std::setw(10) << std::fixed << std::setprecision(2) << gradient_update_time << "s"
                << "\n";
      //should output the magnitude of grad?
    }
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::ClearEnergyAndHoleSamples_(void) {
  energy_samples_.clear();
  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      for (size_t basis = 0; basis < gten_samples_({row, col}).size(); basis++) {
        auto &g_sample = gten_samples_({row, col})[basis];
        auto &ge_sample = g_times_energy_samples_({row, col})[basis];
        for (size_t i = 0; i < g_sample.size(); i++) {
          delete g_sample[i];
          delete ge_sample[i];
        }
        g_sample.clear();
        ge_sample.clear();
      }
    }
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::SampleEnergyAndHols_(void) {
  TensorNetwork2D<TenElemT, QNT> holes(ly_, lx_);
  TenElemT energy_loc = energy_solver_.CalEnergyAndHoles(holes);
  TenElemT inv_psi = 1.0 / tps_sample_.amplitude;
  energy_samples_.push_back(energy_loc);
  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      size_t basis = tps_sample_.config({row, col});
      Tensor *g_ten = new Tensor(), *gten_times_energy = new Tensor();
      *g_ten = inv_psi * holes({row, col});
      *gten_times_energy = energy_loc * (*g_ten);
      gten_samples_({row, col})[basis].push_back(g_ten);
      g_times_energy_samples_({row, col})[basis].push_back(gten_times_energy);
    }
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
  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      const size_t phy_dim = grad_({row, col}).size();
      for (size_t compt = 0; compt < phy_dim; compt++) {
        if (g_times_energy_samples_({row, col})[compt].size() == 0) {
          grad_({row, col})[compt] = Tensor(split_index_tps_({row, col})[compt].GetIndexes());
        } else {
          grad_({row, col})[compt] =
              Mean(g_times_energy_samples_({row, col})[compt], sample_num) +
              (-energy) * Mean(gten_samples_({row, col})[compt], sample_num);
        }

      }
    }
  }
  double grad_norm(0.0);
  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      const size_t phy_dim = grad_({row, col}).size();
      for (size_t compt = 0; compt < phy_dim; compt++) {
        // gather and estimate grad in master (and maybe the error bar of grad)
        grad_({row, col})[compt] = MPIMeanTensor(grad_({row, col})[compt], world_);
        // note here the grad data except in master are clear
      }
    }
  }
  if (world_.rank() == kMasterProc) {
    for (size_t row = 0; row < ly_; row++) {
      for (size_t col = 0; col < lx_; col++) {
        const size_t phy_dim = grad_({row, col}).size();
        for (size_t compt = 0; compt < phy_dim; compt++) {
          grad_norm += grad_({row, col})[compt].Get2Norm();
        }
      }
    }
    grad_norm_.push_back(grad_norm);
  }
  //do not broad cast because only broad cast the updated TPS
  return grad_;
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::GradUpdateTPS_(const VMCPEPSExecutor::SITPST &grad,
                                                                  const double step_len) {
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
          RecvBroadCastGQTensor(world_, split_index_tps_({row, col})[compt], kMasterProc);
        }
      }
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::MCSweepSequentially_(void) {
  tps_sample_.MCSequentiallySweep(split_index_tps_, u_double_);
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
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::LoadTenData(const std::string &tps_path) {

}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::DumpTenData(const std::string &tps_path, const bool release_mem) {
  if (world_.rank() == kMasterProc)
    split_index_tps_.Dump(tps_path, release_mem);
}

}//gqpeps

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H

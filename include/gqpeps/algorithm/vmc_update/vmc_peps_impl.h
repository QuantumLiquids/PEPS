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
#include "axis_update.h"

namespace gqpeps {
using namespace gqten;

// helpers
template<typename DataType>
void DumpVecData(
    const std::string &filename,
    const std::vector<DataType> &data
) {
  std::ofstream ofs(filename, std::ofstream::binary);
  for (auto datum : data) {
    ofs << datum << '\n';
  }
  ofs << std::endl;
  ofs.close();
}

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

///< only rank 0 get the values
std::pair<double, double> StatisticFromProcessors(
    double data,
    MPI_Comm comm) {
  double mean(0), standard_err(0);
  int comm_rank, comm_size;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  double *gather_data;
  if (comm_rank == kMasterProc) {
    gather_data = new double[comm_size];
  }
  int err_msg = ::MPI_Gather((void *) &data, 1, MPI_DOUBLE, (void *) gather_data, 1, MPI_DOUBLE, kMasterProc, comm);

  if (comm_rank == kMasterProc) {
    double sum = 0.0;
    for (size_t i = 0; i < comm_size; i++) {
      sum += *(gather_data + i);
    }
    mean = sum / comm_size;
    if (comm_size > 1) {
      double sum_square = 0.0;
      for (size_t i = 0; i < comm_size; i++) {
        sum_square += gather_data[i] * gather_data[i];
      }
      double variance = sum_square / comm_size - mean * mean;
      standard_err = std::sqrt(variance / (comm_size - 1));
    }
    delete gather_data;
  }
  return std::make_pair(mean, standard_err);
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
    Tensor res = Mean(ten_list, world.size());
    for (auto pten : ten_list) {
      delete pten;
    }
    return res;
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
  return std::sqrt(Variance(data, mean) / (T) data.size());
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
    natural_grad_(ly_, lx_),
//    gten_samples_(ly_, lx_),
//    g_times_energy_samples_(ly_, lx_),
    gten_sum_(ly_, lx_),
    g_times_energy_sum_(ly_, lx_),
    energy_solver_(solver),
    warm_up_(false) {
  random_engine.seed(std::random_device{}() + 10086 * world.rank());
  TPSSample<TenElemT, QNT>::trun_para = BMPSTruncatePara(optimize_para);
  tps_sample_ = TPSSample<TenElemT, QNT>(sitpst_init, optimize_para.init_config);
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
    u_double_(0, 1), grad_(ly_, lx_), natural_grad_(ly_, lx_),
//    gten_samples_(ly_, lx_),
//    g_times_energy_samples_(ly_, lx_),
    gten_sum_(ly_, lx_), g_times_energy_sum_(ly_, lx_),
    energy_solver_(solver), warm_up_(false) {
  TPSSample<TenElemT, QNT>::trun_para = BMPSTruncatePara(optimize_para);
  random_engine.seed(std::random_device{}() + 10086 * world.rank());
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
  if (world_.rank() == 0) {
    energy_trajectory_.reserve(optimize_para.step_lens.size());
    energy_error_traj_.reserve(optimize_para.step_lens.size());
  }

  if (world_.rank() == kMasterProc)
    grad_norm_.reserve(optimize_para.step_lens.size());

  if (optimize_para.update_scheme == StochasticReconfiguration ||
      optimize_para.update_scheme == RandomStepStochasticReconfiguration ||
      optimize_para.update_scheme == NormalizedStochasticReconfiguration) {
    gten_samples_.reserve(optimize_para.mc_samples);
    for (size_t row = 0; row < ly_; row++)
      for (size_t col = 0; col < lx_; col++) {
        size_t dim = split_index_tps_({row, col}).size();
        natural_grad_({row, col}) = std::vector<Tensor>(dim);
      }
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::PrintExecutorInfo_(void) {
  if (world_.rank() == kMasterProc) {
    std::cout << std::left;  // Set left alignment for the output
    std::cout << "\n";
    std::cout << "=====> VARIATIONAL MONTE-CARLO PROGRAM FOR PEPS <=====" << "\n";
    std::cout << std::setw(30) << "System size (lx, ly):" << "(" << lx_ << ", " << ly_ << ")\n";
    std::cout << std::setw(30) << "PEPS bond dimension:" << split_index_tps_.GetMinBondDimension() << "/"
              << split_index_tps_.GetMaxBondDimension() << "\n";
    std::cout << std::setw(30) << "BMPS bond dimension:" << optimize_para.bmps_trunc_para.D_min << "/"
              << optimize_para.bmps_trunc_para.D_max << "\n";
    std::cout << std::setw(30) << "Sampling numbers:" << optimize_para.mc_samples << "\n";
    std::cout << std::setw(30) << "Gradient update times:" << optimize_para.step_lens.size() << "\n";
    std::cout << std::setw(30) << "PEPS update strategy:" << optimize_para.update_scheme << "\n";

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
  if (optimize_para.update_scheme == GradientLineSearch || optimize_para.update_scheme == NaturalGradientLineSearch) {
    LineSearchOptimizeTPS_();
  } else {
    IterativeOptimizeTPS_();
  }
  Measure_();
  DumpData();
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
    std::cout << "Proc " << std::setw(4) << world_.rank() << " warm up completes T = " << elasp_time << "s."
              << std::endl;
    warm_up_ = true;
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::LineSearchOptimizeTPS_(void) {
  size_t flip_bond_num = lx_ * (ly_ - 1) + ly_ * (lx_ - 1);
  size_t cluster_num = 3 * lx_ * ly_;
  size_t bond_flip_accept_num = 0;
  size_t cluster_update_accept_num = 0;
  ClearEnergyAndHoleSamples_();

  Timer grad_calculation_timer("gradient_calculation");
  for (size_t sweep = 0; sweep < optimize_para.mc_samples; sweep++) {
    std::vector<size_t> accept_nums = MCSweep_();
    bond_flip_accept_num += accept_nums[0];
    if (accept_nums.size() > 1) {
      cluster_update_accept_num += accept_nums[1];
    }
    SampleEnergyAndHols_();
  }
  double bond_accept_rate = double(bond_flip_accept_num) / double(flip_bond_num * optimize_para.mc_samples);
  double cluster_accept_rate = double(cluster_update_accept_num) / double(cluster_num * optimize_para.mc_samples);
  GatherStatisticEnergyAndGrad_();
  size_t cgsolver_iter(0);
  double sr_natural_grad_norm(0.0);
  SITPST *search_dir(nullptr);
  switch (optimize_para.update_scheme) {
    case GradientLineSearch: {
      if (world_.rank() == kMasterProc)
        search_dir = &grad_;
      break;
    }
    case NaturalGradientLineSearch: {
      auto init_guess = SITPST(ly_, lx_, split_index_tps_.PhysicalDim());
      cgsolver_iter = CalcNaturalGradient_(grad_, init_guess);
      if (world_.rank() == kMasterProc) {
        search_dir = &natural_grad_;
        sr_natural_grad_norm = natural_grad_.Norm();
      }
      break;
    }
    default: {
      std::cerr << "update scheme is not line search scheme." << std::endl;
      exit(1);
    }
  }

  if (world_.rank() == kMasterProc) {
    double gradient_calculation_time = grad_calculation_timer.Elapsed();
    std::cout << "Initial Search Direction Calculation :\n"
              << "E0 = " << std::setw(14) << std::fixed << std::setprecision(kEnergyOutputPrecision)
              << energy_trajectory_.back()
              << pm_sign << " " << std::setw(10) << std::scientific << std::setprecision(2)
              << energy_error_traj_.back()
              << "Grad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << grad_norm_.back()
              << "Accept rate = " << std::setw(5) << std::fixed << std::setprecision(2) << bond_accept_rate;
    if (optimize_para.mc_sweep_scheme == CompressedLatticeKagomeLocalUpdate) {
      std::cout << std::setw(5) << std::fixed << std::setprecision(2) << cluster_accept_rate;
    }
    if (optimize_para.update_scheme == NaturalGradientLineSearch) {
      std::cout << "SRSolver Iter = " << std::setw(4) << cgsolver_iter;
      std::cout << "NGrad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << sr_natural_grad_norm;
    }
    std::cout << " TotT = " << std::setw(8) << std::fixed << std::setprecision(2) << gradient_calculation_time << "s"
              << "\n";
  }

  double e0_min;
  if (world_.rank() == kMasterProc) {
    e0_min = energy_trajectory_[0];
  }
  SplitIndexTPS tps_min = split_index_tps_;
  for (size_t point = 0; point < optimize_para.step_lens.size(); point++) {
    Timer energy_measure_timer("energy_measure");
    StochGradUpdateTPS_(*search_dir, optimize_para.step_lens[point]);
    std::cout << "good 2" << std::endl;
    ClearEnergyAndHoleSamples_();
    bond_flip_accept_num = 0;
    cluster_update_accept_num = 0;
    for (size_t sweep = 0; sweep < optimize_para.mc_samples; sweep++) {
      std::vector<size_t> accept_nums = MCSweep_();
      bond_flip_accept_num += accept_nums[0];
      if (accept_nums.size() > 1) {
        cluster_update_accept_num += accept_nums[1];
      }
      SampleEnergy_();
    }
    TenElemT en_self = Mean(energy_samples_); //energy value in each processor
    auto [energy, en_err] = StatisticFromProcessors(en_self, MPI_Comm(world_));
    gqten::hp_numeric::MPI_Bcast(&energy, 1, kMasterProc, MPI_Comm(world_));
    if (world_.rank() == 0) {
      energy_trajectory_.push_back(energy);
      energy_error_traj_.push_back(en_err);

      if (energy < e0_min) {
        e0_min = energy;
        tps_min = split_index_tps_;
      }

      double bond_accept_rate = double(bond_flip_accept_num) / double(flip_bond_num * optimize_para.mc_samples);
      double cluster_accept_rate = double(cluster_update_accept_num) / double(cluster_num * optimize_para.mc_samples);

      //cout
      double energy_measure_time = energy_measure_timer.Elapsed();
      double gradient_calculation_time = grad_calculation_timer.Elapsed();
      std::cout << "Stride :" << std::setw(9) << optimize_para.step_lens[point]
                << "E0 = " << std::setw(14) << std::fixed << std::setprecision(kEnergyOutputPrecision)
                << energy
                << pm_sign << " " << std::setw(10) << std::scientific << std::setprecision(2)
                << en_err
                << "Accept rate = " << std::setw(5) << std::fixed << std::setprecision(2) << bond_accept_rate;
      if (optimize_para.mc_sweep_scheme == CompressedLatticeKagomeLocalUpdate) {
        std::cout << std::setw(5) << std::fixed << std::setprecision(2) << cluster_accept_rate;
      }
      std::cout << " TotT = " << std::setw(8) << std::fixed << std::setprecision(2) << energy_measure_time << "s"
                << std::endl;
    }
  }
  if (world_.rank() == kMasterProc) {
    split_index_tps_ = tps_min;
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::IterativeOptimizeTPS_(void) {
  for (size_t iter = 0; iter < optimize_para.step_lens.size(); iter++) {
    IterativeOptimizeTPSStep_(iter);
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::IterativeOptimizeTPSStep_(const size_t iter) {
  size_t flip_bond_num = lx_ * (ly_ - 1) + ly_ * (lx_ - 1);
  size_t cluster_num = 3 * lx_ * ly_;
  size_t bond_flip_accept_num = 0;
  size_t cluster_update_accept_num = 0;
  ClearEnergyAndHoleSamples_();

  Timer grad_update_timer("gradient_update");
  for (size_t sweep = 0; sweep < optimize_para.mc_samples; sweep++) {
    std::vector<size_t> accept_nums = MCSweep_();
    bond_flip_accept_num += accept_nums[0];
    if (accept_nums.size() > 1) {
      cluster_update_accept_num += accept_nums[1];
    }
    SampleEnergyAndHols_();
  }
  double bond_accept_rate = double(bond_flip_accept_num) / double(flip_bond_num * optimize_para.mc_samples);
  double cluster_accept_rate = double(cluster_update_accept_num) / double(cluster_num * optimize_para.mc_samples);
  GatherStatisticEnergyAndGrad_();

  Timer tps_update_timer("tps_update");
  size_t sr_iter;
  double sr_natural_grad_norm;
  SITPST init_guess;
  if (iter == 0) {
    init_guess = SITPST(ly_, lx_, split_index_tps_.PhysicalDim()); //set 0 as initial guess
  } else {
    init_guess = natural_grad_;
  }

  double step_len = optimize_para.step_lens[iter];
  switch (optimize_para.update_scheme) {
    case StochasticGradient:StochGradUpdateTPS_(grad_, step_len);
      break;
    case RandomStepStochasticGradient:step_len *= u_double_(random_engine);
      StochGradUpdateTPS_(grad_, step_len);
      break;
    case StochasticReconfiguration: {
      auto iter_natural_grad_norm = StochReconfigUpdateTPS_(grad_, step_len, init_guess);
      sr_iter = iter_natural_grad_norm.first;
      sr_natural_grad_norm = iter_natural_grad_norm.second;
      break;
    }
    case RandomStepStochasticReconfiguration: {
      step_len *= u_double_(random_engine);
      auto iter_natural_grad_norm = StochReconfigUpdateTPS_(grad_, step_len, init_guess);
      sr_iter = iter_natural_grad_norm.first;
      sr_natural_grad_norm = iter_natural_grad_norm.second;
      break;
    }
    case NormalizedStochasticReconfiguration: {
      auto iter_natural_grad_norm = NormalizedStochReconfigUpdateTPS_(grad_, step_len, init_guess);
      sr_iter = iter_natural_grad_norm.first;
      sr_natural_grad_norm = iter_natural_grad_norm.second;
      break;
    }
    case RandomGradientElement: {
      GradientRandElementSign_();
      StochGradUpdateTPS_(grad_, step_len);
      break;
    }
    case BoundGradientElement:BoundGradElementUpdateTPS_(grad_, step_len);
      break;
    default:std::cout << "update method does not support!" << std::endl;
      exit(2);
  }

  double tps_update_time = tps_update_timer.Elapsed();

  if (world_.rank() == kMasterProc) {
    double gradient_update_time = grad_update_timer.Elapsed();
    std::cout << "Iter " << std::setw(4) << iter
              << "Alpha = " << std::setw(9) << std::scientific << std::setprecision(1) << step_len
              << "E0 = " << std::setw(14) << std::fixed << std::setprecision(kEnergyOutputPrecision)
              << energy_trajectory_.back()
              << pm_sign << " " << std::setw(10) << std::scientific << std::setprecision(2)
              << energy_error_traj_.back()
              << "Grad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << grad_norm_.back()
              << "Accept rate = " << std::setw(5) << std::fixed << std::setprecision(2) << bond_accept_rate;
    if (optimize_para.mc_sweep_scheme == CompressedLatticeKagomeLocalUpdate) {
      std::cout << std::setw(5) << std::fixed << std::setprecision(2) << cluster_accept_rate;
    }
    if (optimize_para.update_scheme == StochasticReconfiguration ||
        optimize_para.update_scheme == RandomStepStochasticReconfiguration ||
        optimize_para.update_scheme == NormalizedStochasticReconfiguration) {
      std::cout << "SRSolver Iter = " << std::setw(4) << sr_iter;
      std::cout << "NGrad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << sr_natural_grad_norm;
    }
    std::cout << "TPS UpdateT = " << std::setw(6) << std::fixed << std::setprecision(2) << tps_update_time << "s"
              << " TotT = " << std::setw(8) << std::fixed << std::setprecision(2) << gradient_update_time << "s"
              << "\n";
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
  if (optimize_para.update_scheme == StochasticReconfiguration ||
      optimize_para.update_scheme == RandomStepStochasticReconfiguration ||
      optimize_para.update_scheme == NormalizedStochasticReconfiguration) {
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
      if (optimize_para.update_scheme == StochasticReconfiguration ||
          optimize_para.update_scheme == RandomStepStochasticReconfiguration ||
          optimize_para.update_scheme == NormalizedStochasticReconfiguration) {
        gten_sample({row, col})[basis] = gten;
      }
    }
  }
  if (optimize_para.update_scheme == StochasticReconfiguration ||
      optimize_para.update_scheme == RandomStepStochasticReconfiguration ||
      optimize_para.update_scheme == NormalizedStochasticReconfiguration) {
    gten_samples_.emplace_back(gten_sample);
  }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::SampleEnergy_(void) {
  TenElemT energy_loc = energy_solver_.CalEnergy(&split_index_tps_, &tps_sample_);
  energy_samples_.push_back(energy_loc);
}

template<typename TenElemT, typename QNT, typename EnergySolver>
SplitIndexTPS<TenElemT, QNT>
VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::GatherStatisticEnergyAndGrad_(void) {
  TenElemT en_self = Mean(energy_samples_); //energy value in each processor
  auto [energy, en_err] = StatisticFromProcessors(en_self, MPI_Comm(world_));
  gqten::hp_numeric::MPI_Bcast(&energy, 1, kMasterProc, MPI_Comm(world_));
  if (world_.rank() == 0) {
    energy_trajectory_.push_back(energy);
    energy_error_traj_.push_back(en_err);
  }

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
        if (optimize_para.update_scheme == StochasticReconfiguration ||
            optimize_para.update_scheme == RandomStepStochasticReconfiguration ||
            optimize_para.update_scheme == NormalizedStochasticReconfiguration) {
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
 * @note Normalization condition: tensors in each site are normalized.
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
  tps_sample_ = TPSSample(split_index_tps_, tps_sample_.config);
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::BoundGradElementUpdateTPS_(VMCPEPSExecutor::SITPST &grad,
                                                                              double step_len) {
  if (world_.rank() == kMasterProc) {
    for (size_t row = 0; row < ly_; row++)
      for (size_t col = 0; col < lx_; col++) {
        const size_t phy_dim = grad_({row, col}).size();
        double norm = 0;
        for (size_t compt = 0; compt < phy_dim; compt++) {
          Tensor &grad_ten = grad({row, col})[compt];
          grad_ten.ElementWiseBoundTo(step_len);
          split_index_tps_({row, col})[compt] += (-step_len) * grad_ten;
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
std::pair<size_t, double> VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::StochReconfigUpdateTPS_(
    const VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::SITPST &grad, double step_len, const SITPST &init_guess) {
  size_t cgsolver_iter = CalcNaturalGradient_(grad, init_guess);
  double natural_grad_norm = natural_grad_.Norm();
  StochGradUpdateTPS_(natural_grad_, step_len);
  return std::make_pair(cgsolver_iter, natural_grad_norm);
}

template<typename TenElemT, typename QNT, typename EnergySolver>
std::pair<size_t, double> VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::NormalizedStochReconfigUpdateTPS_(
    const VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::SITPST &grad, double step_len, const SITPST &init_guess) {
  size_t cgsolver_iter = CalcNaturalGradient_(grad, init_guess);
  double natural_grad_norm = natural_grad_.Norm();
  step_len = step_len / std::sqrt(natural_grad_norm);
  StochGradUpdateTPS_(natural_grad_, step_len);
  return std::make_pair(cgsolver_iter, natural_grad_norm);
}

template<typename TenElemT, typename QNT, typename EnergySolver>
size_t VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::CalcNaturalGradient_(
    const VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::SITPST &grad, const SITPST &init_guess) {
  SITPST *pgten_ave_(nullptr);
  if (world_.rank() == kMasterProc) {
    pgten_ave_ = &gten_ave_;
  }
  SRSMatrix s_matrix(&gten_samples_, pgten_ave_, world_.size());
  s_matrix.diag_shift = cg_params.diag_shift;
  size_t cgsolver_iter;
  natural_grad_ = ConjugateGradientSolver(s_matrix, grad, init_guess,
                                          cg_params.max_iter, cg_params.tolerance,
                                          cg_params.residue_restart_step, cgsolver_iter, world_);
  return cgsolver_iter;
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::GradientRandElementSign_() {
  if (world_.rank() == kMasterProc)
    for (size_t row = 0; row < ly_; row++) {
      for (size_t col = 0; col < lx_; col++) {
        size_t dim = split_index_tps_({row, col}).size();
        for (size_t i = 0; i < dim; i++)
          grad_({row, col})[i].ElementWiseRandSign(u_double_, random_engine);
      }
    }
}

template<typename TenElemT, typename QNT, typename EnergySolver>
std::vector<size_t> VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::MCSweep_(void) {
  size_t bond_flip_times, cluster_flip_times;
  if (optimize_para.mc_sweep_scheme == SequentiallyNNSiteFlip) {
    for (size_t i = 0; i < optimize_para.mc_sweeps_between_sample; i++) {
      bond_flip_times = tps_sample_.MCSequentiallyNNFlipSweep(split_index_tps_, u_double_);;
    }
    return {bond_flip_times};
  } else if (optimize_para.mc_sweep_scheme == CompressedLatticeKagomeLocalUpdate) {
    for (size_t i = 0; i < optimize_para.mc_sweeps_between_sample; i++) {
      tps_sample_.MCCompressedKagomeLatticeSequentiallyLocalUpdateSweepSmoothBoundary(split_index_tps_, u_double_,
                                                                                      cluster_flip_times,
                                                                                      bond_flip_times);
//      tps_sample_.MCCompressedKagomeLatticeSequentiallyLocalUpdateSweep(split_index_tps_, u_double_, cluster_flip_times,
//                                                                        bond_flip_times);
    }
    return {bond_flip_times, cluster_flip_times};
  } else {
    std::cout << "Do not support MC sweep Scheme" << std::endl;
    exit(1);
  }

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
    tps_sample_ = TPSSample<TenElemT, QNT>(split_index_tps_, config);
  } else {
    std::cout << "Loading configuration in rank " << world_.rank()
              << " fails. Use preset configuration and random warm up."
              << std::endl;
    tps_sample_ = TPSSample<TenElemT, QNT>(split_index_tps_, optimize_para.init_config);
    WarmUp_();
  }
  warm_up_ = true;
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::DumpData(const bool release_mem) {
  DumpData(optimize_para.wavefunction_path, release_mem);
}

template<typename TenElemT, typename QNT, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, EnergySolver>::DumpData(const std::string &tps_path, const bool release_mem) {
  std::string energy_data_path = "./energy";
  if (world_.rank() == kMasterProc) {
    split_index_tps_.Dump(tps_path, release_mem);
    if (!gqmps2::IsPathExist(energy_data_path)) {
      gqmps2::CreatPath(energy_data_path);
    }
  }
  world_.barrier(); // configurations dump will collapse when creating path if there is no barrier.
  tps_sample_.config.Dump(tps_path, world_.rank());
  DumpVecData(energy_data_path + "/energy_sample" + std::to_string(world_.rank()), energy_samples_);
  if (world_.rank() == kMasterProc) {
    DumpVecData(energy_data_path + "/energy_trajectory", energy_trajectory_);
    DumpVecData(energy_data_path + "/energy_err_trajectory", energy_error_traj_);
  }
//  DumpVecData(tps_path + "/sum_configs" + std::to_string(world_.rank()), sum_configs_);
}

}//gqpeps

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H

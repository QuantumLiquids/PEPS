// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: QuantumLiquids/PEPS project. Implementation for the variational Monte-Carlo PEPS
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H

#include <iomanip>
#include "qlpeps/algorithm/vmc_update/stochastic_reconfiguration_smatrix.h" //SRSMatrix
#include "qlpeps/utility/conjugate_gradient_solver.h"
#include "qlpeps/utility/helpers.h"                                         //ComplexConjugate
#include "qlpeps/algorithm/vmc_update/axis_update.h"
#include "qlpeps/monte_carlo_tools/statistics.h"

namespace qlpeps {
using namespace qlten;

//helper
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> Mean(const std::vector<QLTensor<TenElemT, QNT> *> &tensor_list,
                             const size_t length) {
  std::vector<TenElemT> coefs(tensor_list.size(), TenElemT(1.0));
  QLTensor<TenElemT, QNT> sum;
  LinearCombine(coefs, tensor_list, TenElemT(0.0), &sum);
  return sum * (1.0 / double(length));
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> MPIMeanTensor(const QLTensor<TenElemT, QNT> &tensor,
                                      const MPI_Comm &comm) {
  using Tensor = QLTensor<TenElemT, QNT>;
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  if (rank == kMPIMasterRank) {
    std::vector<Tensor *> ten_list(mpi_size, nullptr);
    for (size_t proc = 0; proc < mpi_size; proc++) {
      if (proc != kMPIMasterRank) {
        ten_list[proc] = new Tensor();
        ten_list[proc]->MPI_Recv(proc, 2 * proc, comm);
      } else {
        ten_list[proc] = new Tensor(tensor);
      }
    }
    Tensor res = Mean(ten_list, mpi_size);
    for (auto pten : ten_list) {
      delete pten;
    }
    return res;
  } else {
    tensor.MPI_Send(kMPIMasterRank, 2 * rank, comm);
    return Tensor();
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
VMCPEPSExecutor<TenElemT,
                QNT,
                MonteCarloSweepUpdater,
                EnergySolver>::VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                                               const TPST &tps_init,
                                               const MPI_Comm &comm,
                                               const EnergySolver &solver) :
    VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>(optimize_para,
                                                                         SITPST(tps_init),
                                                                         comm,
                                                                         solver) {}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
VMCPEPSExecutor<TenElemT,
                QNT,
                MonteCarloSweepUpdater,
                EnergySolver>::VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                                               const SITPST &sitpst_init,
                                               const MPI_Comm &comm,
                                               const EnergySolver &solver) :
    MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>(sitpst_init,
                                                                      MonteCarloParams(optimize_para),
                                                                      PEPSParams(optimize_para),
                                                                      comm),
    optimize_para(optimize_para),
    energy_solver_(solver),
//    gten_samples_(ly_, lx_),
//    g_times_energy_samples_(ly_, lx_),
    gten_sum_(ly_, lx_),
    g_times_energy_sum_(ly_, lx_),
    grad_(ly_, lx_), natural_grad_(ly_, lx_),
    en_min_(std::numeric_limits<double>::max()),
    tps_lowest_(split_index_tps_) {
  if (std::find(stochastic_reconfiguration_method.cbegin(),
                stochastic_reconfiguration_method.cend(),
                optimize_para.update_scheme) != stochastic_reconfiguration_method.cend()) {
    stochastic_reconfiguration_update_class_ = true;
  } else {
    stochastic_reconfiguration_update_class_ = false;
  }
  ReserveSamplesDataSpace_();
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
VMCPEPSExecutor<TenElemT,
                QNT,
                MonteCarloSweepUpdater,
                EnergySolver>::VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                                               const size_t ly, const size_t lx,
                                               const MPI_Comm &comm,
                                               const EnergySolver &solver):
    MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>(ly, lx,
                                                                      MonteCarloParams(optimize_para),
                                                                      PEPSParams(optimize_para),
                                                                      comm),
    optimize_para(optimize_para),
    energy_solver_(solver),
//    gten_samples_(ly_, lx_),
//    g_times_energy_samples_(ly_, lx_),
    gten_sum_(ly_, lx_), g_times_energy_sum_(ly_, lx_),
    grad_(ly_, lx_), natural_grad_(ly_, lx_),
    en_min_(std::numeric_limits<double>::max()),
    tps_lowest_(split_index_tps_) {

  if (std::find(stochastic_reconfiguration_method.cbegin(),
                stochastic_reconfiguration_method.cend(),
                optimize_para.update_scheme) != stochastic_reconfiguration_method.cend()) {
    stochastic_reconfiguration_update_class_ = true;
  } else {
    stochastic_reconfiguration_update_class_ = false;
  }
  ReserveSamplesDataSpace_();
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::Execute(void) {
  this->SetStatus(ExecutorStatus::EXEING);
  this->WarmUp_();
  if (optimize_para.update_scheme == GradientLineSearch || optimize_para.update_scheme == NaturalGradientLineSearch) {
    LineSearchOptimizeTPS_();
  } else {
    IterativeOptimizeTPS_();
  }
  DumpData();
  this->SetStatus(ExecutorStatus::FINISH);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::ReserveSamplesDataSpace_(void) {
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
      gten_sum_({row, col}) = std::vector<Tensor>(dim);
      for (size_t compt = 0; compt < dim; compt++) {
        gten_sum_({row, col})[compt] = Tensor(split_index_tps_({row, col})[compt].GetIndexes());
      }

      g_times_energy_sum_({row, col}) = gten_sum_({row, col});
    }
  }
  for (size_t row = 0; row < ly_; row++)
    for (size_t col = 0; col < lx_; col++) {
      size_t dim = split_index_tps_({row, col}).size();
      grad_({row, col}) = std::vector<Tensor>(dim);
    }
  if (rank_ == 0) {
    energy_trajectory_.reserve(optimize_para.step_lens.size());
    energy_error_traj_.reserve(optimize_para.step_lens.size());
  }

  if (rank_ == kMPIMasterRank)
    grad_norm_.reserve(optimize_para.step_lens.size());

  if (stochastic_reconfiguration_update_class_) {
    gten_samples_.reserve(optimize_para.mc_samples);
    for (size_t row = 0; row < ly_; row++)
      for (size_t col = 0; col < lx_; col++) {
        size_t dim = split_index_tps_({row, col}).size();
        natural_grad_({row, col}) = std::vector<Tensor>(dim);
      }
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::PrintExecutorInfo_(void) {
  this->PrintCommonInfo_("VARIATIONAL MONTE-CARLO PROGRAM FOR PEPS");
  if (rank_ == kMPIMasterRank) {
    size_t indent = 40;
    std::cout << std::setw(indent) << "PEPS update times:" << optimize_para.step_lens.size() << "\n";
    std::cout << std::setw(indent) << "PEPS update strategy:"
              << WavefunctionUpdateSchemeString(optimize_para.update_scheme) << "\n";
    if (stochastic_reconfiguration_update_class_) {
      if (!optimize_para.cg_params.has_value()) {
        std::cout << "Conjugate gradient parameters have not been set!" << std::endl;
        exit(1);
      }
      std::cout << std::setw(indent) << "Conjugate gradient diagonal shift:"
                << optimize_para.cg_params.value().diag_shift
                << "\n";
    }
  }
  this->PrintTechInfo_();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::LineSearchOptimizeTPS_(void) {
  std::vector<double> accept_rates_accum;
  ClearEnergyAndHoleSamples_();

  Timer grad_calculation_timer("gradient_calculation");
  for (size_t sweep = 0; sweep < optimize_para.mc_samples; sweep++) {
    std::vector<double> accept_rates = this->MCSweep_();
    if (sweep == 0) {
      accept_rates_accum = accept_rates;
    } else {
      for (size_t i = 0; i < accept_rates_accum.size(); i++) {
        accept_rates_accum[i] += accept_rates[i];
      }
    }
    SampleEnergyAndHols_();
  }
  auto accept_rates_avg = accept_rates_accum;
  for (double &rates : accept_rates_avg) {
    rates /= double(optimize_para.mc_samples);
  }
  GatherStatisticEnergyAndGrad_();

  size_t cgsolver_iter(0);
  double sr_natural_grad_norm(0.0);
  SITPST *search_dir(nullptr);
  switch (optimize_para.update_scheme) {
    case GradientLineSearch: {
      if (rank_ == kMPIMasterRank)
        search_dir = &grad_;
      break;
    }
    case NaturalGradientLineSearch: {
      auto init_guess = SITPST(ly_, lx_, split_index_tps_.PhysicalDim());
      cgsolver_iter = CalcNaturalGradient_(grad_, init_guess);
      if (rank_ == kMPIMasterRank) {
        search_dir = &natural_grad_;
        sr_natural_grad_norm = natural_grad_.NormSquare();
      }
      break;
    }
    default: {
      std::cerr << "update scheme is not line search scheme." << std::endl;
      exit(1);
    }
  }

  if (rank_ == kMPIMasterRank) {
    double gradient_calculation_time = grad_calculation_timer.Elapsed();
    std::cout << "Initial Search Direction Calculation :\n"
              << "E0 = " << std::setw(14 * sizeof(TenElemT) / sizeof(double)) << std::fixed
              << std::setprecision(kEnergyOutputPrecision)
              << energy_trajectory_.back()
              << pm_sign << " " << std::setw(10) << std::scientific << std::setprecision(2)
              << energy_error_traj_.back()
              << "Grad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << grad_norm_.back()
              << "Accept rate = [";
    for (double &rate : accept_rates_avg) {
      std::cout << std::setw(5) << std::fixed << std::setprecision(2) << rate;
    }
    std::cout << "]";
    if (stochastic_reconfiguration_update_class_) {
      std::cout << "SRSolver Iter = " << std::setw(4) << cgsolver_iter;
      std::cout << "NGrad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << sr_natural_grad_norm;
    }
    std::cout << " TotT = " << std::setw(8) << std::fixed << std::setprecision(2) << gradient_calculation_time << "s"
              << "\n";
  }
  AcceptanceRateCheck(accept_rates_avg);
  LineSearch_(*search_dir, optimize_para.step_lens);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::LineSearch_(const SplitIndexTPS<TenElemT,
                                                                                                           QNT> &search_dir,
                                                                                       const std::vector<double> &strides) {

  if (rank_ == kMPIMasterRank) {
    en_min_ = Real(energy_trajectory_[0]);
  }
  tps_lowest_ = split_index_tps_;
  double stride = 0.0;
  for (size_t point = 0; point < strides.size(); point++) {
    Timer energy_measure_timer("energy_measure");
    UpdateTPSByVecAndSynchronize_(search_dir, strides[point]);
    ClearEnergyAndHoleSamples_();
    std::vector<double> accept_rates_accum;
    for (size_t sweep = 0; sweep < optimize_para.mc_samples; sweep++) {
      std::vector<double> accept_rates = this->MCSweep_();
      if (sweep == 0) {
        accept_rates_accum = accept_rates;
      } else {
        for (size_t i = 0; i < accept_rates_accum.size(); i++) {
          accept_rates_accum[i] += accept_rates[i];
        }
      }
      SampleEnergy_();
    }
    TenElemT en_self = Mean(energy_samples_); //energy value in each processor
    auto [energy, en_err] = GatherStatisticSingleData(en_self, MPI_Comm(comm_));
    qlten::hp_numeric::MPI_Bcast(&energy, 1, kMPIMasterRank, MPI_Comm(comm_));
    if (rank_ == kMPIMasterRank) {
      energy_trajectory_.push_back(energy);
      energy_error_traj_.push_back(en_err);

      auto accept_rates_avg = accept_rates_accum;
      for (double &rates : accept_rates_avg) {
        rates /= double(optimize_para.mc_samples);
      }
      if (Real(energy) < en_min_) {
        en_min_ = Real(energy);
        tps_lowest_ = split_index_tps_;
      }

      //cout
      double energy_measure_time = energy_measure_timer.Elapsed();
      stride += optimize_para.step_lens[point];
      std::cout << "Stride :" << std::setw(9) << stride
                << "E0 = " << std::setw(14) << std::fixed << std::setprecision(kEnergyOutputPrecision)
                << energy
                << pm_sign << " " << std::setw(10) << std::scientific << std::setprecision(2)
                << en_err
                << "Accept rate = [";
      for (double &rate : accept_rates_avg) {
        std::cout << std::setw(5) << std::fixed << std::setprecision(2) << rate;
      }
      std::cout << "]";

      std::cout << " TotT = " << std::setw(8) << std::fixed << std::setprecision(2) << energy_measure_time << "s"
                << std::endl;
    }
  }
  if (rank_ == kMPIMasterRank) {
    split_index_tps_ = tps_lowest_;
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::IterativeOptimizeTPS_(void) {
  for (size_t iter = 0; iter < optimize_para.step_lens.size(); iter++) {
    IterativeOptimizeTPSStep_(iter);
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT,
                     MonteCarloSweepUpdater,
                     EnergySolver>::IterativeOptimizeTPSStep_(const size_t iter) {
  std::vector<double> accept_rates_accum;
  ClearEnergyAndHoleSamples_();

  Timer grad_update_timer("gradient_update");
  for (size_t sweep = 0; sweep < optimize_para.mc_samples; sweep++) {
    std::vector<double> accept_rates = this->MCSweep_();
    if (sweep == 0) {
      accept_rates_accum = accept_rates;
    } else {
      for (size_t i = 0; i < accept_rates_accum.size(); i++) {
        accept_rates_accum[i] += accept_rates[i];
      }
    }
    SampleEnergyAndHols_();
  }
  std::vector<double> accept_rates_avg = accept_rates_accum;
  for (double &rates : accept_rates_avg) {
    rates /= double(optimize_para.mc_samples);
  }
  TenElemT en_step;
  std::tie(en_step, std::ignore) = GatherStatisticEnergyAndGrad_();

  if (rank_ == kMPIMasterRank && en_min_ > Real(en_step)) {
    en_min_ = Real(en_step);
    tps_lowest_ = split_index_tps_;
  }

  Timer tps_update_timer("tps_update");
  size_t sr_iter;
  double sr_natural_grad_norm;
  SITPST sr_init_guess;
  if (iter == 0) {
    sr_init_guess = SITPST(ly_, lx_, split_index_tps_.PhysicalDim()); //set 0 as initial guess
  } else {
    sr_init_guess = natural_grad_;
  }

  double step_len = optimize_para.step_lens[iter];
  switch (optimize_para.update_scheme) {
    case StochasticGradient:UpdateTPSByVecAndSynchronize_(grad_, step_len);
      break;
    case RandomStepStochasticGradient:step_len *= unit_even_distribution(random_engine_);
      UpdateTPSByVecAndSynchronize_(grad_, step_len);
      break;
    case StochasticReconfiguration: {
      auto iter_natural_grad_norm = StochReconfigUpdateTPS_(grad_, step_len, sr_init_guess, false);
      sr_iter = iter_natural_grad_norm.first;
      sr_natural_grad_norm = iter_natural_grad_norm.second;
      break;
    }
    case RandomStepStochasticReconfiguration: {
      step_len *= unit_even_distribution(random_engine_);
      auto iter_natural_grad_norm = StochReconfigUpdateTPS_(grad_, step_len, sr_init_guess, false);
      sr_iter = iter_natural_grad_norm.first;
      sr_natural_grad_norm = iter_natural_grad_norm.second;
      break;
    }
    case NormalizedStochasticReconfiguration: {
      auto iter_natural_grad_norm = StochReconfigUpdateTPS_(grad_, step_len, sr_init_guess, true);
      sr_iter = iter_natural_grad_norm.first;
      sr_natural_grad_norm = iter_natural_grad_norm.second;
      break;
    }
    case RandomGradientElement: {
      GradientRandElementSign_();
      UpdateTPSByVecAndSynchronize_(grad_, step_len);
      break;
    }
    case BoundGradientElement:BoundGradElementUpdateTPS_(grad_, step_len);
      break;
    default:std::cout << "update method does not support!" << std::endl;
      exit(2);
  }

  double tps_update_time = tps_update_timer.Elapsed();

  if (rank_ == kMPIMasterRank) {
    double gradient_update_time = grad_update_timer.Elapsed();
    std::cout << "Iter " << std::setw(4) << iter
              << "Alpha = " << std::setw(9) << std::scientific << std::setprecision(1) << step_len
              << "E0 = " << std::setw(14) << std::fixed << std::setprecision(kEnergyOutputPrecision)
              << en_step
              << pm_sign << " " << std::setw(10) << std::scientific << std::setprecision(2)
              << energy_error_traj_.back()
              << "Grad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << grad_norm_.back()
              << "Accept rate = [";
    for (double &rate : accept_rates_avg) {
      std::cout << std::setw(5) << std::fixed << std::setprecision(2) << rate;
    }
    std::cout << "]";

    if (stochastic_reconfiguration_update_class_) {
      std::cout << "SRSolver Iter = " << std::setw(4) << sr_iter;
      std::cout << "NGrad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << sr_natural_grad_norm;
    }
    std::cout << "TPS UpdateT = " << std::setw(6) << std::fixed << std::setprecision(2) << tps_update_time << "s"
              << " TotT = " << std::setw(8) << std::fixed << std::setprecision(2) << gradient_update_time << "s"
              << "\n";
  }
  AcceptanceRateCheck(accept_rates_avg);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::ClearEnergyAndHoleSamples_(void) {
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
      size_t dim = split_index_tps_.PhysicalDim({row, col});
      gten_sum_({row, col}) = std::vector<Tensor>(dim);
      for (size_t compt = 0; compt < dim; compt++) {
        gten_sum_({row, col})[compt] = Tensor(split_index_tps_({row, col})[compt].GetIndexes());
      }
      g_times_energy_sum_({row, col}) = gten_sum_({row, col});
    }
  }
  if (stochastic_reconfiguration_update_class_) {
    gten_samples_.clear();
  }
}

/**
 *
 * @param hole_ten      get by CalEnergyAndHoles function, which has be complex conjugated
 * @param split_index_tps_ten   tensor in split index tps, which has not be complex conjugated
 * @return
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> CalGTenForFermionicTensors(
    const QLTensor<TenElemT, QNT> &hole_ten_dag,
    const QLTensor<TenElemT, QNT> &split_index_tps_ten
) {
  auto hole_ten = Dag(hole_ten_dag);
  QLTensor<TenElemT, QNT> psi_ten, hole_dag_psi;
  Contract(&hole_ten, {1, 2, 3, 4}, &split_index_tps_ten, {0, 1, 2, 3}, &psi_ten);
  Contract(&hole_ten_dag, {0}, &psi_ten, {0}, &hole_dag_psi);
  return hole_dag_psi * (1.0 / std::norm(psi_ten.GetElem({0, 0})));
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SampleEnergyAndHols_(void) {
  TensorNetwork2D<TenElemT, QNT> holes(ly_, lx_);
  TenElemT energy_loc = energy_solver_.template CalEnergyAndHoles<TenElemT, QNT, true>(&split_index_tps_,
                                                                                &tps_sample_,
                                                                                holes);
  TenElemT energy_loc_conj = ComplexConjugate(energy_loc);
  TenElemT inv_psi = ComplexConjugate(1.0 / tps_sample_.amplitude); //to divide the holes.
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
      Tensor gten;
      if constexpr (Tensor::IsFermionic()) {
        gten = CalGTenForFermionicTensors(holes({row, col}), tps_sample_.tn({row, col}));
        // tps_sample_.tn({row, col})  is  split_index_tps_({row, col})[basis];
      } else {
        gten = inv_psi * holes({row, col});  //holes should be dag in CalEnergyAndHoles function
      }
      gten_sum_({row, col})[basis] += gten;
      g_times_energy_sum_({row, col})[basis] += energy_loc_conj * gten;
      //? when samples become large, does the summation reliable as the small number are added to large number.
      if (stochastic_reconfiguration_update_class_) {
        gten_sample({row, col})[basis] = gten;
      }
    }
  }
  if (stochastic_reconfiguration_update_class_) {
    gten_samples_.emplace_back(gten_sample);
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
TenElemT VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SampleEnergy_(void) {
  TenElemT energy_loc = energy_solver_.CalEnergy(&split_index_tps_,
                                                                                 &tps_sample_
                                                              );
  energy_samples_.push_back(energy_loc);
  return energy_loc;
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
std::pair<TenElemT, SplitIndexTPS<TenElemT, QNT>>
VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::GatherStatisticEnergyAndGrad_(void) {
  TenElemT en_self = Mean(energy_samples_); //energy value in each processor
  auto [energy, en_err] = GatherStatisticSingleData(en_self, MPI_Comm(comm_));
  qlten::hp_numeric::MPI_Bcast(&energy, 1, kMPIMasterRank, MPI_Comm(comm_));
  if (rank_ == kMPIMasterRank) {
    energy_trajectory_.push_back(energy);
    energy_error_traj_.push_back(en_err);
  }

  //calculate grad in each processor
  const size_t sample_num = optimize_para.mc_samples;
  gten_ave_ = gten_sum_ * (1.0 / sample_num);
  grad_ = g_times_energy_sum_ * (1.0 / sample_num) + ComplexConjugate(-energy) * gten_ave_;

  for (size_t row = 0; row < ly_; row++) {
    for (size_t col = 0; col < lx_; col++) {
      const size_t phy_dim = grad_({row, col}).size();
      for (size_t compt = 0; compt < phy_dim; compt++) {
        // gather and estimate grad in master (and maybe the error bar of grad)
        grad_({row, col})[compt] = MPIMeanTensor(grad_({row, col})[compt], comm_);
        // note here the grad data except in master are clear
        if (stochastic_reconfiguration_update_class_) {
          gten_ave_({row, col})[compt] = MPIMeanTensor(gten_ave_({row, col})[compt], comm_);
        }
      }
    }
  }
  grad_.ActFermionPOps();
  if (rank_ == kMPIMasterRank) {
    grad_norm_.push_back(grad_.NormSquare());
  }
  //do not broadcast because only broadcast the updated TPS
  return std::make_pair(energy, grad_);
}

/**
 * Stochastic gradient descent update peps
 *
 * @param grad
 * @param step_len
 * @note Normalization condition: tensors in each site are normalized.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT,
                     QNT,
                     MonteCarloSweepUpdater,
                     EnergySolver>::UpdateTPSByVecAndSynchronize_(const VMCPEPSExecutor::SITPST &grad,
                                                                  double step_len) {
  if (rank_ == kMPIMasterRank) {
    split_index_tps_ += (-step_len) * grad;
  }
  BroadCast(split_index_tps_, comm_);
  Configuration config = tps_sample_.config;
  tps_sample_ = WaveFunctionComponentT(split_index_tps_, config, TPSWaveFunctionComponent<TenElemT, QNT>::trun_para);
  this->NormTPSForOrder1Amplitude_();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT,
                     MonteCarloSweepUpdater,
                     EnergySolver>::BoundGradElementUpdateTPS_(VMCPEPSExecutor::SITPST &grad,
                                                               double step_len) {
  if (rank_ == kMPIMasterRank) {
    for (size_t row = 0; row < ly_; row++)
      for (size_t col = 0; col < lx_; col++) {
        const size_t phy_dim = grad_({row, col}).size();
        for (size_t compt = 0; compt < phy_dim; compt++) {
          Tensor &grad_ten = grad({row, col})[compt];
          grad_ten.ElementWiseBoundTo(step_len);
          split_index_tps_({row, col})[compt] += (-step_len) * grad_ten;
        }
      }
  }
  BroadCast(split_index_tps_, comm_);
  Configuration config = tps_sample_.config;
  tps_sample_ = WaveFunctionComponentT(split_index_tps_, config, TPSWaveFunctionComponent<TenElemT, QNT>::trun_para);
  this->NormTPSForOrder1Amplitude_();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
std::pair<size_t, double> VMCPEPSExecutor<TenElemT,
                                          QNT,
                                          MonteCarloSweepUpdater,
                                          EnergySolver>::StochReconfigUpdateTPS_(
    const VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SITPST &grad,
    double step_len,
    const SITPST &init_guess,
    const bool normalize_natural_grad) {
  size_t cgsolver_iter = CalcNaturalGradient_(grad, init_guess);
  double natural_grad_norm = natural_grad_.NormSquare();
  if (normalize_natural_grad) step_len /= std::sqrt(natural_grad_norm);
  UpdateTPSByVecAndSynchronize_(natural_grad_, step_len);
  return std::make_pair(cgsolver_iter, natural_grad_norm);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
size_t VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::CalcNaturalGradient_(
    const VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SITPST &grad,
    const SITPST &init_guess) {
  SITPST *pgten_ave_(nullptr);
  if (rank_ == kMPIMasterRank) {
    pgten_ave_ = &gten_ave_;
  }
  const ConjugateGradientParams &cg_params = optimize_para.cg_params.value();
  SRSMatrix s_matrix(&gten_samples_, pgten_ave_, mpi_size_);
  s_matrix.diag_shift = cg_params.diag_shift;
  size_t cgsolver_iter;
  if constexpr (QLTensor<TenElemT, QNT>::IsFermionic()) {
    auto signed_grad = grad;
    signed_grad.ActFermionPOps();   // Act back
    natural_grad_ = ConjugateGradientSolver(s_matrix, signed_grad, init_guess,
                                            cg_params.max_iter, cg_params.tolerance,
                                            cg_params.residue_restart_step, cgsolver_iter, comm_);
    natural_grad_.ActFermionPOps(); // question: why works?
  } else {
    natural_grad_ = ConjugateGradientSolver(s_matrix, grad, init_guess,
                                            cg_params.max_iter, cg_params.tolerance,
                                            cg_params.residue_restart_step, cgsolver_iter, comm_);
  }

  return cgsolver_iter;
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::GradientRandElementSign_() {
  if (rank_ == kMPIMasterRank)
    for (size_t row = 0; row < ly_; row++) {
      for (size_t col = 0; col < lx_; col++) {
        size_t dim = split_index_tps_({row, col}).size();
        for (size_t i = 0; i < dim; i++)
          grad_({row, col})[i].ElementWiseRandSign(unit_even_distribution, random_engine_);
      }
    }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpData(const bool release_mem) {
  DumpData(optimize_para.wavefunction_path, release_mem);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpData(const std::string &tps_path,
                                                                                    const bool release_mem) {
  std::string energy_data_path = "./energy";
  if (rank_ == kMPIMasterRank) {
    split_index_tps_.Dump(tps_path, release_mem);
    tps_lowest_.Dump(tps_path + "lowest", release_mem);
    if (!qlmps::IsPathExist(energy_data_path)) {
      qlmps::CreatPath(energy_data_path);
    }
  }
  MPI_Barrier(comm_); // configurations dump will collapse when creating path if there is no barrier.
  tps_sample_.config.Dump(tps_path, rank_);
  DumpVecData(energy_data_path + "/energy_sample" + std::to_string(rank_), energy_samples_);
  if (rank_ == kMPIMasterRank) {
    DumpVecData(energy_data_path + "/energy_trajectory", energy_trajectory_);
    DumpVecData(energy_data_path + "/energy_err_trajectory", energy_error_traj_);
  }
//  DumpVecData(tps_path + "/sum_configs" + std::to_string(rank_), sum_configs_);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
bool VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::AcceptanceRateCheck(
    const std::vector<double> &accept_rate) const {
  bool too_small = false;
  std::vector<double> global_max(accept_rate.size());
  HANDLE_MPI_ERROR(::MPI_Allreduce(accept_rate.data(),
                                   global_max.data(),
                                   accept_rate.size(),
                                   MPI_DOUBLE,
                                   MPI_MAX,
                                   comm_));
  for (size_t i = 0; i < accept_rate.size(); i++) {
    if (accept_rate[i] < 0.5 * global_max[i]) { //anomaly case
      too_small = true;
      std::cout << "Process " << rank_ << ": Acceptance rate[" << i
                << "] = " << accept_rate[i] << " is too small compared to global max "
                << global_max[i] << std::endl;
    }
  }
  return too_small;
}

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_IMPL_H

// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-25
*
* Description: QuantumLiquids/VMC-SquareLatticePEPS project. Implementation for abstract class of simple update.
*/

#ifndef QLPEPS_VMC_PEPS_SIMPLE_UPDATE_IMPL_H
#define QLPEPS_VMC_PEPS_SIMPLE_UPDATE_IMPL_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qlpeps {

using namespace qlten;

template<typename TenElemT, typename QNT>
using LambdaDiagonalsT = std::vector<std::vector<typename SimpleUpdateExecutor<TenElemT, QNT>::RealT>>;

inline void ValidateAdvancedStopConfig_(const SimpleUpdatePara::AdvancedStopConfig &cfg) {
  if (cfg.energy_abs_tol < 0.0) {
    throw std::invalid_argument("SimpleUpdate advanced stop: energy_abs_tol must be >= 0.");
  }
  if (cfg.energy_rel_tol < 0.0) {
    throw std::invalid_argument("SimpleUpdate advanced stop: energy_rel_tol must be >= 0.");
  }
  if (cfg.lambda_rel_tol < 0.0) {
    throw std::invalid_argument("SimpleUpdate advanced stop: lambda_rel_tol must be >= 0.");
  }
  if (cfg.patience == 0) {
    throw std::invalid_argument("SimpleUpdate advanced stop: patience must be > 0.");
  }
  if (cfg.min_steps == 0) {
    throw std::invalid_argument("SimpleUpdate advanced stop: min_steps must be > 0.");
  }
}

template<typename RealT>
RealT ComputeHybridEnergyTolerance_(const RealT prev_energy,
                                    const RealT curr_energy,
                                    const SimpleUpdatePara::AdvancedStopConfig &cfg) {
  const RealT energy_scale = std::max({RealT(1), std::abs(prev_energy), std::abs(curr_energy)});
  return std::max(static_cast<RealT>(cfg.energy_abs_tol),
                  static_cast<RealT>(cfg.energy_rel_tol) * energy_scale);
}

template<typename RealT>
bool LambdaBondDimensionsStable_(const std::vector<std::vector<RealT>> &prev_lambda_diags,
                                 const std::vector<std::vector<RealT>> &curr_lambda_diags) {
  if (prev_lambda_diags.size() != curr_lambda_diags.size()) {
    return false;
  }
  for (size_t i = 0; i < prev_lambda_diags.size(); ++i) {
    if (prev_lambda_diags[i].size() != curr_lambda_diags[i].size()) {
      return false;
    }
  }
  return true;
}

template<typename RealT>
RealT ComputeMaxRelativeLambdaDrift_(const std::vector<std::vector<RealT>> &prev_lambda_diags,
                                     const std::vector<std::vector<RealT>> &curr_lambda_diags) {
  if (!LambdaBondDimensionsStable_(prev_lambda_diags, curr_lambda_diags)) {
    return std::numeric_limits<RealT>::infinity();
  }
  const RealT eps = std::numeric_limits<RealT>::epsilon();
  RealT max_drift = RealT(0);
  for (size_t i = 0; i < prev_lambda_diags.size(); ++i) {
    RealT diff_norm2 = RealT(0);
    RealT prev_norm2 = RealT(0);
    for (size_t j = 0; j < prev_lambda_diags[i].size(); ++j) {
      const RealT diff = curr_lambda_diags[i][j] - prev_lambda_diags[i][j];
      diff_norm2 += diff * diff;
      prev_norm2 += prev_lambda_diags[i][j] * prev_lambda_diags[i][j];
    }
    const RealT denom = std::max(std::sqrt(prev_norm2), eps);
    const RealT drift = std::sqrt(diff_norm2) / denom;
    max_drift = std::max(max_drift, drift);
  }
  return max_drift;
}

template<typename TenElemT, typename QNT>
LambdaDiagonalsT<TenElemT, QNT> CaptureLambdaDiagonals_(const SquareLatticePEPS<TenElemT, QNT> &peps) {
  using RealT = typename SimpleUpdateExecutor<TenElemT, QNT>::RealT;
  LambdaDiagonalsT<TenElemT, QNT> lambda_diags;
  lambda_diags.reserve(peps.lambda_vert.rows() * peps.lambda_vert.cols()
                       + peps.lambda_horiz.rows() * peps.lambda_horiz.cols());

  auto capture_diag = [&lambda_diags](const QLTensor<RealT, QNT> &lambda) {
    const auto &shape = lambda.GetShape();
    if (shape.size() < 2) {
      lambda_diags.emplace_back();
      return;
    }
    const size_t dim = std::min(shape[0], shape[1]);
    std::vector<RealT> diag(dim, RealT(0));
    for (size_t i = 0; i < dim; ++i) {
      diag[i] = lambda({i, i});
    }
    lambda_diags.push_back(std::move(diag));
  };

  for (size_t row = 0; row < peps.lambda_vert.rows(); ++row) {
    for (size_t col = 0; col < peps.lambda_vert.cols(); ++col) {
      capture_diag(peps.lambda_vert({row, col}));
    }
  }
  for (size_t row = 0; row < peps.lambda_horiz.rows(); ++row) {
    for (size_t col = 0; col < peps.lambda_horiz.cols(); ++col) {
      capture_diag(peps.lambda_horiz({row, col}));
    }
  }
  return lambda_diags;
}

//helper
std::vector<size_t> DuplicateElements(const std::vector<size_t> &input) {
  std::vector<size_t> output;
  output.reserve(input.size() * 2); // Reserve space for efficiency

  for (size_t element : input) {
    output.push_back(element);
    output.push_back(element);
  }

  return output;
}

/**
 * Calculate the exp(-tau*H) with tau a small number by Taylor expansion
 *
 * @param tau
 * @param ham
 *  N site Hamiltonian term with 2*N tensor indexes. The indexes is ordered specifically.
 *  E.g. for 2-site Hamiltonian, the hamiltonian indexes is ordered in the following figure.
 *  The 0,2 indexes will be projected to peps, and 0,1 legs for site 1.
 *
 *          1         3
 *          |         |
 *          v         v
 *          |---ham---|
 *          v         v
 *          |         |
 *          0         2
 *
 *  One more example is the 3-site hamiltonian:
 *          1         3         5
 *          |         |         |
 *          v         v         v
 *          |--------ham--------|
 *          v         v         v
 *          |         |         |
 *          0         2         4
 * @return
 *  2*N indexes tensor which has different order with that of the Hamiltonian.
 *  E.g. for 3-site hamiltonian, the indexes order of the return tensor is
 *
 *          3         4         5
 *          |         |         |
 *          v         v         v
 *          |----exp(-tau*H)----|
 *          v         v         v
 *          |         |         |
 *          0         1         2
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> TaylorExpMatrix(const typename qlten::RealTypeTrait<TenElemT>::type tau, const QLTensor<TenElemT, QNT> &ham) {
  using Tensor = QLTensor<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  const size_t N = ham.Rank() / 2;
  Tensor ham_scale = -tau * ham;
  //transpose so that in leg first.
  std::vector<size_t> transpose_axes(2 * N, 0);
  for (size_t i = 0; i < N; i++) {
    transpose_axes[i] = 2 * i;
    transpose_axes[N + i] = 2 * i + 1;
  }
  ham_scale.Transpose(transpose_axes);
  //generate the Identity tensor
  Tensor id = Tensor(ham.GetIndexes());
  ShapeT shape = ham_scale.GetShape();
  shape.erase(shape.begin() + N, shape.end());
  std::vector<CoorsT> all_coors = GenAllCoors(shape);
  for (const auto &coor : all_coors) {
    id(DuplicateElements(coor)) = RealT(1.0);
  }
  id.Transpose(transpose_axes);
  if (Tensor::IsFermionic() && id.GetIndex(0).GetDir() == OUT) {
    id.ActFermionPOps();
  }

  std::vector<Tensor> taylor_terms = {id, ham_scale};
  taylor_terms.reserve(kMaxTaylorExpansionOrder);
  std::vector<size_t> ctrct_axes1(N), ctrct_axes2(N);
  std::iota(ctrct_axes1.begin(), ctrct_axes1.end(), N);
  std::iota(ctrct_axes2.begin(), ctrct_axes2.end(), 0);
  
  const RealT kEpsilon = (sizeof(RealT) == sizeof(double)) ? kDoubleEpsilon : kFloatEpsilon;

  for (size_t n = 2; n < kMaxTaylorExpansionOrder; n++) {
    Tensor tmp;
    Contract(&taylor_terms.back(), ctrct_axes1, &ham_scale, ctrct_axes2, &tmp);
    tmp *= RealT(1.0) / RealT(n);
    taylor_terms.emplace_back(tmp);
    if (tmp.GetQuasi2Norm() < kEpsilon) {
      std::cout << "calculate the evolution gate taylor series order: " << n << std::endl;
      break;
    }
    if (n == kMaxTaylorExpansionOrder - 1) {
      std::cout << "warning: taylor expansions for evolution gate do not converge "
                << "with precision: " << tmp.GetQuasi2Norm() << "." << std::endl;
    }
  }
  Tensor expH = taylor_terms[0];
  for (size_t n = 1; n < taylor_terms.size(); n++) {
    expH += taylor_terms[n];
  }
  return expH;
}

/**
 *
 * @param update_para
 * @param ham_nn
 *          1         3
 *          |         |
 *          ^         ^
 *          |---ham---|
 *          ^         ^
 *          |         |
 *          0         2
 * @param peps_initial
 * @return
 */
template<typename TenElemT, typename QNT>
SimpleUpdateExecutor<TenElemT, QNT>::SimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                                                          const PEPST &peps_initial)
    : Executor(), lx_(peps_initial.Cols()), ly_(peps_initial.Rows()),
      update_para(update_para), peps_(peps_initial) {
  std::cout << "\n";
  std::cout << "=====> SIMPLE UPDATE PROGRAM FOR Square-Lattice PEPS <=====" << "\n";
  std::cout << std::setw(40) << "System size (lx, ly) : " << "(" << lx_ << ", " << ly_ << ")\n";
  std::cout << std::setw(40) << "SquareLatticePEPS bond dimension : " << update_para.Dmin << "/" << update_para.Dmax
            << "\n";
  std::cout << std::setw(40) << "Trotter step : " << update_para.tau << "\n";

  std::cout << "=====> TECHNICAL PARAMETERS <=====" << "\n";
  std::cout << std::setw(40) << "The number of threads per processor : " << hp_numeric::GetTensorManipulationThreads()
            << "\n";
  SetStatus(qlten::INITED);
}

template<typename TenElemT, typename QNT>
void SimpleUpdateExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(qlten::EXEING);
  SetEvolveGate_();
  last_run_summary_ = RunSummary{};
  last_run_summary_.stop_reason = StopReason::kMaxSteps;
  step_metrics_.clear();
  step_metrics_.reserve(update_para.steps);

  size_t convergence_streak = 0;
  std::optional<RealT> prev_energy;
  LambdaDiagonalsT<TenElemT, QNT> prev_lambda_diags;
  bool has_prev_lambda_diags = false;
  const bool use_advanced_stop = update_para.advanced_stop.has_value();
  if (use_advanced_stop) {
    ValidateAdvancedStopConfig_(update_para.advanced_stop.value());
  }

  size_t prev_dmin = 0, prev_dmax = 0;

  for (size_t step = 0; step < update_para.steps; step++) {
    std::cout << "step = " << step << "\t";
    SweepResult sweep_result = SimpleUpdateSweep_();
    estimated_energy_ = sweep_result.estimated_e0;
    ++last_run_summary_.executed_steps;
    if (estimated_energy_.has_value()) {
      last_run_summary_.final_energy = estimated_energy_.value();
      last_run_summary_.final_estimated_e0 = sweep_result.estimated_e0;
      last_run_summary_.final_estimated_en = sweep_result.estimated_en;
    }

    // Detect bond dimension change
    bool bond_dim_changed = false;
    if (step > 0) {
      bond_dim_changed = (sweep_result.dmin != prev_dmin || sweep_result.dmax != prev_dmax);
    }
    prev_dmin = sweep_result.dmin;
    prev_dmax = sweep_result.dmax;

    // Record step metrics
    StepMetrics metrics;
    metrics.step_index = step;
    metrics.tau = update_para.tau;
    metrics.estimated_e0 = sweep_result.estimated_e0;
    metrics.estimated_en = sweep_result.estimated_en;
    metrics.trunc_err = sweep_result.trunc_err;  // propagate optional
    metrics.elapsed_sec = sweep_result.elapsed_sec;
    metrics.bond_dim_changed = bond_dim_changed;
    step_metrics_.push_back(metrics);

    // Invoke observer callback if set
    if (update_para.step_observer.has_value()) {
      SimpleUpdateStepMetrics<double> cb_metrics;
      cb_metrics.step_index = metrics.step_index;
      cb_metrics.tau = metrics.tau;
      cb_metrics.estimated_e0 = static_cast<double>(metrics.estimated_e0);
      cb_metrics.estimated_en = static_cast<double>(metrics.estimated_en);
      cb_metrics.trunc_err = metrics.trunc_err.has_value()
          ? std::optional<double>(static_cast<double>(metrics.trunc_err.value()))
          : std::nullopt;
      cb_metrics.elapsed_sec = metrics.elapsed_sec;
      cb_metrics.bond_dim_changed = metrics.bond_dim_changed;
      update_para.step_observer.value()(cb_metrics);
    }

    // Emit machine-readable metrics if enabled
    if (update_para.emit_machine_readable_metrics) {
      std::cout << "SU_METRIC"
                << " step=" << step
                << " tau=" << update_para.tau
                << " e0=" << std::setprecision(15) << sweep_result.estimated_e0
                << " en=" << std::setprecision(15) << sweep_result.estimated_en;
      if (sweep_result.trunc_err.has_value()) {
        std::cout << " trunc_err=" << std::setprecision(6) << std::scientific
                  << sweep_result.trunc_err.value() << std::fixed;
      } else {
        std::cout << " trunc_err=N/A";
      }
      std::cout << " elapsed_sec=" << sweep_result.elapsed_sec
                << std::endl;
    }

    if (!use_advanced_stop || !estimated_energy_.has_value()) {
      continue;
    }

    const RealT curr_energy = estimated_energy_.value();
    const auto curr_lambda_diags = CaptureLambdaDiagonals_<TenElemT, QNT>(peps_);
    bool gate_passed = false;

    if (prev_energy.has_value() && has_prev_lambda_diags) {
      const bool dims_stable = LambdaBondDimensionsStable_(prev_lambda_diags, curr_lambda_diags);
      if (dims_stable) {
        const RealT energy_tol = ComputeHybridEnergyTolerance_(
            prev_energy.value(), curr_energy, update_para.advanced_stop.value());
        const RealT energy_delta = std::abs(curr_energy - prev_energy.value());
        const bool energy_ok = (energy_delta <= energy_tol);

        const RealT lambda_drift = ComputeMaxRelativeLambdaDrift_(prev_lambda_diags, curr_lambda_diags);
        const bool lambda_ok =
            (lambda_drift <= static_cast<RealT>(update_para.advanced_stop->lambda_rel_tol));
        const bool min_steps_ok =
            (last_run_summary_.executed_steps >= update_para.advanced_stop->min_steps);
        gate_passed = min_steps_ok && energy_ok && lambda_ok;
      }
    }

    if (gate_passed) {
      ++convergence_streak;
    } else {
      convergence_streak = 0;
    }
    if (convergence_streak >= update_para.advanced_stop->patience) {
      last_run_summary_.converged = true;
      last_run_summary_.stop_reason = StopReason::kAdvancedConverged;
      std::cout << "Stopping: advanced simple-update convergence reached after "
                << last_run_summary_.executed_steps << " sweeps." << std::endl;
      break;
    }

    prev_energy = curr_energy;
    prev_lambda_diags = curr_lambda_diags;
    has_prev_lambda_diags = true;
  }
  SetStatus(qlten::FINISH);
}

template<typename RealT, typename QNT>
void PrintLambda(const QLTensor<RealT, QNT> &lambda) {
  std::cout << std::setprecision(4) << std::scientific;

  // Extract the diagonal elements of lambda into a vector
  std::vector<RealT> diagonal_elements(lambda.GetShape()[0]);
  for (size_t i = 0; i < lambda.GetShape()[0]; i++) {
    diagonal_elements[i] = lambda({i, i});
  }

  // Sort the diagonal elements in descending order
  std::sort(diagonal_elements.begin(), diagonal_elements.end(), std::greater<RealT>());

  // Print the sorted elements
  std::cout << "[";
  for (const auto &element : diagonal_elements) {
    std::cout << " " << element;
  }
  std::cout << " ]" << std::endl;
}
}//qlpeps;
#endif //QLPEPS_VMC_PEPS_SIMPLE_UPDATE_IMPL_H

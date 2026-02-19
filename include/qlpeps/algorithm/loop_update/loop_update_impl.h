// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-08-05
*
* Description: QuantumLiquids/PEPS project. Loop update implementation.
*
*/


#ifndef QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_IMPL_H
#define QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_IMPL_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qlpeps {
using namespace qlten;

template<typename RealT>
using LoopLambdaDiagonalsT = std::vector<std::vector<RealT>>;

inline void LoopUpdateValidateAdvancedStopConfig_(
    const LoopUpdatePara::AdvancedStopConfig &cfg) {
  if (cfg.energy_abs_tol < 0.0) {
    throw std::invalid_argument("LoopUpdate advanced stop: energy_abs_tol must be >= 0.");
  }
  if (cfg.energy_rel_tol < 0.0) {
    throw std::invalid_argument("LoopUpdate advanced stop: energy_rel_tol must be >= 0.");
  }
  if (cfg.lambda_rel_tol < 0.0) {
    throw std::invalid_argument("LoopUpdate advanced stop: lambda_rel_tol must be >= 0.");
  }
  if (cfg.patience == 0) {
    throw std::invalid_argument("LoopUpdate advanced stop: patience must be > 0.");
  }
  if (cfg.min_steps == 0) {
    throw std::invalid_argument("LoopUpdate advanced stop: min_steps must be > 0.");
  }
}

template<typename RealT>
RealT LoopUpdateComputeHybridEnergyTolerance_(
    const RealT prev_energy,
    const RealT curr_energy,
    const LoopUpdatePara::AdvancedStopConfig &cfg) {
  const RealT energy_scale = std::max(
      {RealT(1), std::abs(prev_energy), std::abs(curr_energy)});
  return std::max(static_cast<RealT>(cfg.energy_abs_tol),
                  static_cast<RealT>(cfg.energy_rel_tol) * energy_scale);
}

template<typename RealT>
bool LoopUpdateLambdaBondDimensionsStable_(
    const LoopLambdaDiagonalsT<RealT> &prev_lambda_diags,
    const LoopLambdaDiagonalsT<RealT> &curr_lambda_diags) {
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
RealT LoopUpdateComputeMaxRelativeLambdaDrift_(
    const LoopLambdaDiagonalsT<RealT> &prev_lambda_diags,
    const LoopLambdaDiagonalsT<RealT> &curr_lambda_diags) {
  if (!LoopUpdateLambdaBondDimensionsStable_(prev_lambda_diags, curr_lambda_diags)) {
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
LoopLambdaDiagonalsT<typename qlten::RealTypeTrait<TenElemT>::type> CaptureLoopLambdaDiagonals_(
    const SquareLatticePEPS<TenElemT, QNT> &peps) {
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  LoopLambdaDiagonalsT<RealT> lambda_diags;
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

// returned id_nn can be directly used in the simple update
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> GenerateNNId(const Index<QNT> &phy_idx) {
  auto phy_in = InverseIndex(phy_idx);
  QLTensor<TenElemT, QNT> id_nn({phy_in, phy_idx, phy_in, phy_idx});
  const size_t dim = phy_idx.dim();
  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < dim; j++) {
      id_nn({i, i, j, j}) = 1.0;
    }
  }
  id_nn.Transpose({0, 2, 1, 3});
  return id_nn;
}

template<typename TenElemT, typename QNT>
LoopUpdateExecutor<TenElemT, QNT>::LoopUpdateExecutor(const LoopUpdatePara &para,
                                                      const DuoMatrix<LoopUpdateExecutor<TenElemT,
                                                                                         QNT>::LoopGatesT> &evolve_gates,
                                                      const LoopUpdateExecutor<TenElemT, QNT>::PEPST &peps_initial) :
    Executor(),
    lx_(peps_initial.Cols()),
    ly_(peps_initial.Rows()),
    para_(para),
    evolve_gates_(evolve_gates),
    id_nn_(GenerateNNId<TenElemT, QNT>(peps_initial.Gamma({0, 0}).GetIndex(4))),
    peps_(peps_initial) {
  const auto bc = peps_initial.GetBoundaryCondition();
  if (bc == BoundaryCondition::Periodic) {
    if (lx_ % 2 != 0 || ly_ % 2 != 0) {
      throw std::invalid_argument(
          "LoopUpdateExecutor: PBC loop update requires even lattice dimensions. "
          "Got (" + std::to_string(lx_) + ", " + std::to_string(ly_) + ").");
    }
  }

  // Validate evolve_gates shape matches boundary condition
  const size_t expected_gate_rows = (bc == BoundaryCondition::Periodic) ? ly_ : ly_ - 1;
  const size_t expected_gate_cols = (bc == BoundaryCondition::Periodic) ? lx_ : lx_ - 1;
  if (evolve_gates_.rows() != expected_gate_rows || evolve_gates_.cols() != expected_gate_cols) {
    throw std::invalid_argument(
        "LoopUpdateExecutor: evolve_gates shape mismatch. "
        "Expected (" + std::to_string(expected_gate_rows) + ", " + std::to_string(expected_gate_cols) + ") "
        "for " + std::string(bc == BoundaryCondition::Periodic ? "PBC" : "OBC") + " "
        + std::to_string(ly_) + "x" + std::to_string(lx_) + " lattice, "
        "got (" + std::to_string(evolve_gates_.rows()) + ", " + std::to_string(evolve_gates_.cols()) + ").");
  }

  std::cout << "\n";
  std::cout << "=====> LOOP UPDATE PROGRAM FOR Square-Lattice PEPS <=====" << "\n";
  std::cout << std::setw(40) << "System size (lx, ly) : " << "(" << lx_ << ", " << ly_ << ")\n";
  std::cout << std::setw(40) << "Boundary condition : "
            << (bc == BoundaryCondition::Periodic ? "Periodic" : "Open") << "\n";
  std::cout << std::setw(40) << "Setting bond dimension : " << para_.truncate_para.fet_params.Dmin << "/"
            << para_.truncate_para.fet_params.Dmax
            << "\n";
  std::cout << std::setw(40) << "Evolving steps :" << para_.steps << "\n";
  std::cout << std::setw(40) << "Trotter step : " << para_.tau << "\n";
  std::cout << std::setw(40) << "Gate type : "
            << (para_.gate_type == LoopGateType::kFirstOrder ? "1 - tau*H (first-order)" : "exp(-tau*H) (exponential)")
            << "\n";

  std::cout << "=====> TECHNICAL PARAMETERS <=====" << "\n";
  std::cout << std::setw(40) << "The number of threads per processor : " << hp_numeric::GetTensorManipulationThreads()
            << "\n";
  SetStatus(qlten::INITED);
}

template<typename TenElemT, typename QNT>
LoopUpdateExecutor<TenElemT, QNT>::LoopUpdateExecutor(const LoopUpdateTruncatePara &truncate_para,
                                                      const size_t steps,
                                                      const double tau,
                                                      const DuoMatrix<LoopUpdateExecutor<TenElemT,
                                                                                         QNT>::LoopGatesT> &evolve_gates,
                                                      const LoopUpdateExecutor<TenElemT, QNT>::PEPST &peps_initial) :
    LoopUpdateExecutor(LoopUpdatePara(truncate_para, steps, tau), evolve_gates, peps_initial) {}

template<typename TenElemT, typename QNT>
void LoopUpdateExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(qlten::EXEING);
  last_run_summary_ = RunSummary{};
  last_run_summary_.stop_reason = StopReason::kMaxSteps;
  step_metrics_.clear();
  step_metrics_.reserve(para_.steps);

  size_t convergence_streak = 0;
  std::optional<RealT> prev_energy;
  LoopLambdaDiagonalsT<RealT> prev_lambda_diags;
  bool has_prev_lambda_diags = false;
  const bool use_advanced_stop = para_.advanced_stop.has_value();
  if (use_advanced_stop) {
    LoopUpdateValidateAdvancedStopConfig_(para_.advanced_stop.value());
  }

  size_t prev_dmin = 0, prev_dmax = 0;

  for (size_t step = 0; step < para_.steps; step++) {
    std::cout << "step = " << step << "\n";
    SweepResult sweep_result = LoopUpdateSweep_();
    estimated_energy_ = static_cast<double>(sweep_result.estimated_e0);
    ++last_run_summary_.executed_steps;
    last_run_summary_.final_energy = sweep_result.estimated_e0;
    last_run_summary_.final_estimated_e0 = sweep_result.estimated_e0;
    last_run_summary_.final_estimated_en = sweep_result.estimated_en;

    bool bond_dim_changed = false;
    if (step > 0) {
      bond_dim_changed = (sweep_result.dmin != prev_dmin || sweep_result.dmax != prev_dmax);
    }
    prev_dmin = sweep_result.dmin;
    prev_dmax = sweep_result.dmax;

    StepMetrics metrics;
    metrics.step_index = step;
    metrics.tau = para_.tau;
    metrics.estimated_e0 = sweep_result.estimated_e0;
    metrics.estimated_en = sweep_result.estimated_en;
    metrics.trunc_err = sweep_result.trunc_err;
    metrics.elapsed_sec = sweep_result.elapsed_sec;
    metrics.bond_dim_changed = bond_dim_changed;
    step_metrics_.push_back(metrics);

    if (para_.step_observer.has_value()) {
      LoopUpdateStepMetrics<double> cb_metrics;
      cb_metrics.step_index = metrics.step_index;
      cb_metrics.tau = metrics.tau;
      cb_metrics.estimated_e0 = static_cast<double>(metrics.estimated_e0);
      cb_metrics.estimated_en = static_cast<double>(metrics.estimated_en);
      cb_metrics.trunc_err = metrics.trunc_err.has_value()
          ? std::optional<double>(static_cast<double>(metrics.trunc_err.value()))
          : std::nullopt;
      cb_metrics.elapsed_sec = metrics.elapsed_sec;
      cb_metrics.bond_dim_changed = metrics.bond_dim_changed;
      para_.step_observer.value()(cb_metrics);
    }

    if (para_.emit_machine_readable_metrics) {
      std::cout << "LU_METRIC"
                << " step=" << step
                << " tau=" << para_.tau
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

    if (!use_advanced_stop) {
      continue;
    }

    const RealT curr_energy = sweep_result.estimated_e0;
    const auto curr_lambda_diags = CaptureLoopLambdaDiagonals_<TenElemT, QNT>(peps_);
    bool gate_passed = false;
    if (prev_energy.has_value() && has_prev_lambda_diags) {
      const bool dims_stable = LoopUpdateLambdaBondDimensionsStable_(
          prev_lambda_diags, curr_lambda_diags);
      if (dims_stable) {
        const RealT energy_tol = LoopUpdateComputeHybridEnergyTolerance_(
            prev_energy.value(), curr_energy, para_.advanced_stop.value());
        const RealT energy_delta = std::abs(curr_energy - prev_energy.value());
        const bool energy_ok = (energy_delta <= energy_tol);

        const RealT lambda_drift = LoopUpdateComputeMaxRelativeLambdaDrift_(
            prev_lambda_diags, curr_lambda_diags);
        const bool lambda_ok = (lambda_drift
            <= static_cast<RealT>(para_.advanced_stop->lambda_rel_tol));
        const bool min_steps_ok = (last_run_summary_.executed_steps
            >= para_.advanced_stop->min_steps);
        gate_passed = min_steps_ok && energy_ok && lambda_ok;
      }
    }
    if (gate_passed) {
      ++convergence_streak;
    } else {
      convergence_streak = 0;
    }
    if (convergence_streak >= para_.advanced_stop->patience) {
      last_run_summary_.converged = true;
      last_run_summary_.stop_reason = StopReason::kAdvancedConverged;
      std::cout << "Stopping: advanced loop-update convergence reached after "
                << last_run_summary_.executed_steps << " sweeps." << std::endl;
      break;
    }

    prev_energy = curr_energy;
    prev_lambda_diags = curr_lambda_diags;
    has_prev_lambda_diags = true;
  }
  SetStatus(qlten::FINISH);
}

template<typename TenElemT, typename QNT>
std::pair<double, double> LoopUpdateExecutor<TenElemT, QNT>::UpdateOneLoop(const qlpeps::SiteIdx &site,
                                                                           const qlpeps::LoopUpdateTruncatePara &para,
                                                                           const bool print_time) {
  const LoopGatesT &gate = evolve_gates_(site);
  return this->peps_.LocalSquareLoopProject(gate, site, para, print_time);
}

template<typename TenElemT, typename QNT>
typename LoopUpdateExecutor<TenElemT, QNT>::SweepResult
LoopUpdateExecutor<TenElemT, QNT>::LoopUpdateSweep_(void) {
  Timer loop_update_sweep_timer("loop_update_sweep");

  RealT e0 = 0;
  RealT norm_product = 1;

  const bool is_pbc = (peps_.GetBoundaryCondition() == BoundaryCondition::Periodic);
  const size_t col_limit = is_pbc ? this->lx_ : this->lx_ - 1;
  const size_t row_limit = is_pbc ? this->ly_ : this->ly_ - 1;

  // TODO: add OpenMP parallelization for independent plaquettes (checkerboard decomposition)
  for (size_t start_col : {0, 1})
    for (size_t start_row : {0, 1}) {
      for (size_t col = start_col; col < col_limit; col += 2) {
        for (size_t row = start_row; row < row_limit; row += 2) {
          bool print_time = (row == (this->ly_ / 2) - 1 && col == (this->lx_ / 2 - 1));
          const LoopGatesT &gate = evolve_gates_({row, col});
          auto proj_res = this->peps_.LocalSquareLoopProject(gate, {row, col}, para_.truncate_para, print_time);
          norm_product *= proj_res.first;
          if (para_.gate_type == LoopGateType::kFirstOrder) {
            e0 += (RealT(1) - proj_res.second) / RealT(para_.tau);
          } else {
            e0 += -std::log(proj_res.second) / RealT(para_.tau);
          }
        }
      }
      // TODO: re-enable identity NN simple-update canonicalization sweeps
    }

  if (norm_product <= RealT(0)) {
    throw std::runtime_error("LoopUpdateSweep_: non-positive norm product encountered.");
  }
  RealT estimated_en = -std::log(norm_product) / RealT(para_.tau);
  double sweep_time = loop_update_sweep_timer.Elapsed();
  auto [dmin, dmax] = this->peps_.GetMinMaxBondDim();
  std::cout << "Estimated E0 =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << e0
            << "Estimated En =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << estimated_en
            << " Dmin/Dmax = " << std::setw(2) << std::right << dmin << "/" << std::setw(2) << std::left << dmax
            << " SweepTime = " << std::setw(8) << sweep_time
            << "\n";
  std::cout << "lambda tensors in middle : " << std::endl;
  PrintLambda(this->peps_.lambda_vert({this->ly_ / 2, this->lx_ / 2}));
  PrintLambda(this->peps_.lambda_horiz({this->ly_ / 2, this->lx_ / 2}));
  return {e0, estimated_en, std::nullopt, sweep_time, dmin, dmax};

}

}

#endif //QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_IMPL_H

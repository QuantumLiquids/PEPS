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

namespace qlpeps {
using namespace qlten;

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
  for (size_t step = 0; step < para_.steps; step++) {
    std::cout << "step = " << step << "\n";
    estimated_energy_ = LoopUpdateSweep_();
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
double LoopUpdateExecutor<TenElemT, QNT>::LoopUpdateSweep_(void) {
  Timer loop_update_sweep_timer("loop_update_sweep");

  double e0 = 0.0;

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
          if (para_.gate_type == LoopGateType::kFirstOrder) {
            e0 += (1 - proj_res.second) / para_.tau;
          } else {
            e0 += -std::log(proj_res.second) / para_.tau;
          }
        }
      }
      // TODO: re-enable identity NN simple-update canonicalization sweeps
    }

  double sweep_time = loop_update_sweep_timer.Elapsed();
  auto [dmin, dmax] = this->peps_.GetMinMaxBondDim();
  std::cout << "Estimated E0 =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << e0
            << " Dmin/Dmax = " << std::setw(2) << std::right << dmin << "/" << std::setw(2) << std::left << dmax
            << " SweepTime = " << std::setw(8) << sweep_time
            << "\n";
  std::cout << "lambda tensors in middle : " << std::endl;
  PrintLambda(this->peps_.lambda_vert({this->ly_ / 2, this->lx_ / 2}));
  PrintLambda(this->peps_.lambda_horiz({this->ly_ / 2, this->lx_ / 2}));
  return e0;

}

}

#endif //QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_IMPL_H

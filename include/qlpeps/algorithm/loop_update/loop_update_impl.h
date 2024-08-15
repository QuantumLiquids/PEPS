/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-08-05
*
* Description: QuantumLiquids/VMC-SquareLatticePEPS project. Loop update implementation.
*
*/


#ifndef QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_IMPL_H
#define QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_IMPL_H

namespace qlpeps {
using namespace qlten;

template<typename TenElemT, typename QNT>
LoopUpdateExecutor<TenElemT, QNT>::LoopUpdateExecutor(const LoopUpdateTruncatePara &truncate_para,
                                                      const size_t steps,
                                                      const double tau,
                                                      const DuoMatrix<LoopUpdateExecutor<TenElemT,
                                                                                         QNT>::LoopGatesT> &evolve_gates,
                                                      const LoopUpdateExecutor<TenElemT, QNT>::PEPST &peps_initial) :
    Executor(),
    lx_(peps_initial.Cols()),
    ly_(peps_initial.Rows()),
    steps_(steps),
    tau_(tau),
    evolve_gates_(evolve_gates),
    truncate_para_(truncate_para),
    peps_(peps_initial) {
  std::cout << "\n";
  std::cout << "=====> LOOP UPDATE PROGRAM FOR Square-Lattice PEPS <=====" << "\n";
  std::cout << std::setw(40) << "System size (lx, ly) : " << "(" << lx_ << ", " << ly_ << ")\n";
  std::cout << std::setw(40) << "Setting bond dimension : " << truncate_para.fet_params.Dmin << "/"
            << truncate_para.fet_params.Dmax
            << "\n";
  std::cout << std::setw(40) << "Evolving steps :" << steps << "\n";
  std::cout << std::setw(40) << "Trotter step : " << tau << "\n";

  std::cout << "=====> TECHNICAL PARAMETERS <=====" << "\n";
  std::cout << std::setw(40) << "The number of threads per processor : " << omp_get_thread_num()
            << "\n";
  SetStatus(qlten::INITED);
}

template<typename TenElemT, typename QNT>
void LoopUpdateExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(qlten::EXEING);
  for (size_t step = 0; step < steps_; step++) {
    std::cout << "step = " << step << "\n";
    LoopUpdateSweep_();
  }
  SetStatus(qlten::FINISH);
}

template<typename TenElemT, typename QNT>
double LoopUpdateExecutor<TenElemT, QNT>::UpdateOneLoop(const qlpeps::SiteIdx &site,
                                                        const qlpeps::LoopUpdateTruncatePara &para,
                                                        const bool print_time) {
  const LoopGatesT &gate = evolve_gates_(site);
  return this->peps_.LocalSquareLoopProject(gate, site, para, print_time);
}

template<typename TenElemT, typename QNT>
double LoopUpdateExecutor<TenElemT, QNT>::LoopUpdateSweep_(void) {
  Timer loop_update_sweep_timer("loop_update_sweep");

  double e0 = 0.0;

  if (omp_get_thread_num() == 1) {
    for (size_t col = 0; col < this->lx_ - 1; col++) {
      for (size_t row = 0; row < this->ly_ - 1; row++) {
        double norm;
        if (row == (this->ly_ / 2) - 1 && col == (this->lx_ / 2 - 1)) {
          norm = UpdateOneLoop({row, col}, truncate_para_, true);
        } else {
          norm = UpdateOneLoop({row, col}, truncate_para_, false);
        }
        e0 += -std::log(norm) / tau_;
      }
    }
  } else {
    for (size_t start_col : {0, 1})
      for (size_t start_row : {0, 1}) {
#pragma omp parallel for collapse(2) reduction(+:e0) shared(evolve_gates_, truncate_para_)
        for (size_t col = start_col; col < this->lx_ - 1; col += 2) {
          for (size_t row = start_row; row < this->ly_ - 1; row += 2) {
            double norm;
            const LoopGatesT &gate = evolve_gates_({row, col});
            if (row == (this->ly_ / 2) - 1 && col == (this->lx_ / 2 - 1)) {
              norm = this->peps_.LocalSquareLoopProject(gate, {row, col}, truncate_para_, true);
            } else {
              norm = this->peps_.LocalSquareLoopProject(gate, {row, col}, truncate_para_, false);
            }
            e0 += -std::log(norm) / tau_;
          }
        }
      }
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
  return 0.0;

}

}

#endif //QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_IMPL_H

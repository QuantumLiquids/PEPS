/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-06-21
*
* Description: QuantumLiquids/VMC-SquareLatticePEPS project. Loop update and its implementation.
* Reference: [1] PRB 102, 075147 (2020), "Loop update for iPEPS in 2D".
*
*/


#ifndef QLPEPS_ALGORITHM_SIMPLE_UPDATE_LOOP_UPDATE_H
#define QLPEPS_ALGORITHM_SIMPLE_UPDATE_LOOP_UPDATE_H

#include "qlpeps/algorithm/simple_update/simple_update.h"
#include "qlpeps/two_dim_tn/framework/duomatrix.h"        //DuoMatrix

namespace qlpeps {

using namespace qlten;
template<typename TenElemT, typename QNT>
class LoopUpdate : public SimpleUpdateExecutor<TenElemT, QNT> {
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
  using LoopGateT = std::array<Tensor, 4>;
 public:
  LoopUpdate(const SimpleUpdatePara &simple_update_para,
             const LanczosParams &lanczos_params, // for gauge fixing
             const double fet_tol,                //for FET truncation
             const size_t fet_max_iter,           //for FET truncation
             const ConjugateGradientParams cg_params, //for FET truncation
             const PEPST &peps_initial,
             const DuoMatrix<LoopGateT> &evolve_gates) :
      SimpleUpdateExecutor<TenElemT, QNT>(simple_update_para, peps_initial), evolve_gates_(evolve_gates),
      lanczos_params_(lanczos_params), fet_tol_(fet_tol), fet_max_iter_(fet_max_iter),
      cg_params_(cg_params) {}

 private:
  void SetEvolveGate_(void) override {
    //do nothing
  }

  double SimpleUpdateSweep_(void) override;

  /**
   *
   * @param site: the coordinate of the left-upper site in the loop
   * @return
   */
  double UpdateOneLoop(const SiteIdx &site, const LoopUpdateTruncatePara &para);

  /** The set of the evolve gates
   *  where each gate form a loop and includes 4 tensors, and
   *  represents the local imaginary evolve gate exp(-\tau * h).
   *  sizes of the DuoMatrix are (Ly-1) in rows by (Lx-1) in columns
   *
   *  The order of the 4 tensors in one loop is accord to Ref. [1] Fig. 2 (a)
   *
   *  And the orders of the legs in MPO tensors are
   *
   *        2                  2
   *        |                  |
   *        |                  |
   *  0---[gate 0]---3  0---[gate 1]---3
   *        |                  |
   *        |                  |
   *        1                  1
   *  so on so forth for gate 2 and 3. Note the leg 3 of gate 0 are connected to
   *  the leg 0 of gate 1, so on so forth.
   *  And the legs 1 of the gates are connected to PEPS physical legs.
   *  (The diagrams here are upside down from the Fig. in the Ref. [1].)
   *
   */
  DuoMatrix<LoopGateT> evolve_gates_;
  //addtional parameters
  LanczosParams lanczos_params_;
  double fet_tol_;
  size_t fet_max_iter_;
  ConjugateGradientParams cg_params_;
};

template<typename TenElemT, typename QNT>
double LoopUpdate<TenElemT, QNT>::UpdateOneLoop(const qlpeps::SiteIdx &site,
                                                const qlpeps::LoopUpdateTruncatePara &para) {
  const LoopGateT &gate = evolve_gates_(site);
  return this->peps_.LocalSquareLoopProject(gate, site, para);
}

template<typename TenElemT, typename QNT>
double LoopUpdate<TenElemT, QNT>::SimpleUpdateSweep_(void) {
  Timer loop_update_sweep_timer("loop_update_sweep");
  LoopUpdateTruncatePara para(lanczos_params_,
                              this->update_para.Dmin, this->update_para.Dmax, this->update_para.Trunc_err,
                              fet_tol_, fet_max_iter_, cg_params_);
  double norm;
  double e0 = 0.0;

  for (size_t col = 0; col < this->lx_ - 1; col++) {
    for (size_t row = 0; row < this->ly_ - 1; row++) {
      norm = UpdateOneLoop({row, col}, para);
      e0 += -std::log(norm) / this->update_para.tau;
    }
  }
  double sweep_time = loop_update_sweep_timer.Elapsed();
  auto [dmin, dmax] = this->peps_.GetMinMaxBondDim();
  std::cout << "Estimated E0 =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << e0
            << " Dmin/Dmax = " << std::setw(2) << std::right << dmin << "/" << std::setw(2) << std::left << dmax
            << " SweepTime = " << std::setw(8) << sweep_time
            << std::endl;
  return 0.0;
}

}//qlpeps

#endif //QLPEPS_ALGORITHM_SIMPLE_UPDATE_LOOP_UPDATE_H

// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: GraceQ/VMC-PEPS project. Simple Update.
*/

#ifndef VMC_PEPS_ALGORITHM_SIMPLE_UPDATE_SIMPLE_UPDATE_H
#define VMC_PEPS_ALGORITHM_SIMPLE_UPDATE_SIMPLE_UPDATE_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/peps/peps.h"    //PEPS

namespace gqpeps {

using namespace gqten;

struct SimpleUpdatePara {
  size_t steps;
  double tau;         // Step length

  size_t Dmin;
  size_t Dmax;        // Bond dimension
  double Trunc_err;   // Truncation error

  SimpleUpdatePara(size_t steps, double tau, size_t Dmin, size_t Dmax, double Trunc_err)
      : steps(steps), tau(tau), Dmin(Dmin), Dmax(Dmax), Trunc_err(Trunc_err) {}
};
/** SimpleUpdateExecutor
 * execution for simple update in the optimization of PEPS
 *
 *
 * @tparam TenElemT
 * @tparam QNT
 */
template<typename TenElemT, typename QNT>
class SimpleUpdateExecutor : public Executor {
 public:
  using LocalTenT = GQTensor<TenElemT, QNT>;
  using PEPST = PEPS<TenElemT, QNT>;
  using GateT = Gate<TenElemT, QNT>;
  SimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                       const LocalTenT &ham_nn,
                       const PEPST &peps_initial);

  void Execute(void) override;
  const PEPST &GetPEPS(void) const {
    return peps_;
  }
  bool DumpResult(std::string path, bool release_mem) {
    return peps_.Dump(path, release_mem);
  }

  void SetStepLenth(double tau) {
    update_para.tau = tau;
    TaylorExpMatrix_(tau, ham_nn_);
  }

  SimpleUpdatePara update_para;
 private:

  GateT TaylorExpMatrix_(const double tau, const LocalTenT &
  ham);

  double SimpleUpdateSweep(void);

  LocalTenT ham_nn_;
  PEPST peps_;
  const size_t lx_;
  const size_t ly_;

  std::pair<LocalTenT, LocalTenT> evolve_gate_;
};

/**
 *
 * @tparam TenElemT
 * @tparam QNT
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
 *          1         2
 *          |         |
 *          ^         ^
 *         h1---2 0---h2
 *          ^         ^
 *          |         |
 *          0         1
 * @return
 */
template<typename TenElemT, typename QNT>
SimpleUpdateExecutor<TenElemT, QNT>::SimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                                                          const LocalTenT &ham_nn,
                                                          const PEPST &peps_initial)
    : Executor(), update_para(update_para), ham_nn_(ham_nn), peps_(peps_initial), lx_(peps_initial.Cols()),
      ly_(peps_initial.Rows()) {
  TaylorExpMatrix_(update_para.tau, ham_nn);
  SetStatus(gqten::INITED);
}

template<typename TenElemT, typename QNT>
std::pair<GQTensor<TenElemT, QNT>, GQTensor<TenElemT, QNT>>
SimpleUpdateExecutor<TenElemT, QNT>::TaylorExpMatrix_(const double tau, const LocalTenT &ham) {
  LocalTenT ham_scale = -tau * ham;
  ham_scale.Transpose({0, 2, 1, 3});
  LocalTenT id = LocalTenT(ham_scale.GetIndexes());   //Identity
  for (size_t i = 0; i < id.GetIndex(0).dim(); i++) {
    for (size_t j = 0; j < id.GetIndex(1).dim(); j++) {
      id({i, j, i, j}) = 1.0;
    }
  }
  const size_t kMaxTaylorExpansionOrder = 1000;
  std::vector<LocalTenT> taylor_terms = {id, ham_scale};
  taylor_terms.reserve(kMaxTaylorExpansionOrder);
  for (size_t n = 2; n < kMaxTaylorExpansionOrder; n++) {
    LocalTenT tmp;
    Contract(&taylor_terms.back(), &ham_scale, {{2, 3},
                                                {0, 1}}, &tmp);
    tmp *= 1.0 / double(n);
    taylor_terms.emplace_back(tmp);
    if (tmp.Get2Norm() < kDoubleEpsilon) {
      std::cout << "calculate the evolution gate taylor series order: " << n << std::endl;
      break;
    }
    if (n == kMaxTaylorExpansionOrder - 1) {
      std::cout << "warning: taylor expansion for evolution gate do not converge."
                << "error: " << tmp.Get2Norm() << std::endl;
    }
  }
  LocalTenT expH = taylor_terms[0];
  for (size_t n = 1; n < taylor_terms.size(); n++) {
    expH += taylor_terms[n];
  }

  expH.Transpose({0, 2, 1, 3});
  LocalTenT q, r;
  QR(&expH, 2, ham.Div(), &q, &r);

  evolve_gate_ = std::make_pair(q, r);
  return evolve_gate_;
}

template<typename TenElemT, typename QNT>
double SimpleUpdateExecutor<TenElemT, QNT>::SimpleUpdateSweep(void) {
  Timer simple_update_sweep_timer("simple_update_sweep");
  TruncatePara para(update_para.Dmin, update_para.Dmax, update_para.Trunc_err);
  double norm = 1.0;
  double e0 = 0.0;
#ifdef GQPEPS_TIMING_MODE
  Timer vertical_nn_projection_timer("vertical_nn_projection");
#endif
  for (size_t col = 0; col < lx_; col++) {
    for (size_t row = 0; row < ly_ - 1; row++) {
      norm = peps_.NearestNeighborSiteProject(evolve_gate_, {row, col}, VERTICAL, para);
      e0 += -std::log(norm) / update_para.tau;
    }
  }
#ifdef GQPEPS_TIMING_MODE
  vertical_nn_projection_timer.PrintElapsed();
  Timer horizontal_nn_projection_timer("horizontal_nn_projection");
#endif
  for (size_t col = 0; col < lx_ - 1; col++) {
    for (size_t row = 0; row < ly_; row++) {
      norm = peps_.NearestNeighborSiteProject(evolve_gate_, {row, col}, HORIZONTAL, para);
      e0 += -std::log(norm) / update_para.tau;
    }
  }
#ifdef GQPEPS_TIMING_MODE
  horizontal_nn_projection_timer.PrintElapsed();
#endif
  double sweep_time = simple_update_sweep_timer.Elapsed();
  std::cout << "Estimated E0 =" << std::setw(20) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << e0
            << " Dmax = " << std::setw(5) << peps_.GetMaxBondDimension()
            << " SweepTime = " << std::setw(8) << sweep_time
            << std::endl;

  return norm;
}

template<typename TenElemT, typename QNT>
void SimpleUpdateExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(gqten::EXEING);
  for (size_t step = 0; step < update_para.steps; step++) {
    std::cout << "step = " << step << "\t";
    SimpleUpdateSweep();
  }
  SetStatus(gqten::FINISH);
}

}//gqpeps

#endif //VMC_PEPS_ALGORITHM_SIMPLE_UPDATE_SIMPLE_UPDATE_H

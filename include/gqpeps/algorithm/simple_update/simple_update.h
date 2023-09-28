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
#include "gqpeps/two_dim_tn/peps/square_lattice_peps.h"    //SquareLatticePEPS

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

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> TaylorExpMatrix(const double tau, const GQTensor<TenElemT, QNT> &ham);

/** SimpleUpdateExecutor
 * execution for simple update in the optimization of SquareLatticePEPS
 *
 *
 * @tparam TenElemT
 * @tparam QNT
 */
template<typename TenElemT, typename QNT>
class SimpleUpdateExecutor : public Executor {
 public:
  using Tensor = GQTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;

  SimpleUpdateExecutor(const SimpleUpdatePara &update_para,
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
    SetEvolveGate_();
  }

  SimpleUpdatePara update_para;
 protected:

  virtual void SetEvolveGate_(void) = 0;

  virtual double SimpleUpdateSweep_(void) = 0;

  const size_t lx_;
  const size_t ly_;
  PEPST peps_;
};

}//gqpeps

#include "gqpeps/algorithm/simple_update/simple_update_impl.h"

#endif //VMC_PEPS_ALGORITHM_SIMPLE_UPDATE_SIMPLE_UPDATE_H

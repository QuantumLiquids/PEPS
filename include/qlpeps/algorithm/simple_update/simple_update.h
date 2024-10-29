// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: QuantumLiquids/PEPS project. Abstract class for simple update.
*/

#ifndef QLPEPS_ALGORITHM_SIMPLE_UPDATE_SIMPLE_UPDATE_H
#define QLPEPS_ALGORITHM_SIMPLE_UPDATE_SIMPLE_UPDATE_H

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"    //SquareLatticePEPS

namespace qlpeps {

using namespace qlten;

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
QLTensor<TenElemT, QNT> TaylorExpMatrix(const double tau, const QLTensor<TenElemT, QNT> &ham);

/** SimpleUpdateExecutor
 * abstract class for execution on simple update in SquareLatticePEPS
 */
template<typename TenElemT, typename QNT>
class SimpleUpdateExecutor : public Executor {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;

  SimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                       const PEPST &peps_initial);

  void ResetStepLenth(double tau) {
    update_para.tau = tau;
    SetEvolveGate_();
  }

  void Execute(void) override;

  const PEPST &GetPEPS(void) const {
    return peps_;
  }

  bool DumpResult(std::string path, bool release_mem) {
    return peps_.Dump(path, release_mem);
  }

  double GetEstimatedEnergy(void) const {
    if (estimated_energy_.has_value()) {
      return estimated_energy_.value();
    } else {
      std::cout << "No estimated energy value!" << std::endl;
      return 0.0;
    }
  }

  SimpleUpdatePara update_para;
 protected:

  virtual void SetEvolveGate_(void) = 0;

  // return the estimated energy
  virtual double SimpleUpdateSweep_(void) = 0;

  const size_t lx_;
  const size_t ly_;
  PEPST peps_;

  std::optional<double> estimated_energy_;
};

}//qlpeps

#include "qlpeps/algorithm/simple_update/simple_update_impl.h"

#endif //QLPEPS_ALGORITHM_SIMPLE_UPDATE_SIMPLE_UPDATE_H

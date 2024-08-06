/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-06-21
*
* Description: QuantumLiquids/VMC-SquareLatticePEPS project. Loop update executor class.
* Reference: [1] PRB 102, 075147 (2020), "Loop update for iPEPS in 2D".
*
*/


#ifndef QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_H
#define QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_H

#include "qlpeps/algorithm/simple_update/simple_update.h"
#include "qlpeps/two_dim_tn/framework/duomatrix.h"        //DuoMatrix

namespace qlpeps {

using namespace qlten;

template<typename TenElemT, typename QNT>
class LoopUpdateExecutor : public Executor {
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
  using LoopGateT = std::array<Tensor, 4>;
 public:
  LoopUpdateExecutor(const LoopUpdateTruncatePara &truncate_para,
                     const size_t steps,
                     const double tau,
                     const DuoMatrix<LoopGateT> &evolve_gates,
                     const PEPST &peps_initial);

  void Execute(void) override;

  const PEPST &GetPEPS(void) const {
    return peps_;
  }

  bool DumpResult(std::string path, bool release_mem) {
    return peps_.Dump(path, release_mem);
  }

 private:

  double LoopUpdateSweep_(void);

  /**
   *
   * @param site: the coordinate of the left-upper site in the loop
   * @return
   */
  double UpdateOneLoop(const SiteIdx &site, const LoopUpdateTruncatePara &para);

  const size_t lx_;
  const size_t ly_;
  const size_t steps_;
  const double tau_;

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
  *  so on and so forth for gate 2 and 3. Note the leg 3 of gate 0 are connected to
  *  the leg 0 of gate 1, so on so forth.
  *  And the legs 1 of the gates are connected to PEPS physical legs.
  *  (The diagrams here are upside down from the Fig. in the Ref. [1].)
  *
  */
  DuoMatrix<LoopGateT> evolve_gates_;

  LoopUpdateTruncatePara truncate_para_;

  PEPST peps_;
};

}//qlpeps

#include "qlpeps/algorithm/loop_update/loop_update_impl.h"
#endif //QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_H

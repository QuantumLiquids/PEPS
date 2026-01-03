// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 2025-12-15
 *
 * Description: Exact-sum smoke test for TFIM PBC energy using TRG contraction.
 */

#include "gtest/gtest.h"

#include <cmath>
#include <vector>

#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square_pbc.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/trg/trg_contractor.h"

using namespace qlten;
using namespace qlpeps;

namespace {

template <typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> BuildProductPlusState2x2PBC() {
  using TensorT = QLTensor<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;

  SplitIndexTPS<TenElemT, QNT> sitps(2, 2, 2, BoundaryCondition::Periodic);

  // All virtual indices are dimension-1, but directions must follow TN convention:
  // (l IN, d OUT, r OUT, u IN).
  const IndexT idx_l({QNSctT(QNT(), 1)}, TenIndexDirType::IN);
  const IndexT idx_d({QNSctT(QNT(), 1)}, TenIndexDirType::OUT);
  const IndexT idx_r({QNSctT(QNT(), 1)}, TenIndexDirType::OUT);
  const IndexT idx_u({QNSctT(QNT(), 1)}, TenIndexDirType::IN);

  // Local |+> state in the computational basis: (|0> + |1>)/sqrt(2).
  const TenElemT a = TenElemT(1.0 / std::sqrt(2.0));

  for (size_t r = 0; r < 2; ++r) {
    for (size_t c = 0; c < 2; ++c) {
      std::vector<TensorT> comps(2);
      for (size_t s = 0; s < 2; ++s) {
        TensorT T({idx_l, idx_d, idx_r, idx_u});
        T.Fill(QNT(), TenElemT(0));
        T({0, 0, 0, 0}) = a; // identical for s=0 and s=1
        comps[s] = std::move(T);
      }
      sitps({r, c}) = std::move(comps);
    }
  }
  return sitps;
}

std::vector<Configuration> GenerateAllConfigs2x2Binary() {
  std::vector<Configuration> out;
  out.reserve(16);
  for (size_t mask = 0; mask < 16; ++mask) {
    std::vector<size_t> v(4);
    v[0] = (mask >> 0) & 1;
    v[1] = (mask >> 1) & 1;
    v[2] = (mask >> 2) & 1;
    v[3] = (mask >> 3) & 1;
    out.push_back(Vec2Config(v, /*Lx=*/2, /*Ly=*/2));
  }
  return out;
}

}  // namespace

TEST(TFIMPBC_TRG_ExactSum, ProductPlusStateEnergy) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;

  const double h = 5.0;
  TransverseFieldIsingSquarePBC model(h);

  const auto sitps = BuildProductPlusState2x2PBC<TenElemT, QNT>();
  const auto all_configs = GenerateAllConfigs2x2Binary();

  // TRG truncation params (irrelevant for 2x2 exact contraction, but required by component API).
  const TRGTruncateParams<RealT> trunc_para(/*d_min=*/1, /*d_max=*/4, /*trunc_error=*/0.0);

  auto [energy, gradient, err] =
      ExactSumEnergyEvaluatorMPI<TransverseFieldIsingSquarePBC, TenElemT, QNT, TRGContractor>(
          sitps, all_configs, trunc_para, model, /*Ly=*/2, /*Lx=*/2, MPI_COMM_WORLD, /*rank=*/0, /*mpi_size=*/1);

  (void)gradient;
  EXPECT_NEAR(std::real(energy), -4.0 * h, 1e-12);
  EXPECT_NEAR(err, 0.0, 0.0);
}



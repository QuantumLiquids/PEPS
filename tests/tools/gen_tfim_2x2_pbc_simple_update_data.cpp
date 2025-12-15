// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 2025-12-15
 *
 * Description: Generate 2x2 PBC transverse-field Ising SplitIndexTPS data by simple update.
 *
 * This is a developer tool (not a unit test). Run it manually:
 *   ./gen_tfim_2x2_pbc_simple_update_data
 *
 * It dumps:
 * - tests/test_data/transverse_ising_tps_pbc_double_from_simple_update
 * - tests/test_data/transverse_ising_tps_pbc_complex_from_simple_update
 */

#include <cmath>
#include <complex>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "qlten/qlten.h"
#include "qlpeps/api/conversions.h"
#include "qlpeps/qlpeps.h"

using namespace qlten;
using namespace qlpeps;

namespace {

using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = qlten::QLTEN_Double;
using CTenElemT = qlten::QLTEN_Complex;
using Tensor = qlten::QLTensor<TenElemT, QNT>;
using CTensor = qlten::QLTensor<CTenElemT, QNT>;
using IndexT = qlten::Index<QNT>;
using QNSctT = qlten::QNSector<QNT>;

Tensor MakeZZBondHam(const IndexT& pb_in, const IndexT& pb_out, double J) {
  // H_bond = -J * sigma^z ⊗ sigma^z, sigma^z eigenvalues: +1 (state 0), -1 (state 1).
  Tensor ham({pb_in, pb_out, pb_in, pb_out});
  ham.Fill(QNT(), TenElemT(0));
  ham({0, 0, 0, 0}) = TenElemT(-J);
  ham({0, 0, 1, 1}) = TenElemT(+J);
  ham({1, 1, 0, 0}) = TenElemT(+J);
  ham({1, 1, 1, 1}) = TenElemT(-J);
  return ham;
}

Tensor MakeXOnsiteHam(const IndexT& pb_in, const IndexT& pb_out, double h) {
  // H_site = -h * sigma^x in sigma^z basis.
  Tensor ham({pb_in, pb_out});
  ham.Fill(QNT(), TenElemT(0));
  ham({0, 1}) = TenElemT(-h);
  ham({1, 0}) = TenElemT(-h);
  return ham;
}

SplitIndexTPS<CTenElemT, QNT> MakeComplexFromDouble(
    const SplitIndexTPS<TenElemT, QNT>& sitps_d,
    const std::complex<double> phase) {
  SplitIndexTPS<CTenElemT, QNT> sitps_c(sitps_d.rows(), sitps_d.cols(), sitps_d.PhysicalDim(), sitps_d.GetBoundaryCondition());
  for (size_t r = 0; r < sitps_d.rows(); ++r) {
    for (size_t c = 0; c < sitps_d.cols(); ++c) {
      const auto& comps_d = sitps_d({r, c});
      std::vector<CTensor> comps_c;
      comps_c.reserve(comps_d.size());
      for (const auto& t : comps_d) {
        comps_c.emplace_back(ToComplex(t) * phase);
      }
      sitps_c({r, c}) = std::move(comps_c);
    }
  }
  return sitps_c;
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  const size_t Ly = 2;
  const size_t Lx = 2;
  const double J = 1.0;
  const double h = 1.0;

  // Physical index: dim=2, trivial symmetry.
  const IndexT pb_out({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  const IndexT pb_in = InverseIndex(pb_out);

  // Hamiltonian tensors
  const Tensor ham_nn = MakeZZBondHam(pb_in, pb_out, J);
  const Tensor ham_onsite = MakeXOnsiteHam(pb_in, pb_out, h);

  // Initial PEPS (PBC)
  SquareLatticePEPS<TenElemT, QNT> peps0(pb_out, Ly, Lx, BoundaryCondition::Periodic);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx, 0));  // |0> ferromagnetic
  peps0.Initial(activates);

  // Simple update schedule: keep it small but stable.
  // SimpleUpdatePara(steps, tau, iters, Dmax, trunc_err)
  SimpleUpdatePara update_para(/*steps=*/200, /*tau=*/0.1, /*iter=*/1, /*Dmax=*/4, /*trunc_err=*/1e-12);
  SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT> exe(update_para, peps0, ham_nn, ham_onsite);
  exe.Execute();

  // Convert to SplitIndexTPS and dump.
  auto sitps_d = qlpeps::ToSplitIndexTPS<TenElemT, QNT>(exe.GetPEPS());

  // Use TEST_SOURCE_DIR so the output always goes to the source tree, independent of cwd.
  const std::filesystem::path tests_src_dir = std::filesystem::path(TEST_SOURCE_DIR);
  const std::filesystem::path out_d = tests_src_dir / "test_data/transverse_ising_tps_pbc_double_from_simple_update";
  const std::filesystem::path out_c = tests_src_dir / "test_data/transverse_ising_tps_pbc_complex_from_simple_update";
  std::filesystem::create_directories(out_d);
  std::filesystem::create_directories(out_c);

  sitps_d.Dump(out_d.string());
  std::cout << "Dumped double SplitIndexTPS to: " << out_d << "\n";

  // Complex dataset: deterministic global phase exp(i).
  const std::complex<double> phase = std::exp(std::complex<double>(0.0, 1.0));
  auto sitps_c = MakeComplexFromDouble(sitps_d, phase);
  sitps_c.Dump(out_c.string());
  std::cout << "Dumped complex SplitIndexTPS to: " << out_c << "\n";

  return 0;
}



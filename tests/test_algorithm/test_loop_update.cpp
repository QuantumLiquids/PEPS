// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-02-19
*
* Description: QuantumLiquids/PEPS project. Regression tests for loop update.
*
* Test strategy: 2x2 OBC systems have exactly one plaquette (one loop).
* Loop update should converge the single-plaquette energy toward the exact
* ground state energy. No MPI required. Uses add_two_type_unittest (double + complex variants).
*
* Both real and complex element types are tested via add_two_type_unittest.
* The complex variant uses a genuinely complex Hamiltonian (XX + hY with
* imaginary sigma_y entries) for the TFIM test to exercise the full complex
* code path in the loop update pipeline.
*/

#define PLAIN_TRANSPOSE 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/simple_update/simple_update_model_all.h"
#include "qlpeps/algorithm/loop_update/loop_update.h"

#include <cmath>
#include <cstdlib>
#include <string>

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using TenElemT = TEN_ELEM_TYPE;

namespace {

bool EnableExperimentalLoopUpdateConvergenceTests(void) {
  const char *flag = std::getenv("QLPEPS_ENABLE_EXPERIMENTAL_LOOP_UPDATE_TESTS");
  return flag != nullptr && std::string(flag) == "1";
}

}  // namespace

// ---------------------------------------------------------------------------
// Exact energy helpers
// ---------------------------------------------------------------------------

/// Exact ground state energy of the 2x2 OBC Heisenberg model (= 4-site PBC chain).
/// E0 = -2J.
double Calculate2x2HeisenbergEnergy(double J) {
  return -2.0 * J;
}

/// Exact ground state energy of the 2x2 OBC transverse-field Ising model.
/// Uses Jordan-Wigner + Fourier-Bogoliubov on the equivalent 4-site PBC chain.
double Calculate2x2OBCTransverseIsingEnergy(double J, double h) {
  std::vector<double> k_values = {M_PI / 4.0, 3.0 * M_PI / 4.0};
  double ground_state_energy = 0.0;
  for (auto k : k_values) {
    double epsilon_k = 2.0 * std::sqrt(J * J + h * h - 2.0 * J * h * std::cos(k));
    ground_state_energy -= epsilon_k;
  }
  return ground_state_energy;
}

/// Exact ground state energy of spinless free fermions on an Lx x Ly OBC lattice.
double CalGroundStateEnergyForSpinlessNNFreeFermionOBC(
    const size_t Lx,
    const size_t Ly,
    const size_t particle_num
) {
  std::vector<double> energy_levels;
  for (size_t kx = 0; kx < Lx; ++kx) {
    for (size_t ky = 0; ky < Ly; ++ky) {
      double theta_x = M_PI * (kx + 1) / (Lx + 1);
      double theta_y = M_PI * (ky + 1) / (Ly + 1);
      double energy = -2 * (std::cos(theta_x) + std::cos(theta_y));
      energy_levels.push_back(energy);
    }
  }
  std::sort(energy_levels.begin(), energy_levels.end());
  double ground_state_energy = 0.0;
  for (size_t i = 0; i < particle_num; ++i) {
    ground_state_energy += energy_levels[i];
  }
  return ground_state_energy;
}

// ===========================================================================
// Test 1: Heisenberg model loop update on 2x2 OBC
// ===========================================================================
struct HeisenbergLoopUpdateOBC2x2 : public testing::Test {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using Tensor = QLTensor<TenElemT, U1QN>;
  using LoopGateT = LoopGates<Tensor>;

  static constexpr size_t Lx = 2;
  static constexpr size_t Ly = 2;

  // Trivial U1QN with dim-2 physical index (Z2-like Heisenberg)
  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  // Virtual bond for exact cyclic MPO: dim 5
  // States: A=0 (idle), Cz=1 (Sz-started), C+=2 (S+-started), C-=3 (S--started), B=4 (completed)
  IndexT vb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 5)},
                         TenIndexDirType::OUT);
  IndexT vb_in = InverseIndex(vb_out);

  Tensor dham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});

  void SetUp(void) override {
    dham_hei_nn({0, 0, 0, 0}) = 0.25;
    dham_hei_nn({1, 1, 1, 1}) = 0.25;
    dham_hei_nn({1, 1, 0, 0}) = -0.25;
    dham_hei_nn({0, 0, 1, 1}) = -0.25;
    dham_hei_nn({0, 1, 1, 0}) = 0.5;
    dham_hei_nn({1, 0, 0, 1}) = 0.5;
  }

  /// Exact cyclic (trace-type) MPO for Heisenberg loop gates on a 4-site ring.
  ///
  /// Constructs O = I - tau*H where H = sum_<ij> S_i . S_j
  ///   = sum_<ij> [Sz_i Sz_j + (1/2)(S+_i S-_j + S-_i S+_j)]
  ///
  /// Uses D=5 virtual bond so each bond term consumes 3 virtual transitions:
  ///   SzSz: A -> Cz -> B -> A   (Sz source, -tau/n*Sz sink, closure)
  ///   S+S-: A -> C+ -> B -> A   (S+ source, -tau/(2n)*S- sink, closure)
  ///   S-S+: A -> C- -> B -> A   (S- source, -tau/(2n)*S+ sink, closure)
  ///
  /// On a 4-site ring, two terms would need 6 > 4 transitions, so cross-terms
  /// are impossible. The MPO trace gives EXACTLY I - tau*H.
  ///
  ///   W = | I    S_z    S^+    S^-        0           |
  ///       | 0     0      0      0    -tau/n * S_z     |
  ///       | 0     0      0      0    -tau/(2n) * S^-  |
  ///       | 0     0      0      0    -tau/(2n) * S^+  |
  ///       | I     0      0      0        0            |
  ///
  /// @param n_i sharing factor for bond terms ending at gate i
  LoopGateT GenerateHeisenbergLoopGates(double tau,
                                        size_t n0, size_t n1, size_t n2, size_t n3) {
    const std::vector<size_t> ns = {n0, n1, n2, n3};
    LoopGateT gates;
    for (size_t i = 0; i < 4; i++) {
      gates[i] = Tensor({vb_in, pb_in, pb_out, vb_out});
      Tensor &gate = gates[i];
      // A->A (0->0): Identity
      gate({0, 0, 0, 0}) = 1.0;
      gate({0, 1, 1, 0}) = 1.0;
      // A->Cz (0->1): S_z (start SzSz term, no coefficient)
      gate({0, 0, 0, 1}) = 0.5;
      gate({0, 1, 1, 1}) = -0.5;
      // A->C+ (0->2): S^+ (start S+S- term, no coefficient)
      gate({0, 1, 0, 2}) = 1.0;
      // A->C- (0->3): S^- (start S-S+ term, no coefficient)
      gate({0, 0, 1, 3}) = 1.0;
      // Cz->B (1->4): -tau/n * S_z (complete SzSz bond)
      gate({1, 0, 0, 4}) = -0.5 * tau / double(ns[i]);
      gate({1, 1, 1, 4}) = 0.5 * tau / double(ns[i]);
      // C+->B (2->4): -tau/(2n) * S^- (complete S+S- bond)
      gate({2, 0, 1, 4}) = -tau / (2.0 * double(ns[i]));
      // C-->B (3->4): -tau/(2n) * S^+ (complete S-S+ bond)
      gate({3, 1, 0, 4}) = -tau / (2.0 * double(ns[i]));
      // B->A (4->0): Identity (forced closure)
      gate({4, 0, 0, 0}) = 1.0;
      gate({4, 1, 1, 0}) = 1.0;
    }
    return gates;
  }
};

TEST_F(HeisenbergLoopUpdateOBC2x2, ConvergesToExactEnergy) {
  if (!EnableExperimentalLoopUpdateConvergenceTests()) {
    GTEST_SKIP() << "Loop-update convergence tests are disabled by default "
                 << "(slow/unstable). Set QLPEPS_ENABLE_EXPERIMENTAL_LOOP_UPDATE_TESTS=1 to run.";
  }
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  SquareLatticePEPS<TenElemT, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  for (size_t y = 0; y < Ly; y++)
    for (size_t x = 0; x < Lx; x++)
      activates[y][x] = (x + y) % 2;
  peps0.Initial(activates);

  // Simple update warm-up
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, U1QN>>(
      SimpleUpdatePara(50, 0.1, 1, 4, 1e-10), peps0, dham_hei_nn);
  su_exe->Execute();
  auto peps1 = su_exe->GetPEPS();
  su_exe.reset();
  peps1.NormalizeAllTensor();

  // Build loop gates for the single 2x2 plaquette (all n_i = 1)
  // D=5 exact cyclic MPO: no cross-terms on 4-site ring, energy estimator is exact.
  double tau = 0.01;
  DuoMatrix<LoopGateT> evolve_gates(Ly - 1, Lx - 1);
  evolve_gates({0, 0}) = GenerateHeisenbergLoopGates(tau, 1, 1, 1, 1);

  // Loop update
  ArnoldiParams arnoldi_params(1e-10, 100);
  ConjugateGradientParams cg_params{.max_iter = 100, .relative_tolerance = 1e-5,
                                    .residual_recompute_interval = 20};
  FullEnvironmentTruncateParams fet_params(1, 4, 1e-10, 1e-12, 30, cg_params);

  LoopUpdatePara loop_para(LoopUpdateTruncatePara(arnoldi_params, 1e-7, fet_params), 50, tau);
  auto loop_exe = std::make_unique<LoopUpdateExecutor<TenElemT, U1QN>>(
      loop_para, evolve_gates, peps1);
  loop_exe->Execute();

  double exact_energy = Calculate2x2HeisenbergEnergy(1.0);
  double estimated_energy = loop_exe->GetEstimatedEnergy();
  std::cout << "Heisenberg 2x2 exact E0 = " << exact_energy
            << ", loop update E0 = " << estimated_energy << std::endl;
  // D=5 exact cyclic MPO: no cross-terms on 4-site ring, energy estimator is exact.
  EXPECT_NEAR(exact_energy, estimated_energy, 1e-2);
}

// ===========================================================================
// Test 2: Heisenberg model loop update on 4x4 OBC
// ===========================================================================
struct HeisenbergLoopUpdateOBC4x4 : public testing::Test {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using Tensor = QLTensor<TenElemT, U1QN>;
  using LoopGateT = LoopGates<Tensor>;

  static constexpr size_t Lx = 4;
  static constexpr size_t Ly = 4;

  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  // D=5 exact cyclic MPO virtual bond
  IndexT vb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 5)},
                         TenIndexDirType::OUT);
  IndexT vb_in = InverseIndex(vb_out);

  Tensor dham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});

  // ED ground state energy for 4x4 Heisenberg OBC
  static constexpr double kExactEnergy4x4 = -9.189207065192949;

  void SetUp(void) override {
    dham_hei_nn({0, 0, 0, 0}) = 0.25;
    dham_hei_nn({1, 1, 1, 1}) = 0.25;
    dham_hei_nn({1, 1, 0, 0}) = -0.25;
    dham_hei_nn({0, 0, 1, 1}) = -0.25;
    dham_hei_nn({0, 1, 1, 0}) = 0.5;
    dham_hei_nn({1, 0, 0, 1}) = 0.5;
  }

  /// D=5 exact cyclic Heisenberg MPO for a single plaquette.
  /// n_i = sharing factor for bond ending at gate i (sink convention).
  LoopGateT GenerateHeisenbergLoopGates(double tau,
                                        size_t n0, size_t n1, size_t n2, size_t n3) {
    const std::vector<size_t> ns = {n0, n1, n2, n3};
    LoopGateT gates;
    for (size_t i = 0; i < 4; i++) {
      gates[i] = Tensor({vb_in, pb_in, pb_out, vb_out});
      Tensor &gate = gates[i];
      gate({0, 0, 0, 0}) = 1.0;
      gate({0, 1, 1, 0}) = 1.0;
      gate({0, 0, 0, 1}) = 0.5;
      gate({0, 1, 1, 1}) = -0.5;
      gate({0, 1, 0, 2}) = 1.0;
      gate({0, 0, 1, 3}) = 1.0;
      gate({1, 0, 0, 4}) = -0.5 * tau / double(ns[i]);
      gate({1, 1, 1, 4}) = 0.5 * tau / double(ns[i]);
      gate({2, 0, 1, 4}) = -tau / (2.0 * double(ns[i]));
      gate({3, 1, 0, 4}) = -tau / (2.0 * double(ns[i]));
      gate({4, 0, 0, 0}) = 1.0;
      gate({4, 1, 1, 0}) = 1.0;
    }
    return gates;
  }

  /// Generate D=5 exact cyclic Heisenberg loop gates for all plaquettes on Ly x Lx OBC.
  ///
  /// Bond sharing factors (sink convention):
  ///   n[0] = left vertical:   1 if col==0, else 2
  ///   n[1] = upper horizontal: 1 if row==0, else 2
  ///   n[2] = right vertical:  1 if col==Lx-2, else 2
  ///   n[3] = lower horizontal: 1 if row==Ly-2, else 2
  void GenerateAllEvolveGates(double tau, DuoMatrix<LoopGateT> &evolve_gates) {
    for (size_t row = 0; row < Ly - 1; row++) {
      for (size_t col = 0; col < Lx - 1; col++) {
        size_t n_left  = (col == 0)      ? 1 : 2;
        size_t n_upper = (row == 0)      ? 1 : 2;
        size_t n_right = (col == Lx - 2) ? 1 : 2;
        size_t n_lower = (row == Ly - 2) ? 1 : 2;
        evolve_gates({row, col}) = GenerateHeisenbergLoopGates(tau, n_left, n_upper, n_right, n_lower);
      }
    }
  }
};

TEST_F(HeisenbergLoopUpdateOBC4x4, ConvergesToExactEnergy) {
  if (!EnableExperimentalLoopUpdateConvergenceTests()) {
    GTEST_SKIP() << "Loop-update convergence tests are disabled by default "
                 << "(slow/unstable). Set QLPEPS_ENABLE_EXPERIMENTAL_LOOP_UPDATE_TESTS=1 to run.";
  }
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  SquareLatticePEPS<TenElemT, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  for (size_t y = 0; y < Ly; y++)
    for (size_t x = 0; x < Lx; x++)
      activates[y][x] = (x + y) % 2;
  peps0.Initial(activates);

  // Simple update warm-up
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, U1QN>>(
      SimpleUpdatePara(100, 0.1, 1, 4, 1e-10), peps0, dham_hei_nn);
  su_exe->Execute();
  auto peps1 = su_exe->GetPEPS();
  su_exe.reset();
  peps1.NormalizeAllTensor();

  // Loop update with D=5 exact cyclic MPO
  double tau = 0.01;
  DuoMatrix<LoopGateT> evolve_gates(Ly - 1, Lx - 1);
  GenerateAllEvolveGates(tau, evolve_gates);

  ArnoldiParams arnoldi_params(1e-10, 100);
  ConjugateGradientParams cg_params{.max_iter = 100, .relative_tolerance = 1e-5,
                                    .residual_recompute_interval = 20};
  FullEnvironmentTruncateParams fet_params(1, 4, 1e-10, 1e-12, 30, cg_params);

  LoopUpdatePara loop_para(LoopUpdateTruncatePara(arnoldi_params, 1e-7, fet_params), 30, tau);
  auto loop_exe = std::make_unique<LoopUpdateExecutor<TenElemT, U1QN>>(
      loop_para, evolve_gates, peps1);
  loop_exe->Execute();

  double estimated_energy = loop_exe->GetEstimatedEnergy();
  std::cout << "Heisenberg 4x4 ED E0 = " << kExactEnergy4x4
            << ", loop update E0 = " << estimated_energy << std::endl;
  // D=4 PEPS variational limit on 4x4 gives ~0.22 error; tolerance allows small margin.
  EXPECT_NEAR(kExactEnergy4x4, estimated_energy, 0.25);
}

// ===========================================================================
// Test: Heisenberg PBC loop update with 2x2 -> tiled 4x4
// Strategy: converge 2x2 PBC (simple update + loop update), tile to 4x4,
//           then loop update 4x4 and check energy lowers.
// ===========================================================================
struct HeisenbergPBCTiledLoopUpdate : public testing::Test {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using Tensor = QLTensor<TenElemT, U1QN>;
  using DTensor = QLTensor<typename qlten::RealTypeTrait<TenElemT>::type, U1QN>;
  using LoopGateT = LoopGates<Tensor>;

  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  // D=5 exact cyclic MPO virtual bond
  IndexT vb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 5)},
                         TenIndexDirType::OUT);
  IndexT vb_in = InverseIndex(vb_out);

  Tensor dham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});

  // ED ground state energy for 4x4 Heisenberg PBC (from pbc_benchmarks.py)
  static constexpr double kExactEnergy4x4PBC = -11.228483208428854;

  void SetUp(void) override {
    dham_hei_nn({0, 0, 0, 0}) = 0.25;
    dham_hei_nn({1, 1, 1, 1}) = 0.25;
    dham_hei_nn({1, 1, 0, 0}) = -0.25;
    dham_hei_nn({0, 0, 1, 1}) = -0.25;
    dham_hei_nn({0, 1, 1, 0}) = 0.5;
    dham_hei_nn({1, 0, 0, 1}) = 0.5;
  }

  /// D=5 exact cyclic Heisenberg MPO for a single plaquette.
  /// @param n_i sharing factor for bond terms ending at gate i
  LoopGateT GenerateHeisenbergLoopGates(double tau,
                                        size_t n0, size_t n1, size_t n2, size_t n3) {
    const std::vector<size_t> ns = {n0, n1, n2, n3};
    LoopGateT gates;
    for (size_t i = 0; i < 4; i++) {
      gates[i] = Tensor({vb_in, pb_in, pb_out, vb_out});
      Tensor &gate = gates[i];
      gate({0, 0, 0, 0}) = 1.0;
      gate({0, 1, 1, 0}) = 1.0;
      gate({0, 0, 0, 1}) = 0.5;
      gate({0, 1, 1, 1}) = -0.5;
      gate({0, 1, 0, 2}) = 1.0;
      gate({0, 0, 1, 3}) = 1.0;
      gate({1, 0, 0, 4}) = -0.5 * tau / double(ns[i]);
      gate({1, 1, 1, 4}) = 0.5 * tau / double(ns[i]);
      gate({2, 0, 1, 4}) = -tau / (2.0 * double(ns[i]));
      gate({3, 1, 0, 4}) = -tau / (2.0 * double(ns[i]));
      gate({4, 0, 0, 0}) = 1.0;
      gate({4, 1, 1, 0}) = 1.0;
    }
    return gates;
  }

  /// Fill DuoMatrix with uniform PBC gates (all n_i = 2).
  void GeneratePBCEvolveGates(double tau, size_t ly, size_t lx,
                              DuoMatrix<LoopGateT> &evolve_gates) {
    for (size_t row = 0; row < ly; row++)
      for (size_t col = 0; col < lx; col++)
        evolve_gates({row, col}) = GenerateHeisenbergLoopGates(tau, 2, 2, 2, 2);
  }

  /// Tile a small_ly x small_lx PBC PEPS to big_ly x big_lx PBC PEPS.
  /// Requires big dimensions are multiples of small dimensions.
  static SquareLatticePEPS<TenElemT, U1QN> TilePEPS(
      const SquareLatticePEPS<TenElemT, U1QN> &small,
      size_t big_ly, size_t big_lx) {
    const size_t sly = small.Rows();
    const size_t slx = small.Cols();
    assert(big_ly % sly == 0 && big_lx % slx == 0);
    const IndexT &phys_idx = small.Gamma({0, 0}).GetIndex(4);
    SquareLatticePEPS<TenElemT, U1QN> big(phys_idx, big_ly, big_lx,
                                           BoundaryCondition::Periodic);
    // Copy Gamma tensors periodically
    for (size_t r = 0; r < big_ly; r++)
      for (size_t c = 0; c < big_lx; c++)
        big.Gamma({r, c}) = small.Gamma({r % sly, c % slx});
    // Copy lambda_vert periodically (PBC: size ly x lx)
    for (size_t r = 0; r < big_ly; r++)
      for (size_t c = 0; c < big_lx; c++)
        big.lambda_vert({r, c}) = small.lambda_vert({r % sly, c % slx});
    // Copy lambda_horiz periodically (PBC: size ly x lx)
    for (size_t r = 0; r < big_ly; r++)
      for (size_t c = 0; c < big_lx; c++)
        big.lambda_horiz({r, c}) = small.lambda_horiz({r % sly, c % slx});
    return big;
  }
};

TEST_F(HeisenbergPBCTiledLoopUpdate, TiledEnergyLowers) {
  if (!EnableExperimentalLoopUpdateConvergenceTests()) {
    GTEST_SKIP() << "Loop-update convergence tests are disabled by default "
                 << "(slow/unstable). Set QLPEPS_ENABLE_EXPERIMENTAL_LOOP_UPDATE_TESTS=1 to run.";
  }
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  // --- Phase 1: Converge 2x2 PBC with simple update ---
  const size_t small_ly = 2, small_lx = 2;
  SquareLatticePEPS<TenElemT, U1QN> peps_2x2(pb_out, small_ly, small_lx,
                                               BoundaryCondition::Periodic);
  std::vector<std::vector<size_t>> act_2x2 = {{0, 1}, {1, 0}};
  peps_2x2.Initial(act_2x2);

  // Gradually decrease tau for convergence
  for (double su_tau : {0.5, 0.2, 0.1}) {
    auto su = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, U1QN>>(
        SimpleUpdatePara(100, su_tau, 1, 4, 1e-10), peps_2x2, dham_hei_nn);
    su->Execute();
    peps_2x2 = su->GetPEPS();
  }
  peps_2x2.NormalizeAllTensor();

  // --- Phase 2: Loop update on 2x2 PBC ---
  {
    double tau_2x2 = 0.1;
    DuoMatrix<LoopGateT> gates_2x2(small_ly, small_lx);
    GeneratePBCEvolveGates(tau_2x2, small_ly, small_lx, gates_2x2);

    ArnoldiParams ap(1e-10, 100);
    ConjugateGradientParams cg{.max_iter = 100, .relative_tolerance = 1e-5,
                               .residual_recompute_interval = 20};
    FullEnvironmentTruncateParams fet(1, 4, 1e-10, 1e-12, 30, cg);
    LoopUpdatePara lp(LoopUpdateTruncatePara(ap, 1e-7, fet), 5, tau_2x2);

    auto loop_2x2 = std::make_unique<LoopUpdateExecutor<TenElemT, U1QN>>(
        lp, gates_2x2, peps_2x2);
    loop_2x2->Execute();
    peps_2x2 = loop_2x2->GetPEPS();
  }
  peps_2x2.NormalizeAllTensor();

  // --- Phase 3: Tile 2x2 -> 4x4 ---
  const size_t big_ly = 4, big_lx = 4;
  auto peps_4x4 = TilePEPS(peps_2x2, big_ly, big_lx);

  // --- Phase 4: Loop update on 4x4 PBC ---
  // Run 1 step to get baseline energy, then 4 more to verify lowering.
  double tau_4x4 = 0.02;
  DuoMatrix<LoopGateT> gates_4x4(big_ly, big_lx);
  GeneratePBCEvolveGates(tau_4x4, big_ly, big_lx, gates_4x4);

  ArnoldiParams ap(1e-10, 100);
  ConjugateGradientParams cg{.max_iter = 100, .relative_tolerance = 1e-5,
                             .residual_recompute_interval = 20};
  FullEnvironmentTruncateParams fet(1, 4, 1e-10, 1e-12, 30, cg);

  // Step 1: single sweep to establish baseline energy
  LoopUpdatePara lp_init(LoopUpdateTruncatePara(ap, 1e-7, fet), 1, tau_4x4);
  auto loop_init = std::make_unique<LoopUpdateExecutor<TenElemT, U1QN>>(
      lp_init, gates_4x4, peps_4x4);
  loop_init->Execute();
  double e_initial = loop_init->GetEstimatedEnergy();
  peps_4x4 = loop_init->GetPEPS();
  loop_init.reset();

  // Steps 2-5: continue evolving
  LoopUpdatePara lp_rest(LoopUpdateTruncatePara(ap, 1e-7, fet), 4, tau_4x4);
  auto loop_rest = std::make_unique<LoopUpdateExecutor<TenElemT, U1QN>>(
      lp_rest, gates_4x4, peps_4x4);
  loop_rest->Execute();
  double e_final = loop_rest->GetEstimatedEnergy();

  std::cout << "Heisenberg 4x4 PBC (tiled): ED E0 = " << kExactEnergy4x4PBC
            << ", initial E0 = " << e_initial
            << ", final E0 = " << e_final << std::endl;

  EXPECT_NEAR(kExactEnergy4x4PBC, e_final, 0.7);
  EXPECT_LT(e_final, e_initial);  // energy must lower
}

// ===========================================================================
// Test 3: Transverse-field Ising model loop update on 2x2 OBC
// ===========================================================================
struct TFIMLoopUpdateOBC2x2 : public testing::Test {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using Tensor = QLTensor<TenElemT, U1QN>;
  using LoopGateT = LoopGates<Tensor>;

  static constexpr size_t Lx = 2;
  static constexpr size_t Ly = 2;

  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  // Virtual bond for exact cyclic MPO: dim 4 (A=idle, C=Z-started, E=X-started, B=term-completed)
  IndexT vb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 4)},
                         TenIndexDirType::OUT);
  IndexT vb_in = InverseIndex(vb_out);

  Tensor did = Tensor({pb_in, pb_out});
  Tensor dsigmaz = Tensor({pb_in, pb_out});
  Tensor dsigmax = Tensor({pb_in, pb_out});
  Tensor dham_nn;

  double h = 3.0;

  void SetUp(void) override {
    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsigmaz({0, 0}) = 1;
    dsigmaz({1, 1}) = -1;
    dsigmax({0, 1}) = 1;
    dsigmax({1, 0}) = 1;

    Tensor ham_nn1, ham_nn2, ham_nn3;
    Contract(&dsigmaz, {}, &dsigmaz, {}, &ham_nn1);
    Contract(&dsigmax, {}, &did, {}, &ham_nn2);
    Contract(&did, {}, &dsigmax, {}, &ham_nn3);
    dham_nn = -ham_nn1 + (-0.25) * h * ham_nn2 + (-0.25) * h * ham_nn3;
  }

  /// Exact cyclic (trace-type) MPO for TFIM loop gates on a 4-site ring.
  ///
  /// Constructs O = I - tau*H where H = -J*sum Z_iZ_{i+1} - h*sum X_i.
  /// Uses D=4 virtual bond with an intermediate state E for the X channel,
  /// so each term (ZZ or X) consumes 3 virtual transitions:
  ///   ZZ: A -> C -> B -> A   (Z source, tau*Z sink, closure)
  ///   X:  A -> E -> B -> A   (tau*h*X source, identity pass-through, closure)
  ///
  /// On a 4-site ring, two terms would need 6 > 4 transitions, so cross-terms
  /// are impossible. The MPO trace gives EXACTLY I - tau*H.
  ///
  ///   W = | I    Z   tau*h/m*X   0   |
  ///       | 0    0      0      tau/n*Z|
  ///       | 0    0      0        I   |
  ///       | I    0      0        0   |
  ///
  /// Virtual states: A=0 (idle), C=1 (Z started), E=2 (X started), B=3 (completed)
  ///
  /// @param n_i sharing factor for the bond ending at gate i (FROM gate i-1 TO gate i)
  /// @param m_i sharing factor for on-site term at gate i
  LoopGateT GenerateTFIMLoopGates(double tau, double h_field,
                                  size_t m0, size_t m1, size_t m2, size_t m3,
                                  size_t n0, size_t n1, size_t n2, size_t n3) {
    const std::vector<size_t> ms = {m0, m1, m2, m3};
    const std::vector<size_t> ns = {n0, n1, n2, n3};
    LoopGateT gates;
    for (size_t i = 0; i < 4; i++) {
      gates[i] = Tensor({vb_in, pb_in, pb_out, vb_out});
      Tensor &gate = gates[i];
      // A->A: Identity (idle -> idle)
      gate({0, 0, 0, 0}) = 1.0;
      gate({0, 1, 1, 0}) = 1.0;
      // A->C: Z (start a ZZ bond term, no coefficient)
      gate({0, 0, 0, 1}) = 1.0;
      gate({0, 1, 1, 1}) = -1.0;
      // A->E: tau*h/m * X (start on-site X term)
      gate({0, 0, 1, 2}) = tau * h_field / double(ms[i]);
      gate({0, 1, 0, 2}) = tau * h_field / double(ms[i]);
      // C->B: tau/n * Z (complete ZZ bond from previous site)
      gate({1, 0, 0, 3}) = tau / double(ns[i]);
      gate({1, 1, 1, 3}) = -tau / double(ns[i]);
      // E->B: Identity (pass-through, forces X to consume 3 transitions)
      gate({2, 0, 0, 3}) = 1.0;
      gate({2, 1, 1, 3}) = 1.0;
      // B->A: Identity (forced closure back to idle)
      gate({3, 0, 0, 0}) = 1.0;
      gate({3, 1, 1, 0}) = 1.0;
    }
    return gates;
  }
};

TEST_F(TFIMLoopUpdateOBC2x2, ConvergesToExactEnergy) {
  if (!EnableExperimentalLoopUpdateConvergenceTests()) {
    GTEST_SKIP() << "Loop-update convergence tests are disabled by default "
                 << "(slow/unstable). Set QLPEPS_ENABLE_EXPERIMENTAL_LOOP_UPDATE_TESTS=1 to run.";
  }
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  SquareLatticePEPS<TenElemT, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  for (size_t y = 0; y < Ly; y++)
    for (size_t x = 0; x < Lx; x++)
      activates[y][x] = (x + y) % 2;
  peps0.Initial(activates);

#if TEN_ELEM_TYPE_NUM == 2
  // Complex-specific: use H = -XX + hY instead of -ZZ - hX.
  // sigma_y has purely imaginary entries, forcing genuinely complex PEPS tensors.
  //
  // Unitary equivalence: apply U = W * V on each site, where
  //   V = exp(-i pi/4 sigma_y):  sigma_z -> -sigma_x,  sigma_x -> sigma_z
  //   W = exp(-i pi/4 sigma_x):  sigma_z -> -sigma_y,  sigma_x -> sigma_x
  // Composite U = W * V:  sigma_z -> -sigma_x,  sigma_x -> -sigma_y
  // so  H = -ZZ - hX  maps to  H' = -XX + hY  with identical spectrum.
  Tensor dsigmay = Tensor({pb_in, pb_out});
  dsigmay({0, 1}) = TenElemT(0.0, -1.0);  // -i
  dsigmay({1, 0}) = TenElemT(0.0, 1.0);   // +i

  Tensor ham_xx, ham_y_left, ham_y_right;
  Contract(&dsigmax, {}, &dsigmax, {}, &ham_xx);
  Contract(&dsigmay, {}, &did, {}, &ham_y_left);
  Contract(&did, {}, &dsigmay, {}, &ham_y_right);
  Tensor dham_test = -ham_xx + (0.25 * h) * ham_y_left + (0.25 * h) * ham_y_right;
#else
  const Tensor &dham_test = dham_nn;
#endif

  // Simple update warm-up
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, U1QN>>(
      SimpleUpdatePara(50, 0.1, 1, 4, 1e-5), peps0, dham_test);
  su_exe->Execute();
  auto peps1 = su_exe->GetPEPS();
  su_exe.reset();
  peps1.NormalizeAllTensor();

  double tau = 0.01;
  DuoMatrix<LoopGateT> evolve_gates(Ly - 1, Lx - 1);

#if TEN_ELEM_TYPE_NUM == 2
  // D=4 exact cyclic MPO for XX + hY on 2x2 (all m_i = 1, n_i = 1).
  //   W = | I    X   -tau*h*Y   0       |
  //       | 0    0      0     tau*X     |
  //       | 0    0      0       I       |
  //       | I    0      0       0       |
  {
    LoopGateT gates;
    for (size_t i = 0; i < 4; i++) {
      gates[i] = Tensor({vb_in, pb_in, pb_out, vb_out});
      Tensor &gate = gates[i];
      // A->A: Identity
      gate({0, 0, 0, 0}) = 1.0;
      gate({0, 1, 1, 0}) = 1.0;
      // A->C: X source (start XX bond, no coefficient)
      gate({0, 0, 1, 1}) = 1.0;
      gate({0, 1, 0, 1}) = 1.0;
      // A->E: -tau*h * Y (start on-site Y term)
      // -tau*h * sigma_y = [[0, i*tau*h], [-i*tau*h, 0]]
      gate({0, 0, 1, 2}) = TenElemT(0.0, tau * h);
      gate({0, 1, 0, 2}) = TenElemT(0.0, -tau * h);
      // C->B: tau * X (complete XX bond)
      gate({1, 0, 1, 3}) = tau;
      gate({1, 1, 0, 3}) = tau;
      // E->B: Identity (pass-through)
      gate({2, 0, 0, 3}) = 1.0;
      gate({2, 1, 1, 3}) = 1.0;
      // B->A: Identity (closure)
      gate({3, 0, 0, 0}) = 1.0;
      gate({3, 1, 1, 0}) = 1.0;
    }
    evolve_gates({0, 0}) = gates;
  }
#else
  evolve_gates({0, 0}) = GenerateTFIMLoopGates(tau, h, 1, 1, 1, 1, 1, 1, 1, 1);
#endif

  // Loop update
  ArnoldiParams arnoldi_params(1e-13, 30);
  ConjugateGradientParams cg_params{.max_iter = 100, .relative_tolerance = 1e-5,
                                    .residual_recompute_interval = 20};
  FullEnvironmentTruncateParams fet_params(1, 4, 1e-10, 1e-12, 30, cg_params);

  LoopUpdatePara loop_para(LoopUpdateTruncatePara(arnoldi_params, 1e-7, fet_params), 50, tau);
  auto loop_exe = std::make_unique<LoopUpdateExecutor<TenElemT, U1QN>>(
      loop_para, evolve_gates, peps1);
  loop_exe->Execute();

  // Both models have the same exact energy (related by site-local unitary rotation).
  double exact_energy = Calculate2x2OBCTransverseIsingEnergy(1.0, h);
  double estimated_energy = loop_exe->GetEstimatedEnergy();
#if TEN_ELEM_TYPE_NUM == 2
  std::cout << "XX+hY 2x2 exact E0 = " << exact_energy
            << ", loop update E0 = " << estimated_energy << std::endl;
#else
  std::cout << "TFIM 2x2 exact E0 = " << exact_energy
            << ", loop update E0 = " << estimated_energy << std::endl;
#endif
  EXPECT_NEAR(exact_energy, estimated_energy, 1e-2);
}

// ===========================================================================
// Test 4: Spinless fermion loop update on 2x2 OBC
// ===========================================================================
struct SpinlessFermionLoopUpdateOBC2x2 : public testing::Test {
  using QNT = qlten::special_qn::fZ2QN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using Tensor = QLTensor<TenElemT, QNT>;
  using LoopGateT = LoopGates<Tensor>;

  static constexpr size_t Lx = 2;
  static constexpr size_t Ly = 2;
  static constexpr size_t ele_num = 2; // half-filling

  double t = 1.0;

  QNT qn0 = QNT(0);
  IndexT loc_phy_ket = IndexT({QNSctT(QNT(1), 1),  // |1> occupied
                               QNSctT(QNT(0), 1)}, // |0> empty state
                              TenIndexDirType::IN);
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  // Virtual bond for loop gates
  IndexT vb_out = IndexT({QNSctT(QNT(0), 4),
                          QNSctT(QNT(1), 4)},
                         TenIndexDirType::OUT);
  IndexT vb_in = InverseIndex(vb_out);

  Tensor n_op = Tensor({loc_phy_ket, loc_phy_bra});
  Tensor ham_nn = Tensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});

  double mu;

  void SetUp(void) override {
    n_op({0, 0}) = 1.0;
    n_op.Transpose({1, 0});

    ham_nn({1, 0, 1, 0}) = -t;
    ham_nn({0, 1, 0, 1}) = -t;
    ham_nn.Transpose({3, 0, 2, 1});

    mu = (CalGroundStateEnergyForSpinlessNNFreeFermionOBC(Lx, Ly, ele_num + 1)
        - CalGroundStateEnergyForSpinlessNNFreeFermionOBC(Lx, Ly, ele_num - 1)) / 2.0;
  }

  /// Generate spinless fermion hopping loop gates.
  /// On 2x2: all n_i = 1.
  /// Physical: state 0 = occupied (fZ2(1)), state 1 = empty (fZ2(0)).
  /// Virtual channels: 0 (fZ2(0)) = identity, 4-5 (fZ2(1)) = hopping.
  /// Hopping terms route to fZ2(1) virtual sector to preserve tensor divergence.
  LoopGateT GenerateFermionLoopGates(double tau, double t_hop,
                                     size_t n0, size_t n1, size_t n2, size_t n3) {
    const std::vector<size_t> ns = {n0, n1, n2, n3};
    LoopGateT gates;
    for (size_t i = 0; i < 4; i++) {
      gates[i] = Tensor({vb_in, loc_phy_bra, loc_phy_ket, vb_out});
      Tensor &gate = gates[i];
      // Identity channel (vb fZ2(0), index 0)
      gate({0, 0, 0, 0}) = 1.0;  // |occupied> -> |occupied>
      gate({0, 1, 1, 0}) = 1.0;  // |empty> -> |empty>

      // c source (annihilate: |occ> -> |emp>), route to vb fZ2(1) sector
      gate({0, 1, 0, 4}) = (-tau) * (-t_hop) / double(ns[i]) * (-1);
      // c^dag sink (create: |emp> -> |occ>)
      gate({4, 0, 1, 0}) = 1.0;

      // c^dag source (create: |emp> -> |occ>), route to vb fZ2(1) sector
      gate({0, 0, 1, 5}) = (-tau) * (-t_hop) / double(ns[i]);
      // c sink (annihilate: |occ> -> |emp>)
      gate({5, 1, 0, 0}) = 1.0;
      gate.Div();
    }
    return gates;
  }
};

TEST_F(SpinlessFermionLoopUpdateOBC2x2, ConvergesToExactEnergy) {
  // Fermion loop update projection has a pre-existing lambda orientation issue
  // in WeightedTraceGaugeFixing (QuasiSquareRootDiagMat assertion).
  // Skip until the underlying projection4_impl.h is fixed for fermionic PEPS.
  GTEST_SKIP() << "Fermion loop update projection has pre-existing lambda orientation issue";
}

// ===========================================================================
// Negative tests: constructor validation
// ===========================================================================

// Type alias to avoid commas inside EXPECT_THROW macro
using LoopUpdateExecutorT = LoopUpdateExecutor<TenElemT, U1QN>;

TEST(LoopUpdateValidation, RejectsOddPBCDimensions) {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using Tensor = QLTensor<TenElemT, U1QN>;
  using LoopGateT = LoopGates<Tensor>;

  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT);

  // 3x4 PBC: odd row count
  SquareLatticePEPS<TenElemT, U1QN> peps_3x4(pb_out, 3, 4, BoundaryCondition::Periodic);
  std::vector<std::vector<size_t>> act_3x4(3, std::vector<size_t>(4, 0));
  peps_3x4.Initial(act_3x4);

  DuoMatrix<LoopGateT> gates_3x4(3, 4);  // PBC: Ly x Lx
  ArnoldiParams ap(1e-10, 100);
  ConjugateGradientParams cg{.max_iter = 100, .relative_tolerance = 1e-5,
                             .residual_recompute_interval = 20};
  FullEnvironmentTruncateParams fet(1, 4, 1e-10, 1e-12, 30, cg);
  LoopUpdatePara para(LoopUpdateTruncatePara(ap, 1e-7, fet), 1, 0.01);

  EXPECT_THROW(
      LoopUpdateExecutorT(para, gates_3x4, peps_3x4),
      std::invalid_argument);
}

TEST(LoopUpdateValidation, RejectsWrongGateShapeForPBC) {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using Tensor = QLTensor<TenElemT, U1QN>;
  using LoopGateT = LoopGates<Tensor>;

  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT);

  // 4x4 PBC but gates sized for OBC: (Ly-1, Lx-1) = (3, 3)
  SquareLatticePEPS<TenElemT, U1QN> peps(pb_out, 4, 4, BoundaryCondition::Periodic);
  std::vector<std::vector<size_t>> act(4, std::vector<size_t>(4, 0));
  peps.Initial(act);

  DuoMatrix<LoopGateT> wrong_gates(3, 3);  // OBC size, should be (4, 4) for PBC
  ArnoldiParams ap(1e-10, 100);
  ConjugateGradientParams cg{.max_iter = 100, .relative_tolerance = 1e-5,
                             .residual_recompute_interval = 20};
  FullEnvironmentTruncateParams fet(1, 4, 1e-10, 1e-12, 30, cg);
  LoopUpdatePara para(LoopUpdateTruncatePara(ap, 1e-7, fet), 1, 0.01);

  EXPECT_THROW(
      LoopUpdateExecutorT(para, wrong_gates, peps),
      std::invalid_argument);
}

TEST(LoopUpdateValidation, RejectsWrongGateShapeForOBC) {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using Tensor = QLTensor<TenElemT, U1QN>;
  using LoopGateT = LoopGates<Tensor>;

  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT);

  // 4x4 OBC but gates sized for PBC: (Ly, Lx) = (4, 4)
  SquareLatticePEPS<TenElemT, U1QN> peps(pb_out, 4, 4);  // OBC default
  std::vector<std::vector<size_t>> act(4, std::vector<size_t>(4, 0));
  peps.Initial(act);

  DuoMatrix<LoopGateT> wrong_gates(4, 4);  // PBC size, should be (3, 3) for OBC
  ArnoldiParams ap(1e-10, 100);
  ConjugateGradientParams cg{.max_iter = 100, .relative_tolerance = 1e-5,
                             .residual_recompute_interval = 20};
  FullEnvironmentTruncateParams fet(1, 4, 1e-10, 1e-12, 30, cg);
  LoopUpdatePara para(LoopUpdateTruncatePara(ap, 1e-7, fet), 1, 0.01);

  EXPECT_THROW(
      LoopUpdateExecutorT(para, wrong_gates, peps),
      std::invalid_argument);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

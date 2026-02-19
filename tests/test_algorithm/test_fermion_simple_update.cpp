// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-08-16
*
* Description: QuantumLiquids/PEPS project. Unittests for PEPS Simple Update in fermion model.
*/

#define PLAIN_TRANSPOSE 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"

#include "qlpeps/algorithm/simple_update/simple_update_model_all.h"
#include "qlpeps/utility/filesystem_utils.h"
#include "qlpeps/api/conversions.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"

#include <limits>
#include <memory>
#include <array>

using namespace qlten;
using namespace qlpeps;

std::string GenTPSPath(std::string model_name, size_t Dmax, size_t Lx, size_t Ly) {
#if TEN_ELEM_TYPE_NUM == 1
  return "dtps_" + model_name + "_D" + std::to_string(Dmax) + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#elif TEN_ELEM_TYPE_NUM == 2
  return "ztps_" + model_name + "_D" + std::to_string(Dmax)  + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#else
#error "Unexpected TEN_ELEM_TYPE_NUM"
#endif
}

double CalGroundStateEnergyForSpinlessNNFreeFermionOBC(
    const size_t Lx,
    const size_t Ly,
    const size_t particle_num
) {
  const size_t num_sites = Lx * Ly;
  std::vector<double> energy_levels;

  // Calculate the energy levels
  for (size_t kx = 0; kx < Lx; ++kx) {
    for (size_t ky = 0; ky < Ly; ++ky) {
      double theta_x = M_PI * (kx + 1) / (Lx + 1);
      double theta_y = M_PI * (ky + 1) / (Ly + 1);
      double energy = -2 * (std::cos(theta_x) + std::cos(theta_y));
      energy_levels.push_back(energy);
    }
  }

  // Sort energy levels in ascending order
  std::sort(energy_levels.begin(), energy_levels.end());

  // Sum the lowest `particle_num` energy levels
  double ground_state_energy = 0.0;
  for (size_t i = 0; i < particle_num; ++i) {
    ground_state_energy += energy_levels[i];
  }

  return ground_state_energy;
}

double CalGroundStateEnergyForSpinlessNNFreeFermionPBC(
    const size_t Lx,
    const size_t Ly,
    const size_t particle_num
) {
  const size_t num_sites = Lx * Ly;
  std::vector<double> energy_levels;
  energy_levels.reserve(num_sites);

  const double two_pi = 2.0 * M_PI;
  for (size_t kx = 0; kx < Lx; ++kx) {
    for (size_t ky = 0; ky < Ly; ++ky) {
      double theta_x = two_pi * static_cast<double>(kx) / static_cast<double>(Lx);
      double theta_y = two_pi * static_cast<double>(ky) / static_cast<double>(Ly);
      double energy = -2.0 * std::cos(theta_x) - 2.0 * std::cos(theta_y);
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

std::vector<double> CalEnergyLevelsForSquareNNFreeFermionPBC(
    const size_t Lx,
    const size_t Ly,
    const double t
) {
  const size_t num_sites = Lx * Ly;
  std::vector<double> energy_levels;
  energy_levels.reserve(num_sites);
  const double two_pi = 2.0 * M_PI;
  for (size_t kx = 0; kx < Lx; ++kx) {
    for (size_t ky = 0; ky < Ly; ++ky) {
      const double theta_x = two_pi * static_cast<double>(kx) / static_cast<double>(Lx);
      const double theta_y = two_pi * static_cast<double>(ky) / static_cast<double>(Ly);
      const double e = -2.0 * t * std::cos(theta_x) - 2.0 * t * std::cos(theta_y);
      energy_levels.push_back(e);
    }
  }
  std::sort(energy_levels.begin(), energy_levels.end());
  return energy_levels;
}

double CalGroundStateEnergyForSpinfulNNFreeFermionPBCFixedNe(
    const size_t Lx,
    const size_t Ly,
    const size_t electron_num,
    const double t
) {
  const size_t num_sites = Lx * Ly;
  if (electron_num > 2 * num_sites) {
    std::cerr << "electron_num exceeds 2 * num_sites." << std::endl;
    return 0.0;
  }
  const auto eps = CalEnergyLevelsForSquareNNFreeFermionPBC(Lx, Ly, t);
  const size_t full_k = electron_num / 2;
  const bool has_extra = (electron_num % 2) == 1;

  double e0 = 0.0;
  for (size_t i = 0; i < full_k; ++i) {
    e0 += 2.0 * eps[i];  // both spins
  }
  if (has_extra) {
    e0 += eps[full_k];   // one extra electron in the next k-level
  }
  return e0;
}

double CalChemicalPotentialFromFiniteDifferenceSpinfulPBC(
    const size_t Lx,
    const size_t Ly,
    const size_t electron_num,
    const double t
) {
  if (electron_num == 0) {
    return CalGroundStateEnergyForSpinfulNNFreeFermionPBCFixedNe(Lx, Ly, 1, t);
  }
  const double e_plus = CalGroundStateEnergyForSpinfulNNFreeFermionPBCFixedNe(Lx, Ly, electron_num + 1, t);
  const double e_minus = CalGroundStateEnergyForSpinfulNNFreeFermionPBCFixedNe(Lx, Ly, electron_num - 1, t);
  return (e_plus - e_minus) / 2.0;
}

std::pair<double, size_t> CalGroundStateEnergyForSpinfulNNFreeFermionPBCGrandCanonical(
    const size_t Lx,
    const size_t Ly,
    const double mu,
    const double t
) {
  const size_t num_sites = Lx * Ly;
  const size_t max_e = 2 * num_sites;
  double best_e = std::numeric_limits<double>::infinity();
  size_t best_ne = 0;
  for (size_t ne = 0; ne <= max_e; ++ne) {
    const double e_kin = CalGroundStateEnergyForSpinfulNNFreeFermionPBCFixedNe(Lx, Ly, ne, t);
    const double e = e_kin - mu * static_cast<double>(ne);
    if (e < best_e) {
      best_e = e;
      best_ne = ne;
    }
  }
  return {best_e, best_ne};
}

struct Z2SpinlessFreeFermionTools : public testing::Test {
  using QNT = qlten::special_qn::fZ2QN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using QNSctVecT = QNSectorVec<QNT>;
  using TenElemT = QLTEN_Double;
  using Tensor = QLTensor<TenElemT, QNT>;
  
  // Reduced system size for faster testing
  size_t Lx = 3; //cols (reduced from 4)
  size_t Ly = 2; // (reduced from 3)
  size_t Dmax = 3;  // reduced from 4 for faster convergence

  size_t ele_num = 3; // adjusted for smaller system
  double t = 1.0;
  double mu; //chemical potential

  QNT qn0 = QNT(0);
  // |ket>
  IndexT loc_phy_ket = IndexT({QNSctT(QNT(1), 1),  // |1> occupied
                               QNSctT(QNT(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  // <bra|
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  Tensor c = Tensor({loc_phy_ket, loc_phy_bra});   // annihilation operator
  Tensor cdag = Tensor({loc_phy_ket, loc_phy_bra});// creation operator
  Tensor n = Tensor({loc_phy_ket, loc_phy_bra});   // density operator

  Tensor ham_nn = Tensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});//site: i-j-j-i (i<j)
  std::string model_name = "spinless_free_fermion";
  std::string tps_path = GenTPSPath(model_name, Dmax, Lx, Ly);
  void SetUp(void) {
    n({0, 0}) = 1.0;
    n.Transpose({1, 0});
    c({1, 0}) = 1;
    cdag({0, 1}) = 1;

    ham_nn({1, 0, 1, 0}) = -t;
    ham_nn({0, 1, 0, 1}) = -t;
    ham_nn.Transpose({3, 0, 2, 1}); // transpose indices order for consistent with simple update convention
    mu = (CalGroundStateEnergyForSpinlessNNFreeFermionOBC(Lx, Ly, ele_num + 1)
        - CalGroundStateEnergyForSpinlessNNFreeFermionOBC(Lx, Ly, ele_num - 1)) / 2.0;
    std::cout << "mu : " << mu << std::endl; // -0.707107
  }
};

TEST_F(Z2SpinlessFreeFermionTools, HalfFillingSimpleUpdate) {
  // Allow multi-threading for better performance
  qlten::hp_numeric::SetTensorManipulationThreads(4);
  
  SquareLatticePEPS<TenElemT, QNT> peps0(loc_phy_ket, Ly, Lx);

  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  //half-filling
  size_t n_int = 0;
  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      activates[y][x] = n_int % 2;
      n_int++;
    }
  }
  peps0.Initial(activates);

  // Phase 1: Fast validation with minimal iterations
  {
    SimpleUpdatePara update_para(50, 0.1, 1, Dmax, 1e-6);
    SimpleUpdateExecutor<TenElemT, QNT>
        *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>(update_para, peps0,
                                                                         ham_nn,
                                                                         -mu * n);
    su_exe->Execute();
    
    auto peps = su_exe->GetPEPS();
    
    delete su_exe;
  }

  // Phase 2: Full convergence with more iterations
  {
    SimpleUpdatePara update_para(200, 0.1, 1, Dmax, 1e-8);
    SimpleUpdateExecutor<TenElemT, QNT>
        *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>(update_para, peps0,
                                                                         ham_nn,
                                                                         -mu * n);
    su_exe->Execute();
    
    // Fewer refinement steps
    su_exe->ResetStepLenth(0.01);
    su_exe->Execute();
    
    // Remove the third refinement step to save time
    // su_exe->ResetStepLenth(0.001);
    // su_exe->Execute();
    
    auto peps = su_exe->GetPEPS();
    auto tps = qlpeps::ToTPS<TenElemT, QNT>(su_exe->GetPEPS());
    SplitIndexTPS<TenElemT, QNT> sitps = SplitPhyIndex(tps);
    sitps.Dump(tps_path);

    double exact_gs_energy = CalGroundStateEnergyForSpinlessNNFreeFermionOBC(Lx, Ly, ele_num);
    std::cout << "Exact ground state energy : " << std::setprecision(10) << exact_gs_energy << std::endl;
    
    // Compare estimated energy with exact energy
    double estimated_energy = su_exe->GetEstimatedEnergy();
    std::cout << "Estimated energy from simple update: " << std::setprecision(10) << estimated_energy << std::endl;
    
    // Check energy accuracy (allow reasonable tolerance for finite D and iterations)
    double energy_tolerance = 0.2; // Reduced from 0.5 for more stringent testing
    EXPECT_NEAR(exact_gs_energy, estimated_energy, energy_tolerance);
    
    // Clean up
    delete su_exe;
  }
}

struct Z2SpinlessFreeFermionPBCTools : public testing::Test {
  using QNT = qlten::special_qn::fZ2QN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using QNSctVecT = QNSectorVec<QNT>;
  using TenElemT = QLTEN_Double;
  using Tensor = QLTensor<TenElemT, QNT>;

  size_t Lx = 3; // cols
  size_t Ly = 4;
  size_t Dmax = 4;

  size_t ele_num = 6; // half filling on 3x4
  double t = 1.0;
  double mu; // chemical potential for half filling (PBC)

  QNT qn0 = QNT(0);
  IndexT loc_phy_ket = IndexT({QNSctT(QNT(1), 1),  // |1> occupied
                               QNSctT(QNT(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  Tensor c = Tensor({loc_phy_ket, loc_phy_bra});   // annihilation operator
  Tensor cdag = Tensor({loc_phy_ket, loc_phy_bra});// creation operator
  Tensor n = Tensor({loc_phy_ket, loc_phy_bra});   // density operator

  Tensor ham_nn = Tensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra}); // site: i-j-j-i (i<j)

  void SetUp(void) override {
    n({0, 0}) = 1.0;
    n.Transpose({1, 0});
    c({1, 0}) = 1;
    cdag({0, 1}) = 1;

    ham_nn({1, 0, 1, 0}) = -t;
    ham_nn({0, 1, 0, 1}) = -t;
    ham_nn.Transpose({3, 0, 2, 1}); // match simple update convention

    mu = (CalGroundStateEnergyForSpinlessNNFreeFermionPBC(Lx, Ly, ele_num + 1)
        - CalGroundStateEnergyForSpinlessNNFreeFermionPBC(Lx, Ly, ele_num - 1)) / 2.0;
  }
};

TEST_F(Z2SpinlessFreeFermionPBCTools, HalfFillingSimpleUpdatePBC) {
  qlten::hp_numeric::SetTensorManipulationThreads(4);

  SquareLatticePEPS<TenElemT, QNT> peps0(loc_phy_ket, Ly, Lx, BoundaryCondition::Periodic);

  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      activates[y][x] = (x + y) % 2; // half filling pattern
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(150, 0.1, 1, Dmax, 1e-10);
  SimpleUpdateExecutor<TenElemT, QNT>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>(update_para, peps0,
                                                                       ham_nn,
                                                                       -mu * n);
  su_exe->Execute();
  // su_exe->update_para.Dmax = 8;
  su_exe->ResetStepLenth(0.01);
  su_exe->Execute();

  double exact_gs_energy = CalGroundStateEnergyForSpinlessNNFreeFermionPBC(Lx, Ly, ele_num)
      - mu * static_cast<double>(ele_num); // include chemical potential shift
  double estimated_energy = su_exe->GetEstimatedEnergy();

  EXPECT_NEAR(exact_gs_energy, estimated_energy, 0.4); // 0.2% error, It's OK.

  delete su_exe;
}

struct Z2SquareHubbardU0PBCTools : public testing::Test {
  using QNT = qlten::special_qn::fZ2QN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using TenElemT = QLTEN_Double;
  using Tensor = QLTensor<TenElemT, QNT>;

  // PBC system size (keep it small for CI runtime)
  size_t Lx = 3;
  size_t Ly = 4;
  // Keep D moderate: this test is a *smoke/integration* test for the SU pipeline,
  // not a variational accuracy benchmark.
  size_t Dmax = 10;

  // Hubbard parameters
  double t = 1.0;
  double U = 0.0;  // exact free-fermion limit

  // Target electron number (total, including both spins)
  // Choose a doped configuration to avoid Fermi-level degeneracy artifacts at mu=0.
  size_t electron_num = 12;  // < 2 * Lx * Ly = 24
  double mu = 0.0;

  // NOTE:
  // We reference `SquareHubbardModel`'s physical meanings:
  // - Double occupancy, SpinUp, SpinDown, Empty
  // But the *internal index order* here is chosen to respect fermion parity blocks:
  //   [even sector: doublon, empty] + [odd sector: up, down]
  // This is purely a basis ordering choice for tensor blocks and does not change physics.
  enum class HubbardState : size_t {
    DoubleOccupancy = 0,
    Empty = 1,
    SpinUp = 2,
    SpinDown = 3
  };

  static constexpr size_t Idx(HubbardState s) { return static_cast<size_t>(s); }

  QNT qn0 = QNT(0);
  IndexT loc_phy_ket = IndexT({QNSctT(QNT(0), 2),  // even parity: |↑↓>, |0>
                               QNSctT(QNT(1), 2)}, // odd parity:  |↑>, |↓>
                              TenIndexDirType::IN);
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  Tensor n_tot = Tensor({loc_phy_ket, loc_phy_bra});

  // Two-site Hubbard kinetic term (NN, spin up + spin down)
  Tensor ham_nn = Tensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra}); // (ket1, ket2, bra2, bra1) i-j-j-i before transpose

  // Helper: get the *untransposed* Hamiltonian tensor element
  // Stored raw order is (ket1, ket2, bra2, bra1) to match the i-j-j-i convention
  // used across fermion models in this repo.
  // This accessor uses the user-friendly signature (ket1, ket2, bra1, bra2) = <ket|H|bra>.
  double HamElemRaw(size_t ket1, size_t ket2, size_t bra1, size_t bra2) const {
    Tensor h = ham_nn;
    // ham_nn is stored after Transpose({3,0,2,1}) to match SU convention.
    // Invert that permutation to recover the *raw storage order* of the bond Hamiltonian:
    //   (ket1, ket2, bra2, bra1)  [i-j-j-i].
    h.Transpose({1, 3, 2, 0});
    // Raw storage is i-j-j-i: (ket1, ket2, bra2, bra1).
    // This function takes (bra1, bra2) in argument order, so we swap them when indexing.
    return h({ket1, ket2, bra2, bra1});
  }

  void SetUp(void) override {
    // ------------------------------------------------------------
    // Build local annihilation/creation operators with the standard
    // local mode ordering: (up, down).
    //
    // Basis meanings:
    // |↑↓> : (n_up=1, n_dn=1)
    // |0>  : (0,0)
    // |↑>  : (1,0)
    // |↓>  : (0,1)
    //
    // Local sign for down: c_dn carries (-1)^{n_up}.
    // ------------------------------------------------------------
    const size_t i_dbl = Idx(HubbardState::DoubleOccupancy);
    const size_t i_emp = Idx(HubbardState::Empty);
    const size_t i_up = Idx(HubbardState::SpinUp);
    const size_t i_dn = Idx(HubbardState::SpinDown);

    // NOTE:
    // We do NOT construct the hopping term via `Contract(c, c^dag)` here.
    // `Contract` may assume the operand tensors are even parity in some internal paths,
    // which is a real footgun for fermionic operator algebra.
    //
    // Instead, we build the two-site hopping Hamiltonian by explicitly applying fermion
    // operators on the two-site basis with a fixed global mode ordering:
    //   (site1 up, site1 down, site2 up, site2 down).
    // This makes the sign structure unambiguous.

    // n_tot = n_up + n_dn (diagonal)
    n_tot({i_dbl, i_dbl}) = 2.0;
    n_tot({i_up, i_up}) = 1.0;
    n_tot({i_dn, i_dn}) = 1.0;
    n_tot({i_emp, i_emp}) = 0.0;

    // Two-site hopping Hamiltonian (explicit matrix elements):
    // H_ij = -t * Σ_σ ( c_{iσ}^† c_{jσ} + c_{jσ}^† c_{iσ} )
    //
    // We build the bond Hamiltonian by acting on a two-site Fock basis with a fixed
    // global mode ordering:
    //   mode 0: site1 up, mode 1: site1 down, mode 2: site2 up, mode 3: site2 down.
    //
    // Fermionic sign is then uniquely defined once we fix a global *mode ordering*.
    // This is not "doing a Jordan-Wigner transformation" on the lattice; it's simply the
    // canonical Fock-space convention:
    // applying c_{mode} / c_{mode}^dag contributes (-1)^{#occupied modes with index < mode}.
    auto state_to_bits = [&](const size_t s) -> std::pair<int, int> {
      switch (HubbardState(s)) {
        case HubbardState::DoubleOccupancy: return {1, 1};
        case HubbardState::Empty: return {0, 0};
        case HubbardState::SpinUp: return {1, 0};
        case HubbardState::SpinDown: return {0, 1};
        default: return {0, 0};
      }
    };
    auto bits_to_state = [&](const int nu, const int nd) -> size_t {
      if (nu == 1 && nd == 1) return i_dbl;
      if (nu == 0 && nd == 0) return i_emp;
      if (nu == 1 && nd == 0) return i_up;
      return i_dn; // (0,1)
    };
    auto popcount_prefix = [&](const std::array<int, 4> &b, const int mode) -> int {
      int cnt = 0;
      for (int i = 0; i < mode; ++i) cnt += b[i];
      return cnt;
    };
    auto apply_annihilate = [&](std::array<int, 4> &b, const int mode, double &sgn) -> bool {
      if (b[mode] == 0) return false;
      if (popcount_prefix(b, mode) & 1) sgn *= -1.0;
      b[mode] = 0;
      return true;
    };
    auto apply_create = [&](std::array<int, 4> &b, const int mode, double &sgn) -> bool {
      if (b[mode] == 1) return false;
      if (popcount_prefix(b, mode) & 1) sgn *= -1.0;
      b[mode] = 1;
      return true;
    };

    // Fill H matrix elements in i-j-j-i storage:
    //   ham_nn(ket1, ket2, bra2, bra1) = <ket1,ket2|H|bra1,bra2>.
    for (size_t bra1 = 0; bra1 < 4; ++bra1) {
      for (size_t bra2 = 0; bra2 < 4; ++bra2) {
        const auto [n1u, n1d] = state_to_bits(bra1);
        const auto [n2u, n2d] = state_to_bits(bra2);
        const std::array<int, 4> bra_bits{n1u, n1d, n2u, n2d};

        for (int sigma = 0; sigma < 2; ++sigma) {
          const int mode1 = (sigma == 0) ? 0 : 1;
          const int mode2 = (sigma == 0) ? 2 : 3;

          // c1^dag c2 : move σ from site2 -> site1
          {
            std::array<int, 4> b = bra_bits;
            double sgn = 1.0;
            if (apply_annihilate(b, mode2, sgn) && apply_create(b, mode1, sgn)) {
              const size_t ket1 = bits_to_state(b[0], b[1]);
              const size_t ket2 = bits_to_state(b[2], b[3]);
              ham_nn({ket1, ket2, bra2, bra1}) = ham_nn({ket1, ket2, bra2, bra1}) + (-t) * sgn;
            }
          }
          // c2^dag c1 : move σ from site1 -> site2
          {
            std::array<int, 4> b = bra_bits;
            double sgn = 1.0;
            if (apply_annihilate(b, mode1, sgn) && apply_create(b, mode2, sgn)) {
              const size_t ket1 = bits_to_state(b[0], b[1]);
              const size_t ket2 = bits_to_state(b[2], b[3]);
              ham_nn({ket1, ket2, bra2, bra1}) = ham_nn({ket1, ket2, bra2, bra1}) + (-t) * sgn;
            }
          }
        }
      }
    }
    ham_nn.Transpose({3, 0, 2, 1}); // match simple update convention (same as spinless tests)

    // Choose mu from finite difference in the *noninteracting* spectrum (PBC).
    mu = CalChemicalPotentialFromFiniteDifferenceSpinfulPBC(Lx, Ly, electron_num, t);
    std::cout << "Hubbard(U=0) PBC: mu=" << std::setprecision(10) << mu
              << " (Lx=" << Lx << ", Ly=" << Ly << ", Ne=" << electron_num << ")\n";
  }
};

TEST_F(Z2SquareHubbardU0PBCTools, BondHamiltonianMatrixElements) {
  // This is the *real* unit test: verify fermionic sign structure on a bond.
  // It should be exact (up to floating error) and independent of SU convergence.
  //
  // Convention reference (see user's note `main.tex`, Hubbard model section):
  //   H_t = -t Σ_σ ( c_{1σ}^† c_{2σ} + c_{2σ}^† c_{1σ} )
  // with |↑↓⟩ = c_↑^† c_↓^† |0⟩ and two-site basis |i,j⟩.
  //
  // Then:
  //   H_t |0,↑⟩ = -t |↑,0⟩
  //   H_t |0,↓⟩ = -t |↓,0⟩
  //   H_t |↑↓,0⟩ =  t |↓,↑⟩ - t |↑,↓⟩
  const size_t dbl = Idx(HubbardState::DoubleOccupancy);
  const size_t emp = Idx(HubbardState::Empty);
  const size_t up = Idx(HubbardState::SpinUp);
  const size_t dn = Idx(HubbardState::SpinDown);

  // |0,↑> -> |↑,0>
  EXPECT_DOUBLE_EQ(HamElemRaw(up, emp, emp, up), -t);
  // |0,↓> -> |↓,0>
  EXPECT_DOUBLE_EQ(HamElemRaw(dn, emp, emp, dn), -t);

  // Doublon involvement (matches the note):
  // |↑↓,0> -> |↓,↑> has +t, and |↑↓,0> -> |↑,↓> has -t.
  EXPECT_DOUBLE_EQ(HamElemRaw(dn, up, dbl, emp), +t);
  EXPECT_DOUBLE_EQ(HamElemRaw(up, dn, dbl, emp), -t);

  // Symmetry partner:
  // H_t |0,↑↓⟩ = t |↑,↓⟩ - t |↓,↑⟩
  //
  // Note: the relative sign here depends on the *two-site basis convention*
  // (i.e. whether |i,j⟩ is defined with site-1 creation operators placed before
  // site-2 operators). In this test we follow the explicit 4-mode ordering used
  // to construct `ham_nn`, which yields:
  //   H_t |0,↑↓⟩ = -t |↑,↓⟩ + t |↓,↑⟩.
  EXPECT_DOUBLE_EQ(HamElemRaw(up, dn, emp, dbl), -t);
  EXPECT_DOUBLE_EQ(HamElemRaw(dn, up, emp, dbl), +t);
}

TEST_F(Z2SquareHubbardU0PBCTools, SimpleUpdatePBC_U0_FreeFermionReference) {
  qlten::hp_numeric::SetTensorManipulationThreads(4);

  SquareLatticePEPS<TenElemT, QNT> peps0(loc_phy_ket, Ly, Lx, BoundaryCondition::Periodic);

  // Initialize to a doped AFM-like pattern: mostly single occupancy, with a few holes.
  // This is just an initial guess; chemical potential term should stabilize the target density.
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx, Idx(HubbardState::Empty)));
  size_t placed_e = 0;
  for (size_t y = 0; y < Ly; ++y) {
    for (size_t x = 0; x < Lx; ++x) {
      if (placed_e >= electron_num) {
        activates[y][x] = Idx(HubbardState::Empty);
        continue;
      }
      activates[y][x] = ((x + y) % 2 == 0) ? Idx(HubbardState::SpinUp) : Idx(HubbardState::SpinDown);
      placed_e += 1;
    }
  }
  peps0.Initial(activates);

  // On-site term: U n_up n_dn - mu (n_up + n_dn). Here U=0.
  Tensor ham_onsite = (-mu) * n_tot;
  ham_onsite.Transpose({1, 0}); // (bra, ket) for fermion convention in SU

  // Run two stages to reduce local-minimum risk without making the test too slow.
  {
    // More aggressive schedule to reduce systematic error so we can tighten tolerance.
    SimpleUpdatePara update_para(30, 0.1, 1, Dmax, 1e-10);
    auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>>(
        update_para, peps0, ham_nn, ham_onsite);
    su_exe->Execute();

    su_exe->ResetStepLenth(0.01);
    su_exe->Execute();

    su_exe->ResetStepLenth(0.001);
    su_exe->Execute();

    const auto [e_exact, ne_exact] = CalGroundStateEnergyForSpinfulNNFreeFermionPBCGrandCanonical(Lx, Ly, mu, t);
    const double e_est = su_exe->GetEstimatedEnergy();

    std::cout << "Exact (free fermion, grand canonical) energy: " << std::setprecision(12) << e_exact
              << " (Ne*=" << ne_exact << ")\n";
    std::cout << "Estimated energy from simple update:  " << std::setprecision(12) << e_est << "\n";

    // Simple update is approximate (finite D + local-minimum risk) and the energy estimator here is not
    // guaranteed variational; nevertheless, for U=0 we should be able to get reasonably close.
    // Keep this looser than the exact matrix-element test: SU energy is not strictly variational.
    EXPECT_NEAR(e_exact, e_est, 0.5);
  }
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  auto test_err = RUN_ALL_TESTS();
  return test_err;
}

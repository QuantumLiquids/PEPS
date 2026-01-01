// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: QuantumLiquids/PEPS project. Unittests for Simple Update in PEPS optimization.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/api/conversions.h"
#include "qlpeps/algorithm/simple_update/simple_update_model_all.h"
#include <memory>

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using qlten::special_qn::Z2QN;

using TenElemT = TEN_ELEM_TYPE;

namespace {

template<typename T>
struct MatrixElement {
  std::vector<size_t> coors;
  T elem;
};

std::vector<MatrixElement<double>> GenerateTriElements(
    const std::vector<MatrixElement<double>> &base_elements, size_t i) {

  std::vector<MatrixElement<double>> tri_elements;

  for (const auto &elem : base_elements) {
    // Create new matrix element for each `ham_hei_tri_terms[i]`
    for (size_t j = 0; j < 2; j++) {
      MatrixElement<double> new_elem = elem;

      // Insert {j, j} at position 2*i in the coordinates
      new_elem.coors.insert(new_elem.coors.begin() + 2 * i, {j, j});
      tri_elements.push_back(new_elem);
    }
  }

  return tri_elements;
}

std::string GenPEPSPath(std::string model_name, size_t Dmax) {
#if TEN_ELEM_TYPE_NUM == 1
  return "dpeps_" + model_name + "_D" + std::to_string(Dmax);
#elif TEN_ELEM_TYPE_NUM == 2
  return "zpeps_" + model_name + "_D" + std::to_string(Dmax);
#else
#error "Unexpected TEN_ELEM_TYPE_NUM"
#endif
}

std::string GenTPSPath(std::string model_name, size_t Dmax) {
#if TEN_ELEM_TYPE_NUM == 1
  return "dtps_" + model_name + "_D" + std::to_string(Dmax);
#elif TEN_ELEM_TYPE_NUM == 2
  return "ztps_" + model_name + "_D" + std::to_string(Dmax);
#else
#error "Unexpected TEN_ELEM_TYPE_NUM"
#endif
}

struct UpdateStage {
  size_t D;
  double trunc_err;
  double step_len;
  size_t steps = 0;
};

template<typename ExecutorT, typename QNT>
void RunStages(ExecutorT& exe, const std::string& model_name, const std::vector<UpdateStage>& stages, bool dump_last = false) {
  for (size_t i = 0; i < stages.size(); ++i) {
    const auto& stage = stages[i];
    exe.update_para.Dmax = stage.D;
    exe.update_para.Trunc_err = stage.trunc_err;
    if (stage.steps > 0) exe.update_para.steps = stage.steps;
    exe.ResetStepLenth(stage.step_len);
    exe.Execute();

    // Dump logic (mimicking original test behavior for intermediate dumps)
    // Original test dumped at D=4 and D=8.
    // Here we just dump if D >= 4 for debugging/verification, or as requested.
    if (stage.D >= 4) {
      bool is_last = (i == stages.size() - 1);
      // Keep tensors alive for ToTPS; dump without releasing memory.
      exe.DumpResult(GenPEPSPath(model_name, stage.D), false);
      auto tps = qlpeps::ToTPS<TenElemT, QNT>(exe.GetPEPS());
      tps.Dump(GenTPSPath(model_name, stage.D));
    }
  }
}

} // namespace

// XX + Z
struct TransverseFieldIsing : public testing::Test {
  using IndexT = Index<Z2QN>;
  using QNSctT = QNSector<Z2QN>;
  using QNSctVecT = QNSectorVec<Z2QN>;
  using DTensor = QLTensor<TenElemT, Z2QN>;

  size_t Ly = 4;
  size_t Lx = 4;

  size_t h = 3.0;
  // ED ground state energy = -50.186623882777752
  Z2QN qn0 = Z2QN(0);

  IndexT pb_out = IndexT({
                           QNSctT(Z2QN(0), 1),
                           QNSctT(Z2QN(1), 1)
                         },
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);
  //sigma operators
  DTensor xx_term = DTensor({pb_in, pb_out, pb_in, pb_out});
  DTensor z_term = DTensor({pb_in, pb_out});
  using PEPST = SquareLatticePEPS<TenElemT, Z2QN>;
  // peps0 is initialized in SetUp/InitializePEPS now
  std::optional<PEPST> peps0;

  void SetUp(void) override {
    z_term({0, 0}) = 1.0 * h;
    z_term({1, 1}) = -1.0 * h;

    xx_term({0, 1, 0, 1}) = 1.0;
    xx_term({1, 0, 1, 0}) = 1.0;
    xx_term({1, 0, 0, 1}) = 1.0;
    xx_term({0, 1, 1, 0}) = 1.0;

    InitializePEPS(BoundaryCondition::Open);
  }

  void InitializePEPS(BoundaryCondition bc) {
    peps0.emplace(pb_out, Ly, Lx, bc);
    std::vector<std::vector<size_t> > activates(Ly, std::vector<size_t>(Lx));
    // AFM initial state works better in PBC
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        activates[y][x] = (x + y) % 2;
      }
    }
    peps0->Initial(activates);
  }
};

TEST_F(TransverseFieldIsing, SimpleUpdateOBC) {
  std::string model_name = "square_transverse_field_ising";
  qlten::hp_numeric::SetTensorManipulationThreads(1);
  
  // Initial stage (no dump)
  SimpleUpdatePara update_para(50, 0.1, 1, 2, 1e-5);
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, Z2QN>>(update_para,
                                                                                   *peps0,
                                                                                   xx_term,
                                                                                   z_term);
  su_exe->Execute();

  std::vector<UpdateStage> stages = {
      {4, 1e-10, 0.01},
      {6, 1e-12, 0.001},
      {8, 1e-15, 0.0001, 100}
  };
  
  RunStages<decltype(*su_exe), Z2QN>(*su_exe, model_name, stages, true);

  double E_est = su_exe->GetEstimatedEnergy();
  // check the energy
  double ex_energy = -50.186623882777752;
  EXPECT_NEAR(ex_energy, E_est, 0.2);
}

TEST_F(TransverseFieldIsing, SimpleUpdatePBC) {
  InitializePEPS(BoundaryCondition::Periodic);
  std::string model_name = "square_transverse_field_ising_pbc";
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  SimpleUpdatePara update_para(50, 0.1, 1, 2, 1e-5);
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, Z2QN>>(update_para,
                                                                                   *peps0,
                                                                                   xx_term,
                                                                                   z_term * (1.0/3.0));
  su_exe->Execute();

  std::vector<UpdateStage> stages = {
      {4, 1e-10, 0.1,100},
      {6, 1e-12, 0.001},
      {8, 1e-15, 0.0001, 100}
  };

  RunStages<decltype(*su_exe), Z2QN>(*su_exe, model_name, stages, true);
  EXPECT_TRUE(su_exe->GetPEPS().IsBondDimensionUniform());
  // ED ground state energy = -51.44812913320619 (h=3.0, PBC, 4x4), near the critical point, difficult mode.
  // ED ground state energy = -34.01059755084629 (h=1.0, PBC, 4x4)
  double E_est = su_exe->GetEstimatedEnergy();
  double ex_energy_pbc = -34.01059755084629;
  EXPECT_NEAR(ex_energy_pbc, E_est, 0.01);
}

//spin one-half system with trivial symmetry to match exact energy reference
struct SpinOneHalfSystemSimpleUpdateTrivial : public testing::Test {
  using QNT = qlten::special_qn::TrivialRepQN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using QNSctVecT = QNSectorVec<QNT>;

  using Tensor = QLTensor<TenElemT, QNT>;

  size_t Lx = 3;
  size_t Ly = 4;

  QNT qn0 = QNT();
  IndexT pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  Tensor ham_ising_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
  Tensor ham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
  double j_nnn = -0.52; // next-neighbor interaction in square lattice
  Tensor ham_hei_nnn; // next-neighbor hamiltonian in square lattice

  double pinning_field_strength = 0.1;
  TenMatrix<Tensor> afm_pinning_field = TenMatrix<Tensor>(Ly, Lx);

  Tensor ham_hei_tri; // three-site hamiltonian in triangle lattice
  double j2 = 0.52;
  Tensor ham_hei_tri_j2; // three-site hamiltonian in j1-j2 model
  std::vector<MatrixElement<double> > ham_ising_nn_elements = {
    // Sz_i * Sz_j
    {{0, 0, 0, 0}, 0.25},
    {{1, 1, 1, 1}, 0.25},
    {{1, 1, 0, 0}, -0.25},
    {{0, 0, 1, 1}, -0.25}
  };
  std::vector<MatrixElement<double> > ham_hei_nn_elements = {
    // Sz_i * Sz_j
    {{0, 0, 0, 0}, 0.25},
    {{1, 1, 1, 1}, 0.25},
    {{1, 1, 0, 0}, -0.25},
    {{0, 0, 1, 1}, -0.25},
    // 0.5 * S^+_i * S^-_j
    {{0, 1, 1, 0}, 0.5},
    // 0.5 * S^-_i * S^+j
    {{1, 0, 0, 1}, 0.5},
  };
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
  std::optional<PEPST> peps0;

  void SetUp(void) override {
    for (const auto &element : ham_ising_nn_elements) {
      ham_ising_nn(element.coors) = element.elem;
    }

    for (const auto &element : ham_hei_nn_elements) {
      ham_hei_nn(element.coors) = element.elem;
    }
    ham_hei_nnn = j_nnn * ham_hei_nn;

    Tensor ham_hei_tri_terms[3];
    for (size_t i = 0; i < 3; i++) {
      std::vector<MatrixElement<double> > tri_elements = GenerateTriElements(ham_hei_nn_elements, i);
      ham_hei_tri_terms[i] = Tensor({pb_in, pb_out, pb_in, pb_out, pb_in, pb_out});
      for (const auto &element : tri_elements) {
        ham_hei_tri_terms[i](element.coors) = element.elem;
      }
    }
    ham_hei_tri = ham_hei_tri_terms[0] + ham_hei_tri_terms[1] + ham_hei_tri_terms[2];
    ham_hei_tri_j2 = 0.5 * ham_hei_tri_terms[0] + j2 * ham_hei_tri_terms[1] + 0.5 * ham_hei_tri_terms[2];
    // AFM pinning field
    Tensor sz({pb_in, pb_out});
    sz({0, 0}) = 0.5;
    sz({1, 1}) = -0.5;
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        int sign = ((x + y) % 2 == 0) ? -1 : 1; //compatible with the following init state
        afm_pinning_field({y, x}) = pinning_field_strength * sign * sz;
      }
    }

    InitializePEPS(BoundaryCondition::Open);
  }

  void InitializePEPS(BoundaryCondition bc) {
    peps0.emplace(pb_out, Ly, Lx, bc);
    std::vector<std::vector<size_t> > activates(Ly, std::vector<size_t>(Lx));
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        size_t sz_int = x + y;
        activates[y][x] = sz_int % 2;
      }
    }
    peps0->Initial(activates);
  }
};

//spin one-half system with U1 symmetry (relaxed accuracy)
struct SpinOneHalfSystemSimpleUpdateU1 : public testing::Test {
  using QNT = U1QN;
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using QNSctVecT = QNSectorVec<U1QN>;

  using Tensor = QLTensor<TenElemT, U1QN>;

  size_t Lx = 3;
  size_t Ly = 4;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});

  IndexT pb_out = IndexT({
                           QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                           QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)
                         },
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);

  Tensor ham_ising_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
  Tensor ham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});

  double pinning_field_strength = 0.1;
  TenMatrix<Tensor> afm_pinning_field = TenMatrix<Tensor>(Ly, Lx);

  Tensor ham_hei_tri; // three-site hamiltonian in triangle lattice
  double j2 = 0.52;
  Tensor ham_hei_tri_j2; // three-site hamiltonian in j1-j2 model
  std::vector<MatrixElement<double> > ham_ising_nn_elements = {
    // Sz_i * Sz_j
    {{0, 0, 0, 0}, 0.25},
    {{1, 1, 1, 1}, 0.25},
    {{1, 1, 0, 0}, -0.25},
    {{0, 0, 1, 1}, -0.25}
  };
  std::vector<MatrixElement<double> > ham_hei_nn_elements = {
    // Sz_i * Sz_j
    {{0, 0, 0, 0}, 0.25},
    {{1, 1, 1, 1}, 0.25},
    {{1, 1, 0, 0}, -0.25},
    {{0, 0, 1, 1}, -0.25},
    // 0.5 * S^+_i * S^-_j
    {{0, 1, 1, 0}, 0.5},
    // 0.5 * S^-_i * S^+j
    {{1, 0, 0, 1}, 0.5},
  };
  using PEPST = SquareLatticePEPS<TenElemT, U1QN>;
  std::optional<PEPST> peps0;

  void SetUp(void) override {
    for (const auto &element : ham_ising_nn_elements) {
      ham_ising_nn(element.coors) = element.elem;
    }

    for (const auto &element : ham_hei_nn_elements) {
      ham_hei_nn(element.coors) = element.elem;
    }

    Tensor ham_hei_tri_terms[3];
    for (size_t i = 0; i < 3; i++) {
      std::vector<MatrixElement<double> > tri_elements = GenerateTriElements(ham_hei_nn_elements, i);
      ham_hei_tri_terms[i] = Tensor({pb_in, pb_out, pb_in, pb_out, pb_in, pb_out});
      for (const auto &element : tri_elements) {
        ham_hei_tri_terms[i](element.coors) = element.elem;
      }
    }
    ham_hei_tri = ham_hei_tri_terms[0] + ham_hei_tri_terms[1] + ham_hei_tri_terms[2];
    ham_hei_tri_j2 = 0.5 * ham_hei_tri_terms[0] + j2 * ham_hei_tri_terms[1] + 0.5 * ham_hei_tri_terms[2];
    // AFM pinning field
    Tensor sz({pb_in, pb_out});
    sz({0, 0}) = 0.5;
    sz({1, 1}) = -0.5;
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        int sign = ((x + y) % 2 == 0) ? -1 : 1; //compatible with the following init state
        afm_pinning_field({y, x}) = pinning_field_strength * sign * sz;
      }
    }

    InitializePEPS(BoundaryCondition::Open);
  }

  void InitializePEPS(BoundaryCondition bc) {
    peps0.emplace(pb_out, Ly, Lx, bc);
    std::vector<std::vector<size_t> > activates(Ly, std::vector<size_t>(Lx));
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        size_t sz_int = x + y;
        activates[y][x] = sz_int % 2;
      }
    }
    peps0->Initial(activates);
  }
};

TEST_F(SpinOneHalfSystemSimpleUpdateTrivial, AFM_ClassicalIsingOBC) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);
  SimpleUpdatePara update_para(50, 0.01, 1, 1, 1e-5);
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>>(update_para,
                                                                                    *peps0,
                                                                                    ham_ising_nn);
  su_exe->Execute();
  double ex_energy = -0.25 * ((Lx - 1) * Ly + (Ly - 1) * Lx);
  EXPECT_NEAR(ex_energy, su_exe->GetEstimatedEnergy(), 1e-10);
}

TEST_F(SpinOneHalfSystemSimpleUpdateTrivial, AFM_ClassicalIsingPBC) {
  InitializePEPS(BoundaryCondition::Periodic);
  qlten::hp_numeric::SetTensorManipulationThreads(1);
  SimpleUpdatePara update_para(50, 0.01, 1, 1, 1e-5);
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>>(update_para,
                                                                                    *peps0,
                                                                                    ham_ising_nn);
  su_exe->Execute();
  // For Lx=3 (odd) PBC, the horizontal ring is frustrated:
  // each row has 2 satisfied bonds (-0.25) and 1 frustrated bond (+0.25) => -0.25 per row.
  // Vertical direction (Ly=4 even) is unfrustrated: 3 columns * 4 bonds/col * (-0.25) = -3.
  // Total = -0.25*4 + (-3) = -4.0
  double ex_energy = -4.0;
  EXPECT_NEAR(ex_energy, su_exe->GetEstimatedEnergy(), 1e-6);
}

TEST_F(SpinOneHalfSystemSimpleUpdateU1, SquareNNHeisenbergOBC) {
  std::string model_name = "square_nn_hei";
  
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>>(SimpleUpdatePara(50, 0.2, 1, 2, 1e-5),
                                                                                    *peps0,
                                                                                    ham_hei_nn);
  su_exe->Execute();

  std::vector<UpdateStage> stages = {
      {4, 1e-6, 0.1},
      {8, 1e-10, 0.02, 50}
  };

  RunStages<decltype(*su_exe), QNT>(*su_exe, model_name, stages, true);

  double en_exact = -6.691680193514947;
  EXPECT_NEAR(su_exe->GetEstimatedEnergy(), en_exact, 0.5);
}

TEST_F(SpinOneHalfSystemSimpleUpdateU1, SquareNNHeisenbergPBC) {
  InitializePEPS(BoundaryCondition::Periodic);
  std::string model_name = "square_nn_hei_pbc";
  
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>>(SimpleUpdatePara(50, 0.2, 1, 2, 1e-5),
                                                                                    *peps0,
                                                                                    ham_hei_nn);
  su_exe->Execute();

  std::vector<UpdateStage> stages = {
      {4, 1e-6, 0.1},
      {8, 1e-10, 0.02, 50}
  };

  RunStages<decltype(*su_exe), QNT>(*su_exe, model_name, stages, true);

  double en_exact = -7.368217694134078; // ED ground state energy (3x4 PBC)
  EXPECT_NEAR(su_exe->GetEstimatedEnergy(), en_exact, 0.5);
}

TEST_F(SpinOneHalfSystemSimpleUpdateTrivial, SquareNNHeisenbergWithAMFPinningFieldOBC) {
  std::string model_name = "square_nn_hei_pin";
  
  auto su_exe = std::make_unique<SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>>(SimpleUpdatePara(50, 0.1, 1, 2, 1e-5),
                                                                                    *peps0,
                                                                                    ham_hei_nn,
                                                                                    afm_pinning_field);
  su_exe->Execute();

  std::vector<UpdateStage> stages = {
      {4, 1e-6, 0.05},
      {8, 1e-10, 0.01, 50}
  };

  RunStages<decltype(*su_exe), QNT>(*su_exe, model_name, stages, true);

  double en_exact = -6.878533413625821;
  EXPECT_NEAR(su_exe->GetEstimatedEnergy(), en_exact, 0.3);
}

TEST_F(SpinOneHalfSystemSimpleUpdateTrivial, SquareNNNHeisenbergWithAMFPinningFieldOBC) {
  std::string model_name = "square_nnn_hei_pin_obc";
  
  auto su_exe = std::make_unique<SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>>(SimpleUpdatePara(50, 0.1, 1, 2, 1e-5),
                                                                                     *peps0,
                                                                                     ham_hei_nn,
                                                                                     ham_hei_nnn,
                                                                                     afm_pinning_field);
  su_exe->Execute();

  std::vector<UpdateStage> stages = {
      {4, 1e-6, 0.05},
      {8, 1e-10, 0.01, 50}
  };

  RunStages<decltype(*su_exe), QNT>(*su_exe, model_name, stages, true);

  double en_exact = -8.2563506175000985;
  EXPECT_NEAR(su_exe->GetEstimatedEnergy(), en_exact, 0.5);
}

TEST_F(SpinOneHalfSystemSimpleUpdateTrivial, SquareNNNHeisenbergWithAMFPinningFieldPBC) {
  InitializePEPS(BoundaryCondition::Periodic);
  std::string model_name = "square_nnn_hei_pin_pbc";
  
  auto su_exe = std::make_unique<SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>>(SimpleUpdatePara(50, 0.1, 1, 2, 1e-5),
                                                                                     *peps0,
                                                                                     ham_hei_nn,
                                                                                     ham_hei_nnn,
                                                                                     afm_pinning_field);
  su_exe->Execute();

  std::vector<UpdateStage> stages = {
      {4, 1e-6, 0.05},
      {8, 1e-10, 0.01, 50}
  };

  RunStages<decltype(*su_exe), QNT>(*su_exe, model_name, stages, true);

  double en_exact = -8.9324003607025837;
  EXPECT_NEAR(su_exe->GetEstimatedEnergy(), en_exact, 0.5);
}

TEST_F(SpinOneHalfSystemSimpleUpdateTrivial, TriangleNNHeisenbergOBC) {
  std::string model_name = "tri_nn_hei";
  SimpleUpdatePara update_para(20, 0.1, 1, 2, 1e-5);

  auto su_exe = std::make_unique<TriangleNNModelSquarePEPSSimpleUpdateExecutor<TenElemT, QNT>>(update_para,
                                                                                              *peps0,
                                                                                              ham_hei_nn,
                                                                                              ham_hei_tri);
  su_exe->Execute();

  std::vector<UpdateStage> stages = {
      {4, 1e-6, 0.05}
  };
  
  RunStages<decltype(*su_exe), QNT>(*su_exe, model_name, stages, true);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

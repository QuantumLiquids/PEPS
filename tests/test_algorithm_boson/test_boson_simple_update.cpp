// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: QuantumLiquids/PEPS project. Unittests for Simple Update in PEPS optimization.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/simple_update/simple_update_model_all.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using qlten::special_qn::Z2QN;

using TenElemT = TEN_ELEM_TYPE;

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

  IndexT pb_out = IndexT({QNSctT(Z2QN(0), 1),
                          QNSctT(Z2QN(1), 1)},
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);
  //sigma operators
  DTensor xx_term = DTensor({pb_in, pb_out, pb_in, pb_out});
  DTensor z_term = DTensor({pb_in, pb_out});
  using PEPST = SquareLatticePEPS<TenElemT, Z2QN>;
  PEPST peps0 = PEPST(pb_out, Ly, Lx);
  void SetUp(void) {
    z_term({0, 0}) = 1.0 * h;
    z_term({1, 1}) = -1.0 * h;

    xx_term({0, 1, 0, 1}) = 1.0;
    xx_term({1, 0, 1, 0}) = 1.0;
    xx_term({1, 0, 0, 1}) = 1.0;
    xx_term({0, 1, 1, 0}) = 1.0;

    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx, 1));
    peps0.Initial(activates);
  }
};

TEST_F(TransverseFieldIsing, SimpleUpdate) {
  std::string model_name = "square_transverse_field_ising";
  qlten::hp_numeric::SetTensorManipulationThreads(1);
  // stage 1, D = 2
  SimpleUpdatePara update_para(50, 0.1, 1, 2, 1e-5);
  SimpleUpdateExecutor<TenElemT, Z2QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, Z2QN>(update_para, peps0,
                                                                        xx_term,
                                                                        z_term);
  su_exe->Execute();

  // stage 2, D = 4
  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-10;
  su_exe->ResetStepLenth(0.01); // call to re-evaluate the evolution gates
  su_exe->Execute();
  auto tps_d4 = TPS<TenElemT, Z2QN>(su_exe->GetPEPS());
  su_exe->DumpResult(GenPEPSPath(model_name, su_exe->update_para.Dmax), false);
  tps_d4.Dump(GenTPSPath(model_name, su_exe->update_para.Dmax));

  // stage 3, D = 6
  su_exe->update_para.Dmax = 6;
  su_exe->update_para.Trunc_err = 1e-12;
  su_exe->ResetStepLenth(0.001); // call to re-evaluate the evolution gates
  su_exe->Execute();

  // stage 4, D = 8
  su_exe->update_para.Dmax = 8;
  su_exe->update_para.Trunc_err = 1e-15;
  su_exe->update_para.steps = 100;
  su_exe->ResetStepLenth(0.0001);
  su_exe->Execute();
  auto tps_d8 = TPS<TenElemT, Z2QN>(su_exe->GetPEPS());
  su_exe->DumpResult(GenPEPSPath(model_name, su_exe->update_para.Dmax), true);
  tps_d8.Dump(GenTPSPath(model_name, su_exe->update_para.Dmax));
  double E_est = su_exe->GetEstimatedEnergy();
  // check the energy
  double ex_energy = -50.186623882777752;
  EXPECT_NEAR(ex_energy, E_est, 0.2);
  delete su_exe;
}

//spin one-half system with U1 symmetry
struct SpinOneHalfSystemSimpleUpdate : public testing::Test {
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using QNSctVecT = QNSectorVec<U1QN>;

  using Tensor = QLTensor<TenElemT, U1QN>;

  size_t Lx = 4;
  size_t Ly = 3;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
#ifdef U1SYM
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         TenIndexDirType::OUT
  );
#else
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT
  );
#endif
  IndexT pb_in = InverseIndex(pb_out);

  Tensor ham_ising_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
  Tensor ham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});

  double pinning_field_strength = 0.1;
  TenMatrix<Tensor> afm_pinning_field = TenMatrix<Tensor>(Ly, Lx);

  Tensor ham_hei_tri;  // three-site hamiltonian in triangle lattice
  double j2 = 0.52;
  Tensor ham_hei_tri_j2;  // three-site hamiltonian in j1-j2 model
  std::vector<MatrixElement<double>> ham_ising_nn_elements = {
      // Sz_i * Sz_j
      {{0, 0, 0, 0}, 0.25},
      {{1, 1, 1, 1}, 0.25},
      {{1, 1, 0, 0}, -0.25},
      {{0, 0, 1, 1}, -0.25}
  };
  std::vector<MatrixElement<double>> ham_hei_nn_elements = {
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
  PEPST peps0 = PEPST(pb_out, Ly, Lx);
  void SetUp(void) {
    for (const auto &element : ham_ising_nn_elements) {
      ham_ising_nn(element.coors) = element.elem;
    }

    for (const auto &element : ham_hei_nn_elements) {
      ham_hei_nn(element.coors) = element.elem;
    }

    Tensor ham_hei_tri_terms[3];
    for (size_t i = 0; i < 3; i++) {
      std::vector<MatrixElement<double>> tri_elements = GenerateTriElements(ham_hei_nn_elements, i);
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

    // initial peps as classical Neel state
    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));

    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        size_t sz_int = x + y;
        activates[y][x] = sz_int % 2;
      }
    }
    peps0.Initial(activates);
  }
};

TEST_F(SpinOneHalfSystemSimpleUpdate, AFM_ClassicalIsing) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);
  SimpleUpdatePara update_para(100, 0.01, 1, 1, 1e-5);
  SimpleUpdateExecutor<TenElemT, U1QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, U1QN>(update_para, peps0,
                                                                        ham_ising_nn);
  su_exe->Execute();
  double ex_energy = -0.25 * ((Lx - 1) * Ly + (Ly - 1) * Lx);
  EXPECT_NEAR(ex_energy, su_exe->GetEstimatedEnergy(), 1e-10);
  delete su_exe;
}

TEST_F(SpinOneHalfSystemSimpleUpdate, SquareNNHeisenberg) {
  std::string model_name = "square_nn_hei";
  // stage 1, D = 2

  SimpleUpdateExecutor<TenElemT, U1QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, U1QN>(SimpleUpdatePara(50, 0.1, 1, 2, 1e-5),
                                                                        peps0,
                                                                        ham_hei_nn);
  su_exe->Execute();

  // stage 2, D = 4
  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->ResetStepLenth(0.05); // call to re-evaluate the evolution gates
  su_exe->Execute();
  auto tps_d4 = TPS<TenElemT, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult(GenPEPSPath(model_name, su_exe->update_para.Dmax), false);
  tps_d4.Dump(GenTPSPath(model_name, su_exe->update_para.Dmax));

  // stage 3, D = 8
  su_exe->update_para.Dmax = 8;
  su_exe->update_para.Trunc_err = 1e-10;
  su_exe->update_para.steps = 100;
  su_exe->ResetStepLenth(0.01);
  su_exe->Execute();
  auto tps_d8 = TPS<TenElemT, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult(GenPEPSPath(model_name, su_exe->update_para.Dmax), true);
  tps_d8.Dump(GenTPSPath(model_name, su_exe->update_para.Dmax));

  double en_exact = -6.691680193514947;
  EXPECT_NEAR(su_exe->GetEstimatedEnergy(), en_exact, 0.03);
  delete su_exe;
}

TEST_F(SpinOneHalfSystemSimpleUpdate, SquareNNHeisenbergWithAMFPinningField) {
  std::string model_name = "square_nn_hei_pin";
  // stage 1, D = 2

  SimpleUpdateExecutor<TenElemT, U1QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, U1QN>(SimpleUpdatePara(50, 0.1, 1, 2, 1e-5),
                                                                        peps0,
                                                                        ham_hei_nn,
                                                                        afm_pinning_field);
  su_exe->Execute();

  // stage 2, D = 4
  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->ResetStepLenth(0.05); // call to re-evaluate the evolution gates
  su_exe->Execute();
  auto tps_d4 = TPS<TenElemT, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult(GenPEPSPath(model_name, su_exe->update_para.Dmax), false);
  tps_d4.Dump(GenTPSPath(model_name, su_exe->update_para.Dmax));

  // stage 3, D = 8
  su_exe->update_para.Dmax = 8;
  su_exe->update_para.Trunc_err = 1e-10;
  su_exe->update_para.steps = 100;
  su_exe->ResetStepLenth(0.01);
  su_exe->Execute();
  auto tps_d8 = TPS<TenElemT, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult(GenPEPSPath(model_name, su_exe->update_para.Dmax), true);
  tps_d8.Dump(GenTPSPath(model_name, su_exe->update_para.Dmax));

  double en_exact = -6.878533413625821;
  EXPECT_NEAR(su_exe->GetEstimatedEnergy(), en_exact, 0.5);
  delete su_exe;
}

TEST_F(SpinOneHalfSystemSimpleUpdate, TriangleNNHeisenberg) {
  std::string model_name = "tri_nn_hei";
  SimpleUpdatePara update_para(20, 0.1, 1, 2, 1e-5);

  SimpleUpdateExecutor<TenElemT, U1QN> *su_exe
      = new TriangleNNModelSquarePEPSSimpleUpdateExecutor<TenElemT, U1QN>(update_para, peps0,
                                                                          ham_hei_nn,
                                                                          ham_hei_tri);
  su_exe->Execute();

  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->ResetStepLenth(0.05);
  su_exe->Execute();

  auto tps4 = TPS<TenElemT, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult(GenPEPSPath(model_name, su_exe->update_para.Dmax), true);
  tps4.Dump(GenTPSPath(model_name, su_exe->update_para.Dmax));
  delete su_exe;
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

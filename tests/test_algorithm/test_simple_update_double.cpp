// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: QuantumLiquids/PEPS project. Unittests for Simple Update in PEPS optimization.
*/

#define PLAIN_TRANSPOSE 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/algorithm/simple_update/simple_update_model_all.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

using qlmps::CaseParamsParserBasic;

char *params_file;

struct SystemSizeParams : public CaseParamsParserBasic {
  SystemSizeParams(const char *f) : CaseParamsParserBasic(f) {
    Lx = ParseInt("Lx");
    Ly = ParseInt("Ly");
  }

  size_t Ly;
  size_t Lx;
};

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

// Test spin systems
struct SpinOneHalfSystemSimpleUpdate : public testing::Test {
  SystemSizeParams params = SystemSizeParams(params_file);
  size_t Lx = params.Lx; //cols
  size_t Ly = params.Ly;

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

  DTensor dham_ising_nn = DTensor({pb_in, pb_out, pb_in, pb_out});
  DTensor dham_hei_nn = DTensor({pb_in, pb_out, pb_in, pb_out});
  DTensor dham_hei_tri;  // three-site hamiltonian in triangle lattice
  double j2 = 0.52;
  DTensor dham_hei_tri_j2;  // three-site hamiltonian in j1-j2 model
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
  using PEPST = SquareLatticePEPS<QLTEN_Double, U1QN>;
  PEPST peps0 = PEPST(pb_out, Ly, Lx);
  void SetUp(void) {
    for (const auto &element : ham_ising_nn_elements) {
      dham_ising_nn(element.coors) = element.elem;
    }

    for (const auto &element : ham_hei_nn_elements) {
      dham_hei_nn(element.coors) = element.elem;
    }

    DTensor ham_hei_tri_terms[3];
    for (size_t i = 0; i < 3; i++) {
      std::vector<MatrixElement<double>> tri_elements = GenerateTriElements(ham_hei_nn_elements, i);
      ham_hei_tri_terms[i] = DTensor({pb_in, pb_out, pb_in, pb_out, pb_in, pb_out});
      for (const auto &element : tri_elements) {
        ham_hei_tri_terms[i](element.coors) = element.elem;
      }
    }
    dham_hei_tri = ham_hei_tri_terms[0] + ham_hei_tri_terms[1] + ham_hei_tri_terms[2];
    dham_hei_tri_j2 = 0.5 * ham_hei_tri_terms[0] + j2 * ham_hei_tri_terms[1] + 0.5 * ham_hei_tri_terms[2];
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
  SimpleUpdatePara update_para(5, 0.01, 1, 1, 1e-5);
  SimpleUpdateExecutor<QLTEN_Double, U1QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Double, U1QN>(update_para, peps0,
                                                                            dham_ising_nn);
  su_exe->Execute();
  delete su_exe;
}

TEST_F(SpinOneHalfSystemSimpleUpdate, SquareNNHeisenberg) {
  // stage 1, D = 2
  SimpleUpdatePara update_para(50, 0.1, 1, 2, 1e-5);
  SimpleUpdateExecutor<QLTEN_Double, U1QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Double, U1QN>(update_para, peps0,
                                                                            dham_hei_nn);
  su_exe->Execute();

  // stage 2, D = 4
  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->SetStepLenth(0.05); // call to re-evaluate the evolution gates
  su_exe->Execute();
  auto tps_d4 = TPS<QLTEN_Double, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult("peps_square_nn_hei_D4", false);
  tps_d4.Dump("tps_square_nn_hei_D4");

  // stage 2, D = 8
  su_exe->update_para.Dmax = 8;
  su_exe->update_para.Trunc_err = 1e-10;
  su_exe->update_para.steps = 100;
  su_exe->SetStepLenth(0.01);
  su_exe->Execute();
  auto tps_d8 = TPS<QLTEN_Double, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult("peps_square_nn_hei_D8", true);
  tps_d8.Dump("tps_square_nn_hei_D8");
  delete su_exe;
}

TEST_F(SpinOneHalfSystemSimpleUpdate, TriangleNNHeisenberg) {
  SimpleUpdatePara update_para(20, 0.1, 1, 2, 1e-5);

  SimpleUpdateExecutor<QLTEN_Double, U1QN> *su_exe
      = new TriangleNNModelSquarePEPSSimpleUpdateExecutor<QLTEN_Double, U1QN>(update_para, peps0,
                                                                              dham_hei_nn,
                                                                              dham_hei_tri);
  su_exe->Execute();

  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->SetStepLenth(0.05);
  su_exe->Execute();

  auto tps4 = TPS<QLTEN_Double, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult("peps_tri_nn_heisenberg_D4", true);
  tps4.Dump("tps_tri_nn_heisenberg_D4");
  delete su_exe;
}

TEST_F(SpinOneHalfSystemSimpleUpdate, SquareJ1J2Heisenberg) {
  SimpleUpdatePara update_para(10, 0.1, 1, 2, 1e-5);

  SimpleUpdateExecutor<QLTEN_Double, U1QN>
      *su_exe = new SquareLatticeNNNSimpleUpdateExecutor<QLTEN_Double, U1QN>(update_para, peps0,
                                                                             dham_hei_nn,
                                                                             dham_hei_tri_j2);
  su_exe->Execute();

  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-6;
  su_exe->SetStepLenth(0.05);
  su_exe->Execute();

  auto tps4 = TPS<QLTEN_Double, U1QN>(su_exe->GetPEPS());
  su_exe->DumpResult("peps_square_j1j2_hei_D4", true);
  tps4.Dump("tps_square_j1j2_hei_D4");
  delete su_exe;
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  params_file = argv[1];
  return RUN_ALL_TESTS();
}

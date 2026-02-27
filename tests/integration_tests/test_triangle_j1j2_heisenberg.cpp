// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-12-19
*
* Description: QuantumLiquids/PEPS project. Integration testing for Triangle J1-J2 Heisenberg model with VMC optimization.
*/

#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "integration_test_framework.h"

using namespace qlten;
using namespace qlpeps;

using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = TEN_ELEM_TYPE;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;

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

class TriangleJ1J2HeisenbergSystem
    : public IntegrationTestFramework<QNT, TriangleJ1J2HeisenbergSystem> {
  protected:
    double j2 = 0.2;
    Tensor ham_hei_nn; // nearest-neighbor hamiltonian
    Tensor ham_hei_tri; // three-site hamiltonian in triangle lattice
    std::vector<MatrixElement<double>> ham_hei_nn_elements;

    void SetUpIndices() override {
      pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
      pb_in = InverseIndex(pb_out);
    }

    void SetUpHamiltonians() override {
      // Define matrix elements for nearest-neighbor Heisenberg Hamiltonian
      ham_hei_nn_elements = {
        // Sz_i * Sz_j
        {{0, 0, 0, 0}, 0.25},
        {{1, 1, 1, 1}, 0.25},
        {{1, 1, 0, 0}, -0.25},
        {{0, 0, 1, 1}, -0.25},
        // 0.5 * S^+_i * S^-_j
        {{0, 1, 1, 0}, 0.5},
        // 0.5 * S^-_i * S^+_j
        {{1, 0, 0, 1}, 0.5},
      };
      
      // Construct nearest-neighbor Heisenberg Hamiltonian
      ham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
      for (const auto &element : ham_hei_nn_elements) {
        ham_hei_nn(element.coors) = element.elem;
      }

      // Triangle lattice three-site Hamiltonian with J1-J2 interactions
      Tensor ham_hei_tri_terms[3];
      for (size_t i = 0; i < 3; i++) {
        std::vector<MatrixElement<double>> tri_elements = GenerateTriElements(ham_hei_nn_elements, i);
        ham_hei_tri_terms[i] = Tensor({pb_in, pb_out, pb_in, pb_out, pb_in, pb_out});
        for (const auto &element : tri_elements) {
          ham_hei_tri_terms[i](element.coors) = element.elem;
        }
      }
      // Apply J1-J2 coupling: J1=1.0 for bonds 0 and 2, J2 for bond 1
      ham_hei_tri = ham_hei_tri_terms[0] + j2 * ham_hei_tri_terms[1] + ham_hei_tri_terms[2];
    }

    void SetUpParameters() override {
      model_name = "triangle_j1j2_heisenberg";
      energy_ed = -8.5; // Approximate expected energy for triangle J1-J2 lattice

      optimize_para.emplace(
          OptimizerFactory::CreateStochasticReconfiguration(40, ConjugateGradientParams(100, 3e-3, 20, 0.001), 0.3),
          MonteCarloParams(5000, 100, 1,
                           Configuration(Ly, Lx,
                                         OccupancyNum({Lx * Ly / 2, Lx * Ly / 2})),
                           false), // Sz = 0, not warmed up initially
          PEPSParams(BMPSTruncateParams<qlten::QLTEN_Double>(6, 12, 1e-15,
                                      CompressMPSScheme::SVD_COMPRESS,
                                      std::make_optional<double>(1e-14),
                                      std::make_optional<size_t>(10))));

      Configuration measure_config{Ly, Lx, OccupancyNum(std::vector<size_t>(2, Lx * Ly / 2))};
      MonteCarloParams measure_mc_params{50000, 1000, 1, measure_config, false}; // not warmed up initially
      PEPSParams measure_peps_params{BMPSTruncateParams<qlten::QLTEN_Double>(Dpeps, 2 * Dpeps, 1e-15,
                                                      CompressMPSScheme::SVD_COMPRESS,
                                                      std::make_optional<double>(1e-14),
                                                      std::make_optional<size_t>(10))};
      measure_para.emplace(measure_mc_params, measure_peps_params);
  }
};

TEST_F(TriangleJ1J2HeisenbergSystem, SimpleUpdate) {
  if (rank == hp_numeric::kMPIMasterRank) {
    SquareLatticePEPS<TenElemT, QNT> peps0(pb_out, Ly, Lx);
    std::vector<std::vector<size_t> > activates(Ly, std::vector<size_t>(Lx));
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        size_t sz_int = x + y;
        activates[y][x] = sz_int % 2;
      }
    }
    peps0.Initial(activates);

    SimpleUpdatePara update_para(1000, 0.1, 1, 4, 1e-15);
    auto su_exe = new TriangleNNModelSquarePEPSSimpleUpdateExecutor<TenElemT, QNT>(
      update_para,
      peps0,
      ham_hei_nn,
      ham_hei_tri);

    RunSimpleUpdate(su_exe);
    delete su_exe;
  }
}

TEST_F(TriangleJ1J2HeisenbergSystem, ZeroUpdate) {
  using Model = SpinOneHalfTriJ1J2HeisenbergSqrPEPS;
  Model trianglej1j2_hei_solver(j2);
  RunZeroUpdateTest<Model, MCUpdateSquareNNExchange>(trianglej1j2_hei_solver);
}

TEST_F(TriangleJ1J2HeisenbergSystem, StochasticReconfigurationOpt) {
  using Model = SpinOneHalfTriJ1J2HeisenbergSqrPEPS;
  Model trianglej1j2_hei_solver(j2);

  // VMC optimization
  RunVMCOptimization<Model, MCUpdateSquareNNExchange>(trianglej1j2_hei_solver);

  // Monte Carlo measurement
  RunMCMeasurement<Model, MCUpdateSquareNNExchange>(trianglej1j2_hei_solver);
}

TEST_F(TriangleJ1J2HeisenbergSystem, StochasticGradientOpt) {
  using Model = SpinOneHalfTriJ1J2HeisenbergSqrPEPS;
  Model trianglej1j2_hei_solver(j2);

  // Change to stochastic gradient
  optimize_para->optimizer_params = OptimizerFactory::CreateSGDWithDecay(40, 0.1, 1.0, 1000);

  // VMC optimization
  RunVMCOptimization<Model, MCUpdateSquareNNExchange>(trianglej1j2_hei_solver);

  // Monte Carlo measurement
  RunMCMeasurement<Model, MCUpdateSquareNNExchange>(trianglej1j2_hei_solver);
}

TEST_F(TriangleJ1J2HeisenbergSystem, LBFGSOptimization) {
  using Model = SpinOneHalfTriJ1J2HeisenbergSqrPEPS;
  Model trianglej1j2_hei_solver(j2);

  // L-BFGS update (iterative optimizer path; not LineSearchOptimize).
  optimize_para->optimizer_params = OptimizerFactory::CreateLBFGS(40, 0.01);

  // VMC optimization
  RunVMCOptimization<Model, MCUpdateSquareNNExchange>(trianglej1j2_hei_solver);

  // Monte Carlo measurement
  RunMCMeasurement<Model, MCUpdateSquareNNExchange>(trianglej1j2_hei_solver);
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}

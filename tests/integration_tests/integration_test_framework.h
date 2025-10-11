// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-12-19
*
* Description: QuantumLiquids/PEPS project. Common framework for integration tests.
*/

#ifndef PEPS_INTEGRATION_TEST_FRAMEWORK_H
#define PEPS_INTEGRATION_TEST_FRAMEWORK_H

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "../test_mpi_env.h"
#include "../utilities.h"

#include <type_traits>
#include <utility>

using namespace qlten;
using namespace qlpeps;

template<typename QNT, typename DerivedT>
class IntegrationTestFramework : public MPITest {
protected:
  using TenElemT = TEN_ELEM_TYPE;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using Tensor = QLTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using SplitIndexTPST = SplitIndexTPS<TenElemT, QNT>;

  // Common parameters
  size_t Lx = 3;
  size_t Ly = 4;
  size_t Dpeps = 8;
  QNT qn0 = QNT();
  
  // Physical indices
  IndexT pb_out;
  IndexT pb_in;
  
  // Paths
  std::string model_name;
  std::string tps_path;
  
  // Expected energy for validation
  double energy_ed;
  
  // VMC optimization parameters
  VMCPEPSOptimizerParams optimize_para;
  
  // Monte Carlo measurement parameters
  MCMeasurementParams measure_para;

  virtual void SetUpIndices() = 0;
  virtual void SetUpHamiltonians() = 0;
  virtual void SetUpParameters() = 0;
  
  void SetUp() override {
    MPITest::SetUp();
    SetUpIndices();
    SetUpHamiltonians();
    SetUpParameters();
    
    // Set common paths
    tps_path = GenTPSPath(model_name, Dpeps, Lx, Ly);
    // New API: TPS path is handled by the caller, not stored in parameters
  }

  // Common simple update workflow
  template<typename SimpleUpdateExecutorT>
  void RunSimpleUpdate(SimpleUpdateExecutorT* executor) {
    if (rank == hp_numeric::kMPIMasterRank) {
      executor->Execute();
      
      // Refine with smaller step lengths
      executor->update_para.Dmax = 6;
      executor->update_para.Trunc_err = 1e-15;
      executor->ResetStepLenth(0.01);
      executor->Execute();

      executor->update_para.Dmax = Dpeps;
      executor->update_para.Trunc_err = 1e-15;
      executor->ResetStepLenth(0.001);
      executor->Execute();

      // Save TPS
      auto tps = ToTPS<TenElemT, QNT>(executor->GetPEPS());
      for (auto &ten : tps) {
        ten *= (1.0 / ten.GetMaxAbs());
      }
      SplitIndexTPST sitps = ToSplitIndexTPS<TenElemT, QNT>(tps);
      sitps.Dump(tps_path);
    }
  }

  // Common VMC optimization workflow
  template<typename ModelT, typename MCUpdaterT>
  void RunVMCOptimization(const ModelT& model) {
    MPI_Barrier(comm);
    
    SplitIndexTPST tps(Ly, Lx);
    tps.Load(tps_path);

    auto executor = new VMCPEPSOptimizer<TenElemT, QNT, MCUpdaterT, ModelT>(
        optimize_para, tps, comm, model);
    
    size_t start_flop = flop;
    Timer vmc_timer("vmc");
    
    executor->Execute();
    
    size_t end_flop = flop;
    double elapsed_time = vmc_timer.PrintElapsed();
    double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
    std::cout << "VMC Gflops = " << Gflops / elapsed_time << std::endl;

    tps = executor->GetState();
    delete executor;
    
    // Save optimized TPS
    tps.Dump(tps_path);
  }

  // Common Monte Carlo measurement workflow
  template<typename ModelT, typename MCUpdaterT>
  void RunMCMeasurement(const ModelT& model) {
    SplitIndexTPST tps(Ly, Lx);
    tps.Load(tps_path);

    auto measure_exe = new MCPEPSMeasurer<TenElemT, QNT, MCUpdaterT, ModelT>(
        tps, measure_para, comm, model);
    
    size_t start_flop = flop;
    Timer measure_timer("measurement");
    
    measure_exe->Execute();
    
    size_t end_flop = flop;
    double elapsed_time = measure_timer.PrintElapsed();
    double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
    std::cout << "Measurement Gflops = " << Gflops / elapsed_time << std::endl;

    if (EnableFrameworkEnergyCheck()) {
      auto [energy, en_err] = measure_exe->OutputEnergy();
      (void)en_err;
      EXPECT_NEAR(std::real(energy), energy_ed, FrameworkEnergyTolerance());
    }
    
    InvokeValidateMeasurementResults<ModelT, MCUpdaterT>(*measure_exe);

    delete measure_exe;
  }

  // Zero update test to verify TPS doesn't change
  template<typename ModelT, typename MCUpdaterT>
  void RunZeroUpdateTest(const ModelT& model) {
    MPI_Barrier(comm);
    optimize_para.optimizer_params.base_params.learning_rate = 0.0;
    
    SplitIndexTPST tps(Ly, Lx);
    tps.Load(tps_path);
    auto init_tps = tps;
    
    auto executor = new VMCPEPSOptimizer<TenElemT, QNT, MCUpdaterT, ModelT>(
        optimize_para, tps, comm, model);
    
    size_t start_flop = flop;
    Timer vmc_timer("vmc");
    executor->Execute();
    size_t end_flop = flop;
    double elapsed_time = vmc_timer.Elapsed();
    double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
    std::cout << "flop = " << end_flop - start_flop << std::endl;
    std::cout << "Gflops = " << Gflops << std::endl;
    
    SplitIndexTPST result_sitps = executor->GetState();
    auto diff = init_tps + (-result_sitps);
    EXPECT_NE(diff.NormSquare(), 1e-14);
    
    delete executor;
  }

  /**
   * @brief Allow derived tests to opt out of the default energy benchmark.
   */
  virtual bool EnableFrameworkEnergyCheck() const { return true; }

  /**
   * @brief Default tolerance used when the framework checks the ED energy.
   */
  virtual double FrameworkEnergyTolerance() const { return 1e-3; }

private:
  template<typename T, typename ModelT, typename MCUpdaterT, typename = void>
  struct HasValidateMeasurementResults : std::false_type {};

  template<typename T, typename ModelT, typename MCUpdaterT>
  struct HasValidateMeasurementResults<
      T, ModelT, MCUpdaterT,
      std::void_t<decltype(std::declval<const T>().template ValidateMeasurementResults<ModelT, MCUpdaterT>(
          std::declval<const MCPEPSMeasurer<TenElemT, QNT, MCUpdaterT, ModelT>&>()))>>
      : std::true_type {};

  template<typename ModelT, typename MCUpdaterT>
  void InvokeValidateMeasurementResults(
      const MCPEPSMeasurer<TenElemT, QNT, MCUpdaterT, ModelT> &measurer) const {
    if constexpr (HasValidateMeasurementResults<DerivedT, ModelT, MCUpdaterT>::value) {
      static_cast<const DerivedT *>(this)->template ValidateMeasurementResults<ModelT, MCUpdaterT>(measurer);
    }
  }
};

#endif // PEPS_INTEGRATION_TEST_FRAMEWORK_H 
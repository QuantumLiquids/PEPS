// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-06-11
*
* Description: QuantumLiquids/PEPS project. Utilities for tests.
*/


#ifndef PEPS_TESTS_UTILITIES_H
#define PEPS_TESTS_UTILITIES_H

#include <string>
#include <filesystem>

/**
 * @brief Generate test output path directly in build directory (cleaner approach)
 * @param test_name Name of the test (e.g., "vmc_peps_optimizer", "exact_sum_optimization") 
 * @param data_subdir Subdirectory for data (e.g., "tps_square_heisenberg4x4D8Double")
 * @return Path directly in build directory (current working directory when running ctest)
 */
std::string GetTestOutputPath(const std::string& test_name, const std::string& data_subdir = "") {
  std::string path;
  if (!data_subdir.empty()) {
    // Simple format: testname_datasubdir (e.g., "vmc_test_heisenberg_double")
    path = test_name + "_" + data_subdir;
  } else {
    path = test_name;
  }
  
  // Create directory if it doesn't exist (relative to current working directory = build/)
  std::filesystem::create_directories(path);
  
  return path;
}

/**
 * @brief Get reference test data path (read-only, in source directory)
 * @param data_subdir Subdirectory under test_data (e.g., "heisenberg_tps")
 * @return Path to source test data
 */
std::string GetTestDataPath(const std::string& data_subdir) {
  return std::string(TEST_SOURCE_DIR) + "/test_data/" + data_subdir;
}

std::string GenTPSPath(std::string model_name, size_t Dmax, size_t Lx, size_t Ly) {
#if TEN_ELEM_TYPE == QLTEN_Double
  return "dtps_" + model_name + "_D" + std::to_string(Dmax) + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#elif TEN_ELEM_TYPE == QLTEN_Complex
  return "ztps_" + model_name + "_D" + std::to_string(Dmax)  + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#else
#error "Unexpected TEN_ELEM_TYPE"
#endif
}

#endif //PEPS_TESTS_UTILITIES_H

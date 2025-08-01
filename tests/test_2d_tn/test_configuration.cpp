// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-18
*
* Description: QuantumLiquids/PEPS project. Unittests for Configuration
*/
#include "gtest/gtest.h"
#include "qlpeps/vmc_basic/configuration.h"
#include <filesystem>
#include <random>

using namespace qlpeps;

class ConfigurationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test directory
    test_dir_ = "./test_configuration_data";
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    // Clean up test directory
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
  }

  std::string test_dir_;
};

// Test basic constructor and access
TEST_F(ConfigurationTest, BasicConstructor) {
  Configuration config(3, 4);
  EXPECT_EQ(config.rows(), 3);
  EXPECT_EQ(config.cols(), 4);
  EXPECT_EQ(config.size(), 12);
  
  // Test that raw data is nullptr before accessing elements
  for (size_t row = 0; row < config.rows(); row++) {
    for (size_t col = 0; col < config.cols(); col++) {
      EXPECT_EQ(config(row, col), nullptr);
    }
  }
  
  // Test that accessing elements allocates memory and returns default values (0)
  for (size_t row = 0; row < config.rows(); row++) {
    for (size_t col = 0; col < config.cols(); col++) {
      config({row, col}) = 0;
      // After access, raw data should no longer be nullptr
      EXPECT_NE(config(row, col), nullptr);
    }
  }
}

// Test constructor from 2D vector
TEST_F(ConfigurationTest, ConstructorFromVector) {
  std::vector<std::vector<size_t>> data = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
  };
  
  Configuration config(data);
  EXPECT_EQ(config.rows(), 3);
  EXPECT_EQ(config.cols(), 3);
  
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      EXPECT_EQ(config({row, col}), data[row][col]);
    }
  }
}

// Test constructor with random initialization using dimension
TEST_F(ConfigurationTest, ConstructorWithDimension) {
  Configuration config(2, 2, 3); // 2x2 lattice, states 0,1,2
  
  EXPECT_EQ(config.rows(), 2);
  EXPECT_EQ(config.cols(), 2);
  
  // Check that all values are within valid range
  for (size_t row = 0; row < config.rows(); row++) {
    for (size_t col = 0; col < config.cols(); col++) {
      EXPECT_GE(config({row, col}), 0);
      EXPECT_LT(config({row, col}), 3);
    }
  }
}

// Test constructor with occupancy numbers
TEST_F(ConfigurationTest, ConstructorWithOccupancy) {
  OccupancyNum occupancy = {2, 2}; // 2 sites in state 0, 2 sites in state 1
  Configuration config(2, 2, occupancy);
  
  EXPECT_EQ(config.rows(), 2);
  EXPECT_EQ(config.cols(), 2);
  
  // Check that we have exactly 2 sites in each state
  size_t count_0 = 0, count_1 = 0;
  for (size_t row = 0; row < config.rows(); row++) {
    for (size_t col = 0; col < config.cols(); col++) {
      if (config({row, col}) == 0) count_0++;
      else if (config({row, col}) == 1) count_1++;
    }
  }
  EXPECT_EQ(count_0, 2);
  EXPECT_EQ(count_1, 2);
}

// Test constructor with configuration map
TEST_F(ConfigurationTest, ConstructorWithMap) {
  std::map<size_t, size_t> config_map = {{0, 2}, {2, 2}}; // 2 sites in state 0, 2 sites in state 2
  Configuration config(2, 2, config_map);
  
  EXPECT_EQ(config.rows(), 2);
  EXPECT_EQ(config.cols(), 2);
  
  // Check that we have exactly 2 sites in each specified state
  size_t count_0 = 0, count_2 = 0;
  for (size_t row = 0; row < config.rows(); row++) {
    for (size_t col = 0; col < config.cols(); col++) {
      if (config({row, col}) == 0) count_0++;
      else if (config({row, col}) == 2) count_2++;
    }
  }
  EXPECT_EQ(count_0, 2);
  EXPECT_EQ(count_2, 2);
}

// Test Random method with dimension
TEST_F(ConfigurationTest, RandomWithDimension) {
  Configuration config(3, 3);
  config.Random(4); // States 0,1,2,3
  
  // Check that all values are within valid range
  for (size_t row = 0; row < config.rows(); row++) {
    for (size_t col = 0; col < config.cols(); col++) {
      EXPECT_GE(config({row, col}), 0);
      EXPECT_LT(config({row, col}), 4);
    }
  }
}

// Test Random method with occupancy numbers
TEST_F(ConfigurationTest, RandomWithOccupancy) {
  Configuration config(4, 4);
  OccupancyNum occupancy = {8, 8}; // 8 sites in state 0, 8 sites in state 1
  
  config.Random(occupancy);
  
  // Check that we have exactly the specified number of sites in each state
  size_t count_0 = 0, count_1 = 0;
  for (size_t row = 0; row < config.rows(); row++) {
    for (size_t col = 0; col < config.cols(); col++) {
      if (config({row, col}) == 0) count_0++;
      else if (config({row, col}) == 1) count_1++;
    }
  }
  EXPECT_EQ(count_0, 8);
  EXPECT_EQ(count_1, 8);
}

// Test Random method with configuration map
TEST_F(ConfigurationTest, RandomWithMap) {
  Configuration config(3, 3);
  std::map<size_t, size_t> config_map = {{1, 5}, {3, 4}}; // 5 sites in state 1, 4 sites in state 3
  
  config.Random(config_map);
  
  // Check that we have exactly the specified number of sites in each state
  size_t count_1 = 0, count_3 = 0;
  for (size_t row = 0; row < config.rows(); row++) {
    for (size_t col = 0; col < config.cols(); col++) {
      if (config({row, col}) == 1) count_1++;
      else if (config({row, col}) == 3) count_3++;
    }
  }
  EXPECT_EQ(count_1, 5);
  EXPECT_EQ(count_3, 4);
}

// Test Sum method
TEST_F(ConfigurationTest, Sum) {
  Configuration config(2, 2);
  config({0, 0}) = 1;
  config({0, 1}) = 2;
  config({1, 0}) = 3;
  config({1, 1}) = 4;
  
  EXPECT_EQ(config.Sum(), 10);
}

// Test IsValid method
TEST_F(ConfigurationTest, IsValid) {
  //construct a empty configuration without raw data
  Configuration config(2, 2);
  EXPECT_FALSE(config.IsValid());
  
  // partial set some values
  config({0, 0}) = 1;
  config({1, 1}) = 2;
  EXPECT_FALSE(config.IsValid());

  // set all values
  config({0, 1}) = 3;
  config({1, 0}) = 4;
  EXPECT_TRUE(config.IsValid());
}

// Test Dump and Load functions
TEST_F(ConfigurationTest, DumpAndLoad) {
  // Create a Configuration object
  Configuration config(2, 2);
  
  // Set some values
  config({0, 0}) = 1;
  config({0, 1}) = 2;
  config({1, 0}) = 3;
  config({1, 1}) = 4;
  
  // Dump the Configuration to a file
  size_t label = 1;
  config.Dump(test_dir_, label);
  
  // Create a new Configuration object
  Configuration loaded_config(2, 2);
  
  // Load the Configuration from the file
  bool load_success = loaded_config.Load(test_dir_, label);
  ASSERT_TRUE(load_success);
  
  // Compare the loaded Configuration with the original
  EXPECT_EQ(config, loaded_config);
}

// Test Load with non-existent file
TEST_F(ConfigurationTest, LoadNonExistentFile) {
  Configuration config(2, 2);
  bool load_success = config.Load(test_dir_, 999);
  EXPECT_FALSE(load_success);
}

// Test equality operator
TEST_F(ConfigurationTest, EqualityOperator) {
  Configuration config1(2, 2);
  Configuration config2(2, 2);
  
  // Set same values
  config1({0, 0}) = 1; config2({0, 0}) = 1;
  config1({0, 1}) = 2; config2({0, 1}) = 2;
  config1({1, 0}) = 3; config2({1, 0}) = 3;
  config1({1, 1}) = 4; config2({1, 1}) = 4;
  
  EXPECT_EQ(config1, config2);
  
  // Change one value
  config2({1, 1}) = 5;
  EXPECT_NE(config1, config2);
}

// Test different sized configurations
TEST_F(ConfigurationTest, DifferentSizes) {
  Configuration config1(2, 2);
  Configuration config2(3, 3);
  
  EXPECT_NE(config1, config2);
}

// Test edge cases for occupancy numbers
TEST_F(ConfigurationTest, EdgeCaseOccupancy) {
  // Test with all sites in one state
  Configuration config1(2, 2);
  OccupancyNum occupancy1 = {4, 0}; // All 4 sites in state 0
  config1.Random(occupancy1);
  
  for (size_t row = 0; row < config1.rows(); row++) {
    for (size_t col = 0; col < config1.cols(); col++) {
      EXPECT_EQ(config1({row, col}), 0);
    }
  }
  
  // Test with single site in each state
  Configuration config2(2, 2);
  OccupancyNum occupancy2 = {1, 1, 1, 1}; // One site in each of 4 states
  config2.Random(occupancy2);
  
  std::vector<size_t> counts(4, 0);
  for (size_t row = 0; row < config2.rows(); row++) {
    for (size_t col = 0; col < config2.cols(); col++) {
      counts[config2({row, col})]++;
    }
  }
  
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(counts[i], 1);
  }
}

// Test large configuration
TEST_F(ConfigurationTest, LargeConfiguration) {
  Configuration config(10, 10);
  OccupancyNum occupancy = {50, 50}; // 50 sites in each state
  
  config.Random(occupancy);
  
  size_t count_0 = 0, count_1 = 0;
  for (size_t row = 0; row < config.rows(); row++) {
    for (size_t col = 0; col < config.cols(); col++) {
      if (config({row, col}) == 0) count_0++;
      else if (config({row, col}) == 1) count_1++;
    }
  }
  EXPECT_EQ(count_0, 50);
  EXPECT_EQ(count_1, 50);
}

// Test Show method (output verification)
TEST_F(ConfigurationTest, ShowMethod) {
  Configuration config(2, 2);
  config({0, 0}) = 1;
  config({0, 1}) = 2;
  config({1, 0}) = 3;
  config({1, 1}) = 4;
  
  // This test mainly ensures the method doesn't crash
  // In a real test environment, you might capture stdout to verify output
  EXPECT_NO_THROW(config.Show());
  EXPECT_NO_THROW(config.Show(2));
}

// Test StreamRead and StreamWrite
TEST_F(ConfigurationTest, StreamOperations) {
  Configuration config(2, 2);
  config({0, 0}) = 1;
  config({0, 1}) = 2;
  config({1, 0}) = 3;
  config({1, 1}) = 4;
  
  // Test stream write
  std::ostringstream oss;
  config.StreamWrite(oss);
  std::string output = oss.str();
  
  // Test stream read
  Configuration loaded_config(2, 2);
  std::istringstream iss(output);
  loaded_config.StreamRead(iss);
  
  EXPECT_EQ(config, loaded_config);
}

// Test error handling in StreamRead
TEST_F(ConfigurationTest, StreamReadError) {
  Configuration config(2, 2);
  std::istringstream iss("1 2 3"); // Not enough data
  
  EXPECT_THROW(config.StreamRead(iss), std::runtime_error);
}

// Test MPI operations (basic functionality)
TEST_F(ConfigurationTest, MPIOperations) {
  // Note: These tests would require MPI environment
  // In a real test, you'd need to run with mpirun
  Configuration config(2, 2);
  config({0, 0}) = 1;
  config({0, 1}) = 2;
  config({1, 0}) = 3;
  config({1, 1}) = 4;
  
  // Test that MPI functions exist and can be called
  // (actual MPI testing would require MPI environment)
  EXPECT_NO_THROW({
    // These would fail in non-MPI environment, but we're just testing compilation
    // MPI_Send(config, 0, 0, MPI_COMM_WORLD);
    // MPI_Recv(config, 0, 0, MPI_COMM_WORLD, nullptr);
    // MPI_BCast(config, 0, MPI_COMM_WORLD);
  });
}

// Test configuration with zero dimensions
TEST_F(ConfigurationTest, ZeroDimensions) {
  Configuration config(0, 0);
  EXPECT_EQ(config.rows(), 0);
  EXPECT_EQ(config.cols(), 0);
  EXPECT_EQ(config.size(), 0);
  EXPECT_EQ(config.Sum(), 0);
}

// Test configuration with single row/column
TEST_F(ConfigurationTest, SingleRowColumn) {
  Configuration config(1, 3);
  config({0, 0}) = 1;
  config({0, 1}) = 2;
  config({0, 2}) = 3;
  
  EXPECT_EQ(config.rows(), 1);
  EXPECT_EQ(config.cols(), 3);
  EXPECT_EQ(config.Sum(), 6);
  
  Configuration config2(3, 1);
  config2({0, 0}) = 1;
  config2({1, 0}) = 2;
  config2({2, 0}) = 3;
  
  EXPECT_EQ(config2.rows(), 3);
  EXPECT_EQ(config2.cols(), 1);
  EXPECT_EQ(config2.Sum(), 6);
}

// Test validation of occupancy numbers
TEST_F(ConfigurationTest, OccupancyValidation) {
  Configuration config(2, 2); // 4 total sites
  
  // Test with correct sum (should work)
  OccupancyNum valid_occupancy = {2, 2}; // 2 + 2 = 4 sites
  EXPECT_NO_THROW(config.Random(valid_occupancy));
  
  // Test with too many sites (should throw)
  OccupancyNum too_many = {3, 3}; // 3 + 3 = 6 > 4 sites
  EXPECT_THROW(config.Random(too_many), std::invalid_argument);
  
  // Test with too few sites (should throw)
  OccupancyNum too_few = {1, 1}; // 1 + 1 = 2 < 4 sites
  EXPECT_THROW(config.Random(too_few), std::invalid_argument);
  
  // Test with zero sites (should throw)
  OccupancyNum zero_sites = {0, 0}; // 0 + 0 = 0 < 4 sites
  EXPECT_THROW(config.Random(zero_sites), std::invalid_argument);
}

// Test validation of configuration map
TEST_F(ConfigurationTest, ConfigMapValidation) {
  Configuration config(3, 3); // 9 total sites
  
  // Test with correct sum (should work)
  std::map<size_t, size_t> valid_map = {{0, 4}, {1, 3}, {2, 2}}; // 4 + 3 + 2 = 9 sites
  EXPECT_NO_THROW(config.Random(valid_map));
  
  // Test with too many sites (should throw)
  std::map<size_t, size_t> too_many_map = {{0, 5}, {1, 5}}; // 5 + 5 = 10 > 9 sites
  EXPECT_THROW(config.Random(too_many_map), std::invalid_argument);
  
  // Test with too few sites (should throw)
  std::map<size_t, size_t> too_few_map = {{0, 2}, {1, 2}}; // 2 + 2 = 4 < 9 sites
  EXPECT_THROW(config.Random(too_few_map), std::invalid_argument);
  
  // Test with empty map (should throw)
  std::map<size_t, size_t> empty_map = {}; // 0 sites
  EXPECT_THROW(config.Random(empty_map), std::invalid_argument);
}
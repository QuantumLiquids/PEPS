// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-18
*
* Description: QuantumLiquids/PEPS project. Unittests for Configuration
*/
#include "gtest/gtest.h"
#include "qlpeps/two_dim_tn/tps/configuration.h"

using namespace qlpeps;

// Test case for Dump and Load functions (should be inverse operations)
TEST(ConfigurationTest, DumpAndLoad) {
  // Create a Configuration object
  Configuration config(2, 2); // Example dimensions: 2 rows, 2 columns

  // Set some values in the Configuration
  config({0, 0}) = 1;
  config({0, 1}) = 2;
  config({1, 0}) = 3;
  config({1, 1}) = 4;

  // Dump the Configuration to a file
  std::string path = "./configuration_test_data";
  size_t label = 1;
  config.Dump(path, label);

  // Create a new Configuration object
  Configuration loaded_config(2, 2);

  // Load the Configuration from the file
  bool load_success = loaded_config.Load(path, label);
  ASSERT_TRUE(load_success); // Ensure the file was successfully loaded

  // Compare the loaded Configuration with the original Configuration
  EXPECT_EQ(config, loaded_config);
}

// Test case for Random function with given occupancy numbers
TEST(ConfigurationTest, RandomInitWithOccupancy) {
  // Create a Configuration object
  Configuration config(10, 10); // Example dimensions: 2 rows, 2 columns

  // Set the occupancy numbers
  std::vector<size_t> occupancy_num = {40, 30, 20, 10}; // Example occupancy numbers

  // Initialize the Configuration randomly using the given occupancy numbers
  config.Random(occupancy_num);

  size_t sum = 0;
  for (size_t i = 0; i < occupancy_num.size(); i++) {
    sum += occupancy_num[i] * i;
  }
  EXPECT_EQ(config.Sum(), sum);
}
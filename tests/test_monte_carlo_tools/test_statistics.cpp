/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-07
*
* Description: GraceQ/VMC-PEPS project. Unittests for Monte-Carlo Statistic functions.
*/

#include "gtest/gtest.h"
#include "gqpeps/monte_carlo_tools/statistics.h"

class StatisticsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up any necessary resources before each test
  }

  void TearDown() override {
    // Clean up any resources after each test
  }
};

TEST_F(StatisticsTest, MeanTest) {
  std::vector<int> data = {1, 2, 3, 4, 5};
  int expectedMean = 3;
  int actualMean = gqpeps::Mean(data);
  EXPECT_EQ(expectedMean, actualMean);
}

TEST_F(StatisticsTest, StandardErrorTest) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float mean = gqpeps::Mean(data);
  float expectedStdError = std::sqrt(2.0) / 2.0;
  float actualStdError = gqpeps::StandardError(data, mean);
  EXPECT_FLOAT_EQ(expectedStdError, actualStdError);
}

TEST_F(StatisticsTest, AveListOfDataTest) {
  std::vector<std::vector<double>> data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  std::vector<double> expectedAve = {4.0, 5.0, 6.0};
  std::vector<double> actualAve = gqpeps::AveListOfData(data);
  EXPECT_EQ(expectedAve, actualAve);
}

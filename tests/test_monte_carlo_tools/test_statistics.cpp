/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-07
*
* Description: QuantumLiquids/PEPS project. Unittests for Monte-Carlo Statistic functions.
*/

#include "gtest/gtest.h"
#include "qlpeps/monte_carlo_tools/statistics.h"

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
  int expected_mean = 3;
  int actual_mean = qlpeps::Mean(data);
  EXPECT_EQ(expected_mean, actual_mean);
}

TEST_F(StatisticsTest, StandardErrorTest) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float mean = qlpeps::Mean(data);
  float expected_std_error = std::sqrt(2.0f) / 2.0f;
  float actual_std_error = qlpeps::StandardError(data, mean);
  EXPECT_FLOAT_EQ(expected_std_error, actual_std_error);
}

TEST_F(StatisticsTest, AveListOfDataTest) {
  std::vector<std::vector<double>> data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  std::vector<double> expected_ave = {4.0, 5.0, 6.0};
  std::vector<double> actual_ave = qlpeps::AveListOfData(data);
  EXPECT_EQ(expected_ave, actual_ave);
}

TEST_F(StatisticsTest, MeanComplexTest) {
  std::vector<std::complex<double>> data = {
      std::complex<double>(1.0, 2.0),
      std::complex<double>(2.0, 3.0),
      std::complex<double>(3.0, 4.0)};
  std::complex<double> expected_mean(2.0, 3.0);
  std::complex<double> actual_mean = qlpeps::Mean(data);
  EXPECT_EQ(expected_mean, actual_mean);
  float expected_std_error = std::sqrt(2.0 / 3.0);
  float actual_std_error = qlpeps::StandardError(data, actual_mean);
  EXPECT_EQ(expected_std_error, actual_std_error);
}

TEST_F(StatisticsTest, AveListOfComplexDataTest) {
  std::vector<std::vector<std::complex<double>>> data = {
      {std::complex<double>(1.0, 2.0), std::complex<double>(2.0, 3.0), std::complex<double>(3.0, 4.0)},
      {std::complex<double>(4.0, 5.0), std::complex<double>(5.0, 6.0), std::complex<double>(6.0, 7.0)},
      {std::complex<double>(7.0, 8.0), std::complex<double>(8.0, 9.0), std::complex<double>(9.0, 10.0)}};
  std::vector<std::complex<double>> expected_ave = {
      std::complex<double>(4.0, 5.0), std::complex<double>(5.0, 6.0), std::complex<double>(6.0, 7.0)};
  std::vector<std::complex<double>> actual_ave = qlpeps::AveListOfData(data);
  EXPECT_EQ(expected_ave, actual_ave);
}


/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-15
*
* Description: GraceQ/VMC-PEPS project. Unittests for Non-detailed balance Markov-chain Monte-Carlo.
*/


#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "gqpeps/monte_carlo_tools/non_detailed_balance_mcmc.h"
#include "gqpeps/monte_carlo_tools/statistics.h"      //Mean

using namespace gqpeps;

// Potts Model Monte Carlo Simulation Class
class PottsModel {
 public:
  PottsModel(size_t size, size_t q, double temperature)
      : size_(size), q_(q), temperature_(temperature), spins_(size * size) {}

  void Initialize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, q_ - 1);

    for (size_t i = 0; i < size_ * size_; ++i) {
      spins_[i] = dis(gen);
    }
    energy_samples_.reserve(1000000);
  }

  void MonteCarloStep() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (size_t i = 0; i < size_ * size_; ++i) {
      size_t spin = spins_[i];
      std::vector<double> weights(q_, 0.0);

      for (size_t j = 0; j < q_; ++j) {
        spins_[i] = j;
        weights[j] = std::exp(-GetLocalE_(i) / temperature_);
//        weights[j] = std::exp(-CalculateEnergy_() / temperature_);
      }

      spins_[i] = NonDBMCMCStateUpdate(spin, weights, dis(gen));
    }
  }
  void MonteCarloSample() {
    energy_samples_.push_back(CalculateEnergy_());
  }

  double GetEnergy() const {
    return Mean(energy_samples_);
  }
  double GetEnergyErr(size_t bin_num) const {
    size_t data_size = energy_samples_.size();
    size_t bin_size = data_size / bin_num;

    std::vector<double> means(bin_num);

    // Calculate the mean for each bin
    for (size_t i = 0; i < bin_num; ++i) {
      size_t start_index = i * bin_size;
      size_t end_index = start_index + bin_size;
      std::vector<double> bin_data(energy_samples_.begin() + start_index,
                                   energy_samples_.begin() + end_index);
      means[i] = Mean(bin_data);
    }

    // Calculate the standard error using the means
    double energy_mean = Mean(energy_samples_);
    double energy_err = StandardError(means, energy_mean);

    return energy_err;
  }

  size_t GetSpin(size_t x, size_t y) const {
    return spins_[x * size_ + y];
  }

  double CalculateExactEnergy() const {
    double partition_function = 0.0;
    double energy_weighted_sum = 0.0;

    // Iterate over all possible spin configurations
    std::vector<size_t> configuration(size_ *size_);
    for (size_t i = 0; i < pow(q_, size_ * size_); ++i) {
      // Convert the decimal index 'i' to the corresponding spin configuration
      size_t value = i;
      for (size_t j = 0; j < size_ * size_; ++j) {
        configuration[j] = value % q_;
        value /= q_;
      }

      // Calculate the energy for the current spin configuration
      double energy = 0.0;
      for (size_t x = 0; x < size_; ++x) {
        for (size_t y = 0; y < size_; ++y) {
          size_t spin = configuration[x * size_ + y];
          size_t sum_neighbors = (configuration[((x + 1) % size_) * size_ + y] == spin) +
              (configuration[x * size_ + (y + 1) % size_] == spin);
          energy += (-1.0 * sum_neighbors);
        }
      }

      // Accumulate the partition function and energy weighted by Boltzmann factor
      double boltzmann_factor = std::exp(-energy / temperature_);
      partition_function += boltzmann_factor;
      energy_weighted_sum += energy * boltzmann_factor;
    }

    // Calculate the exact energy using the accumulated values
    return energy_weighted_sum / partition_function;
  }

 private:
  double GetLocalE_(const size_t site_index) const {
    size_t x = site_index / size_;
    size_t y = site_index % size_;
    size_t spin = spins_[site_index];
    size_t sum_neighbors = (GetSpin((x + size_ - 1) % size_, y) == spin) +
        (GetSpin((x + 1) % size_, y) == spin) +
        (GetSpin(x, (y + size_ - 1) % size_) == spin) +
        (GetSpin(x, (y + 1) % size_) == spin);

    return -1.0 * sum_neighbors;//ferromagnetic Potts model
  }

  double CalculateEnergy_() const {
    double energy = 0.0;
    for (size_t x = 0; x < size_; ++x) {
      for (size_t y = 0; y < size_; ++y) {
        size_t spin = GetSpin(x, y);
        size_t sum_neighbors = (GetSpin((x + 1) % size_, y) == spin) +
            (GetSpin(x, (y + 1) % size_) == spin);
        energy += (-1.0 * sum_neighbors);
      }
    }
    return energy;
  }

  size_t size_;  // Lattice size
  size_t q_;     // Number of spin states
  double temperature_;  // Temperature
  std::vector<size_t> spins_;
  std::vector<double> energy_samples_;
};

// Test Potts Model Monte Carlo Simulation
TEST(PottsModelTest, MonteCarloSimulationTest) {
  const size_t size = 3;
  const size_t q = 3;
  const double temperature = 1.0 / std::log(1 + std::sqrt((double) q));//critical point
  const size_t num_iterations = 1000000;

  PottsModel model(size, q, temperature);
  model.Initialize();

  for (size_t i = 0; i < num_iterations; ++i) {
    model.MonteCarloStep(); //warm up
  }

  for (size_t i = 0; i < num_iterations; ++i) {
    model.MonteCarloStep();
    model.MonteCarloSample();
  }

  // Add your assertions here to check the properties of the Potts model after the simulation
  double energy_ex = model.CalculateExactEnergy();
  EXPECT_NEAR(model.GetEnergy(), energy_ex, model.GetEnergyErr(10));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

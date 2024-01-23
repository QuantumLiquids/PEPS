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
#include "gqten/utility/timer.h"                                  //Timer
#include "gqpeps/monte_carlo_tools/non_detailed_balance_mcmc.h"
#include "gqpeps/monte_carlo_tools/statistics.h"                  //Mean

using gqten::Timer;
using namespace gqpeps;

void TestSingleModeNonDBMarkovChainDistribution(
    const std::vector<double> &weights,
    const size_t num_iterations
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  // Create a map to store the count of each state
  std::vector<size_t> state_counts(weights.size(), 0);

  // Generate the Markov chain and count the occurrences of each state
  size_t state = 0;
  for (int i = 0; i < num_iterations; ++i) {
    state = gqpeps::NonDBMCMCStateUpdate(state, weights, dis(gen));
    state_counts[state]++;
  }

  // Calculate the distribution of states in the Markov chain
  std::vector<double> state_distribution(weights.size(), 0.0);
  for (size_t i = 0; i < state_counts.size(); i++) {
    state_distribution[i] = static_cast<double>(state_counts[i]) / num_iterations;
  }

  // Verify if the generated distribution matches the initial weights
  for (size_t i = 0; i < weights.size(); ++i) {
    EXPECT_NEAR(state_distribution[i], weights[i], 1e-3);
  }
}

TEST(NonDBMCMCTest, SingleModeMarkovChainDistribution) {
  TestSingleModeNonDBMarkovChainDistribution({0.4, 0.35, 0.25}, 1000000);
  TestSingleModeNonDBMarkovChainDistribution({0.01, 0.1, 0.6, 0.29}, 1000000);
}

///< Potts Model Monte Carlo Simulation Class
///< E = \sum_{<i,j>} [1 - \delta_{sigma_i, sigma_j}]
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

  void MonteCarloSweep() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (size_t i = 0; i < size_ * size_; ++i) {
      const size_t spin = spins_[i];
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
  // another idea is use transfer matrix
  double CalculateExactEnergy() {
    p_n_ = std::vector<size_t>(2 * size_ * size_ + 1, 0);
    // Iterate over all possible spin configurations
    std::vector<size_t> configuration(size_ *size_);
    const size_t total_num_configurations = pow(q_, size_ * size_ - 1);
    for (size_t i = 0; i < total_num_configurations; ++i) {//fix last site spin = 0  by Zq symmetry
      // Convert the decimal index 'i' to the corresponding spin configuration
      size_t value = i;
      for (size_t j = 0; j < size_ * size_; ++j) {
        configuration[j] = value % q_;
        value /= q_;
      }

      // Calculate the energy for the current spin configuration
      size_t energy = 2 * size_ * size_; // note the energy definition is >= 0
      for (size_t x = 0; x < size_; ++x) {
        for (size_t y = 0; y < size_; ++y) {
          size_t spin = configuration[x * size_ + y];
          size_t sum_neighbors = (configuration[((x + 1) % size_) * size_ + y] == spin) +
              (configuration[x * size_ + (y + 1) % size_] == spin);
          energy -= sum_neighbors;
        }
      }
      p_n_[energy] += 1;
    }
    for (auto &pn : p_n_) {
      pn *= q_; // less count by symmetry
    }

    // Accumulate the partition function and energy weighted by Boltzmann factor
    double partition_function = 0.0;
    double energy_weighted_sum = 0.0;
    double u = std::exp(-1.0 / temperature_);
    for (int energy = p_n_.size() - 1; energy >= 0; energy--) {
      double boltzmann_factor = pow(u, energy);
      //double boltzmann_factor = std::exp(-energy / temperature_);
      partition_function += p_n_[energy] * boltzmann_factor;
      energy_weighted_sum += energy * p_n_[energy] * boltzmann_factor;
    }

    // Calculate the exact energy using the accumulated values
    return energy_weighted_sum / partition_function;
  }

  ///< https://arxiv.org/pdf/hep-lat/9301017.pdf
  double CalLowTemperatureExpansionEnergy() const {
    double u = std::exp(-1.0 / temperature_);
    double e = 0;
    for (int j = en_per_site_q3_low_temperature_expansion_coeff_.size() - 1; j >= 0; j--) {
      e += en_per_site_q3_low_temperature_expansion_coeff_[j] * pow(u, j);
    }
    return e;
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
    double energy = 2 * size_ * size_;
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

  const size_t size_;  // Lattice size
  const size_t q_;     // Number of spin states
  const double temperature_;  // Temperature
  std::vector<size_t> spins_;
  std::vector<double> energy_samples_;

  std::vector<size_t> p_n_; //partition function polynomial coefficient
  const static std::vector<long long> en_per_site_q3_low_temperature_expansion_coeff_;
};

///< https://arxiv.org/pdf/hep-lat/9301017.pdf, it's an approxmation for thermal dynamic limit so it may not good to used as a benchmark here.
const std::vector<long long>PottsModel::en_per_site_q3_low_temperature_expansion_coeff_ =
    {0, 0, 0, 0, 8, 0, 24, 28, 32, 216, 160, 660, 2072, 1664, 11760, 17700, 41088, 156468, 207240, 849300, 1817048,
     4021780, 13178264, 25754296, 75653408, 193458400, 440725376, 1296485460, 3009317200, 7977739920, 21217637824,
     51359965976, 140885970816, 354038121756, 916153258448, 2439917838708, 6161990034800, 16397314674708,
     42540620667584, 110314458936968, 292427669006272, 756553239055504, 1994873374110312, 5238354130103568,
     136864019707177088, 36195015152016276};

// Test Potts Model Monte Carlo Simulation
TEST(PottsModelTest, MonteCarloSimulationTest) {
  const double energy_q3_4x4_ex = 4.88543;
  const size_t size = 4;
  const size_t q = 3;
  const double temperature = std::log(1 + std::sqrt((double) q));//critical point
  const size_t num_iterations = 200000;

  PottsModel model(size, q, temperature);
  model.Initialize();

  Timer exact_summation_timer("exact summation");
//  double energy_ex = model.CalculateExactEnergy();
  double energy_ex = energy_q3_4x4_ex;
  exact_summation_timer.PrintElapsed();

  Timer monte_carlo_timer("monte carlo");
  for (size_t i = 0; i < 1000; ++i) {
    model.MonteCarloSweep(); //warm up
  }

  for (size_t i = 0; i < num_iterations; ++i) {
    model.MonteCarloSweep();
    model.MonteCarloSample();
  }
  monte_carlo_timer.PrintElapsed();

  EXPECT_NEAR(model.GetEnergy(), energy_ex, model.GetEnergyErr(10));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


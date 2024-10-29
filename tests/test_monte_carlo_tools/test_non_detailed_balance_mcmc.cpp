/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-15
*
* Description: QuantumLiquids/PEPS project. Unittests for Non-detailed balance Markov-chain Monte-Carlo.
*/


#include <gtest/gtest.h>
#include <random>
#include "qlten/utility/timer.h"                                  //Timer
#include "qlpeps/monte_carlo_tools/non_detailed_balance_mcmc.h"
#include "qlpeps/monte_carlo_tools/statistics.h"                  //Mean

using qlten::Timer;
using namespace qlpeps;

void TestSingleModeNonDBMarkovChainDistribution(
    const std::vector<double> &weights,
    const size_t num_iterations,
    const size_t init_state = 0
) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // Create a map to store the count of each state
  std::vector<size_t> state_counts(weights.size(), 0);

  // Generate the Markov chain and count the occurrences of each state
  size_t state = init_state;
  for (int i = 0; i < num_iterations; ++i) {
    state = qlpeps::NonDBMCMCStateUpdate(state, weights, gen);
    state_counts[state]++;
  }

  // Calculate the distribution of states in the Markov chain
  std::vector<double> state_distribution(weights.size(), 0.0);
  for (size_t i = 0; i < state_counts.size(); i++) {
    state_distribution[i] = static_cast<double>(state_counts[i]) / static_cast<double>(num_iterations);
  }

  // Verify if the generated distribution matches the initial weights
  double weight_sum = 0.0;
  for (auto w : weights) {
    weight_sum += w;
  }
  size_t effective_sample_size = num_iterations;  // assume no autocorrelation
  for (size_t i = 0; i < weights.size(); ++i) {
    double pi = weights[i] / weight_sum;
    double std_error = std::sqrt(pi * (1 - pi) / static_cast<double>(effective_sample_size));
    double tolerance = 3 * std_error; // ~99.7% confidence interval, Central limit theorem
    EXPECT_NEAR(state_distribution.at(i), pi, tolerance);
  }
}

TEST(NonDBMCMCTest, SingleModeMarkovChainDistribution) {
  TestSingleModeNonDBMarkovChainDistribution({1, 0.5, 0.3, 0.01, 0.06, 2}, 1e6);
  TestSingleModeNonDBMarkovChainDistribution({1, 0.3, 1e-30}, 1e6, 0);
  TestSingleModeNonDBMarkovChainDistribution({9.6, 9.6, 1}, 1e6, 1);
}

///< Potts Model Monte Carlo Simulation Class
///< E = \sum_{<i,j>} [1 - \delta_{sigma_i, sigma_j}]
class PottsModel {
 public:
  PottsModel(size_t size, size_t q, double temperature)
      : size_(size), q_(q), temperature_(temperature), spins_(size * size) {
    Initialize();
  }

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

    for (size_t i = 0; i < size_ * size_; ++i) {
      const size_t spin = spins_[i];
      std::vector<double> weights(q_, 0.0);

      for (size_t j = 0; j < q_; ++j) {
        spins_[i] = j;
        weights[j] = std::exp(-GetLocalE_(i) / temperature_);
      }

      spins_[i] = NonDBMCMCStateUpdate(spin, weights, gen);
    }
  }
  void MonteCarloSample() {
    energy_samples_.push_back(CalculateEnergy_());
  }

  [[nodiscard]] double GetEnergy() const {
    return Mean(energy_samples_);
  }
  [[nodiscard]] double GetEnergyErr(size_t bin_num) const {
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
};

// Test Potts Model Monte Carlo Simulation
TEST(PottsModelTest, MonteCarloSimulationTest) {
  const double energy_q3_4x4_ex = 4.88543;
  const size_t size = 4;
  const size_t q = 3;
  const double temperature = std::log(1 + std::sqrt((double) q));//critical point
  const size_t num_iterations = 200000;

  PottsModel model(size, q, temperature);

  Timer exact_summation_timer("exact summation");
//  double energy_ex = model.CalculateExactEnergy();
  double energy_ex = energy_q3_4x4_ex;
  exact_summation_timer.PrintElapsed();

  Timer monte_carlo_timer("monte carlo");
  for (size_t i = 0; i < num_iterations; ++i) {
    model.MonteCarloSweep(); //warm up
  }

  for (size_t i = 0; i < num_iterations; ++i) {
    model.MonteCarloSweep();
    model.MonteCarloSample();
  }
  monte_carlo_timer.PrintElapsed();

  EXPECT_NEAR(model.GetEnergy(), energy_ex, model.GetEnergyErr(10));
}

///< Potts Model Monte Carlo Simulation with Exchange Dynamics and OBC
///< Monte Carlo update only exchanges spins between neighboring sites
class ConservedPottsModel {
 public:
  ConservedPottsModel(size_t size, size_t q, double temperature)
      : size_(size), q_(q), temperature_(temperature), spins_(size * size),
        gen(rd()), dis(0, 1.0) {
    // Calculate the total number of spins
    size_t total_spins = size_ * size_;

    // Calculate the number of spins for each state
    size_t spins_per_state = total_spins / q_;
    size_t remaining_spins = total_spins % q_; // Remainder spins to distribute

    // Initialize the spins
    size_t index = 0;
    for (size_t state = 0; state < q_; ++state) {
      // Assign the spins for the current state
      for (size_t count = 0; count < spins_per_state; ++count) {
        if (index < total_spins) {
          spins_[index++] = state;
        }
      }
      // Distribute any remaining spins among the states
      if (remaining_spins > 0) {
        if (index < total_spins) {
          spins_[index++] = state;
          remaining_spins--;
        }
      }
    }

    // Shuffle the spins array to randomize the configuration
    std::shuffle(spins_.begin(), spins_.end(), gen);

    energy_samples_.reserve(1000000);
  }
  void MonteCarloSweep() {
    std::uniform_int_distribution<size_t> dis_site(0, size_ * size_ - 1);
    for (size_t i = 0; i < size_ * size_; ++i) {
      // Pick two neighboring sites randomly
      size_t site1 = i;
      size_t site2 = dis_site(gen);

      // If no valid neighbor, skip the exchange
      if (site2 == site1) {
        continue;
      }

      // Exchange spins between site1 and site2
      size_t spin1 = spins_[site1];
      size_t spin2 = spins_[site2];
      if (spin1 == spin2) {
        continue;
      }

      // Compute energy difference if spins are exchanged
      double energy_old = CalculateEnergy_();
      std::swap(spins_[site1], spins_[site2]);
      double energy_new = CalculateEnergy_();
      double deltaE = energy_new - energy_old;
      // Metropolis acceptance criterion
      if (deltaE < 0 || dis(gen) < std::exp(-deltaE / temperature_)) {
      } else {
        std::swap(spins_[site1], spins_[site2]);
      }
    }
  }

  void MonteCarloSweepNN() {
    for (size_t i = 0; i < size_ * size_; ++i) {
      // Pick two neighboring sites randomly
      size_t site1 = i;
      size_t site2 = GetRandomNeighbor_(site1, gen);

      // Exchange spins between site1 and site2
      size_t spin1 = spins_[site1];
      size_t spin2 = spins_[site2];
      if (spin1 == spin2) {
        continue;
      }

      // Compute energy difference if spins are exchanged
      double energy_old = CalculateEnergy_();
      std::swap(spins_[site1], spins_[site2]);
      double energy_new = CalculateEnergy_();
      double deltaE = energy_new - energy_old;
      // Metropolis acceptance criterion
      if (deltaE < 0 || dis(gen) < std::exp(-deltaE / temperature_)) {
      } else {
        std::swap(spins_[site1], spins_[site2]);
      }
    }
  }

  void SequentialExchange() {
    // Iterate over the lattice
    for (size_t x = 0; x < size_; ++x) {
      for (size_t y = 0; y < size_ - 1; ++y) { // horizontal pairs
        size_t site1 = x * size_ + y;
        size_t site2 = x * size_ + (y + 1);
        ExchangeNNUpdate(site1, site2, dis(gen));
      }
    }

    for (size_t x = 0; x < size_ - 1; ++x) {
      for (size_t y = 0; y < size_; ++y) { // vertical pairs
        size_t site1 = x * size_ + y;
        size_t site2 = (x + 1) * size_ + y;
        ExchangeNNUpdate(site1, site2, dis(gen));
      }
    }
  }

  void MonteCarloSweep3() {
    // Iterate through the lattice to randomly select connected triplets of sites
    for (size_t i = 0; i < 2 * size_ * size_; ++i) {
      // Randomly choose a starting site
      size_t row = rand() % size_;
      size_t col = rand() % size_;
      size_t site1 = row * size_ + col;

      if (dis(gen) < 0.5) {
        // Randomly decide whether to choose a horizontal or vertical triplet
        if (col < size_ - 2) {  // Ensure we don't go out of bounds for horizontal
          // Randomly select to either use horizontal or vertical exchange
          // Choose three horizontal connected sites
          Exchange3Update(site1, site1 + 1, site1 + 2);
        }
      } else if (row < size_ - 2) { // Ensure we don't go out of bounds for vertical
        // Choose three vertical connected sites
        Exchange3Update(site1, site1 + size_, site1 + 2 * size_);
      }
    }
  }

  void SequentialExchange3() {
    for (size_t row = 0; row < size_; row++) {
      for (size_t col = 0; col < size_ - 2; col++) {
        size_t site1 = row * size_ + col;
        Exchange3Update(site1, site1 + 1, site1 + 2);
      }
    }

    for (size_t row = 0; row < size_ - 2; row++) {
      for (size_t col = 0; col < size_; col++) {
        size_t site1 = row * size_ + col;
        Exchange3Update(site1, site1 + size_, site1 + 2 * size_);
      }
    }
  }

  bool Exchange3Update(size_t site1, size_t site2, size_t site3) {
    // Get the spins at the three sites
    size_t spin1 = spins_[site1];
    size_t spin2 = spins_[site2];
    size_t spin3 = spins_[site3];

    std::vector<size_t> spins = {spin1, spin2, spin3};

    // If all spins are the same, no update is needed
    if (spin1 == spin2 && spin2 == spin3) {
      return false;
    }

    // Sort the spins vector to ensure lexicographic order of permutations
    std::sort(spins.begin(), spins.end());

    // Generate all permutations of the spins in sorted (lexicographic) order
    std::vector<std::vector<size_t>> permutations;

    do {
      permutations.push_back(spins);
    } while (std::next_permutation(spins.begin(), spins.end()));

//    std::sort(permutations.begin(), permutations.end());
    // Now find the initial configuration (init_state) in the sorted permutation list
    std::vector<size_t> initial_spins = {spin1, spin2, spin3};
    size_t init_state =
        std::distance(permutations.begin(), std::find(permutations.begin(), permutations.end(), initial_spins));

    // Prepare a vector to store the weights of each permutation
    std::vector<double> weights(permutations.size());

    // Calculate the energy for each permutation and store the weight
    for (size_t i = 0; i < permutations.size(); ++i) {
      // Apply the i-th permutation
      spins_[site1] = permutations[i][0];
      spins_[site2] = permutations[i][1];
      spins_[site3] = permutations[i][2];

      // Calculate energy and weight
      double E = CalculateEnergy_();
      weights[i] = std::exp(-E / temperature_);
    }

    // Call NonDBMCMCStateUpdate to select the final state
    size_t final_state = NonDBMCMCStateUpdate(init_state, weights, gen);

    // set spins_, even if final_state = init_state
    spins_[site1] = permutations[final_state][0];
    spins_[site2] = permutations[final_state][1];
    spins_[site3] = permutations[final_state][2];
    // If the final state is the same as the initial state, return false
    if (final_state == init_state) {
      return false;
    }
    return true;
  }

  bool ExchangeNNUpdate(size_t site1, size_t site2, double rand_num) {
    size_t spin1 = spins_[site1];
    size_t spin2 = spins_[site2];

    // Ensure spins are different for exchange
    if (spin1 == spin2) {
      return false; // No exchange if spins are the same
    }

    // Calculate the energy change
    double energy_old = CalculateEnergy_();
    std::swap(spins_[site1], spins_[site2]);
    double energy_new = CalculateEnergy_();
    double deltaE = energy_new - energy_old;
    // Metropolis acceptance criterion
    if (deltaE < 0 || rand_num < std::exp(-deltaE / temperature_)) {
      return true;
    } else {
      std::swap(spins_[site1], spins_[site2]);
    }
    return false; // Exchange not performed
  }

  void MonteCarloSample() {
    energy_samples_.push_back(CalculateEnergy_());
  }

  [[nodiscard]] double GetEnergy() const {
    return Mean(energy_samples_);
  }

  [[nodiscard]] double GetEnergyErr(size_t bin_num) const {
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

  void ClearSamples() {
    energy_samples_.clear();
//    energy_samples_.shrink_to_fit();  // Optional: To release allocated memory
  }

  std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<double> dis;
 private:
  size_t GetRandomNeighbor_(size_t site, std::mt19937 &gen) const {
    size_t x = site / size_;
    size_t y = site % size_;

    // Define potential neighbors based on OBC
    std::vector<size_t> neighbors;

    if (x > 0) {  // Up neighbor
      neighbors.push_back((x - 1) * size_ + y);
    }
    if (x < size_ - 1) {  // Down neighbor
      neighbors.push_back((x + 1) * size_ + y);
    }
    if (y > 0) {  // Left neighbor
      neighbors.push_back(x * size_ + (y - 1));
    }
    if (y < size_ - 1) {  // Right neighbor
      neighbors.push_back(x * size_ + (y + 1));
    }

    // If no neighbors (shouldn't happen), return the same site
    if (neighbors.empty()) {
      std::cout << "no neighbor expected." << std::endl;
      exit(1);
    }

    // Randomly choose a neighbor
    std::uniform_int_distribution<size_t> int_dis(0, neighbors.size() - 1);
    return neighbors[int_dis(gen)];
  }

  [[nodiscard]] double GetEnergyDiff_(size_t site, size_t new_spin, size_t old_spin) const {
    size_t x = site / size_;
    size_t y = site % size_;

    // Calculate energy contribution from neighbors based on OBC
    int sum_neighbors_old = 0;
    int sum_neighbors_new = 0;

    // Check Up
    if (x > 0) {
      sum_neighbors_old += (GetSpin(x - 1, y) == old_spin);
      sum_neighbors_new += (GetSpin(x - 1, y) == new_spin);
    }
    // Check Down
    if (x < size_ - 1) {
      sum_neighbors_old += (GetSpin(x + 1, y) == old_spin);
      sum_neighbors_new += (GetSpin(x + 1, y) == new_spin);
    }
    // Check Left
    if (y > 0) {
      sum_neighbors_old += (GetSpin(x, y - 1) == old_spin);
      sum_neighbors_new += (GetSpin(x, y - 1) == new_spin);
    }
    // Check Right
    if (y < size_ - 1) {
      sum_neighbors_old += (GetSpin(x, y + 1) == old_spin);
      sum_neighbors_new += (GetSpin(x, y + 1) == new_spin);
    }

    assert(std::abs(sum_neighbors_new - sum_neighbors_old) < 3);
    // Energy change from swapping the spins
    return -1.0 * (sum_neighbors_new - sum_neighbors_old);
  }

  [[nodiscard]] double CalculateEnergy_() const {
    size_t sum_neighbors = 0;
    for (size_t x = 0; x < size_; ++x) {
      for (size_t y = 0; y < size_; ++y) {
        size_t spin = GetSpin(x, y);
        // Only consider neighbors within OBC limits
        if (x < size_ - 1) {
          sum_neighbors += (GetSpin(x + 1, y) == spin);
        }
        if (y < size_ - 1) {
          sum_neighbors += (GetSpin(x, y + 1) == spin);
        }
      }
    }
    double energy = (-1.0 * double(sum_neighbors));
    return energy;
  }

  const size_t size_;  // Lattice size
  const size_t q_;     // Number of spin states
  const double temperature_;  // Temperature
  std::vector<size_t> spins_;
  std::vector<double> energy_samples_;
};

TEST(ConservedPottsModelTest, MonteCarloSimulationTest) {
  const size_t size = 4;
  const size_t q = 3;
  const double temperature = std::log(1 + std::sqrt((double) q));//critical point
  const size_t num_iterations = 500000;

  ConservedPottsModel model(size, q, temperature);

  Timer monte_carlo_timer("monte carlo");
  for (size_t i = 0; i < num_iterations; ++i) {
    model.MonteCarloSweep(); //warm up
  }

  for (size_t i = 0; i < num_iterations; ++i) {
    model.SequentialExchange();
    model.MonteCarloSample();
  }
  monte_carlo_timer.PrintElapsed();
  double energy1 = model.GetEnergy();
  double en_err1 = model.GetEnergyErr(10);

  model.ClearSamples();

  for (size_t i = 0; i < num_iterations; ++i) {
    model.SequentialExchange3();
    model.SequentialExchange3();
    model.MonteCarloSample();
  }
  double energy2 = model.GetEnergy();
  double en_err2 = model.GetEnergyErr(10);
  EXPECT_NEAR(energy1, energy2, en_err1 + en_err2);

  model.ClearSamples();

  for (size_t i = 0; i < num_iterations; ++i) {
    model.MonteCarloSweep3();
    model.MonteCarloSample();
  }
  double energy3 = model.GetEnergy();
  double en_err3 = model.GetEnergyErr(10);

  EXPECT_NEAR(energy1, energy3, en_err1 + en_err3);
}

int FindDifferentSpin(const std::array<bool, 3> &spins) {
  if (spins[0] == spins[1] && spins[0] != spins[2]) {
    return 2;  // spin3 (index 2) is different
  } else if (spins[0] == spins[2] && spins[0] != spins[1]) {
    return 1;  // spin2 (index 1) is different
  } else if (spins[1] == spins[2] && spins[1] != spins[0]) {
    return 0;  // spin1 (index 0) is different
  }

  return -1;  // If all spins are the same (or there's an error)
}

class ConservedIsingModel {
 public:
  ConservedIsingModel(size_t size, double temperature)
      : size_(size), temperature_(temperature), spins_(size, false),
        gen(rd()), dis(0, 1.0) {

    size_t total_spins = size_;

    // Set half spins to up (true), half to down (false)
    size_t up_spins = total_spins / 2;

    // Initialize spins: first half up, rest down
    for (size_t i = 0; i < up_spins; ++i) {
      spins_[i] = true;
    }

    // Shuffle the spin configuration to randomize
    std::shuffle(spins_.begin(), spins_.end(), gen);

    energy_samples_.reserve(10000000);
  }

  void MonteCarloSweep() {
    std::uniform_int_distribution<size_t> dis_site(0, size_ - 1);
    for (size_t i = 0; i < size_; ++i) {
      // Pick two random sites for exchange
      size_t site1 = dis_site(gen);
      size_t site2 = dis_site(gen);

      // If the spins are the same, skip exchange
      if (spins_[site1] == spins_[site2]) {
        continue;
      }

      // Compute the energy difference if spins are exchanged
      double energy_old = CalculateEnergy_();
      std::swap(spins_[site1], spins_[site2]);
      double energy_new = CalculateEnergy_();
      double deltaE = energy_new - energy_old;

      // Metropolis acceptance criterion
      if (deltaE < 0 || dis(gen) < std::exp(-deltaE / temperature_)) {
        // Accept the swap (already done)
      } else {
        // Reject the swap, revert back
        std::swap(spins_[site1], spins_[site2]);
      }
    }
  }

  void MonteCarloSweep3() {
    std::uniform_int_distribution<size_t> dis_site(0, size_ - 1);
    for (size_t i = 0; i < size_; ++i) {
      // Pick two random sites for exchange
      size_t site1 = i;
      size_t site2 = dis_site(gen);
      size_t site3 = dis_site(gen);

      // If the sites are the same, skip exchange
      if (site1 == site2 || site2 == site3 || site1 == site3) {
        continue;
      }
      Exchange3Update(site1, site2, site3);
    }
  }

  void SequentialExchange3() {
    for (size_t row = 0; row < size_; row++) {
      size_t site1 = row;
      size_t site2 = (row + 1) % size_;
      size_t site3 = (row + 2) % size_;
      Exchange3Update(site1, site2, site3);
    }
  }

  double CalculateExactEnergy() {
    p_n_ = std::vector<size_t>(2 * size_ + 1, 0);
    size_t total_spins = size_;
    std::cout << "size : " << size_ << std::endl;

    // Iterate over all configurations with the same number of up spins
    std::vector<char> configuration(total_spins, '-');
    size_t up_spins = total_spins / 2;
    std::fill(configuration.begin(), configuration.begin() + up_spins, '+');

    // Permute the configurations to generate all possible ones
    size_t config_count = 0;
    do {
      // Calculate the energy of the current configuration
      size_t energy = CalculateEnergyAbsInt(configuration);
      p_n_.at(energy) += 1;
      config_count++;
    } while (std::next_permutation(configuration.begin(), configuration.end()));
    std::cout << "total config:" << config_count << std::endl;

    // Accumulate the partition function and energy weighted by the Boltzmann factor
    double partition_function = 0.0;
    double energy_weighted_sum = 0.0;
    double u = std::exp(1.0 / temperature_);
    for (size_t energy = 0; energy < p_n_.size(); energy++) {
      double boltzmann_factor = std::pow(u, energy);
//      std::cout << "(pn, boltzmann_factor) : " << p_n_[energy] << "," << boltzmann_factor << std::endl;
      partition_function += p_n_[energy] * boltzmann_factor;
      energy_weighted_sum += energy * p_n_[energy] * boltzmann_factor;
    }

    // Calculate the exact energy
    return -energy_weighted_sum / partition_function;
  }

  void MonteCarloSample() {
    energy_samples_.push_back(CalculateEnergy_());
  }

  [[nodiscard]] double GetEnergy() const {
    return Mean(energy_samples_);
  }

  [[nodiscard]] double GetEnergyErr(size_t bin_num) const {
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

  size_t GetSpin(size_t x) const {
    return spins_[x];
  }

  void ClearSamples() {
    energy_samples_.clear();
  }

  std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<double> dis;

 private:

  bool Exchange3Update(size_t site1, size_t site2, size_t site3) {
    std::array<bool, 3> local_spins = {spins_[site1], spins_[site2], spins_[site3]};

    // If all spins are the same, no update is needed
    if (spins_[site1] == spins_[site2] && spins_[site2] == spins_[site3]) {
      return false;
    }

    std::vector<double> weights(3);
    size_t init_state = FindDifferentSpin(local_spins);
    // Default weight for no change
    weights[init_state] = std::exp(-CalculateEnergy_() / temperature_);;

    std::swap(spins_[site1], spins_[site2]);
    std::swap(spins_[site2], spins_[site3]);

    // Calculate energy differences for each possible configuration
    weights[(init_state + 2) % 3] = std::exp(-CalculateEnergy_() / temperature_);

    std::swap(spins_[site1], spins_[site2]);
    std::swap(spins_[site2], spins_[site3]);
    weights[(init_state + 1) % 3] = std::exp(-CalculateEnergy_() / temperature_);

    std::swap(spins_[site1], spins_[site2]);
    std::swap(spins_[site2], spins_[site3]);//back to original

    // Choose the final state using NonDBMCMCStateUpdate
    size_t final_state = NonDBMCMCStateUpdate(init_state, weights, gen);
//    size_t final_state = DBMCMCStateUpdate(0, weights,gen);
    if (int(spins_[site1]) + int(spins_[site2]) + int(spins_[site3]) == 1) {
      //1 true, 2 false
      switch (final_state) {
        case 0: {
          spins_[site1] = true;
          spins_[site2] = false;
          spins_[site3] = false;
          return true;
        }
        case 1: {
          spins_[site1] = false;
          spins_[site2] = true;
          spins_[site3] = false;
          return true;
        }
        case 2: {
          spins_[site1] = false;
          spins_[site2] = false;
          spins_[site3] = true;
          return true;
        }
        default: {
          std::cerr << "Unexpected final state in Exchange3Update." << std::endl;
          exit(1);
        }
      }
    } else {
      // 2 true, 1 false
      switch (final_state) {
        case 0: {
          spins_[site1] = false;
          spins_[site2] = true;
          spins_[site3] = true;
          return true;
        }
        case 1: {
          spins_[site1] = true;
          spins_[site2] = false;
          spins_[site3] = true;
          return true;
        }
        case 2: {
          spins_[site1] = true;
          spins_[site2] = true;
          spins_[site3] = false;
          return true;
        }
        default: {
          std::cerr << "Unexpected final state in Exchange3Update." << std::endl;
          exit(1);
        }
      }
    }
    // Perform the corresponding update based on the final state

  }

  [[nodiscard]] double CalculateEnergy_() const {
    size_t sum_neighbors = 0;
    for (size_t y = 0; y < size_; ++y) {
      bool spin = spins_[y];
      sum_neighbors += (spins_[(y + 1) % size_] == spin);
    }
    return (-1.0 * double(sum_neighbors));
  }
  // for ED
  [[nodiscard]] size_t CalculateEnergyAbsInt(std::vector<char> spins) const {
    size_t sum_neighbors = 0;
    for (size_t x = 0; x < size_; ++x) {
      char spin = spins[x];
//	if(x<size_-1) // for OBC
      sum_neighbors += (spins[(x + 1) % size_] == spin);
    }
    return sum_neighbors;
  }

  const size_t size_;
  const double temperature_;
  std::vector<bool> spins_;
  std::vector<double> energy_samples_;
  std::vector<size_t> p_n_;  // Energy probability density
};

TEST(ConservedIsingModelTEST, 1D) {
  const size_t size = 10;
  const double temperature = std::log(1 + std::sqrt((double) 2));//critical point
  const size_t num_iterations = 100000;

  ConservedIsingModel model(size, temperature);

  double e_ex = model.CalculateExactEnergy();
  std::cout << "e_ex : " << e_ex << std::endl;

  for (size_t i = 0; i < num_iterations; ++i) {
    model.MonteCarloSweep(); //warm up
  }

  for (size_t i = 0; i < num_iterations; ++i) {
    model.MonteCarloSweep();
    model.MonteCarloSweep();
    model.MonteCarloSample();
  }
  double energy0 = model.GetEnergy();
  double en_err0 = model.GetEnergyErr(10);

  std::cout << "e0 : " << energy0 << " pm " << en_err0 << std::endl;

  model.ClearSamples();

  for (size_t i = 0; i < num_iterations; ++i) {
    model.SequentialExchange3();
    model.MonteCarloSweep();
    model.MonteCarloSample();
  }
  double energy2 = model.GetEnergy();
  double en_err2 = model.GetEnergyErr(30);
  std::cout << "e2 : " << energy2 << " pm " << en_err2 << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


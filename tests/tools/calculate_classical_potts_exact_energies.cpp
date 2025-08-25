/*
* Standalone program to pre-calculate exact energies for Monte Carlo tests
* Run this once to generate reference values, then use results in unit tests
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// Copy the PottsModel class energy calculation logic
class ExactEnergyCalculator {
public:
    ExactEnergyCalculator(size_t size, size_t q, double temperature)
        : size_(size), q_(q), temperature_(temperature) {}

    double CalculateExactEnergy() {
        p_n_ = std::vector<size_t>(2 * size_ * size_ + 1, 0);
        
        // Iterate over all possible spin configurations
        std::vector<size_t> configuration(size_ * size_);
        const size_t total_num_configurations = std::pow(q_, size_ * size_ - 1);
        
        std::cout << "Calculating exact energy for " << size_ << "x" << size_ 
                  << " system with q=" << q_ << " states..." << std::endl;
        std::cout << "Total configurations: " << total_num_configurations << std::endl;
        
        for (size_t i = 0; i < total_num_configurations; ++i) {
            if (i % 100000 == 0) {
                std::cout << "Progress: " << (100.0 * i / total_num_configurations) << "%\r" << std::flush;
            }
            
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
            double boltzmann_factor = std::pow(u, energy);
            partition_function += p_n_[energy] * boltzmann_factor;
            energy_weighted_sum += energy * p_n_[energy] * boltzmann_factor;
        }

        std::cout << "\nPartition function: " << partition_function << std::endl;
        return energy_weighted_sum / partition_function;
    }

private:
    const size_t size_;
    const size_t q_;
    const double temperature_;
    std::vector<size_t> p_n_;
};

int main() {
    std::cout << std::fixed << std::setprecision(8);
    
    // Calculate critical temperature for q=3 Potts model
    const size_t q = 3;
    const double T_critical = std::log(1 + std::sqrt((double)q));
    
    std::cout << "=== Potts Model Exact Energy Calculator ===" << std::endl;
    std::cout << "q = " << q << std::endl;
    std::cout << "Critical temperature T_c = " << T_critical << std::endl;
    std::cout << std::endl;

    // Calculate energies for different system sizes and temperatures
    struct CalculationParams {
        size_t size;
        double temp_factor;  // multiplier of T_critical
        std::string description;
    };
    
    std::vector<CalculationParams> calculations = {
        {3, 0.3, "3x3, T = 0.3*T_c (low temp)"},
        {3, 0.7, "3x3, T = 0.7*T_c (near critical)"},
        {3, 1.5, "3x3, T = 1.5*T_c (high temp)"},
        {4, 0.3, "4x4, T = 0.3*T_c (low temp)"},
        {4, 0.7, "4x4, T = 0.7*T_c (near critical)"},
        {4, 1.0, "4x4, T = T_c (critical point)"},
        {4, 1.5, "4x4, T = 1.5*T_c (high temp)"}
    };

    std::ofstream results_file("potts_exact_energies.txt");
    if (!results_file.is_open()) {
        std::cerr << "Error: Failed to open output file 'potts_exact_energies.txt'" << std::endl;
        return 1;
    }
    results_file << std::fixed << std::setprecision(8);
    if (results_file.fail()) {
        std::cerr << "Error: Failed to set precision for output file" << std::endl;
        return 1;
    }
    results_file << "# Exact energies for q=" << q << " Potts model\n";
    results_file << "# T_critical = " << T_critical << "\n";
    results_file << "# Format: size temp_factor temperature exact_energy\n";
    if (results_file.fail()) {
        std::cerr << "Error: Failed to write header to output file" << std::endl;
        return 1;
    }

    for (const auto& calc : calculations) {
        double temperature = calc.temp_factor * T_critical;
        
        std::cout << "\n" << calc.description << std::endl;
        std::cout << "Temperature = " << temperature << std::endl;
        
        if (calc.size <= 4) {  // Only calculate for reasonable sizes
            ExactEnergyCalculator calculator(calc.size, q, temperature);
            double exact_energy = calculator.CalculateExactEnergy();
            
            std::cout << "Exact energy = " << exact_energy << std::endl;
            
            // Write to file
            results_file << calc.size << " " << calc.temp_factor << " " 
                        << temperature << " " << exact_energy << "\n";
            if (results_file.fail()) {
                std::cerr << "Error: Failed to write results to output file" << std::endl;
                return 1;
            }
        } else {
            std::cout << "Skipping (too large for exact calculation)" << std::endl;
        }
    }
    
    results_file.close();
    if (results_file.fail()) {
        std::cerr << "Error: Failed to close output file properly" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Results Summary ===" << std::endl;
    std::cout << "Results saved to 'potts_exact_energies.txt'" << std::endl;
    std::cout << "Use these values in your unit tests for fast, accurate comparisons." << std::endl;
    
    return 0;
}
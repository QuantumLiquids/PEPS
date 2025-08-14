# Monte Carlo Testing Best Practices Analysis

## Current Issues in `test_non_detailed_balance_mcmc.cpp`

### Problems Identified:
1. **Insufficient sampling** - 200k iterations may not be enough at critical temperature
2. **Poor error estimation** - Only 10 bins for error analysis is inadequate
3. **No autocorrelation analysis** - Error bars may be underestimated
4. **Testing at critical point** - Maximizes fluctuations and autocorrelation times

## Recommended Monte Carlo Testing Strategy

### 1. **Multi-Temperature Testing**
Instead of testing only at critical temperature, test at multiple temperatures:
- Well below critical temperature (fast convergence)
- At critical temperature (if necessary)
- Well above critical temperature (fast convergence)

### 2. **Autocorrelation Time Measurement**
```cpp
// Measure autocorrelation time for energy
std::vector<double> energy_series = model.GetEnergySeries();
double tau_int = MeasureAutocorrelationTime(energy_series);
size_t effective_samples = energy_series.size() / (2 * tau_int + 1);
```

### 3. **Adaptive Error Estimation**
```cpp
// Use autocorrelation-aware error estimation
double GetAutocorrelationAwareError(const std::vector<double>& data) {
    double tau = MeasureAutocorrelationTime(data);
    double variance = Variance(data);
    size_t N_eff = data.size() / (2 * tau + 1);
    return sqrt(variance / N_eff);
}
```

### 4. **Convergence Diagnostics**
```cpp
// Test for equilibration
bool IsEquilibrated(const std::vector<double>& energy_series, 
                   size_t window_size = 1000) {
    // Split series into windows and check for drift
    // Use statistical tests like Kolmogorov-Smirnov
}
```

### 5. **Algorithm Comparison Framework**
```cpp
// For comparing different Monte Carlo algorithms
struct MCAlgorithmTest {
    virtual void RunMonteCarlo(size_t steps) = 0;
    virtual double GetEnergy() const = 0;
    virtual double GetErrorEstimate() const = 0;
};

void CompareAlgorithms(const std::vector<MCAlgorithmTest*>& algorithms) {
    // Run each algorithm
    // Compare results using proper statistical tests
    // Account for different autocorrelation times
}
```

### 6. **Robust Statistical Comparison**
```cpp
// Use Welch's t-test for comparing algorithms with different variances
bool AlgorithmsAgree(double mean1, double err1, 
                    double mean2, double err2,
                    double confidence_level = 0.95) {
    double t_stat = abs(mean1 - mean2) / sqrt(err1*err1 + err2*err2);
    // Compare with appropriate critical value
}
```

## Specific Fixes for Current Tests

### Fix 1: Increase Sampling
```cpp
// For critical temperature tests
const size_t thermalization_steps = 1000000;  // 10x increase
const size_t sampling_steps = 2000000;        // 10x increase
```

### Fix 2: Better Error Analysis
```cpp
// Use more bins and autocorrelation correction
double GetImprovedError(size_t min_bins = 50) {
    size_t optimal_bins = std::min(min_bins, energy_samples_.size() / 100);
    double bin_error = GetEnergyErr(optimal_bins);
    
    // Multiply by autocorrelation correction factor
    double tau = EstimateAutocorrelationTime();
    return bin_error * sqrt(2 * tau + 1);
}
```

### Fix 3: Off-Critical Testing
```cpp
// Test at easier temperatures first
const double T_low = 0.5 * T_critical;   // Fast convergence
const double T_high = 2.0 * T_critical;  // Fast convergence

TEST(PottsModelTest, OffCriticalTemperatures) {
    // These should have smaller error bars and faster convergence
}
```

### Fix 4: Validate Non-DB MCMC Core
```cpp
TEST(NonDBMCMCTest, DetailedBalanceCheck) {
    // For simple 2-state system, verify detailed balance
    // Compare transition matrix with equilibrium distribution
}

TEST(NonDBMCMCTest, EquilibriumDistribution) {
    // Test on simple system with known equilibrium
    // Use very long runs to verify correctness
}
```

## Literature-Standard Monte Carlo Tests

1. **Exact Solutions**: Test on systems with known exact solutions (2D Ising, free fermions)
2. **Finite Size Scaling**: Verify correct scaling behavior near critical points
3. **Thermodynamic Relations**: Check energy/heat capacity relationships
4. **Ergodicity Tests**: Verify all states are accessible
5. **Detailed Balance**: For equilibrium algorithms, verify microscopic reversibility

## Conclusion

The current test failures suggest the Monte Carlo implementation itself may be correct, but the test conditions (critical temperature, insufficient sampling) are challenging the statistical analysis. The solution is to:

1. Use more conservative test conditions initially
2. Implement proper autocorrelation analysis
3. Use adaptive sampling until convergence criteria are met
4. Compare algorithms only after validating each individually
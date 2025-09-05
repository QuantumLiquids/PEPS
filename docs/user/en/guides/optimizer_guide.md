# PEPS Optimizer User Guide

## Overview

PEPS optimizer supports multiple optimization algorithms for VMC-PEPS (Variational Monte Carlo PEPS) optimization. Uses variant-based type-safe design with APIs similar to popular libraries.

### PEPS Optimization Challenges

1. **High-dimensional non-convex optimization**: High parameter space dimensionality with many local minima
2. **Gradient noise**: Statistical noise from Monte Carlo sampling  
3. **Slow convergence**: Flat energy landscapes

## Core Architecture

### Type-safe Parameter System

```cpp
// Modern C++ design: variant-based algorithm dispatch
using AlgorithmParams = std::variant<
  SGDParams,
  AdamParams, 
  StochasticReconfigurationParams,
  LBFGSParams,
  AdaGradParams
>;

struct OptimizerParams {
  BaseParams base_params;           // Common parameters
  AlgorithmParams algorithm_params; // Algorithm-specific parameters
};
```

Features:
- Unified interface for all algorithms
- Compile-time type checking
- Easy to extend with new algorithms

### Three-layer Parameter Architecture

```
OptimizerParams
├── BaseParams (shared by all algorithms)
│   ├── max_iterations        // Maximum iterations
│   ├── energy_tolerance      // Energy convergence criterion
│   ├── gradient_tolerance    // Gradient convergence criterion
│   ├── plateau_patience      // Plateau patience parameter
│   ├── learning_rate         // Unified learning rate interface
│   └── lr_scheduler          // Optional learning rate scheduler
└── AlgorithmParams (algorithm-specific)
    ├── SGDParams            // Stochastic Gradient Descent
    ├── AdamParams           // Adam optimizer
    ├── StochasticReconfigurationParams  // Stochastic Reconfiguration
    ├── LBFGSParams          // Limited-memory BFGS
    └── AdaGradParams        // Adaptive Gradient
```

## Optimization Algorithms

### 1. Stochastic Reconfiguration (Natural Gradient)

**Mathematical principle**:

Stochastic Reconfiguration is the gold standard for PEPS optimization, based on natural gradient:

```
Standard gradient descent: θ_{n+1} = θ_n - η ∇E(θ)
Natural gradient descent: θ_{n+1} = θ_n - η S^{-1} ∇E(θ)
```

Where S is the Fisher information matrix (S-matrix in VMC):

```
S_{ij} = ⟨O_i^* O_j⟩ - ⟨O_i^*⟩⟨O_j⟩
∇E_i = ⟨E_{loc} O_i^*⟩ - ⟨E_{loc}⟩⟨O_i^*⟩
```

Here O_i = ∂ln|ψ⟩/∂θ_i is the logarithmic derivative operator.

Principle:
- Considers parameter space geometric structure
- S-matrix characterizes parameter correlations
- Avoids pathological behavior of standard gradients

**Parameter settings**:

```cpp
// CG solver parameters (critical!)
ConjugateGradientParams cg_params{
  100,      // max_iter: S-matrix usually ill-conditioned, needs more iterations
  1e-5,     // tolerance: balance precision and computational cost
  20,       // restart_step: avoid numerical accumulation errors
  0.001     // diag_shift: regularization to prevent singular matrix
};

// SR-specific parameters
StochasticReconfigurationParams sr_params{
  cg_params,
  false,    // normalize_update: whether to normalize updates
  0.0       // adaptive_diagonal_shift: dynamic regularization
};
```

**Convergence characteristics**:
- **Pros**: Theoretically optimal convergence rate, robust for complex energy landscapes
- **Cons**: Requires solving linear system each step, high computational cost O(N³)
- **Use case**: High-precision optimization, moderate parameter count systems

### 2. Adam (Adaptive Moment Estimation)

**Mathematical principle**:

Adam combines momentum and adaptive learning rate:

```
m_t = β₁ m_{t-1} + (1-β₁) g_t        // First moment (momentum)
v_t = β₂ v_{t-1} + (1-β₂) g_t²       // Second moment (adaptive rate)
m̂_t = m_t / (1-β₁^t)                // Bias correction
v̂_t = v_t / (1-β₂^t)                // Bias correction
θ_{t+1} = θ_t - η m̂_t / (√v̂_t + ε)  // Parameter update
```

Parameter explanation:
- **β₁**: momentum parameter, remembers historical gradients
- **β₂**: adaptive learning rate parameter
- **Bias correction**: corrects initial bias

**Parameter setting guide**:

```cpp
AdamParams adam_params{
  0.9,      // beta1: first moment decay rate, controls momentum strength
  0.999,    // beta2: second moment decay rate, controls adaptivity
  1e-8,     // epsilon: numerical stability, prevents division by zero
  0.0       // weight_decay: L2 regularization strength
};
```

**Tuning strategy**:
- **Learning rate**: PEPS usually needs smaller learning rates (1e-4 ~ 1e-2)
- **β₁**: default 0.9 usually works, reduce to 0.8 for high noise
- **β₂**: default 0.999, reduce to 0.99 for non-stationary problems
- **ε**: increase to 1e-6 if numerical instability occurs

### 3. SGD with Momentum

**Mathematical principle**:

Classic stochastic gradient descent with momentum:

```
Standard SGD: θ_{t+1} = θ_t - η g_t
With momentum: v_t = μ v_{t-1} + η g_t
              θ_{t+1} = θ_t - v_t

Nesterov variant: v_t = μ v_{t-1} + η ∇f(θ_t - μ v_{t-1})
                 θ_{t+1} = θ_t - v_t
```

**Parameter understanding**:
- **μ (momentum)**: usually 0.9, controls historical gradient influence
- **Nesterov**: provides "look-ahead" gradient computation, usually converges faster

### 4. AdaGrad (Adaptive Gradient)

**Mathematical principle**:

Automatically adjusts learning rate based on historical gradients:

```
G_t = G_{t-1} + g_t ⊙ g_t          // Accumulate gradient squares
θ_{t+1} = θ_t - η g_t / (√G_t + ε)  // Parameter update
```

**Features**:
- **Adaptive**: learning rate automatically decreases for frequently updated parameters
- **Problem**: learning rate monotonically decreases, may stop too early
- **Use case**: sparse gradient scenarios, or early exploration phases

### 5. L-BFGS (Limited-memory BFGS)

**Mathematical principle**:

Quasi-Newton method based on Hessian approximation:

```
H_k ≈ ∇²f(x_k)                    // Hessian approximation
d_k = -H_k^{-1} ∇f(x_k)           // Search direction
x_{k+1} = x_k + α_k d_k            // Line search update
```

L-BFGS constructs Hessian approximation using limited history, avoiding full matrix storage.

**Use cases**:
- Deterministic optimization problems
- Small gradient noise situations
- When superlinear convergence rate is needed

## Learning Rate Scheduling Strategies

### 1. Exponential Decay

```cpp
auto scheduler = std::make_unique<ExponentialDecayLR>(
  0.01,     // initial_lr
  0.95,     // decay_rate: decay 5% every decay_steps
  100       // decay_steps: decay every 100 steps
);
```

**Use case**: stable convergence, suitable for long-term training

### 2. Step Decay

```cpp
auto scheduler = std::make_unique<StepLR>(
  0.01,     // initial_lr
  200,      // step_size: reduce every 200 steps
  0.5       // gamma: halve each time
);
```

**Use case**: phase-wise adjustment, suitable for clear training phases

### 3. Plateau-based (Energy Plateau Detection)

```cpp
auto scheduler = std::make_unique<PlateauLR>(
  0.01,     // initial_lr
  0.5,      // factor: halve when plateau detected
  20,       // patience: 20 steps without improvement = plateau
  1e-5      // threshold: energy improvement threshold
);
```

**Use case**: best choice for PEPS optimization, automatically adjusts based on physical convergence

## Practical Usage Guide

### Quick Start: Factory Methods

```cpp
#include "qlpeps/optimizer/optimizer_params.h"

// 1. Stochastic Reconfiguration (recommended for high-precision optimization)
ConjugateGradientParams cg_params{100, 1e-5, 20, 0.001};
auto sr_params = OptimizerFactory::CreateStochasticReconfiguration(
  1000,        // max_iterations
  cg_params,   // CG solver parameters
  0.1          // learning_rate
);

// 2. Adam (recommended for rapid prototyping)
auto adam_params = OptimizerFactory::CreateAdam(
  1000,        // max_iterations
  1e-3         // learning_rate
);

// 3. SGD with decay (simple and robust)
auto sgd_params = OptimizerFactory::CreateSGDWithDecay(
  1000,        // max_iterations
  0.01,        // initial_learning_rate
  0.95,        // decay_rate
  100          // decay_steps
);
```

### Advanced Configuration: Builder Pattern

```cpp
auto params = OptimizerParamsBuilder()
  .SetMaxIterations(2000)
  .SetEnergyTolerance(1e-8)
  .SetGradientTolerance(1e-6)
  .SetPlateauPatience(50)
  .SetLearningRate(0.01, std::make_unique<PlateauLR>(0.01, 0.5, 20))
  .WithStochasticReconfiguration(cg_params, true, 0.001)  // enable normalize_update
  .Build();
```

### Gradient Clipping

To improve numerical stability for first-order optimizers (SGD/AdaGrad/Adam), two optional preprocessing steps are provided:

- Per-element magnitude clipping (complex-safe, preserve phase): if |g| > c then g ← polar(c, arg(g)); for real tensors this reduces to sign-preserving absolute clipping.
- Global L2 norm clipping: let r = sqrt(Σ |g_j|^2). If r > C, scale g ← (C/r)·g uniformly.

Usage via Builder:

```cpp
auto params = OptimizerParamsBuilder()
  .SetMaxIterations(2000)
  .SetLearningRate(0.01)
  .WithSGD(0.9, false)
  .SetClipValue(0.1)   // per-element magnitude threshold
  .SetClipNorm(10.0)   // global L2 threshold
  .Build();
```

Notes:
- Clipping applies only to first-order methods; SR/L-BFGS do not use clipping by default.
- SetClipValue/SetClipNorm require BaseParams to be initialized first (e.g., via SetMaxIterations/SetLearningRate).
- MPI semantics follow first-order updates: clipping is performed on master rank only.

### VMCPEPS Integration

```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/api/conversions.h"  // explicit PEPS/TPS/SITPS conversions

// Complete VMCPEPS optimization parameters
VMCPEPSOptimizerParams vmc_params{
  optimizer_params,    // optimizer parameters created above
  mc_params,          // Monte Carlo sampling parameters
  peps_params,        // PEPS truncation parameters  
  "optimized_state"   // output file prefix
};

// Create and execute optimizer
// Example of explicit conversions if starting from a PEPS
auto tps      = qlpeps::ToTPS<ComplexDouble, QNZ2>(peps_initial);
auto sitps    = qlpeps::ToSplitIndexTPS<ComplexDouble, QNZ2>(tps);

VMCPEPSOptimizer<ComplexDouble, QNZ2> executor(
  vmc_params, sitps, comm, energy_solver
);
executor.Execute();
```

## Tuning Best Practices

### 1. Algorithm Selection Decision Tree

```
Need highest precision?
├── Yes → Stochastic Reconfiguration
│   ├── Parameter count < 10⁴ → Standard SR
│   └── Parameter count ≥ 10⁴ → SR + larger diag_shift
└── No → Look at training stage
    ├── Fast prototyping/exploration → Adam
    ├── Production/stability → SGD + decay
    └── Special needs → AdaGrad/L-BFGS
```

### 2. Learning Rate Setting Strategy

```cpp
// Conservative strategy (recommended for production)
double conservative_lr = 0.001;  // slow but stable

// Aggressive strategy (for rapid experiments)  
double aggressive_lr = 0.1;     // fast but may be unstable

// Adaptive strategy (best practice)
auto adaptive_lr = std::make_unique<PlateauLR>(0.01, 0.5, 20);
```

### 3. Convergence Criteria Setting

```cpp
// High-precision physics calculations
BaseParams high_precision{
  5000,     // max_iterations: give enough time
  1e-10,    // energy_tolerance: chemical accuracy
  1e-8,     // gradient_tolerance: strict gradient convergence
  100,      // plateau_patience: avoid premature stopping
  0.01
};

// Rapid prototyping
BaseParams prototype{
  500,      // max_iterations: quick feedback
  1e-6,     // energy_tolerance: sufficient precision
  1e-4,     // gradient_tolerance: more relaxed
  20,       // plateau_patience: quick convergence decision
  0.01
};
```

### 4. Common Problem Diagnosis

**Slow convergence**:
```cpp
// 1. Increase learning rate
// 2. Use Adam or Stochastic Reconfiguration
// 3. Check if gradient calculation is correct
// 4. Consider preconditioning
```

**Numerical instability**:
```cpp
// 1. Reduce learning rate
// 2. Increase epsilon (Adam)
// 3. Increase diag_shift (SR)
// 4. Use gradient clipping
```

**Memory issues**:
```cpp
// 1. Avoid L-BFGS for large systems
// 2. Reduce max_iter for SR CG solver
// 3. Use more aggressive PEPS truncation
```

## Performance Benchmarks （TODO table)

Based on our test data:

| Algorithm | Convergence Speed | Memory Usage | Numerical Stability | Recommended Use |
|-----------|-------------------|--------------|---------------------|-----------------|

## Summary

Effective PEPS optimization requires:
1. **Correct algorithm choice**: based on problem scale and precision requirements
2. **Reasonable parameter settings**: based on physical intuition and experience
3. **Appropriate learning rate scheduling**: combined with energy convergence characteristics
4. **Sufficient patience**: PEPS optimization is inherently difficult

Recommend testing these parameter configurations on actual problems.

## Advanced Mathematical Principles

### Stochastic Reconfiguration Details

#### Fisher Information Matrix Construction

In VMC, we have wavefunction |ψ(θ)⟩, where θ are variational parameters. Fisher information matrix is defined as:

```
S_{ij} = ⟨∂_i ln ψ|∂_j ln ψ⟩ - ⟨∂_i ln ψ⟩⟨∂_j ln ψ⟩
```

Where ∂_i = ∂/∂θ_i, and:

```
O_i = ∂_i ln ψ = ∂_i ln ψ / ψ = (1/ψ) ∂_i ψ
```

#### Why Natural Gradient is Better?

Consider the Riemannian structure of parameter space. On curved manifolds, the steepest descent direction is not the Euclidean gradient, but the natural gradient:

```
g^{natural} = S^{-1} g^{Euclidean}
```

The S-matrix encodes correlations between parameters. If two parameters are highly correlated, changing one automatically affects the other, so updates must consider this coupling.

#### Gradient Calculation in PEPS

For PEPS, local derivative is:

```
O_i^{(x,y)} = ∂ ln ψ / ∂ T^{(x,y)}_i

where T^{(x,y)}_i is the i-th component of tensor at site (x,y)
```

In VMC sampling:

```
⟨O_i^* O_j⟩ = (1/N) Σ_k O_i^*(config_k) O_j(config_k)
⟨E_loc O_i^*⟩ = (1/N) Σ_k E_loc(config_k) O_i^*(config_k)
```

### Adam Algorithm Convergence Analysis

#### Necessity of Bias Correction

Adam's moment estimates have bias in early stages:

```
E[m_t] = E[g_t] (1-β₁^t) / (1-β₁)
E[v_t] = E[g_t²] (1-β₂^t) / (1-β₂)
```

Without bias correction, early estimates severely underestimate true moments, leading to oversized update steps.

#### Effective Learning Rate Scaling

Adam's effective learning rate is:

```
η_eff = η / (√v̂_t + ε) ≈ η / σ_t

where σ_t is RMS of historical gradients
```

This means:
- Parameters with consistently large gradients: small effective learning rate
- Parameters with noisy but small average gradients: large effective learning rate

### Special Considerations for Optimization Algorithms in PEPS

#### 1. Gauge Freedom

PEPS has gauge freedom: can insert U†U on bonds without changing physical state. This leads to:
- Flat directions in parameter space
- Singular Hessian matrix
- Need for gauge fixing or regularization

#### 2. Entanglement Constraints

PEPS bond dimension limits representable entanglement. During optimization:
- Gradients may point to unreachable states
- Need projected gradient methods
- SVD truncation introduces additional noise

#### 3. Monte Carlo Noise

VMC gradient estimates contain statistical error:

```
g_estimated = g_true + noise
Var[g_estimated] ∝ 1/N_samples
```

Optimization algorithms need to be robust to noisy gradients:
- Adam's momentum helps filter noise
- SR's preconditioning can amplify or suppress noise depending on S-matrix condition number

## Advanced Techniques and Troubleshooting

### 1. Gradient Clipping

When gradient norm is too large, clip to reasonable range:

```cpp
// In optimizer implementation
double grad_norm = CalculateGradientNorm(gradient);
if (grad_norm > clip_threshold) {
  ScaleGradient(gradient, clip_threshold / grad_norm);
}
```

**Use case**: prevent numerical explosion, especially in optimization early stages.

### 2. Warm Restart

Periodically reset momentum/adaptive terms:

```cpp
// Reset Adam's moment estimates every N steps
if (iteration % restart_period == 0) {
  ResetMomentEstimates();
}
```

**Use case**: escape local minima, re-explore parameter space.

### 3. Learning Rate Warmup

Gradually increase learning rate from 0:

```cpp
double warmup_lr = base_lr * min(1.0, iteration / warmup_steps);
```

**Use case**: avoid large early updates that destroy pre-trained state.

### 4. Preconditioning

Add problem-specific preconditioning to gradient:

```cpp
// Example: normalize by parameter magnitude
preconditioned_grad[i] = gradient[i] / (abs(parameter[i]) + epsilon);
```

### 5. Multi-stage Optimization

Use different strategies for different stages:

```cpp
// Stage 1: Exploration (high learning rate, Adam)
// Stage 2: Refinement (medium learning rate, SGD)  
// Stage 3: Polishing (low learning rate, SR)
```

## Complete Code Examples

### Example 1: Basic Heisenberg Model Optimization

```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/optimizer/optimizer_params.h"

void OptimizeHeisenbergModel() {
  using TenElemT = qlten::QLTEN_Complex;
  using QNT = qlten::QNZ2;
  
  // 1. Set Monte Carlo parameters
  MonteCarloParams mc_params;
  mc_params.sample_num = 8000;        // sufficient statistical samples
  mc_params.warmup_sample_num = 2000; // adequate thermalization
  mc_params.mc_samples_dumpinterval = 100;
  mc_params.filename_postfix = "heisenberg";
  
  // 2. Set PEPS parameters  
  PEPSParams peps_params;
  peps_params.bond_dim = 4;           // moderate bond dimension
  peps_params.truncate_para = BMPSTruncatePara(
    peps_params.bond_dim,
    peps_params.bond_dim * 20,  // cutoff
    1e-10,                      // trunc_err
    QLTensor<TenElemT, QNT>::GetQNSectorSet().GetQNSctNum(),
    &world
  );
  
  // 3. Set Stochastic Reconfiguration optimizer
  ConjugateGradientParams cg_params{
    100,    // max_iter: sufficient CG iterations
    1e-5,   // tolerance: balance precision and speed
    20,     // restart_step: avoid numerical accumulation errors
    0.001   // diag_shift: regularization parameter
  };
  
  auto sr_params = OptimizerFactory::CreateStochasticReconfigurationAdvanced(
    1000,      // max_iterations
    1e-8,      // energy_tolerance: high precision requirement
    1e-6,      // gradient_tolerance
    50,        // plateau_patience: give enough patience
    cg_params,
    0.1,       // learning_rate: moderate learning rate
    std::make_unique<PlateauLR>(0.1, 0.5, 30, 1e-5)  // energy plateau detection
  );
  
  // 4. Combine VMC parameters
  VMCPEPSOptimizerParams vmc_params{
    sr_params,
    mc_params, 
    peps_params,
    "heisenberg_optimized"  // output file prefix
  };
  
  // 5. Create energy solver (assume defined)
  SpinOneHalfHeisenbergSquare energy_solver(4, 4, 1.0);  // 4x4 lattice, J=1
  
  // 6. Create and execute optimizer
  try {
    VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, 
                            SpinOneHalfHeisenbergSquare> executor(
      vmc_params, 
      "initial_state_dir/",  // initial state path
      MPI_COMM_WORLD, 
      energy_solver
    );
    
    executor.Execute();
    
    // 7. Get results
    std::cout << "Optimization completed. Final energy: " 
              << executor.GetMinimumEnergy() << std::endl;
              
  } catch (const std::exception& e) {
    std::cerr << "Optimization failed: " << e.what() << std::endl;
  }
}
```

### Example 2: End-to-end VMCPEPS Workflow

```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_heisenberg_square.h"

int main(int argc, char* argv[]) {
  // 1. MPI initialization
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  try {
    // 2. System parameter setup
    const size_t Lx = 4, Ly = 4;
    const double J = 1.0;  // Heisenberg coupling
    const size_t bond_dim = 4;
    
    // 3. Create energy solver
    SpinOneHalfHeisenbergSquare energy_solver(Ly, Lx, J);
    
    // 4. Set optimization parameters
    if (rank == 0) {
      std::cout << "Setting up optimization parameters..." << std::endl;
    }
    
    // Monte Carlo parameters
    MonteCarloParams mc_params;
    mc_params.sample_num = 10000;
    mc_params.warmup_sample_num = 2000;
    mc_params.mc_samples_dumpinterval = 500;
    mc_params.filename_postfix = "heisenberg_4x4";
    
    // PEPS parameters
    PEPSParams peps_params;
    peps_params.bond_dim = bond_dim;
    peps_params.truncate_para = BMPSTruncatePara(
      bond_dim, bond_dim * 20, 1e-10, 
      QLTensor<qlten::QLTEN_Complex, qlten::QNZ2>::GetQNSectorSet().GetQNSctNum(),
      &MPI_COMM_WORLD
    );
    
    // Optimizer parameters (Stochastic Reconfiguration)
    ConjugateGradientParams cg_params{100, 1e-5, 20, 0.001};
    auto opt_params = OptimizerFactory::CreateStochasticReconfigurationAdvanced(
      1500,                    // max_iterations
      1e-8,                   // energy_tolerance
      1e-6,                   // gradient_tolerance
      80,                     // plateau_patience
      cg_params,
      0.1,                    // learning_rate
      std::make_unique<PlateauLR>(0.1, 0.5, 30, 1e-5)
    );
    
    // 5. Combine parameters
    VMCPEPSOptimizerParams vmc_params{
      opt_params, mc_params, peps_params, "heisenberg_4x4_D4"
    };
    
    // 6. Create and execute optimizer
    VMCPEPSOptimizer<qlten::QLTEN_Complex, qlten::QNZ2, 
                            SquareNNUpdater, SpinOneHalfHeisenbergSquare> 
        executor(vmc_params, "random_init/", MPI_COMM_WORLD, energy_solver);
    
    if (rank == 0) {
      std::cout << "Starting optimization..." << std::endl;
    }
    
    executor.Execute();
    
    // 7. Output results
    if (rank == 0) {
      std::cout << "Optimization completed!" << std::endl;
      std::cout << "Final minimum energy: " << executor.GetMinimumEnergy() << std::endl;
      std::cout << "Energy per site: " << executor.GetMinimumEnergy() / (Lx * Ly) << std::endl;
      
      // Save optimization trajectory
      const auto& energy_traj = executor.GetEnergyTrajectory();
      const auto& error_traj = executor.GetEnergyErrorTrajectory();
      
      std::ofstream traj_file("optimization_trajectory.dat");
      if (!traj_file.is_open()) {
        throw std::ios_base::failure("Failed to open optimization_trajectory.dat");
      }
      traj_file << "# Iteration Energy Error\n";
      for (size_t i = 0; i < energy_traj.size(); ++i) {
        traj_file << i << " " << energy_traj[i] << " " << error_traj[i] << "\n";
        if (traj_file.fail()) {
          throw std::ios_base::failure("Failed to write trajectory data");
        }
      }
      traj_file.close();
      if (traj_file.fail()) {
        throw std::ios_base::failure("Failed to close optimization_trajectory.dat");
      }
      
      std::cout << "Optimization trajectory saved to optimization_trajectory.dat" << std::endl;
    }
    
  } catch (const std::exception& e) {
    if (rank == 0) {
      std::cerr << "Error during optimization: " << e.what() << std::endl;
    }
    MPI_Finalize();
    return -1;
  }
  
  MPI_Finalize();
  return 0;
}
```

---

Test and adjust these code examples according to your specific system.
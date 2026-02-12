# PEPS Documentation

**PEPS** (Projected Entangled Pair States) is a high-performance C++ library for simulating 2D strongly correlated electron systems using tensor network methods. It provides efficient implementations of finite-size PEPS algorithms, variational Monte Carlo optimization, and quantum many-body calculations.

## Prerequisites knowledge

- **Quantum many-body physics** background 
- **PEPS && Variational Monte Carlo methods**, like SR technical
- **Modern C++ (C++20)** programming
- **MPI** for parallel computing


## Quick Start

**New to PEPS? Start here:**

1. **[Installation Guide](tutorials/01_installation.html)** - Build and install PEPS
2. **[Heisenberg Model Full Workflow](tutorials/02_quick_start.html)** - A complete example: J1-J2 Heisenberg Model.
3. **[PEPS Basics](tutorials/03_peps_basics.html)** - Understanding Data structure

## User Tutorials

### Define your model in simple update

### Define your model in VMC

### Define your Monte-Carlo update strategies.



## Developer Resources

### API Reference
- **[Core API](api/core.html)** - Main PEPS classes and tensor network framework
- **[Algorithm API](api/algorithm.html)** - PEPS update algorithms and optimization methods
- **[VMC API](api/vmc.html)** - Variational Monte Carlo implementation
- **[Model Solvers](api/model_solvers.html)** - Built-in model implementations

### Development
- **[Developer Guide](developer/index.html)** - Architecture, design principles, and contribution guidelines
- **[Testing Guide](developer/testing.html)** - How to run tests and contribute code
- **[Performance Guide](developer/performance.html)** - Optimization and benchmarking

## Key Features

- **Finite-Size PEPS** - Efficient implementation for finite 2D systems
- **Multiple Update Algorithms** - Simple update, loop update, and variational methods
- **Variational Monte Carlo** - VMC-based optimization with stochastic reconfiguration
- **Multiple Lattice Geometries** - Square, triangular, and custom lattice support
- **Model-Specific Solvers** - Built-in implementations for common quantum models
- **MPI Parallelization** - Distributed computing support
- **Header-Only Design** - Easy integration into existing projects

## External Resources

- **[GitHub Repository](https://github.com/QuantumLiquids/PEPS)**
- **[Issue Tracker](https://github.com/QuantumLiquids/PEPS/issues)**
- **[Dependencies](https://github.com/QuantumLiquids/TensorToolkit)** - TensorToolkit library
- **[Related Work](https://github.com/QuantumLiquids/UltraDMRG)** - UltraDMRG project

---

*Need help? Check the [tutorials](tutorials/index.html) or open an issue on GitHub.*


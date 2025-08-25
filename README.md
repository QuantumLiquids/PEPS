# PEPS : Tensor Network Library for simulating 2D strongly correlated electron systems
**QuantumLiquids/PEPS** is a library for efficiently simulating finite-size PEPS
for two-dimensional quantum many-body problems. It is under active development.

---

## Features
- [x] Variational Monte Carlo (VMC) optimizer for finite-size PEPS
- [x] Optimizer framework with SR and first-order methods (e.g., AdaGrad/Adam)
- [x] Fermion support 
- [x] Extensible model and Monte Carlo updater interfaces (plugin-style)
- [x] Monte Carlo measurement toolkit (energy, one-/two-point functions, autocorrelation)
- [x] built-in common lattice models
- [x] Header-only core; seamless with TensorToolkit

---

## Documentation

The PEPS project provides comprehensive documentation organized into several categories:

- **📚 API Reference (Doxygen)**: run `./docs/build_docs.sh`, then open `docs/build/html/index.html`
- **📖 User Guides (EN)**: `docs/user/en/guides/`
- **📖 User Guides (ZH)**: `docs/user/zh/guides/`
- **🧰 Installation (ZH)**: `docs/user/zh/installation.md`
- **🚀 Getting Started (ZH)**: `docs/user/zh/getting_started.md`
- **🛠️ Development**: `docs/dev/`
- **💻 Examples**: `examples/`

### Quick Start

1. **Build API Docs**: `./docs/build_docs.sh` (from project root), then open `docs/build/html/index.html`
2. **Read Guides**: `docs/user/en/guides/` or `docs/user/zh/guides/`
3. **Run Examples**: Build and run from the `examples/` directory

---

## Dependencies
**PEPS** is header-only and requires no installation dependencies.
However, building test cases or practical PEPS programs based on this project requires:

- **C++ Compiler:** C++20 or above
- **Build System:** CMake (version 3.12 or higher)
- **Math Libraries:** Intel MKL or OpenBLAS
- **Parallelization:** MPI
- **Tensor/MPS:** [QuantumLiquids/TensorToolkit](https://github.com/QuantumLiquids/TensorToolkit); [QuantumLiquids/UltraDMRG](https://github.com/QuantumLiquids/UltraDMRG)
- **Testing (optional):** GoogleTest

---

## Installation

Clone the repository:

```
git clone https://github.com/QuantumLiquids/PEPS.git
cd PEPS
```

The installation is now complete. You can directly use it as a header-only dependency (recommended):

```cmake
# CMake (example)
target_include_directories(your_target PRIVATE /path/to/PEPS/include)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

```

Optional install (copy headers to a prefix) via CMake:

```
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
make install

```
---

## Author

Hao-Xin Wang

For any inquiries or questions regarding the project,
you can reach out to Hao-Xin via email at wanghaoxin1996@gmail.com.

---

## Acknowledgments

We would like to express our gratitude to the following individuals for their contributions and guidance:

- Wen-Yuan Liu, expert in the variational Monte-Carlo PEPS.
- Zhen-Cheng Gu, my postdoc advisor, one of the pioneers in the field of tensor network.

Their expertise and support have been invaluable in the development of QuantumLiquids/PEPS.


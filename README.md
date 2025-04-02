# PEPS : Tensor Network Libraray for simulating 2D Strongly Correlated Electron Systems
**QuantumLiquids/PEPS** is a library for efficiently simulation the finite-size PEPS 
for two-dimension quantum many-body problems. It is still under building.

---

## Features
- [x] variational-Monte-Carlo-based finite-size PEPS optimization and measurements

---

## To-Do List

- [ ] PESS

---

## Dependence
**PEPS** is header-only and requires no installation dependencies.
However, building test cases or practical PEPS programs based on this project requires:

- **C++ Compiler:** C++20 or above
- **Build System:** CMake (version 3.12 or higher)
- **Math Libraries:** Intel MKL or OpenBLAS
- **Parallelization:** MPI
- **Tensor Operations:** [QuantumLiquids/TensorToolkit](https://github.com/QuantumLiquids/TensorToolkit); [QuantumLiquids/UltraDMRG](https://github.com/QuantumLiquids/UltraDMRG)
- **Testing (optional):** GoogleTest

---

## Install

Clone the repository into a desired directory and change into that location:

```
git clone https://github.com/QuantumLiquids/PEPS.git
cd PEPS
```

Using CMake:

```
mkdir build && cd build
cmake ..
make -j4 && make install

```

You may want to specify `CMAKE_CXX_COMPILER` as your favorite C++ compiler,
and `CMAKE_INSTALL_PREFIX` as your install directory when you're calling `cmake`

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


# PEPS
QuantumLiquids/PEPS is a library for efficiently simulation the finite-size PEPS 
for two-dimension quantum many-body problems. It is still under building.


## To-Do List

- [ ] PESS

## Dependence
Please note that the project requires the following dependencies to be installed 
in order to build and run successfully:

- C++17 Compiler
- CMake (version 3.12 or higher)
- Intel MKL or OpenBlas
- MPI
- [QuantumLiquids/TensorToolkit](https://github.com/QuantumLiquids/TensorToolkit)
- [QuantumLiquids/UltraDMRG](https://github.com/QuantumLiquids/UltraDMRG)
- GoogleTest (if testing is required)

## Install

Clone the repository into a desired directory and change into that location:

```
git clone https://github.com/QuantumLiquids/PEPS.git
cd UltraDMRG
```

Using CMake:

```
mkdir build && cd build
cmake ..
make -j4 && make install

```

You may want to specify `CMAKE_CXX_COMPILER` as your favorite C++ compiler,
and `CMAKE_INSTALL_PREFIX` as your install directory when you're calling `cmake`

## Author

Hao-Xin Wang

For any inquiries or questions regarding the project,
you can reach out to Hao-Xin via email at wanghaoxin1996@gmail.com.

## Acknowledgments

We would like to express our gratitude to the following individuals for their contributions and guidance:

- Wen-Yuan Liu, expert in the variational Monte-Carlo PEPS.
- Zhen-Cheng Gu, my postdoc advisor, one of the pioneers in the field of tensor network.

Their expertise and support have been invaluable in the development of QuantumLiquids/PEPS.


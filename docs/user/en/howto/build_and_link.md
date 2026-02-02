# Build and link (CMake)

Use this page when you want to **compile an executable** that links to BLAS/LAPACK, MPI, OpenMP, and uses PEPS + TensorToolkit.

## Recommended compiler versions

To avoid known issues in older compilers (e.g. lambda + structured bindings), use at least:

| Compiler | Recommended |
|---|---|
| GCC | 11.0+ |
| Clang/LLVM | 15.0+ |
| Intel oneAPI icpx | 2024.0+ (LLVM 17 baseline) |

## What PEPS is (from a build perspective)

- PEPS itself is header-only under `include/`.
- Your executable still needs **BLAS/LAPACK + MPI + OpenMP**, and typically `hptt`.

## Optional: install headers via CMake

If you prefer an “installed” include tree (instead of referencing this repo directly), you can use the repo’s CMake install:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build . --target install
```

Notes:

- If you omit `CMAKE_INSTALL_PREFIX`, many Unix systems default to `/usr/local` and you may need admin permissions.
- On HPC, installing into a user-writable prefix is usually the easiest.

## Minimal CMake skeleton

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_peps_app CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

find_path(hptt_INCLUDE_DIR "hptt.h")
find_library(hptt_LIBRARY "libhptt.a")

add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE
  /path/to/PEPS/include
  /path/to/TensorToolkit/include
  /path/to/UltraDMRG/include
  ${hptt_INCLUDE_DIR}
)
target_link_libraries(my_app PRIVATE
  ${hptt_LIBRARY}
  BLAS::BLAS
  LAPACK::LAPACK
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
)
```

## Notes on BLAS vendors and include paths

- On x86_64, MKL is often faster than OpenBLAS for typical workloads; on Apple Silicon you usually use OpenBLAS.
- CMake’s `FindBLAS` does not always populate an include variable. If you need it (e.g. MKL), you may have to set it yourself (example: `set(BLAS_INCLUDE_DIR \"$ENV{MKLROOT}/include\")`).

## Quick sanity check: build the examples

```bash
cd examples
mkdir -p build && cd build
cmake ..
make -j
```

## Related

- Tutorials: `../tutorials/index.md`

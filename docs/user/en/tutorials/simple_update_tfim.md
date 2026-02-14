# Simple Update (TFIM) tutorial

This tutorial runs a minimal **4×4 transverse-field Ising model** (TFIM) Simple Update and dumps a PEPS to disk.

Source code:

- `examples/transverse_field_ising_simple_update.cpp`

## What you’ll build

- A working example binary: `transverse_field_ising_simple_update`
- A dumped PEPS directory: `./peps/` (used by the VMC tutorial next)

## Prerequisites

See: `../howto/build_and_link.md`.

Minimum requirements (typical HPC setup):

- C++20 compiler
- CMake
- BLAS/LAPACK
- MPI + OpenMP (recommended, and used by many components even if a specific example file doesn’t include `<mpi.h>`)
- `hptt`

## If you start a fresh project

You can either:

- build and run the provided `examples/` directly (recommended), or
- create your own project directory and copy the example source.

Minimal project directory:

```bash
mkdir TransverseFieldIsingPEPS
```

## Model and operator conventions (important)

The example implements:

\[
H = - \sum_{\langle i,j\rangle} Z_i Z_j - h \sum_i X_i
\]

where \(X,Z\) are Pauli matrices with matrix elements ±1.

For the NN bond operator \(-Z\otimes Z\), the code uses a rank-4 tensor with the fixed index order:

\[
(\text{in}_1, \text{out}_1, \text{in}_2, \text{out}_2).
\]

In the provided example, `Contract(&opZ, {}, &opZ, {}, &ham_zz)` (outer product) produces exactly this ordering.

Index-order picture (numbers are index positions):

```
      1            3          // out_1, out_2
      |            |
      ^            ^
      |----  H  ----|
      ^            ^
      |            |
      0            2          // in_1,  in_2
```

Terminology:

- `in` / `out` correspond to bra/ket legs (or “input/output” legs) of the operator tensor.
- The subscript `1,2` identify which site the leg belongs to.

## Using this example in your own project (CMake snippet)

If you copy `examples/transverse_field_ising_simple_update.cpp` into your own CMake project, the simplest approach is:

```cmake
add_executable(transverse_field_ising_simple_update
  ${CMAKE_SOURCE_DIR}/examples/transverse_field_ising_simple_update.cpp)

target_link_libraries(transverse_field_ising_simple_update
  ${hptt_LIBRARY}
  BLAS::BLAS
  LAPACK::LAPACK
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX)
```

Note:

- Even if your source does not include `<mpi.h>`, you may still need to link `MPI::MPI_CXX` because TensorToolkit/PEPS components use MPI internally.

## Steps

### 1) Build the examples

```bash
cd examples
mkdir -p build && cd build
cmake ..
make -j
```

### 2) Run Simple Update

```bash
cd examples/build
./transverse_field_ising_simple_update
```

### Optional: enable advanced automatic stop

The default example uses fixed `steps`. If you want automatic convergence stop, replace
the `SimpleUpdatePara` construction with:

```cpp
auto su_para = SimpleUpdatePara::Advanced(
    /*steps=*/1000,      // hard cap
    /*tau=*/0.05,
    /*Dmin=*/1,
    /*Dmax=*/4,
    /*Trunc_err=*/1e-14,
    /*energy_abs_tol=*/1e-8,
    /*energy_rel_tol=*/1e-10,
    /*lambda_rel_tol=*/1e-6,
    /*patience=*/3,
    /*min_steps=*/10);
```

After `Execute()`, you can inspect why it stopped:

```cpp
executor.Execute();
std::cout << "Converged: " << std::boolalpha << executor.LastRunConverged() << "\\n";
std::cout << "Executed sweeps: " << executor.LastRunExecutedSteps() << "\\n";
```

Expected output:

- A directory `examples/build/peps/` containing the dumped PEPS.
- Console logs showing sweep progress and an estimated energy.
- If you rerun the binary, it overwrites `peps/` in the current working directory.

## Next steps

- Continue with VMC: `vmc_optimize_tfim.md`

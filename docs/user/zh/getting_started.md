# Getting Started
本部分将展示如何搞一个二维横场Ising模型的Simple Update作为这个包的开始

## Prerequisites
- **C++ Compiler:** C++20 or above
- **Build System:** CMake (version 3.12 or higher)
- **BLAS/LAPACK:** Intel MKL, AOCL or OpenBLAS
- **Parallelization:** MPI
- **Tensor&MPS:** [QuantumLiquids/TensorToolkit](https://github.com/QuantumLiquids/TensorToolkit); [QuantumLiquids/UltraDMRG](https://github.com/QuantumLiquids/UltraDMRG)

### 编译器支持与最低版本
为避免已知的 lambda + 结构化绑定在旧编译器中的问题，推荐使用下列及以上版本：

| 编译器 |  推荐版本 |
| --- | --- |
| GCC | 11.0+ |
| Clang/LLVM |  15.0+ |
| Intel oneAPI icpx | 2024.0+（LLVM 17 基线） |


## 创建项目
我们从新建文件夹开始。比如我们叫它
```bash
mkdir TransverseFieldIsingPEPS
```

我们建议在项目的开始就配置好编译工具。我们也建议采用现代C++标准化的CMake构建编译系统而非Makefile。
当然如果你热爱手搓Makefile也可以。以下我们以CMakeList.txt文件的内容为例说明如何构建编译系统。
手搓Makefile需自行翻译。

构建编译系统的基本要素是引入对上述Prerequisites。因而需包含
```cmake
set(CMAKE_CXX_STANDARD 20)

FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED) 

find_path(TENSOR_HEADER_PATH "qlten")
find_path(MPS_HEADER_PATH "qlmps")
find_path(PEPS_HEADER_PATH "qlpeps")
find_path(hptt_INCLUDE_DIR "hptt.h")
find_library(hptt_LIBRARY "libhptt.a")

include_directories(
        ${MPI_CXX_HEADER_DIR}
        ${BLAS_INCLUDE_DIR}
        ${hptt_INCLUDE_DIR}
        ${TENSOR_HEADER_PATH}
        ${MPS_HEADER_PATH}
        ${PEPS_HEADER_PATH}
)

link_libraries(
        ${hptt_LIBRARY}
        BLAS::BLAS
        LAPACK::LAPACK
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
)
```

推荐在x86_64系统（包括intel和amd）设置set(BLA_VENDOR Intel10_64lp_seq)；Arm64芯片上set(BLA_VENDOR OpenBLAS).
AMD芯片上MKL表现要强过OpenBLAS。为什么不用AOCL-BLAS？因为里面缺矩阵转置操作的函数（属于blas extension）。
日后AOCL-BLAS支持这一函数以后QuantumLiquids/PEPS也会支持。

我们假设上述package每个都有并且只装了一个，这通常在HPC上可以通过module 系统做到。
但是即便如此，还有幺蛾子。cmake标准中
FindBlas(https://cmake.org/cmake/help/latest/module/FindBLAS.html)
并未返回BLAS_INCLUDE_DIR Variable. 这一点需要手动设置。MKL环境可设置为set(BLAS_INCLUDE_DIR "$ENV{MKLROOT}/include")。

接下来，当我们写好后续代码，只需要通过add_executable命令添加二进制文件名和source源码即可。

我们
```bash
mkdir src/
```

## 创建代码

### 从示例复制
  本仓库已提供可直接运行的最小示例， 位于`examples/transverse_field_ising_simple_update.cpp`。在编译运行之前，我们先对代码做一些解读。
  
### 模型与算符（- ZZ - h X）
- 我们使用的哈密顿量是：

  \[ H = - \sum_{\langle i,j \rangle} Z_i Z_j - h \sum_i X_i \]

  其中：\(X = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix},\; Z = \begin{pmatrix}1 & 0 \\ 0 & -1\end{pmatrix}\)

NN两体算符和 on-site 算符是我们要传入给程序的
### NN 两体哈密顿量张量的索引约定（重要）
- 对于最近邻两体算符 \(Z_i Z_j\)，我们在张量里使用 4 条腿，索引顺序固定为：

  \[ (\text{in}_1,\; \text{out}_1,\; \text{in}_2,\; \text{out}_2) \]
其中，下角标1,2标记格点， in/out对应 |cat>/<bra|
  图示约定（数字为索引序号）：

  ```
        1           3          // out_1, out_2
        |           |
        ^           ^
        |----  H  ----|
        ^           ^
        |           |
        0           2          // in_1, in_2
  ```

  在示例代码里，两个 on-site 算子（此处为 Z 与 Z）做外积即可得到该顺序：先是第一个算子的 (in,out)，再接第二个算子的 (in,out)：

```53:55:/Users/wanghaoxin/GitHub/PEPS/examples/transverse_field_ising_simple_update.cpp
    Tensor ham_zz;
    Contract(&opZ, {}, &opZ, {}, &ham_zz); // outer product yields correct index order
    ham_zz *= -1.0;
```

### 使用 SU Executor（SquareLatticeNNSimpleUpdateExecutor）
- 我们采用 `SquareLatticeNNSimpleUpdateExecutor` 来做 Simple Update：
  - NN 两体项传入 `ham_nn = - Z ⊗ Z`（索引顺序如上）。
  - on-site 项传入 `ham_onsite = - h * X`（rank-2）。
  - 这会在内部对每条 NN 键构造 `exp(-τ H_bond)` 并做截断，完成一次扫。


### 将可执行文件添加到你的 CMake（要点：源码里不需要包含 mpi.h，但编译需链接 MPI）
- 如果你在自己的工程里使用该示例，最小化地添加一个目标并链接必要依赖（BLAS/LAPACK/OpenMP/MPI 与 hptt）：

```cmake
add_executable(transverse_field_ising_simple_update
    ${CMAKE_SOURCE_DIR}/examples/transverse_field_ising_simple_update.cpp)

target_link_libraries(transverse_field_ising_simple_update
    ${hptt_LIBRARY}
    BLAS::BLAS
    LAPACK::LAPACK
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX)  # 仅链接，源码无需包含 mpi.h
```

> 说明：qlten 内部使用 MPI，因此虽然示例源码没有 `#include <mpi.h>`，编译/链接阶段仍需要 `MPI::MPI_CXX`。
## 编译
```bash
mkdir build/ && cd build && cmake ..
```
解决cmake ..报错。可以问问GPT
```bash
make
```

## 运行
```bash
./transverse_field_ising_simple_update
```



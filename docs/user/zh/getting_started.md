# Getting Started
本部分将展示如何搞一个二维横场Ising模型的Simple Update作为这个包的开始

## Prerequisites
- **C++ Compiler:** C++20 or above
- **Build System:** CMake (version 3.12 or higher)
- **BLAS/LAPACK:** Intel MKL or OpenBLAS
- **Parallelization:** MPI
- **Tensor&MPS:** [QuantumLiquids/TensorToolkit](https://github.com/QuantumLiquids/TensorToolkit); [QuantumLiquids/UltraDMRG](https://github.com/QuantumLiquids/UltraDMRG)

## 创建项目
臃肿的C++是这样的，当你开始一个项目的时候，需要从新建文件夹开始。比如我们叫它
```bash
mkdir TransverseIsingPEPS
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
兄弟们等我在example给你们放个代码，然后你把代码拷贝过去。
还得给你们一句一句解释。

然后在CMakeList.txt file中放入
```cmake
add_executable(simple_update src/xxxx.cpp)
```
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
./simple_update
```



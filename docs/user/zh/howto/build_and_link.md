# 编译与链接（CMake）

当你想编译一个可执行程序，并链接 BLAS/LAPACK、MPI、OpenMP，同时使用 PEPS + TensorToolkit 时，请阅读本页。

> 同步状态：英文文档（`docs/user/en/`）为权威版本；如与代码行为冲突，请以英文版本和头文件为准。

## 推荐编译器版本

为避免旧编译器中已知问题（例如 lambda + 结构化绑定的兼容性问题），建议至少使用：

| 编译器 | 推荐版本 |
|---|---|
| GCC | 11.0+ |
| Clang/LLVM | 15.0+ |
| Intel oneAPI icpx | 2024.0+（LLVM 17 基线） |

## 从构建角度理解 PEPS

- PEPS 本身在 `include/` 下是头文件库（header-only）。
- 你的可执行文件仍需要链接 **BLAS/LAPACK + MPI + OpenMP**，并且通常需要 `hptt`。

## （可选）用 CMake 安装头文件

如果你更喜欢使用“安装后的 include 树”（而不是在工程里直接引用本仓库路径），可以使用本仓库的 CMake install：

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build . --target install
```

说明：

- 如果不设置 `CMAKE_INSTALL_PREFIX`，很多 Unix 系统默认安装到 `/usr/local`，可能需要管理员权限。
- HPC 环境通常建议安装到用户可写的 prefix。

## 最小 CMake 骨架

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

## BLAS 厂商与 include path 的注意事项

- 在 x86_64 上，MKL 对许多典型工作负载往往比 OpenBLAS 更快；在 Apple Silicon 上通常用 OpenBLAS。
- CMake 的 `FindBLAS` 不一定会填充 include 变量。如果你需要（例如 MKL），可能需要手动设置（例：`set(BLAS_INCLUDE_DIR \"$ENV{MKLROOT}/include\")`）。

## 快速自检：编译 examples

```bash
cd examples
mkdir -p build && cd build
cmake ..
make -j
```

## 相关阅读

- Tutorials：`../tutorials/index.md`


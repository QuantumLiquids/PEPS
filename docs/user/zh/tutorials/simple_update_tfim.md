# Simple Update（TFIM）教程

本教程运行一个最小的 **4×4 横场 Ising 模型**（TFIM）Simple Update，并把得到的 PEPS 导出到磁盘。

源代码：

- `examples/transverse_field_ising_simple_update.cpp`

## 你会得到什么

- 一个可运行的示例程序：`transverse_field_ising_simple_update`
- 一个导出的 PEPS 目录：`./peps/`（下一篇 VMC 教程会用到）

## 前置条件

编译/链接相关说明见：`../howto/build_and_link.md`。

最小需求（典型 HPC 环境）：

- C++20 编译器
- CMake
- BLAS/LAPACK
- MPI + OpenMP（推荐；即使某个示例源码不 `#include <mpi.h>`，底层组件也可能依赖 MPI）
- `hptt`

## 模型与算符约定（重要）

示例实现的哈密顿量为：

\[
H = - \sum_{\langle i,j\rangle} Z_i Z_j - h \sum_i X_i
\]

其中 \(X,Z\) 是 Pauli 矩阵（矩阵元为 ±1）。

对最近邻两体算符 \(-Z\otimes Z\)，代码使用 4 阶张量，索引顺序固定为：

\[
(\text{in}_1,\; \text{out}_1,\; \text{in}_2,\; \text{out}_2).
\]

在示例中，`Contract(&opZ, {}, &opZ, {}, &ham_zz)`（外积）会生成恰好这个顺序。

索引顺序示意（数字为索引位置）：

```
      1            3          // out_1, out_2
      |            |
      ^            ^
      |----  H  ----|
      ^            ^
      |            |
      0            2          // in_1,  in_2
```

术语说明：

- `in` / `out` 对应算符张量的 bra/ket 腿（也可理解为输入/输出腿）。
- 下标 `1,2` 表示该腿属于哪个格点。

## 步骤

### 1）编译 examples

```bash
cd examples
mkdir -p build && cd build
cmake ..
make -j
```

### 2）运行 Simple Update

```bash
cd examples/build
./transverse_field_ising_simple_update
```

预期输出：

- 当前工作目录下生成一个 `peps/` 目录（例如在 `examples/build/peps/`）。
- 终端输出包含 sweep 进度与一个估计能量。
- 重复运行会覆盖当前目录下的 `peps/`。

## 下一步

- 继续做 VMC：`vmc_optimize_tfim.md`


# VMC 优化（TFIM）教程

本教程从 Simple Update 导出的 `./peps/` 开始，对横场 Ising 模型（TFIM）运行 VMC 优化。

源代码：

- `examples/transverse_field_ising_vmc_optimize.cpp`

## 你会得到什么

- 一个可运行的示例程序：`transverse_field_ising_vmc_optimize`
- 一个导出的优化后 `SplitIndexTPS` 目录：`./optimized_tps/`

## 前置条件

1. 你已经运行过 Simple Update，并得到了 `./peps/`：
   - `simple_update_tfim.md`
2. 编译/链接说明见：`../howto/build_and_link.md`

## 关键概念（状态类型）

本仓库中常见的三种状态表示：

- **PEPS**：可能带有显式 bond weight（Simple Update 常见输出形式）。
- **TPS**：不显式携带 bond weight 的张量乘积态（“全局算法”更常用）。
- **SplitIndexTPS**：将物理指标提前拆分的 TPS 变体，用于更快的蒙特卡洛投影/幅度计算。

在蒙特卡洛相关接口中，`SplitIndexTPS` 是主要工作状态。

相关阅读：

- 状态转换：`../howto/state_conversions.md`
- 术语表：`../reference/glossary.md`

## 运行 VMC 需要哪些输入

VMC 优化通常需要提供：

1. **模型能量求解器**（哈密顿量逻辑）：TFIM 使用 `TransverseFieldIsingSquareOBC`。
2. **蒙特卡洛更新器 + 采样参数**（组态如何演化）。
3. **优化算法 + 参数**（SR / Adam / SGD 等）。
4. **收缩参数**（`PEPSParams`，BMPS/TRG 精度与开销权衡）。

本示例为 4×4 的最小演示，很多参数都写死在代码里。

## 示例代码在做什么（快速走读）

TFIM VMC 示例大致按下面步骤执行：

1. 从 `./peps/` 加载由 Simple Update 导出的 `SquareLatticePEPS`。
2. 通过 `ToSplitIndexTPS(...)` 转为 `SplitIndexTPS`（`#include "qlpeps/api/conversions.h"`）。
3. 构造：
   - `MonteCarloParams`（样本数、预热、样本间隔、初始 `Configuration`、可选 `config_dump_path`）
   - `PEPSParams`（BMPS 截断策略/精度）
   - `OptimizerParams`（选择 SR/SGD/Adam 等；本例使用 SR）
4. 组装为 `VMCPEPSOptimizerParams`，并调用一键接口：
   - `auto result = VmcOptimize(params, sitps, MPI_COMM_WORLD, model, MCUpdateSquareNNFullSpaceUpdate{});`
5. 导出优化后的态：
   - `result.state.Dump(params.tps_dump_path);`

性能小提示：

- 示例通过 `hp_numeric::SetTensorManipulationThreads(1)` 强制每个 MPI rank 使用 1 线程，避免集群上线程过度订阅。

## 步骤

### 1）编译 examples（如果你还没编译）

```bash
cd examples
mkdir -p build && cd build
cmake ..
make -j
```

### 2）用 MPI 运行 VMC 优化

```bash
cd examples/build
mpirun -n 4 ./transverse_field_ising_vmc_optimize
```

示例固定使用：

- 格点：4×4
- 局域维数：2
- 横场：`h = 0.5`

预期输出：

- `examples/build/optimized_tps/`：包含优化后的 `SplitIndexTPS`。
- 可选的 `./vmc_configs/`：若示例设置了 `MonteCarloParams.config_dump_path`，会把最终组态导出到此目录。

## 参数直觉（快速理解）

- `MonteCarloParams.total_samples`：所有 MPI rank 的总样本数；引擎内部按 `ceil(total_samples / mpi_size)` 计算每个 rank 的样本数。
- `num_warmup_sweeps`：正式采样前的预热 sweep。
- `sweeps_between_samples`：采样点之间的去相关 sweep 间隔。

对优化器而言：

- SR（stochastic reconfiguration）每步要解一个线性方程；CG 参数很重要。
- 若运行噪声大/不稳定，通常先提升收缩精度，并增加预热/样本数。

### 示例参数速查

蒙特卡洛参数：

| 字段 | 含义 |
|---|---|
| `total_samples` | 所有 rank 的总样本数 |
| `num_warmup_sweeps` | 预热 sweep 数 |
| `sweeps_between_samples` | 两次采样之间的间隔 sweep 数 |
| `initial_config` | 初始组态（本示例为“半上半下”的随机 Ising 组态） |
| `is_warmed_up` | 初始组态是否已平衡 |
| `config_dump_path` | 可选：导出最终组态的目录 |

SR（stochastic reconfiguration）使用的 CG 参数：

| 字段 | 含义 |
|---|---|
| `max_iter` | CG 最大迭代数 |
| `tolerance` | CG 残差收敛阈值 |
| `restart` | CG 重启间隔 |
| `diag_shift` | 对角正则（改善条件数） |

## 下一步

- 测量可观测量：`mc_measure_tfim.md`


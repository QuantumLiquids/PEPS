# 端到端工作流：Simple Update → VMC → Measurement

本页描述本仓库推荐的端到端使用路径：

1. 用 **Simple Update** 得到一个可用的初始 PEPS（快、通常不需要 MPI）。
2. 转换为 `SplitIndexTPS`（蒙特卡洛采样主要使用的表示）。
3. 运行 **VMC 优化**（MPI 并行采样，SR/一阶优化器）。
4. 对优化后态做 **蒙特卡洛测量**。

如果你只想跑 TFIM 演示，直接按教程走：

- Simple Update（TFIM）：`simple_update_tfim.md`
- VMC 优化（TFIM）：`vmc_optimize_tfim.md`
- 蒙特卡洛测量（TFIM）：`mc_measure_tfim.md`

## 适用范围

- 有限尺寸方格晶格的工作流目前支持最完整（尤其是 OBC/BMPS）。
- 只要满足接口契约，模型能量求解器与蒙特卡洛更新器都可以自由替换。
- 支持单进程或多进程 MPI（采样天然并行）。

## 分阶段说明（每一阶段会产出什么）

### Step 1：Simple Update（初态）

- 输出：一个 PEPS 导出目录（示例使用 `./peps/`）。
- 目的：快速得到一个“可用”的初态；VMC 通常会在此基础上进一步降低能量。

### Step 2：转换为 `SplitIndexTPS`（用于蒙特卡洛）

- 推荐：使用显式转换函数（`#include "qlpeps/api/conversions.h"`）。
- 实践中建议：做一次轻量的规范化/缩放，让采样扇区里的典型幅度为 \(O(1)\)。

相关阅读：`../howto/state_conversions.md`。

### Step 3：VMC 优化

- 输入：`SplitIndexTPS`、模型能量求解器、更新器，以及参数结构体：
  - `OptimizerParams`、`MonteCarloParams`、`PEPSParams`，组合到 `VMCPEPSOptimizerParams` 中。
- 输出：导出优化后的 `SplitIndexTPS`（示例使用 `./optimized_tps/`）。

相关阅读：`../explanation/optimizer_algorithms.md`、`../howto/choose_mc_updater.md`、`../howto/data_persistence.md`。

### Step 4：蒙特卡洛测量

- 输入：优化后的 `SplitIndexTPS` + 测量求解器（很多模型类同时实现能量求解与测量接口）。
- 输出：聚合后的 CSV 统计结果写入 `measurement_data_dump_path`（示例使用 `./mc_measure_output/stats/`）。

相关阅读：`../reference/model_observables_registry.md`。

## 输出路径（示例默认）

TFIM 示例使用的默认目录：

- Simple Update：`./peps/`
- VMC 优化：`./optimized_tps/`
- 测量结果：`./mc_measure_output/stats/`

## 常见坑

- **幅度尺度问题**：转换后若幅度极小/极大，采样会更脆弱；尝试做规范化/缩放（见 `../howto/state_conversions.md`）。
- **收缩截断过猛**：若 `psi-consistency` 警告很大，通常需要提高 BMPS/TRG 精度。
- **更新器/求解器不匹配**：例如更新器只在守恒扇区里走，而求解器假设全空间（或相反）。

## 相关阅读

- 状态转换：`../howto/state_conversions.md`
- 顶层 API：`../howto/top_level_apis.md`
- 数据持久化/导出：`../howto/data_persistence.md`

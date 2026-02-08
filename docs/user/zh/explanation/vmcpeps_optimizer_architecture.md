# VMCPEPSOptimizer：架构与扩展点

## 它是什么

`VMCPEPSOptimizer` 是本仓库中用于 **PEPS/TPS 的变分蒙特卡洛（VMC）优化** 的执行引擎。
它的设计目标是“小而可预期”：你提供若干 *策略组件*，执行器负责把它们按固定流程组织起来。

从高层看，一次 VMC 运行总是由三个核心想法组成：

1. **组态如何演化**（Monte Carlo sweep updater）
2. **能量/梯度如何计算**（模型能量求解器）
3. **参数如何更新**（优化算法 + 参数）

用户常用入口：

- 一键接口：`VmcOptimize(...)`（`include/qlpeps/api/vmc_api.h`）
- 底层类模板：`VMCPEPSOptimizer<...>`（`include/qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h`）

## 心智模型（运行时发生了什么）

概念上的执行循环是：

1. **初始化** `SplitIndexTPS` 状态与初始 `Configuration`。
2. 对每次优化迭代：
   1. 每个 MPI rank 跑蒙特卡洛 sweep 生成样本（独立马尔可夫链）。
   2. 对每个样本，**能量求解器**计算：
      - 局域能量 \(E_{\mathrm{loc}}(S)\)，以及
      - （可选）用于梯度/SR 的 “holes”。
   3. 把样本统计量在 MPI ranks 间做 **聚合**。
   4. **优化算法**更新态参数（SR / Adam / SGD / …）。
   5. 更新后的波函数需要让所有 ranks 在下一轮迭代中一致使用。

两个实用的后果：

- 收缩截断（BMPS/TRG）可能导致不同收缩位置得到的波函数幅度略有不一致，进而出现 `psi-consistency` 警告。出现警告时通常需要提高收缩精度。见：`model_energy_solver_math.md`。
- 转换到 `SplitIndexTPS` 后做轻量的规范化/缩放，常常能改善数值稳定性。见：`../howto/state_conversions.md`。

## 组件契约（你需要提供什么）

### 1）Monte Carlo sweep updater

更新器是一个 functor，负责提出/接受组态更新，并保持缓存数据一致。

契约（概念性职责）：

- 更新组态（`tps_component.config`）。
- 更新缓存的波函数幅度（`tps_component.amplitude`）以及框架期望的缓存收缩对象。
- 维护详细平衡（并在目标扇区里遍历充分/可达）。
- 写出接受率等诊断信息。

从这里开始：

- 选型：`../howto/choose_mc_updater.md`
- 自定义更新器（PXP 例子）：`../howto/write_mc_updater_pxp.md`
- 顶层 API：`../howto/top_level_apis.md`

### 2）模型能量求解器

能量求解器封装哈密顿量逻辑，并负责局域能量的约定（复数波函数下尤其关键）。

从这里开始：

- 数学与约定：`model_energy_solver_math.md`
- 自定义求解器：`../howto/write_custom_energy_solver.md`

### 3）优化算法与参数

优化算法由 `OptimizerParams` 选择（SR / Adam / SGD / AdaGrad / L-BFGS），并支持学习率调度与可选的梯度预处理。

从这里开始：

- 算法与数学：`optimizer_algorithms.md`
- 参数设置：`../howto/set_optimizer_parameter.md`

## MPI 语义（需要知道的事）

采样是天然并行的，但并不是“各 rank 完全独立”：

- 每个 rank 跑自己的马尔可夫链并贡献样本。
- 聚合阶段是 collective（能量/梯度估计需要跨 rank 归约）。
- 状态更新需要协调，使得所有 ranks 在下一轮迭代中从同一个更新后的波函数出发。

集群上的实践建议：

- 避免线程过度订阅：示例中通过 `hp_numeric::SetTensorManipulationThreads(1)` 强制每个 rank 使用 1 线程。

## 从哪里开始（用户入口）

- 端到端：`../tutorials/end_to_end_workflow.md`
- 教程：`../tutorials/index.md`
- How-to：`../howto/index.md`
- Reference：`../reference/index.md`

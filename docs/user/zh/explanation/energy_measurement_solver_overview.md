# 能量与测量求解器概览

当你想理解“能量求解器（energy solver）”与“测量求解器（measurement solver）”如何嵌入 VMC + 测量流程时，请阅读本页。

## 后端说明（OBC/BMPS 与 PBC/TRG）

- OBC 工作流使用 BMPS 收缩。
- PBC 工作流使用 TRG 收缩。

在公共一键接口（`VmcOptimize` / `MonteCarloMeasure`）中，`SplitIndexTPS` 的边界条件会与 `PEPSParams` 交叉检查，不匹配时会直接报错。

## 能量求解器（VMC）

能量求解器负责：

- 对给定的蒙特卡洛组态计算局域能量；
- （可选）为优化计算梯度相关的张量（“holes”）。

它作为 `EnergySolver` 模板参数传给 `VMCPEPSOptimizer` / `VmcOptimize(...)`。

## 测量求解器（测量）

测量求解器负责：

- 对每个蒙特卡洛组态计算可观测量；
- 通过 `DescribeObservables()` 提供元数据（registry key + shape）。

它传给 `MCPEPSMeasurer` / `MonteCarloMeasure(...)`。

## 相关阅读

- 数学与约定（复数梯度）：`model_energy_solver_math.md`
- 自定义能量求解器：`../howto/write_custom_energy_solver.md`
- registry key 参考：`../reference/model_observables_registry.md`

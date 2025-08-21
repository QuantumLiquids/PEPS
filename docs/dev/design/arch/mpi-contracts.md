---
title: MPI Contracts for Optimizer / Evaluator / Executor
last_updated: 2025-08-21
---

目标：用一页清单明确 MPI 职责边界与“主/全进程有效性”契约，避免隐式广播与重复归约。

约定（术语）
- 主节点/全进程有效：返回值或副作用在哪个范围内可被依赖。
- 广播/归约所有权：哪个组件负责发起 MPI_Bcast/MPI_Reduce/Allreduce 等。
- Master：主节点（rank 0）。“仅主进程有效”指仅 master 的值可被依赖。
- SR 工件：O* 样本与其均值。（由评估器在各进程收集并提供给 Optimizer。）

总则
- 谁使用，谁分发（广播状态由使用该状态的人负责）。
- 优化器不广播波函数，仅在主进程更新状态；SR 内部并行仅限算法矩阵-向量计算。
- 评估器负责状态广播与采样归约；是否广播能量属于实现细节，不在契约内。

组件契约
1) Optimizer（含 SR）
- 输入：来自 Evaluator 的 (energy, gradient)，仅主进程有效。SR类算法中，由所有节点的Evaluator共同提供sr_artifacts 给Optimizer.
- 行为：在主进程基于梯度/自然梯度更新状态；SR 可在内部使用 MPI 求解线性系统。
- 输出：更新后的状态仅主进程有效；不发起状态广播。

2) Energy Evaluator（MC 或 Exact-Sum）
- 输入：状态仅主进程有效（来自 Optimizer）。
- 行为：
  - 广播状态到所有进程（单次）。
  - 在各进程执行采样/枚举与本地累加。SR 算法保留采样的 O* 样本与其均值，用于后续提供给 Optimizer。
  - 归约标量到主进程；梯度张量在主进程收集/累加。
- 输出：返回 (energy, gradient, error)，仅主进程有效。

(Energy Evaluator计算过程中如需用于协方差项一致性的计算，应在评估器内部广播能量 E。这属于评估器实现细节，对外接口仍以主节点为准。)

3) Executor（VMC/Measurement）
- 输入：主进程持有“最新状态”。
- 行为：负责在需要时做最终状态同步与持久化（不在每次迭代内）。
- 输出：对外可选的全进程可见副作用（落盘/日志）。

工作流
每次迭代：
1. Energy Evaluator 接收来自 Optimizer 的状态（仅 master 持有）。
2. Energy Evaluator 广播状态至所有进程（单次）。
3. 各进程并行采样/枚举并本地累加；SR 记录 O* 样本与其均值。
4. 将能量/误差等标量归约到 master；梯度张量在 master 收集/累加；SR 额外将 O* 样本与其均值提供给 Optimizer。
5. Optimizer 在 master 基于（自然）梯度更新状态；SR 内部如需用 MPI 求解线性系统，但不广播状态。
6. Executor 仅在需要时做最终状态同步与持久化（非每迭代），然后进入下一轮。

有效性矩阵（主/全进程）
| 项目 | Optimizer | Evaluator | Executor |
| --- | --- | --- | --- |
| 输入态 | 主 | 主→评估器内部广播 | 主 |
| 能量E | 主 | 主 | 可选全（日志/可视化） |
| 梯度∇E | 主 | 主 | - |
| O*样本(仅适用SR) | 全 | 全 | - |
| 状态更新 | 主 | - | 最终全（按需） |

错误与负例（必须检测）
- 评估器未广播状态：多进程结果不一致（检测能量轨迹差异）。
- Optimizer 在非 SR 情况下发起状态广播：多余通信（禁用）。
- 评估器返回的能量依赖内部广播但外部又广播：重复通信（告警）。

测试要点
- 单/多进程一致性：固定随机种子时，E/∥∇∥ 轨迹在主节点一致。


参考
- 设计背景：`design/arch/overview_cn.md`
- 示例评估器：`include/qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h`



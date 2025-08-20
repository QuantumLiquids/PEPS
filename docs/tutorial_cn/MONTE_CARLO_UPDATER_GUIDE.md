# 蒙特卡洛更新器指南

## 概述

蒙特卡洛更新器定义在采样过程中“组态如何演化”。它是 VMC 框架的第一环节：给定当前组态与 TPS 状态，提出并执行局域更新，维持详细平衡与遍历性，从而生成无偏样本。

本指南给出接口契约、内置类型、选型建议与最小使用范式。进阶算法与推导后续单独补充。

## 蒙特卡洛基础：顺序更新为何成立（Balance vs Detailed Balance）

顺序（sequential）扫描更新常被质疑“破坏 detailed balance”。事实是确实如此：顺序扫描通常确实不满足逐步的 detailed balance（逐步等式对称），但只要整体满足 balance condition，即可保证目标分布不变性并收敛到正确平衡态。

记目标分布为 \(\pi(x)\)。
- 详细平衡（detailed balance）：\(\pi(x) P(x\to y) = \pi(y) P(y\to x)\)。
- 平衡条件（balance）：\(\sum_x \pi(x) P(x\to y) = \pi(y)\)。

关键结论：
- 若单步核 \(K_i\) 各自满足对 \(\pi\) 的平衡条件，则复合核 \(K = K_m \cdots K_2 K_1\) 仍保持 \(\pi\) 不变。因此“先横后纵”的顺序扫描是可行的。
- 对于“多候选局域更新”，可以采用满足 balance 而非逐项 detailed balance 的选择核；实现细节与实例请见《蒙特卡洛更新器：自定义（PXP）基础篇》。

实践要点：
- 顺序与遍历顺序需与当前状态无关，避免违反 balance condition。
- 若使用 NonDB 的多局部候选选择，务必遵循固定局部候选列表与顺序的约束，否则将引入非常难以发现但致命的偏差。

## 接口契约

```cpp
template<typename TenElemT, typename QNT>
void operator()(const SplitIndexTPS<TenElemT, QNT>& sitps,
                TPSWaveFunctionComponent<TenElemT, QNT>& tps_component,
                std::vector<double>& accept_rates);
```

必须遵守的职责与不变量：
- 更新 `tps_component.config`：粒子/自旋组态。
- 更新 `tps_component.amplitude`：对应新组态的波函数振幅。
- 更新 `tps_component.tn`：表示该组态的单层二维张量网络缓存。
- 填写 `accept_rates` 以便诊断（按“横向键后纵向键”的遍历顺序）。
- 维持 蒙特卡洛的 balance condition；不要在算子内部做跨进程通信。

建议：
- 同一随机种子与相同初态应给出可复现实验结果。
- 对于守恒律（如 U(1) 粒子数/磁量子数）要么显式遵守，要么在文档中明确声明不守恒并说明适用模型。

## 内置更新器

位于 `include/qlpeps/vmc_basic/configuration_update_strategies/`。

### 1) MCUpdateSquareNNExchange
- 适用：自旋 1/2 海森堡、t-J 等 U(1) 守恒且通过交换可遍历采样空间的模型。
- 思路：依次遍历相邻格点对（先横后纵），提出交换更新；按振幅比决定接受，满足详细平衡。
- 优势：高效、限制到守恒子空间，通常更快。

### 2) MCUpdateSquareNNFullSpaceUpdate
- 适用：几乎所有模型，尤其无严格守恒律（如横场 Ising）。
- 思路：对每个键考虑所有局域配置，按权重采样。
- 备注：对 PXP 一类“特殊约束”模型不适配（需专用投影）。

### 3) MCUpdateSquareTNN3SiteExchange
- 适用：提升交换更新的接受率（阻挫系统常见）。
- 思路：遍历相邻三格点（先横后纵），进行三体交换提议。

## 选型建议（极简决策树）

```
只靠交换即可遍历目标子空间？
├─ 是 → 交换接受率足够高？
│   ├─ 是 → MCUpdateSquareNNExchange
│   └─ 否 → MCUpdateSquareTNN3SiteExchange
└─ 否 → MCUpdateSquareNNFullSpaceUpdate
```

注意：蒙特卡洛更新器引入的希尔伯特子空间约束一般不应少于张量网络的对称性约束。

## 最小使用范式

```cpp
using UpdaterType = MCUpdateSquareNNExchange; // 或 FullSpace / 3Site

VMCPEPSOptimizerExecutor<TenElemT, QNT, UpdaterType, EnergySolver>
  executor(vmc_params, initial_tps, MPI_COMM_WORLD, solver);
executor.Execute();
```

## 自定义更新器

实现上述 `operator()` 即可，务必：
- 明确遍历顺序与随机性来源；
- 保证 detailed balance 与期望的遍历性；
- 不在算子内部做全局通信；
- 正确维护 `config / amplitude / tn / accept_rates` 的一致性。

接口正确性比“聪明的优化”更重要。先写对，再优化。

进一步示例请参考：
- 《蒙特卡洛更新器：自定义（PXP）基础篇》：`MONTE_CARLO_UPDATER_PXP_CUSTOM_BASICS.md`

## 常见陷阱

- 组件不匹配：更新器不守恒但能量求解器假设守恒，导致偏差。
- 缓存未同步：更新了 `config` 却忘了更新 `amplitude` 或 `tn`。
- 误用通信：在更新器内部做 MPI 通信，破坏“尴尬并行”的采样模型。

## 与 MPI 的关系

采样是“尴尬并行”的：每个进程独立跑自己的 Markov 链；优化阶段通过 `MPI_Allreduce` 汇总统计，再广播更新后的参数。因而蒙卡更新器不必考虑MPI效果。

---

本篇为基础指南。更复杂的 cluster/loop 更新、硬约束模型、以及混合策略将在进阶文档中给出。



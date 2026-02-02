# 选择蒙特卡洛更新器（sweep updater）

当你在为 **VMC 优化**或**蒙特卡洛测量**选择 sweep updater 时，请阅读本页。

在本仓库中，“updater”定义了采样过程中 **Configuration 如何演化**。选型会影响：

- **正确性**：马尔可夫链必须保持目标分布不变。
- **效率**：接受率与（实践意义上的）遍历性决定收敛速度。

## 正确性：balance 与 detailed balance（顺序扫）

很多人担心“按固定顺序扫”（例如先横键再竖键）会破坏 detailed balance。逐步来看通常确实如此，但只要整体转移核保持目标分布不变，算法仍然是正确的（满足 balance/stationarity）。

记目标分布为 \(\pi(x)\)：

- Detailed balance：\(\pi(x) P(x\to y) = \pi(y) P(y\to x)\)
- Balance（stationarity）：\(\sum_x \pi(x) P(x\to y) = \pi(y)\)

关键点：

- 如果每个子核 \(K_i\) 都保持 \(\pi\)，那么它们的复合核 \(K = K_m \cdots K_2 K_1\) 也保持 \(\pi\)。
- 这解释了为什么“固定顺序扫”在很多实现中是正确的，即使每个微小步本身不是对称的。

实用规则：

- sweep 的顺序必须**不依赖当前状态**。
- 对于非详细平衡（NonDB）的“多候选局部选择”，候选集合与其**顺序必须固定**，不能随当前状态重排。

## Updater 接口契约

方格晶格的 sweep updater 都是 functor，契约为：

```cpp
template<typename TenElemT, typename QNT>
void operator()(const SplitIndexTPS<TenElemT, QNT>& sitps,
                TPSWaveFunctionComponent<TenElemT, QNT>& tps_component,
                std::vector<double>& accept_rates);
```

你必须一致地更新：

- `tps_component.config`（组态）
- `tps_component.amplitude`（该组态下的波函数幅度）
- `tps_component.tn`（该组态对应的缓存单层张量网络）
- `accept_rates`（诊断信息，通常按 sweep 记录）

不要在 updater 内做跨 rank 通信。

## 内置更新器（方格晶格）

头文件位于：

- `include/qlpeps/vmc_basic/configuration_update_strategies/`

### `MCUpdateSquareNNExchange`

- 适合：存在守恒律、且交换更新在目标扇区里遍历性好（例如 Heisenberg，许多 t-J 设置）。
- 思路：提出最近邻交换，按幅度比接受/拒绝（详细平衡式）。
- 优点：限制在守恒扇区采样，通常更高效。

### `MCUpdateSquareNNFullSpaceUpdate`

- 适合：无严格守恒律（例如 TFIM）或交换更新不遍历。
- 思路：对每条键枚举局部替代，并按权重抽样（full local state space）。
- 说明：硬约束模型（例如 PXP）通常需要自定义投影/候选选择策略。

### `MCUpdateSquareTNN3SiteExchange`

- 适合：在受阻/阻挫系统中提升接受率（NN exchange 太“粘”时）。
- 思路：在三站点三角形 plaquette 图案中做 exchange move（仍按固定顺序扫）。

## 最小决策树

```
你的模型是否守恒粒子数/磁化？
├─ 是 → NN 交换是否遍历且接受率好？
│   ├─ 是 → MCUpdateSquareNNExchange
│   └─ 否 → MCUpdateSquareTNN3SiteExchange
└─ 否 → MCUpdateSquareNNFullSpaceUpdate
```

## 最小用法范式

```cpp
using Updater = MCUpdateSquareNNFullSpaceUpdate; // 例子
auto result = VmcOptimize(params, sitps, MPI_COMM_WORLD, solver, Updater{});
```

## 常见坑

- **契约不一致**：更新了 `config` 但忘了更新 `amplitude` / `tn`。
- **隐蔽偏差**：NonDB 的候选集合/顺序依赖当前状态。
- **物理不匹配**：updater 强制守恒扇区，但模型/求解器假设全空间（或反之）。

## 与 MPI 的关系

采样是“尴尬并行”的：每个 rank 独立跑自己的马尔可夫链，统计量在外层做聚合。updater 不需要关心 MPI。

## 后端说明（OBC/BMPS 与 PBC/TRG）

更新器需要与状态与截断参数所暗含的收缩后端相匹配：

- OBC 使用 BMPS 收缩；
- PBC 使用 TRG 收缩。

公共一键接口（`VmcOptimize` / `MonteCarloMeasure`）会将 `sitps.GetBoundaryCondition()` 与 `PEPSParams` 交叉检查，不匹配时直接抛异常。

## 相关阅读

- 顶层 API：`top_level_apis.md`
- 自定义 PXP 更新器示例：`write_mc_updater_pxp.md`
- VMC 架构（updater 在哪里）：`../explanation/vmcpeps_optimizer_architecture.md`

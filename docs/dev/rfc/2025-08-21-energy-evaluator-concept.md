---
title: Energy Evaluator Concept (C++20)
status: draft
last_updated: 2025-08-21
applies_to: [module/optimizer, module/vmc_update]
tags: [design, rfc, mpi, concepts]
---

# Energy Evaluator 概念化（C++20 concept）提案

类型: 设计提案 (RFC)

## 背景与目标
- 背景：当前 `Optimizer::IterativeOptimize` 通过 `std::function<std::tuple<E, Grad, Err>(const State&)>` 接收能量评估器。虽然灵活，但对 MPI 职责边界依赖文档约定，存在自定义评估器“破约”的风险。
- 目标：用 C++20 concept 固定评估器的最小接口和语义契约，使错误在编译期暴露，减少运行期不一致。

## 范围与不做的事
- 范围：仅约束“评估器接口 + MPI 语义契约”；不改变现有外部 API（短期），不强制改写现有实现。
- 不做：此提案不引入运行时多态、不重塑执行器/优化器的数据流，不要求立刻迁移所有调用点。

## 概念定义（草案）
```cpp
// 假设别名
using Ten = TenElemT;
using State = SplitIndexTPS<TenElemT, QNT>;

// 评估器概念：提供 Evaluate(const State&) -> tuple<E, Grad, Err>
// 语义契约详见下节
template<class Eval>
concept EnergyEvaluator = requires(Eval e, const State& s) {
  { e.Evaluate(s) } -> std::same_as<std::tuple<Ten, State, double>>;
};

// 可选：对 SR 友好（非强制），暴露是否需要 O* 样本
template<class Eval>
concept EnergyEvaluatorSRHints = requires(const Eval& e) {
  { e.NeedsOStarSamples() } -> std::same_as<bool>;
};
```

说明：
- 返回的第二个元素用 `State` 表示梯度张量容器（与现有实现一致）。
- 若未来需要扩展误差类型，可用 traits 替代硬编码的 `double`。

## 语义契约（强制）
- 输入：`state` 仅保证在 master 节点有效。
- 评估器职责：
  - 必须在内部完成“单次状态广播”，保证所有进程用于 MC 采样的一致性。
  - 必须在所有进程执行 MC 采样与本地累加；并在评估内部或返回前完成梯度的标准归约，使“返回的梯度在 master 节点有效”。
- 输出：
  - `(energy, gradient, error)`；其中 `gradient` 仅 master 有效，`error` 仅 master 有效。
  - “能量 energy 的广播”不作为评估器对外契约，调用方（执行器）可自行选择是否广播（当前实现为执行器广播）。

## SR 兼容（说明性，不强制）
- 当采用 SR（自然梯度）算法时，优化器需要 `O*` 样本集合与其均值。
- 现状：执行器在采样阶段内部构造 `Ostar_samples_` 与 `Ostar_mean_` 并传入优化器；评估器无需改变返回签名。
- 可选扩展：若 `EnergyEvaluatorSRHints` 可用，则执行器可在运行期通过 `NeedsOStarSamples()` 决定是否收集 `O*` 样本，避免执行器中显式分支写法；该扩展不改变已有路径。

## 典型使用（不改变现有 API）
```cpp
// 维持现有接口（std::function）以兼容旧代码
OptimizationResult IterativeOptimize(
  const State& initial_state,
  std::function<std::tuple<Ten, State, double>(const State&)> energy_evaluator,
  const OptimizationCallback& callback = {}
  /* SR inputs kept as-is */
);

// 新增模板重载：仅在调用点显式传入“满足 concept 的评估器实例”时启用
template<class Evaluator>
  requires EnergyEvaluator<Evaluator>
OptimizationResult IterativeOptimize(
  const State& initial_state,
  Evaluator&& energy_evaluator,
  const OptimizationCallback& callback = {}
  /* SR inputs kept as-is */
);
```

- 兼容策略：
  - 旧调用（`std::function`）不变；
  - 新调用（满足 concept 的类型）享受编译期检查；
  - 评估器的 MPI 语义由文档和概念共同约束。

## 迁移与实施计划
- 阶段 1（本提案阶段）：仅文档与示例，不改动生产代码。
- 阶段 2（可选）：添加 `IterativeOptimize` 概念重载；不移除旧重载。
- 阶段 3（可选）：为内置评估器（如精确求和）提供 `Evaluate` 方法适配；添加单测覆盖“master-only state + 状态广播 + 梯度规约”契约。

## 测试要点
- 单进程与多进程一致性：能量轨迹、梯度范数一致。
- 状态广播缺失的负例：概念版本应在类型层通过；运行时版本在测试中失败。
- SR on/off：`O*` 样本开关行为正确，数值不变。

## 风险评估
- 二重接口（`std::function` 与模板重载）会增加维护点，但换取了平滑迁移与编译期保障。
- 若未来统一到概念重载，可 deprecate 旧接口。

## 结论
- 使用 C++20 concept 限定能量评估器接口是低侵入、高收益的结构性改进。
- 建议按“文档→模板重载→内置适配”的顺序推进，确保零破坏性。



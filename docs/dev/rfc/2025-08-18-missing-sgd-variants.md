---
title: Missing SGD Variants and Line Search
status: documented
date: 2025-08-18
last_updated: 2025-08-24
applies_to: [module/optimizer]
tags: [rfc, optimizer]
---

## 变更记录

- 2025-08-24: 已实现 Clip-by-Value 梯度预处理；删除相关提案段落（原“2. ClipValueSGD(rename from legacy BoundGradientElement)”以及文末“独立实现计划：梯度预处理（裁剪）”）。

# Missing Optimization Algorithms from Legacy VMCPEPSExecutor

## Status: PARTIALLY IMPLEMENTED — 已完成 Clip-by-Value；其余项待实现

## 概述

在从 `VMCPEPSExecutor` 迁移到 `VMCPEPSOptimizer` 的过程中，发现有多个特殊的优化算法未在新的优化器架构中实现：

### SGD变体与梯度预处理：
1. **RandomGradientElement** - 随机化梯度元素的幅值但保留相位

### Line Search算法：
2. **GradientLineSearch** - 梯度方向的线搜索优化
3. **NaturalGradientLineSearch** - 自然梯度方向的线搜索优化

## 详细功能说明

### 1. RandomGradientElement

**功能**: 对梯度的每个tensor element随机化其magnitude但保留phase/sign。

**原始实现** (来自 `vmc_peps_impl.h:755-764`):
```cpp
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::GradientRandElementSign_() {
  if (rank_ == kMPIMasterRank)
    for (size_t row = 0; row < ly_; row++) {
      for (size_t col = 0; col < lx_; col++) {
        size_t dim = split_index_tps_({row, col}).size();
        for (size_t i = 0; i < dim; i++)
          grad_({row, col})[i].ElementWiseRandomizeMagnitudePreservePhase(unit_even_distribution, random_engine_);
      }
    }
}
```

**使用场景**: 在梯度下降过程中引入随机性，可能有助于跳出局部最优解。

**调用方式** (原始实现):
```cpp
case RandomGradientElement: {
  GradientRandElementSign_();
  UpdateTPSByVecAndSynchronize_(grad_, step_len);
  break;
}
```

### 2. GradientLineSearch

**功能**: 沿着梯度方向进行线搜索优化，寻找最优步长。

**原始enum定义**: `GradientLineSearch = 7`

**使用场景**: 
- 提高梯度下降的收敛性
- 自动选择最优步长，避免手动调参
- 适用于对收敛速度要求较高的场景

**特点**:
- 使用标准梯度方向作为搜索方向
- 通过多次能量评估寻找最优步长
- 比固定步长SGD更稳定

**建议（MC-heavy 场景的务实策略）**:
- 低频触发：每 N 步（如 N∈{10,20,50}）才执行一次线搜索，平时用固定/调度学习率；
- 双档/少档候选步长：仅试 {η, η/2} 或 {η, η/2, η/4}；
- 触发条件：仅当近期能量显著恶化时触发（避免常态开销）；
- 能量方差约束：若测得方差过大，直接跳过线搜索结果，沿保守步长前进；
- 停止策略：第一次改善即停（first improvement），避免无谓探测；
- 方向选择：目前仅用梯度方向，未来可选“动量方向”线搜索（需维护 look-ahead 的 velocity 状态）。

### 3. NaturalGradientLineSearch

**功能**: 沿着自然梯度(natural gradient)方向进行线搜索优化。

**原始enum定义**: `NaturalGradientLineSearch = 8`

**使用场景**:
- 结合自然梯度的优势和线搜索的稳定性
- 适用于需要高精度优化的物理问题
- 特别适合处理ill-conditioned的优化landscape

**特点**:
- 使用Stochastic Reconfiguration计算自然梯度方向
- 在自然梯度方向上进行线搜索
- 比标准线搜索更加physically motivated

## 实现计划

### Phase 1: 扩展OptimizerParams

需要在 `OptimizerParams` 中添加对这些算法变体的支持：

#### SGD变体参数与梯度预处理：

```cpp
/**
 * @struct RandomizedSGDParams  
 * @brief Parameters for randomized SGD (RandomGradientElement variant)
 */
struct RandomizedSGDParams {
  double momentum;
  bool nesterov;
  std::mt19937* random_engine;  // RNG handle
  
  RandomizedSGDParams(double momentum = 0.0, bool nesterov = false, 
                      std::mt19937* engine = nullptr)
      : momentum(momentum), nesterov(nesterov), random_engine(engine) {}
};

// Modern design: use independent gradient preprocessing instead of a separate
// ClipValueSGD algorithm variant. Extend BaseParams with optional clip fields.
//
// struct OptimizerParams::BaseParams {
//   ...
//   std::optional<double> clip_value;  // per-element value clipping threshold
//   std::optional<double> clip_norm;   // global norm clipping threshold
// };
```

### Phase 2: 扩展参数承载（AlgorithmParams 保持不变）

```cpp
using AlgorithmParams = std::variant<
  SGDParams,
  AdamParams, 
  StochasticReconfigurationParams,
  LBFGSParams,
  AdaGradParams,
  RandomizedSGDParams    // 如需；RandomizedSign 后续另起 RFC
>;
// 裁剪作为 BaseParams 的独立预处理选项承载，不新增算法变体。
```

### Phase 3: 在Optimizer中实现算法逻辑

需要在 `Optimizer` 类中添加对应的update方法：

```cpp
// 在 optimizer_impl.h 中添加
template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::ApplyRandomizedSGDUpdate(
    SplitIndexTPS<TenElemT, QNT>& state,
    const SplitIndexTPS<TenElemT, QNT>& gradient,
    const RandomizedSGDParams& params) {
  auto randomized_grad = gradient;
  RandomizeGradientElements(randomized_grad, params.random_engine);
  ApplySGDUpdate(state, randomized_grad, static_cast<SGDParams>(params));
}

// 梯度预处理（独立于算法）：在通用更新路径中应用（示意）
// SITPST processed_grad = gradient;
// if (base.clip_value) { ElementWiseClipTo(processed_grad, *base.clip_value); }
// if (base.clip_norm)  { ClipByGlobalNorm(processed_grad, *base.clip_norm); }
// 然后进入算法分派（SGD/Adam/AdaGrad/...），SR/L-BFGS 默认跳过裁剪。
```

### Phase 4: 扩展OptimizerParamsBuilder

```cpp
class OptimizerParamsBuilder {
  // ...
  
  OptimizerParamsBuilder& WithRandomizedSGD(double momentum = 0.0, 
                                           bool nesterov = false,
                                           std::mt19937* engine = nullptr) {
    RandomizedSGDParams sgd_params(momentum, nesterov, engine);
    algorithm_params_ = sgd_params;
    return *this;
  }

  // 裁剪作为独立预处理选项（默认不启用）
  OptimizerParamsBuilder& SetClipValue(double clip_value) {
    if (!base_params_) { base_params_ = OptimizerParams::BaseParams(1000, 1e-10, 1e-30, 20, 0.01); }
    base_params_->clip_value = clip_value;
    return *this;
  }

  OptimizerParamsBuilder& SetClipNorm(double clip_norm) {
    if (!base_params_) { base_params_ = OptimizerParams::BaseParams(1000, 1e-10, 1e-30, 20, 0.01); }
    base_params_->clip_norm = clip_norm;
    return *this;
  }
};
```

## 优先级

**Priority: LOW-MEDIUM** - 这些是specialized功能，不是核心功能
- 不影响基本的优化算法使用
- 主要用于高级调优和特殊场景
- 可以作为后续enhancement实现

## 实施时间线

建议在以下milestone后实现：
1. ✅ 完成基本的Legacy清理和迁移
2. ✅ 确保所有现有测试通过  
3. ✅ 基本优化算法(SGD, Adam, SR, L-BFGS)功能完整
4. 🔄 然后实现这些specialized SGD variants

**预计工作量**: 2-3天

## 测试策略

需要添加相应的单元测试：
1. 测试RandomizedSGD的随机性效果
2. 对比新实现与legacy实现的数值一致性
3. 验证MPI环境下的正确性

## 注意事项

1. **随机数生成器管理**: RandomizedSGD需要careful管理随机数状态
2. **MPI**: 1阶方法仅对Master进程生效。
3. **Tensor API依赖**: 依赖于 `ElementWiseRandomizeMagnitudePreservePhase`、`ElementWiseClipTo`、`ClipByGlobalNorm`（由SplitIndexTPS负责实现）。
4. **默认与作用范围**: `clip_value/clip_norm` 默认未设置即不启用；空出clip_scope作为place_holder，但预处理仅对一阶方法生效（内部 clip_scope 控制），SR/L-BFGS 不应用。

---

**作者**: Linus-style Code Review  
**状态**: 部分实现

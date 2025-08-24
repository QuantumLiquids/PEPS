---
title: Missing SGD Variants and Line Search
status: documented
date: 2025-08-18
last_updated: 2025-08-18
applies_to: [module/optimizer]
tags: [rfc, optimizer]
---

# Missing Optimization Algorithms from Legacy VMCPEPSExecutor

## Status: DOCUMENTED - To be implemented in future releases

## 概述

在从 `VMCPEPSExecutor` 迁移到 `VMCPEPSOptimizer` 的过程中，发现有多个特殊的优化算法未在新的优化器架构中实现：

### SGD变体与梯度预处理：
1. **RandomGradientElement** - 随机化梯度元素的幅值但保留相位
2. **Clip-by-value 梯度预处理（rename 自 legacy BoundGradientElement）** - 在更新前对梯度做按值裁剪；`clip_value` 独立于学习率，可与任何算法组合

### Line Search算法：
3. **GradientLineSearch** - 梯度方向的线搜索优化
4. **NaturalGradientLineSearch** - 自然梯度方向的线搜索优化

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

### 2. ClipValueSGD（legacy BoundGradientElement 的现代命名）

**功能**: 对梯度进行按元素“值裁剪”（clip-by-value），将每个元素限制在 [-clip_value, clip_value] 区间，然后按 SGD（可含 Momentum/Nesterov）更新。`clip_value` 是独立超参数，不与 `learning_rate` 混用。

**原始实现（legacy 参考）** (来自 `vmc_peps_impl.h:691-708`):
```cpp
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSExecutor<TenElemT, QNT,
                     MonteCarloSweepUpdater,
                     EnergySolver>::BoundGradElementUpdateTPS_(VMCPEPSExecutor::SITPST &grad,
                                                               double step_len) {
  if (rank_ == kMPIMasterRank) {
    for (size_t row = 0; row < ly_; row++)
      for (size_t col = 0; col < lx_; col++) {
        const size_t phy_dim = grad_({row, col}).size();
        for (size_t compt = 0; compt < phy_dim; compt++) {
          Tensor &grad_ten = grad({row, col})[compt];
          grad_ten.ElementWiseBoundTo(step_len);  // 核心操作：限制element幅值
          split_index_tps_({row, col})[compt] += (-step_len) * grad_ten;
        }
      }
  }
  BroadCast(split_index_tps_, comm_);
  Configuration config = tps_sample_.config;
  tps_sample_ = WaveFunctionComponentT(split_index_tps_, config, tps_sample_.trun_para);
  this->NormTPSForOrder1Amplitude_();
}
```

**使用场景**: 防止梯度爆炸，特别是在某些 tensor elements 出现异常大的梯度时。现代命名“ClipValueSGD”对齐 PyTorch/Keras/Optax 的“clip by value”术语。

**实现方式** :
现代实现顺序建议：裁剪梯度作为梯度预处理步骤，再按 SGD（含 Momentum/Nesterov），Adagrad，Adam等一阶方法更新


**复数语义与默认行为**:
- 复数按“幅值裁剪保相位”：若 |g|>clip_value，则 g←std::polar(clip_value, arg(g))；否则不变。与 `ElementWiseClipTo` 一致。
- 实数等价为“限制绝对值”且保留符号。
- `clip_value`、`clip_norm` 为可选项，默认未设置即“不裁剪”。
- 预处理默认仅对一阶方法（SGD/AdaGrad/Adam）生效；对 SR/L-BFGS 默认不应用。

### 3. GradientLineSearch

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

### 4. NaturalGradientLineSearch

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
2. 测试裁剪预处理（ElementWiseClipTo）的逐元素幅值限制与复数保相位
3. 测试全局范数裁剪（ClipByGlobalNorm）对步幅的统一缩放
4. 对比新实现与legacy实现的数值一致性
5. 验证MPI环境下的正确性

## 注意事项

1. **随机数生成器管理**: RandomizedSGD需要careful管理随机数状态
2. **MPI**: 1阶方法仅对Master进程生效。
3. **Tensor API依赖**: 依赖于 `ElementWiseRandomizeMagnitudePreservePhase`、`ElementWiseClipTo`、`ClipByGlobalNorm`（由SplitIndexTPS负责实现）。
4. **默认与作用范围**: `clip_value/clip_norm` 默认未设置即不启用；空出clip_scope作为place_holder，但预处理仅对一阶方法生效（内部 clip_scope 控制），SR/L-BFGS 不应用。

---

**作者**: Linus-style Code Review  
**创建日期**: 2025-01-29  
**状态**: 等待实现



## 独立实现计划：梯度预处理（裁剪）

本章节将“梯度裁剪”作为与其他算法变体解耦的独立功能，单独定义语义、API 与落地步骤。此计划可独立推进与上线，不依赖 RandomizedSGD/Line Search 等实现。

### 目标与范围
- 目标：提供可选的梯度预处理（裁剪）能力，以提高一阶优化器在数值不稳场景下的鲁棒性。
- 适用范围：默认仅对一阶方法（SGD/AdaGrad/Adam）生效；SR/L-BFGS 默认不应用。

### 数学与语义
- 元素幅值裁剪（complex-safe，默认）：若 |g| > clip_value，则 g ← std::polar(clip_value, arg(g))，否则不变；对实数等价于“限制绝对值且保留符号”。
- 全局范数裁剪（complex-safe）：令 r = sqrt(Σ_i |g_i|^2)。若 r > clip_norm，则对所有元素 g_i ← (clip_norm / r) · g_i，否则不变。

### API 变更（仅新增，可选，默认禁用）
- `OptimizerParams::BaseParams` 中新增两可选字段：
  - `std::optional<double> clip_value;`  // 元素级幅值裁剪阈值
  - `std::optional<double> clip_norm;`   // 全局 L2 范数裁剪阈值
- 不需要： `clip_mode/clip_eps`；`clip_scope` 作为内部开关，固定为“仅一阶方法”。

示意：
```cpp
// struct OptimizerParams::BaseParams {
//   ...
//   std::optional<double> clip_value;  // unset -> 不裁剪
//   std::optional<double> clip_norm;   // unset -> 不裁剪
// };
```

### Builder 接口（示意）
```cpp
class OptimizerParamsBuilder {
  // ...
  OptimizerParamsBuilder& SetClipValue(double clip_value);
  OptimizerParamsBuilder& SetClipNorm(double clip_norm);
};
```

### 实现顺序（优化器通用路径）
1) 若 `clip_value`：对梯度调用 `ElementWiseClipTo(clip_value)`（幅值裁剪，保相位）。
2) 若 `clip_norm`：对梯度调用 `ClipByGlobalNorm(clip_norm)` 统一缩放。
3) 进入一阶优化器更新（SGD/AdaGrad/Adam），动量/Nesterov/自适应在裁剪之后应用。
4) SR/L-BFGS：默认跳过裁剪（内部 `clip_scope` 控制）。

### MPI 与Complex number/费米张量
- MPI：裁剪与一阶更新保持一致，仅在 Master 进程执行。
- 复杂数/费米张量：幅值裁剪保相位语义与 `ElementWiseClipTo` 保持一致；全局范数裁剪对模长操作，天然适配。

### 回溯兼容与命名
- `ElementWiseBoundTo` 无需保留向后兼容。文档与新代码统一使用 `ElementWiseClipTo`。

### 测试清单
1. 复数张量幅值裁剪：|g|>c 时幅值被裁剪且相位不变；|g|≤c 时不变。
2. 实数张量幅值裁剪：|g|>c 时被裁剪为 ±c，符号正确；未超界时不变。
3. 全局范数裁剪：当 ||g||2>c 时统一缩放，缩放比准确；保持方向不变。
4. 与一阶法集成：裁剪发生在动量/自适应之前；数值稳定性改善。
5. MPI 一致性：仅 Master 裁剪，行为与现有分工一致。

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

在从 `VMCPEPSExecutor` 迁移到 `VMCPEPSOptimizerExecutor` 的过程中，发现有多个特殊的优化算法未在新的优化器架构中实现：

### SGD变体：
1. **RandomGradientElement** - 随机化梯度元素的幅值但保留相位
2. **BoundGradientElement** - 限制梯度元素的幅值但保留符号

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

### 2. BoundGradientElement

**功能**: 限制梯度的每个tensor element的幅值不超过指定值，但保留其符号/相位。

**原始实现** (来自 `vmc_peps_impl.h:691-708`):
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

**使用场景**: 防止梯度爆炸，特别是在某些tensor elements出现异常大的梯度时。

**调用方式** (原始实现):
```cpp
case BoundGradientElement:
  BoundGradElementUpdateTPS_(grad_, step_len);
  break;
```

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

#### SGD变体参数：

```cpp
/**
 * @struct RandomizedSGDParams  
 * @brief Parameters for randomized SGD (RandomGradientElement variant)
 */
struct RandomizedSGDParams {
  double momentum;
  bool nesterov;
  std::mt19937* random_engine;  // 随机数生成器引用
  
  RandomizedSGDParams(double momentum = 0.0, bool nesterov = false, 
                     std::mt19937* engine = nullptr)
    : momentum(momentum), nesterov(nesterov), random_engine(engine) {}
};

/**
 * @struct BoundedSGDParams
 * @brief Parameters for bounded SGD (BoundGradientElement variant)  
 */
struct BoundedSGDParams {
  double momentum;
  bool nesterov;
  double element_bound;  // 每个element的最大幅值
  
  BoundedSGDParams(double momentum = 0.0, bool nesterov = false, 
                  double bound = 1.0)
    : momentum(momentum), nesterov(nesterov), element_bound(bound) {}
};
```

### Phase 2: 扩展AlgorithmParams variant

```cpp
using AlgorithmParams = std::variant<
  SGDParams,
  AdamParams, 
  StochasticReconfigurationParams,
  LBFGSParams,
  AdaGradParams,
  RandomizedSGDParams,    // 新增
  BoundedSGDParams        // 新增
>;
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
  
  // 1. 随机化梯度elements
  auto randomized_grad = gradient;
  RandomizeGradientElements(randomized_grad, params.random_engine);
  
  // 2. 应用标准SGD更新
  ApplySGDUpdate(state, randomized_grad, static_cast<SGDParams>(params));
}

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::ApplyBoundedSGDUpdate(
    SplitIndexTPS<TenElemT, QNT>& state,
    const SplitIndexTPS<TenElemT, QNT>& gradient,
    const BoundedSGDParams& params) {
  
  // 1. 限制梯度elements幅值
  auto bounded_grad = gradient;
  BoundGradientElements(bounded_grad, params.element_bound);
  
  // 2. 应用标准SGD更新  
  ApplySGDUpdate(state, bounded_grad, static_cast<SGDParams>(params));
}
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
  
  OptimizerParamsBuilder& WithBoundedSGD(double momentum = 0.0, 
                                        bool nesterov = false,
                                        double element_bound = 1.0) {
    BoundedSGDParams sgd_params(momentum, nesterov, element_bound);
    algorithm_params_ = sgd_params;
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
2. 测试BoundedSGD的element限制功能
3. 对比新实现与legacy实现的数值一致性
4. 验证MPI环境下的正确性

## 注意事项

1. **随机数生成器管理**: RandomizedSGD需要careful管理随机数状态
2. **MPI一致性**: 确保所有进程上的随机化行为一致
3. **性能影响**: Element-wise操作可能影响性能，需要benchmark
4. **Tensor API依赖**: 依赖于 `ElementWiseRandomizeMagnitudePreservePhase` 和 `ElementWiseBoundTo` 方法

---

**作者**: Linus-style Code Review  
**创建日期**: 2025-01-29  
**状态**: 等待实现



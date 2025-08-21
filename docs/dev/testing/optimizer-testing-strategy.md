---
title: Optimizer Testing Strategy
last_updated: 2025-08-21
---

# Optimizer 测试策略：消除 Monte-Carlo 复杂性

## 问题分析

### 当前测试挑战

Monte-Carlo PEPS + Optimizer 测试包含太多变量：
- Monte-Carlo 采样误差
- PEPS 状态表示
- Optimizer 算法逻辑
- MPI 并行化

当测试失败时，很难确定问题的根源。这违反了测试的基本原则：**每个测试应该只验证一个关注点**。

### 现有解决方案

我们为测试增添 `ExactSumEnergyEvaluator`，它通过枚举所有配置来精确计算能量和梯度：

```cpp
template<typename ModelT, typename TenElemT, typename QNT>
std::tuple<TenElemT, SplitIndexTPS<TenElemT, QNT>, double> ExactSumEnergyEvaluator(
    const SplitIndexTPS<TenElemT, QNT> &split_index_tps,
    const std::vector<Configuration> &all_configs,
    const BMPSTruncatePara &trun_para,
    ModelT &model,
    size_t Ly, size_t Lx
)
```
+ MPI (即将支持）
这从算法和代码的测试都有帮助。

优势：
- 1. 完全确定性，无随机误差
- 2 可在小系统上更快计算

这一接口的测试：
 `tests/test_algorithm/test_exact_summation_evaluator.cpp`

## 纯 Optimizer 测试布局与命名

- 目录：`tests/test_optimizer/`
- AdaGrad 用例：`tests/test_optimizer/test_optimizer_adagrad_exact_sum.cpp`
- 命名规则：`test_optimizer_<algorithm>_exact_sum.cpp`
  - 未来每个优化算法新增一个独立的纯优化器测试文件
  - 统一采用 ExactSumEnergyEvaluator，确保梯度确定性，隔离 Monte-Carlo 噪声

备注：CMake 中对应的双类型目标命名为 `test_optimizer_adagrad_exact_sum_double` 与 `test_optimizer_adagrad_exact_sum_complex`，并提供 MPI 运行用例。


### 缺口
对SR的支持不是那么直接。虽然也有可能可以，但是还需要一些数学推导工作。



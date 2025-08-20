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
+ MPI 支持
这从算法和代码的测试都有帮助。

优势：
- ✅ 完全确定性，无随机误差
- ✅ 可在小系统上更快计算

这一接口的测试：
 test_exact_summation_evaluator.h

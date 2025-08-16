# Legacy Code Migration Plan: VMCPEPSExecutor → VMCPEPSOptimizerExecutor

## Date: 2025-01-29
## Status: PROPOSED - Ready for execution

## 【核心判断】
✅ **立即开始迁移，但先打保险tag！**

虽然代码库处于开发阶段无外部用户，但当前已有大量refactor工作在进行中。为了避免"两个大变更同时进行"的混乱，建议先创建一个保险tag，然后在当前分支上继续legacy清理工作。

## Git版本管理策略

### **推荐方案: Staged Migration with Safety Tag**

```bash
# 1. 在当前分支创建保险tag (标记refactor状态)  
git add .
git commit -m "WIP: Optimizer params refactor - before legacy cleanup"
git tag -a v2.0-pre-legacy-cleanup -m "Safe point before VMCPEPSExecutor removal"

# 2. 继续在当前分支进行legacy清理
# 3. 完成后合并到main并打正式tag
```

**原因**:
- ✅ **最小风险**: Tag作为回滚点，避免丢失当前refactor进度
- ✅ **开发效率**: 在单个分支完成所有现代化工作
- ✅ **逻辑一致性**: OptimizerParams重构 + Legacy清理 = 完整现代化
- ✅ **Testing便利**: 一次性验证所有变更的互操作性

## Legacy代码影响分析

### **1. 核心Legacy组件清单**

#### **VMCPEPSExecutor类族** (需完全删除)
```
include/qlpeps/algorithm/vmc_update/vmc_peps.h         - 类定义 (35-123行)
include/qlpeps/algorithm/vmc_update/vmc_peps_impl.h    - 实现代码
include/qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h - VMCOptimizePara
```

#### **受影响的测试文件** (需迁移到VMCPEPSOptimizerExecutor)
```
tests/test_algorithm/test_vmc_peps.cpp                          - 5处VMCOptimizePara使用
tests/integration_tests/test_square_tj_model.cpp               - 1处使用
tests/integration_tests/test_triangle_j1j2_heisenberg.cpp      - 1处使用  
tests/integration_tests/test_triangle_heisenberg.cpp           - 1处使用
tests/integration_tests/test_square_nn_spinless_free_fermion.cpp - 2处使用
tests/integration_tests/test_square_j1j2_xxz_legacy_vmcpeps.cpp - 1处使用(已标记legacy)
```

#### **示例代码** (需更新)
```
examples/migration_example.cpp - 3处VMCOptimizePara使用，需展示新API
```

#### **文档** (需更新)
```
docs/tutorial/MIGRATION_SUMMARY.md - 迁移指南需要更新
tests/integration_tests/integration_test_framework.h - 基础框架需要更新
```

### **2. 依赖关系严重性分析**

**🔴 CRITICAL Dependencies** (必须立即处理):
- `VMCOptimizePara` 在5个integration tests中使用
- `VMCPEPSExecutor` 在测试框架中被引用
- 构造函数参数不兼容 (VMCOptimizePara vs VMCPEPSOptimizerParams)

**🟡 MEDIUM Dependencies** (可平滑迁移):
- Example代码可以展示新旧两种方式 (短期)
- 文档可以并存 (无代码依赖)

**🟢 LOW Risk** (清理即可):
- Header file inclusion (include路径变更)
- Template实例化 (只是类名变更)

## 迁移实施计划

### **Phase 1: Git Safety & Preparation** (30分钟)

```bash
# 1.1 提交当前progress并打tag
git add .
git commit -m "feat: Optimizer params refactor - learning rate scheduling & modern C++ design

- Unified learning_rate parameter in BaseParams
- Learning rate scheduler interface (ConstantLR, ExponentialDecayLR, StepLR, PlateauLR)
- Factory methods and builder pattern for common configurations
- Type-safe algorithm parameter handling with std::variant
- Backward compatibility maintained through legacy fields"

git tag -a v2.0-pre-legacy-cleanup -m "Safe checkpoint: Modern optimizer params implemented, legacy cleanup next"

# 1.2 创建备份分支 (可选，额外安全)
git checkout -b backup-before-legacy-cleanup
git checkout optimizer-params-refactor
```

### **Phase 2: VMCOptimizePara Structure Migration** (2小时)

#### **2.1 删除VMCOptimizePara定义** (30分钟)
```cpp
// 🗑️ 删除文件内容:
// include/qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h
// - 整个VMCOptimizePara结构定义
// - 相关的constructor和helper函数

// ✅ 保留但标记过时:
// - 文件保留，添加deprecation notice和包含新头文件的指向
```

#### **2.2 Migration Helper Function** (30分钟)
```cpp
// 临时迁移助手 (在头文件中)
inline VMCPEPSOptimizerParams ConvertToModern(const VMCOptimizePara& legacy) {
  // 从legacy params提取参数并构造现代结构
  OptimizerParams opt_params = OptimizerFactory::CreateStochasticReconfiguration(
    legacy.mc_params.sweeps, legacy.cg_params, 0.01);
  MonteCarloParams mc_params = legacy.mc_params;
  PEPSParams peps_params = legacy.peps_params;
  return VMCPEPSOptimizerParams(opt_params, mc_params, peps_params);
}
```

#### **2.3 Update Integration Test Framework** (1小时)
```cpp
// integration_test_framework.h 需要修改基类
template<typename QNT>
class IntegrationTestFramework {
protected:
  // ✅ 新版本参数结构
  VMCPEPSOptimizerParams GetDefaultParams() {
    OptimizerParams opt_params = OptimizerFactory::CreateStochasticReconfiguration(
      40, ConjugateGradientParams(100, 1e-5, 20, 0.001), 0.01);
    // ... 其他参数
    return VMCPEPSOptimizerParams(opt_params, mc_params, peps_params);
  }
  
  // 🗑️ 删除legacy方法
  // VMCOptimizePara GetLegacyParams() { ... }
};
```

### **Phase 3: Test File Migration** (3小时)

#### **3.1 Mass Test Migration** (2小时)
逐个迁移测试文件，转换模式：

```cpp
// 🗑️ OLD Pattern:
VMCOptimizePara optimize_para(/*parameters*/);
VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model> executor(
  optimize_para, tps_init, comm, model);

// ✅ NEW Pattern:  
VMCPEPSOptimizerParams params = GetModernParams();  // 使用测试框架
VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model> executor(
  params, tps_init, comm, model);
```

**迁移顺序** (按复杂度):
1. `test_square_j1j2_xxz_legacy_vmcpeps.cpp` ✅ 已有modern版本，直接删除
2. `test_square_nn_spinless_free_fermion.cpp` 
3. `test_triangle_heisenberg.cpp`
4. `test_triangle_j1j2_heisenberg.cpp`
5. `test_square_tj_model.cpp`
6. `test_vmc_peps.cpp` (最复杂，有多个test case)

#### **3.2 Build & Test Validation** (1小时)
```bash
# 每迁移一个文件，立即测试
cd build && make -j$(nproc)
ctest -R test_square_nn_spinless_free_fermion --verbose
# 确保迁移后测试仍然通过
```

### **Phase 4: VMCPEPSExecutor Class Deletion** (1小时)

#### **4.1 删除类定义** (30分钟)
```cpp
// 🗑️ 完全删除:
// include/qlpeps/algorithm/vmc_update/vmc_peps.h (lines 35-123)
// include/qlpeps/algorithm/vmc_update/vmc_peps_impl.h (整个实现)

// ✅ 保留文件，但只留deprecation notice:
#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_H

#warning "VMCPEPSExecutor has been removed. Please use VMCPEPSOptimizerExecutor instead."
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"

#endif
```

#### **4.2 清理include dependencies** (30分钟)
- 移除所有指向已删除文件的include
- 更新algorithm_all.h和qlpeps.h中的包含

### **Phase 5: Examples & Documentation Update** (1小时)

#### **5.1 Update Migration Example** (30分钟)
```cpp
// examples/migration_example.cpp 
// ✅ 只展示新API，删除legacy示例
// ✅ 展示Factory methods, Builder pattern等现代用法
// ✅ 展示learning rate scheduling示例
```

#### **5.2 Update Documentation** (30分钟)
```markdown
// docs/tutorial/MIGRATION_SUMMARY.md
# 更新内容:
- ❌ 删除"迁移指南"章节 (已完成迁移)
- ✅ 更新为"Modern API指南"
- ✅ 展示OptimizerFactory和Builder pattern使用
- ✅ 学习率调度示例
```

### **Phase 6: Final Cleanup & Testing** (2小时)

#### **6.1 Complete Legacy Params Cleanup** (1小时)
按照`LEGACY_CLEANUP_ANALYSIS.md`中的方案，删除optimizer_params.h中的所有legacy兼容代码:

```cpp
// 🗑️ 删除:
// - WAVEFUNCTION_UPDATE_SCHEME enum (lines 32-79)
// - OptimizerParams中的legacy fields (lines 428-445) 
// - SetLegacyFields() method
// - Testing-only default constructor
// - Factory methods中的default parameters
// - 所有default constructors
```

#### **6.2 Comprehensive Testing** (1小时)
```bash
# 完整构建测试
cd build && make clean && make -j$(nproc)

# 运行所有相关测试
ctest -R "test_.*optimization|test_.*vmc_peps" --verbose

# 验证integration tests
ctest -R "test_square_|test_triangle_" --verbose

# 确保没有compilation errors
```

## 风险评估与缓解

### **High Risk 因素**

#### **🔴 Risk: 测试破坏**
- **可能性**: 高 (参数结构完全变化)
- **影响**: 中 (可以修复，但耗时)
- **缓解**: 逐个文件迁移 + 立即测试

#### **🔴 Risk: 性能回归**  
- **可能性**: 低 (新架构更clean)
- **影响**: 中 (physics结果必须一致)
- **缓解**: 在代表性物理问题上运行before/after对比

### **Medium Risk 因素**

#### **🟡 Risk: MPI协调问题**
- **可能性**: 中 (MPI代码复杂)
- **影响**: 高 (并行计算错误很难debug)  
- **缓解**: 重点测试MPI相关integration tests

#### **🟡 Risk: 遗漏的依赖**
- **可能性**: 中 (代码库较大)
- **影响**: 中 (编译错误，易发现易修复)
- **缓解**: 全量grep搜索 + 完整编译测试

### **Low Risk 因素**

#### **🟢 Risk: 文档不一致**  
- **可能性**: 高 (文档更新容易遗漏)
- **影响**: 低 (不影响功能)
- **缓解**: 最后阶段专门检查文档一致性

## 实施时间线

### **总时间预估: 8-10小时 (1.5个工作日)**

| Phase | 任务 | 时间 | 累计 |
|-------|------|------|------|
| 1 | Git Safety & Preparation | 0.5h | 0.5h |
| 2 | VMCOptimizePara Migration | 2h | 2.5h |
| 3 | Test File Migration | 3h | 5.5h |
| 4 | VMCPEPSExecutor Deletion | 1h | 6.5h |  
| 5 | Examples & Docs Update | 1h | 7.5h |
| 6 | Final Cleanup & Testing | 2h | 9.5h |
| 缓冲 | 调试与polish | 0.5h | 10h |

### **Milestone检查点**

**✅ Milestone 1** (2.5h): VMCOptimizePara完全移除，helper转换函数可用
**✅ Milestone 2** (5.5h): 所有测试文件迁移完成，integration tests通过  
**✅ Milestone 3** (7.5h): VMCPEPSExecutor类完全删除，example更新完成
**✅ Milestone 4** (9.5h): Legacy optimizer params清理完成，所有测试通过

## 【Linus式最终判断】

**这个迁移计划是正确的**。我们不是在"破坏"代码，而是在**消除技术债务**。

**核心原则**:
1. **安全第一**: Tag保护，逐步迁移，每步验证
2. **一次做对**: 现代化整个优化器栈，不留历史包袱  
3. **零妥协**: 删除所有legacy代码，强制使用modern API
4. **实用主义**: 专注于代码质量，不为"向后兼容"浪费时间

现在就开始！这个项目没有外部用户，这是进行彻底现代化的最佳时机。等到有外部依赖的时候，这种clean-up就会变得困难百倍。

**"Perfect is the enemy of good enough"的相反情况**: 在开发阶段，**good enough就是perfect的敌人**。现在把代码做到最好，future you will thank present you.

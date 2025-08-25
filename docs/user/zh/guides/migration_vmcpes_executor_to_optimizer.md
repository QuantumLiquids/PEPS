# 迁移指南：从 VMCPEPSExecutor 到 VMCPEPSOptimizer

## 概览

本指南帮助你从旧的 `VMCPEPSExecutor`（v0.0.1) 迁移到新的 `VMCPEPSOptimizer`。新执行器在保持外部接口一致的同时，带来更好的模块化、职责分离与可维护性。

## TL;DR — 有哪些变化


### 1. 参数结构
```cpp
// OLD: 单一结构，显式传 ly/lx 与占据数组
VMCOptimizePara optimize_para(
    truncate_para, num_samples, num_warmup_sweeps, sweeps_between_samples,
    {1, 1, 2}, // occupancy array
    4, 4,         // ly, lx
    step_lengths, update_scheme, cg_params, wavefunction_path);

// NEW: 拆分结构
Configuration initial_config(4, 4, OccupancyNum({1, 1, 2}));

// Monte-Carlo parameters
MonteCarloParams mc_params(
    num_samples, num_warmup_sweeps, sweeps_between_samples,
    initial_config, /*is_warmed_up=*/false, /*config_dump_path=*/"./configs/"
);
// Tensor-network parameters
PEPSParams peps_params(truncate_para);

// Optimizer parameters
// 方式A：推荐工厂方法（更少样板、避免误配）
auto opt_params = OptimizerFactory::CreateStochasticReconfiguration(
    /*max_iterations=*/40,
    ConjugateGradientParams(/*max_iter=*/100, /*tol=*/1e-5, /*restart=*/20, /*diag_shift=*/1e-3),
    /*learning_rate=*/0.3
);

// 方式B：需要更细控制时使用 Builder（示例）
/*
auto opt_params = OptimizerParamsBuilder()
    .SetMaxIterations(40)
    .SetLearningRate(0.3)
    .WithStochasticReconfiguration(ConjugateGradientParams(100, 1e-5, 20, 1e-3),
                                   /*normalize=*/false, /*adaptive_shift=*/0.0)
    // .SetClipValue(...) / .SetClipNorm(...) 仅对一阶优化器生效
    .Build();
*/

VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params, /*dump tps path bas=*/"./tps_dump");
```

## 关键差异
- 由 `VMCOptimizePara` → `OptimizerParams`/`MonteCarloParams`/`PEPSParams` 组合
- Initial configuration 由 tps_path + fallback_init_config -> init_config + is warmedup
- 新增config_dump_path（之前采用tps_path作为config_dump_path)
- tps_path -> tps_path_base_name


### 2. 构造模式
```cpp
// OLD: 多种重载
VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    optimize_para, tps_init, comm, model);

// NEW: 两种清晰模式
// 模式A：用户提供显式 TPS
VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    params, tps_init, comm, model);

// 模式B：从路径加载 TPS（ly/lx 由 initial_config 推断）
VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    params, "/path/to/tps", comm, model);
```

### 3. 新增算法
Monmentum, AdaGrad
ElementWiseBoundedSign -> ClipTo 梯度预处理。



## 迁移步骤

### Step 1：更新参数结构
```cpp
#include "qlpeps/qlpeps.h"

Configuration initial_config(4, 4, OccupancyNum({1, 1, 2}));

MonteCarloParams mc_params(
    1000, 100, 10,
    initial_config,
    false,  // is_warmed_up
    "configs"      // config_dump_path
);

PEPSParams peps_params(BMPSTruncatePara(/* ... */));

OptimizerParams opt_params;/* 参考优化器指南 */

VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params, "./tps");
```

### Step 2：更新构造调用
```cpp
// 模式A：显式 TPS
VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    params, tps_init, comm, model);

// 模式B：文件路径（推荐用于加载已保存态）
VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    params, "/path/to/tps", comm, model);
```


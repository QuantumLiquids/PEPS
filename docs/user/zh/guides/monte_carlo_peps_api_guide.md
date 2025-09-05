# Monte Carlo PEPS API 使用指南

## 概览
API 提供两种清晰的构造模式：
1. 显式控制：用户显式提供 TPS 与 Configuration
2. 便捷加载：用户给出 TPS 路径，格点尺寸由 Configuration 推断

两种模式共享同一套统一参数结构，初始化逻辑一致，无隐式差异。

## 模式一：显式控制（适合自定义/内存态）

测量：
```cpp
#include "qlpeps/qlpeps.h"
#include "qlpeps/api/conversions.h" // 显式转换 PEPS/TPS/SITPS

SplitIndexTPS<TenElemT, QNT> user_tps(ly, lx);
user_tps.Random();
// 如果你持有 PEPS，可显式转换：
// auto tps   = qlpeps::ToTPS<TenElemT, QNT>(peps);
// user_tps   = qlpeps::ToSplitIndexTPS<TenElemT, QNT>(tps);

Configuration user_config(ly, lx, OccupancyNum({num_up, num_down, num_empty}));

MonteCarloParams mc_params(1000, 100, 5, user_config, false);
PEPSParams peps_params(BMPSTruncatePara(/*...*/));
MCMeasurementParams params(mc_params, peps_params, "./output");

MCPEPSMeasurer<TenElemT, QNT, UpdaterType, SolverType> executor(
    user_tps, params, comm, solver);
```

优化：
```cpp
SplitIndexTPS<TenElemT, QNT> initial_tps(ly, lx);
initial_tps.Load("/path/to/initial/tps");

Configuration initial_config(ly, lx);
bool load_success = initial_config.Load("/path/to/config.dat", 0);
if (!load_success) {
  initial_config = Configuration(ly, lx, OccupancyNum({num_up, num_down, num_holes}));
}

OptimizerParams opt_params(/*...*/);
MonteCarloParams mc_params(500, 100, 3, initial_config, load_success);
PEPSParams peps_params(BMPSTruncatePara(/*...*/));
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);

VMCPEPSOptimizer<TenElemT, QNT, UpdaterType, SolverType> optimizer(
    params, initial_tps, comm, solver);
```

## 模式二：便捷加载（适合常规科研工作流）

测量：
```cpp
Configuration analysis_config(4, 4);
analysis_config.Random(std::vector<size_t>(2, 8));

MonteCarloParams mc_params(2000, 200, 10, analysis_config, false);
PEPSParams peps_params(BMPSTruncatePara(/*...*/));
MCMeasurementParams params(mc_params, peps_params, "./measurement_output");

MonteCarloMeasurementExecutor<TenElemT, QNT, UpdaterType, SolverType> executor(
    "/path/to/saved/tps", params, comm, solver);
```

优化：
```cpp
size_t total = 6 * 6;
size_t holes = total - 18 - 18;
Configuration opt_config(6, 6, OccupancyNum({18, 18, holes}));

OptimizerParams opt_params(/*...*/);
MonteCarloParams mc_params(1000, 200, 5, opt_config, false);
PEPSParams peps_params(BMPSTruncatePara(/*...*/));
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);

VMCPEPSOptimizer<TenElemT, QNT, UpdaterType, SolverType> optimizer(
    params, "/path/to/initial/tps", comm, solver);
```

## 设计要点
- 尺寸推断：`ly = initial_config.rows()`，`lx = initial_config.cols()`
- 统一参数：两种模式完全共享同一参数结构与初始化逻辑
- 显式所有权：用户清楚自己提供了什么数据与 dump 策略

## 迁移指引（旧 API → 新 API）
```cpp
// 旧：混淆的多实参构造
// 新：两种固定构造；要么显式数据，要么路径+config推断
```

## 最佳实践
1. 常规工作流优先用“便捷加载”
2. 特殊初始化/研究需求用“显式控制”
3. 始终显式设置 dump 路径，重要数据不要依赖默认
4. 配置尺寸要与 TPS 完全匹配，发现不符应快速失败



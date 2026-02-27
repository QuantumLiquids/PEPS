# 顶层 API（Simple Update / VMC / 测量）

本仓库主要服务两个管线：

- **VMC 优化**：`VmcOptimize(...)`
- **蒙特卡洛测量**：`MonteCarloMeasure(...)`

这两个一键接口位于：

- `include/qlpeps/api/vmc_api.h`

Simple Update 作为一组 executor，位于：

- `include/qlpeps/algorithm/simple_update/`

## 这些 API 做什么（以及不做什么）

- `VmcOptimize(...)` 与 `MonteCarloMeasure(...)` 面向内存中的 `SplitIndexTPS`。
- 它们**不会**替你从磁盘加载 TPS。加载是状态类型自己的方法（`SplitIndexTPS::Load`）。
- 输出路径由参数显式控制（`tps_dump_path`、`measurement_data_dump_path`、`config_dump_path`）。

## 后端（OBC/BMPS 与 PBC/TRG）

两个一键接口都会从：

- `sitps.GetBoundaryCondition()`（OBC 或 PBC）

推断后端，并在发现与你的参数不一致时**快速失败**：

- OBC 需要 `PEPSParams` 中持有 BMPS 截断参数
- PBC 需要 `PEPSParams` 中持有 TRG 截断参数

这是刻意的设计：避免“看似跑通，但其实跑了错误后端”的隐蔽问题。

## VMC 优化：`VmcOptimize(...)`

最小骨架：

```cpp
#include <mpi.h>
#include "qlten/qlten.h"
#include "qlpeps/api/vmc_api.h"                  // VmcOptimize
#include "qlpeps/optimizer/optimizer_params.h"   // OptimizerFactory, ConjugateGradientParams
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h" // MCUpdateSquareNNFullSpaceUpdate

using TenElemT = qlten::QLTEN_Double;
using QNT = qlten::special_qn::TrivialRepQN;

// sitps: SplitIndexTPS<TenElemT, QNT>（已构造）
// solver: 模型能量求解器对象（例如 TransverseFieldIsingSquareOBC(h)）

MonteCarloParams mc_params(
    /*total_samples=*/500,
    /*num_warmup_sweeps=*/200,
    /*sweeps_between_samples=*/2,
    /*initial_config=*/Configuration(/*Ly=*/4, /*Lx=*/4),
    /*is_warmed_up=*/false,
    /*config_dump_path=*/"");

auto trunc = BMPSTruncateParams<double>::SVD(/*D_min=*/2, /*D_max=*/8, /*trunc_err=*/1e-14);
PEPSParams peps_params(trunc);

auto opt_params = OptimizerFactory::CreateStochasticReconfiguration(
    /*max_iterations=*/40,
    ConjugateGradientParams(/*max_iter=*/100, /*relative_tolerance=*/3e-3, /*restart=*/20, /*diag_shift=*/1e-3),
    /*learning_rate=*/0.1);

VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params, /*tps_dump_path=*/"./optimized_tps");

auto result = qlpeps::VmcOptimize(params, sitps, MPI_COMM_WORLD, solver,
                                 MCUpdateSquareNNFullSpaceUpdate{});
```

## 蒙特卡洛测量：`MonteCarloMeasure(...)`

最小骨架：

```cpp
#include <mpi.h>
#include "qlten/qlten.h"
#include "qlpeps/api/vmc_api.h" // MonteCarloMeasure
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h" // MCUpdateSquareNNFullSpaceUpdate

MCMeasurementParams meas_params(
    /*mc_params=*/mc_params,
    /*peps_params=*/peps_params,
    /*measurement_data_dump_path=*/"./mc_measure_output");

auto meas = qlpeps::MonteCarloMeasure(sitps, meas_params, MPI_COMM_WORLD, solver,
                                     MCUpdateSquareNNFullSpaceUpdate{});
```

说明：

- 可观测量的 key/shape 由模型的 `DescribeObservables()` 元数据定义。
- registry key 参考：`../reference/model_observables_registry.md`。

## 状态 I/O：从磁盘加载（显式）

加载由状态类型提供：

```cpp
SplitIndexTPS<TenElemT, QNT> sitps;
if (!sitps.Load("./optimized_tps")) {
  throw std::runtime_error("Failed to load SplitIndexTPS.");
}
```

## Simple Update（executor API）

Simple Update 目前没有类似 `VmcOptimize(...)` 的一键 wrapper，直接使用 executor。

参考实现：

- `examples/transverse_field_ising_simple_update.cpp`

方格晶格最近邻 Simple Update 的主要入口头文件：

- `qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h`

## 相关阅读

- 端到端工作流：`../tutorials/end_to_end_workflow.md`
- 选择更新器：`choose_mc_updater.md`
- 状态转换：`state_conversions.md`

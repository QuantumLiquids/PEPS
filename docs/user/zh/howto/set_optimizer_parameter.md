# 设置优化器参数（How-to）

本页展示如何为 VMC-PEPS 优化构造 `OptimizerParams`。
仅关注优化器参数（不包含 MC/PEPS 参数）。
算法数学见 `../explanation/optimizer_algorithms.md`。
建议先从简单配置开始，确有需要再加入高级参数。

> 同步状态：英文文档（`docs/user/en/`）为权威版本；如与代码行为冲突，请以英文版本和头文件为准。

## 头文件

```cpp
#include "qlpeps/optimizer/optimizer_params.h"
```

## 1. 创建优化器参数（从简单到高级）

### SGD（常数学习率）

```cpp
auto opt_params = qlpeps::OptimizerParamsBuilder()
    .SetMaxIterations(1000)
    .SetLearningRate(1e-2)
    .WithSGD(/*momentum=*/0.9, /*nesterov=*/false)
    .Build();
```

如需 weight decay，请显式构造 `SGDParams`：

```cpp
qlpeps::OptimizerParams::BaseParams base_params(
    /*max_iterations=*/1000,
    /*energy_tolerance=*/1e-8,
    /*gradient_tolerance=*/1e-6,
    /*plateau_patience=*/50,
    /*learning_rate=*/1e-2);

qlpeps::SGDParams sgd_params(
    /*momentum=*/0.9,
    /*nesterov=*/false,
    /*weight_decay=*/1e-4);

qlpeps::OptimizerParams opt_params(base_params, sgd_params);
```

### Adam

```cpp
auto opt_params = qlpeps::OptimizerFactory::CreateAdam(
    /*max_iterations=*/1000,
    /*learning_rate=*/1e-3,
    /*beta1=*/0.9,
    /*beta2=*/0.999);
```

需要完全控制（epsilon、weight decay）时可用 builder：

```cpp
auto opt_params = qlpeps::OptimizerParamsBuilder()
    .SetMaxIterations(1000)
    .SetLearningRate(1e-3)
    .WithAdam(/*beta1=*/0.9, /*beta2=*/0.999, /*epsilon=*/1e-8, /*weight_decay=*/0.0)
    .Build();
```

### AdaGrad

```cpp
auto opt_params = qlpeps::OptimizerFactory::CreateAdaGrad(
    /*max_iterations=*/1000,
    /*learning_rate=*/1e-2,
    /*epsilon=*/1e-8,
    /*initial_accumulator=*/0.0);
```

### L-BFGS

```cpp
auto opt_params = qlpeps::OptimizerFactory::CreateLBFGS(
    /*max_iterations=*/200,
    /*learning_rate=*/1.0,
    /*history_size=*/10);
```

`CreateLBFGS(...)` 保持向后兼容，默认使用固定步长模式：
- `step_mode = LBFGSStepMode::kFixed`
- 推荐用于 MC 场景（对噪声更稳健）。

deterministic / exact-sum 场景建议显式配置 strong-Wolfe：

```cpp
qlpeps::LBFGSParams lbfgs(
    /*history_size=*/10,
    /*tol_grad=*/1e-8,
    /*tol_change=*/1e-12,
    /*max_eval=*/32,
    /*step_mode=*/qlpeps::LBFGSStepMode::kStrongWolfe,
    /*wolfe_c1=*/1e-4,
    /*wolfe_c2=*/0.9,
    /*min_step=*/1e-8,
    /*max_step=*/1.0,
    /*min_curvature=*/1e-12,
    /*use_damping=*/true,
    /*max_direction_norm=*/1e3,
    /*allow_fallback_to_fixed_step=*/false,
    /*fallback_fixed_step_scale=*/0.2);

auto opt_params = qlpeps::OptimizerFactory::CreateLBFGSAdvanced(
    /*max_iterations=*/300,
    /*energy_tolerance=*/1e-15,
    /*gradient_tolerance=*/1e-30,
    /*plateau_patience=*/100,
    /*learning_rate=*/0.05,
    lbfgs);
```

strong-Wolfe 失败策略：
- 默认：直接报错（fail-fast）。
- 仅在显式开启 `allow_fallback_to_fixed_step=true` 时才允许降级为固定步长。
- `tol_change` 控制线搜索 bracket/步长区间的终止阈值；取值越小通常会增加线搜索评估次数。
- `tol_grad` 是曲率条件中的绝对下限（`|phi'(alpha)| <= max(c2*|phi'(0)|, tol_grad)`）。

### 随机重构（SR）

```cpp
qlpeps::ConjugateGradientParams cg_params(
    /*max_iter=*/100,
    /*tolerance=*/1e-5,
    /*restart_step=*/20,
    /*diag_shift=*/1e-3);

auto opt_params = qlpeps::OptimizerFactory::CreateStochasticReconfiguration(
    /*max_iterations=*/1000,
    cg_params,
    /*learning_rate=*/0.1);
```

## 2. 高级停止条件 + 调度器

如需显式停止条件，可使用 `*Advanced` 工厂或直接构造 `OptimizerParams::BaseParams`。

```cpp
auto scheduler = std::make_unique<qlpeps::PlateauLR>(
    /*initial_lr=*/0.1,
    /*factor=*/0.5,
    /*patience=*/30,
    /*threshold=*/1e-5);

auto opt_params = qlpeps::OptimizerFactory::CreateStochasticReconfigurationAdvanced(
    /*max_iterations=*/2000,
    /*energy_tolerance=*/1e-8,
    /*gradient_tolerance=*/1e-6,
    /*plateau_patience=*/50,
    cg_params,
    /*learning_rate=*/0.1,
    std::move(scheduler));
```

## 3. 学习率调度器（可用列表）

构造函数签名：

```cpp
ConstantLR(double lr)
ExponentialDecayLR(double initial_lr, double decay_rate, size_t decay_steps)
StepLR(double initial_lr, size_t step_size, double gamma = 0.1)
MultiStepLR(double initial_lr, std::vector<size_t> milestones, double gamma)
CosineAnnealingLR(double eta_max, size_t T_max, double eta_min = 0.0)
WarmupLR(double base_lr, size_t warmup_steps, double start_lr = 0.0)
PlateauLR(double initial_lr, double factor = 0.5, size_t patience = 10, double threshold = 1e-4)
```

使用方式：将 `std::unique_ptr<LearningRateScheduler>` 传给
`OptimizerParams::BaseParams` 或 `*Advanced` 工厂方法。

## 4. 高级配置

### 梯度裁剪

```cpp
auto opt_params = qlpeps::OptimizerParamsBuilder()
    .SetMaxIterations(2000)
    .SetLearningRate(1e-3)
    .WithAdam()
    .SetClipValue(/*per_element=*/0.1)
    .SetClipNorm(/*global_l2=*/10.0)
    .Build();
```

说明：
- 裁剪仅适用于一阶优化器（SGD/AdaGrad/Adam）。
- 设置裁剪前先调用 `SetMaxIterations` 或 `SetLearningRate`。

### 自动步长选择器（面向 MC，v1）

`IterativeOptimize` 可选启用自动步长选择器，用少量候选步长应对 MC 噪声。

```cpp
auto opt_params = qlpeps::OptimizerParamsBuilder()
    .SetMaxIterations(1000)
    .SetLearningRate(0.1)
    .WithSGD()
    .SetPeriodicStepSelector(
        /*enabled=*/true,
        /*every_n_steps=*/10,
        /*phase_switch_ratio=*/0.3,
        /*enable_in_deterministic=*/false)
    .Build();
```

v1 行为：
- 支持算法：仅 SGD 与 SR。
- 候选集合：`{eta, eta/2}`。
- 触发频率：仅在迭代号可被 `every_n_steps` 整除时触发；若最后一步不整除，则该步不会触发选择器。
- 写回策略：选中步长会写回，且保持单调不增。
- 相位策略：前期（`iter < ratio * max_iterations`）偏激进按均值选；后期要求相对误差条有显著改进才降步长。
- 触发步 MC 成本：每次触发总计会评估 3 次能量（主路径 1 次 + 候选试算 2 次）。
- 若同时启用 `SetInitialStepSelector(enabled=true, max_line_search_steps=k, ...)`，则第 0 步会评估 `1 + k` 次能量。

重要约束：
- 默认仅 MC 模式可用（`enable_in_deterministic=false`）；deterministic 评估器需显式开启。
- v1 中 `lr_scheduler` 与自动步长选择器不能同时启用（fail-fast）。
- L-BFGS 行为不变，不使用该功能。
- 功能实现在 `IterativeOptimize`。

### Checkpointing

```cpp
auto opt_params = qlpeps::OptimizerParamsBuilder()
    .SetMaxIterations(2000)
    .SetLearningRate(1e-3)
    .WithAdam()
    .SetCheckpoint(/*every_n_steps=*/100, /*base_path=*/"./checkpoints")
    .Build();
```

### 尖峰恢复

```cpp
qlpeps::SpikeRecoveryParams spike;
spike.enable_auto_recover = true;
spike.redo_mc_max_retries = 2;
spike.enable_rollback = false;

auto opt_params = qlpeps::OptimizerParamsBuilder()
    .SetMaxIterations(2000)
    .SetLearningRate(1e-3)
    .WithAdam()
    .SetSpikeRecovery(spike)
    .Build();
```

参见：
- How-to：`../howto/spike_recovery.md`
- 数学细节：`../explanation/spike_recovery_math.md`

## 5. 接入 VMCPEPS

```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"

qlpeps::VMCPEPSOptimizerParams vmc_params(
    opt_params,
    mc_params,
    peps_params,
    /*tps_dump_path=*/"./optimized_tps");
```

实现路径说明：
- 本仓库中的 L-BFGS 走 `Optimizer::IterativeOptimize` 主链路。

## 相关阅读

- 优化算法：`../explanation/optimizer_algorithms.md`
- 顶层 API：`top_level_apis.md`
- 尖峰恢复：`spike_recovery.md`

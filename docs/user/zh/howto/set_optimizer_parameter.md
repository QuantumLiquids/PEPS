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

## 相关阅读

- 优化算法：`../explanation/optimizer_algorithms.md`
- 顶层 API：`top_level_apis.md`
- 尖峰恢复：`spike_recovery.md`

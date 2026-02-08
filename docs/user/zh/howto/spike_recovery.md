# 尖峰恢复（How-to）

本页展示如何为 VMC-PEPS 配置自动尖峰检测与恢复。

尖峰恢复由 `OptimizerParams::spike_recovery_params` 控制，并通过
`VmcOptimize(...)` 的 `VMCPEPSOptimizerParams` 生效。

> 同步状态：英文文档（`docs/user/en/`）为权威版本；如与代码行为冲突，请以英文版本和头文件为准。

## 功能概览

- **S1**：检测异常大的能量误差条。
- **S2**：检测异常大的梯度范数。
- **S3**：检测自然梯度异常（仅 SR）。
- **S4**：能量上冲时的可选回滚（需显式开启）。
- **回滚兜底**：若开启回滚，在 S1-S3 的重采样重试次数耗尽后，也可能触发回滚（最后手段）。

默认启用 S1-S3，并触发 **MC 重采样**；回滚默认关闭。

## 最小配置

```cpp
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"

qlpeps::SpikeRecoveryParams spike;
spike.enable_auto_recover = true;      // S1-S3
spike.redo_mc_max_retries = 2;         // 每步最多重采样次数
spike.enable_rollback = false;         // 回滚关闭（默认）

auto params = qlpeps::OptimizerParamsBuilder()
    .SetMaxIterations(2000)
    .SetLearningRate(1e-3)
    .WithAdam(0.9, 0.999, 1e-8, 0.0)
    .SetSpikeRecovery(spike)
    .Build();

qlpeps::VMCPEPSOptimizerParams vmc_params(
    params, mc_params, peps_params, /*tps_dump_path=*/"./optimized_tps");

auto result = qlpeps::VmcOptimize(vmc_params, sitps, MPI_COMM_WORLD, solver,
                                 MCUpdateSquareNNFullSpaceUpdate{});
```

## 完整参数列表

所有字段位于 `qlpeps::SpikeRecoveryParams`。

- `enable_auto_recover`：S1-S3 重采样总开关。
- `redo_mc_max_retries`：每步最大重采样次数，超过后“带警告接受”
  （若开启回滚且存在上一步状态，则可能回滚）。
- `factor_err`：S1 能量误差条阈值因子。
- `factor_grad`：S2 梯度范数阈值因子。
- `factor_ngrad`：S3 自然梯度范数阈值因子（仅 SR）。
- `sr_min_iters_suspicious`：S3 触发阈值（CG 迭代次数异常偏小）。
- `enable_rollback`：回滚总开关（默认关闭）。
  开启后会启用 S4 回滚，并允许在 S1-S3 重采样耗尽时以回滚作为兜底动作。
- `ema_window`：所有信号的 EMA 窗口大小。
- `sigma_k`：S4 能量尖峰阈值倍数。
- `log_trigger_csv_path`：可选的 CSV 触发日志路径。

## 推荐用法

- 在噪声较大的 Monte Carlo 场景下保持 S1-S3 开启。
- 只有在理解回滚限制后再开启回滚（见下方说明）。
- 对 SR，同步调整 `factor_ngrad` 与 `sr_min_iters_suspicious`。
- 诊断不稳定时可设置 `log_trigger_csv_path`。

## 说明与注意事项

- 回滚只恢复 **波函数状态**，不会恢复优化器内部累积量（如 Adam 动量、AdaGrad 历史）。
- S3 仅对 **随机重构（SR）** 有意义。
- 重采样会重做同一步的 **能量/梯度评估**，不会推进优化器状态。

## 相关阅读

- 数学与阈值：`../explanation/spike_recovery_math.md`
- 优化器设置：`set_optimizer_parameter.md`
- 顶层 API：`top_level_apis.md`

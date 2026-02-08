# Spike recovery (How-to)

This page shows how to configure automatic spike detection and recovery for VMC-PEPS.

Spike recovery is controlled by `OptimizerParams::spike_recovery_params` and is used by
`VmcOptimize(...)` via `VMCPEPSOptimizerParams`.

## What it does

- **S1**: Detects unusually large energy error bars.
- **S2**: Detects unusually large gradient norms.
- **S3**: Detects anomalous natural-gradient behavior (SR only).
- **S4**: Optional rollback when energy spikes upward (opt-in).
- **Fallback rollback**: if rollback is enabled, S1-S3 may also roll back after the resampling
  retry budget is exhausted (last-resort).

By default, S1-S3 are enabled and trigger **MC resampling**. Rollback is disabled.

## Minimal configuration

```cpp
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"

qlpeps::SpikeRecoveryParams spike;
spike.enable_auto_recover = true;      // S1-S3
spike.redo_mc_max_retries = 2;         // retries per step
spike.enable_rollback = false;         // rollback off (default)

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

## Full parameter list

All fields live in `qlpeps::SpikeRecoveryParams`.

- `enable_auto_recover`: master switch for S1-S3 resampling.
- `redo_mc_max_retries`: maximum resample attempts per step before accept-with-warning
  (or rollback if enabled and a previous state is available).
- `factor_err`: S1 threshold factor for the energy error bar.
- `factor_grad`: S2 threshold factor for gradient norm.
- `factor_ngrad`: S3 threshold factor for natural-gradient norm (SR only).
- `sr_min_iters_suspicious`: S3 trigger if CG iterations are unusually small.
- `enable_rollback`: master switch for rollback (default off).
  Enables S4 rollback and allows S1-S3 to roll back as a last-resort after resampling retries.
- `ema_window`: EMA window size for all tracked signals.
- `sigma_k`: S4 threshold multiplier for energy spike.
- `log_trigger_csv_path`: optional CSV path for spike logs.

## Recommended usage

- Keep S1-S3 on for noisy Monte Carlo regimes.
- Turn on rollback only if you understand rollback limitations
  (see notes below).
- For SR, tune `factor_ngrad` and `sr_min_iters_suspicious` together.
- Use `log_trigger_csv_path` when diagnosing instability.

## Notes and caveats

- Rollback restores the **wavefunction state only**. It does not restore
  optimizer-internal accumulators (e.g., Adam moments, AdaGrad history).
- S3 is only relevant for **Stochastic Reconfiguration**.
- A resample retries the **energy/gradient evaluation** for the same step;
  it does not advance the optimizer state.

## Related

- Math and thresholds: `../explanation/spike_recovery_math.md`
- Optimizer setup: `set_optimizer_parameter.md`
- Top-level APIs: `top_level_apis.md`

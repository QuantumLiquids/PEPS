# Set optimizer parameters (How-to)

This page shows how to construct `OptimizerParams` for VMC-PEPS optimization.
It focuses on optimizer parameters only (not MC/PEPS params).
For algorithm math, see `../explanation/optimizer_algorithms.md`.
Start simple and add advanced knobs only when you need them.

## Headers

```cpp
#include "qlpeps/optimizer/optimizer_params.h"
```

## 1. Create optimizer params (easy to advanced)

### SGD (constant learning rate)

```cpp
auto opt_params = qlpeps::OptimizerParamsBuilder()
    .SetMaxIterations(1000)
    .SetLearningRate(1e-2)
    .WithSGD(/*momentum=*/0.9, /*nesterov=*/false)
    .Build();
```

If you need weight decay, construct `SGDParams` explicitly:

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

For full control (epsilon, weight decay), use the builder:

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

`CreateLBFGS(...)` is backward-compatible and defaults to fixed-step mode:
- `step_mode = LBFGSStepMode::kFixed`
- Recommended for MC runs (noise-robust baseline).

For deterministic/exact-sum runs, use strong-Wolfe explicitly:

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

Strong-Wolfe failure policy:
- Default: throw (fail fast).
- Fallback to fixed-step is opt-in only (`allow_fallback_to_fixed_step=true`).
- `tol_change` controls the line-search bracket/step-interval termination tolerance; smaller values usually increase line-search evaluations.
- `tol_grad` is an absolute floor in the curvature check (`|phi'(alpha)| <= max(c2*|phi'(0)|, tol_grad)`).

### Stochastic Reconfiguration (SR)

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

## 2. Advanced stopping + schedulers

If you need explicit stopping criteria, use the `*Advanced` factories or build
`OptimizerParams::BaseParams` directly.

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

## 3. Learning-rate schedulers (available)

Constructor signatures:

```cpp
ConstantLR(double lr)
ExponentialDecayLR(double initial_lr, double decay_rate, size_t decay_steps)
StepLR(double initial_lr, size_t step_size, double gamma = 0.1)
MultiStepLR(double initial_lr, std::vector<size_t> milestones, double gamma)
CosineAnnealingLR(double eta_max, size_t T_max, double eta_min = 0.0)
WarmupLR(double base_lr, size_t warmup_steps, double start_lr = 0.0)
PlateauLR(double initial_lr, double factor = 0.5, size_t patience = 10, double threshold = 1e-4)
```

Use them by passing a `std::unique_ptr<LearningRateScheduler>` into
`OptimizerParams::BaseParams` or the `*Advanced` factory methods.

## 4. Advanced configuration

### Gradient clipping

```cpp
auto opt_params = qlpeps::OptimizerParamsBuilder()
    .SetMaxIterations(2000)
    .SetLearningRate(1e-3)
    .WithAdam()
    .SetClipValue(/*per_element=*/0.1)
    .SetClipNorm(/*global_l2=*/10.0)
    .Build();
```

Notes:
- Clipping applies only to first-order optimizers (SGD/AdaGrad/Adam).
- Call `SetMaxIterations` or `SetLearningRate` before setting clip values.

### Checkpointing

```cpp
auto opt_params = qlpeps::OptimizerParamsBuilder()
    .SetMaxIterations(2000)
    .SetLearningRate(1e-3)
    .WithAdam()
    .SetCheckpoint(/*every_n_steps=*/100, /*base_path=*/"./checkpoints")
    .Build();
```

### Spike recovery

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

See:
- How-to: `../howto/spike_recovery.md`
- Math: `../explanation/spike_recovery_math.md`

## 5. Wire into VMCPEPS

```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"

qlpeps::VMCPEPSOptimizerParams vmc_params(
    opt_params,
    mc_params,
    peps_params,
    /*tps_dump_path=*/"./optimized_tps");
```

Implementation path note:
- L-BFGS in this repo is implemented in `Optimizer::IterativeOptimize`.
- `LineSearchOptimize` is not part of the L-BFGS production path.

## Related

- Optimizer algorithms: `../explanation/optimizer_algorithms.md`
- Top-level APIs: `top_level_apis.md`
- Spike recovery: `spike_recovery.md`

# Optimizer algorithms (VMC-PEPS)

## Overview

This page explains the optimizer algorithms and math used by VMC-PEPS in this repository.
It assumes you have read `vmcpeps_optimizer_architecture.md`.
If you only want the knobs and code, jump to `../howto/set_optimizer_parameter.md`.

## Parameter setup: the simple structure

```cpp
// Variant-based algorithm dispatch
using AlgorithmParams = std::variant<
  SGDParams,
  AdamParams,
  StochasticReconfigurationParams,
  LBFGSParams,
  AdaGradParams
>;

struct OptimizerParams {
  BaseParams base_params;           // Common parameters
  AlgorithmParams algorithm_params; // Algorithm-specific parameters
};
```

This keeps the common knobs (`BaseParams`) separate from the algorithm-specific ones.
The result is one config object that is hard to misuse.

### Parameter hierarchy

```
OptimizerParams
├── BaseParams (shared by all algorithms)
│   ├── max_iterations        // Maximum iterations
│   ├── energy_tolerance      // Energy convergence criterion
│   ├── gradient_tolerance    // Gradient convergence criterion
│   ├── plateau_patience      // Plateau patience parameter
│   ├── learning_rate         // Unified learning rate interface
│   ├── lr_scheduler          // Optional learning rate scheduler
│   └── auto_step_selector    // Optional MC-oriented auto step-size selector (v1)
├── AlgorithmParams (algorithm-specific)
│   ├── SGDParams            // Stochastic Gradient Descent
│   ├── AdamParams           // Adam optimizer
│   ├── AdaGradParams        // Adaptive Gradient
│   ├── LBFGSParams          // Limited-memory BFGS
│   └── StochasticReconfigurationParams  // Stochastic Reconfiguration
├── CheckpointParams         // Optional periodic TPS checkpointing
└── SpikeRecoveryParams      // Optional spike detection/recovery
```

## Notation

- $\theta$: vector of variational parameters.
- $\psi(S;\theta)$: wavefunction amplitude for configuration $S$.
- $E_{\mathrm{loc}}(S)$: local energy for configuration $S$.
- $O_i = \partial \ln \psi / \partial \theta_i$: log-derivative operator.
- $g$: gradient vector (algorithm-dependent definition).
- $\eta$: learning rate.

## Algorithms (easy to advanced)

### 1. SGD (first-order)

**Update rule**:

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

**Momentum**:

$$
v_t = \mu v_{t-1} + \eta g_t
$$
$$
\theta_{t+1} = \theta_t - v_t
$$

**Nesterov momentum**:

$$
v_t = \mu v_{t-1} + \eta \nabla f(\theta_t - \mu v_{t-1})
$$
$$
\theta_{t+1} = \theta_t - v_t
$$

**Decoupled weight decay** (as implemented in `SGDParams`):

$$
\theta \leftarrow (1 - \eta \lambda)\,\theta
$$

Then apply the gradient update. This is separate from learning-rate scheduling.

**Properties**:
- Low per-step memory and compute cost.
- Sensitive to learning rate and noise; momentum helps smooth noise.

### 2. Adam (adaptive moment estimation)

**Update rule**:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\hat{m}_t = m_t / (1-\beta_1^t)
$$
$$
\hat{v}_t = v_t / (1-\beta_2^t)
$$
$$
\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
$$

**Properties**:
- Maintains first and second moment estimates of gradients.
- Higher memory cost (stores $m_t$ and $v_t$).

### 3. AdaGrad (adaptive gradient)

**Update rule**:

$$
G_t = G_{t-1} + g_t \odot g_t
$$
$$
\theta_{t+1} = \theta_t - \eta g_t / (\sqrt{G_t} + \epsilon)
$$

**Properties**:
- Per-parameter step sizes shrink over time.
- Can stall if learning rate becomes too small.

### 4. L-BFGS (limited-memory BFGS)

**Update rule**:

$$
d_k = -H_k g_k
$$
$$
x_{k+1} = x_k + \alpha_k d_k
$$

Here \(H_k\) is built from limited history pairs \((s_i, y_i)\) via two-loop recursion.
In this codebase, curvature uses the parameter-space inner product
\(\mathrm{Re}\langle s_i, y_i \rangle\), with damping/skip guards for low-curvature pairs.

**Step mode in implementation**:
- `LBFGSStepMode::kStrongWolfe`: recommended for deterministic / exact-sum runs.
- `LBFGSStepMode::kFixed`: recommended for MC runs with noisy gradients.

Strong-Wolfe conditions (used when `kStrongWolfe`):

$$
\phi(\alpha) \le \phi(0) + c_1 \alpha \phi'(0)
$$
$$
|\phi'(\alpha)| \le \max\!\left(c_2 |\phi'(0)|,\ \texttt{tol\_grad}\right),\quad 0 < c_1 < c_2 < 1
$$

Failure policy:
- Default: throw (fail fast).
- Fixed-step fallback is opt-in only (`allow_fallback_to_fixed_step=true`).
- `tol_change` is the bracket/step-interval termination tolerance in strong-Wolfe line search; smaller values typically require more evaluations.

Implementation path:
- L-BFGS is implemented in `Optimizer::IterativeOptimize`.
- `LineSearchOptimize` is not part of the L-BFGS production path.

**Properties**:
- Uses limited history to approximate inverse Hessian.
- `kStrongWolfe` provides robust deterministic step control.
- `kFixed` avoids unstable line-search behavior under MC noise.

### 5. Stochastic Reconfiguration (natural gradient)

**Natural gradient update**:

$$
\theta_{t+1} = \theta_t - \eta S^{-1} g
$$

**S-matrix and gradient (VMC)**:

$$
S_{ij} = \langle O_i^* O_j \rangle - \langle O_i^* \rangle \langle O_j \rangle
$$
$$
g_i = \langle E_{\mathrm{loc}} O_i^* \rangle - \langle E_{\mathrm{loc}} \rangle \langle O_i^* \rangle
$$

**Monte Carlo estimates**:

$$
\langle A \rangle \approx \frac{1}{N} \sum_{k=1}^N A(S_k)
$$

**Properties**:
- Solves a linear system each step (typically with CG + diagonal shift).
- Sensitive to S-matrix conditioning; regularization is usually required.

## Learning-rate schedulers

All schedulers implement $\eta_t = \eta(t)$ and are optional.

**Constant**:

$$
\eta_t = \eta_0
$$

**Exponential decay**:

$$
\eta_t = \eta_0 \cdot \gamma^{t / s}
$$

**Step decay**:

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}
$$

**Multi-step decay**:

$$
\eta_t = \eta_0 \cdot \gamma^{k},\quad k = \#\{m \in \text{milestones} : m \le t\}
$$

**Cosine annealing**:

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\,[1 + \cos(\pi t / T_{\max})]
$$

**Warmup (linear)**:

$$
\eta_t = \eta_{\mathrm{start}} + (\eta_{\mathrm{base}} - \eta_{\mathrm{start}})\,\frac{t+1}{T_{\mathrm{warmup}}}
$$

**Plateau-based**:

Track the best energy and reduce the learning rate by a factor when the energy
improves by less than a threshold for a fixed patience window.

## Auto step selector (v1, IterativeOptimize)

This repository also provides an optional auto step-size selector in
`Optimizer::IterativeOptimize` for MC-noisy runs.

Current v1 scope:
- Algorithms: SGD and SR only.
- Candidate set: `{eta, eta/2}`.
- Trigger: only when the iteration index is divisible by `every_n_steps` (no trigger on non-divisible final iterations).
- Writeback: selected `eta` is persisted and forced to be non-increasing.
- Two-phase policy:
  - Early phase: aggressive mean-energy comparison.
  - Late phase: halve only if improvement is significant versus error bars.

Compatibility and safety:
- Default mode assumes MC error bars are available.
- Deterministic use must be explicitly enabled (`enable_in_deterministic=true`).
- `lr_scheduler` cannot be combined with auto step selector in v1 (fail fast).
- L-BFGS remains unchanged and does not use this selector.

## Monte Carlo noise (shared context)

VMC gradient estimates contain statistical noise:

$$
g_{\mathrm{estimated}} = g_{\mathrm{true}} + \mathrm{noise}
$$
$$
\mathrm{Var}[g_{\mathrm{estimated}}] \propto 1 / N_{\mathrm{samples}}
$$

Noise affects all optimizers; momentum or natural-gradient preconditioning can
help but do not remove sampling variance.

## Related

- Parameter setup: `../howto/set_optimizer_parameter.md`
- Optimizer architecture: `vmcpeps_optimizer_architecture.md`
- Spike recovery math: `spike_recovery_math.md`

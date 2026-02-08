# Spike recovery math (EMA-based detection)

This page summarizes the math used by spike detection and recovery in the VMC optimizer.
It assumes you already know VMC, SR, and CG.

## Motivation (why spike recovery exists)

Monte Carlo estimates are stochastic. Rare configurations or under-sampling can
produce sudden spikes in the energy error bar, gradient norm, or natural-gradient
norm. Updating parameters on these outliers can destabilize optimization.

Spike recovery protects the optimizer by comparing each signal against its recent
history (via EMA statistics) and then **resampling** or **rolling back** when a
step looks anomalous.

## EMA tracking (Exponential Moving Average)

EMA stands for **Exponential Moving Average**. We track an EMA mean and variance
for each scalar signal.

Let $x_t$ be a scalar signal at step $t$. Let $w$ be `ema_window`.

EMA and variance (as implemented):

- $\alpha = 2 / (w + 1)$.
- Mean: $m_t = m_{t-1} + \alpha (x_t - m_{t-1})$.
- Variance: $v_t = (1 - \alpha)\,(v_{t-1} + \alpha (x_t - m_{t-1})^2)$.
- Std: $\sigma_t = \sqrt{v_t}$.

EMA trackers are updated **only on accepted steps**.

## Signal-specific EMA notation

We maintain separate EMA statistics for each signal. Notation:

- $m_t^{(e)}$: EMA mean of the energy error bar $e_t$.
- $m_t^{(g)}$: EMA mean of the gradient norm $g_t$.
- $m_t^{(n)}$: EMA mean of the natural-gradient norm $n_t$.
- $m_t^{(E)}$: EMA mean of the energy $E_t$.
- $\sigma_t^{(E)}$: EMA standard deviation of the energy $E_t$.

## Signals and thresholds

### S1: energy error bar spike

Let $e_t$ be the energy error bar (MC error).

Trigger if:

- $e_t > \kappa_e \cdot m_t^{(e)}$, where $\kappa_e$ is the tunable threshold `factor_err`.

### S2: gradient norm spike

Let $g_t = \|\nabla E\|$.

Trigger if:

- $g_t > \kappa_g \cdot m_t^{(g)}$, where $\kappa_g$ is the tunable threshold `factor_grad`.

### S3: natural-gradient anomaly (SR only)

Let $n_t = \|\nabla_{\text{SR}} E\|$ and $k_t$ be CG iterations.

Trigger if either:

- $n_t > \kappa_n \cdot m_t^{(n)}$, where $\kappa_n$ is the tunable threshold `factor_ngrad`, or
- $k_t \le k_\text{min}$, where $k_\text{min}$ is the tunable threshold `sr_min_iters_suspicious`.

### S4: upward energy spike (rollback, opt-in)

Let $E_t$ be the energy (real part).

Trigger if **both**:

- $E_t$ increases relative to the EMA mean, and
- $E_t > m_t^{(E)} + \sigma_k \cdot \sigma_t^{(E)}$, where `sigma_k` is the threshold multiplier.

S4 is disabled by default and must be enabled explicitly.

## Actions

Actions depend on the signal and the retry budget:

- **Resample** (S1-S3): re-run Monte Carlo evaluation (same step, new samples).
- **Rollback** (opt-in): restore the previous accepted state.
  - Always used for S4 when enabled.
  - Can also be used as a last-resort for S1-S3 when resampling retries are exhausted
    (if rollback is enabled and a previous state is available).
- **Accept with warning**: if resampling retries are exhausted and rollback is disabled/unavailable.

Retry budget is controlled by `redo_mc_max_retries`.

## Practical implications

- S1-S3 address stochastic noise in MC sampling and SR solves.
- S4 is useful for rare energy blow-ups, but only restores the state.
  Optimizer accumulators (Adam/AdaGrad/SGD momentum) are **not** restored.
- Thresholds are multiplicative and scale with EMA statistics, so they are
  relatively robust across models once a stable EMA forms.

## Related

- How to configure: `../howto/spike_recovery.md`
- Optimizer setup: `../howto/set_optimizer_parameter.md`

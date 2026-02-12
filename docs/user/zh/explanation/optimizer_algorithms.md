# 优化算法（VMC-PEPS）

## 概述

本页解释本仓库 VMC-PEPS 优化器使用的算法与数学。
假设你已阅读 `vmcpeps_optimizer_architecture.md`。
如果只想看参数和代码，请跳到 `../howto/set_optimizer_parameter.md`。

## 参数设置：简单结构

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
  BaseParams base_params;
  AlgorithmParams algorithm_params;
};
```

这会把通用参数（`BaseParams`）与算法特定参数分离。
最终得到一个不易误用的配置对象。

### 参数层级

```
OptimizerParams
├── BaseParams（所有算法共享）
│   ├── max_iterations        // 最大迭代次数
│   ├── energy_tolerance      // 能量收敛阈值
│   ├── gradient_tolerance    // 梯度收敛阈值
│   ├── plateau_patience      // 平台期耐心参数
│   ├── learning_rate         // 统一学习率接口
│   ├── lr_scheduler          // 可选学习率调度器
│   └── auto_step_selector    // 可选 MC 导向自动步长选择器（v1）
├── AlgorithmParams（算法特定）
│   ├── SGDParams            // 随机梯度下降
│   ├── AdamParams           // Adam
│   ├── AdaGradParams        // AdaGrad
│   ├── LBFGSParams          // L-BFGS
│   └── StochasticReconfigurationParams  // 随机重构（SR）
├── CheckpointParams         // 可选 TPS 周期性 checkpoint
└── SpikeRecoveryParams      // 可选尖峰检测/恢复
```

## 记号

- $\theta$：变分参数向量。
- $\psi(S;\theta)$：组态 $S$ 的波函数幅度。
- $E_{\mathrm{loc}}(S)$：组态 $S$ 的局域能量。
- $O_i = \partial \ln \psi / \partial \theta_i$：对数导数算符。
- $g$：梯度向量（定义随算法不同而变化）。
- $\eta$：学习率。

## 算法（从简单到复杂）

### 1. SGD（一阶方法）

**更新式**：

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

**动量**：

$$
v_t = \mu v_{t-1} + \eta g_t
$$
$$
\theta_{t+1} = \theta_t - v_t
$$

**Nesterov 动量**：

$$
v_t = \mu v_{t-1} + \eta \nabla f(\theta_t - \mu v_{t-1})
$$
$$
\theta_{t+1} = \theta_t - v_t
$$

**解耦式 weight decay**（`SGDParams` 的实现）：

$$
\theta \leftarrow (1 - \eta \lambda)\,\theta
$$

随后再应用梯度更新。这与学习率调度相互独立。

**性质**：
- 单步计算与内存开销低。
- 对学习率与噪声敏感；动量有助于平滑噪声。

### 2. Adam（自适应一阶矩/二阶矩）

**更新式**：

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

**性质**：
- 维护一阶与二阶矩估计。
- 内存开销较高（需要存 $m_t$ 与 $v_t$）。

### 3. AdaGrad（自适应梯度）

**更新式**：

$$
G_t = G_{t-1} + g_t \odot g_t
$$
$$
\theta_{t+1} = \theta_t - \eta g_t / (\sqrt{G_t} + \epsilon)
$$

**性质**：
- 各参数步长会随时间递减。
- 学习率过小后可能停滞。

### 4. L-BFGS（有限内存 BFGS）

**更新式**：

$$
d_k = -H_k g_k
$$
$$
x_{k+1} = x_k + \alpha_k d_k
$$

其中 \(H_k\) 由有限历史对 \((s_i, y_i)\) 通过 two-loop recursion 构造。
本代码中曲率标量使用参数空间内积 \(\mathrm{Re}\langle s_i, y_i \rangle\)，
并对低曲率对做 damping/skip 保护。

**实现中的步长模式**：
- `LBFGSStepMode::kStrongWolfe`：推荐 deterministic / exact-sum 场景。
- `LBFGSStepMode::kFixed`：推荐 MC 场景（含采样噪声）。

`kStrongWolfe` 使用强 Wolfe 条件：

$$
\phi(\alpha) \le \phi(0) + c_1 \alpha \phi'(0)
$$
$$
|\phi'(\alpha)| \le \max\!\left(c_2 |\phi'(0)|,\ \texttt{tol\_grad}\right),\quad 0 < c_1 < c_2 < 1
$$

失败策略：
- 默认：直接报错（fail-fast）。
- 仅在显式开启 `allow_fallback_to_fixed_step=true` 时才允许降级固定步长。
- `tol_change` 是 strong-Wolfe 线搜索中 bracket/步长区间的终止阈值；取值越小通常需要更多评估次数。

实现路径：
- L-BFGS 走 `Optimizer::IterativeOptimize` 主链路。
- `LineSearchOptimize` 不属于当前 L-BFGS 生产路径。

**性质**：
- 用有限历史近似逆 Hessian。
- `kStrongWolfe` 适合 deterministic 步长控制。
- `kFixed` 能避免 MC 噪声下线搜索不稳定。

### 5. 随机重构（SR，自然梯度）

**自然梯度更新**：

$$
\theta_{t+1} = \theta_t - \eta S^{-1} g
$$

**S 矩阵与梯度（VMC）**：

$$
S_{ij} = \langle O_i^* O_j \rangle - \langle O_i^* \rangle \langle O_j \rangle
$$
$$
g_i = \langle E_{\mathrm{loc}} O_i^* \rangle - \langle E_{\mathrm{loc}} \rangle \langle O_i^* \rangle
$$

**蒙特卡洛估计**：

$$
\langle A \rangle \approx \frac{1}{N} \sum_{k=1}^N A(S_k)
$$

**性质**：
- 每步需要解线性方程（通常用 CG + 对角正则）。
- 对 S 矩阵条件数敏感，通常需要正则化。

## 学习率调度器

所有调度器都实现 $\eta_t = \eta(t)$，且均为可选。

**常数**：

$$
\eta_t = \eta_0
$$

**指数衰减**：

$$
\eta_t = \eta_0 \cdot \gamma^{t / s}
$$

**阶梯衰减**：

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}
$$

**多阶梯衰减**：

$$
\eta_t = \eta_0 \cdot \gamma^{k},\quad k = \#\{m \in \text{milestones} : m \le t\}
$$

**余弦退火**：

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\,[1 + \cos(\pi t / T_{\max})]
$$

**线性 warmup**：

$$
\eta_t = \eta_{\mathrm{start}} + (\eta_{\mathrm{base}} - \eta_{\mathrm{start}})\,\frac{t+1}{T_{\mathrm{warmup}}}
$$

**平台触发**：

跟踪最优能量，在固定的耐心窗口内能量提升小于阈值时，将学习率按比例下降。

## 自动步长选择器（v1，IterativeOptimize）

本仓库在 `Optimizer::IterativeOptimize` 中提供了可选自动步长选择器，
用于处理 MC 噪声场景下的步长选择。

当前 v1 范围：
- 算法：仅 SGD 和 SR。
- 候选集合：`{eta, eta/2}`。
- 触发频率：仅在迭代号可被 `every_n_steps` 整除时触发（最后一步若不整除则不触发）。
- 写回策略：选中 `eta` 会写回，并保持单调不增。
- 两阶段策略：
  - 前期：偏激进，按均值能量比较。
  - 后期：只有当相对误差条有显著改进时才降为 `eta/2`。

兼容性与安全性：
- 默认假设有 MC 误差条。
- deterministic 场景需显式开启（`enable_in_deterministic=true`）。
- v1 中 `lr_scheduler` 与自动步长选择器不可同时启用（fail-fast）。
- L-BFGS 行为不变，不使用该选择器。

## 蒙特卡洛噪声（共享背景）

VMC 的梯度估计包含统计噪声：

$$
g_{\mathrm{estimated}} = g_{\mathrm{true}} + \mathrm{noise}
$$
$$
\mathrm{Var}[g_{\mathrm{estimated}}] \propto 1 / N_{\mathrm{samples}}
$$

噪声会影响所有优化器；动量或自然梯度预条件可以缓解，但无法消除采样方差。

## 相关阅读

- 参数设置：`../howto/set_optimizer_parameter.md`
- 优化器架构：`vmcpeps_optimizer_architecture.md`
- 尖峰恢复数学：`spike_recovery_math.md`

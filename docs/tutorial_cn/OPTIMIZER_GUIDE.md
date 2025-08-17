# 用于变分蒙卡优化PEPS的 优化器（Optimizer） 使用指南

## 概述

PEPS optimizer支持多种优化算法，用于VMC-PEPS (Variational Monte Carlo PEPS) 优化。采用variant-based类型安全设计，API与常用库相似。

### PEPS优化的挑战

1. **高维非凸优化**：参数空间维度高，local minima多
2. **梯度噪声**：Monte Carlo sampling引入statistical noise  
3. **收敛缓慢**：能量landscape平坦

## 核心架构

### 类型安全的参数系统

```cpp
// 现代C++设计：variant-based algorithm dispatch
using AlgorithmParams = std::variant<
  SGDParams,
  AdamParams, 
  StochasticReconfigurationParams,
  LBFGSParams,
  AdaGradParams
>;

struct OptimizerParams {
  BaseParams base_params;           // 通用参数
  AlgorithmParams algorithm_params; // 算法特定参数
};
```

特点：
- 统一接口处理所有算法
- 编译时类型检查
- 容易扩展新算法

### 三层参数架构

```
OptimizerParams
├── BaseParams (所有算法共享)
│   ├── max_iterations        // 最大迭代次数
│   ├── energy_tolerance      // 能量收敛判据
│   ├── gradient_tolerance    // 梯度收敛判据
│   ├── plateau_patience      // 平台期耐心参数
│   ├── learning_rate         // 统一学习率接口
│   └── lr_scheduler          // 可选的学习率调度器
└── AlgorithmParams (算法特定)
    ├── SGDParams            // 随机梯度下降
    ├── AdamParams           // Adam优化器
    ├── StochasticReconfigurationParams  // 随机重构
    ├── LBFGSParams          // 限制内存BFGS
    └── AdaGradParams        // 自适应梯度
```

## 优化算法详解

### 1. Stochastic Reconfiguration (自然梯度)

**数学原理**：

Stochastic Reconfiguration是PEPS优化的金标准，基于自然梯度的思想：

```
普通梯度下降：θ_{n+1} = θ_n - η ∇E(θ)
自然梯度下降：θ_{n+1} = θ_n - η S^{-1} ∇E(θ)
```

其中S是Fisher信息矩阵（在VMC中称为S-matrix）：

```
S_{ij} = ⟨O_i^* O_j⟩ - ⟨O_i^*⟩⟨O_j⟩
∇E_i = ⟨E_{loc} O_i^*⟩ - ⟨E_{loc}⟩⟨O_i^*⟩
```

这里 O_i = ∂ln|ψ⟩/∂θ_i 是对数导数算子。

原理：
- 考虑参数空间几何结构
- S-matrix表征参数间相关性
- 避免普通梯度的病态行为

**参数设置**：

```cpp
// CG求解器参数（关键！）
ConjugateGradientParams cg_params{
  100,      // max_iter: S-matrix通常ill-conditioned，需要更多迭代
  1e-5,     // tolerance: 平衡精度与计算成本
  20,       // restart_step: 避免数值积累误差
  0.001     // diag_shift: 正则化防止singular matrix
};

// SR特定参数
StochasticReconfigurationParams sr_params{
  cg_params,
  false,    // normalize_update: 是否归一化更新
  0.0       // adaptive_diagonal_shift: 动态正则化
};
```

**收敛特性**：
- **优点**：理论上optimal的收敛率，对复杂能量landscape robust
- **缺点**：每步需要求解线性方程组，计算成本高O(N³)
- **适用场景**：高精度优化，parameter数量适中的系统

### 2. Adam (Adaptive Moment Estimation)

**数学原理**：

Adam结合了momentum和adaptive learning rate：

```
m_t = β₁ m_{t-1} + (1-β₁) g_t        // First moment (momentum)
v_t = β₂ v_{t-1} + (1-β₂) g_t²       // Second moment (adaptive rate)
m̂_t = m_t / (1-β₁^t)                // Bias correction
v̂_t = v_t / (1-β₂^t)                // Bias correction
θ_{t+1} = θ_t - η m̂_t / (√v̂_t + ε)  // Parameter update
```

参数说明：
- **β₁**：momentum参数，记忆历史梯度
- **β₂**：自适应学习率参数
- **Bias correction**：修正初始偏差

**参数设置指南**：

```cpp
AdamParams adam_params{
  0.9,      // beta1: 一阶矩衰减率，控制momentum强度
  0.999,    // beta2: 二阶矩衰减率，控制自适应程度
  1e-8,     // epsilon: 数值稳定性，防止除零
  0.0       // weight_decay: L2正则化强度
};
```

**调参策略**：
- **学习率**：PEPS通常需要较小的学习率 (1e-4 ~ 1e-2)
- **β₁**: 默认0.9通常有效，noise较大时可减小到0.8
- **β₂**: 默认0.999，对于non-stationary problems可减小到0.99
- **ε**: 如果出现numerical instability，可增大到1e-6

### 3. SGD with Momentum

**数学原理**：

经典的带动量随机梯度下降：

```
标准SGD：θ_{t+1} = θ_t - η g_t
带动量：v_t = μ v_{t-1} + η g_t
        θ_{t+1} = θ_t - v_t

Nesterov变种：v_t = μ v_{t-1} + η ∇f(θ_t - μ v_{t-1})
             θ_{t+1} = θ_t - v_t
```

**参数理解**：
- **μ (momentum)**：通常0.9，控制历史梯度的影响
- **Nesterov**: 提供"预见性"的梯度计算，通常收敛更快

### 4. AdaGrad (Adaptive Gradient)

**数学原理**：

根据历史梯度自动调整学习率：

```
G_t = G_{t-1} + g_t ⊙ g_t          // 累积梯度平方
θ_{t+1} = θ_t - η g_t / (√G_t + ε)  // 参数更新
```

**特点**：
- **自适应**：频繁更新的参数学习率自动减小
- **问题**：学习率单调递减，可能过早停止
- **适用**：稀疏梯度场景，或初期探索阶段

### 5. L-BFGS (Limited-memory BFGS)

**数学原理**：

基于Hessian近似的准牛顿法：

```
H_k ≈ ∇²f(x_k)                    // Hessian近似
d_k = -H_k^{-1} ∇f(x_k)           // 搜索方向
x_{k+1} = x_k + α_k d_k            // 线搜索更新
```

L-BFGS用有限历史构造Hessian近似，避免存储完整矩阵。

**适用场景**：
- 确定性优化问题
- 梯度noise较小的情况
- 需要superlinear收敛率

## 学习率调度策略

### 1. Exponential Decay

```cpp
auto scheduler = std::make_unique<ExponentialDecayLR>(
  0.01,     // initial_lr
  0.95,     // decay_rate: 每decay_steps衰减5%
  100       // decay_steps: 每100步衰减一次
);
```

**使用场景**：稳定收敛，适合长期训练

### 2. Step Decay

```cpp
auto scheduler = std::make_unique<StepLR>(
  0.01,     // initial_lr
  200,      // step_size: 每200步降低一次
  0.5       // gamma: 每次减半
);
```

**使用场景**：阶段性调整，适合有明确training phase的场景

### 3. Plateau-based (能量平台检测)

```cpp
auto scheduler = std::make_unique<PlateauLR>(
  0.01,     // initial_lr
  0.5,      // factor: 检测到平台时减半
  20,       // patience: 20步没有改进才算平台
  1e-5      // threshold: 能量改进阈值
);
```

**使用场景**：PEPS优化的最佳选择，根据物理收敛自动调整

## 实际使用指南

### 快速开始：Factory Methods

```cpp
#include "qlpeps/optimizer/optimizer_params.h"

// 1. Stochastic Reconfiguration (推荐用于高精度优化)
ConjugateGradientParams cg_params{100, 1e-5, 20, 0.001};
auto sr_params = OptimizerFactory::CreateStochasticReconfiguration(
  1000,        // max_iterations
  cg_params,   // CG求解器参数
  0.1          // learning_rate
);

// 2. Adam (推荐用于快速原型开发)
auto adam_params = OptimizerFactory::CreateAdam(
  1000,        // max_iterations
  1e-3         // learning_rate
);

// 3. SGD with decay (简单且robust)
auto sgd_params = OptimizerFactory::CreateSGDWithDecay(
  1000,        // max_iterations
  0.01,        // initial_learning_rate
  0.95,        // decay_rate
  100          // decay_steps
);
```

### 高级配置：Builder Pattern

```cpp
auto params = OptimizerParamsBuilder()
  .SetMaxIterations(2000)
  .SetEnergyTolerance(1e-8)
  .SetGradientTolerance(1e-6)
  .SetPlateauPatience(50)
  .SetLearningRate(0.01, std::make_unique<PlateauLR>(0.01, 0.5, 20))
  .WithStochasticReconfiguration(cg_params, true, 0.001)  // 启用normalize_update
  .Build();
```

### VMCPEPS集成

```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"

// 完整的VMCPEPS优化参数
VMCPEPSOptimizerParams vmc_params{
  optimizer_params,    // 上面创建的optimizer参数
  mc_params,          // Monte Carlo采样参数
  peps_params,        // PEPS truncation参数  
  "optimized_state"   // 输出文件前缀
};

// 创建并执行optimizer
VMCPEPSOptimizerExecutor<ComplexDouble, QNZ2> executor(
  vmc_params, initial_tps, comm, energy_solver
);
executor.Execute();
```

## 调参最佳实践

### 1. 算法选择决策树

```
是否需要最高精度？
├── 是 → Stochastic Reconfiguration
│   ├── 参数数量 < 10⁴ → 标准SR
│   └── 参数数量 ≥ 10⁴ → SR + 更大diag_shift
└── 否 → 看训练阶段
    ├── 快速原型/exploration → Adam
    ├── 生产环境/稳定性 → SGD + decay
    └── 特殊需求 → AdaGrad/L-BFGS
```

### 2. 学习率设置策略

```cpp
// 保守策略（推荐用于生产）
double conservative_lr = 0.001;  // 慢但稳定

// 激进策略（用于快速实验）  
double aggressive_lr = 0.1;     // 快但可能不稳定

// 自适应策略（最佳实践）
auto adaptive_lr = std::make_unique<PlateauLR>(0.01, 0.5, 20);
```

### 3. 收敛判据设置

```cpp
// 高精度物理计算
BaseParams high_precision{
  5000,     // max_iterations: 给足时间
  1e-10,    // energy_tolerance: 化学精度
  1e-8,     // gradient_tolerance: 梯度严格收敛
  100,      // plateau_patience: 避免过早停止
  0.01
};

// 快速原型开发
BaseParams prototype{
  500,      // max_iterations: 快速反馈
  1e-6,     // energy_tolerance: 足够精度
  1e-4,     // gradient_tolerance: 较宽松
  20,       // plateau_patience: 快速判断收敛
  0.01
};
```

### 4. 常见问题诊断

**收敛太慢**：
```cpp
// 1. 增大学习率
// 2. 使用Adam或Stochastic Reconfiguration
// 3. 检查gradient calculation是否正确
// 4. 考虑preconditioning
```

**数值不稳定**：
```cpp
// 1. 减小学习率
// 2. 增大epsilon (Adam)
// 3. 增大diag_shift (SR)
// 4. 使用gradient clipping
```

**内存不足**：
```cpp
// 1. 避免L-BFGS for large systems
// 2. SR的CG求解器减小max_iter
// 3. 使用更aggressive的PEPS truncation
```

## 性能基准

基于我们的测试数据：

| 算法 | 收敛速度 | 内存使用 | 数值稳定性 | 推荐场景 |
|------|----------|----------|------------|----------|
| Stochastic Reconfiguration | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高精度计算 |
| Adam | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 快速原型 |
| SGD + Momentum | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 生产环境 |
| AdaGrad | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 稀疏参数 |
| L-BFGS | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 小规模系统 |

## 总结

优秀的PEPS优化需要：
1. **正确的算法选择**：根据问题规模和精度要求
2. **合理的参数设置**：基于物理直觉和经验
3. **适当的学习率调度**：结合能量收敛特性
4. **充分的patience**：PEPS优化本质上是困难的

建议在实际问题中测试这些参数配置。

## 深入数学原理

### Stochastic Reconfiguration详解

#### Fisher Information Matrix的构造

在VMC中，我们有wavefunction |ψ(θ)⟩，其中θ是variational parameters。Fisher information matrix定义为：

```
S_{ij} = ⟨∂_i ln ψ|∂_j ln ψ⟩ - ⟨∂_i ln ψ⟩⟨∂_j ln ψ⟩
```

这里∂_i = ∂/∂θ_i，且：

```
O_i = ∂_i ln ψ = ∂_i ln ψ / ψ = (1/ψ) ∂_i ψ
```

#### 为什么自然梯度更优？

考虑parameter space的Riemannian structure。在curved manifold上，最steep descent direction不是Euclidean gradient，而是natural gradient：

```
g^{natural} = S^{-1} g^{Euclidean}
```

**物理直觉**：S-matrix编码了参数间的correlation。如果两个参数highly correlated，改变其中一个会自动影响另一个，所以更新时需要考虑这种coupling。

#### PEPS中的gradient calculation

对于PEPS，local derivative为：

```
O_i^{(x,y)} = ∂ ln ψ / ∂ T^{(x,y)}_i

其中T^{(x,y)}_i是site (x,y)上tensor的第i个分量
```

在VMC sampling中：

```
⟨O_i^* O_j⟩ = (1/N) Σ_k O_i^*(config_k) O_j(config_k)
⟨E_loc O_i^*⟩ = (1/N) Σ_k E_loc(config_k) O_i^*(config_k)
```

### Adam算法的收敛性分析

#### Bias correction的必要性

Adam的moment estimates在初期有bias：

```
E[m_t] = E[g_t] (1-β₁^t) / (1-β₁)
E[v_t] = E[g_t²] (1-β₂^t) / (1-β₂)
```

如果不做bias correction，初期的estimates会严重underestimate真实moments，导致过大的update steps。

#### 学习率的effective scaling

Adam的effective learning rate为：

```
η_eff = η / (√v̂_t + ε) ≈ η / σ_t

其中σ_t是历史梯度的RMS
```

这意味着：
- 梯度consistently large的参数：effective learning rate小
- 梯度noisy但average small的参数：effective learning rate大

### 优化算法在PEPS中的特殊考虑

#### 1. Gauge Freedom

PEPS具有gauge freedom：可以在bond上插入U†U而不改变physical state。这导致：
- Parameter space存在flat directions
- Hessian matrix singular
- 需要gauge fixing或regularization

#### 2. Entanglement Constraint

PEPS的bond dimension限制了能表示的entanglement。优化过程中：
- 梯度可能指向unreachable states
- 需要projected gradient methods
- SVD truncation引入additional noise

#### 3. Monte Carlo Noise

VMC gradient估计包含statistical error：

```
g_estimated = g_true + noise
Var[g_estimated] ∝ 1/N_samples
```

优化算法需要robust to noisy gradients：
- Adam's momentum helps filter noise
- SR's preconditioning can amplify or suppress noise depending on S-matrix condition number

## 高级技巧和故障排除

### 1. Gradient Clipping

当gradient norm过大时，clip到合理范围：

```cpp
// 在optimizer实现中
double grad_norm = CalculateGradientNorm(gradient);
if (grad_norm > clip_threshold) {
  ScaleGradient(gradient, clip_threshold / grad_norm);
}
```

**使用场景**：防止numerical explosion，特别是在optimization early stage。

### 2. Warm Restart

周期性重置momentum/adaptive terms：

```cpp
// 每N步重置Adam的moment estimates
if (iteration % restart_period == 0) {
  ResetMomentEstimates();
}
```

**使用场景**：跳出local minima，重新探索parameter space。

### 3. Learning Rate Warmup

Gradually increase learning rate from 0：

```cpp
double warmup_lr = base_lr * min(1.0, iteration / warmup_steps);
```

**使用场景**：避免初期的large updates破坏预训练state。

### 4. Preconditioning

为gradient添加problem-specific preconditioning：

```cpp
// 例如：normalize by parameter magnitude
preconditioned_grad[i] = gradient[i] / (abs(parameter[i]) + epsilon);
```

### 5. Multi-stage Optimization

不同阶段使用不同策略：

```cpp
// Stage 1: Exploration (高学习率，Adam)
// Stage 2: Refinement (中学习率，SGD)  
// Stage 3: Polishing (低学习率，SR)
```

## 实际案例研究

### 案例1：2D Heisenberg Model (4×4)

**问题**：收敛到错误的local minimum
**诊断**：Initial state太差，gradient信息misleading
**解决方案**：
```cpp
// 1. 更好的initialization
// 2. 多次random restart
// 3. Simulated annealing style learning rate
auto scheduler = std::make_unique<ExponentialDecayLR>(0.1, 0.99, 50);
```

### 案例2：Large system (10×10) with Adam

**问题**：内存不足，收敛缓慢
**诊断**：Adam需要存储per-parameter moments
**解决方案**：
```cpp
// 1. 切换到SGD + momentum
// 2. 减小batch size
// 3. Use gradient accumulation
auto sgd_params = OptimizerFactory::CreateSGDWithDecay(2000, 0.01, 0.98, 100);
```

### 案例3：t-J model的复杂能量landscape

**问题**：频繁震荡，无法稳定收敛
**诊断**：能量surface highly non-convex，梯度noise大
**解决方案**：
```cpp
// 1. 使用Stochastic Reconfiguration
ConjugateGradientParams cg_params{200, 1e-6, 30, 0.01}; // 更保守的参数
// 2. 更大的sampling size减少noise
// 3. Plateau-aware learning rate
auto scheduler = std::make_unique<PlateauLR>(0.05, 0.3, 50, 1e-6);
```

## 完整的代码示例

### 示例1：基础Heisenberg模型优化

```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/optimizer/optimizer_params.h"

void OptimizeHeisenbergModel() {
  using TenElemT = qlten::QLTEN_Complex;
  using QNT = qlten::QNZ2;
  
  // 1. 设置Monte Carlo参数
  MonteCarloParams mc_params;
  mc_params.sample_num = 8000;        // 足够的统计样本
  mc_params.warmup_sample_num = 2000; // 充分的热化
  mc_params.mc_samples_dumpinterval = 100;
  mc_params.filename_postfix = "heisenberg";
  
  // 2. 设置PEPS参数  
  PEPSParams peps_params;
  peps_params.bond_dim = 4;           // 适中的bond dimension
  peps_params.truncate_para = BMPSTruncatePara(
    peps_params.bond_dim,
    peps_params.bond_dim * 20,  // cutoff
    1e-10,                      // trunc_err
    QLTensor<TenElemT, QNT>::GetQNSectorSet().GetQNSctNum(),
    &world
  );
  
  // 3. 设置Stochastic Reconfiguration优化器
  ConjugateGradientParams cg_params{
    100,    // max_iter: 足够的CG迭代
    1e-5,   // tolerance: 平衡精度和速度
    20,     // restart_step: 避免数值累积误差
    0.001   // diag_shift: 正则化参数
  };
  
  auto sr_params = OptimizerFactory::CreateStochasticReconfigurationAdvanced(
    1000,      // max_iterations
    1e-8,      // energy_tolerance: 高精度要求
    1e-6,      // gradient_tolerance
    50,        // plateau_patience: 给足耐心
    cg_params,
    0.1,       // learning_rate: 适中的学习率
    std::make_unique<PlateauLR>(0.1, 0.5, 30, 1e-5)  // 能量平台检测
  );
  
  // 4. 组合VMC参数
  VMCPEPSOptimizerParams vmc_params{
    sr_params,
    mc_params, 
    peps_params,
    "heisenberg_optimized"  // 输出文件前缀
  };
  
  // 5. 创建能量求解器（假设已定义）
  SpinOneHalfHeisenbergSquare energy_solver(4, 4, 1.0);  // 4x4 lattice, J=1
  
  // 6. 创建并执行优化器
  try {
    VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, 
                            SpinOneHalfHeisenbergSquare> executor(
      vmc_params, 
      "initial_state_dir/",  // 初始state路径
      MPI_COMM_WORLD, 
      energy_solver
    );
    
    executor.Execute();
    
    // 7. 获取结果
    std::cout << "Optimization completed. Final energy: " 
              << executor.GetMinimumEnergy() << std::endl;
              
  } catch (const std::exception& e) {
    std::cerr << "Optimization failed: " << e.what() << std::endl;
  }
}
```

### 示例2：快速原型开发with Adam

```cpp
void FastPrototyping() {
  // 快速实验设置：小系统，Adam优化器
  MonteCarloParams mc_params;
  mc_params.sample_num = 2000;        // 较少样本，快速反馈
  mc_params.warmup_sample_num = 500;
  
  PEPSParams peps_params;
  peps_params.bond_dim = 2;           // 小bond dimension
  
  // Adam with aggressive learning rate
  auto adam_params = OptimizerFactory::CreateAdamAdvanced(
    500,        // max_iterations: 快速迭代
    1e-6,       // energy_tolerance: 适中精度
    1e-4,       // gradient_tolerance: 宽松梯度要求
    20,         // plateau_patience: 快速判断收敛
    1e-2,       // learning_rate: 相对激进
    0.9, 0.999, // Adam的标准参数
    std::make_unique<StepLR>(1e-2, 100, 0.8)  // 阶段性衰减
  );
  
  VMCPEPSOptimizerParams vmc_params{adam_params, mc_params, peps_params, "prototype"};
  
  // ... 执行优化
}
```

### 示例3：Production级别的稳健配置

```cpp
void ProductionOptimization() {
  // 生产环境：稳健性优先
  MonteCarloParams mc_params;
  mc_params.sample_num = 20000;       // 大样本量，减少noise
  mc_params.warmup_sample_num = 5000; // 充分热化
  
  PEPSParams peps_params;
  peps_params.bond_dim = 8;           // 较大bond dimension
  
  // 保守的SGD配置
  auto sgd_params = OptimizerFactory::CreateSGDWithDecayAdvanced(
    5000,       // max_iterations: 给足时间
    1e-10,      // energy_tolerance: 化学精度
    1e-8,       // gradient_tolerance: 严格梯度收敛
    100,        // plateau_patience: 避免过早停止
    0.005,      // initial_learning_rate: 保守
    0.98,       // decay_rate: 缓慢衰减
    200         // decay_steps: 较长的decay interval
  );
  
  VMCPEPSOptimizerParams vmc_params{sgd_params, mc_params, peps_params, "production"};
  
  // ... 执行优化，包含更多的error handling和logging
}
```

### 示例4：Multi-stage优化策略

```cpp
class MultiStageOptimizer {
public:
  void OptimizeWithStages(const std::string& initial_state_path) {
    // Stage 1: 快速exploration with Adam
    std::cout << "Stage 1: Exploration phase..." << std::endl;
    auto adam_params = OptimizerFactory::CreateAdam(500, 1e-2);
    auto result1 = RunOptimization(adam_params, initial_state_path, "stage1");
    
    // Stage 2: 稳定收敛with SGD
    std::cout << "Stage 2: Convergence phase..." << std::endl;
    auto sgd_params = OptimizerFactory::CreateSGDWithDecay(1000, 0.005, 0.95, 100);
    auto result2 = RunOptimization(sgd_params, result1.final_state_path, "stage2");
    
    // Stage 3: 高精度polishing with SR
    std::cout << "Stage 3: Polishing phase..." << std::endl;
    ConjugateGradientParams cg_params{150, 1e-6, 25, 0.0005};
    auto sr_params = OptimizerFactory::CreateStochasticReconfiguration(500, cg_params, 0.05);
    auto result3 = RunOptimization(sr_params, result2.final_state_path, "stage3");
    
    std::cout << "Multi-stage optimization completed. Final energy: " 
              << result3.final_energy << std::endl;
  }
  
private:
  struct OptimizationResult {
    double final_energy;
    std::string final_state_path;
  };
  
  OptimizationResult RunOptimization(const OptimizerParams& opt_params,
                                   const std::string& input_path,
                                   const std::string& stage_name) {
    // 实现单阶段优化...
    VMCPEPSOptimizerParams vmc_params{opt_params, mc_params_, peps_params_, stage_name};
    
    VMCPEPSOptimizerExecutor executor(vmc_params, input_path, MPI_COMM_WORLD, energy_solver_);
    executor.Execute();
    
    return {executor.GetMinimumEnergy(), stage_name + "_final"};
  }
  
  MonteCarloParams mc_params_;
  PEPSParams peps_params_;
  Energysolver energy_solver_;
};
```

### 示例5：自定义学习率调度器

```cpp
class PhysicsAwareLRScheduler : public LearningRateScheduler {
private:
  double initial_lr_;
  double min_lr_;
  std::vector<double> energy_history_;
  size_t stagnation_count_;
  double stagnation_threshold_;
  
public:
  PhysicsAwareLRScheduler(double initial_lr, double min_lr = 1e-5, 
                         double stagnation_threshold = 1e-6)
    : initial_lr_(initial_lr), min_lr_(min_lr), 
      stagnation_threshold_(stagnation_threshold), stagnation_count_(0) {}
  
  double GetLearningRate(size_t iteration, double current_energy) const override {
    energy_history_.push_back(current_energy);
    
    // 检测能量是否停滞
    if (energy_history_.size() > 10) {
      double recent_improvement = energy_history_[energy_history_.size()-10] - current_energy;
      if (recent_improvement < stagnation_threshold_) {
        stagnation_count_++;
      } else {
        stagnation_count_ = 0;
      }
    }
    
    // 根据停滞情况调整学习率
    double lr_factor = 1.0;
    if (stagnation_count_ > 20) {
      lr_factor = 0.1;  // 大幅降低学习率
    } else if (stagnation_count_ > 10) {
      lr_factor = 0.5;  // 适度降低学习率
    }
    
    return std::max(min_lr_, initial_lr_ * lr_factor);
  }
  
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    return std::make_unique<PhysicsAwareLRScheduler>(initial_lr_, min_lr_, stagnation_threshold_);
  }
};

// 使用自定义调度器
void UseCustomScheduler() {
  auto custom_scheduler = std::make_unique<PhysicsAwareLRScheduler>(0.01, 1e-5, 1e-7);
  
  OptimizerParams::BaseParams base_params(1000, 1e-8, 1e-6, 50, 0.01, std::move(custom_scheduler));
  SGDParams sgd_params{0.9, false};  // momentum SGD
  
  OptimizerParams params(base_params, sgd_params);
}
```

## VMCPEPS完整工作流程

### 端到端的优化流程

```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_heisenberg_square.h"

int main(int argc, char* argv[]) {
  // 1. MPI初始化
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  try {
    // 2. 系统参数设置
    const size_t Lx = 4, Ly = 4;
    const double J = 1.0;  // Heisenberg coupling
    const size_t bond_dim = 4;
    
    // 3. 创建能量求解器
    SpinOneHalfHeisenbergSquare energy_solver(Ly, Lx, J);
    
    // 4. 设置优化参数
    if (rank == 0) {
      std::cout << "Setting up optimization parameters..." << std::endl;
    }
    
    // Monte Carlo参数
    MonteCarloParams mc_params;
    mc_params.sample_num = 10000;
    mc_params.warmup_sample_num = 2000;
    mc_params.mc_samples_dumpinterval = 500;
    mc_params.filename_postfix = "heisenberg_4x4";
    
    // PEPS参数
    PEPSParams peps_params;
    peps_params.bond_dim = bond_dim;
    peps_params.truncate_para = BMPSTruncatePara(
      bond_dim, bond_dim * 20, 1e-10, 
      QLTensor<qlten::QLTEN_Complex, qlten::QNZ2>::GetQNSectorSet().GetQNSctNum(),
      &MPI_COMM_WORLD
    );
    
    // Optimizer参数（Stochastic Reconfiguration）
    ConjugateGradientParams cg_params{100, 1e-5, 20, 0.001};
    auto opt_params = OptimizerFactory::CreateStochasticReconfigurationAdvanced(
      1500,                    // max_iterations
      1e-8,                   // energy_tolerance
      1e-6,                   // gradient_tolerance
      80,                     // plateau_patience
      cg_params,
      0.1,                    // learning_rate
      std::make_unique<PlateauLR>(0.1, 0.5, 30, 1e-5)
    );
    
    // 5. 组合参数
    VMCPEPSOptimizerParams vmc_params{
      opt_params, mc_params, peps_params, "heisenberg_4x4_D4"
    };
    
    // 6. 创建和执行优化器
    VMCPEPSOptimizerExecutor<qlten::QLTEN_Complex, qlten::QNZ2, 
                            SquareNNUpdater, SpinOneHalfHeisenbergSquare> 
        executor(vmc_params, "random_init/", MPI_COMM_WORLD, energy_solver);
    
    if (rank == 0) {
      std::cout << "Starting optimization..." << std::endl;
    }
    
    executor.Execute();
    
    // 7. 输出结果
    if (rank == 0) {
      std::cout << "Optimization completed!" << std::endl;
      std::cout << "Final minimum energy: " << executor.GetMinimumEnergy() << std::endl;
      std::cout << "Energy per site: " << executor.GetMinimumEnergy() / (Lx * Ly) << std::endl;
      
      // 保存优化轨迹
      const auto& energy_traj = executor.GetEnergyTrajectory();
      const auto& error_traj = executor.GetEnergyErrorTrajectory();
      
      std::ofstream traj_file("optimization_trajectory.dat");
      traj_file << "# Iteration Energy Error\n";
      for (size_t i = 0; i < energy_traj.size(); ++i) {
        traj_file << i << " " << energy_traj[i] << " " << error_traj[i] << "\n";
      }
      traj_file.close();
      
      std::cout << "Optimization trajectory saved to optimization_trajectory.dat" << std::endl;
    }
    
  } catch (const std::exception& e) {
    if (rank == 0) {
      std::cerr << "Error during optimization: " << e.what() << std::endl;
    }
    MPI_Finalize();
    return -1;
  }
  
  MPI_Finalize();
  return 0;
}
```

### 批量参数扫描

```cpp
class ParameterSweepRunner {
public:
  void RunLearningRateSweep() {
    std::vector<double> learning_rates = {0.001, 0.005, 0.01, 0.05, 0.1, 0.2};
    std::vector<std::pair<double, double>> results;  // (lr, final_energy)
    
    for (double lr : learning_rates) {
      std::cout << "Testing learning rate: " << lr << std::endl;
      
      auto opt_params = OptimizerFactory::CreateAdam(500, lr);
      VMCPEPSOptimizerParams vmc_params{opt_params, mc_params_, peps_params_, 
                                       "lr_sweep_" + std::to_string(lr)};
      
      try {
        VMCPEPSOptimizerExecutor executor(vmc_params, initial_state_path_, 
                                         MPI_COMM_WORLD, energy_solver_);
        executor.Execute();
        
        double final_energy = executor.GetMinimumEnergy();
        results.emplace_back(lr, final_energy);
        
        std::cout << "Learning rate " << lr << " -> Energy: " << final_energy << std::endl;
        
      } catch (const std::exception& e) {
        std::cout << "Learning rate " << lr << " failed: " << e.what() << std::endl;
        results.emplace_back(lr, std::numeric_limits<double>::max());
      }
    }
    
    // 找到最佳学习率
    auto best_result = *std::min_element(results.begin(), results.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; });
      
    std::cout << "Best learning rate: " << best_result.first 
              << " with energy: " << best_result.second << std::endl;
  }
  
private:
  MonteCarloParams mc_params_;
  PEPSParams peps_params_;
  std::string initial_state_path_;
  EnergysolverType energy_solver_;
};
```

---

请根据具体系统测试和调整这些代码示例。

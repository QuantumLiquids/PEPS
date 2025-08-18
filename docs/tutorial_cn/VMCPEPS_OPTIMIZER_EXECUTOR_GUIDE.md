# VMCPEPSOptimizerExecutor 完整指南

## 概述

`VMCPEPSOptimizerExecutor` 是用于 PEPS (Projected Entangled Pair States) 变分蒙特卡洛优化的**统一执行引擎**。它通过清晰的模板化架构协调三个基础组件，消除复杂性而非管理复杂性。

**核心设计哲学**：一个执行器，三种策略，零特殊情况。

## 架构：三策略组合模式

### 核心组件

```cpp
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
class VMCPEPSOptimizerExecutor
```

执行器精确组合了三个策略组件：

1. **蒙特卡洛扫描更新器** (MonteCarloSweepUpdater) - 通过具体的更新策略函子定义组态在蒙卡中如何演化
2. **模型能量求解器** (Model Energy Solver) - 定义能量和梯度在特定组态（蒙卡样本）中如何计算  
3. **优化算法** (Optimizer Algorithm) - 定义张量乘积态的参数如何更新

**设计洞察**：每个组件都有单一职责。执行器只是协调它们的交互，没有复杂的条件逻辑。

---

## 组件一：蒙特卡洛更新器

### 概念：配置演化策略

蒙特卡洛更新器定义**组态如何变化**的采样过程。它是一个函子，在保持detailed balance的前提下更新配置，确保正确的统计采样。

### 为什么重要

更新器策略决定了采样效率和遍历性。

### 内置更新器类型

位于：`include/qlpeps/vmc_basic/configuration_update_strategies/`

#### 1. 最近邻交换 (`MCUpdateSquareNNExchange`)
```cpp
// 适用于：spin-1/2 海森堡模型，t-J模型等具有自旋-粒子数守恒的系统，
// 不适用于Hubbard model, Spin-1 Heisenberg model
using UpdaterType = MCUpdateSquareNNExchange;
```

**算法**：依次遍历所有相邻格点对，先遍历所有横向的键，再遍历所有纵向的键。对于每一对相邻格点，提出粒子交换的更新提议，并根据波函数振幅的比值决定是否接受该交换，从而实现详细平衡和有效采样。

**使用场景**：
- 具有U(1)对称性的自旋1/2海森堡模型
- 粒子数守恒的t-J模型

#### 2. 全配置空间 (`MCUpdateSquareNNFullSpaceUpdate`)
```cpp
// 适用于：几乎所有模型，尤其是没有严格守恒律的模型, e.g. Transever Ising model.
using UpdaterType = MCUpdateSquareNNFullSpaceUpdate;
```

**算法**：对每个键，考虑所有可能的局域配置，按概率权重采样。顺序也是先遍历所有横向的键，再遍历所有纵向的键。

**使用场景**：
- 几乎所有模型

**不适用场景**
- 严格投影掉最近邻sites上同时激发的PXP model

**注意**
对于spin-1/2 海森堡模型，t-J模型而言，虽然也可以使用MCUpdateSquareNNFullSpaceUpdate，
采用MCUpdateSquareNNExchange相当于在U1 conserved subspace来采样，可能可以增加采样效率，并因为波函数投影的原因降低TPS最后优化的能量结果。


#### 3. 三格点三角更新 (`MCUpdateSquareTNN3SiteExchange`)  
```cpp
using UpdaterType = MCUpdateSquareTNN3SiteExchange;
```

**算法**：每次更新依次遍历所有相邻三格点，先遍历所有横向的三格点组元，再遍历所有纵向的键三格点组元。

**使用场景**：
提高MCUpdateSquareNNExchange的接受率

### 更新器接口契约

所有更新器必须实现：
```cpp
template<typename TenElemT, typename QNT>
void operator()(const SplitIndexTPS<TenElemT, QNT>& sitps,
                TPSWaveFunctionComponent<TenElemT, QNT>& tps_component,
                std::vector<double>& accept_rates);
```

**职责**：
- 更新 `tps_component.config` (粒子配置)
- 更新 `tps_component.amplitude` (波函数振幅)
- 更新 `tps_component.tn` （代表波函数分量的单层二维张量网络）
- 记录接受率用于诊断
- 维持detailed balance以保证正确采样

### 选择合适的更新器

**决策树**：
```
你的模型的组态只需要做交换就可以遍历基态子空间的基矢吗？
├── 是 → 接受效率高吗？
│   ├── 是 → MCUpdateSquareNNExchange 
│   └── 否 → MCUpdateSquareTNN3SiteExchange (一定程度提高接受效率)
└── 否 → MCUpdateSquareNNFullSpaceUpdate (全配置空间)
```

---

## 组件二：模型能量求解器

### 概念：能量和梯度计算引擎

模型能量求解器计算特定粒子配置和TPS状态的**局域能量和梯度信息**。它封装了所有模型特定的哈密顿量细节，是连接物理问题和优化算法的桥梁。

VMC优化的核心是最小化能量期望值 $\langle H \rangle$，这需要计算每个蒙特卡洛样本的局域能量 $E_{\text{loc}} = \frac{\langle \text{config}|H|\psi \rangle}{\langle \text{config}|\psi \rangle}$ 和波函数对参数的梯度。能量求解器的职责就是高效、精确地完成这个计算。

### 使用方式：模板参数注入

与蒙特卡洛更新器一样，能量求解器也是一个**策略对象**，通过模板参数注入到 `VMCPEPSOptimizerExecutor` 中：

```cpp
// 声明一个使用特定求解器的执行器类型
using MyExecutor = VMCPEPSOptimizerExecutor<TenElemT, QNT, MyUpdater, MyEnergySolver>;

// 实例化求解器并传入
MyEnergySolver solver(...);
MyExecutor executor(params, tps, comm, solver);
```

这种设计将物理模型的复杂性与优化流程完全解耦。

### 内置能量求解器

我们提供了一系列针对标准物理模型的内置求解器：
- `SquareSpinOneHalfXXZModel`
- `SpinOneHalfTriangleHeisenbergSquarePEPS`
- `SpinOneHalfTriangleHeisenbergJ1J2SquarePEPS`
- `SquaretJModel`

### 扩展性：添加新模型

系统是完全可扩展的。你可以通过继承 `ModelEnergySolver` 基类并实现必要的接口来支持任何新的物理模型。

**详细的技术实现、API契约和自定义指南，请参考独立的[模型能量求解器指南](MODEL_ENERGY_SOLVER_GUIDE.md)。**

---

## VMC中的梯度计算数学原理

变分蒙特卡洛(VMC)优化的核心是寻找最优变分参数 $\{\theta_i\}$ 以最小化能量期望值。本节详细推导复数参数情况下的梯度计算公式。

### 问题设置

设变分波函数为 $|\Psi(\theta)\rangle = \sum_S \Psi(S; \theta) |S\rangle$，其中：
- $\{S\}$ 是配置空间的基矢
- $\{\theta_i\}$ 是变分参数（在PEPS中是张量元素）
- 目标：最小化 $E(\theta) = \frac{\langle \Psi(\theta)| H |\Psi(\theta)\rangle}{\langle \Psi(\theta)|\Psi(\theta)\rangle}$

### 复数参数的微积分

当参数 $\theta_i$ 是复数时，将 $\theta_i$ 和 $\theta_i^*$ 视为独立变量：
- $\Psi(S; \theta)$ 是 $\{\theta_i\}$ 的全纯函数
- $\Psi^*(S; \theta^*)$ 是 $\{\theta_i^*\}$ 的全纯函数

梯度下降的方向是 $-\frac{\partial E}{\partial \theta_i^*}$。

### 梯度公式推导

从能量期望值的定义开始：
\[
E = \frac{\sum_S |\Psi(S)|^2 E_{\mathrm{loc}}(S)}{\sum_S |\Psi(S)|^2}
\]

其中局域能量为：
\[
E_{\mathrm{loc}}(S) = \sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)} \langle S'| H | S\rangle
\]

应用链式法则计算梯度：
\[
\frac{\partial E}{\partial \theta_i^*} = \frac{\partial_{\theta_i^*}(\langle \Psi|H|\Psi\rangle)}{\langle \Psi|\Psi\rangle} - E \frac{\partial_{\theta_i^*} (\langle \Psi|\Psi\rangle)}{\langle \Psi|\Psi\rangle}
\]

通过变分分析，可以证明：
\[
\frac{\partial E}{\partial \theta_i^*} = \langle E_{\mathrm{loc}}^* \cdot O_i^* \rangle - \langle E_{\mathrm{loc}}^* \rangle \langle O_i^* \rangle
\]

其中：
- $O_i^*(S) = \frac{\partial \ln \Psi^*(S; \theta_i^*)}{\partial \theta_i^*}$ 是对数导数
- $E_{\mathrm{loc}}^*$ 是局域能量的复共轭

### 实现细节

**梯度计算的关键点**：

1. **局域能量**：使用 $E_{\mathrm{loc}}(S) = \sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)} \langle S'| H | S\rangle$

2. **复共轭处理**：在代码中通过 `ComplexConjugate(local_energy)` 实现 $E_{\mathrm{loc}}^*$

3. **对数导数**：通过张量网络的"洞"计算 $O_i^*$

**为什么需要复共轭**：
- **数学要求**：复数微积分的自然结果
- **数值稳定性**：在随机重构(SR)中提供更好的收敛性
- **物理一致性**：保证优化过程中能量期望值始终为实数

### 算法实现

代码中的梯度累积：
```cpp
// 每个蒙特卡洛样本
TenElemT local_energy_conjugate = ComplexConjugate(local_energy);
gten_sum_ += gradient_tensor;                              // ∑ O_i*
g_times_energy_sum_ += local_energy_conjugate * gradient_tensor;  // ∑ E_loc* · O_i*
```

最终梯度：
```cpp
gradient = (g_times_energy_sum_ - mean_energy * gten_sum_) / n_samples;
```

这精确实现了理论公式 $\frac{\partial E}{\partial \theta_i^*} = \langle E_{\mathrm{loc}}^* \cdot O_i^* \rangle - \langle E_{\mathrm{loc}}^* \rangle \langle O_i^* \rangle$。

---

## 组件三：优化算法

第三个组件是优化算法本身。这在[优化器指南](OPTIMIZER_GUIDE.md)中有详细介绍，这里说明它如何集成：

### 与VMC框架的集成

```cpp
VMCPEPSOptimizerParams params{
  optimizer_params,  // 算法选择 (Adam, SGD, Stochastic Reconfiguration)
  mc_params,         // 蒙特卡洛采样参数
  peps_params        // PEPS键维数和截断
};
```

**工作流程集成**：
1. **采样阶段**：蒙特卡洛更新器生成一个新配置，能量求解器立即计算该配置的能量和梯度样本
2. **重复采样**：重复步骤1，收集大量能量和梯度样本
3. **MPI统计**：统计所有进程的样本，计算能量和梯度的平均值
4. **优化更新**：
   - 对于普通算法：将平均能量和梯度送入优化器
   - 对于SR算法：将所有能量和梯度样本数据送入优化器进行随机重构
5. **参数更新与广播**：优化器更新TPS参数并广播到所有进程
6. **重复**：返回步骤1，直到收敛

---

## 完整集成示例

### 基本使用模式

```cpp
#include "qlpeps/qlpeps.h"

using TenElemT = qlten::QLTEN_Complex;
using QNT = qlten::QNZ2;

// 1. 选择你的三种策略
using MonteCarloUpdater = MCUpdateSquareNNExchange;
using EnergySolver = SquareSpinOneHalfXXZModel;

// 2. 配置参数
OptimizerParams opt_params = /* 参见优化器指南 */;
MonteCarloParams mc_params = /* 参见蒙特卡洛API指南 */;
PEPSParams peps_params = /* 参见VMC数据持久化指南 */;

VMCPEPSOptimizerParams vmc_params{opt_params, mc_params, peps_params, "output"};

// 3. 初始化求解器
EnergySolver energy_solver(ly, lx, J_coupling);

// 4. 创建并执行
VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloUpdater, EnergySolver> 
  executor(vmc_params, initial_tps, MPI_COMM_WORLD, energy_solver);

executor.Execute();
```

### 高级：自定义组件组合

```cpp
// 阻挫系统的自定义三格点更新器
using CustomUpdater = MCUpdateSquareTNN3SiteExchange;

// 自定义J1-J2能量求解器
using CustomSolver = SpinOneHalfTriangleHeisenbergJ1J2SquarePEPS;

// 高精度的随机重构
auto opt_params = OptimizerFactory::CreateStochasticReconfiguration(
  1000, cg_params, 0.1
);

VMCPEPSOptimizerExecutor<TenElemT, QNT, CustomUpdater, CustomSolver>
  executor(vmc_params, initial_tps, MPI_COMM_WORLD, custom_solver);
```

---

## 组件交互工作流程

### 详细执行流程

```
每次优化迭代：
  ┌─────────────────────────────────────────┐
  │ 1. 单步采样&评估 (重复N次)                 │
  │   ├─ 蒙特卡洛更新器生成新配置              │
  │   ├─ 能量求解器立即计算 E_loc             │
  │   └─ 能量求解器立即计算梯度样本            │
  └─────────────────────────────────────────┘
                      ↓
  ┌─────────────────────────────────────────┐
  │ 2. MPI数据收集与统计                      │
  │   ├─ 收集所有进程的能量和梯度样本          │
  │   ├─ 计算能量和梯度的平均值               │
  │   └─ SR类算法保留原始样本数据             │
  └─────────────────────────────────────────┘
                      ↓
  ┌─────────────────────────────────────────┐
  │ 3. 优化器更新                     │
  │   ├─ 普通算法：使用平均能量和梯度(主进程)      │
  │   ├─ SR算法：使用所有样本数据进行随机重构（所有进程 │
  └─────────────────────────────────────────┘
                      ↓
  ┌─────────────────────────────────────────┐
  │ 4. 参数广播                               │
  │   └─ 更新的TPS参数广播到所有进程          │
  └─────────────────────────────────────────┘
```

### MPI并行化策略

**令人尴尬的并行采样**：
- 每个MPI进程运行独立的蒙特卡洛链
- 采样与处理器数量完美缩放
- 采样阶段无通信

**集体优化**：
- 通过 `MPI_Allreduce` 收集能量/梯度统计
- 在主进程上计算参数更新
- 通过 `MPI_Bcast` 广播更新的参数

---

## 设计原则：为什么这个架构有效

### 1. 单一职责原则
- **蒙特卡洛更新器**：只处理配置更新
- **能量求解器**：只处理能量/梯度计算
- **优化器**：只处理参数更新
- **执行器**：只协调交互

### 2. 模板化策略模式
```cpp
template<typename MonteCarloUpdater, typename EnergySolver>
```
- 编译时策略选择
- 零运行时开销
- 类型安全组合
- 易于扩展新策略

### 3. CRTP性能优化
- 通过CRTP实现静态多态
- 无虚函数开销
- 编译时优化机会
- 干净接口，无运行时成本

### 4. 快速失败验证
- 初始化时配置验证
- 振幅一致性检查
- 无效状态时早期终止
- 清晰的错误诊断

---


## 与其他组件的集成

### 与现有教程的连接

此执行器与其他几个系统组件集成：

1. **参数管理**：算法详情参见[优化器指南](OPTIMIZER_GUIDE.md)
2. **数据持久化**：I/O控制参见[VMC数据持久化指南](VMC_DATA_PERSISTENCE_GUIDE.md)
3. **API模式**：构造模式参见[蒙特卡洛PEPS API指南](MONTE_CARLO_PEPS_API_GUIDE.md)
4. **高层概述**：生态系统上下文参见[顶级API](TOP_LEVEL_APIs.md)

### 测量集成

相同的蒙特卡洛更新器和能量求解器策略可以重用于测量：

```cpp
// 优化阶段
VMCPEPSOptimizerExecutor<TenElemT, QNT, UpdaterType, SolverType> 
  optimizer(opt_params, initial_tps, comm, solver);
optimizer.Execute();

// 使用相同策略的测量阶段
MonteCarloMeasurementExecutor<TenElemT, QNT, UpdaterType, MeasurementSolverType>
  measurement(measurement_params, optimized_tps, comm, measurement_solver);
measurement.Execute();
```

---

## 故障排除和最佳实践

### 组件选择指南

**基于以下条件选择蒙特卡洛更新器**：
- 模型中的守恒律
- 所需的遍历性质
- 计算效率需求

### 常见陷阱

1. **组件不匹配**：确保更新器遵守与能量求解器相同的守恒律
2. 一般来说，蒙特卡洛更新器带来的希尔伯特子空间约束可以比张量网络对称性的约束更多，但是一般不会更少。
---

## 实际应用案例

### 案例1：4×4海森堡模型优化

```cpp
// 系统参数
const size_t ly = 4, lx = 4;
const double J = 1.0;
const size_t bond_dim = 4;

// 选择策略组件
using UpdaterType = MCUpdateSquareNNExchange;      // 最近邻交换
using SolverType = SpinOneHalfHeisenbergSquare;    // 海森堡求解器

// 配置优化参数
ConjugateGradientParams cg_params{100, 1e-5, 20, 0.001};
auto sr_params = OptimizerFactory::CreateStochasticReconfiguration(
  1000, cg_params, 0.1
);

// 蒙特卡洛参数
MonteCarloParams mc_params(8000, 2000, 5, initial_config, false, "");

// PEPS参数
PEPSParams peps_params;
peps_params.bond_dim = bond_dim;
peps_params.truncate_para = BMPSTruncatePara(/*...*/);

// 创建执行器
VMCPEPSOptimizerParams vmc_params{sr_params, mc_params, peps_params, "heisenberg_4x4"};
SolverType energy_solver(ly, lx, J);

VMCPEPSOptimizerExecutor<QLTEN_Complex, QNZ2, UpdaterType, SolverType> 
  executor(vmc_params, initial_tps, MPI_COMM_WORLD, energy_solver);

executor.Execute();
```

### 案例2：阻挫J1-J2模型

```cpp
// 阻挫系统需要三格点更新
using FrustrationUpdater = MCUpdateSquareTNN3SiteExchange;
using FrustrationSolver = SpinOneHalfTriangleHeisenbergJ1J2SquarePEPS;

// J1-J2参数
double J1 = 1.0, J2 = 0.5;  // 阻挫比率
FrustrationSolver solver(ly, lx, J1, J2);

// 使用Adam优化器进行快速收敛
auto adam_params = OptimizerFactory::CreateAdam(1000, 1e-3);

VMCPEPSOptimizerExecutor<QLTEN_Complex, QNZ2, FrustrationUpdater, FrustrationSolver>
  executor(vmc_params, initial_tps, MPI_COMM_WORLD, solver);
```

### 案例3：t-J模型高精度计算

```cpp
// t-J模型特定设置
using tJUpdater = MCUpdateSquareNNExchange;  // 守恒粒子数
using tJSolver = /* 适当的t-J求解器 */;

// 高精度随机重构设置
ConjugateGradientParams precise_cg{200, 1e-7, 30, 0.0001};
auto precise_sr = OptimizerFactory::CreateStochasticReconfigurationAdvanced(
  2000,      // max_iterations
  1e-10,     // energy_tolerance
  1e-8,      // gradient_tolerance  
  100,       // plateau_patience
  precise_cg,
  0.05,      // conservative learning_rate
  std::make_unique<PlateauLR>(0.05, 0.3, 50, 1e-6)
);

// 大样本量减少噪声
MonteCarloParams precise_mc(20000, 5000, 10, config, false, "");
```

---

## 总结

`VMCPEPSOptimizerExecutor` 体现了良好的软件设计：

- **组合优于继承**：组合三个策略组件
- **单一职责**：每个组件都有一个明确定义的工作
- **基于模板的灵活性**：易于扩展而无需代码更改
- **无特殊情况**：无论组件选择如何，都有干净、统一的接口

执行器的力量来自于**在架构层面消除复杂性**而不是试图管理它。通过清晰地分离三个关注点（采样、能量评估、参数更新），使得独立理解、扩展和优化每个组件变得直接了当。

**记住**：最好的复杂性是你永远不必思考的复杂性。

---

## 相关英文文档

如需更详细的技术信息，请参考英文版文档：

- [VMCPEPSOptimizerExecutor Complete Guide (English)](VMCPEPS_OPTIMIZER_EXECUTOR_GUIDE.md)
- [Optimizer Guide (English)](../OPTIMIZER_GUIDE.md)
- [Monte Carlo PEPS API Guide (English)](../MONTE_CARLO_PEPS_API_GUIDE.md)
- [VMC Data Persistence Guide (English)](../VMC_DATA_PERSISTENCE_GUIDE.md)

本指南涵盖了VMCPEPSOptimizerExecutor的核心概念，重点解释了model energy solver和monte carlo updater这两个关键组件的作用和选择策略。


//TODO for this doc: Paramters
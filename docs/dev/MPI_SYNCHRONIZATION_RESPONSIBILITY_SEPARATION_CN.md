# VMC PEPS Executor MPI 同步职责分离

**类型:** 架构设计指南  
**范围:** 核心框架  
**受众:** PEPS 开发者  

## 概述

本文档定义了 PEPS 架构中的 MPI 同步职责分离，建立清晰的组件边界并消除冗余的通信模式。

## 设计决策

### 核心原则
谁用谁分发。

### 具体描述
**非SR类：**

- Optimizer 组件对波函数的更新不执行 MPI 广播。仅负责Master上的波函数更新。
- Energy evaluator 输入态仅保证master有效，自己分发用于能量梯度计算。
- Energy evaluator 输出仅需在主节点提供能量和梯度结果。

**SR类：**

- Optimizer 组件对波函数的更新不执行 MPI 广播。 仅负责Master上的波函数更新。SR求解中自己负责广播能量期望值。
- Energy evaluator 输入态仅保证master有效，自己分发用于能量梯度计算。
- Energy evaluator 输出仅需在主节点提供能量和梯度结果。给Optimizer提供所有节点的能量和梯度样本数据。


### 工作流文档

```text
迭代流程:
1. Energy evaluator 接收波函数                 ← 仍为 master 节点  
2. Energy evaluator 为 MC 采样广播           
3. 分布式 Monte Carlo 执行。SR类存下样本数据。    ← 所有节点
4. 能量梯度收集到 master 节点，喂给Optimzer。对SR类也喂入能量、梯度样本数据。
5. Optimizer 计算更新状态 (仅 master 节点)       ← 除SR需要用到MPI外,  slave节点无职责。
6. 返回步骤 1                                  

```

## 相关代码、文档

- **实现**: `include/qlpeps/optimizer/optimizer_impl.h`, `optimizer.h`
- **Energy Evaluator**: `include/qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h`
- **测试策略**: `OPTIMIZER_TESTING_STRATEGY.md`

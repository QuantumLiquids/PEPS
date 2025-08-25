# 项目概览
QuantumLiquids/PEPS 是一个用于有限尺寸 Projected Entangled Pair States (PEPS)
与相关张量网络算法的开源 C++ 库，用于高效模拟二维有限尺寸量子多体系统。

## 目标用户
面向量子多体领域的爱折腾的科研牛马。场景包括：
- 凝聚态强关联体系：磁性与超导
- 量子计算与量子模拟平台
- 冷原子体系
等

## Why PEPS?
PEPS 是研究二维强关联系统的强大张量网络方法。当格点系统的宽度逐渐加宽，DMRG边际效益逐渐递减。
截止作者写下这一文档（2025年8月），state-of-the-art的DMRG code 也无法计算超越宽度10的t-J/Hubbard. 
PEPS 在超越这一尺寸的计算上非常有优势。

**关联项目**: [QuantumLiquids/UltraDMRG](https://github.com/QuantumLiquids/UltraDMRG)

## Why QuantumLiquids/PEPS?
 - **高性能 C++ 实现** 采用现代 C++ 编写，保证数值计算的高性能和可扩展性，适合大规模模拟与 HPC 环境。也很容易迁移至GPU环境。[^1]
 - **模块化与可维护性** 清晰的架构设计，基于 CMake 构建，避免“屎山”式代码；仅头文件库，用户可轻松在自己的科研项目中集成与扩展。
 - **Friendly APIs** 提供从最小示例到复杂计算的完整链条，便于快速验证想法并开展研究。
 - 其实我不知道有没有同类开源竞品

## 项目能做什么
本项目提供了对有限尺寸PEPS/TPS的Simple Update和VMC的优化的基础支持，以及通过MC采样来计算关联函数的支持。
提供部分内置常用模型（XXZ, t-J, Hubbard等），对模型的扩展和采样策略保持扩展性。
优化算法包含了机器学习领域的标准一阶算法(SGD, AdaGrad, Adam等)，二阶算法（LBFGS),
以及标准的VMC方法SR。

[^1]: 我就没见过哪个HPC集群预装过Julia的。Julia的GPU统一编程框架就是个笑话。
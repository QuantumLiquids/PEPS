# 01: VMC 优化：横场 Ising 模型

本篇介绍如何用Variational Monte Carlo的方法优化横场 Ising 模型的PEPS/TPS波函数。
由于本篇采用Simple Update得到的PEPS作为初始态，我们要求读者已经先行运行了《Getting Started》中的Simple Update 示例。这一示例会导出波函数在目录`peps`中。VMC将在相同目录下运行。

**示例代码位置：** `examples/transverse_field_ising_vmc_optimize.cpp`
请将代码拷贝至与simple update教程中相同的目录下。并以相同的方式在CMakeLists.txt中add_executable。

## 目标模型
- 哈密顿量：\[ H = - \sum_{\langle i,j \rangle} Z_i Z_j - h \sum_i X_i \]
- 本例依然采用 `h = 0.5`，格点 `4x4`，局域希尔伯特空间维度 `2`。

## 代码解释

### 波函数的数据类型
在张量网络领域的术语中，PEPS(Projected entangled pair state)与TPS(Tensor Product State)往往不做区分。在本库中，我们约定用PEPS来表示由bond上的weight tensor和（局部正则）张量而构成的二维张量网络；用TPS来表示没有bond上的weight tensor，格点和格点间张量直接连接的张量网络。构造两种数据类型是必要的：对于simple update 而言，局部的weight tensor可以用来做近似的环境；而对于global update (e.g. VMC)，引入weight tensor会带来冗余的自由度。

在QuantumLiquids/PEPS中，我们还引入了一种特殊的数据类型，SplitIndexTPS. 基于蒙特卡洛采样的测量需要将张量网络投影至direct product state上，并计算相应的播函数振幅。处于对性能的考虑，提前将张量按照物理物理指标拆分可以规避测量中显示的投影。因而我们设计了这一数据类型。

我们在头文件"qlpeps/api/conversions.h"中提供了这三种数据类型的显示类型转换的有名函数。另外，在早期的实现中，考虑到TPS和SplitIndexTPS有相同的物理意义，我们提供了他们之间的隐式类型转换。但这在软件工程上可能不是好的设计，我们正在考虑将其移除。

在示例代码中
1) 从硬盘中加载 Simple update得到的PEPS
- 使用 `SquareLatticePEPS::Load(path)` 逐个张量加载 `Gamma/Lambda`。

2) 转换为 `SplitIndexTPS`
- 显式 API（推荐）：
  ```cpp
  #include "qlpeps/api/conversions.h"
  auto sitps = qlpeps::ToSplitIndexTPS<TenElemT, QNT>(peps);
  ```
`sitps`将后续喂给VMC作为初始态。

### VMC 的输入信息

运行VMC除了要给初始态，必要的信息还包括
1. 模型（哈密顿量）
2. 蒙特卡洛采样策略以及蒙特卡洛采样参数
3. 优化算法及参数
4. 对于张量网络算法，还包含张量网络收缩参数。

本库提供了flexible的方法来输入以上信息。同时对于不熟悉Cpp的人来说（我知道你们大多数做物理的都不熟悉），在tutorial中解释在示例中的代码是如何运作的，是很困难的。好在代码是相当抽象且模块化的，我相信对于熟悉Cpp的人也无需多解释。因而以下仅仅说明这些信息是如何对应示例代码中的。

#### 模型：哈密顿量
位于95行的
    TransverseFieldIsingSquare model(h);
定义了计算的模型，并在108行将模型传递给VMC。这是本库build-in 模型。

#### 蒙特卡洛采样策略与参数
位于108行VMC参数的最后一个参数
```cpp
MCUpdateSquareNNFullSpaceUpdate{}
```
定义了采用最近邻键顺序全希尔伯特空间更新的蒙特卡洛更新策略。

蒙特卡洛参数集由69到75行
```cpp
 MonteCarloParams mc_params(
        /*num_samples=*/500,
        /*num_warmup_sweeps=*/200,
        /*sweeps_between_samples=*/2,
        /*initial_config=*/init_config,
        /*is_warmed_up=*/false,
        /*config_dump_path=*/"./vmc_configs");
```
定义。


| 算法参数 | 变量 | 说明 |
|--------------|----------|----------|
| **样本数量** | `num_samples=500` | 每次能量/梯度评估中，每条马尔可夫链上采样次数 |
| **预热步数** | `num_warmup_sweeps=200` | 正式采样之前，单条马尔可夫链上达到平衡前的扫描次数 |
| **样本间隔** | `sweeps_between_samples=2` | 每次能量/梯度采样之间，样本间间隔的扫描次数 |
| **初始组态** | `init_config` | 初始组态 |
| **初始组态是否已经预热** | `is_warmed_up=false` | 初始组态是否已经预热 |
| **组态存储路径** | `"./vmc_configs"` | 优化流程结束后，最终采样组态存储路径（可用于后续蒙卡计算而无需重新预热） |

这里采用的初始组态为4x4格点上8个自旋朝上和8个自旋朝下的随机组态。


#### 优化算法及参数
位于88-91行创建了 SR 优化器参数并指定共轭梯度(CG)线性求解器与学习率：
```cpp
ConjugateGradientParams cg_params(/*max_iter=*/100, /*tolerance=*/1e-5, /*restart=*/20, /*diag_shift=*/1e-3);
auto opt_params = OptimizerFactory::CreateStochasticReconfiguration(
    /*max_iterations=*/40, cg_params, /*learning_rate=*/0.1);
```

| 项 | 变量 | 说明 |
|----|------|------|
| **优化算法** | SR | Stochasitic reconfiguration. VMC基操 |
| **最大优化步数** | `max_iterations=40` | 总优化迭代次数 |
| **CG 最大迭代步数** | `max_iter=100` | 共轭梯度最大迭代次数 |
| **CG 收敛阈值** | `tolerance=1e-5` | 残差收敛判据 |
| **CG 重启周期** | `restart=20` | 共轭梯度重启周期 |
| **对角修正** | `diag_shift=1e-3` | 关联矩阵(S)对角正则化，改善条件数 |
| **学习率** | `learning_rate=0.1` | 参数更新步长 |

优化参数与采样/收缩参数在92行组装为统一的 `VMCPEPSOptimizerParams`：
```cpp
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params, /*tps_dump_path=*/"./optimized_tps");
```

#### 张量网络收缩参数
位于78-85行设置了 BMPS 边界收缩的截断策略，并封装为 `PEPSParams`：
```cpp
// SVD 压缩工厂方法，仅需截断参数
BMPSTruncatePara trunc_para = BMPSTruncatePara::SVD(
    /*D_min=*/2,
    /*D_max=*/8,
    /*trunc_err=*/1e-14);
PEPSParams peps_params(trunc_para);
```

| 项 | 变量 | 说明 |
|----|------|------|
| **截断方案** | `SVD()` | 采用 SVD 压缩方案（仅需 D_min/D_max/trunc_err） |
| **最小截断维数** | `D_min=2` | Boundary MPS 的最小保留态 |
| **最大截断维数** | `D_max=8` | Boundary MPS 的最大保留态 |
| **截断误差阈值** | `trunc_err=1e-14` | 允许的截断误差 |


这些参数控制 PEPS 收缩的精度-成本权衡：更大的 `D_max`/更小的 `trunc_err` 提升精度但计算更昂贵。事实上，SVD本身是至少O(D^7)的计算复杂度，比变分方法要更加昂贵。这里采用SVD作为学习例子。生产环境建议采用变分方法。

### VMC 执行与导出
一键接口会在内部完成优化流程并返回已执行完成的执行器指针：
```cpp
auto executor = VmcOptimize(
    params,                // 组合了优化器/采样/收缩/导出路径
    sitps,                 // 初始 SplitIndexTPS
    MPI_COMM_WORLD,        // MPI 通信器
    model,                 // 横场 Ising 模型
    MCUpdateSquareNNFullSpaceUpdate{}); // 采样更新策略

executor->DumpData(/*release_mem=*/false);
```

- 执行：`VmcOptimize(...)` 会按 `peps_params` 收缩，按 `mc_params` 采样，按 `opt_params` 进行 SR(+CG) 更新。
- 导出：`DumpData(false)` 将
  - 把优化40步后的 TPS 写入 `./optimized_tpsfinal/`以及优化中得到的评估能量最低的态写入 `./optimized_tpslowest/`（来自 `VMCPEPSOptimizerParams` 中的 `tps_dump_path`）。
  - 把最终蒙卡组态写入 `./vmc_configs/`（来自 `MonteCarloParams::config_dump_path`）。
- `release_mem` 表示是否释放存储的数据在优化器中占据的内存。

## 编译
与simple update教程中相同。

## 运行: 
- 运行命令：
```bash
mpirun -n 4 ./examples/transverse_field_ising_vmc_optimize
```
MPI，多进程，每进程单线程。每次能量计算所用的Markov 采样数为 500 * 4 = 2000。可按需调整 `-n` 进程数，也是蒙特卡洛链的数目。


## 后记
不知道读者是否觉得参数有点多。连Conjugate Gradient这种参数都暴露给用户。这也是本库设计的核心：No magic number. 用户有权利通过参数设置了解程序具体做了什么，同时也对程序的稳定运行负责。

事实上，正如我前面所言，我们提供了flexible的接口来设置不同优化器、模型、蒙特卡洛采样策略等。这使得程序充满了工厂方法和冗长的参数列表。曾经有人建议我的DMRG程序可以设置的用户友好一些。我认为这是把自己推向深渊的思想。好用的话，老板自己就算了，要你做什么？
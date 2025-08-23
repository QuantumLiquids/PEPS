### 推荐工作流：Simple Update → VMCPEPSOptimizer → MCPEPSMeasurer

本页提供项目推荐的端到端使用路径，帮助用户用最少的概念实现从初始化到优化、再到测量的完整闭环。

## 适用范围
- 有限尺寸、方格晶格（当前仅 OBC 的 Square lattice PEPS 完整支持）
- 任意模型能量求解器与蒙特卡洛更新器，只要满足对应接口契约
- 单/多进程 MPI 运行

## 总览
1) 用 Simple Update 得到一个“可用”的初始 PEPS （无MPI）
2) 转成 SplitIndexTPS，做必要的规范化
3) 通过 VMCPEPSOptimizer 进行变分优化（SR/SGD/AdaGrad等）
4) 用 MCPEPSMeasurer 对优化后态进行可观测量测量

## 步骤 1：Simple Update 生成初态
- 入口：`qlpeps/algorithm/simple_update/simple_update.h` 及具体实现（如 Square lattice 最近邻）
- 典型参数：`SimpleUpdatePara{steps, tau, Dmin, Dmax, Trunc_err}`
- 产物：`SquareLatticePEPS<TenElemT, QNT>`（或 dump 到目录）

要点：
- Simple Update 是用户“第一步”，用于快速得到一个合理初态；之后的 VMC 会显著改进能量
- 输出的 PEPS 可通过 `DumpResult(path)` 落盘，便于后续复现

## 步骤 2：TPS → SplitIndexTPS 转换与规范化
- 转换：`SplitIndexTPS(const TPS& tps)` 构造函数
- 规范化（可选）：`NormalizeAllSite()` 或按站点 `NormalizeSite()`；必要时可做 `ScaleMaxAbsForAllSite()` 避免后续计算数值过大或过小。（加入波函数归一，平均每个波函数份量的大小会是1/2^N量级，这对数值计算是灾难！应该让采样空间里占主导的波函数份量的大小在O(1).)

示例：
```cpp
using TenElemT = qlten::QLTEN_Complex;
using QNT = qlten::special_qn::TrivialRepQN;
using PEPST = qlpeps::SquareLatticePEPS<TenElemT, QNT>;
using SITPST = qlpeps::SplitIndexTPS<TenElemT, QNT>;

PEPST peps = /* 来自 Simple Update 的结果或从文件加载 */;
SITPST sitps(peps);   // TPS → SplitIndexTPS
sitps.NormalizeAllSite();
```

## 步骤 3：VMCPEPSOptimizer 变分优化
- 入口：`qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h`
- 参数：`VMCPEPSOptimizerParams{ OptimizerParams, MonteCarloParams, PEPSParams, dump_path }`
- 模板策略：`MonteCarloSweepUpdater`（用户入参）与 `EnergySolver`
- 输出：优化轨迹（能量、误差、梯度范数），最佳状态与当前状态（可 Dump）

MPI 契约（开发者文档内容）：
- 优化器回调（能量、梯度）仅在主进程有效；优化更新仅在主进程生效
- SR 中间环节需要全进程参与 CG 求解（算法内部并行），但最终更新仍只在主进程
- 能量评估器负责“状态广播”；是否广播能量不在契约中

极简骨架：
```cpp
using Updater    = MCUpdateSquareNNExchange<TenElemT, QNT>; // 示例更新器
using EnergySolv = SquareSpinOneHalfXXZModel;               // 示例模型

OptimizerParams opt = /* OptimizerFactory 或 Builder 生成 */;
MonteCarloParams mc = /* 样本、预热、步进、初始配置 */;
PEPSParams pepsp     = /* BMPS 截断与波函数路径 */;
VMCPEPSOptimizerParams vmc_params{opt, mc, pepsp, "output"};

EnergySolv solver(/*ly,lx,模型参数...*/);
VMCPEPSOptimizer<TenElemT, QNT, Updater, EnergySolv>
  executor(vmc_params, sitps, MPI_COMM_WORLD, solver);
executor.Execute();
```

选择 MCSweepUpdater 的建议：
- 仅交换即可遍历的守恒模型：`MCUpdateSquareNNExchange`（必要时 `MCUpdateSquareTNN3SiteExchange` 提升接受率）
- 非守恒或交换无法遍历的模型：`MCUpdateSquareNNFullSpaceUpdate`

## 步骤 4：MCPEPSMeasurer 测量
- 入口：`qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.h`
- 参数：`MCMeasurementParams{ MonteCarloParams, PEPSParams, measurement_data_dump_path }`
- 模板策略：与优化阶段可共享 `MCSweepUpdater`/选择匹配的 `MeasurementSolver`
- 输出：能量及误差、键能量、单点/两点函数、自相关（同时转储 CSV 与二进制）

示例：
```cpp
using MeasUpdater = Updater; // 与优化阶段相同或按需替换
using MeasSolver  = /* 与能量一致的测量求解器 */;

MCMeasurementParams measp = /* 设置样本与 BMPS 截断 */;
MeasSolver ms;
MCPEPSMeasurer<TenElemT, QNT, MeasUpdater, MeasSolver>
  meas(sitps /*或从文件加载优化后态*/, measp, MPI_COMM_WORLD, ms);
meas.Execute();
```

## I/O 与路径组织
- `PEPSParams.wavefunction_path` 建议设置为与运行目录相对的稳定路径，便于跨阶段共享

## 常见陷阱（后三条是开发者文档内容）
- 初态幅度过大/过小导致数值不稳定：在转换后进行 `NormalizeAllSite()` 或 `ScaleMaxAbsForAllSite()`
- 扩展 Simple Update 时，指标顺序与演化门的一致性（参见 `TaylorExpMatrix` 与 SU 文档），费米子符号问题。
- 费米子系统的符号与奇偶操作：使用内置 `ActFermionPOps/CalGTenForFermionicTensors`
- 在优化器回调之外广播状态会破坏职责分离（不建议）

## 进一步阅读
- `VMCPEPS_OPTIMIZER_EXECUTOR_GUIDE.md`（执行器细节、组件选择）
- `OPTIMIZER_GUIDE.md`（优化算法与参数）
- `MODEL_ENERGY_SOLVER_GUIDE.md`（模型能量求解器）
- `TOP_LEVEL_APIs.md`（顶层执行器概览）

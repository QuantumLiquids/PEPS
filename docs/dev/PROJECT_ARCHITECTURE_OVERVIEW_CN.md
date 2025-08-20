### 项目架构总览（高密度 | 开发者导向）

本页给出 PEPS 项目核心模块的职责边界、调用关系、依赖层级与关键代码索引，帮助新开发者（含 AI/自动化工具）在最短时间内建立整体认知并准确落点到代码。

## 一、顶层入口与聚合

- 顶层聚合头：`include/qlpeps/qlpeps.h`
  - 仅定义命名空间并聚合 `algorithm_all.h`。
  - 注意：`algorithm_all.h` 会将 VMC、Simple/Loop Update 等全部对外暴露，属于“全量入口”。

- 算法聚合头：`include/qlpeps/algorithm/algorithm_all.h`
  - 聚合：Simple Update、Loop Update、VMC（优化器执行器、测量、能量评估器、模型 solver）。

建议：顶层聚合对外用，但在内部开发中，尽量只按需包含“具体组件头”以减少编译耦合与隐藏的循环包含风险（详见“依赖卫生”）。

## 二、模块边界与职责

- 核心数据结构（Tensor Network 层）
  - `two_dim_tn/framework/`: 基础容器与矩阵类（`ten_matrix.h`、`site_idx.h`、`duomatrix.h`）。
  - `two_dim_tn/tps/`: TPS 与 SplitIndexTPS 实现（`tps.h`、`split_index_tps.h`）。
    - 关键：`SplitIndexTPS<T>` 将物理指标拆分，面向 VMC 抽样与梯度计算。
  - `two_dim_tn/peps/`: PEPS 基类与方格格点实现（当前 `peps.h` 为空壳，可能需要扩展为PBC系或其他晶格形状。目前仅有OBC Square lattice PEPS）。

- Monte Carlo 基础设施（VMC 基础层）
  - `vmc_basic/`: 配置、统计与抽样工具（`configuration.h`、`statistics.h` 等）。
  - 向上被 VMC 执行器与能量评估器使用。

- 算法层
  - Simple Update：`algorithm/simple_update/`
    - 抽象执行器：`simple_update.h`（`SimpleUpdateExecutor<T>`）。
  - Loop Update：`algorithm/loop_update/`(暂时遗弃这一模块，算法效率不行)
  - VMC Update：`algorithm/vmc_update/`
    - 优化执行器：`vmc_peps_optimizer.h`（`VMCPEPSOptimizerExecutor<T>`）
    - 测量执行器：`monte_carlo_peps_measurement.h`（`MonteCarloMeasurementExecutor<T>`）
    - 能量评估器：`exact_summation_energy_evaluator.h`（精确求和，MPI 并行枚举）（非外部暴露接口，对测试十分有效，可消除蒙卡不确定性。）
    - 物理模型求解器（能量/观测）：`model_energy_solver.h` 与 `model_solvers/*`
    - 采样更新器（用户入参）：`MCSweepUpdater`（详见 VMCPEPSOptimizerExecutor 中文用户文档）

- 优化器层
  - 参数系统：`optimizer/optimizer_params.h`
    - 现代化参数：`OptimizerParams` + `AlgorithmParams(std::variant)`，支持 `SGD/AdaGrad/SR/Adam/L-BFGS` 等。
    - 学习率调度：`LearningRateScheduler`（`ConstantLR/ExponentialDecayLR/StepLR/PlateauLR`）
    - 工厂与 Builder：快速构建常用优化配置。
  - 优化器：`optimizer/optimizer.h` + `optimizer_impl.h`
    - 统一接口：`LineSearchOptimize` / `IterativeOptimize` / `CalculateNaturalGradient` 等。
    - 关键架构：MPI 责任分离（见下一节“关键调用链”）。本层刻意与执行器解耦，目的是更好地测试，同时避免庞大对象。但其直接服务对象仍是 `VMCPEPSOptimizerExecutor`。其测试可以通过精确能量评估器来测试，以消除蒙卡不确定性。
  - SR 矩阵：`optimizer/stochastic_reconfiguration_smatrix.h`

- 通用工具
  - `utility/helpers.h`：小型数学/张量辅助（含费米子梯度张量工具）。
  - `base/mpi_signal_guard.h`：紧急停止与 MPI 信号守护。

## 三、关键调用链（谁驱动谁）

- VMC 优化主流程（以 `VMCPEPSOptimizerExecutor` 为例）
  1. 用户通过 `VMCPEPSOptimizerExecutor` 传入：`OptimizerParams` + 初态 `SplitIndexTPS` + 物理模型 solver。
  2. 执行器内部持有 `Optimizer<T>`，并提供能量评估回调（默认或自定义）。
  3. 每次迭代：
     - Optimizer 在主进程更新状态（绝不在此处广播状态）。
     - 调用能量评估器（下方流程），由评估器负责广播状态，执行 MC 或枚举，回传梯度与能量。
     - 若算法为 SR，自身内部发起 MPI 协作的 CG 求解（只处理算法内矩阵-向量，非状态分发）。
     - Optimizer 根据梯度更新本地状态（主进程），记录轨迹/判断停止条件。
  4. 结束时由上层（执行器）保证最终态的同步与输出。

说明：项目大部分模块服务于 `VMCPEPSOptimizerExecutor` 及其测量器。`Simple Update` 常作为用户的第一步（生成初态），其结果可经转换用于 VMC 优化。此工作流细节请参考用户教程，而非本架构页。

- 精确求和能量评估（`ExactSumEnergyEvaluatorMPI`）
  1. 输入：主进程上的 `SplitIndexTPS`（来自 Optimizer）。
  2. 评估器广播态给所有进程（评估器“拥有状态分发权”）。
  3. 根据粒子数/自旋约束生成全部配置，并按 Round-Robin 分配到各进程。
  4. 每个进程独立计算局域能量与梯度样本的加权和。
  5. 归约：标量用 MPI_Reduce，梯度张量主进程收集并累加。
  6. 主进程计算最终能量与梯度；按契约，回调返回的能量与梯度仅在主进程有效。能量评估器可选择“额外广播能量”用于日志/调试，但调用方不应依赖该行为。

- Simple Update 执行（`SimpleUpdateExecutor`）
  - 以 `SquareLatticePEPS` 为目标态，内部维护演化门与扫掠，抽象 `Execute()` 与 `SimpleUpdateSweep_()`。

## 四、MPI 责任分离（架构要点）

- 原则：
  - Optimizer：只负责算法本身（如 SR 的 CG 并行），不负责“状态广播”。
  - Energy Evaluator：唯一“状态广播者”，对 Monte Carlo/精确枚举所需的态进行广播与统计归约。
  - Executor（VMC/Measurement）：负责最终状态一致性与数据落盘。

- 好处：
  - 将每一步的“多余三次广播”压缩到“一次”，广播成本下降约 67%。
  - 模块职责单一，避免特殊分支与重复通信。

- 回调契约（优化器视角）：
  - 能量与梯度仅在主进程有效；SR 的中间并行仅用于求解自然梯度，输出更新仍只在主进程生效。
  - 能量评估器负责“状态广播”；是否广播能量不在契约中（若实现广播，视为实现细节）。

参考实现位置：`optimizer/optimizer_impl.h`（类注释与实现中有清晰约束）与 `algorithm/vmc_update/exact_summation_energy_evaluator.h`（通信流程注释完整）。

## 五、依赖分层与“依赖卫生”评审

- 建议的分层（下层被上层依赖）：
  - Utility / Base（helpers、MPI 守护）
  - two_dim_tn（framework/tps/peps）
  - vmc_basic（配置/统计/抽样组件）
  - algorithm（simple/loop/vmc 及 model_solvers）
  - optimizer（算法策略与调度）
  - 顶层聚合（`qlpeps.h`）

- 发现的问题（需要“清洁化”）：
  - 循环包含风险：`exact_summation_energy_evaluator.h` 包含 `qlpeps/qlpeps.h`，而 `qlpeps.h` 又通过 `algorithm_all.h` 聚合回 `exact_summation_energy_evaluator.h`。
    - 虽然 include guard 可避免致命循环，但这是不必要耦合，建议：
      - 移除评估器对顶层聚合头的包含，改为最小必要头（`split_index_tps.h`、`model_energy_solver.h`、所需模型 solver、MPI 包装等）。
      - 保持“叶子模块→聚合头”的单向禁令：叶子模块不应包含顶层聚合。
  - 头文件 `using namespace qlten;`：出现在多个公共头中（例如 `simple_update.h`、`optimizer.h` 等）。
    - 建议移除头文件中的 `using namespace`，改为最小限定名或在 `.cpp/.impl.h` 内部使用，减少命名污染。
  - `optimizer_impl.h` 依赖 `vmc_basic/monte_carlo_tools/statistics.h`
    - 优化器本应与 MC 统计解耦；若仅为打印/统计辅助，应迁移到上层执行器或抽出更通用的 utility 组件。
  - `two_dim_tn/peps/peps.h` 是空壳：
    - 若为历史过渡，建议补齐注释说明或移除/合并，以避免“名存实亡”的 API 误导。

## 六、关键代码定位索引（按主题速查）

- 顶层 API 与聚合
  - `include/qlpeps/qlpeps.h`
  - `include/qlpeps/algorithm/algorithm_all.h`

- 数据结构
  - `two_dim_tn/tps/split_index_tps.h`（VMC 主态）
  - `two_dim_tn/tps/tps.h`
  - `two_dim_tn/framework/ten_matrix.h`、`site_idx.h`

- VMC 执行与测量
  - 优化执行器：`algorithm/vmc_update/vmc_peps_optimizer.h`（及 `vmc_peps_optimizer_impl.h`）
  - 测量执行器：`algorithm/vmc_update/monte_carlo_peps_measurement.h`
  - Monte Carlo 基类/参数：`monte_carlo_peps_base.h`、`monte_carlo_peps_params.h`

- 能量评估与模型
  - 精确求和评估器：`algorithm/vmc_update/exact_summation_energy_evaluator.h`
  - 通用模型能量接口：`algorithm/vmc_update/model_energy_solver.h`
  - 内置模型 solver：`algorithm/vmc_update/model_solvers/*`

- 优化器与参数
  - `optimizer/optimizer.h`、`optimizer_impl.h`
  - 现代参数系统：`optimizer/optimizer_params.h`
  - SR 矩阵/CG：`optimizer/stochastic_reconfiguration_smatrix.h`、`utility/conjugate_gradient_solver.h`

- 工具与基础
  - `utility/helpers.h`（费米子梯度构造等）
  - `base/mpi_signal_guard.h`

- 示例与测试（理解用法最直接）
  - `examples/`（整体流程示例）
  - `tests/test_algorithm/test_exact_sum_optimization.cpp`
  - `tests/test_algorithm/test_exact_summation_evaluator.cpp`
  - 其余 `tests/` 下集成/单元测试：查看如何拼装参数、模型与执行器。

## 七、开发建议（可操作）

- 依赖清洁化
  - 叶子模块（评估器/solver/具体算法实现）避免包含 `qlpeps.h`；只引入所需具体头。
  - 去除公共头文件的 `using namespace`。
  - 优化器层避免直接依赖 MC 统计；将“仅用于打印/统计”的功能上移到执行器或抽取至通用工具。

- API 一致性
  - 统一 MPI 包装接口命名与职责（广播/归约的所有权由评估器承担，优化器仅进行算法内部并行）。
  - 新增/修改接口时，明确“主进程有效/全进程有效”的返回约定（代码注释已采用此规范，新增处保持一致）。

- 文档化
  - 在 `two_dim_tn/peps/peps.h` 增加文件级注释，说明其状态（占位/过渡/计划）。

## 八、快速上手（开发者）

- 作为库使用：包含 `qlpeps/qlpeps.h`，按需选择 VMC 执行器或 Simple Update。
- VMC 优化最小骨架：
  - 构造 `SplitIndexTPS` 初态与模型 solver
  - 通过 `OptimizerFactory` 或 `OptimizerParamsBuilder` 构造优化器参数
  - 创建 `VMCPEPSOptimizerExecutor`，注入自定义能量评估器或采用默认
  - 调用 `Execute()`，读取 `GetEnergyTrajectory()`、`GetBestState()` 并落盘

---

若需要我对“循环包含清理方案”、“优化器与 MC 统计解耦”提供具体 edits，请告诉我当前对外 API 的兼容性要求（Never break userspace）与可接受的最小改动范围，我会给出最短路径的安全改造方案与对应提交。



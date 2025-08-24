# PEPS 执行器教程

本教程介绍库中暴露的三个高级执行器，并展示如何正确使用它们：

- Simple Update：用于有限方格晶格 PEPS 的虚时间投影
- VMC PEPS 优化器执行器：用于变分蒙特卡洛优化  
- 蒙特卡洛测量执行器：用于并行可观测量测量

这些类都是仅头文件的模板。请将模板参数替换为项目中使用的具体类型（张量元素类型和量子数类型/对称性）。

## 前置条件

- 已编译的 PEPS 项目（构建方法见顶级 README）
- 熟悉 `TPS`/`SplitIndexTPS` 和基本 MPI 使用
- 不确定时请包含便利的伞状头文件：`qlpeps/algorithm/algorithm_all.h`

---

## Simple Update（虚时间投影）

- 头文件：
  - `qlpeps/algorithm/simple_update/simple_update.h` （抽象基类）
  - `qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h` （方格晶格最近邻模型）
- 关键类：
  - `qlpeps::SimpleUpdateExecutor<TenElemT, QNT>` （抽象）
  - `qlpeps::SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>` （具体实现）

### 功能说明

在有限方格晶格 PEPS 上执行 Trotter 步骤，通过应用 exp(-τ H_bond) 门并在 `[Dmin, Dmax]` 范围内截断键维来进行局部截断。支持均匀最近邻项以及可选的均匀或位点相关的单点项。

### 重要 API

- 参数：`SimpleUpdatePara { size_t steps, double tau, size_t Dmin, size_t Dmax, double Trunc_err }`
- 构造函数（典型用法）：
  - 均匀单点项（可选）：
    - `SquareLatticeNNSimpleUpdateExecutor(const SimpleUpdatePara&, const PEPST&, const Tensor& ham_nn, const Tensor& ham_onsite = Tensor())`
  - 非均匀单点项：
    - `SquareLatticeNNSimpleUpdateExecutor(const SimpleUpdatePara&, const PEPST&, const Tensor& ham_nn, const TenMatrix<Tensor>& ham_onsite_terms)`
- 执行：
  - `void Execute()` 通过 `TaylorExpMatrix(tau, H_bond)` 构建演化门，并使用 `SquareLatticePEPS::NearestNeighborSiteProject` 扫描垂直和水平键。
- 工具函数：
  - `void ResetStepLenth(double tau)` 更新 τ 并在下次执行时重建门
  - `const PEPST& GetPEPS() const`
  - `bool DumpResult(std::string path, bool release_mem)`
  - `double GetEstimatedEnergy() const`

### 注意事项和正确性细节

- 门通过泰勒展开 exp(-τ H) 计算，使用与 `TaylorExpMatrix` 中实现的投影约定匹配的指标重排序。
- 对于包含单点项的模型（如 TFIM、带化学势的 t-J），键哈密顿量构造为
  `H_ij = H_two_site + h_i ⊗ I + I ⊗ h_j`，边界权重为 0.5/0.375/0.25，如实现中所示。
- 每次扫描打印的诊断信息包括中间键 λ 谱、从局部期望值和范数衰减估计的能量、中间键的截断误差，以及扫描计时。

### 最小使用示例

```cpp
#include "qlten/qlten.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"
using namespace qlten;
using namespace qlpeps;

using TenElemT = QLTEN_Double;     // or QLTEN_Complex
using QNT = qlten::special_qn::TrivialRepQN;  // Use TrivialRepQN for no symmetry (trivial quantum number)

using Tensor = QLTensor<TenElemT, QNT>;
using PEPST  = SquareLatticePEPS<TenElemT, QNT>;

// 构建哈密顿量片段（用户责任）
Tensor ham_nn = /* two-site operator with 4 legs (in-out per site) */;
Tensor ham_onsite = /* on-site operator with 2 legs (in-out) */; // optional

// 初始 PEPS
PEPST peps_init(/* ly, lx, bond_dim, phys_dim, ... */);

// 参数
SimpleUpdatePara su_para(/*steps=*/100, /*tau=*/1e-2, /*Dmin=*/8, /*Dmax=*/10, /*Trunc_err=*/1e-8);

// 执行器和运行
SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT> su(su_para, peps_init, ham_nn, ham_onsite);
su.Execute();

// 结果
double e_est = su.GetEstimatedEnergy();
su.DumpResult("output/peps_", /*release_mem=*/false);
```

---

## VMC PEPS 优化器（变分优化）

- 头文件：`qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h`
- 基类（采样核心）：`qlpeps::MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>`
- 执行器：`qlpeps::VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>`

### 功能说明

使用蒙特卡洛采样局部能量和梯度来优化 TPS/PEPS 波函数。支持线搜索方案和包括随机重构（SR）在内的迭代更新。采样是 MPI 并行的；状态更新在主进程上进行，然后广播到所有进程。

### 参数

使用统一的参数结构：

`VMCPEPSOptimizerParams { OptimizerParams optimizer_params; MonteCarloParams mc_params; PEPSParams peps_params; }`

- `OptimizerParams` 控制更新方案、步长和（对于 SR）共轭梯度参数。
- `MonteCarloParams` 控制样本数量、预热扫描、样本间扫描数和初始配置。
- `PEPSParams` 控制 BMPS 截断和 IO 路径（`wavefunction_path`）。

内置的蒙特卡洛扫描更新器和模型能量求解器位于：

- `qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h`
- `qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h`
- 精确求和（小系统）辅助器：`qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h`

### 执行流程（正确性关键行为）

1. 如需要，预热马尔可夫链，验证所有进程的配置。
2. 对于每个优化步骤，构建一个能量评估器：
   - 将当前 TPS 状态广播到所有进程，
   - 更新波函数组件以反映新的 TPS，
   - 标准化 TPS 使波函数振幅在各进程间为 O(1)，
   - 运行 `num_samples` 扫描，累积局部能量和每位点梯度张量，
   - 将统计数据（能量均值/误差、平均梯度，对于 SR 还有平均 g 张量）收集到主进程。
3. 主进程更新状态（线搜索或 SR），验证，广播，然后继续。
4. 转储当前和最佳 TPS、配置、样本能量和轨迹。

### 最小使用示例

```cpp
#include "mpi.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
using namespace qlten;
using namespace qlpeps;

using TenElemT = QLTEN_Double;
using QNT      = qlten::special_qn::TrivialRepQN;

using MonteCarloSweepUpdater = MCUpdateSquareNNExchange<TenElemT, QNT>; // example updater
using EnergySolver          = ExactSummationEnergyEvaluator;            // or a model-specific solver

// 适当填充参数
OptimizerParams opt_params;             // step_lengths, update_scheme, cg_params, ...
MonteCarloParams mc_params;             // num_samples, warmup, sweeps_between_samples, init config
PEPSParams peps_params;                 // BMPS truncate_para, wavefunction_path
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);

// 初始状态
SplitIndexTPS<TenElemT, QNT> tps_init(/* ly, lx */);

MPI_Comm comm = MPI_COMM_WORLD;
EnergySolver solver;

VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver> exe(params, tps_init, comm, solver);
exe.Execute();

// 访问结果
const auto &state_opt  = exe.GetOptimizedState();
const auto &state_best = exe.GetBestState();
double E_min           = exe.GetMinEnergy();
exe.DumpData("output/vmc_peps_", /*release_mem=*/false);
```

---

## 蒙特卡洛测量（可观测量）

- 头文件：`qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.h`
- 基类（采样核心）：`qlpeps::MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>`
- 执行器：`qlpeps::MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>`

### 功能说明

运行 MPI 并行蒙特卡洛采样来估计能量、键能量、单点和两点函数以及短时自相关，并提供误差棒。提供副本测试功能来探测遍历性。

### 参数和求解器

- `MCMeasurementParams` 结合了用于测量运行的 `MonteCarloParams` 和 `PEPSParams`。
- `MeasurementSolver` 函子在调用时必须返回 `ObservablesLocal<TenElemT>`：
  `measurement_solver_(&split_index_tps_, &tps_sample_)`。
  内置模型测量求解器位于
  `qlpeps/algorithm/vmc_update/model_measurement_solver.h` 和
  `qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h`。

### 执行流程和输出

- `Execute()` 预热（如需要），逐样本测量，然后在 MPI 进程间收集统计数据。
- 转储二进制和 CSV 输出，包括：
  - `energy_statistics` 和 `energy_statistics.csv`（能量和误差棒），
  - `bond_energys.csv`、`one_point_functions.csv`、`two_point_functions.csv`，
  - 每进程原始样本在 `energy_sample_data/`、`wave_function_amplitudes/`，
  - 单点/两点函数样本在 `one_point_function_samples/` 和 `two_point_function_samples/` 中的 CSV。
- 支持通过 `MPISignalGuard` 紧急停止以安全转储中间结果。

### 最小使用示例

```cpp
#include "mpi.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
using namespace qlten;
using namespace qlpeps;

using TenElemT = QLTEN_Double;
using QNT      = qlten::special_qn::TrivialRepQN;

using MonteCarloSweepUpdater = MCUpdateSquareNNExchange<TenElemT, QNT>;
using MeasurementSolver      = SomeModelMeasurementSolver; // choose a concrete one

MCMeasurementParams meas_para; // set mc_samples, warmup, sweeps_between_samples, BMPS truncate

size_t ly = /*...*/, lx = /*...*/;
MPI_Comm comm = MPI_COMM_WORLD;
MeasurementSolver solver;

// 从文件路径加载 TPS 或使用内存中的 TPS
SplitIndexTPS<TenElemT, QNT> tps(ly, lx);
tps.Load(meas_para.peps_params.wavefunction_path); // or load from your path

MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>
  meas(tps, meas_para, comm, solver);
meas.Execute();

auto [E, dE] = meas.OutputEnergy();
const auto &res = meas.GetMeasureResult(); // contains bond/one-/two-point functions and auto-correlations
```

---

## 选择执行器

- 使用 Simple Update 从合理的初始猜测快速获得局部/最近邻模型的投影 PEPS 状态。
- 使用 VMC 优化器通过严格的随机梯度和 SR 变分地精化 TPS/PEPS。
- 使用 MC 测量为给定的 TPS/PEPS 状态计算可观测量（带误差）。

## 常见陷阱

- 确保哈密顿量张量指标顺序与所需约定匹配（参见 `TaylorExpMatrix` 文档和示例）。
- 对于费米子系统，依赖内置的宇称操作和 `CalGTenForFermionicTensors`；不要手动重新应用符号。
- 设置 `wavefunction_path` 以便执行器能够一致地转储/加载 TPS 和配置。
- 在 MPI 运行中，避免进程相关的文件路径，除非在文档中说明（每进程原始样本和配置）。

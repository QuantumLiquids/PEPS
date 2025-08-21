## RFC: ExactSum Evaluator MPI — 从契约版假并行到高效真并行

- Status: Draft (low priority)
- Owner: @Hao-Xin Wang
- Related: `docs/dev/design/arch/mpi-contracts.md`, `include/qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h`, tests under `tests/test_algorithm/`

### 背景与动机
当前我们已提供“契约版假并行”的 MPI 评估器重载 `ExactSumEnergyEvaluatorMPI`：
- 仅 master 进程执行单进程实现；
- 广播输入态用于满足“谁用谁分发”的契约；
- 返回值只在 master 有效；
- 用于测试链路替换单进程 API，保证单/多进程测试一致性。

这满足测试需要，但不能发挥并行能力。本 RFC 讨论在不破坏现有 API 的前提下，演进为“高效真并行”：各 rank 分担配置枚举，主进程聚合标量与梯度，提升 ExactSum 的吞吐与可扩展性。

### 目标
- 在保持现有 MPI 重载签名不变的前提下，实现真实并行：
  - 广播输入态与必要元数据；
  - 静态切分 `all_configs`；
  - 各 rank 本地计算局部标量与梯度贡献；
  - 主进程聚合，输出与单进程一致的 `(energy, gradient, error)`；
- 单进程与多进程结果在数值上严格一致（浮点相容范围内）。

### 非目标
- SR 支持（暂不考虑）。
- 变更外部 Optimizer API（保持优化器以 master-only 逻辑使用评估结果）。
- 引入新的头文件（继续在现有头文件内实现）。

### 契约与兼容性（Never break userspace）
- 输入态仅在 master 有效；评估器内部负责广播（参见 `mpi-contracts.md`）。
- 返回值仅在 master 有效；非 master 返回占位（零形状梯度，标量未定义不使用）。
- `mpi_size==1` 时等价于单进程实现。
- 优先“正确与简单”，消除特殊情况而非堆叠分支。

### 设计方案（真并行）
- 广播输入态与元数据
  - 广播 `Ly, Lx, 物理维`、模型参数与截断参数。
  - 广播 `SplitIndexTPS`：请采用已有 `MPI_Bcast` (split_index_tps_impl.h)

- 静态切分配置集合 `all_configs`
  - 采用连续区间切分，范围定义（确定性、可重现）：
    - `start = total * rank / mpi_size`
    - `end   = total * (rank + 1) / mpi_size`
  - 对于空分片（end==start）直接产生零贡献。

- 本地计算（每个 rank）
  - 对局部配置子集，逐个计算：
    - `weight_sum = \sum w(cfg)`
    - `e_loc_weight_sum = \sum E_loc(cfg) * w(cfg)`
    - `O*_weighted_sum`（用于梯度）
    - `E_loc^* O*_weighted_sum`（梯度的能量修正项）
  - 内存控制：边算边累加，避免存储全量临时对象。

- 归约与聚合
   参考VMCPEPSOptimizerExecutor

- 结果计算（master）


### API 
- 继续沿用当前重载签名：
  - `ExactSumEnergyEvaluatorMPI<ModelT, TenElemT, QNT>(..., MPI_Comm comm, int rank, int mpi_size)`

### 可重复性与数值一致性
- 测试用例以单进程结果为金标准，比较能量与梯度（逐元素）误差阈值 `<= 1e-12`。

### 测试策略
- 复用现有：
  - `tests/test_algorithm/test_exact_summation_evaluator.cpp`：
    - 单/多进程运行；
    - 断言仅在 master 进行；
  - `tests/test_optimizer/test_optimizer_adagrad_exact_sum.cpp`：
    - 断言与日志限定 master；
    - 保证优化器对 master-only 输出的容忍与稳定。
- 新增（可选）：
  - 大规模配置枚举的多进程压力测试（仅在 CI 夜间/手动触发）。


### RFC: RNG 可复现性与种子管理改进

- title: RNG 可复现性与种子管理改进（include/ 范围）
- status: Draft
- last_updated: 2025-08-27
- tags: rng, reproducibility, mpi, vmc
- applies_to: vmc_update, optimizer, vmc_basic

### 背景与目标

当前随机数主要通过 `std::random_device` → `std::mt19937` 初始化，分布在：
- `Configuration::Random(...)`（函数局部、每次重播种）
- `MonteCarloSweepUpdaterBase`（成员 RNG）
- `NonDBMCMCStateUpdate`（调用方注入）

问题：
- 不可复现：缺少统一播种入口；平台差异导致 `random_device` 非一致。
- 可维护性差：对象复制会复制 RNG 状态；`% dim` 取模偏差。
- 语义分裂：`MonteCarloEngine` 内分布对象与引擎分离，易误导（建议精简或注释解释）。

目标：
- 在不破坏默认“随机即可用”的前提下，引入“显式可复现”的标准化播种方案与 API。

### 设计原则（Linus 式）

- 好品味：消除特殊情况；以“有显式种子→确定性；无→退回非确定性”为统一规则。
- Never break userspace：默认行为保持不变（未提供种子时继续使用 `random_device`）。
- 实用主义：仅引入最小 API 以满足实验/测试复现与调试需求。

### 提案（最小变更集）

1) 引入统一的显式种子入口（参数层）
- 在 `MonteCarloParams`、`OptimizerParams` 增加可选字段 `std::optional<uint64_t> seed`。
- 缺省 `nullopt` 表示维持旧行为（`random_device`）。

2) 派生规则（MPI 友好且稳定）
- 若提供 `seed`，则按 rank 派生子种子：
  - `seed_per_rank = mix64(seed) ^ (0x9E3779B97F4A7C15ULL * (uint64_t)rank)`
  - 其中 `mix64(x)` 可用 splitmix64 风格混合，或 `std::seed_seq` 混合 `{seed, rank}` 生成引擎状态。

3) 组件播种策略
- MonteCarloSweepUpdaterBase：构造时优先使用 `seed_per_rank`；否则 `random_device`。
- Configuration::Random(...)：
  - 方案 A（最小侵入）：保持现状；用于一次性初始化的“非复现随机”。
  - 方案 B（可选）：若调用方设置了“全局种子上下文”，则使用该上下文的引擎或经 `seed_seq` 派生的临时引擎。

1) 运行时接口
- 为 Updater 与 Optimizer 提供 `void SetSeed(uint64_t base_seed, int rank)`：在运行中重播种（测试/调试用途）。
- 保证与构造策略一致的派生规则。

1) 取模偏差修复
- 将 `Configuration::Random(dim)` 的 `rand_num_gen() % dim` 替换为 `std::uniform_int_distribution<size_t>(0, dim-1)`。

### 兼容性

- 未提供 `seed` 的用户：行为完全不变（继续 `random_device`）。
- 提供 `seed` 的用户：获得跨平台、跨 MPI 规模稳定的可复现随机序列（前提：同版本与同编译器标准库实现）。

### 实施步骤

1) 参数扩展：为 `MonteCarloParams`、`OptimizerParams` 增加可选 `seed` 字段，默认 `nullopt`。
2) 组件播种：在 `MonteCarloEngine` 构造 Updater、`Optimizer` 构造函数中按“显式种子优先”播种。
3) 可选：提供 `SetSeed` 方法；在 `ClearUp()` 文档中明确 RNG 不会自动重播种。
4) 修复 `Configuration::Random(dim)` 的取模偏差。
5) 文档与示例：在指南与测试中示范如何设置固定种子并验证复现。

### 测试计划

- 单测：
  - 同一 `seed`、同一 rank 序列一致；不同 rank 序列不同。
  - 未提供 `seed` 时结果分布性质不变（KS 检验/频数均匀性）。
- 集成测试：
  - 固定 `seed` 下端到端 MC 采样与优化步骤可复现（统计量/轨迹相等）。

### 风险与权衡

- 不同标准库/平台对 `std::seed_seq`、`mt19937` 序列兼容性：通常一致，但需在 CI 上做多平台样本校验。
- 复制含 RNG 的对象仍可能复制状态：在文档与评审中强调“避免复制”或复制后强制重播种。

### 里程碑与验收

- M1：参数层支持 + Updater/Optimizer 显式播种；通过单测。
- M2：端到端复现用例通过；文档完备。
- 验收标准：指定 `seed` 时，多次运行在相同环境下结果完全一致；未指定时行为不变。

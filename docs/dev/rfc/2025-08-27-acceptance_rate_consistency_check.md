## RFC: MPI 多节点 Monte Carlo 接受率一致性检查（Engine 下放）

Status: Draft

Author: H.-X. Wang (proposal consolidated)

### 背景与动机

当前在评估器侧存在基于“阈值相对最大值”的简单检查（如 rate < 0.5×global_max）。该方案依赖 magic number、对样本规模不敏感，且位置不当（Evaluator），无法复用到 MC Measurer。

目标：
- 在不破坏现有 API 行为的前提下，提供一个稳健、可配置、代价极低的跨节点一致性哨兵；
- 将检查逻辑下放到 `MonteCarloEngine`，供 Evaluator 与 Measurer 复用；
- 当计数不可用时保持与旧行为兼容（回退策略）。

### 统计学原理（LaTeX → Unicode 双表示）

记第 i 个 rank 在某一 update-type 上的“接受数/尝试数”为 (a_i, n_i)，接受率 p_i = a_i/n_i。

- 全局接受率估计：

  \[\hat p = \frac{\sum_i a_i}{\sum_i n_i}\]

  Unicode: p̂ = (Σ aᵢ) / (Σ nᵢ)

- 单 rank 正态近似 z 检验（大样本）：

  \[ z_i = \frac{p_i - \hat p}{\sqrt{\hat p(1-\hat p)/n_i}} \]

  Unicode: zᵢ = (pᵢ − p̂) / √( p̂(1−p̂)/nᵢ )

  当 |zᵢ| > z_threshold（如 4 或 5）时，标记该 rank 的该 update-type 异常。为避免 p̂≈0 或 1 的零方差问题，引入稳定项 ε：

  \[ \sigma_i = \sqrt{\max\{\hat p(1-\hat p),\,\varepsilon\}/n_i} \]

  Unicode: σᵢ = √( max{ p̂(1−p̂), ε } / nᵢ )

- 整体一致性（可选）卡方检验：

  \[ \chi^2 = \sum_i \frac{n_i\,(p_i-\hat p)^2}{\hat p(1-\hat p)} \sim \chi^2_{R-1} \]

  Unicode: χ² = Σ nᵢ (pᵢ−p̂)² / (p̂(1−p̂)) 近似 χ² 自由度 (R−1)

注意：当 nᵢ < n_min（最小样本数）时跳过 z 检验，以避免小样本误报。

### 设计概要

- 数据收集最小化：引入“计数”而非“率”。每 sweep 累加 per-update-type 的 `accepted` 与 `attempts`。
- 位置：实现于 `MonteCarloEngine` 内部（或紧耦合的内部组件），暴露只读接口与一次性诊断函数；Evaluator 与 Measurer 统一复用。
- 兼容性：若无计数（旧引擎或未启用），回退到现有“相对最大值”检查，不破坏用户空间。

### 拟新增数据结构与接口（示意）

```cpp
// Engine 内部：累积器（与 update-type 一一对应）
struct AcceptanceCounters {
  std::vector<uint64_t> accepted;  // per update-type
  std::vector<uint64_t> attempts;  // per update-type
  void Reset();
};

// Engine 对外（const 只读）
const AcceptanceCounters& GetAcceptanceCounters() const;

// Engine 工具：基于 MPI 统计生成诊断
struct AcceptanceDiagnostics {
  struct PerType {
    double p_hat;              // global rate estimate
    double min_rate, max_rate; // local rates
    int    min_rank, max_rank;
    double worst_abs_z;        // max_i |z_i|
    bool   flagged;            // 是否触发 z 检验阈值
  };
  std::vector<PerType> types;
};

AcceptanceDiagnostics ComputeAcceptanceDiagnostics(
    const AcceptanceCounters& local,
    MPI_Comm comm,
    double z_threshold,
    uint64_t min_samples,
    double variance_epsilon) const;
```

实现要点：
- MPI 只做 `SUM`（accepted/attempts），可选 `MINLOC/MAXLOC` 拿到 (rate, rank)。
- 复杂度 O(num_update_types)，内存与传输开销极小。
- 不改变现有采样流程；仅在 sweep 处累加计数。

### 参数与默认值（可配置）

- `z_threshold`：默认 5.0（极低误报概率，保守）。
- `min_samples`：默认 100 或由系统规模自适应；低于该值不做 z 检验。
- `variance_epsilon`：默认 1e-9，避免 p̂≈0/1 的零方差。

### 回退策略（保持兼容）

- 当 `attempts[type] == 0` 或计数容器未启用：
  - 返回 `flagged=false`，并可选执行旧有“rate vs global_max”检查以提示潜在问题；
  - 日志沿用既有格式，避免打断现有工作流。

### 日志与行为

- 每种 update-type 输出：`type, p_hat, min_rate(rank), max_rate(rank), worst|z|, flagged`。
- 当 `flagged==true`：
  - 默认仅警告（不终止），保持“Never break userspace”；
  - 允许通过配置将其升级为硬错误（调试/CI 模式）。

### 测试计划

- 单元测试（伪造计数）：
  - 正常情形：各 rank p_i≈p̂，|z| 全部小于阈值。
  - 异常情形：单 rank 人为降低/提高接受率，命中阈值且定位到该 rank。
  - 小样本情形：nᵢ < min_samples，不应误报。
- 端到端：在多进程环境下随机扰动单 rank 的提案分布，验证诊断一致。

### 迁移与落地步骤

1. 在 Engine 内部添加 `AcceptanceCounters` 并在 sweep 中累加（不改对外 API）。
2. 提供 `GetAcceptanceCounters()` 与 `ComputeAcceptanceDiagnostics(...)`；Measurer/Evaluator 改为调用该诊断。
3. 保留旧检查作为 fallback；为阈值常量提供配置项。
4. 分阶段启用更严格的动作（例如 CI 失败）。

### 复杂度与性能

- 计数累加为 O(1) 操作；
- MPI 约简是小向量（update-type 数目通常很小），开销可忽略；
- 无额外张量或样本数据的分发与收集。

### 风险与边界条件

- p̂→0 或 1：使用 `variance_epsilon` 稳定分母；必要时切换 Wilson 区间判定亦可拓展。
- 样本极不均衡：小样本 rank 自动跳过 z 检验，避免误报。
- update-type 动态变化：初始化时确定长度；或在首次 sweep 同步类型数。

### 结论

该方案以极低代价，引入统计学有效的一致性诊断；将逻辑下放到 Engine，统一 Evaluator/Measurer 复用，并在计数缺失时保持旧行为回退，不破坏用户空间。



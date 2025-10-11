---
title: Observable Registry and Results Organization for VMC/PEPS Measurements
date: 2025-09-11
status: draft
owners: [PEPS Core]
---

## Motivation
固定的 energy/one-point/two-point 分类限制扩展性和可用性：用户必须理解内部模型细节才能匹配到结果。我们提出“可注册的观测量注册表 (key+meta)”，统一组织样本与统计，跨模型一致，对用户透明。

## Core Abstractions
观测量元数据：
```c++
struct ObservableMeta {
  std::string key;                    // "energy", "spin_z", "charge", "bond_energy", "SzSz", "SC_dwave", ...
  std::string description;            // English, concise physical meaning
  std::vector<size_t> shape;          // runtime shape. Scalars use {}; lattice-aware entries use {Ly, Lx}, etc.
  std::vector<std::string> index_labels; // semantic tags, e.g., {"y","x"} or {"bond_y","bond_x"}
};
```

求值接口（由 `MeasurementSolver` 实现）：
- `DescribeObservables(size_t ly, size_t lx): std::vector<ObservableMeta>`
- `EvaluateSample(): std::unordered_map<std::string, std::vector<T>>`（扁平数组，长度为∏shape）

Psi 摘要（特殊通道，非注册表成员）：
在张量网络收缩的计算中，因为裁剪会引入裁剪误差，因而每个样本的波函数Amplitude Psi的具体数值是无法严格计算的。
在张量网络收缩中，我们会记录在不同行和不同列最终Trace的波函数Amplitude，这一计算是在观测量计算当中顺带计算的。因而不引入额外的计算量。我们把他存为psi_list. 我们可以通过这些不同的Psi的估计值来估计Psi的误差。

```c++
struct PsiSummary {
  std::complex<double> psi_mean;  // mean of psi_list
  double psi_rel_err;             // relative radius: max_i |psi_i - psi_mean| / |psi_mean|
};
```
- `EvaluatePsiSummary(): PsiSummary`（每个样本调用一次；不进入观测量注册表，不参与统计学分箱/误差估计）

`MCPEPSMeasurer` 负责：
- 缓冲：`key -> vector<flat_values>`（按采样时间堆叠）
- 统计：对每个 key 的每个分量运行 RFC《Binning+IPS SE》中的扫描，得 μ、SE、b*、τ_est
- 输出：按 key dump 统计（CSV/二进制）
- Psi 专属样本通道：逐样本收集 `PsiSummary` 并单独 dump（不走注册表，不做统计）

## User Experience
- 用户通过 key 直接查找：不需要了解 solver 内部分类。
- 可配置订阅的观测量集合，避免昂贵计算；solver 侧按 flag 构建元数据与求值。
- 统一的统计 dump 与文档：每个 key 有独立的 `stats/<key>.csv` 与可选 `stats/<key>_bin_scan.csv`。
- Psi 样本独立 dump：`samples/psi.csv`（逐样本），与任何观测量统计结果解耦。

## Compatibility
- 兼容层：保留 `res.energy` 等旧字段，从 `key="energy"` 等映射填充。
- 文档声明旧分类接口 deprecate，给出迁移指南（如何从 key 获取原先的一体/二体量）。
-. Psi 摘要不属于观测量注册表，不提供旧字段映射；仅保证新增的样本级输出存在且格式稳定。

## Dump Specification
样本：
- Psi 样本专属文件 `samples/psi.csv`（逐样本一行），列：`psi_mean_re,psi_mean_im,psi_rel_err` (real number 没有psi_mean_im)
- 暂不导出其他观测量的逐样本原始值（避免数据膨胀）。

统计：
- v1（当前实现）：`stats/<key>.csv`，列：`index,mean,stderr`
- v2（计划中）：添加 `chosen_b,tau_est,unstable`，并导出 `stats/<key>_bin_scan.csv`（见 SE RFC）
  - `index` 为扁平索引；若 `index_labels` 可构造多维索引，另存 `stats/<key>_index_map.txt` 说明

二进制（可选）：
- 写入魔数、版本、条目 meta（key、shape、类型）、数据尺寸、数据块。跨平台可解析。

## Implementation Plan (Phased)
0. 没有第三方用户，无需向后兼容。
1. 引入注册表抽象与 `MeasurementSolver` 接口；在 `MCPEPSMeasurer` 中实现缓冲与统计（并行复用现有 MPI 组件）。
2. 为现有能量与已有观测量注册 key（energy、bond_energy、spin_z、charge、SzSz 等）。
3. 引入 `PsiSummary` 专属接口与样本 dump 流水线；不进入注册表与统计。
4. 导出 CSV/二进制，完善 Doxygen 与开发者文档。

## Relation to RFC: SE via Binning+IPS
对每个 key 的每个分量单独进行分箱扫描与 τ 估计，保证跨观测量的一致统计学处理与可视化。

## Psi consistency handling
- `PsiSummary` 为样本级输出，不属于观测量注册表，不参与任何统计（不分箱、不估计 SE）。
- 仅存储：
  - `psi_mean`: 波函数振幅的样本均值（复标量）
  - `psi_rel_err`: 相对半径，定义为 \(\mathrm{radius\_rel} = \max_i |\psi_i - \overline{\psi}| / |\overline{\psi}|\)
- `psi_list` 为中间量，仅用于计算 `PsiSummary`，不落盘、不聚合。计算位置建议：
  - 在 `ModelMeasurementSolver` 基类提供受保护的通用工具函数以完成从 `psi_list` 到 `PsiSummary` 的转换；
  - 具体模型负责提供其 `psi_list` 的生成逻辑；
  - `MCPEPSMeasurer` 在每个样本结束时调用 `EvaluatePsiSummary()` 收集并写入 `samples/psi.csv`。

## Current Status (2025-10)

- All built-in solvers have been migrated to the registry interface. Keys are documented in
  `docs/user/en/guides/model_observables.md`.
- `MCPEPSMeasurer::Result` exposure has been reduced to the energy compatibility shim; all other
  consumers should query registries or CSV dumps directly.
- Recent fixes:
  - `TransverseFieldIsingSquare` now advertises `sigma_x` and `SzSz_row` via `DescribeObservables()`.
  - `SquareHubbardModel` exposes `double_occupancy` explicitly, matching the legacy dump.
- Remaining legacy discrepancies:
  - No raw `psi_list` in registry (by design).
  - Per-model gaps must be tracked as they surface; see below for planned tests.
  - Base metadata supplied by `SquareNNNModelMeasurementSolver` is intentionally minimal; each
    concrete solver must enrich the entries (shape/index labels) to match its lattice geometry。

## API Refactor Plan (No Backward Compatibility Required)

### Goals
- 彻底去除 “shape = {0,0} + 猜方向” 的隐式约定。
- 让 `DescribeObservables` 在被调用时就拿到格点尺寸，写入真实 shape。
- 统一 `index_labels` 语义，允许可选的第三轴标签标注方向/类型。
- 更新 `MCPEPSMeasurer` 以使用新的 metadata，删除现有的 fallback 猜测逻辑。

### Proposed Changes
1. **API 签名**：`DescribeObservables(size_t ly, size_t lx)`
   - 基类默认实现返回空向量。
   - 派生类必须使用传入尺寸填充真实 shape。
2. **Metadata 规范**
   - `shape` 必须与数据长度匹配。标量 `{}`；site 级 `{ly, lx}`；横向 bond `{ly, lx-1}`；纵向 `{ly-1, lx}`；对角 `{ly-1, lx-1}`。
  - `index_labels` 可空；若填写，应与 shape 轴一一对应（如 `{ "bond_y", "bond_x" }` 表示起点坐标）。
3. **Measurer 更新**
   - `MCPEPSMeasurer` 在构造时调用 `DescribeObservables(engine_.Ly(), engine_.Lx())`。
   - Dump 逻辑直接利用 `shape` 创建矩阵；shape 与数据不匹配时抛出异常。
   - 旧的尺寸猜测和静默 fallback 逻辑全部移除。
4. **模型迁移 Checklist**
   - 更新所有派生模型的 `DescribeObservables` 签名与实现，填入真实 shape/labels。
   - 清理遗留的 `"bond_id"` 等魔法字符串。

### Developer Guide 更新
- 在开发者文档新增 “扩展 Measurement Solver 的步骤”：说明新签名、shape/index_labels 写法及常见示例。

### Rollout Notes
- 无需兼容旧签名；编译失败会直接提醒开发者调整。
- 重构需与 `MCPEPSMeasurer` 改动同行提交，避免中间状态。

### Follow-up
- 重构完成后，更新本文档的 “Remaining legacy discrepancies” 段落。
- 增加自动化测试，断言 `DescribeObservables(ly,lx)` 的 shape 与实际数据吻合。

## Test Roadmap

1. **Registry contract tests**
   - Build a parameterised gtest suite that instantiates every built-in solver on a minimal 2×2
     `SplitIndexTPS` and asserts that each key declared in `DescribeObservables()` appears in the
     returned `ObservableMap`.
   - For models with conditional keys (e.g., superconducting order, NNN bonds), cover both enabled
     and disabled cases.

2. **Smoke tests with bundled TPS data**
   - Reuse sample states in `tests/slow_tests/test_data/` to run `MCPEPSMeasurer::Execute()` and
     verify that `stats/<key>.csv` exists for every advertised key.
   - Keep these tests under `RUN_SLOW_TESTS` to avoid extending the default CI time.

3. **Physics regression tests**
   - Where reference data exists (e.g., 4×4 Heisenberg, 2×2 transverse Ising), compare registry
     means against expected values within statistical tolerance.
   - For models lacking references, construct deterministic product states with analytic
     expectations for sanity checks.

4. **Golden data integration**
   - Prepare QuSpin (or similar) ED scripts for tiny lattices; store the resulting observables as
     JSON/CSV fixtures in `tests/resources/` and compare against registry outputs.
   - Plan follow-up integration with DMRG pipelines for larger systems when data is available.

5. **Automation skeleton**
   - Extend `tests/test_algorithm/test_mc_peps_measure.cpp` or add a new suite that parameterises
     over model classes, lattice sizes, and expected registry keys, reducing boilerplate.
   - Provide helpers to read registry metadata at runtime, so new keys automatically enter the
     assertions.



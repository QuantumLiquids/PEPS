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
  std::vector<size_t> shape;          // e.g., {Ly, Lx} or {Nbonds} or {Ly,Lx,Ly,Lx}
  bool is_complex;                    // data type
  std::vector<std::string> index_labels; // e.g., {"y","x"} or {"bond_id"}
};
```

求值接口（由 `MeasurementSolver` 实现）：
- `DescribeObservables(): std::vector<ObservableMeta>`
- `EvaluateSample(): std::unordered_map<std::string, std::vector<T>>`（扁平数组，长度为∏shape）

`MCPEPSMeasurer` 负责：
- 缓冲：`key -> vector<flat_values>`（按采样时间堆叠）
- 统计：对每个 key 的每个分量运行 RFC《Binning+IPS SE》中的扫描，得 μ、SE、b*、τ_est
- 输出：按 key dump 样本与统计（CSV/二进制）

## User Experience
- 用户通过 key 直接查找：不需要了解 solver 内部分类。
- 可配置订阅的观测量集合，避免昂贵计算；solver 侧按 flag 构建元数据与求值。
- 统一的 dump 与文档：每个 key 有独立的 `stats/<key>.csv` 与可选 `stats/<key>_bin_scan.csv`。

## Compatibility
- 兼容层：保留 `res.energy` 等旧字段，从 `key="energy"` 等映射填充。
- 文档声明旧分类接口 deprecate，给出迁移指南（如何从 key 获取原先的一体/二体量）。

## Dump Specification
样本（可选）：
- `samples/<key>/rank_<r>/sample_<r>.csv`（header 含 shape、复数列名 re,im）

统计：
- `stats/<key>.csv`，列：`index,mean,stderr,chosen_b,tau_est,unstable`
  - `index` 为扁平索引；若 `index_labels` 可构造多维索引，另存 `stats/<key>_index_map.txt` 说明
- `stats/<key>_bin_scan.csv`（见 SE RFC）

二进制（可选）：
- 写入魔数、版本、条目 meta（key、shape、类型）、数据尺寸、数据块。跨平台可解析。

## Implementation Plan (Phased)
1. 引入注册表抽象与 `MeasurementSolver` 接口；在 `MCPEPSMeasurer` 中实现缓冲与统计（并行复用现有 MPI 组件）。
2. 为现有能量与已有观测量注册 key（energy、bond_energy、spin_z、charge、SzSz 等），保证旧字段可从新结果派生。
3. 导出 CSV/二进制，完善 Doxygen 与开发者文档。
4. 标注旧分类接口为 deprecated；观测到稳定后移除。

## Relation to RFC: SE via Binning+IPS
对每个 key 的每个分量单独进行分箱扫描与 τ 估计，保证跨观测量的一致统计学处理与可视化。



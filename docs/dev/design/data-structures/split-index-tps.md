---
title: SplitIndexTPS 数据结构
---

## 概览

`SplitIndexTPS<TenElemT, QNT>` 表示将物理指标拆分后的二维的张量网络乘积态（TPS）。每个格点保存一个长度为 `phy_dim` 的张量向量 `std::vector<QLTensor<...>>`，其中第 i 个分量对应物理局域希尔伯特空间的第 i 个基态投影后的张量。该表示法便于 VMC 采样与基于配置的投影收缩。

实现位置：`include/qlpeps/two_dim_tn/tps/split_index_tps.h` 与 `.../split_index_tps_impl.h`。

## 索引约定与玻色/费米差异

- 玻色：站点张量通常只有 4 条虚指数，物理指数被“拆分”掉，因此分量张量收缩时使用索引集 {0,1,2,3}。
- 费米：站点张量包含额外的 1 维奇偶索引（放在最后），用于保证配分的一致性。内积或张量收缩时使用 {0,1,2,3,4} 并在必要处调用 `ActFermionPOps()`。
- 从常规 TPS 转换为拆分格式时：
  - 玻色：使用 Kronecker 投影（对物理索引打 delta，逐分量投影）。
  - 费米：按量子数扇区匹配投影，确保量子数守恒与奇偶一致。

物理索引位置：在非拆分的常规 TPS 中，物理索引约定为位置 4。由 `SplitIndexTPS(const TPS&)` 及 `GroupIndices()` 的实现可见。

## 主要 API（简要）

- 构造与转换：
  - `SplitIndexTPS(rows, cols)`：创建空矩阵；元素为默认态（未分配张量）。
  - `SplitIndexTPS(rows, cols, phy_dim)`：为每个站点创建长度为 `phy_dim` 的分量向量。
  - `SplitIndexTPS(const TPS&)`：从常规 TPS 生成拆分表示（按物理分量逐个投影）。
  - `TPST GroupIndices(const Index<QNT>& phy_idx) const`：将拆分的分量合并回常规 TPS。费米使用 `Expand(..., {4})`；玻色使用带物理腿的投影张量并求和。

- 配置投影：
  - `TensorNetwork2D Project(const Configuration&) const`：根据组态在每个格点选择一个分量，形成 2D 张量网络。

- 代数与内积：
  - `operator+(...)`, `operator-`, `operator*(scalar)` 与就地版本。
  - `TenElemT operator*(const SplitIndexTPS&) const`：内积。费米情形先对 `Dag(T)` 调用 `ActFermionPOps()`，再以 {0,1,2,3,4} 收缩；玻色以 {0,1,2,3} 收缩。

- 归一化与缩放：
  - `double NormSquare() const`
    - 先给出 LaTeX 公式：
      \[\sum_{r,c,i} \|T_{r,c}^{(i)}\|_{2,\mathrm{quasi}}^2\]
    - Unicode 公式：`sum_{r,c,i} ||T_{r,c}^{(i)}||_{2,quasi}^2`
    - 说明：使用“拟 2 范数”（quasi-2-norm），对费米张量总是良定；区别于可能失效的分级 2 范数（graded 2-norm）。
  - `double NormalizeSite(const SiteIdx&)`
    - LaTeX：\[\sum_{i=0}^{d-1} \|T_{r,c}^{(i)}\|_{2,\mathrm{quasi}}^2 = 1\]
    - Unicode：`sum_{i=0..d-1} ||T_{r,c}^{(i)}||_{2,quasi}^2 = 1`
    - 返回站点的归一化因子。
  - `void NormalizeAllSite()`：逐站点调用上式。
  - `double ScaleMaxAbsForSite(const SiteIdx&, double aiming_max_abs)`：将站点内所有分量按最大绝对值缩放到目标值；返回缩放因子的倒数。
  - `void ScaleMaxAbsForAllSite(double)`：逐站点调用上式。

- 尺寸与一致性：
  - `GetMinBondDimension()`, `GetMaxBondDimension()`, `GetMinMaxBondDimension()`
  - `GetAllInnerBondDimensions()`：扫描内部键尺寸。
  - `IsBondDimensionEven()`：检查内部键尺寸是否在体区一致。

- I/O：
  - `void Dump(const std::string& tps_path, bool release_mem=false)`：
    - Tensor 文件：`kTpsTenBaseName + "row_col_compt" + "." + kQLTenFileSuffix`
    - 元数据：`tps_meta.txt`，内容为 `rows cols phy_dim`
  - `bool Load(const std::string& tps_path)`：兼容新旧两种元数据格式（新：`tps_meta.txt`；旧：`phys_dim`）。
  - 也提供 `DumpTen/LoadTen` 操作单个分量。

- MPI：
  - `MPI_Send(const SplitIndexTPS&, int dest, const MPI_Comm&, int tag=0)`：先发送尺寸，再逐分量发送张量。
  - `MPI_Recv(SplitIndexTPS&, int src, const MPI_Comm&, int tag=0)`：先接收尺寸并初始化，再逐分量接收。
  - `MPI_Bcast(SplitIndexTPS&, const MPI_Comm&, int root=0)`：先广播尺寸，再广播每个张量。

以上接口名称遵循 MPI 标准命名（无多余下划线）。

## 关键实现要点与注意事项

- 默认态处理：站点分量张量可能为默认态（未分配），遍历使用 `ForEachValidTensor_()` 自动跳过默认态以节省内存/计算。
- 费米处理：内积和某些操作需要先对 `Dag(T)` 调用 `ActFermionPOps()` 以保证奇偶算符序的正确性，再进行 5 索引收缩。
- 数值稳定性：费米张量的归一化统一使用拟 2 范数，避免分级 2 范数在奇数块占优时的非正定问题。

## 最小示例

```cpp
using SplitTPS = qlpeps::SplitIndexTPS<QLTEN_Double, qlten::special_qn::U1QN>;
SplitTPS s(4, 4, 2);
s.NormalizeAllSite();
auto ns = s.NormSquare();
```

从常规 TPS 转换：

```cpp
using TPS = qlpeps::TPS<QLTEN_Double, qlten::special_qn::U1QN>;
TPS t(4, 4);
// ... 初始化 t ...
SplitTPS s_from_t(t);
```

合并回常规 TPS：

```cpp
auto phy = t({0,0}).GetIndex(4); // 物理索引
TPS recovered = s_from_t.GroupIndices(phy);
```

## 与其他组件的关系

- `TensorNetwork2D<TenElemT, QNT>`：`Project(Configuration)` 生成对应的 2D 网络以进行 BMPS 收缩。
- `TPS<TenElemT, QNT>`：提供互相转换（拆分/合并）。

## 约束与假设

- 常规 TPS 的物理索引约定在位置 4。
- 费米张量需要遵循约定的站点顺序与奇偶索引顺序；调用者在进行高级收缩或定制 trace 时应保证与默认约定一致。



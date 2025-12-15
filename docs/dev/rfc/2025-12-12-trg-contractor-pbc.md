---
title: Finite-size TRG Contractor for PBC TensorNetwork2D
status: draft
last_updated: 2025-12-15
applies_to: [module/two_dim_tn, module/tensor_network_2d, module/vmc_update]
tags: [design, rfc, pbc, trg, caching, incremental-update]
---

# RFC：为 PBC 的 `TensorNetwork2D` 引入 Finite-size Navy–Levin TRG Contractor

类型：设计提案 (RFC)

## 0. 需求理解确认（先把话说清楚）
基于现有信息，我理解你的需求是：
- 我们已经把 `TensorNetwork2D` 的 OBC 收缩逻辑从数据容器里拆出去，形成了 `BMPSContractor`，并且 **VMC 更新路径依赖** `InvalidateEnvs + Replace*Trace (+ PunchHole)` 语义。
- 现在要为 **PBC** 实现一个新的 contractor：`TRGContractor`，用 **finite-size** 版本的 **Navy–Levin TRG**（每次 RG：格点数减半，线尺度乘 \(1/\sqrt{2}\)，并伴随 45° 旋转）。
- TRG 必须支持：
  1) 非平移不变：每个 tensor 都可能不同，但仍要保留 AB 子格（决定分解方向/截断）；
  2) 缓存每个 scale 的网络信息，供上层 VMC 重复利用；
  3) 初期可先只支持 \(n\times n\), \(n=2^m\)；
  4) 提供类似 OBC 的 ReplaceTrace：能够判断“替换一个局域 tensor”在多尺度上影响哪些 coarse tensor，并做增量更新；
  5) Hole/punch-hole 类功能先不实现，但要预留设计钩子。

请确认以上理解是否准确；如果你希望 RG 的“格点减半”具体规则与我理解不同（例如你用的不是 checkerboard plaquette coarse-graining），必须先对齐，否则实现会跑偏。

## 0.1 实施进度更新（2025-12）
已落地的内容（代码已合入工作区并通过单测）：
- `TRGContractor` 已创建：`include/qlpeps/two_dim_tn/tensor_network_2d/trg_contractor.h/.cpp (header-only impl)`
- **Finite-size bosonic `Trace()` 已实现**：checkerboard plaquette TRG（even→odd→even…直到最后的 even 2×2），并且保证 even-scale leg order 始终为 `[L,D,R,U]`，避免奇偶步的腿约定漂移。
- **终结收缩改为 2×2 精确收缩**：不再做最后一次 2×2→1×1 的 TRG coarse-graining（避免多一次 SVD/截断；也避免额外特殊情况）。
- **显式截断参数**：`TRGContractor` 不再持有“隐藏默认截断参数”，调用方必须 `SetTruncateParams()`（否则 `Trace()` 抛异常）。
- **可读性**：在 TRG 的关键业务步骤（SVD split、plaquette contraction、diamond contraction、final 2×2 contraction / 1×1 trace helper）旁边添加了 ASCII 张量网络图注释。
- **对照测试**：
  - `tests/test_2d_tn/test_trg_contractor.cpp`：使用 Python transfer-matrix 生成的严格参考值，对比 TRG 输出。
  - 覆盖：
    - 4×4 torus、Z2 对称、均匀耦合（对比 \(Z\)）
    - 4×4 torus、Z2 对称、non-uniform 耦合（对比 \(Z\)）
    - 8×8 torus、Z2 对称、non-uniform 耦合（对比 \(\log Z\)，避免数值爆炸）
- **2×2 PunchHole 终结器（base case）已实现**：
  - `TRGContractor::PunchHole(tn, site)` 当前支持 **2×2 PBC torus**，通过精确收缩 3 个 tensor 得到 rank-4 hole 张量。
  - 单测 `PunchHole2x2U1Random` 覆盖了 U1 block-sparse 且 bond 维度不一致的情形，并验证
    \(\langle \mathrm{hole}_i, T_i\rangle = Z\)（对四个 site 全部成立）。

未完成/明确推迟的内容：
- TRG 的 **增量更新**（`InvalidateEnvs(site)` 目前仅记录 dirty seed，没有影响域传播与局部重算）。
- `Replace*Trace`（局域替换比值）尚未实现。
- general `PunchHole`（任意尺寸/递归向下的 hole environment）尚未实现（当前只有 2×2 terminator）。
- 与 `TPSWaveFunctionComponent` 的 contractor 选择/适配尚未进行（当前测试直接构造 `TensorNetwork2D`）。

## 1. Linus 的三个问题（先问再做）
1) 这是个真问题还是臆想？——是 **真问题**：PBC 不可能靠 BMPS 那套 OBC 环境硬塞进去，TRG 是合理路径，且 VMC 需要缓存/增量更新，不是“学术洁癖”。
2) 有更简单的方法吗？——短期最简是“每次 ReplaceTrace 全量重建 TRG”，但 VMC 会慢到不可用。必须设计 **缓存 + 影响域传播**，否则就是自欺欺人。
3) 会破坏什么吗？——最大风险是破坏现有 VMC 用户代码。必须 **不改现有 `BMPSContractor` 的调用方式**；TRG 以并行扩展方式接入。

## 2. 核心判断
✅ 值得做：这是 PBC 的必经之路，并且必须以“数据结构优先”的方式设计，否则会变成 `if (bc==PBC)` 的垃圾分支地狱。

## 3. 关键洞察（数据结构先行）
> “Bad programmers worry about the code. Good programmers worry about data structures.”

TRG 的麻烦点（非平移不变、45° 旋转、影响域扩散）本质上都是 **索引/邻接关系** 的问题，不是张量 SVD 的问题。

因此本 RFC 的核心是：把每个 scale 的网络表示为一个 **“4-正则图 + 每个点的 leg 标号”**（或等价的解析坐标系统），让“旋转/奇偶步”不再是特殊情况，而是同一个数据结构的不同实例。

## 4. 范围与不做的事
### 范围（本 RFC 要求）
- 引入 `TRGContractor`（与 `BMPSContractor` 并列），用于 `BoundaryCondition::Periodic` 的 finite-size contraction。
- 提供与现有 VMC 更新路径兼容的最小 API（见 §6），至少覆盖：
  - `Init(tn)`
  - `Trace(...)`（兼容签名）
  - `ReplaceOneSiteTrace / ReplaceNNSiteTrace / ReplaceTNNSiteTrace ...`（先支持常用的 1/2/3-site）
  - `InvalidateEnvs(site)`（增量更新的入口）
- 缓存多尺度网络，并实现“局域替换→影响域传播→局部重算 coarse tensors”的增量更新机制。

### 不做（明确排除）
- 暂不实现 general `PunchHole`/hole environment（仅预留扩展点；目前只支持 2×2 terminator）。
- 暂不支持任意尺寸；初期限定 \(n\times n, n=2^m\)。
- 暂不追求无限系统（iTRG）；本 RFC 针对 finite-size。

> 更新：2×2 的 PunchHole 终结器已落地（见 §0.1），但 general PunchHole 仍未实现。

## 5. 约束与数学定义（finite-size Navy–Levin TRG）
### 5.1 尺寸约束
- 初版仅支持：`see tn.rows()==tn.cols()==n` 且 `n` 是 \(2^m\)。
- 若未来要扩展到矩形或非 2 的幂，必须先明确 coarse-graining 的边界处理规则（否则 debug 是灾难）。

### 5.2 每一步 RG 的规则（与“格点数减半”一致）
采用 checkerboard plaquette coarse-graining：
- 在 scale \(s\) 的网络上，选择一个子格（黑格 plaquette），每个 plaquette 涉及 4 个 rank-4 tensors（或等价的 4 个“分解后片段”）。
- 每个 tensor 先按 AB 子格规则做一次分解（SVD/截断），得到两块 rank-3 tensor 放到 plaquette 边上。
- 每个黑格 plaquette 收缩 4 个 rank-3，形成一个新的 rank-4 coarse tensor。
- 因为只取一半的 plaquettes，coarse tensor 个数变为原来的 \(1/2\)，线尺度乘 \(1/\sqrt{2}\)，并出现 45° 旋转的嵌入变化。

这正对应你给的例子：\(8\times 8\)（64 tensors）→ 32 tensors → 16 tensors（可嵌入为 \(4\times 4\)）。

## 6. 目标 API（柔性调用，不绑死 BMPS 的“脑回路”）
### 6.1 现状：`TPSWaveFunctionComponent` 把 contractor 写死成 BMPS（问题）
`TPSWaveFunctionComponent` 当前成员固定为：
- `TensorNetwork2D tn;`
- `BMPSContractor contractor;`
并在 `UpdateLocal` 时调用 `contractor.InvalidateEnvs(site)`。

这意味着如果我们想支持 PBC(TRG)，不能要求用户改掉所有 solver/updater。否则就是破坏 userspace。

### 6.2 提案：以“concept/adapter”方式柔性接入（默认仍走 BMPS）
TRG 与 BMPS 的收缩流程与缓存形态完全不同，**不应该为了“对齐”而伪造一堆同名接口**。正确做法是：
- 上层（VMC/solver）依赖一个**最小语义接口**；
- 具体 contractor（BMPS/TRG/未来 CTMRG）各自实现自己的内部流程；
- 需要兼容旧调用点时，用 **adapter** 做“薄转换”，避免在业务代码里散落 `if (bc)`。

最小侵入的实现路径是：给 `TPSWaveFunctionComponent` 引入默认模板参数，使旧代码零改动：

```cpp
// PSEUDO CODE (design intent)
template<typename TenElemT, typename QNT, typename Dress = NoDress,
         template<class, class> class ContractorT = BMPSContractor>
struct TPSWaveFunctionComponent {
  TensorNetwork2D<TenElemT, QNT> tn;
  ContractorT<TenElemT, QNT> contractor;
  // ...
};
```

- 旧路径：`TPSWaveFunctionComponent<..., BMPSContractor>`（默认）。
- 新路径：`TPSWaveFunctionComponent<..., TRGContractor>`（PBC）。

### 6.3 `TRGContractor` 的接口：允许更“钉死”的 Trace
TRG 下的 `Trace` 通常就是“给定整个网络 → 返回标量 amplitude”的钉死流程，因此 **建议 TRG 的核心接口**是：
- `TenElemT Trace(const TensorNetwork2D&) const;`

但为了不破坏现有调用点（目前大量写的是 `contractor.Trace(tn, site, orient)`），我们建议用 **adapter** 提供兼容签名：
- `Trace(tn, SiteIdx, BondOrientation)` 内部直接转发到 `Trace(tn)`（忽略后两者）。

### 6.4 最小语义接口（contractor concept，文档约束）
上层应当依赖“语义”而不是 BMPS 的具体过程名。建议把最小接口定义成（仅文档约束，先不强制 concept 编译）：
- `Init(const TensorNetwork2D&)`
- `Trace(const TensorNetwork2D&)`
- `InvalidateEnvs(SiteIdx)`（允许实现为 no-op；TRG 用它驱动 dirty 传播）
- 可选：`ReplaceOneSiteTrace / ReplaceNNSiteTrace / ReplaceTNNSiteTrace ...`

说明：
- 对 TRG 来说，`Replace*Trace` 可以先只做 1-site/2-site（VMC 最常用），其余逐步补齐。
- 对 BMPS 来说，现有的 `Replace*Trace` 是“面向过程的收缩管线”，TRG 不需要照搬同名/同参，只要能通过 adapter 让上层拿到同样的“比值评估能力”即可。

## 7. 数据结构设计（消除 45° 旋转这个特殊情况）
### 7.1 最小可行的表示：每个 scale 一个“图”
定义 scale \(s\) 的网络为：
- `nodes`: \(N_s\) 个张量（\(N_{s+1}=N_s/2\) 直到终止）
- 每个 node 有 4 条外腿（legs 0..3），并且每条腿连到另一个 node 的某条腿（PBC 下无边界）

核心是 **邻接关系**，不是几何坐标。

建议的数据结构（仅说明，不是最终代码）：

```cpp
struct TrgNeighbor {
  uint32_t node;
  uint8_t  leg;   // 0..3
};

struct TrgGraph {
  std::vector<std::array<TrgNeighbor, 4>> nbr; // size = N
  std::vector<uint8_t> sublattice;             // 0 = A, 1 = B
  std::vector<uint8_t> orientation;            // 0/1: which leg-pair is "x" vs "y"
  // invariant: nbr[n][l] points back to (n,l)
};
```

这样：
- odd-step 的“旋转 45°”不需要特殊处理：只要 graph 正确，收缩/更新都一样。
- AB 子格与“分解方向”只依赖 `sublattice/orientation`，不会散落成 if/else。

### 7.2 为什么不直接用 (row,col) 表示 rotated lattice？
你已经预感到了：odd 次 RG 后的格子“转了 45°”。如果硬用 `rows/cols` 来解释它，会产生大量特殊情况（越修越烂）。

用 graph 表示，旋转只是“邻接表不同”，数学上更干净。

> 如果你强烈希望保留解析坐标（性能/缓存局部性），也可以在 graph 上额外存一个 embedding（比如对角基 `u=(x+y)/2`, `v=(x-y)/2` 的模运算）。但那是优化，不是第一版必需品。

## 8. 缓存设计（每个 scale 的信息必须可复用）
### 8.1 缓存内容（建议）
每个 scale \(s\) 缓存：
- `graph_s`: 该 scale 的邻接与 AB/orientation
- `ten_s[i]`: 该 scale 的张量集合
- `local_decomp_s[i]`（可选，但强烈建议）：`i` 点分解得到的 rank-3 片段/等距映射（用于增量更新减少重复 SVD）
- `norm_s`（建议）：数值稳定的归一化因子（避免溢出/下溢）

整体缓存：
- `scales_`: vector<ScaleCache>
- `dirty_frontier_`: 每个 scale 的“脏节点集合”，用于增量更新传播

### 8.2 收敛终止
当 \(N_s\ remind\) 足够小时（例如 1、2 或 4 个张量），用直接收缩得到标量 \(Z\)。

## 9. 增量更新与 ReplaceTrace（真正的难点）
### 9.1 语义对齐（必须）
OBC/BMPS 的语义是：
- 更新某个 site tensor 后调用 `InvalidateEnvs(site)`，环境缓存被截断/失效；
- 后续 `Replace*Trace` 或 `Trace` 会使用“重建后的缓存”。

PBC/TRG 必须提供同样的操作模型，否则上层 VMC/updater 都要重写。

### 9.2 TRG 的影响域扩散（你给的“1→2→3”）
在 checkerboard TRG 中，一个 fine tensor 参与 **两个** coarse plaquette 的构造，因此：
- scale \(s\) 改一个 node → scale \(s+1\) 至少影响 2 个 coarse nodes；
- 继续上去影响域像光锥一样扩散（1→2→3→...）。

### 9.3 关键数据：fine-to-coarse / coarse-to-fine 映射
为了增量更新必须预计算并缓存：
- `fine_to_coarse_s[fine_node] -> array<coarse_node, 2>`
- `coarse_to_fine_{s+1}[coarse_node] -> array<fine_node, 4>`

有了它，增量更新就变成确定性的图传播：
1. 用户更新 TN 的 site tensor；
2. `InvalidateEnvs(site)` 把 scale0 的对应 node 标记为 dirty；
3. 对每个 scale s：
   - 把 dirty fine nodes 映射成 dirty coarse nodes（通常 2 倍扩张）；
   - 仅重算这些 coarse nodes 的张量（需要其对应的 4 个 fine nodes；必要时也重算周围分解片段）；
   - 产生下一层 dirty 集合，继续传播；
4. 最终更新终止层的标量 \(Z\)。

这就是“动态实现”，并且它完全由映射决定，没有隐含的几何特判。

### 9.4 ReplaceTrace 的实现策略（分阶段，避免一口吞掉大象）
#### 阶段 A（先要正确）：全量重建
`ReplaceOneSiteTrace(...)` 的最朴素实现是：
- 在 scale0 构造一份临时 tensor 列表（只替换局域 tensor）；
- 从 scale0 开始一路 TRG 到终止，得到标量。

这是 O(\(N\log N\))，会慢，但方便作为金标准测试。

#### 阶段 B（可用）：增量更新 + 复用多尺度缓存
把 ReplaceTrace 变成：
- 保存原缓存；
- 在 scale0 只替换局域 tensor，并以该局域 node 为 dirty 起点执行增量传播；
- 得到新 \(Z'\)；
- 恢复缓存（或用“小对象/RAII”做临时覆盖）。

这能把单次 proposal 的成本压到“光锥大小 × 每层局部 SVD/contract 的成本”，通常远小于全量。

> 注意：缓存恢复策略要小心，不要做大量深拷贝。建议用“版本号 + 覆盖记录”或 “copy-on-write”。

## 10. Hole / PunchHole（先别做，但必须预留）
BMPS 的 `PunchHole` 是“给定 site，返回一个 rank-4 环境张量”，用于构造局域观测/梯度项。

TRG 下的 hole 更复杂，因为你需要同时 coarse-grain “带 impurity 的网络”，并在每层保存/应用粗粒化等距映射。

本 RFC 的最低要求：
- `TRGContractor` 中预留接口（可以先 `throw` 或 `assert(false)`）：
  - `Tensor PunchHole(...) const;`
- 缓存结构里保留未来可扩展字段：`local_isometry_s`（每点分解得到的等距张量）。

### 10.1 近期折中计划：2×2 terminator + 递归向上
我们现在采用的路线是把 PunchHole 拆成两层：
- **终结器（已实现）**：当规模到达 even 2×2 时，用精确收缩得到 4 个 hole（或按 site 逐个算 hole）。
- **递归/迭代层（待实现）**：对更大系统，通过保存每层 coarse-graining 的等距映射（isometry）把 hole 从 coarse 层“往下推回去”，直到 scale-0。

这能把“深水区”风险隔离开：先用 2×2 把 leg-order / index-direction / 共轭等约定钉死，再逐层推广。

## 11. 数值稳定性（可选项：先保持 amplitude 语义）
你们当前 VMC 需要的是 **amplitude（也就是收缩得到的复/实标量 \(Z\)）**，而不是经典统计那套 `log_norm_` 叙事。

数值稳定性确实可能成为问题，但不必在第一版强行引入复杂的标度分离。建议分阶段：
- 阶段 1（默认）：直接返回 `TenElemT Z`，不引入 `log_norm_`。
- 阶段 2（可选，保持接口干净）：在 contractor 内部做轻量规范化（例如每步除以一个实标度因子），同时累积一个 `RealT log_scale`，最终返回 `Z = Z_scaled * exp(log_scale)`。

关键要求：**对上层暴露的依然是 amplitude**，数值稳定机制是实现细节或可选扩展，不要污染 solver/VMC 逻辑。

## 12. 对 `BMPSContractor` 现有接口的批评与改良建议（不破坏 userspace）
`BMPSContractor` 现在能跑，但接口层面有几个明显的坏味道（会拖累未来的 TRG/CTMRG 并行接入）：
- **过程名泄漏实现细节**：`GrowFullBTen/InitBTen/BTen2MoveStep/...` 这些名字把 BMPS 的内部步骤暴露给上层，导致调用方必须“懂 BMPS 才会用”，这是设计失败。
- **`Trace` 参数语义混乱**：`Trace(tn, site, bond_dir)` 对 OBC 来说像是在“选一个局域 bond”，但返回的是全局收缩标量；这让 TRG 这种“全局一次性收缩”实现显得别扭。
- **Replace*Trace 家族过多且耦合调用顺序**：上层经常要先 Grow 环境再 Replace，这种“必须先做 A 再做 B”是典型的易错 API。

不破坏现有代码的改良路线（建议，不强制）：
1. **新增一层更干净的 facade（不动旧接口）**：例如 `ContractorFacade` 暴露 `Trace(tn) / ReplaceLocal(tn, updates)`，内部对 BMPS 做必要的 Grow/Init。
2. **把 “局域替换”统一成一个数据结构**：例如 `LocalUpdate{site, replaced_tensor}` 或 `std::span<LocalUpdate>`，避免 `ReplaceNNNSiteTrace/ReplaceSqrt5...` 这种爆炸式命名。
3. **旧接口保留但标记为“BMPS 专用管线 API”**：文档上明确“仅供性能敏感路径直接操控环境”，其余调用走 facade。

## 13. 测试计划（先不强制“严格解”，但要有金标准路径）
### 13.1 正确性金标准（TRG internal cross-check）
- 小系统（比如 \(2\times 2\), \(4\times 4\)）：
  - 直接暴力收缩（无截断或 chi 足够大） vs TRG 结果一致；
  - ReplaceOneSite/ReplaceNNSite 的结果与“替换后全量重建”的 TRG 一致。

### 13.2 增量更新一致性
- 构造随机 TN（或来自 PEPS projection）；
- 计算 \(Z\)；
- 替换一个 site tensor：
  - 路径 1：全量重建得到 \(Z'\)
  - 路径 2：增量更新得到 \(Z''\)
  - 断言 \(Z'\approx Z''\)

### 13.3 影响域大小回归
- 统计单点更新触发的 dirty node 数在各 scale 的增长是否符合预期（1→2→3…），防止映射 bug 导致“漏更新”。

## 14. 风险与权衡
- **实现复杂度**：TRG 本身不复杂，复杂的是“索引与缓存一致性”。所以必须先把 graph/mapping 写对。
- **性能**：如果 ReplaceTrace 走全量重建，VMC 会死。必须尽快落地阶段 B。
- **API 漂移**：如果 TRG contractor 的接口和 BMPS 不一致，上层会出现 `if (bc)` 分支爆炸。必须用默认模板参数确保旧代码不变，新代码可选。

## 15. 结论
- 这是一个值得做且必须做的结构性扩展。
- 成败取决于数据结构：用 “scale graph + 映射 + dirty 传播” 消除 45°/奇偶步等特殊情况。
- 建议按“正确性（金标准）→ 增量更新 → 预留 hole”三段推进，确保不破坏现有 userspace，同时让 PBC 真正可用。



---
title: Unified Contractor Contract (BMPS/TRG) and API Layering
status: draft
last_updated: 2025-12-23
applies_to: [module/two_dim_tn, module/tensor_network_2d, module/vmc_update]
tags: [design, rfc, contractor, caching, performance, compatibility]
---

## 0. 需求理解确认（把“契约”写死，别靠口头约定）

我们要解决的是真问题：`TRGContractor`（PBC）已经重构得比较干净，但在改动上下层接口约定时，把平级的 `BMPSContractor`（OBC）以及上层 VMC/solver 的调用语义搞乱了。

症状不是“某处加个 if 就好”，而是**缺乏统一的、可执行的接口契约**：

- 上层同时存在两条更新语义：
  - **Trial/Commit**：`BeginTrialWithReplacement()` → `CommitTrial()`
  - **Direct-Update + Invalidate**：外部直接改 `TensorNetwork2D` 后调用 `InvalidateEnvs(site)`
- 现有 BMPS 还存在第三类“专家 API”（手动挡）：`GrowBTenStep/GrowFullBTen/BTenMoveStep/...`
- TRG 是“全局 coarse-graining + 多尺度 cache”，BMPS 是“边界环境 + 游标式推进”；强行把它们塞成同一套过程名会产生分支地狱。

本 RFC 的目标是：在**不破坏现有可跑代码**（Never break userspace）的前提下，定义一个清晰的层次化 API，让 BMPS/TRG 既能在上层统一使用，又能保留 BMPS 的手动高性能路径。

## 0.2 变更规则（本 RFC 允许改，但不允许瞎改）

本 RFC 是 **living document**：实现过程中可以、也应该根据事实修正设计。但必须遵守以下规则，否则就是在制造混乱。

### 规则 A：允许修改的触发条件（必须是“真问题”）

只有在出现以下任一情况时，才允许修改本 RFC：

- 代码按 RFC 实现后出现**语义矛盾/脚枪**（例如 TN 与 cache 一致性无法保证）
- 实现被证明**不可落地**（例如必须引入大量 `if` 特判、或出现不可控的复杂度爆炸）
- 出现**性能回退**且有证据（profiling/benchmark/可复现用例）
- 新增/变更需求（例如外部 userspace 约束）导致原设计不再成立

### 规则 B：修改 RFC 时必须写清楚（必须带证据与影响面）

每一次修改必须在文档里新增一段 “Revision / Rationale”，至少包含：

- **旧假设**：原本的设计前提是什么
- **新事实**：是什么证据推翻了旧假设（link 到测试/benchmark/issue/讨论记录）
- **新结论**：现在要怎么改，具体改哪条契约/接口
- **兼容性影响**：会破坏什么吗？如何做到 *Never break userspace*
- **迁移计划**：调用方怎么从旧行为迁移到新行为（必要时提供 deprecation 路线）

### 规则 C：禁止 silent drift（代码与 RFC 不一致必须修）

当实现与 RFC 不一致时，必须二选一：

- 改代码以匹配 RFC；或
- 改 RFC 并按规则 A/B 写清楚原因与影响。

不允许“代码先跑起来，文档以后再说”的长期悬置状态。

## 1. Linus 的三个问题

1) 这是个真问题吗？——是。当前“统一接口”并没有统一语义，只是统一了名字，导致调用方踩雷。

2) 有更简单的方法吗？——最简单且正确的方法是**把接口分层**：通用层只承诺少量语义；BMPS 手动挡保留，但不再冒充“通用接口”。

3) 会破坏什么吗？——最大的破坏来自“重命名/删 API”。本 RFC 明确：**旧接口保留**，通过 adapter/facade 提供新契约，逐步迁移。

## 2. 核心判断

✅ 值得做：这是 PBC(TRG) 与 OBC(BMPS) 并存的前提，也是防止未来 CTMRG/其他 contractor 再次把代码搞烂的唯一办法。

## 3. 关键洞察（数据结构优先）

> “Bad programmers worry about the code. Good programmers worry about data structures.”

BMPS/TRG 的差异不是“函数名不同”，而是**状态的形态不同**：

- **BMPS**：状态是 *一组边界 MPS + 边界张量*，并且大量操作是“移动游标/增长一层”。
- **TRG**：状态是 *多尺度图 + 每层 tensor cache + fine↔coarse 映射*，更新是“影响域光锥传播”。

所以正确的统一方式是：统一**语义**（contract），而不是统一**过程步骤**（Grow/Move）。

## 4. 分层 API（必须明确什么是“通用”，什么是“BMPS 专家”）

### 4.1 Layer 0：数据容器（已完成：split-tensornetwork2d）

`TensorNetwork2D` 只负责存 tensor 与边界条件，不存算法环境。

### 4.2 Layer 1：通用收缩契约（Contraction Contract）

通用层只提供 VMC 必需的最小能力：**给定一个 TN，评估 amplitude；对局域替换做 trial，并能 commit**。

**关键设计原则：Contractor 不拥有 TensorNetwork2D**

Contractor 是"纯计算"组件，不持有 TN 的所有权。所有读取或修改 TN 的操作都需要显式传入 `TensorNetwork2D` 引用：

```cpp
// Contractor 不持有 TN，所有操作都需要显式传入
TenElemT Trace(const TensorNetwork2D& tn) const;
void CommitTrial(TensorNetwork2D& tn, Trial&&);
//              ^^^^^^^^^^^^^^^^^^^^^^^^
//              Explicit TN reference required
```

这样设计的原因：
1. Contractor 可以被多个 TN 复用（例如测量不同的 trial configurations）
2. Contractor 可以脱离 `WaveFunctionComponent` 独立使用（例如测量代码）
3. 避免 split-tensornetwork2d 之前的 God-class 设计（TN 和算法耦合）

**Layer 1 接口**（文档约束；不要求立即用 C++20 concept 强制）：

- `void Init(const TensorNetwork2D&)`
- `TenElemT Trace(const TensorNetwork2D&) const`
- `TenElemT EvaluateReplacement(const TensorNetwork2D&, std::span<const LocalReplacement>) const`
- `Trial BeginTrialWithReplacement(std::span<const LocalReplacement>) const`
- `void CommitTrial(TensorNetwork2D& tn, Trial&&)`
- `void ClearCache()`

其中 `LocalReplacement` 是唯一的"局域更新数据结构"，避免 ReplaceNN/NNN/... 的函数名爆炸：

```cpp
// English comments only.
struct LocalReplacement {
  SiteIdx site;
  Tensor  tensor;
};
```

**硬约束（必须写入契约）**：

- `EvaluateReplacement()` **只读接口**：评估替换后的振幅，但**不修改**内部 cache，也**不保存**任何状态用于后续 commit。
  - **使用场景**：能量计算、测量（例如 $\langle O_i O_j \rangle$ 的单次查询）
  - **性能特征**：contractor 可以使用临时缓存，计算完即丢弃（不需要保存 Trial 状态）
- `BeginTrialWithReplacement()` **带状态接口**：评估替换后的振幅，并**保存内部状态**以便后续 `CommitTrial` 使用。
  - **使用场景**：VMC 更新（需要 accept-reject 决策，接受后需要 commit）
  - **性能特征**：contractor 必须保存环境快照/增量传播 seed（BMPS 需要保存 BTen 状态，TRG 需要保存 coarse-grain seed）
  - **语义约束**：**不得修改**内部持久缓存（只生成 trial token）
  - **生命周期安全**：`Trial` 对象必须**拥有** replacement tensors 的副本（深拷贝或移动）。
    - 原因：`BeginTrial` 传入的 `LocalReplacement` 可能引用临时变量（如 `op * tensor`），如果 `Trial` 只存引用，在 `CommitTrial` 时会发生 Use-After-Free。
  - **一致性安全（Version Check）**：`Trial` 必须记录 TN 的版本号/状态快照。
    - 在 Debug 模式下，`CommitTrial` 必须校验 TN 当前版本与生成 Trial 时的版本一致，防止"版本漂移"（即生成 Trial 后 TN 被修改，然后又 Commit 旧 Trial）。
  - **注意**：contractor 不拥有 TN，因此 Trial 对象必须保存 replacement tensors（而不是 site+config），
    以便 `CommitTrial` 能够写回 TN
- `CommitTrial(tn, trial)` **唯一的 commit 接口**：
  - 必须**同时**：(1) 把 trial 的 replacement tensors 写回 `tn`；(2) 更新 contractor 内部 cache
  - 必须**保证**：后续 `Trace(tn)` 返回与 `trial.amplitude` 一致的值
  - **显式传入 TN**：因为 contractor 不拥有 TN，调用方必须显式传入（通常是 `WaveFunctionComponent` 内部调用）
  - **不提供** `CommitTrial(trial)` 版本（那是脚枪：只更新 cache 不更新 TN，会导致 cache-TN 不一致）

**设计理由：为什么需要两个接口？**

- 测量场景（`EvaluateReplacement`）：只要振幅值，不需要后续 commit，不应付出"保存 Trial 状态"的成本
- 更新场景（`BeginTrial` + `CommitTrial`）：需要在 accept-reject 决策之间保持状态一致性

两者的实现可以共享核心计算逻辑（例如 BMPS 的 `BeginTrial` 可以内部调用 `EvaluateReplacement`），但语义和使用场景不同。

### 4.2.1 Higher Layer：WaveFunctionComponent 的封装

`WaveFunctionComponent` 是"状态管理"组件，拥有 TN + contractor + config 的所有权，
并负责维护三者的一致性。它提供更高层的接口，隐藏 TN 的显式传递：

```cpp
template<typename TenElemT, typename QNT>
class TPSWaveFunctionComponent {
public:
  TensorNetwork2D<TenElemT, QNT> tn;        // Owned
  Contractor<TenElemT, QNT> contractor;      // Owned
  Configuration config;                      // Owned
  const SplitIndexTPS<TenElemT, QNT>* sitps_;  // External reference
  
  // High-level interface: encapsulates TN + contractor + config
  Trial BeginTrialWithReplacement(std::span<const LocalReplacementWithConfig> reps) {
    // 1. Project config to tensors (using sitps_)
    std::vector<LocalReplacement> tensor_reps;
    for (const auto& rep : reps) {
      tensor_reps.push_back({rep.site, (*sitps_)(rep.site)[rep.new_config]});
    }
    
    // 2. Call contractor Layer 1 API (passing internal tn)
    auto trial = contractor.BeginTrialWithReplacement(tensor_reps);
    
    // 3. Save config changes for later commit
    trial.config_changes = reps;
    
    return trial;
  }
  
  void CommitTrial(Trial&& trial) {
    // 1. Update TN + contractor cache (delegate to Layer 1)
    contractor.CommitTrial(tn, std::move(trial));
    //                     ^^
    //                     使用内部的 tn，不需要调用方传入
    
    // 2. Update config to match TN
    for (const auto& change : trial.config_changes) {
      config(change.site) = change.new_config;
    }
    
    // Now: tn, contractor cache, and config are all synchronized
  }
};

// Helper struct for WaveFunctionComponent
struct LocalReplacementWithConfig {
  SiteIdx site;
  size_t new_config;  // config index (not tensor)
};
```

**关键差异**：

| | Layer 1 (Contractor) | Higher Layer (WaveFunctionComponent) |
|---|---|---|
| **拥有 TN** | ❌ 不拥有 | ✅ 拥有 |
| **接口参数** | 需要显式传入 TN | 不需要传入 TN |
| **输入类型** | `LocalReplacement` (tensor) | `LocalReplacementWithConfig` (config index) |
| **同步责任** | 只同步 TN + cache | 同步 TN + cache + config |
| **独立性** | 可独立使用（测量） | 依赖 contractor |

**上层 VMC 代码示例**：

```cpp
// VMC update loop
auto trial = tps_sample->BeginTrialWithReplacement({
  {site1, new_config1}, {site2, new_config2}
});

if (AcceptProposal(trial.amplitude, old_amplitude)) {
  tps_sample->CommitTrial(std::move(trial));
  // 一行搞定：TN、contractor cache、config 全部同步
}
```

**为什么这样分层？**

1. **Contractor（Layer 1）**：纯计算组件，可以脱离 `WaveFunctionComponent` 独立使用
   - 测量代码可能只需要 contractor + TN，不需要完整的 WaveFunctionComponent
   - Contractor 可以被多个 TN 复用（例如测量不同的 trial configurations）
   - 保持接口简单（只处理 tensor 级别的操作）

2. **WaveFunctionComponent（Higher layer）**：状态管理组件，封装一致性
   - VMC 主路径通过 WaveFunctionComponent 操作，自动保证 TN/cache/config 同步
   - 隐藏 TN 的显式传递，简化上层代码
   - 处理 config→tensor 的投影逻辑

### 4.3 Layer 2：局域观测/洞环境（Observable / Hole Contract）

`PunchHole` 是"高层需求"，但不是所有 contractor 都应该在第一版支持。把它放在独立层：

- `Tensor PunchHole(...) const`

如果某个 solver 需要 hole，就声明它依赖 Layer 2；不满足的 contractor 不参与该 solver。

### 4.5 Layer 4：高性能 Correlation 测量（BMPS 专家场景）

**问题陈述**：为什么通用接口无法满足需求？

对于需要计算大量 correlation 的测量（例如 $\langle S^+_i S^-_j \rangle$ 对所有 $(i,j)$ 组合），
Layer 1 的 `EvaluateReplacement` 接口虽然语义清晰，但无法达到 BMPS 的最优性能。

**根本原因**：BMPS 的最优性能来自"调用方显式控制遍历顺序"。

**示例：同一行的 correlation 测量**

```cpp
// Generic path (works for any contractor, but slower for BMPS)
for (size_t j = 0; j < lx; j++) {
  corr[j] = contractor.EvaluateReplacement(tn, {
    {site_i, op_i}, {SiteIdx{row, j}, op_j}
  });
}
// 每次调用都会重新计算边界环境 → O(lx * lx * D^6)
```

```cpp
// BMPS expert path (optimal)
// 1. 临时修改 site_i
tn.UpdateSiteTensor(site_i, flipped_tensor_i, sitps);
contractor.InvalidateEnvs(site_i);

// 2. 准备环境到 site_i 右侧
contractor.GrowBTenStep(tn, LEFT);
contractor.GrowFullBTen(tn, RIGHT, row, start_col + 2, false);

// 3. 循环：只移动游标，复用边界 MPS
for (size_t j = 0; j < lx; j++) {
  corr[j] = contractor.ReplaceOneSiteTrace(tn, {row, j}, op_j, HORIZONTAL);
  contractor.BTenMoveStep(tn, RIGHT);  // O(D^6) 增量更新
}

// 4. 恢复
tn.UpdateSiteTensor(site_i, original_tensor_i, sitps);
contractor.InvalidateEnvs(site_i);
// 总复杂度：O(lx * D^6)，比通用路径快 lx 倍
```

**性能对比**：

| 测量类型 | 通用接口复杂度 | BMPS 专家路径复杂度 | 加速比 |
|---------|--------------|-------------------|-------|
| 同一行 correlation (lx 个点) | $O(L_x^2 D^6)$ | $O(L_x D^6)$ | $L_x^2$ |
| Structure Factor (全格点) | $O(N^3 D^6)$ | $O(N^2 D^6)$ | $N$ |

**解决方案：保留 BMPS 手动挡作为 Layer 4**

对于这类性能关键的测量场景，允许使用 Layer 3 的 BMPS 专家 API，但必须：

1. **明确文档说明**：这是 BMPS/OBC 专用，不适用于 TRG/CTMRG
2. **提供 fallback**：要么提供通用实现（慢但正确），要么编译期拒绝非 BMPS contractor
3. **隔离影响范围**：把专家路径封装在独立函数中，清晰标记使用了 Layer 3/4 API
4. **基于特性的检测（Trait-based Detection）**：不要硬编码 `is_same_v<BMPSContractor>`，而是检测能力。

**代码模式示例**：

```cpp
// measurement_correlation.h
template<typename Contractor, typename TenElemT, typename QNT>
std::vector<TenElemT> MeasureCorrelationInRow(
    Contractor& contractor,
    TensorNetwork2D<TenElemT, QNT>& tn,
    SiteIdx site_i, /* ... */
) {
  // Good Taste: Check for capability, not specific class
  if constexpr (Contractor::supports_cursor_invalidation) {
    // Fast path: BMPS expert API (Layer 4)
    return MeasureCorrelationInRowExpert(contractor, tn, site_i, /* ... */);
  } else {
    // Slow path or throw
    return MeasureCorrelationInRowGeneric(contractor, tn, site_i, /* ... */);
  }
}

// Expert implementation (Layer 4)
template<typename Contractor, typename TenElemT, typename QNT>
std::vector<TenElemT> MeasureCorrelationInRowExpert(
    Contractor& contractor,  // Use template type, don't hardcode BMPSContractor
    TensorNetwork2D<TenElemT, QNT>& tn,
    SiteIdx site_i, /* ... */
) {
  // Use Layer 3 API: GrowBTenStep, BTenMoveStep, InvalidateEnvs, ...
  // (see example above)
}
```

**为什么用 Trait/Capability 而不是 `is_same_v`？**
- **解耦**：如果未来有新的 Contractor（例如 `NewBMPSContractor`）也支持游标操作，只需要声明 `supports_cursor_invalidation = true` 即可，不需要修改所有测量代码。
- **清晰**：代码表达的是“我需要游标能力”，而不是“我需要这个具体的类”。

**关于 Structure Factor**：

Structure Factor 需要 $O(N^2)$ 次查询，即使用 BMPS 专家路径也需要 $O(N^2 D^3)$。
但通过"按行遍历 + 复用边界环境"，可以将系数降低到最小。

TRG 可能更适合 Structure Factor 测量：
- TRG 的多尺度缓存可能自然支持任意位置查询
- 不需要手动控制遍历顺序
- 建议在 TRG 实现成熟后，对比 BMPS 专家路径和 TRG 通用路径的性能

**迁移策略**：

- 短期：保留现有的 BMPS 专家测量代码（如 `MeasureSpinOneHalfOffDiagOrderInRow`），标记为 Layer 4
- 中期：将专家路径封装到独立函数，提供 `if constexpr` 分发
- 长期：评估 TRG 的 Structure Factor 性能，可能统一到 Layer 1

### 4.4 Layer 3：BMPS 专家 API（Manual Workspace）

像 `GrowBTenStep/GrowFullBTen/BTenMoveStep/InvalidateEnvs/...` 这种 API 是**手动挡**，必须明确：

- 它们是 *BMPS 专用*，不属于 Layer 1；
- 调用方必须理解 BMPS 的内部机制（游标位置、环境长度、MPS 方向等），否则就是用户错误；
- VMC 通用路径不应依赖这些 API。

这层可以继续存在以保证性能，在需要手动控制环境的场景（例如高性能 correlation 测量，见 §4.5）使用。

#### 4.4.1 实现策略：命名空间标记，不拆分类

**关键原则：只有一个 `BMPSContractor` 类，所有方法都在这个类里。**

不创建新类（例如 `BMPSContractorExpert`），因为：
- BMPS 的内部状态（`bmps_set_`, `bten_set_`）是唯一的，拆成两个类会导致状态重复/同步问题
- 大量现有代码直接使用 `BMPSContractor`，拆分会制造巨大迁移成本

**推荐的标记方式**（二选一）：

**方案 A：Doxygen 分组（最简单，推荐短期使用）**

```cpp
template<typename TenElemT, typename QNT>
class BMPSContractor {
public:
  /// @name General Contraction Contract (Layer 1)
  /// Compatible with TRG/CTMRG and other contractors
  /// @{
  TenElemT EvaluateReplacement(...) const;
  Trial BeginTrialWithReplacement(...) const;
  void CommitTrial(TensorNetwork2D&, Trial&&);
  /// @}
  
  /// @name BMPS Expert API (Layer 3 - OBC only)
  /// @warning Requires deep understanding of BMPS internals.
  ///          Not compatible with TRG/CTMRG.
  /// @{
  void GenerateBMPSApproach(...);
  void GrowBTenStep(...);
  void BTenMoveStep(...);
  void InvalidateEnvs(const SiteIdx&);
  
  [[deprecated("Use EvaluateReplacement")]]
  TenElemT ReplaceOneSiteTrace(...) const;
  [[deprecated("Use EvaluateReplacement")]]
  TenElemT ReplaceNNSiteTrace(...) const;
  /// @}
  
private:
  BMPSSet bmps_set_;
  BTensorSet bten_set_;
  // ... (唯一一份内部状态)
};
```

**方案 B：命名空间重导出（可选，用于更清晰的调用点）**

创建 `bmps_contractor_expert.h`（可选的单独头文件）：

```cpp
namespace qlpeps::bmps::expert {

// These are "forwarding functions" that call BMPSContractor methods.
// They serve as documentation: "expert::" prefix tells readers this is expert API.

template<typename TenElemT, typename QNT>
void PrepareRowEnvironment(BMPSContractor<TenElemT,QNT>& contractor,
                          const TensorNetwork2D<TenElemT,QNT>& tn,
                          size_t row, size_t start_col) {
  contractor.InitBTen(tn, LEFT, row);
  contractor.GrowFullBTen(tn, RIGHT, row, start_col + 1, true);
}

template<typename TenElemT, typename QNT>
void InvalidateEnvs(BMPSContractor<TenElemT,QNT>& contractor, const SiteIdx& site) {
  contractor.InvalidateEnvs(site);
}

// ... other forwarding functions

} // namespace qlpeps::bmps::expert
```

**调用示例**：

```cpp
// 方案 A：直接调用（现有代码不需要改）
contractor.GrowBTenStep(tn, LEFT);
contractor.InvalidateEnvs(site);

// 方案 B：通过命名空间（可选，让调用点更清晰）
using namespace qlpeps::bmps::expert;
PrepareRowEnvironment(contractor, tn, row, start_col);
InvalidateEnvs(contractor, site);
```

**零成本**：编译器会内联命名空间中的转发函数，运行时无额外开销。

**迁移要求**：
- 短期（1 个月）：使用方案 A（Doxygen 分组），现有代码无需修改
- 中期（3-6 个月）：可选实现方案 B，逐步迁移高性能测量代码
- 长期：VMC 主路径完全不依赖 Layer 3

## 5. 缓存与“谁拥有数据”（必须写死，否则会反复踩坑）

统一原则：**`TensorNetwork2D` 是 source-of-truth；contractor 的 cache 是派生数据**。

### 5.1 两条更新语义必须兼容

#### (A) Trial/Commit 语义（推荐）

**两层接口，各司其职**：

**Layer 1 (Contractor)**：
```cpp
auto trial = contractor.BeginTrialWithReplacement({{site, tensor}});
if (accept) {
  contractor.CommitTrial(tn, std::move(trial));
  //                     ^^
  //                     必须显式传入 TN（contractor 不拥有它）
}
```

**Higher Layer (WaveFunctionComponent)**：
```cpp
auto trial = tps_sample->BeginTrialWithReplacement({{site, config_index}});
if (accept) {
  tps_sample->CommitTrial(std::move(trial));
  //                      ^^
  //                      不需要传入 TN（内部已经封装）
}
```

**Contractor 的 `CommitTrial(tn, trial)` 接口设计**

- **语义**：同时 (1) 把 trial 的 replacement tensors 写回 `tn`；(2) 更新 contractor 内部 cache
- **保证**：后续 `Trace(tn)` 与 `trial.amplitude` 一致
- **显式传入 TN**：因为 contractor 不拥有 TN，必须由调用方传入
  - 通常的调用方是 `WaveFunctionComponent::CommitTrial`
  - 也可以是测量代码（直接使用 contractor + TN）

**为什么不提供 `CommitTrial(trial)` 版本（不传入 TN）？**

因为"只更新 contractor cache、不更新 TN"是典型的脚枪（foot-gun）：
- 如果 contractor cache 与 TN 不同步，任何从 TN 重新加载的逻辑（例如 `ClearCache` 后的重算、不同 contractor 的读取）都会得到错误结果
- 调用方容易忘记先调用 `tn.UpdateSiteTensor`，导致难以调试的隐蔽错误

**为什么 contractor 不拥有 TN？**

contractor **不应该拥有** `TensorNetwork2D`（否则又回到 split-tensornetwork2d 之前的 God-class 设计），原因：
- Contractor 是"纯计算"组件，应该可以脱离状态管理独立使用
- Contractor 可以被多个 TN 复用（例如测量不同的 trial configurations）
- TN 的所有权应该在更高层（`WaveFunctionComponent`），由它负责 TN + cache + config 的一致性

但"数据同步"必须是 contractor 的责任，所以 `CommitTrial` 必须同时更新 TN 和 cache，
不能依赖调用方记得做两步操作。

#### (B) Direct-Update + Invalidate 语义（历史遗留，不推荐新代码使用）

这是历史遗留路径，主要用于 BMPS 专家测量（Layer 4）：

1. 调用方直接修改 `TensorNetwork2D`（例如 `tn.UpdateSiteTensor(...)`）
2. 调用方调用 `InvalidateEnvs(site)`（BMPS 专家层）或 `ClearCache()`（Layer 1）
3. 后续调用 `EvaluateReplacement()` 时，contractor 必须保证缓存与 TN 一致

**`InvalidateEnvs` 不属于 Layer 1，而是 Layer 3（BMPS 专家 API）**：

- **历史原因**：在 `main`（pre-split）里，`TensorNetwork2D::UpdateSiteTensor(..., check_envs=true)` 
  会立刻截断 BMPS 环境（`bmps_set_`）。split 后这个逻辑被迁移成 `BMPSContractor::InvalidateEnvs(site)`。
- **现状**：`InvalidateEnvs` 被用于 BMPS 专家测量路径（例如 `MeasureSpinOneHalfOffDiagOrderInRow`），
  这些代码临时修改 TN 来准备环境，然后在循环中复用边界 MPS。
- **语义**：`InvalidateEnvs(site)` 只"标记 dirty"（截断 BMPS 游标），不做大规模重算。

**为什么不强制 TRG 提供 `InvalidateEnvs`？**

- TRG 的"失效"语义是"光锥传播"（fine→coarse 多尺度更新），与 BMPS 的"游标截断"完全不同
- 强行让 TRG 提供 `InvalidateEnvs(site)` 只会产生两个烂选择：
  1. 退化为 `ClearCache()`（丢失细粒度优化）
  2. 实现"伪游标"来模拟 BMPS 语义（制造复杂性）
- **结论**：`InvalidateEnvs` 是 BMPS 专家层（Layer 3），不属于通用契约（Layer 1）

**VMC 主路径不应依赖 `InvalidateEnvs`**：

- 在 `refactor/split-tensornetwork2d` 的历史实现里，`WaveFunctionComponent` 无条件调用
  `contractor.InvalidateEnvs(site)`，这导致它成为模板参数的硬依赖。
- **本 RFC 的结论**：VMC 主路径应改为**只用 Trial+Commit**（见 (A)），不再依赖 `InvalidateEnvs`。
- `InvalidateEnvs` 只保留给 BMPS 专家测量路径（Layer 4）使用。

**迁移路径**：

- **短期**：`WaveFunctionComponent` 继续调用 `InvalidateEnvs`（如果存在），但不强制要求
- **中期**：VMC 主路径完全切换到 Trial+Commit，移除对 `InvalidateEnvs` 的调用
- **长期**：`InvalidateEnvs` 只存在于 `BMPSContractor` 中，作为 Layer 3 API（通过 Doxygen 或 `bmps::expert` 命名空间标记）

### 5.2 最小支持集合与接口统一（消除函数名爆炸）

**问题**：不同 contractor 支持的"局域替换几何"不同，是事实。

**错误的统一方式**：为每种几何创建一个函数（`ReplaceOneSiteTrace`, `ReplaceNNSiteTrace`, `ReplaceNNNSiteTrace`, `ReplaceTNNSiteTrace`, `ReplaceSqrt5DistTwoSiteTrace`, ...）
- 结果：函数名爆炸，每种新几何都要加新函数
- TRG/CTMRG 无法自然地支持这些几何分类（它们不区分 NN/NNN，只看"影响的 site 集合"）

**正确的统一方式**：用数据结构统一（`LocalReplacement`），而不是用函数名：

```cpp
// Old: 函数名爆炸
TenElemT amp1 = contractor.ReplaceOneSiteTrace(tn, site, tensor, orient);
TenElemT amp2 = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient, ten1, ten2);
TenElemT amp3 = contractor.ReplaceNNNSiteTrace(tn, left_up_site, diag_dir, orient, ten1, ten2);

// New: 统一接口
TenElemT amp1 = contractor.EvaluateReplacement(tn, {{site, tensor}});
TenElemT amp2 = contractor.EvaluateReplacement(tn, {{site1, ten1}, {site2, ten2}});
TenElemT amp3 = contractor.EvaluateReplacement(tn, {{site_nnn1, ten1}, {site_nnn2, ten2}});
```

**最小支持集合**：

- **Layer 1 必须支持**：2-site NN（最近邻）替换（这是当前 VMC updater 最常用的更新形态）
- **Layer 1 可选支持**：1-site / 3-site / 任意多 site / 任意几何
- **不支持的几何**：contractor 应明确抛出 `std::logic_error`，由调用方选择 fallback

**迁移策略**：

- `ReplaceOneSiteTrace/ReplaceNNSiteTrace/ReplaceNNNSiteTrace/...` 标记为 `[[deprecated("Use EvaluateReplacement")]]`
- BMPS 的实现：`EvaluateReplacement` 内部根据 `replacements.size()` 和几何关系分发到现有实现
- 长期（v2.0）：删除 `Replace*Trace` 系列函数

## 6. 推荐实现策略

### 6.1 BMPS：在同一个类中实现 Layer 1 + Layer 3

**关键原则：不创建新类，只在 `BMPSContractor` 中添加 Layer 1 接口。**

```cpp
template<typename TenElemT, typename QNT>
class BMPSContractor {
public:
  // Trait for Layer 4 detection
  static constexpr bool supports_cursor_invalidation = true;

  // ===== Layer 1: General Contraction Contract =====
  void Init(const TensorNetwork2D<TenElemT,QNT>&);
  TenElemT Trace(const TensorNetwork2D<TenElemT,QNT>&) const;
  
  // NEW: Pure read-only interface for measurement/energy
  TenElemT EvaluateReplacement(
      const TensorNetwork2D<TenElemT,QNT>&,
      std::span<const LocalReplacement>) const;
  
  // NEW: Stateful trial for VMC update
  Trial BeginTrialWithReplacement(std::span<const LocalReplacement>) const;
  void CommitTrial(TensorNetwork2D<TenElemT,QNT>&, Trial&&);
  
  void ClearCache();
  
  // ===== Layer 3: BMPS Expert API =====
  // (Marked as expert via Doxygen @name or bmps::expert namespace)
  void GenerateBMPSApproach(...);
  void InitBTen(...);
  void GrowBTenStep(...);
  void GrowFullBTen(...);
  void BTenMoveStep(...);
  void BMPSMoveStep(...);
  void InvalidateEnvs(const SiteIdx&);
  
  // Deprecated: use EvaluateReplacement instead
  [[deprecated("Use EvaluateReplacement")]]
  TenElemT ReplaceOneSiteTrace(...) const;
  [[deprecated("Use EvaluateReplacement")]]
  TenElemT ReplaceNNSiteTrace(...) const;
  [[deprecated("Use EvaluateReplacement")]]
  TenElemT ReplaceNNNSiteTrace(...) const;
  
private:
  BMPSSet bmps_set_;
  BTensorSet bten_set_;
  // ... (同一份内部状态)
};
```

**实现示例**：

```cpp
// EvaluateReplacement: dispatch to existing Replace*Trace
template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::EvaluateReplacement(
    const TensorNetwork2D<TenElemT, QNT>& tn,
    std::span<const LocalReplacement> reps) const 
{
  if (reps.size() == 1) {
    return ReplaceOneSiteTrace(tn, reps[0].site, reps[0].tensor, /*...*/);
  } else if (reps.size() == 2 && IsNearestNeighbor(reps[0], reps[1])) {
    // Determine orientation and dispatch to ReplaceNNSiteTrace
    return ReplaceNNSiteTrace(tn, /*...*/);
  } else if (reps.size() == 2 && IsNextNearestNeighbor(reps[0], reps[1])) {
    return ReplaceNNNSiteTrace(tn, /*...*/);
  } else {
    throw std::logic_error("BMPSContractor::EvaluateReplacement: unsupported geometry");
  }
}

// BeginTrialWithReplacement: reuse EvaluateReplacement + save state
template<typename TenElemT, typename QNT>
Trial BMPSContractor<TenElemT, QNT>::BeginTrialWithReplacement(
    std::span<const LocalReplacement> reps) const 
{
  Trial trial;
  trial.amplitude = EvaluateReplacement(*tn_ptr_, reps);
  trial.replacements = {reps.begin(), reps.end()};  // save for CommitTrial
  // Optionally: save environment snapshot for incremental update
  return trial;
}

// CommitTrial: write to TN + update cache
template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::CommitTrial(
    TensorNetwork2D<TenElemT, QNT>& tn, Trial&& trial)
{
  // 1. Write replacement tensors to TN
  for (const auto& rep : trial.replacements) {
    tn(rep.site) = rep.tensor;
  }
  
  // 2. Invalidate affected cache (using existing expert API)
  for (const auto& rep : trial.replacements) {
    InvalidateEnvs(rep.site);
  }
}
```

**性能保证**：

- `EvaluateReplacement` 直接调用现有 `Replace*Trace`（零额外开销）
- Layer 3 API 完全保留（性能路径不变）
- Layer 1 的分发逻辑在高层（不污染核心张量核）

### 6.2 TRG：实现 Layer 1，不强制提供 `InvalidateEnvs`

TRG 的自然接口是 `Trace(tn)` 和多尺度传播，可以直接实现 Layer 1：

```cpp
template<typename TenElemT, typename QNT>
class TRGContractor {
public:
  // Trait: TRG does not support cursor-based invalidation
  static constexpr bool supports_cursor_invalidation = false;

  // Layer 1: Natural for TRG
  void Init(const TensorNetwork2D<TenElemT,QNT>&);
  TenElemT Trace(const TensorNetwork2D<TenElemT,QNT>&) const;
  
  TenElemT EvaluateReplacement(
      const TensorNetwork2D<TenElemT,QNT>&,
      std::span<const LocalReplacement>) const;
  
  Trial BeginTrialWithReplacement(std::span<const LocalReplacement>) const;
  void CommitTrial(TensorNetwork2D<TenElemT,QNT>&, Trial&&);
  
  void ClearCache();
  
  // NO InvalidateEnvs: TRG uses full cache invalidation
  // NO GrowBTenStep/BTenMoveStep: these are BMPS-specific
};
```

**兼容性说明**：

- 旧代码中的 `contractor.InvalidateEnvs(site)` 需要改为 `contractor.ClearCache()`（如果是 TRG）
- 或者 VMC 主路径改为只用 Trial+Commit（不再依赖 `InvalidateEnvs`）

## 7. 上层（VMC/solver）的使用规则

### 7.1 `TPSWaveFunctionComponent`（VMC 主路径）

VMC 主路径通过 `WaveFunctionComponent` 封装操作，**只允许依赖 Layer 1**（必要时 Layer 2），不得依赖 BMPS 手动挡（Layer 3）。

**推荐用法（通过 WaveFunctionComponent）**：

```cpp
// VMC update step
auto trial = tps_sample->BeginTrialWithReplacement({
  {site1, new_config1}, {site2, new_config2}
});

if (AcceptProposal(trial.amplitude, old_amplitude)) {
  tps_sample->CommitTrial(std::move(trial));
  // 一行搞定：TN、contractor cache、config 全部同步
}
```

**内部实现（WaveFunctionComponent）**：

```cpp
template<typename TenElemT, typename QNT>
void TPSWaveFunctionComponent<TenElemT, QNT>::CommitTrial(Trial&& trial) {
  // 1. Call contractor Layer 1 API (passing internal tn)
  contractor.CommitTrial(tn, std::move(trial));
  
  // 2. Update config to match TN
  for (const auto& change : trial.config_changes) {
    config(change.site) = change.new_config;
  }
}
```

**禁止用法**：

```cpp
// ❌ 不要绕过 WaveFunctionComponent 直接操作 contractor
tps_sample->contractor.GrowBTenStep(...);  // Layer 3, BMPS-only
tps_sample->contractor.InvalidateEnvs(site);  // Layer 3, BMPS-only

// ❌ 不要直接修改 TN 而不通过 WaveFunctionComponent
tps_sample->tn(site) = new_tensor;  // 会导致 config 不同步
```

**原因**：
1. TRG/未来 CTMRG 不可能实现 BMPS 的 Grow/Move 语义，强行依赖只会导致到处写 `if (backend==BMPS)`
2. 直接操作 contractor 或 TN 会破坏 TN/cache/config 的一致性，由 WaveFunctionComponent 封装可以防止这类错误

### 7.2 Energy Solver（能量计算）

**推荐用法**（Layer 1）：

```cpp
// 旧代码
TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient, ten1, ten2);
TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);

// 新代码（推荐）
TenElemT psi_ex = contractor.EvaluateReplacement(tn, {{site1, ten1}, {site2, ten2}});
TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
```

**迁移说明**：

- 现有 `Replace*Trace` 调用可以机械替换为 `EvaluateReplacement`
- 性能零损失（BMPS 的 `EvaluateReplacement` 内部调用现有实现）

### 7.3 Measurement Solver（简单测量）

对于单次或少量测量（例如单个 bond 的关联函数），使用 Layer 1：

```cpp
// 测量单个 bond 的 <S+_i S-_j>
auto amp = contractor.EvaluateReplacement(tn, {
  {site_i, op_plus}, {site_j, op_minus}
});
auto correlation = ComplexConjugate(amp * inv_psi);
```

### 7.4 High-Performance Correlation Measurement（Layer 4）

对于需要遍历大量 correlation 的测量（例如 structure factor、同一行 correlation），
如果性能是关键需求，允许使用 BMPS 专家 API（Layer 3），但必须：

1. **明确文档说明**：这是 BMPS/OBC 专用
2. **提供 fallback 或拒绝编译**：
   ```cpp
   if constexpr (std::is_same_v<Contractor, BMPSContractor<TenElemT, QNT>>) {
     // Fast path: use BMPS expert API
     return MeasureCorrelationBMPSExpert(contractor, tn, ...);
   } else {
     // Slow path: use Layer 1 interface
     return MeasureCorrelationGeneric(contractor, tn, ...);
     // Or: static_assert(false, "Requires BMPSContractor");
   }
   ```

**示例：同一行 correlation 测量**

```cpp
// File: measurement_correlation_bmps.h
// @note This measurement uses BMPS expert API (Layer 3/4) for optimal performance.
//       It is BMPS/OBC specific and does not work with TRGContractor.

template<typename TenElemT, typename QNT>
std::vector<TenElemT> MeasureCorrelationInRowBMPSExpert(
    BMPSContractor<TenElemT, QNT>& contractor,
    TensorNetwork2D<TenElemT, QNT>& tn,
    const SplitIndexTPS<TenElemT, QNT>* sitps,
    SiteIdx site_i,
    size_t row,
    /* ... */
) {
  const size_t lx = tn.cols();
  std::vector<TenElemT> correlations(lx / 2);
  
  // 1. Temporarily flip site_i
  tn.UpdateSiteTensor(site_i, flipped_config, *sitps);
  contractor.InvalidateEnvs(site_i);  // Layer 3 API
  
  // 2. Prepare environment
  contractor.GrowBTenStep(tn, LEFT);  // Layer 3 API
  contractor.GrowFullBTen(tn, RIGHT, row, start_col + 2, false);  // Layer 3 API
  
  // 3. Loop: reuse boundary MPS
  for (size_t j = 0; j < lx / 2; j++) {
    correlations[j] = contractor.ReplaceOneSiteTrace(tn, {row, start_col + j}, op_j, HORIZONTAL);
    contractor.BTenMoveStep(tn, RIGHT);  // Layer 3 API: O(D^3) incremental update
  }
  
  // 4. Restore
  tn.UpdateSiteTensor(site_i, original_config, *sitps);
  contractor.InvalidateEnvs(site_i);
  
  return correlations;
}
```

**性能对比**：

- 通用路径（Layer 1）：$O(L_x^2 D^3)$
- BMPS 专家路径（Layer 4）：$O(L_x D^3)$（快 $L_x$ 倍）

**未来方向**：

- TRG 可能通过多尺度缓存自然支持高效 structure factor 测量
- 在 TRG 成熟后，评估是否可以统一到 Layer 1

## 8. 迁移与兼容策略（Never break userspace）

### 8.1 短期（1 个月内）

**BMPS 端**：
- 在 `BMPSContractor` 中添加 Layer 1 接口：
  - `TenElemT EvaluateReplacement(const TensorNetwork2D&, std::span<const LocalReplacement>) const`
  - `Trial BeginTrialWithReplacement(std::span<const LocalReplacement>) const`
  - `void CommitTrial(TensorNetwork2D&, Trial&&)`
- 标记现有 `Replace*Trace` 为 `[[deprecated("Use EvaluateReplacement")]]`
- 用 Doxygen `@name` 分组标记 Layer 1 vs Layer 3 API
- **不删除任何现有接口**

**上层代码**：
- 不需要立即修改（`Replace*Trace` 仍然可用）
- 可选：逐步迁移 energy solver 中的 `Replace*Trace` 调用（机械替换，零风险）

**TRG 端**：
- 继续完善 Layer 1 接口
- **不强制实现 `InvalidateEnvs`**（用 `ClearCache` 代替）

### 8.2 中期（3-6 个月）

**VMC 主路径**：
- `WaveFunctionComponent` 改为只用 Trial+Commit，移除对 `InvalidateEnvs` 的调用
- 确保 VMC 主路径不依赖任何 Layer 3 API

**Measurement Solver**：
- 迁移 energy solver：把 `Replace*Trace` 改为 `EvaluateReplacement`
- 保留高性能 correlation 测量的 BMPS 专家路径（Layer 4）：
  - 添加 `if constexpr` 分发（BMPS 专家路径 vs 通用路径）
  - 在文档中明确标记"BMPS/OBC only"

**可选**：
- 创建 `bmps_contractor_expert.h`，提供 `bmps::expert::*` 命名空间函数
- 迁移 Layer 4 测量代码到新命名空间

### 8.3 长期（v2.0）

- 删除 `[[deprecated]]` 的 `Replace*Trace` 系列函数
- 评估 TRG 的 structure factor 性能，考虑是否可以统一到 Layer 1
- 如果 TRG 性能足够好，考虑逐步移除 Layer 4 的 BMPS 专家路径

### 8.4 兼容性保证

**不会破坏的用户代码**：
- 现有直接调用 `BMPSContractor` 的代码（Layer 3 API 完全保留）
- 现有 `Replace*Trace` 调用（标记 deprecated 但保留）
- 现有 `InvalidateEnvs` 调用（保留在 `BMPSContractor` 中）

**需要修改的代码**（有清晰迁移路径）：
- VMC 主路径：改为 Trial+Commit（新接口，不破坏旧代码）
- 新的 solver：使用 `EvaluateReplacement` 而不是 `Replace*Trace`（更通用）

## 9. 测试（必须能防回归）

### 9.1 Layer 1 接口测试（BMPS 和 TRG 都必须通过）

**Test 1: EvaluateReplacement 一致性**
```cpp
// Setup: create TN with known state
TensorNetwork2D tn = CreateTestTN();
TenElemT baseline = contractor.Trace(tn);

// Replace one site
auto amplitude = contractor.EvaluateReplacement(tn, {{site, new_tensor}});

// Verify: EvaluateReplacement does not modify TN or cache
EXPECT_EQ(contractor.Trace(tn), baseline);  // TN unchanged

// Manually apply replacement and verify amplitude matches
tn(site) = new_tensor;
contractor.ClearCache();
EXPECT_NEAR(contractor.Trace(tn), amplitude, tolerance);
```

**Test 2: Trial-Commit 一致性（Contractor Layer 1）**
```cpp
// Setup: contractor does not own TN
TensorNetwork2D tn = CreateTestTN();
BMPSContractor contractor;
contractor.Init(tn);

// Begin trial (passing TN for read)
auto trial = contractor.BeginTrialWithReplacement({{site, new_tensor}});

// Verify: BeginTrial does not modify TN or persistent cache
TenElemT baseline = contractor.Trace(tn);
EXPECT_NE(baseline, trial.amplitude);  // different amplitudes

// Commit and verify (passing TN for write)
contractor.CommitTrial(tn, std::move(trial));
//                     ^^
//                     Must explicitly pass TN
EXPECT_NEAR(contractor.Trace(tn), trial.amplitude, tolerance);
EXPECT_EQ(tn(site), new_tensor);  // TN updated by CommitTrial
```

**Test 2b: Trial-Commit 一致性（WaveFunctionComponent Higher Layer）**
```cpp
// Setup: WaveFunctionComponent owns TN + contractor + config
TPSWaveFunctionComponent tps_sample(tn_rows, tn_cols, sitps);

// Begin trial (no need to pass TN)
auto trial = tps_sample.BeginTrialWithReplacement({{site, new_config}});

// Verify: BeginTrial does not modify internal state
auto baseline = tps_sample.contractor.Trace(tps_sample.tn);
EXPECT_NE(baseline, trial.amplitude);

// Commit and verify (no need to pass TN)
tps_sample.CommitTrial(std::move(trial));
EXPECT_NEAR(tps_sample.contractor.Trace(tps_sample.tn), trial.amplitude, tolerance);
EXPECT_EQ(tps_sample.tn(site), (*sitps)(site)[new_config]);  // TN updated
EXPECT_EQ(tps_sample.config(site), new_config);  // config updated
```

**Test 3: EvaluateReplacement vs BeginTrial 语义一致**
```cpp
auto amp1 = contractor.EvaluateReplacement(tn, {{site, new_tensor}});
auto trial = contractor.BeginTrialWithReplacement({{site, new_tensor}});
EXPECT_NEAR(amp1, trial.amplitude, tolerance);
```

**Test 4: Multi-site replacement（如果支持）**
```cpp
// 2-site NN replacement
auto amplitude = contractor.EvaluateReplacement(tn, {
  {site1, tensor1}, {site2, tensor2}
});
// Verify against manual replacement + Trace
```

### 9.2 Layer 3 专家 API 测试（仅 BMPS）

**Test 5: InvalidateEnvs + 环境一致性**
```cpp
// Setup BMPS environment
contractor.GenerateBMPSApproach(tn, UP, trunc_para);
TenElemT baseline = contractor.Trace(tn, site, HORIZONTAL);

// Modify TN and invalidate
tn(site) = new_tensor;
contractor.InvalidateEnvs(site);

// Subsequent Trace should reflect modification
TenElemT new_amp = contractor.Trace(tn, site, HORIZONTAL);
EXPECT_NE(baseline, new_amp);
```

**Test 6: BTenMoveStep 增量更新正确性**
```cpp
// Prepare environment
contractor.InitBTen(tn, LEFT, row);
contractor.GrowFullBTen(tn, RIGHT, row, 2, true);

// Get amplitude at col=0
auto amp0 = contractor.ReplaceOneSiteTrace(tn, {row, 0}, tensor, HORIZONTAL);

// Move step and get amplitude at col=1
contractor.BTenMoveStep(tn, RIGHT);
auto amp1 = contractor.ReplaceOneSiteTrace(tn, {row, 1}, tensor, HORIZONTAL);

// Verify: amp1 should match ground truth (full recalculation)
contractor.ClearCache();
contractor.GenerateBMPSApproach(tn, UP, trunc_para);
auto amp1_truth = contractor.ReplaceOneSiteTrace(tn, {row, 1}, tensor, HORIZONTAL);
EXPECT_NEAR(amp1, amp1_truth, tolerance);
```

### 9.3 性能回归测试（可选但推荐）

**Benchmark: Correlation in row（BMPS）**
- 通用路径（Layer 1）：循环调用 `EvaluateReplacement`
- 专家路径（Layer 4）：手动控制环境 + `BTenMoveStep`
- 验证：专家路径应快 $O(L_x)$ 倍

### 9.4 Deprecation 警告测试

```cpp
// Verify deprecated functions still work (but emit warnings)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
auto amp = contractor.ReplaceNNSiteTrace(tn, site1, site2, HORIZONTAL, ten1, ten2);
#pragma GCC diagnostic pop

// Verify matches new interface
auto amp_new = contractor.EvaluateReplacement(tn, {{site1, ten1}, {site2, ten2}});
EXPECT_NEAR(amp, amp_new, tolerance);
```

---

## 10. 快速参考：API 分层总结

### 10.1 Contractor Layer（纯计算组件）

| Layer | 接口 | 使用场景 | 支持者 | 备注 |
|-------|------|---------|--------|------|
| **Layer 0** | `TensorNetwork2D` | 数据容器 | All | Source of truth |
| **Layer 1** | `EvaluateReplacement(tn, ...)` | 测量、能量计算（纯读） | BMPS, TRG, CTMRG | 替代 `Replace*Trace` |
| | `BeginTrialWithReplacement(...)` | VMC 更新（带状态） | BMPS, TRG, CTMRG | 需要后续 commit |
| | `CommitTrial(tn, trial)` | 提交试探更新 | BMPS, TRG, CTMRG | **显式传入 TN** |
| | `Trace(tn)` | 完整收缩 | BMPS, TRG, CTMRG | **显式传入 TN** |
| | `ClearCache()` | 清除所有缓存 | BMPS, TRG, CTMRG | |
| **Layer 2** | `PunchHole(tn, site)` | 梯度计算 | BMPS, (TRG future) | 可选 |
| **Layer 3** | `GenerateBMPSApproach(tn, ...)` | BMPS 环境准备 | BMPS only | 专家 API |
| | `GrowBTenStep(tn, ...)` | BMPS 游标控制 | BMPS only | 专家 API |
| | `InvalidateEnvs(site)` | BMPS 游标截断 | BMPS only | 专家 API，不是 Layer 1 |
| | `ReplaceOneSiteTrace(tn, ...)` | 旧测量接口 | BMPS only | **Deprecated**，用 `EvaluateReplacement` |
| **Layer 4** | 高性能 correlation 测量 | Structure factor 等 | BMPS expert path | 复用 Layer 3，文档标记 |

### 10.2 WaveFunctionComponent Layer（状态管理组件）

| 组件 | 接口 | 使用场景 | 拥有数据 | 备注 |
|------|------|---------|---------|------|
| **WaveFunctionComponent** | `BeginTrialWithReplacement(configs)` | VMC 更新 | TN, contractor, config | **不需要传入 TN** |
| | `CommitTrial(trial)` | 提交更新 | TN, contractor, config | 内部调用 `contractor.CommitTrial(tn, trial)` |
| | `UpdateSingleSite(site, config)` | 单点更新 | TN, contractor, config | 同步 TN/cache/config |

### 10.3 关键差异对比

| | Contractor (Layer 1) | WaveFunctionComponent (Higher Layer) |
|---|---|---|
| **拥有 TN** | ❌ 不拥有 | ✅ 拥有 |
| **接口参数** | 需要显式传入 TN | 不需要传入 TN |
| **输入类型** | `LocalReplacement` (tensor) | `LocalReplacementWithConfig` (config index) |
| **同步责任** | 只同步 TN + cache | 同步 TN + cache + config |
| **独立性** | 可独立使用（测量） | 依赖 contractor |
| **使用场景** | 测量、能量计算、底层操作 | VMC 主路径、用户界面 |

### 关键设计决策

1. **`EvaluateReplacement` vs `BeginTrial`**：
   - 前者纯读（不保存状态），后者带状态（用于 commit）
   - 语义清晰，性能优化空间不同

2. **`InvalidateEnvs` 不是 Layer 1**：
   - BMPS 的"游标截断"与 TRG 的"光锥传播"语义不同
   - 强行统一只会制造伪统一或复杂性爆炸

3. **Layer 4 保留 BMPS 专家路径**：
   - 高性能 correlation 测量（$O(L_x)$ vs $O(L_x^2)$）依赖手动控制遍历顺序
   - 通用接口无法达到最优性能（这是事实，不是设计缺陷）
   - 用 `if constexpr` 分发 + 文档标记，不破坏模块化

4. **不拆分类，只标记层次**：
   - `BMPSContractor` 只有一个类，所有方法在同一个类里
   - 用 Doxygen 分组或 `bmps::expert` 命名空间标记专家 API
   - 零运行时开销，清晰的文档分层

### 迁移优先级

| 优先级 | 任务 | 预期时间 |
|--------|------|---------|
| **P0** | BMPS 实现 `EvaluateReplacement`（复用现有 `Replace*Trace`） | 1 周 |
| **P0** | BMPS 实现 `BeginTrial` + `CommitTrial` | 1 周 |
| **P1** | Doxygen 分组标记 Layer 1 vs Layer 3 | 1 天 |
| **P1** | 标记 `Replace*Trace` 为 deprecated | 1 天 |
| **P2** | 迁移 energy solver 到 `EvaluateReplacement` | 2 周 |
| **P2** | VMC 主路径改为 Trial+Commit | 1 周 |
| **P3** | 封装 Layer 4 测量（`if constexpr` 分发） | 2 周 |
| **P3** | TRG 实现 Layer 1 接口（已部分完成） | 持续 |
| **Future** | 删除 deprecated `Replace*Trace` | v2.0 |



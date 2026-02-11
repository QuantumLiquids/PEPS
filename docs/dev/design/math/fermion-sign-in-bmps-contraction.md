# 费米子BMPS收缩中的符号一致性

本文档解释为什么在费米子PEPS/TPS系统中，计算局域能量（或其他可观测量的比值形式）时，需要局部重新计算波函数振幅 $\Psi(S)$，而不能复用全局缓存的值。

## 1. 背景：SplitIndexTPS的费米子结构

### 1.1 玻色子 vs 费米子的索引差异

将 `TPS` 投影到 `SplitIndexTPS` 时：

- **玻色子**：每个站点张量有 4 条虚拟指标 `{0,1,2,3}`
- **费米子**：每个站点张量有 5 条指标 `{0,1,2,3,4}`，其中第 4 个是**1维的parity索引**

这个额外的parity索引用于保证费米子张量网络中量子数（$\mathbb{Z}_2$ parity）的守恒。

### 1.2 投影后的张量网络

当用 configuration $S$ 投影 `SplitIndexTPS` 生成单层 `TensorNetwork2D` 时：
- 玻色子：得到的张量只有虚拟指标
- 费米子：每个站点张量仍保留一条1维的parity索引

如果不做任何处理，收缩整个 $L_y \times L_x$ 的网络后，会得到一个有 $N = L_y \times L_x$ 条1维索引的"标量张量"，从中读取唯一元素即为 $\Psi(S)$。

## 2. FuseIndex策略

### 2.1 为什么要FuseIndex

保留 $N$ 条parity索引会导致：
- 每次张量操作都需要处理额外的索引
- 中间张量的rank增加（即使每条索引dim=1）

BMPS收缩采用 `FuseIndex` 策略：**在每一步局部收缩后，将新产生的parity索引与已有的parity索引融合**，始终保持只有1条"累积parity"索引。

### 2.2 FuseIndex的数学含义

设两个费米子张量 $A$ 和 $B$ 各有一条parity索引，收缩后产物 $C$ 本来会有两条parity索引。`FuseIndex` 将它们融合为一条：

$$
C^{(p_A, p_B)}_{...} \to C^{(p_{\text{fused}})}_{...}
$$

其中 $p_{\text{fused}} = p_A \oplus p_B$（$\mathbb{Z}_2$ 加法）。这在数值上是正确的，因为最终网络的总parity是守恒的。

## 3. 符号一致性问题

### 3.1 问题根源

`FuseIndex` 的**顺序**会影响最终张量元素的符号。考虑三个张量 $A$, $B$, $C$ 的收缩：

- 路径1：先 $(A, B)$ 融合parity，再与 $C$ 融合
- 路径2：先 $(B, C)$ 融合parity，再与 $A$ 融合

两种路径得到的标量结果在**绝对值**上相同，但**符号**可能相差一个费米交换相位（取决于具体的费米子配置）。

### 3.2 对能量计算的影响

局域能量的计算涉及振幅比：

$$
E_{\mathrm{loc}}(S) = \sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)} \langle S'|H|S\rangle
$$

如果 $\Psi(S)$ 是用一种收缩路径（比如全局Trace），而 $\Psi(S')$ 是用另一种收缩路径（比如ReplaceNNSiteTrace），它们各自的符号虽然"内部自洽"，但**相对符号可能不一致**。

### 3.3 解决方案：局部重新计算

**核心原则**：计算 $\Psi(S')/\Psi(S)$ 时，两者必须使用**相同的收缩路径**。

在 `BMPSContractor` 中，这意味着：
- `Trace(tn, site_a, site_b, orient)` 和 `ReplaceNNSiteTrace(tn, site_a, site_b, orient, ten_a, ten_b)` 使用相同的BMPS环境和收缩顺序
- 因此在 `EvaluateBondEnergy` 中，应该用 `Trace(...)` 局部计算 $\Psi(S)$，再用 `ReplaceNNSiteTrace(...)` 计算 $\Psi(S')$

## 4. 代码中的体现

### 4.1 费米子EvaluateBondEnergy接口

费米子模型的键能量接口返回 `std::optional<TenElemT> &psi` 而非输入 `TenElemT inv_psi`：

```cpp
// 玻色子接口
TenElemT EvaluateBondEnergy(..., const TenElemT inv_psi);

// 费米子接口
TenElemT EvaluateBondEnergy(..., std::optional<TenElemT> &psi);
```

### 4.2 正确的实现模式

```cpp
// 费米子系统中计算键能量
if (config1 != config2) {
  // CRITICAL: 必须局部重新计算psi
  psi = contractor.Trace(tn, site1, site2, orient);
  TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient,
                                                   split_index_tps_on_site1[config2],
                                                   split_index_tps_on_site2[config1]);
  TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
  // ... 使用 ratio 计算能量贡献
}
```

### 4.3 NN vs NNN：psi 重计算策略的差异

NN 和 NNN 能量计算中 psi 的处理机制不同：

#### NN 能量 (EvaluateBondEnergy)

```cpp
// 每次都重新计算 psi
psi = contractor.Trace(tn, site1, site2, orient);
TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient, ...);
TenElemT ratio = psi_ex / psi.value();
```

**原因**：`Trace` 和 `ReplaceNNSiteTrace` 都使用相同的 BTen 环境（`bten_set_.at(LEFT)[col]` 等），保证收缩路径相同。每条 NN 键都有独立的环境设置，所以必须每次重新计算。

#### NNN 能量 (EvaluateNNNEnergy)

```cpp
// 条件计算：同一 plaquette 内可复用
if (!psi.has_value()) {
    psi = contractor.ReplaceNNNSiteTrace(tn, left_up_site, diagonal_dir, HORIZONTAL,
                                          ten_site1[config1], ten_site2[config2]);
}
TenElemT psi_ex = contractor.ReplaceNNNSiteTrace(tn, left_up_site, diagonal_dir, HORIZONTAL,
                                                  ten_site1[config2], ten_site2[config1]);
TenElemT ratio = psi_ex / psi.value();
```

**原因**：NNN 涉及 2×2 plaquette，基类在同一个 plaquette 内调用两次 `EvaluateNNNEnergy`（两条对角线），共享相同的 BTen2 环境：

```cpp
// 在 SquareNNNModelEnergySolver 基类中
std::optional<TenElemT> psi;  // 为整个 plaquette 创建

// 第1条对角线: LEFTUP_TO_RIGHTDOWN
nnn_energy = EvaluateNNNEnergy(..., LEFTUP_TO_RIGHTDOWN, ..., psi);
// psi 现在有值

// 第2条对角线: LEFTDOWN_TO_RIGHTUP（复用 psi）
nnn_energy += EvaluateNNNEnergy(..., LEFTDOWN_TO_RIGHTUP, ..., psi);

contractor.ShiftBTen2Window(...);  // 移动到下一个 plaquette，psi 超出作用域
```

**符号一致性保持**：两次调用都使用 `ReplaceNNNSiteTrace`，在相同的 BTen2 环境下收缩，路径相同。

#### 对比总结

| 方面 | NN | NNN |
|------|-----|-----|
| 计算 psi 的函数 | `Trace` | `ReplaceNNNSiteTrace` |
| psi 复用 | ❌ 每次重新计算 | ✅ 同一 plaquette 内复用 |
| 环境类型 | BTen（单行/列） | BTen2（双行/列） |
| 基类管理 | 每条键独立 | 同一 plaquette 两条对角线共享 psi |

### 4.4 FuseIndex在BMPS收缩中的位置

在 `bmps_contractor_impl.h` 中，每次局部收缩后都有 `FuseIndex` 调用：

```cpp
// 收缩环境张量与站点张量
Contract<TenElemT, QNT, true, true>(up_mps_ten_a, bten_set_.at(LEFT)[col_a], 2, 0, 1, tmp[0]);
tmp[0].FuseIndex(0, 5);  // 融合parity索引
Contract(tmp, {2, 3}, &ten_a, {3, 0}, tmp + 1);
// ...
tmp[2].FuseIndex(0, 5);  // 再次融合
```

## 5. 替代方案比较

### 5.1 当前方案：FuseIndex + 局部重计算

**优点**：
- 中间张量rank不增长
- 内存效率高
- 收缩操作简洁

**缺点**：
- 需要在每个键上局部重新计算 $\Psi(S)$
- 用户需理解符号约定

### 5.2 替代方案：保留所有parity索引

**做法**：不做 `FuseIndex`，让最终结果带有 $N$ 条1维索引

**优点**：
- 符号完全由索引值决定，无需重新计算
- 概念上更清晰

**缺点**：
- 每次张量操作都要处理额外索引
- 中间张量rank增加（虽然dim=1）
- 索引簿记复杂度高

### 5.3 权衡选择

当前PEPS项目选择了FuseIndex方案，因为：
1. 局部重计算的额外开销通常可接受（Trace本身已是O(1)操作，环境已经建好）
2. 代码结构更清晰，符合"局部计算局部负责"的原则
3. 避免了在所有张量操作中传递parity索引的复杂性

## 6. 常见陷阱

### 6.1 错误：复用全局psi

```cpp
// ❌ 错误示例
TenElemT global_psi = tps_sample.amplitude;  // 在别处计算的
TenElemT psi_ex = contractor.ReplaceNNSiteTrace(...);
TenElemT ratio = psi_ex / global_psi;  // 符号可能不一致！
```

### 6.2 正确：局部配对计算

```cpp
// ✅ 正确示例
TenElemT psi = contractor.Trace(tn, site1, site2, orient);  // 局部计算
TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient, ...);
TenElemT ratio = psi_ex / psi;  // 符号一致
```

### 6.3 NNN 的正确用法

NNN 使用 `ReplaceNNNSiteTrace` 而非 `Trace`，且 psi 可在同一 plaquette 内复用：

```cpp
// ✅ 正确：条件计算 psi
if (!psi.has_value()) {
    // 第一次调用（第一条对角线）：计算 psi
    psi = contractor.ReplaceNNNSiteTrace(tn, left_up_site, diagonal_dir, HORIZONTAL,
                                          ten_site1[config1], ten_site2[config2]);
}
// 后续调用（第二条对角线）：复用 psi
TenElemT psi_ex = contractor.ReplaceNNNSiteTrace(tn, left_up_site, diagonal_dir, HORIZONTAL,
                                                  ten_site1[config2], ten_site2[config1]);
TenElemT ratio = psi_ex / psi.value();
```

```cpp
// ❌ 错误：用 Trace 计算 NNN 的 psi
psi = contractor.Trace(tn, site1, site2, HORIZONTAL);  // Trace 只适用于 NN！
TenElemT psi_ex = contractor.ReplaceNNNSiteTrace(...);
// 符号不一致：Trace 和 ReplaceNNNSiteTrace 的收缩路径不同
```

**关键点**：
- NNN 的 psi 和 psi_ex 都必须用 `ReplaceNNNSiteTrace` 计算
- 同一 plaquette 内两条对角线共享 psi（基类管理 `std::optional<TenElemT> psi` 的生命周期）
- 切换到新 plaquette 后，psi 自动失效（超出作用域）

## 7. 总结

| 方面 | 玻色子 | 费米子 |
|------|--------|--------|
| SplitIndexTPS索引数 | 4 | 5（含parity） |
| BMPS收缩 | 直接收缩 | 需FuseIndex |
| 振幅比计算 | 可用全局inv_psi | 必须局部配对计算 |
| EvaluateBondEnergy接口 | 输入inv_psi | 输出psi |

核心要点：**费米子系统中，$\Psi(S)$ 和 $\Psi(S')$ 必须在相同的收缩环境下计算，以保证符号一致性。**

## 参考

- `include/qlpeps/two_dim_tn/tps/split_index_tps.h` - SplitIndexTPS定义
- `include/qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor_impl.h` - BMPS收缩实现
- `docs/user/zh/howto/write_custom_energy_solver.md` - 用户指南中的费米子接口说明
- `docs/dev/design/math/fermion-vmc-math.md` - 费米子VMC数学定义（\(O^*\) 与 \(\Pi\)）
- `docs/dev/design/math/fermion-vmc-implementation.md` - 当前实现约定与代码映射（含SR路径）

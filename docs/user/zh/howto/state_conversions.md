# 状态转换：PEPS / TPS / SplitIndexTPS

本页规范本仓库中常见的三种状态表示之间的转换方式，并推荐使用显式 API。

> 同步状态：英文文档（`docs/user/en/`）为权威版本；如与代码行为冲突，请以英文版本和头文件为准。

## 推荐用法（显式转换）

包含头文件：

```cpp
#include "qlpeps/api/conversions.h"
```

然后使用显式的自由函数：

```cpp
using qlten::special_qn::U1QN;

// PEPS -> TPS
auto tps = qlpeps::ToTPS<double, U1QN>(peps);

// TPS -> SplitIndexTPS
auto sitps = qlpeps::ToSplitIndexTPS<double, U1QN>(tps);

// PEPS -> SplitIndexTPS（一步到位）
auto sitps2 = qlpeps::ToSplitIndexTPS<double, U1QN>(peps);
```

## 为什么推荐显式 API

- 避免隐式转换带来的隐藏开销（尤其是可能发生的深拷贝/重排）。
- 将语义集中到统一位置，避免在调用点“看不出”发生了状态重建。
- 为兼容旧代码保留了部分 legacy 接口，但新代码应当使用显式 API。

## Legacy 接口（已弃用）

- `SquareLatticePEPS::operator TPS()`
- `SplitIndexTPS(const TPS&)`

这些接口为了兼容仍然存在，但已标注 `[[deprecated]]`。新代码请优先使用上面的显式函数。

## 物理指标约定

- 在非 split 的 TPS/PEPS 张量中，物理指标固定在第 4 个位置。
- 费米子张量会多一个奇偶（parity）腿（最后一个指标）。转换过程会保持量子数一致性。

## 转换后：规范化与幅度尺度（推荐）

在蒙特卡洛采样中，如果采样扇区里典型的波函数振幅量级接近 \(O(1)\)，数值通常更稳定。
如果振幅极小/极大，接受率与局域能量计算可能会变得脆弱。

典型流程（在 `ToSplitIndexTPS(...)` 之后）：

```cpp
// 逐站点规范化（局部缩放）。
sitps.NormalizeAllSite();

// 可选：把每个站点张量的最大元素幅值缩放到目标附近。
// 这是实用的数值预处理；不改变物理态。
sitps.ScaleMaxAbsForAllSite(/*aiming_max_abs=*/1.0);
```

说明：

- `NormalizeAllSite()` 通常是便宜且安全的“让它合理”步骤，建议在 VMC/测量前做一次。
- `ScaleMaxAbsForAllSite(...)` 是可选项；当你观察到幅度上溢/下溢、或接受率极低时再启用。


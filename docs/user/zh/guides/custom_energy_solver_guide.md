# 自定义能量求解器开发指南：基础用法

本指南详细说明如何基于PEPS框架的基础基类开发自定义能量求解器，适用于方格格子上最多包含次近邻相互作用的物理模型。

## 概述

本指南覆盖PEPS框架中最常用的两个基础基类，能够处理绝大多数标准物理模型：

### 适用模型

**典型场景** 二维方格晶格最近邻(NN)和次近邻(NNN)哈密顿量

**邪修** Kitaev model, Triangle NN Heisenberg model。能在NNN范围内塞进到二维方格晶格里都可以。

### 基类选择
1. **最近邻模型** - `SquareNNModelEnergySolver`
   - 适用于只包含最近邻相互作用的模型
   - 如：海森堡模型、简单Hubbard模型、横场Ising模型等

2. **次近邻模型** - `SquareNNNModelEnergySolver`  
   - 适用于包含次近邻相互作用的模型
   - 如：J1-J2模型、扩展Hubbard模型、挫败自旋系统等

**注意** 能用最近邻模型接口的不要用次近邻模型接口然后设置次近邻相互作用强度=0。这样能规避不必要的基类的张量网络收缩操作计算以提高效率。

## 基础用法1：最近邻(NN)模型

### 适用场景
- 只包含最近邻相互作用的模型
- 如：Spin-1 最近邻海森堡模型， 包含最近邻电子density interaction的Hubbard 模型、Transverse Field Ising 模型等。
- 邪修：Kitaev model

### 继承基类
```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"

class MyNNModel : public SquareNNModelEnergySolver<MyNNModel> {
```

### 必须实现的方法

#### 1. 键能量评估方法

**玻色子/自旋系统接口**：
```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,           // 两个相邻格点
    const size_t config1, const size_t config2,         // 格点上的局域状态
    const BondOrientation orient,                       // 键方向：HORIZONTAL/VERTICAL
    const TensorNetwork2D<TenElemT, QNT> &tn,          // 张量网络
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    const TenElemT inv_psi                              // 波函数倒数（输入参数）
);
```

**费米子系统接口**：
```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,           // 两个相邻格点
    const size_t config1, const size_t config2,         // 格点上的局域状态
    const BondOrientation orient,                       // 键方向：HORIZONTAL/VERTICAL
    const TensorNetwork2D<TenElemT, QNT> &tn,          // 张量网络
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi                        // 波函数振幅（返回值，用于数值检查）
);
```

**参数说明**：
- `site1, site2`：相邻的两个格点坐标
- `config1, config2`：每个格点上的粒子配置（编码为整数）
- `orient`：键的方向，用于张量网络收缩
- `tn`：当前的二维张量网络
- `split_index_tps_on_site1/2`：每个格点的所有可能态张量

**关键差异**：
- **玻色子系统**：最后参数是 `const TenElemT inv_psi`（输入），基类已经计算好波函数倒数
- **费米子系统**：最后参数是 `std::optional<TenElemT> &psi`（输出），需要在函数内部计算并返回波函数振幅用于数值检查
这一差异产生的原因是，费米子psi的符号不能完全由Configuration来决定，因而inv_psi不是一个global的量。inv_psi或psi值的计算，下放给EvaluateBondEnergy内部来计算。使得用于计算非对角项能量中Psi(S')/Psi(S)，的S'和S具有相同的指标外围指标顺序，以确保符号正确。

### 数学定义：EvaluateBondEnergy 的数学定义

设当前采样组态为 \(S\)，考察一条最近邻键 \(\langle i,j\rangle\)。记作用在该键上的哈密顿子算符为 \(\hat{H}^{\text{bond}}_{ij}\)（或能量密度 \(\hat{h}_{ij}\)）。则
\[
E_{\text{bond}}(S; i,j)
\;=\; \sum_{\sigma'_i,\sigma'_j}\, \big\langle \sigma'_i\sigma'_j \big|\, \hat{H}^{\text{bond}}_{ij} \,\big| \sigma_i\sigma_j \big\rangle\; \frac{\Psi^*(S')}{\Psi^*(S)} ,
\]
其中 \(S'\) 仅在站点 \(i,j\) 的物理占据由 \((\sigma_i,\sigma_j)\) 替换为 \((\sigma'_i,\sigma'_j)\)。当 \((\sigma'_i,\sigma'_j)=(\sigma_i,\sigma_j)\) 时对应对角贡献；否则为非对角过程（自旋翻转、费米子跃迁等），每一项都乘以幅度比 \(\Psi^*(S')/\Psi^*(S)\)。

对应到实现：
- 定义 `psi_ex = tn.ReplaceNNSiteTrace(...)` 为把 \(i,j\) 上的局域张量替换为 \((\sigma'_i,\sigma'_j)\) 后得到的 \(\Psi(S')\)。
- 玻色子接口使用 `inv_psi`（即 \(1/\Psi(S)\)），计算
  \[
  \frac{\Psi^*(S')}{\Psi^*(S)}\;=\;\big(\psi_{\text{ex}}\cdot \text{inv\_psi}\big)^* \;=\; \text{ComplexConjugate}(\psi_{\text{ex}} \cdot \text{inv\_psi}).
  \]
- 费米子接口必须在函数内部先用 `psi = tn.Trace(...)` 重新得到“同一外部指标顺序”下的 \(\Psi(S)\)，再计算
  \[
  \frac{\Psi^*(S')}{\Psi^*(S)}\;=\;\Big(\frac{\psi_{\text{ex}}}{\psi}\Big)^* \;=\; \text{ComplexConjugate}(\psi_{\text{ex}} / \psi).
  \]

直观表述：对角项直接给出能量常数；非对角项的能量权重是“哈密顿矩阵元”乘以“幅度比的复共轭”。这与《模型能量求解器指南》中局域能量的统一定义严格一致：
\[
E_{\mathrm{loc}}(S) = \sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)} \langle S'| \, \hat{H} \, | S\rangle .
\]
等价文字描述：非对角项能量贡献等于相应跃迁矩阵元乘以 \(\Psi^*(S')/\Psi^*(S)\)。

#### 2. 在位能量评估方法
```cpp
double EvaluateTotalOnsiteEnergy(const Configuration &config);
```
计算所有格点的在位对角能量总和，如化学势、磁场等单体项。

### 设计动机：为何保留 EvaluateTotalOnsiteEnergy（可选）

原则上，接口可以不需要单独的 `EvaluateTotalOnsiteEnergy`，因为在位项也能并入每条键的处理里（或统一视为对角键）。但保留该接口有一个很实用的优势：

- 当去掉 onsite 之后，if bond energy is uniform（键项在实现层面形式一致、无特殊分支），代码更简洁可读；
- onsite 的所有细节集中在一处实现，减少在 `EvaluateBondEnergy` 内部的条件分支与特殊情况，符合“消除特殊情况优于增加判断”的准则。

数学上，在位能量是
\[
E_{\text{onsite}}(S)\;=\;\sum_i h_i(\sigma_i) ,
\]
即把每个格点的单体项累加即可；亦可理解为对每个站点算符 \(\hat{h}_i\) 的对角矩阵元求和。在位能量等于对每个 site 的单点哈密顿算符贡献的求和。

### 玻色子系统示例：带交错磁场的自旋1/2海森堡模型

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"
#include "qlpeps/utility/helpers.h"

namespace qlpeps {

/**
 * 带交错磁场的海森堡模型
 * \hat{H} = J * \sum_{\langle i,j \rangle} (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j) + h * \sum_i (-1)^{i+j} S^z_i
 */
class StaggeredFieldHeisenbergModel : public SquareNNModelEnergySolver<StaggeredFieldHeisenbergModel> {
public:
    StaggeredFieldHeisenbergModel(double J, double h) : J_(J), h_(h) {}
    
    static constexpr bool requires_density_measurement = false;
    static constexpr bool requires_spin_sz_measurement = true;
    
    // 键能量计算（玻色子系统接口）
    template<typename TenElemT, typename QNT>
    TenElemT EvaluateBondEnergy(
        const SiteIdx site1, const SiteIdx site2,
        const size_t config1, const size_t config2,
        const BondOrientation orient,
        const TensorNetwork2D<TenElemT, QNT> &tn,
        const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
        const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
        const TenElemT inv_psi  // 玻色子系统：输入参数
    ) {
        if (config1 == config2) {
            // 对角项：J * <S^z_i S^z_j> = J * (±1/4)
            return 0.25 * J_;
        } else {
            // 非对角项：J * <S^x_i S^x_j + S^y_i S^y_j>
            // 计算 <config'|H|config> 其中 config' 是翻转后的配置
            TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                                   split_index_tps_on_site1[config2],
                                                   split_index_tps_on_site2[config1]);
            TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
            return (-0.25 * J_ + ratio * 0.5 * J_);
        }
    }
    
    // 在位能量：交错磁场
    double EvaluateTotalOnsiteEnergy(const Configuration &config) {
        double energy = 0.0;
        for (size_t row = 0; row < config.rows(); row++) {
            for (size_t col = 0; col < config.cols(); col++) {
                double sz = double(config({row, col})) - 0.5;  // 0->-0.5, 1->+0.5
                double stagger_sign = ((row + col) % 2 == 0) ? 1.0 : -1.0;
                energy += h_ * stagger_sign * sz;
            }
        }
        return energy;
    }

private:
    double J_;  // 海森堡交换积分
    double h_;  // 交错磁场强度
};

} // namespace qlpeps
```

### 费米子系统示例：简化t-J模型

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"
#include "qlpeps/utility/helpers.h"
#include "qlpeps/vmc_basic/tj_single_site_state.h"

namespace qlpeps {

/**
 * 简化 t-J 模型 (基于内置模型)
 * \hat{H} = -t * \sum_{\langle i,j \rangle,\sigma} (c^\dag_{i,\sigma} c_{j,\sigma} + h.c.)
 *         + J * \sum_{\langle i,j \rangle} (\mathbf{S}_i \cdot \mathbf{S}_j - \tfrac{1}{4} n_i n_j)
 */
class SimpleTJModel : public SquareNNModelEnergySolver<SimpleTJModel> {
public:
    SimpleTJModel(double t, double J) : t_(t), J_(J) {}
    
    static constexpr bool requires_density_measurement = true;
    static constexpr bool requires_spin_sz_measurement = true;
    
    // 键能量计算（费米子系统接口）
    template<typename TenElemT, typename QNT>
    TenElemT EvaluateBondEnergy(
        const SiteIdx site1, const SiteIdx site2,
        const size_t config1, const size_t config2,
        const BondOrientation orient,
        const TensorNetwork2D<TenElemT, QNT> &tn,
        const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
        const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
        std::optional<TenElemT> &psi  // 费米子系统：输出参数
    ) {
        if (config1 == config2) {
            psi.reset(); // 这里可能不是一个好的设计，在上层代码中引入负担。需考虑基类设计。
            if (tJSingleSiteState(config1) == tJSingleSiteState::Empty) {
                return 0.0;
            } else {
                // 自旋相互作用对角项
                return 0.0;  // sz * sz - 1/4 * n * n = 0 for parallel spins
            }
        } else {
            psi = tn.Trace(site1, site2, orient);// CRITICAL: 费米子系统中，需要用到波函数振幅psi时，要在局部重新计算psi，以确保和psi_ex符号的一致性。这一情况在NNN系统中会变得更加复杂。用户需完整理解费米子符号如何在能量计算中起作用！
            TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                                   split_index_tps_on_site1[config2],
                                                   split_index_tps_on_site2[config1]);
            TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
            
            if (tJSingleSiteState(config1) == tJSingleSiteState::Empty ||
                tJSingleSiteState(config2) == tJSingleSiteState::Empty) {
                // 跳跃项
                return -t_ * ratio;
            } else {
                // 自旋翻转项
                return (-0.5 + ratio * 0.5) * J_;
            }
        }
    }
    
    double EvaluateTotalOnsiteEnergy(const Configuration &config) {
        return 0.0;  // 无化学势
    }

private:
    double t_;  // 跳跃强度
    double J_;  // 交换积分
};

} // namespace qlpeps
```

## 基础用法2：次近邻(NNN)模型

### 适用场景
- 包含次近邻相互作用的模型
- 如：J1-J2模型、扩展Hubbard模型、自旋模型的挫败系统等
- 邪修： Triangle NN Heisenberg model

### 继承基类
```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_energy_solver.h"
class MyNNNModel : public SquareNNNModelEnergySolver<MyNNNModel> {
```

### 必须实现的方法

除了NN模型的所有方法外，还需要实现：

#### 次近邻能量评估方法
**玻色子系统接口**：
```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateNNNEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,                    // 对角方向
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    const TenElemT inv_psi
);
```

**费米子系统接口**：
```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateNNNEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi //这里的sign需要比较仔细的考虑！可以参考次近邻 tJ 模型的实现。
);
```

**新增参数**：
- `diagonal_dir`：对角线方向，`LEFTUP_TO_RIGHTDOWN` 或 `LEFTDOWN_TO_RIGHTUP`

**其他方法**：与NN模型相同（`EvaluateBondEnergy`、`EvaluateTotalOnsiteEnergy`等）

### 数学定义：EvaluateNNNEnergy 的精确定义

与 NN 情况相同，只是成键对改为次近邻对 \(\langle\!\langle i,j\rangle\!\rangle\)：
\[
E_{\text{diag}}(S; i,j)
\;=\; H_{\text{diag}}^{\text{(NNN)}}(\sigma_i,\sigma_j),\quad
E_{\text{off}}(S; i,j)
\;=\; \sum_{\sigma'_i,\sigma'_j} H_{\text{off}}^{\text{(NNN)}}\big((\sigma_i,\sigma_j)\to(\sigma'_i,\sigma'_j)\big)\, \frac{\Psi^*(S')}{\Psi^*(S)}.
\]
实现层面：
- 玻色子接口仍用 `ComplexConjugate(psi_ex * inv_psi)`；
- 费米子接口仍需在本地计算 `psi`，再用 `ComplexConjugate(psi_ex / psi)`，以确保与 `psi_ex` 使用同一外部指标顺序与费米符号约定。

等价文字描述：把 NN 的计算替换为 NNN 对，并保持相同的“矩阵元 × 幅度比（取复共轭）”的结构即可。

### 记号与顺序约定（Fermion 注意事项）

- 本文档为叙述清晰，统一使用“玻色子式”的组态记号 \(S\) 与 \(S'\)。对应代码中：
  - \(\psi \equiv \Psi(S)\) 与 `psi = tn.Trace(...)`；
  - \(\psi_{\text{ex}} \equiv \Psi(S')\) 与 `psi_ex = tn.ReplaceNNSiteTrace(...)`（如偏好，可记 \(\Psi'\)）。
- 费米子振幅严格依赖于“外部指标顺序”（全局模式/格点的规范排序）。因此，任何用于构造 \(\psi\) 与 \(\psi_{\text{ex}}\) 的张量收缩，必须保证两者在完全相同的外部指标顺序下取张量元，否则会引入额外的费米交换符号（相位）。

### 费米子 NNN hopping 的易错点（如何避免符号不一致）

考虑 2×2 方块 4个sites（编号如图）：

| 1 | 2 |
|---|---|
| 3 | 4 |

计算 NNN 键 (2,3) 的 hopping 贡献时，建议遵循以下步骤：
1) 局部环境：先把包含 1、2、3、4 的单层张量与其环境收缩到“仅保留 1、2、3、4 的物理腿”为止；
2) 物理腿重排：将站点 2 与 3 的物理腿在张量中的顺序调整为相邻，且该顺序与用于计算 `psi = tn.Trace(..)` 的“外部指标顺序”一致；
3) 取张量元：
   - 原组态 \(S\) 的局部取值给出 \(\psi = \Psi(S)\)；
   - 交换/跃迁后的组态 \(S'\) 的局部取值给出 \(\psi_{\text{ex}} = \Psi(S')\)；
4) 计算比值并取复共轭：\(\text{ratio} = ((\psi_{\text{ex}}/\psi))^*\)。

要点：步骤 (2) 确保了 \(\psi\) 与 \(\psi_{\text{ex}}\) 在相同的外部指标顺序下定义，从而不会因为张量元素访问次序不同而引入额外的费米交换相位。NN 情形由 `ReplaceNNSiteTrace`/`Trace` 自然保证该一致性；NNN 情形需要你在局部把待作用的两个站点的物理腿先挪到一起，再取元素。



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

#### 2. 在位能量评估方法
```cpp
double EvaluateTotalOnsiteEnergy(const Configuration &config);
```
计算所有格点的在位对角能量总和，如化学势、磁场等单体项。

### 玻色子系统示例：带交错磁场的自旋1/2海森堡模型

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"
#include "qlpeps/utility/helpers.h"

namespace qlpeps {

/**
 * 带交错磁场的海森堡模型
 * H = J * sum_{<i,j>} (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j) + h * sum_i (-1)^{i+j} S^z_i
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
 * 简化t-J模型 (基于内置模型)
 * H = -t * sum_{<i,j>,sigma} (c^dag_{i,sigma} c_{j,sigma} + h.c.) 
 *     + J * sum_{<i,j>} (S_i · S_j - 1/4 n_i n_j)
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


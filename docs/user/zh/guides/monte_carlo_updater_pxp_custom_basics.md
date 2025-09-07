# 蒙特卡洛更新器：自定义（PXP）基础篇

## 目标

以 square lattice PXP 模型为例，演示如何在蒙特卡洛更新器中投影掉“最近邻同时激发”的非法局域配置，同时保持采样满足 balance condition 并收敛到目标分布。

PXP 约束：两相邻站点不允许同时处于激发态（记作 1）。即任意相邻对 \((i,j)\) 上的局域配置 \((1,1)\) 被硬性排除。

## 约束实现的两种方式

- 方式 A（建议）：在候选生成阶段直接剔除非法候选；保持候选顺序“固定且与当前状态无关”，然后用 NonDB 进行抽样。
- 方式 B：对非法候选赋权重 0（或将振幅比权重设为 0），仍使用固定候选顺序与 NonDB 进行选择。

两者等价的核心是在“候选集合与其顺序”对每一步都固定，不依赖当前起始状态，从而满足 PRL 105, 120603 (2010) 的非详细平衡选择核要求。

## 接口与工具

- 非详细平衡多候选选择：`SuwaTodoStateUpdate(init_state, weights, rng)`
  - 头文件：`qlpeps/vmc_basic/monte_carlo_tools/suwa_todo_update.h`
  - 约束：候选顺序在整个模拟中固定，且不依赖 init_state；否则违反 balance condition。

## 函数签名与 CRTP 集成

在 square lattice 的最近邻两站点更新器体系中，需实现如下成员函数以接入 CRTP 基类：

```cpp
template<typename TenElemT, typename QNT>
bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1,
                              const SiteIdx &site2,
                              BondOrientation bond_dir,
                              const SplitIndexTPS<TenElemT, QNT> &sitps,
                              TPSWaveFunctionComponent<TenElemT, QNT> &tps_component);
```

该函数由基类 `MCUpdateSquareNNUpdateBase<Derived>` 在横向-纵向顺序扫描过程中多次调用。你只需定义“如何在一个 NN 键上从 (config1, config2) 提议并接受一个新配置”的局部逻辑。

基类位置与用法参考：`include/qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h`。

## 实践要点

- 固定候选顺序（例如词典序），且在整个模拟期间不变；不依赖当前 `init_state`。
- 非法候选可直接置零或不加入候选表；推荐置零以保持候选长度恒定，便于验证顺序不变。
- 使用 `SuwaTodoStateUpdate` 时，严格遵守其头文件中的顺序约束，避免引入隐蔽偏差。

## 完整示例：MCUpdateSquareNNFullSpacePXP（可直接作为自定义更新器）

下面给出一个完整的自定义类，继承 `MCUpdateSquareNNUpdateBase`，在两站点 full-space 更新上加入 PXP 投影，并通过 NonDB 进行多候选选择。

```cpp
#include "qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h"   // MCUpdateSquareNNUpdateBase
#include "qlpeps/vmc_basic/monte_carlo_tools/suwa_todo_update.h"         // SuwaTodoStateUpdate

namespace qlpeps {

class MCUpdateSquareNNFullSpacePXP : public MCUpdateSquareNNUpdateBase<MCUpdateSquareNNFullSpacePXP> {
 public:
  template<typename TenElemT, typename QNT>
  bool TwoSiteNNUpdateLocalImpl(const SiteIdx &site1, const SiteIdx &site2, BondOrientation bond_dir,
                                const SplitIndexTPS<TenElemT, QNT> &sitps,
                                TPSWaveFunctionComponent<TenElemT, QNT> &tps_component) {
    auto &tn = tps_component.tn;
    const size_t dim = sitps.PhysicalDim();
    // 可选：对 PXP 进行防御式断言
    // assert(dim == 2 && "PXP expects two-level local Hilbert space");

    // 固定候选顺序（词典序）
    const size_t init_config = tps_component.config(site1) * dim + tps_component.config(site2);
    std::vector<TenElemT> alternative_psi(dim * dim);
    alternative_psi[init_config] = tps_component.amplitude;

    for (size_t c1 = 0; c1 < dim; ++c1) {
      for (size_t c2 = 0; c2 < dim; ++c2) {
        const size_t id = c1 * dim + c2;
        if (id == init_config) continue;
        const bool forbidden = (c1 == 1 && c2 == 1); // PXP 硬约束
        if (forbidden) {
          alternative_psi[id] = TenElemT(0);
          continue;
        }
        alternative_psi[id] = tn.ReplaceNNSiteTrace(site1, site2, bond_dir,
                                                    sitps(site1)[c1],
                                                    sitps(site2)[c2]);
      }
    }

    // NonDB 权重
    std::vector<double> weights(dim * dim, 0.0);
    const TenElemT &psi_old = tps_component.amplitude;
    for (size_t i = 0; i < alternative_psi.size(); ++i) {
      if (i == init_config) { weights[i] = 1.0; continue; }
      const double r = std::abs(alternative_psi[i] / psi_old);
      weights[i] = r * r;
    }

    const size_t final_state = SuwaTodoStateUpdate(init_config, weights, random_engine_);
    if (final_state == init_config) return false;

    tps_component.UpdateLocal(sitps, alternative_psi[final_state],
                              std::make_pair(site1, final_state / dim),
                              std::make_pair(site2, final_state % dim));
    return true;
  }
};

} // namespace qlpeps
```

## 如何在执行器中使用

像使用内置更新器一样，将 `MCUpdateSquareNNFullSpacePXP` 作为模板参数传给执行器即可：

```cpp
using UpdaterType = qlpeps::MCUpdateSquareNNFullSpacePXP;
using SolverType  = /* 你的能量求解器 */;

VMCPEPSOptimizer<TenElemT, QNT, UpdaterType, SolverType>
  executor(vmc_params, initial_tps, MPI_COMM_WORLD, solver);
executor.Execute();
```

## 与三站点/更复杂更新的关系

三站点（TNN）或 cluster/loop 更新可采用同样的思路：
- 先枚举固定顺序的候选置换或子空间；
- 对不合法（违反 PXP 或其它硬约束）的候选置零权重；
- 调用 `SuwaTodoStateUpdate` 完成无偏选择；
- 最后用 `UpdateLocal` 完成一次原子更新。

## 参考

- 非详细平衡选择核：PRL 105, 120603 (2010)
- 接口：`qlpeps/vmc_basic/monte_carlo_tools/suwa_todo_update.h`



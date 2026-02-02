# 自定义能量求解器（How-to）

本页说明如何基于本仓库提供的方格晶格基类，实现一个自定义的模型能量求解器（energy solver）。
主要面向方格晶格上的最近邻（NN）/次近邻（NNN）哈密顿量（以及“能塞进 NN/NNN 局域结构里”的模型）。

## 选择一个基类（方格晶格）

基类位于：

- `include/qlpeps/algorithm/vmc_update/model_solvers/base/`

请选择与你的哈密顿量局域结构匹配的**最简单**基类：

1. 仅 NN：`SquareNNModelEnergySolver<YourModel>`
2. NN + NNN：`SquareNNNModelEnergySolver<YourModel>`

性能提示：

- 如果你的模型没有 NNN 项，不要用 NNN 基类再把 “J2=0”；它会引入额外的收缩工作。

适用范围提示：

- 该框架名义上面向方格晶格 NN/NNN，但只要你能把模型写成相同的局域结构并保持收缩逻辑一致，也可以覆盖一些“非常规映射模型”。

## Part 1：NN 模型（`SquareNNModelEnergySolver`）

### 继承基类

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"

class MyNNModel : public SquareNNModelEnergySolver<MyNNModel> {
  // ...
};
```

### 必需接口：`EvaluateBondEnergy`

你必须在两种签名（自旋/玻色子 vs 费米子）中**二选一**实现，不能混用。

#### 自旋 / 玻色子接口

```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    const TenElemT inv_psi);
```

关键点：

- `inv_psi` 由基类提供（`1 / Psi(S)`），并使用一致的收缩路径计算。

#### 费米子接口

```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi);
```

关键点：

- 对费米子，幅度的符号可能依赖指标顺序。通常你需要在局部用与 `psi_ex = Psi(S')` 相同的收缩约定重新计算 `psi = Psi(S)`，然后用 `ratio = conj(psi_ex / psi)`。
- 参考：`docs/dev/design/math/fermion-sign-in-bmps-contraction.md`。

### 可选接口：对角的 on-site 项

如果你的模型包含纯对角的 on-site 项（化学势、外场等），实现：

```cpp
double EvaluateTotalOnsiteEnergy(const Configuration &config);
```

单独保留这个接口的原因：

- 当键能量逻辑是统一形式时，它能让 `EvaluateBondEnergy` 更干净、少分支；
- on-site 的细节集中在一处实现，避免在每条键里塞特殊情况。

### 数学定义（你在计算什么）

对一条 NN 键 \(\langle i,j\rangle\) 与一个蒙特卡洛组态 \(S\)，计算：

\[
E_{\text{bond}}(S;i,j)=\sum_{\sigma'_i,\sigma'_j}
\langle \sigma'_i\sigma'_j|\hat{H}^{\text{bond}}_{ij}|\sigma_i\sigma_j\rangle
\cdot \frac{\Psi^*(S')}{\Psi^*(S)} ,
\]

其中 \(S'\) 仅在站点 \(i,j\) 上与 \(S\) 不同。

实现层面的经验法则：

- 对角项：仅依赖 \((\sigma_i,\sigma_j)\) 的常数贡献；
- 非对角项：使用 “矩阵元 × 幅度比的复共轭”。

该约定与本仓库的局域能量定义一致，详见：

- `../explanation/model_energy_solver_math.md`

### 示例（自旋）：海森堡 + 交错场

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"
#include "qlpeps/utility/helpers.h" // ComplexConjugate

namespace qlpeps {

class StaggeredFieldHeisenbergModel
    : public SquareNNModelEnergySolver<StaggeredFieldHeisenbergModel> {
 public:
  StaggeredFieldHeisenbergModel(double J, double h) : J_(J), h_(h) {}

  static constexpr bool requires_density_measurement = false;
  static constexpr bool requires_spin_sz_measurement = true;

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      const TenElemT inv_psi) {
    if (config1 == config2) {
      // Diagonal: J * <Sz_i Sz_j> = +J/4 for parallel spins (using 0/1 encoding).
      return TenElemT(0.25 * J_);
    }
    // Off-diagonal: spin flip terms, weighted by conj(psi_ex / psi).
    const TenElemT psi_ex = contractor.ReplaceNNSiteTrace(
        tn, site1, site2, orient,
        split_index_tps_on_site1[config2],
        split_index_tps_on_site2[config1]);
    const TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
    return TenElemT(-0.25 * J_) + ratio * TenElemT(0.5 * J_);
  }

  double EvaluateTotalOnsiteEnergy(const Configuration &config) {
    double e = 0.0;
    for (size_t row = 0; row < config.rows(); ++row) {
      for (size_t col = 0; col < config.cols(); ++col) {
        const double sz = static_cast<double>(config({row, col})) - 0.5; // 0->-0.5, 1->+0.5
        const double sign = ((row + col) % 2 == 0) ? 1.0 : -1.0;
        e += h_ * sign * sz;
      }
    }
    return e;
  }

 private:
  double J_;
  double h_;
};

} // namespace qlpeps
```

### 示例（费米子）：t–J 风格的 hopping + exchange（模式示意）

这是一个最小草图，强调**符号一致性规则**：用相同约定计算 `psi` 与 `psi_ex`。

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"
#include "qlpeps/utility/helpers.h" // ComplexConjugate

namespace qlpeps {

class SimpleTJModel : public SquareNNModelEnergySolver<SimpleTJModel> {
 public:
  SimpleTJModel(double t, double J) : t_(t), J_(J) {}

  static constexpr bool requires_density_measurement = true;
  static constexpr bool requires_spin_sz_measurement = true;

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi) {
    if (config1 == config2) {
      psi.reset(); // no need to compute amplitude for diagonal-only contribution
      return TenElemT(0);
    }

    // Critical: recompute Psi(S) locally using the same contraction path/order.
    psi = contractor.Trace(tn, site1, site2, orient);
    const TenElemT psi_ex = contractor.ReplaceNNSiteTrace(
        tn, site1, site2, orient,
        split_index_tps_on_site1[config2],
        split_index_tps_on_site2[config1]);
    const TenElemT ratio = ComplexConjugate(psi_ex / psi.value());

    // Your model logic chooses the matrix element (hopping vs exchange vs ...)
    return -t_ * ratio;
  }

  double EvaluateTotalOnsiteEnergy(const Configuration &) { return 0.0; }

 private:
  double t_;
  double J_;
};

} // namespace qlpeps
```

## Part 2：NNN 模型（`SquareNNNModelEnergySolver`）

### 继承基类

```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_energy_solver.h"

class MyNNNModel : public SquareNNNModelEnergySolver<MyNNNModel> {
  // ...
};
```

### 额外必需接口：`EvaluateNNNEnergy`

除了 NN 的接口外，你还必须实现 NNN 的能量评估：

#### 自旋 / 玻色子接口

```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateNNNEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    const TenElemT inv_psi);
```

#### 费米子接口

```cpp
template<typename TenElemT, typename QNT>
TenElemT EvaluateNNNEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi);
```

新增参数：

- `diagonal_dir`：对角方向（`LEFTUP_TO_RIGHTDOWN` 或 `LEFTDOWN_TO_RIGHTUP`）。

### 费米子 NNN hopping：很容易忽略的符号问题

NNN 项经常要求你显式处理局部物理腿的顺序。对一个 2×2 plaquette（站点 1–4），一个更稳妥的思路是：

1. 把局部环境收缩到只剩 plaquette 的物理腿；
2. 重新排列物理腿，使得“被作用的两个站点”相邻，且顺序与 `psi = Psi(S)` 的定义一致；
3. 用相同的外腿顺序分别得到 \(S\) 的 `psi` 与 \(S'\) 的 `psi_ex`；
4. 使用 `ratio = conj(psi_ex / psi)`。

NN 项很多时候可以依赖辅助接口（`Trace` / `ReplaceNNSiteTrace`）来保证顺序一致；NNN 项往往不行。

数学上，NNN 的结构与 NN 相同，只是作用在对角键 \(\langle\!\langle i,j\rangle\!\rangle\) 上：

\[
E_{\text{off}}^{\text{(NNN)}}(S;i,j)
= \sum_{\sigma'_i,\sigma'_j}
H_{\text{off}}^{\text{(NNN)}}\big((\sigma_i,\sigma_j)\to(\sigma'_i,\sigma'_j)\big)\,
\frac{\Psi^*(S')}{\Psi^*(S)}.
\]

实现层经验法则仍然是：

- 自旋/玻色子：`ratio = ComplexConjugate(psi_ex * inv_psi)`
- 费米子：局部计算 `psi`，并使用 `ratio = ComplexConjugate(psi_ex / psi)`

## 相关阅读

- 数学与约定：`../explanation/model_energy_solver_math.md`
- 内置模型（参考实现入口）：`../reference/model_observables_registry.md`


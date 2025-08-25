## 费米子（Z2 graded）VMC 梯度与 SR 推导（from scratch）

本笔记在 Z2 分级（fermionic, graded）张量网络的框架下，从零定义并推导变分蒙特卡洛（VMC）中能量梯度与 Stochastic Reconfiguration（SR，自然梯度）线性方程，给出与当前实现完全一致的复数统一形式。文档采用本项目与 TensorToolkit 的共同约定：

- 费米子约定：`|ket⟩` 对应张量索引方向 IN；`⟨bra|` 对应 OUT；`⟨bra|ket⟩` 的收缩不额外产生负号。所有由费米子交换引入的符号，均由分级结构与 `ActFermionPOps` 内部处理。
- 复数统一约定：对变分参数的最速下降/自然梯度方向取对共轭参数的导数（Wirtinger 风格），即以 \(\partial/\partial\theta_i^*\) 为基本导数。

记号与玻色子版文档 `sr-boson.md` 一致，差异仅在费米子分级与 `ActFermionPOps` 的角色与放置。

### 1. 设定与记号（Z2 graded）

- 变分波函数与抽样测度：
  \[
  |\Psi\rangle = \sum_{S} \Psi(S;\boldsymbol{\theta})\,|S\rangle,\quad
  p(S) = \frac{|\Psi(S)|^2}{Z},\; Z=\sum_S |\Psi(S)|^2.
  \]
- 局域能量（与实现一致，使用统一复数约定）：
  \[
  E_{\mathrm{loc}}(S)
  = \sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)}\, \langle S'|H|S\rangle.
  \]
- 对数导数算符（以共轭参数为自变量）：
  \[
  O_i^*(S) = \frac{\partial\, \ln \Psi^*(S;\theta_i^*)}{\partial\, \theta_i^*}.
  \]

上述定义在玻色与费米两类系统形式相同；费米子情形的“分级”只影响如何具体构造与收缩 \(O_i^*(S)\) 与内积，不改变公式形态。

### 2. 能量与统一梯度公式

能量期望值与对数导数梯度（按 \(\partial/\partial\theta_i^*\)）满足：
\[
E = \langle E_{\mathrm{loc}}(S) \rangle = \langle E_{\mathrm{loc}}^*(S) \rangle,
\]
\[
\frac{\partial E}{\partial \theta_i^*}
\;=\; \big\langle E_{\mathrm{loc}}^*\, O_i^* \big\rangle
\; -\; \big\langle E_{\mathrm{loc}}^* \big\rangle \big\langle O_i^* \big\rangle.
\]

实现细节（与代码一致）：梯度估计中乘以 \(E_{\mathrm{loc}}^*\)（而非 \(E_{\mathrm{loc}}\)），协方差扣项使用 \(E^*\)。

### 3. 费米子对数导数的构造（与实现一致）

- 玻色子时，采样配置 \(S\) 下有 \(O^*(S) = (\Psi^*(S))^{-1}\,\partial_{\theta^*}\Psi^*(S)\)。实现中以“洞”张量（`holes`）乘以 \((\Psi^*)^{-1}\) 得到。
- 费米子（Z2 graded）时，显式的 \((\Psi^*)^{-1}\) 与“洞”之直乘不再直接使用；实现采用分级安全的构造
  \[
  O^*(S) \;\equiv\; \mathrm{CalGTenForFermionicTensors}\big(\text{holes},\; \text{tn}\big),
  \]
  即以分级规则将“洞”张量与现场张量（`tn`）组合成正确的 \(O^*(S)\) 张量。

按照我们的 bra/ket 约定，`⟨bra|ket⟩` 的收缩不出负号；分级交换带来的符号由 `ActFermionPOps` 统一吸收，见 §6。

### 4. MC 估计器（与实现一致）

对 \(M\) 个采样 \(S^{(k)}\)：

- 样本平均：\( \overline{X} = M^{-1}\sum_k X(S^{(k)}) \)。
- 估计量：
  \[
  \widehat{S}_{ij} = \overline{O_i^* O_j} - \overline{O_i^*}\,\overline{O_j},\quad
  \widehat{F}_j = \overline{E_{\mathrm{loc}}^* O_j} - \overline{E_{\mathrm{loc}}^*}\,\overline{O_j}.
  \]

实现映射：每个样本累加 \(\sum O^*\) 与 \(\sum E_{\mathrm{loc}}^* O^*\)，最终取平均并作协方差扣项（用 \(E^*\)）。费米子情形下，样本的 \(O^*(S)\) 来自 `CalGTenForFermionicTensors`（§3），最后对得到的梯度整体调用一次 `ActFermionPOps`（§6）。

### 5. 精确求和估计器（与实现一致）

当体系可遍历：
\[
\langle X \rangle = \frac{\sum_S |\Psi(S)|^2 X(S)}{\sum_S |\Psi(S)|^2}.
\]
实现中以原始权重 \(w(S)=|\Psi(S)|^2\) 累加 \(S_O = \sum w\,O^*\)、\(S_{EO} = \sum w\,E_{\mathrm{loc}}^* O^*\) 与能量分子分母，最后同样计算 \(\nabla E = \frac{S_{EO} - E^* S_O}{\sum w}\)，并对最终梯度调用一次 `ActFermionPOps`（§6）。

### 6. `ActFermionPOps` 的数学角色

`ActFermionPOps` 是 TensorToolkit 中对“费米子奇偶算符”的一致实现。其核心作用是在“IN 方向”的腿上施加奇偶算符，使得：

- 在我们的约定下（IN=ket，OUT=bra），`⟨bra|ket⟩` 的收缩不再显式携带交换符号；所有因分级交换产生的 \((-1)\) 因子被折算为对 IN 腿的局部奇偶作用。
- 在线性代数层面，`ActFermionPOps` 是线性、对合（自反）的：\(P^2=I\)。实现上对每个数据块按 IN 索引之奇偶选择乘以 \(\pm 1\)。
- 该算符在内积与运算符作用中位置固定：
  - SplitIndexTPS 的“内积”实现为 \(\langle A, B\rangle = \sum \mathrm{Dag}(A) \cdot B\)，其中对 fermion 先对 \(\mathrm{Dag}(A)\) 调用一次 `ActFermionPOps` 再收缩，从而保证内积不显式携带交换号。
  - VMC 中，费米子样本的 \(O^*(S)\) 已按分级构造；样本聚合后，对“整体梯度张量场”调用一次 `ActFermionPOps`，即可得到与“无显式负号内积”约定匹配的梯度方向。

重要性质与放置（与实现一致）：

- 梯度聚合后：`gradient.ActFermionPOps()` 调用一次。
- SR 求解前后：对右端项先作用一次以进入“奇偶配准表象”，求解后再作用一次把结果拉回原表象（利用 \(P^2=I\)）。
- S 矩阵的乘子使用 `SplitIndexTPS::operator*` 的分级安全内积（其中对 `Dag(left)` 自动做一次 `ActFermionPOps`）。

### 7. SR（自然梯度）与线性方程

以对数导数算符之协方差为度量（Fisher 信息）：
\[
S_{ij} = \langle O_i^* O_j \rangle - \langle O_i^* \rangle\langle O_j \rangle,\quad
F_j = \langle E_{\mathrm{loc}}^* O_j \rangle - \langle E_{\mathrm{loc}}^* \rangle\langle O_j \rangle.
\]
自然梯度（或虚时演化投影）对应的步长 \(\delta\boldsymbol{\theta}\) 解线性方程：
\[
\big(S + \lambda I\big)\, \delta\boldsymbol{\theta} = -\, \alpha\, \boldsymbol{F},\quad \lambda\ge 0,\; \alpha>0.
\]

实现映射：

- S 矩阵在向量 \(v\) 上的作用以样本实现（省略 MPI 前缀）：
  \[
  (S\,v) \approx \overline{\,O^*\,(O^* v)\,} - (\overline{O^*}\,v)\,\overline{O^*} + \lambda v,
  \]
  其中内积与乘法均通过 `SplitIndexTPS` 的分级安全运算实现。
- 费米子：右端项进入 CG 前先 `ActFermionPOps()`，解出自然梯度后再 `ActFermionPOps()` 一次以回到原表象。

### 8. 与玻色子版公式的一致性与差异

- 公式层面：梯度与 SR 的所有公式与玻色子文档 `sr-boson.md` 完全同形（统一复数约定）。
- 差异仅体现在实现细节：
  - \(O^*(S)\) 的构造：玻色子为 \((\Psi^*)^{-1}\,\text{holes}\)；费米子为分级安全的 `CalGTenForFermionicTensors(holes, tn)`。
  - `ActFermionPOps` 的位置：费米子在“整体梯度”处与 SR 前后各一次；S 矩阵的内积内部对 Dag(left) 亦隐式调用一次。
  - 这些放置确保“`⟨bra|ket⟩` 收缩不出负号”的全局约定被严格满足。

### 9. 与当前代码的逐点映射（关键片段）

- 采样与梯度样本累计（费米子构造 \(O^*\) + 使用 \(E_{\mathrm{loc}}^*\)）：见 `vmc_peps_optimizer_impl.h` 中 `SampleEnergyAndHoles_()` 与统计聚合：

```333:376:include/qlpeps/algorithm/vmc_update/vmc_peps_optimizer_impl.h
// ∂E/∂θ* = <E_loc^* O*> − E^* <O*>
if constexpr (Tensor::IsFermionic()) {
  Ostar_tensor = CalGTenForFermionicTensors(holes({row, col}), this->tps_sample_.tn({row, col}));
} else {
  Ostar_tensor = inverse_amplitude * holes({row, col});
}
ELocConj_Ostar_sum_({row, col})[basis_index] += local_energy_conjugate * Ostar_tensor;
```

```421:461:include/qlpeps/algorithm/vmc_update/vmc_peps_optimizer_impl.h
Ostar_mean_ = Ostar_sum_ * (1.0 / sample_num);
grad_ = ELocConj_Ostar_sum_ * (1.0 / sample_num) + ComplexConjugate(-energy) * Ostar_mean_;
grad_.ActFermionPOps();
```

- 精确求和后对梯度一次性施加奇偶算符：

```151:219:include/qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h
SplitIndexTPSType gradient = (ELocConj_Ostar_weighted_sum - ComplexConjugate(energy) * Ostar_weighted_sum) * (1.0 / weight_sum);
if constexpr (Index<QNT>::IsFermionic()) {
  gradient.ActFermionPOps();
}
```

- SR：S 矩阵乘子与“前后各一次”的奇偶作用：

```16:51:include/qlpeps/optimizer/stochastic_reconfiguration_smatrix.h
// S·v ≈ mean_i [ O*_i (O*_i v) ] − (O*_mean v) O*_mean + diag_shift v
```

```637:649:include/qlpeps/optimizer/optimizer_impl.h
auto signed_gradient = gradient; signed_gradient.ActFermionPOps();
natural_gradient = ConjugateGradientSolver(s_matrix, signed_gradient, ...);
natural_gradient.ActFermionPOps();
```

- `SplitIndexTPS` 的分级安全内积（对 Dag(left) 施加一次 `ActFermionPOps` 后再收缩）：

```338:353:include/qlpeps/two_dim_tn/tps/split_index_tps.h
Tensor ten_dag = Dag(ten);
if constexpr (Tensor::IsFermionic()) {
  ten_dag.ActFermionPOps();
  Contract(&ten_dag, {0,1,2,3,4}, &right({row, col})[i], {0,1,2,3,4}, &scalar);
}
```

### 10. 结论

在 Z2 分级的 TN/VMC 框架中，SR 与梯度的“公式层”与玻色子完全一致；差别仅在“如何构造 \(O^*\) 与如何在张量层实现分级安全的内积”。通过在恰当位置应用 `ActFermionPOps`（一次于内积左矢、一次于整体梯度、SR 前后各一次），我们与“IN=ket、OUT=bra、`⟨bra|ket⟩` 收缩不生负号”的全局约定保持严谨一致，并与当前代码实现一一对应。



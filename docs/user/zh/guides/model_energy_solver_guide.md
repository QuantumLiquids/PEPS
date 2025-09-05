# 模型能量求解器 (Model Energy Solver) 指南

## 概念：能量和梯度计算引擎

模型能量求解器计算特定粒子配置和TPS状态的**局域能量和梯度信息**。它封装了所有模型特定的哈密顿量细节。

## 为什么重要

VMC优化需要：
- **能量样本**：$E_{\text{loc}}$ 对每个采样配置
- **梯度样本**：$\frac{\partial \ln|\psi(S)|}{\partial \theta_i}$ (对数导数) 

能量样本和梯度样本除了用来计算能量和梯度，也对SR的计算有作用。

能量求解器将所有复杂的张量网络收缩隐藏在干净的接口后面。

### 复数波函数与局域能量

我们的实现统一处理复数波函数情况。局域能量计算涉及复共轭操作，本节详细说明实现方法。对于实数情况，我们采用相同的代码框架，这种统一性在下面的基础说明中会变得清晰。

#### 复数梯度的基础概念

为了理解我们的实现选择，先看一个简单例子：

**实数情况**：设 $f(x) = x^2$，其中 $x$ 是实数，则 $f'(x) = 2x$。

**复数情况**：设 $f(z, z^*) = |z|^2 = z^* z$，其中 $z$ 是复数。根据微积分，最速下降方向是 $\frac{\partial f}{\partial z^*} = z$的反方向，而不是对 $z$ 求导。

**Mismatch**：$\frac{\partial f}{\partial z^*} = z$ 与实数情况的 $f'(x) = 2x$ 相差因子2。

**统一约定**：为保持代码的一致性，我们在整个框架中都遵循复数的微积分约定。即使对于实数参数，我们也将其视为复数的特殊情况来处理。即对于Real number不计入因子2在梯度计算中。这种统一的处理方式简化了实现，避免了针对不同数据类型的分支逻辑。

#### 数学基础

对于复数波函数，能量期望值必须是实数：
\[
E = \frac{\langle \Psi| H|\Psi\rangle}{\langle \Psi| \Psi\rangle}  = \frac{\sum_S |\Psi(S)|^2 E_{\mathrm{loc}}(S)}{\sum_S |\Psi(S)|^2} = \langle E_{\mathrm{loc}}(S) \rangle
\]

#### 局域能量的定义

在我们的实现中，局域能量定义为：
\[
E_{\mathrm{loc}}(S) = \sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)} \langle S'| H | S\rangle
\]

这个定义的关键点是需要对波函数振幅的比值做**复共轭**。我们的实现将这项**复共轭**的责任放在能量求解器内部，而非外部的变分蒙特卡洛 PEPS 优化器组合。其中函数 `ComplexConjugate` 可辅助用户扩展自定义模型；关于最近邻/次近邻键能的幅度比实现，见《自定义能量求解器开发指南》中 EvaluateBondEnergy/EvaluateNNNEnergy。

优化器日志中输出的能量 $E = \langle E_{\mathrm{loc}} \rangle$ 按上述定义计算（尤其在复数情形）。

### 梯度计算的数学定义

当参数 $\theta_i$ 是复数时，根据复变函数理论，我们需要对 $\theta_i^*$ 求导来获得最速下降方向。

下面公式请读者自行推导。

代码中采用的**最终梯度公式**：
\[
\frac{\partial E}{\partial \theta_i^*} = \langle E_{\mathrm{loc}}^* \cdot O_i^* \rangle - \langle E_{\mathrm{loc}}^* \rangle \langle O_i^* \rangle
\]

其中 $O_i^* = \frac{\partial \ln \Psi^*(\theta_i^*)}{\partial \theta_i^*}$ 是对数导数。

#### 代码实现要点
梯度计算公式中$\langle E_{\mathrm{loc}}^* \rangle$是能量期望值，看似平平无奇，实则暗藏玄机。选$\langle E_{\mathrm{loc}}^* \rangle$还是$\langle E_{\mathrm{loc}} \rangle$还是$\langle Re(E_{\mathrm{loc}}) \rangle$对SR的数值稳定性有影响。根据在下的经验选择的是$\langle E_{\mathrm{loc}}^* \rangle$。

## 基础接口定义
模型求解器是一个定义有以下成员函数签名的类
```cpp
template<typename TenElemT, typename QNT, bool calchols>
TenElemT CalEnergyAndHoles(
    const SplitIndexTPS<TenElemT, QNT>* sitps,           // 输入：代表波函数 |Psi>的split-index TPS的指针
    TPSWaveFunctionComponent<TenElemT, QNT>* tps_sample, // 输入：指向当前蒙特卡洛样本的指针，包含组态、组态对应的的单层张量网络、和波函数振幅的信息
         TensorNetwork2D<TenElemT, QNT>& hole_res             // 输出：梯度"洞"张量，已经过复共轭处理（Dag操作）
);
```
其中
- `sitps`: 代表波函数 |Psi> 的split-index TPS指针
- `tps_sample`: 当前蒙特卡洛样本，包含配置、单层张量网络和波函数振幅
- `hole_res`: 输出参数，存储梯度"洞"张量，**在能量求解器内部已经进行了复共轭操作**（`Dag`函数），将在外部直接用于梯度计算
- `calchols`: 布尔型模板参数。在模型求解器内部实现中用`calchols`的条件分支括住梯度的计算部分。使得当`true`时同时计算梯度和能量，`false`表示只计算能量。当外部优化器只需要能量而不需要梯度时，可以跳过梯度计算节省时间。作为模版参数，在函数实现时可以用constexpr声明以减少分支预测提高程序性能。

**返回值**：
- `TenElemT`: 当前配置的局域能量 $E_{\mathrm{loc}}(S)$，类型和张量网络的数据类型一致，即可能是复数

### hole_res 的数学定义（玻色子）

对每个组态 $S$ 与每个站点（以及其物理基底索引）定义对应的局部参数 $\theta_i$（等价为局部张量元 $A_{\text{site},\text{basis}}$）。在玻色子情况下，能量求解器返回的
$\texttt{hole\_res}$ 满足：
\[
  (\text{hole\_res})_i(S) 
  \;\equiv\; \frac{\partial \, \Psi^*(S)}{\partial \, \theta_i^*}
  \;=\; \frac{\partial \, \Psi^*(S)}{\partial \, A_{\text{site},\text{basis}}^*} .
\]
这与对数导数算符 $O_i^*(S)$ 的关系为：
\[
  O_i^*(S) 
  \,=\, \frac{\partial \, \ln \Psi^*(S)}{\partial \, \theta_i^*}
  \,=\, \frac{1}{\Psi^*(S)}\, (\text{hole\_res})_i(S) .
\]
因此：
- 在 MC 采样路径中使用 $O_i^*(S)$ 时，会计算 $\frac{1}{\Psi^*(S)}\,(\text{hole\_res})_i(S)$；
- 在“精确求和”路径中累加 $\langle O_i^* \rangle$ 的加权和时，利用
  \[ \sum_S |\Psi(S)|^2\, O_i^*(S) = \sum_S \Psi(S)\, (\text{hole\_res})_i(S), \]
  即直接将 $\Psi(S)$ 与 $\text{hole\_res}$ 相乘再求和，可避免显式除法与数值不稳定。

实现提示：在代码中 `hole_res(site)` 由 `Dag(tn.PunchHole(site, ...))` 得到，正对应
$\partial \Psi^*(S)/\partial \theta_i^*$ 的“洞”张量；后续按照上式进行规一化或加权求和。

注：费米子情形由于奇偶算符与符号的参与更为复杂，代码中通过 `EvaluateLocalPsiPartialPsiDag` 与最终的 `gradient.ActFermionPOps()` 处理。本文档此处先给出玻色子精确定义，费米子细节将在专门章节讨论。

## 用法
在VMCPEPSOptimizer中作为最后一个模版参数EnergySolver传入，并在构造函数中，传入其具体的对象。其对象可能包含物理模型的参数。

## 一般约定：
位于：`include/qlpeps/algorithm/vmc_update/model_energy_solver.h`
中我们有class ModelEnergySolver的CRTP base class. 用户可以以CTRP方式继承这一base class，并且必须实现 `CalEnergyAndHolesImpl()`的方法。这一方法返回参数列表中多了一个
psi_list
用于收集在计算能量时，boundary MPS收缩到不同位置时候计算的波函数amplitude的值。原则上，只有当这一数值时一致的，波函数才是well-defined. 但是由于二维张量网络收缩时截断误差的引入，boundary MPS收缩到不同行列时波函数amplitude的值会有不一致性。ModelEnergySolver基类会检查这一一致性并在误差较大时报warning给标准输出，提醒用户在必要时提高boundary MPS收缩的虚拟指标的维度(bond dimension)。


## 内置能量求解器

位于：`include/qlpeps/algorithm/vmc_update/model_solvers/`

### 自旋/玻色子模型

#### 1. 方格XXZ模型
**头文件**：`#include "qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_xxz_model.h"`

**类名**：`SquareSpinOneHalfXXZModel`

**构造函数**：
- `SquareSpinOneHalfXXZModel()` - 各向同性海森堡模型，J=1，无外场
- `SquareSpinOneHalfXXZModel(double jz, double jxy, double pinning00)`
  - `jz`：Ising相互作用强度，控制 $S^z_i S^z_j$ 项
  - `jxy`：XY相互作用强度，控制 $(S^x_i S^x_j + S^y_i S^y_j)$ 项  
  - `pinning00`：角落磁场强度，作用于 $(0,0)$ 位置

**哈密顿量**：$$H = \sum_{\langle i,j \rangle} (J_z S^z_i S^z_j + J_{xy} (S^x_i S^x_j + S^y_i S^y_j)) - h_{00} S^z_{00}$$

#### 2. 方格J1-J2 XXZ模型
**头文件**：`#include "qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_j1j2_xxz_model.h"`

**类名**：`SquareSpinOneHalfJ1J2XXZModel`

**构造函数**：
- `SquareSpinOneHalfJ1J2XXZModel(double j2)` - J1-J2海森堡模型，J1=1
- `SquareSpinOneHalfJ1J2XXZModel(double jz, double jxy, double jz2, double jxy2, double pinning_field00)`
  - `jz`, `jxy`：最近邻Ising和XY相互作用强度
  - `jz2`, `jxy2`：次近邻Ising和XY相互作用强度
  - `pinning_field00`：角落磁场强度

**哈密顿量**：$$H = \sum_{\langle i,j \rangle} (J_{z1} S^z_i S^z_j + J_{xy1} (S^x_i S^x_j + S^y_i S^y_j)) + \sum_{\langle\langle i,j \rangle\rangle} (J_{z2} S^z_i S^z_j + J_{xy2} (S^x_i S^x_j + S^y_i S^y_j))$$

**注意**：当J2=0时，此模型在数值上等价于XXZ模型，但计算效率不同。

#### 3. 三角格点海森堡模型
**头文件**：`#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_triangle_heisenberg_sqrpeps.h"`

**类名**：`SpinOneHalfTriHeisenbergSqrPEPS`

**构造函数**：
- `SpinOneHalfTriHeisenbergSqrPEPS()` - 默认构造，J=1

**哈密顿量**：$$H = J \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j$$
**特性**：三角格点几何，包含对角键，几何阻挫

#### 4. 三角格点J1-J2海森堡模型
**头文件**：`#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_triangle_heisenbergJ1J2_sqrpeps.h"`

**类名**：`SpinOneHalfTriJ1J2HeisenbergSqrPEPS`

**构造函数**：
- `SpinOneHalfTriJ1J2HeisenbergSqrPEPS(double j2)`
  - `j2`：次近邻相互作用强度，J1固定为1

**哈密顿量**：$$H = J_1 \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j + J_2 \sum_{\langle\langle i,j \rangle\rangle} \vec{S}_i \cdot \vec{S}_j$$

#### 5. 横场伊辛模型
**头文件**：`#include "qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square.h"`

**类名**：`TransverseFieldIsingSquare`

**构造函数**：
- `TransverseFieldIsingSquare(double h)`
  - `h`：横向磁场强度，控制量子涨落与经典序参量的竞争

**哈密顿量**：$$H = -\sum_{\langle i,j \rangle} \sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i$$

### 费米子模型

#### 6. t-J模型系列
**头文件**：`#include "qlpeps/algorithm/vmc_update/model_solvers/square_tJ_model.h"`

##### 6.1 最近邻t-J模型
**类名**：`SquaretJNNModel`

**构造函数**：
- `SquaretJNNModel(double t, double J, double mu)`
  - `t`：近邻跳跃积分强度
  - `J`：反铁磁交换相互作用强度  
  - `mu`：化学势，控制电子填充

**哈密顿量**：$$H = -t\sum_{\langle i,j\rangle,\sigma} (c_{i,\sigma}^\dagger c_{j,\sigma} + h.c.) + J \sum_{\langle i,j\rangle} (\vec{S}_i \cdot \vec{S}_j - \frac{1}{4} n_i n_j) - \mu \sum_{i,\sigma} n_{i,\sigma}$$

**特性**：经典t-J模型，无双占据约束

##### 6.2 次近邻t-J模型
**类名**：`SquaretJNNNModel`

**构造函数**：
- `SquaretJNNNModel(double t, double t2, double J, double mu)`
  - `t`：近邻跳跃积分强度
  - `t2`：次近邻跳跃积分强度
  - `J`：反铁磁交换相互作用强度
  - `mu`：化学势，控制电子填充

**哈密顿量**：$$H = -t\sum_{\langle i,j\rangle,\sigma} (c_{i,\sigma}^\dagger c_{j,\sigma} + h.c.) -t_2\sum_{\langle\langle i,j\rangle\rangle,\sigma} (c_{i,\sigma}^\dagger c_{j,\sigma} + h.c.) + J \sum_{\langle i,j\rangle} (\vec{S}_i \cdot \vec{S}_j - \frac{1}{4} n_i n_j) - \mu \sum_{i,\sigma} n_{i,\sigma}$$

**特性**：包含次近邻跳跃，更精确描述铜氧化物等材料的电子结构

##### 6.3 t-J-V模型
**类名**：`SquaretJVModel`

**构造函数**：
- `SquaretJVModel(double t, double t2, double J, double V, double mu)`
  - `t`：近邻跳跃积分强度
  - `t2`：次近邻跳跃积分强度
  - `J`：反铁磁交换相互作用强度
  - `V`：近邻密度相互作用强度
  - `mu`：化学势，控制电子填充

**哈密顿量**：$$H = -t\sum_{\langle i,j\rangle,\sigma} (c_{i,\sigma}^\dagger c_{j,\sigma} + h.c.) -t_2\sum_{\langle\langle i,j\rangle\rangle,\sigma} (c_{i,\sigma}^\dagger c_{j,\sigma} + h.c.) + J \sum_{\langle i,j\rangle} (\vec{S}_i \cdot \vec{S}_j - \frac{1}{4} n_i n_j) + V \sum_{\langle i,j\rangle} n_i n_j - \mu \sum_{i,\sigma} n_{i,\sigma}$$

**特性**：
- 当V=J/4时，V项精确抵消J项中的 $-\frac{1}{4}n_i n_j$ 部分，简化为纯自旋交换模型

#### 7. Hubbard模型
**头文件**：`#include "qlpeps/algorithm/vmc_update/model_solvers/square_hubbard_model.h"`

**类名**：`SquareHubbardModel`

**构造函数**：
- `SquareHubbardModel(double t, double U, double mu)`
  - `t`：近邻跳跃积分强度
  - `U`：在位库仑排斥能，U>>t时为Mott绝缘体
  - `mu`：化学势，控制电子密度

**哈密顿量**：$$H = -t \sum_{\langle i,j \rangle, \sigma} (c_{i,\sigma}^\dagger c_{j,\sigma} + h.c.) + U \sum_i n_{i,\uparrow} n_{i,\downarrow} - \mu \sum_{i,\sigma} n_{i,\sigma}$$

#### 8. 无自旋费米子模型
**头文件**：`#include "qlpeps/algorithm/vmc_update/model_solvers/square_spinless_fermion.h"`

**类名**：`SquareSpinlessFermion`

**构造函数**：
- `SquareSpinlessFermion(double t, double V)` - 仅最近邻跳跃
- `SquareSpinlessFermion(double t, double t2, double V)` - 含次近邻跳跃
  - `t`：近邻跳跃积分强度
  - `t2`：次近邻跳跃积分强度
  - `V`：近邻密度排斥相互作用强度

**哈密顿量**：$$H = -t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + h.c.) - t_2 \sum_{\langle\langle i,j \rangle\rangle} (c_i^\dagger c_j + h.c.) + V \sum_{\langle i,j \rangle} n_i n_j$$



## 自定义能量求解器开发

对于需要实现新的物理模型，PEPS框架提供了灵活的自定义能量求解器接口。

详细的开发指南请参阅：**[自定义能量求解器开发指南](CUSTOM_ENERGY_SOLVER_GUIDE.md)**

该指南包含：
- **基础用法**：最近邻(NN)模型开发
- **中级用法**：次近邻(NNN)模型开发  
- **进阶用法**：完全自定义模型开发
- **完整代码示例**和最佳实践

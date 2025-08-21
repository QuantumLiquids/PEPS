## 玻色子 SR 推导（from scratch）

本笔记从零推导变分蒙特卡洛（VMC）中针对玻色子（或等价的无费米符号问题）波函数的 Stochastic Reconfiguration（SR）方法，给出能量梯度、对数导数与 SR 线性方程的统一复数形式，以及蒙特卡洛与“精确求和”两种估计器的一致表述。记号与实现约定与 `docs/tutorial_cn/MODEL_ENERGY_SOLVER_GUIDE.md` 保持一致。

### 1. 设定与记号

- 变分波函数：\( |\Psi\rangle = \sum_{S} \Psi(S;\boldsymbol{\theta}) \, |S\rangle \)。参数 \(\boldsymbol{\theta} = (\theta_1,\theta_2,\dots)\) 允许为复数。
- 采样测度：\( p(S) = |\Psi(S)|^2 / Z \)，其中 \( Z = \sum_S |\Psi(S)|^2 \)。
- 局域能量（统一复数约定，参见能量求解器文档）：
\[
E_{\mathrm{loc}}(S) \,=\, \sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)} \, \langle S'| H | S\rangle .
\]
- 对数导数算符（以复数共轭参数为自变量，保证最速下降方向一致性）：
\[
O_i^*(S) \;=\; \frac{\partial \, \ln \Psi^*(S;\theta_i^*)}{\partial \, \theta_i^*}
\quad (\text{玻色子情形与一般复数情形一致}).
\]

为简洁起见，后续期望 \(\langle \cdot \rangle\) 均指对测度 \(p(S)\) 的统计平均。

### 2. 能量与统一梯度公式

能量期望值满足
\[
E \,=\, \frac{\langle \Psi| H|\Psi\rangle}{\langle \Psi| \Psi\rangle}
 \,=\, \langle E_{\mathrm{loc}}(S) \rangle \,=\, \langle E_{\mathrm{loc}}^*(S) \rangle .
\]

对复参量 \(\theta_i\) 的最速下降方向需取 \(\partial/\partial\theta_i^*\)。严格推导给出统一梯度（与实现一致）：
\[
\frac{\partial E}{\partial \theta_i^*}
\;=\; \big\langle E_{\mathrm{loc}}^* \, O_i^* \big\rangle
\; - \; \big\langle E_{\mathrm{loc}}^* \big\rangle \, \big\langle O_i^* \big\rangle .
\]

当参数实值时，可将 \(\theta_i\) 与 \(\theta_i^*\) 视作独立坐标的等价写法；本项目统一采用复数约定，避免分支与额外因子问题。


### 3. SR 的变分原理与线性方程

SR 可由“自然梯度/虚时演化”的二次型投影得到：在小步长 \(\delta\boldsymbol{\theta}\) 下，以费舍尔度量（由对数导数的协方差给出）约束下行进能量最速下降方向。定义
\[
\Delta O_i^*(S) = O_i^*(S) - \langle O_i^* \rangle .
\]
度量矩阵（协方差）与右端项为
\[
S_{ij} \,=\, \big\langle \Delta O_i^* \, \Delta O_j \big\rangle
\;=\; \big\langle O_i^* O_j \big\rangle - \big\langle O_i^* \big\rangle \big\langle O_j \big\rangle ,
\]
\[
F_j \,=\, \big\langle E_{\mathrm{loc}}^* \, \Delta O_j \big\rangle
\;=\; \big\langle E_{\mathrm{loc}}^* O_j \big\rangle - \big\langle E_{\mathrm{loc}}^* \big\rangle \big\langle O_j \big\rangle .
\]
SR 步进由线性方程给出：
\[
\big(S + \lambda I\big) \, \delta\boldsymbol{\theta} \;=\; - \, \alpha \, \boldsymbol{F} ,
\]
其中 \(\lambda \ge 0\) 为Tikhonov正则（或对角 shift）以增强数值稳定性，\(\alpha>0\) 为学习率。矩阵 \(S\) 厄米半正定，玻色子与一般复数波函数情形均成立。

实现上推荐使用共轭梯度（CG）或最小残量法（MINRES）求解该厄米系统；当 \(S\) 病态时，需适当调大 \(\lambda\)。

### 4. 蒙特卡洛估计器（MC 路径）

在 VMC 采样得到样本 \(\{S^{(k)}\}_{k=1}^M\) 后：

- 经验均值：\( \langle X \rangle \approx M^{-1} \sum_k X(S^{(k)}) \)。
- 估计量：
  - \( \widehat{S}_{ij} = \overline{O_i^* O_j} - \overline{O_i^*} \, \overline{O_j} \)。
  - \( \widehat{F}_j = \overline{E_{\mathrm{loc}}^* O_j} - \overline{E_{\mathrm{loc}}^*} \, \overline{O_j} \)。
  - 其中横线表示样本平均。为降低方差，建议对 \(O\) 做居中（即使用 \(\Delta O\)）。

数值细节：
- 能量平均推荐使用 \(\langle E_{\mathrm{loc}}^* \rangle\)（经验更稳），与梯度/右端项定义保持自洽。
- 计算 \(O_i^*\) 可借助“洞”张量（见 §6）避免显式写出 \((\Psi^*)^{-1}\) 的除法，但这只是等价重排，并不自动带来数值稳定性提升。

### 5. 精确求和估计器（Exact Summation 路径）

当体系尺寸较小或可遍历全部组态 \(S\) 时，用全和替代采样平均：
\[
\langle X \rangle \,=\, \frac{\sum_S |\Psi(S)|^2 \, X(S)}{\sum_S |\Psi(S)|^2} .
\]
在实现中，若能直接得到 \(\Psi(S)\) 与“洞”张量 \(\mathrm{hole\_res}_i(S)\equiv \partial \Psi^*(S)/\partial \theta_i^*\)，可避免显式除法（等价重排，非“更稳定”的保证）。定义原始权重 \(w(S)=|\Psi(S)|^2\) 及以下累加量（玻色子情形）：
\[
W = \sum_S w(S),\quad E_{\text{num}} = \sum_S w(S)\,E_{\mathrm{loc}}(S),\quad E = \frac{E_{\text{num}}}{W}.
\]
对数导数的“加权和”与二阶项：
\[
S_{O,i} = \sum_S w(S)\, O_i^*(S) = \sum_S \Psi(S)\, \mathrm{hole\_res}_i(S),
\]
\[
S_{O,j}^{\,\text{(non\,conj)}} = \sum_S w(S)\, O_j(S) = \sum_S \Psi^*(S)\, \mathrm{hole\_res}_j(S)^*,
\]
\[
S_{OO,ij} = \sum_S w(S)\, O_i^*(S) O_j(S) = \sum_S \mathrm{hole\_res}_i(S)\, \mathrm{hole\_res}_j(S)^*.
\]
右端项：
\[
S_{EO,j} = \sum_S w(S)\, E_{\mathrm{loc}}^*(S)\, O_j(S) = \sum_S E_{\mathrm{loc}}^*(S)\, \Psi^*(S)\, \mathrm{hole\_res}_j(S)^*.
\]
则有
\[
\langle O_i^* \rangle = \frac{S_{O,i}}{W},\qquad \langle O_j \rangle = \frac{S_{O,j}^{\,\text{(non\,conj)}}}{W},\qquad
\langle O_i^* O_j \rangle = \frac{S_{OO,ij}}{W},\qquad \langle E_{\mathrm{loc}}^* O_j \rangle = \frac{S_{EO,j}}{W}.
\]
从而 SR 的精确求和配方为
\[
S_{ij} = \langle O_i^* O_j \rangle - \langle O_i^* \rangle\, \langle O_j \rangle,\qquad
F_j = \langle E_{\mathrm{loc}}^* O_j \rangle - \langle E_{\mathrm{loc}}^* \rangle\, \langle O_j \rangle,
\]
并解 \( (S+\lambda I)\,\delta\boldsymbol{\theta} = -\alpha\,\boldsymbol{F} \)。上述表达与 MC 配方完全同形，仅以全和替代样本平均；当扩展到费米子时需加入奇偶与算符次序修正（另文给出）。




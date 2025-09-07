### RFC: MC 能量标准差/标准误差计算与并行聚合方案（含自相关修正）

- title: MC energy stddev/SE and parallel aggregation (IID + autocorrelation)
- status: Draft
- last_updated: 2025-08-27
- tags: vmc, statistics, mpi, mcmc, variance, IAT
- applies_to: vmc_update, optimizer, vmc_basic

#### 背景
现状问题：
- 单进程下能量标准误返回为无穷大，测试与数值分析不便。
- 多进程误差计算基于“各进程均值”的方差，未纳入进程内样本方差与自相关，统计意义不足且可能低估不确定性。

#### 目标
1) 提供在单/多进程统一且正确的“全样本”均值与标准误（SE）计算。
2) 默认采用“短窗初始正序列（IPS）的方差膨胀因子 g”进行自相关修正；当 g≈1 时自动退化为 IID-SE；保留 blocking 作为可选互证。

---

### 数学基础

设能量样本为 \(\{E_t\}_{t=1}^N\)。

#### 符号与定义
- 均值：\(\mu = \mathbb{E}[E_t]\)，样本均值 \(\bar E = \frac{1}{N}\sum_{t=1}^{N} E_t\)。
- 自协方差（理论）：\(\Gamma_k = \operatorname{Cov}(E_t, E_{t+k}) = \mathbb{E}[(E_t-\mu)(E_{t+k}-\mu)]\)。
- 自协方差（样本估计）：\(\hat{\Gamma}_k = \frac{1}{N}\sum_{t=1}^{N-k}(E_t-\bar E)(E_{t+k}-\bar E)\)。
- 归一化自相关系数：\(\rho_k = \Gamma_k / \Gamma_0\)；样本 \(\hat\rho_k = \hat\Gamma_k / \hat\Gamma_0\)。
- 初始正序列准则量：\(\gamma_k = \Gamma_k + \Gamma_{k+1}\)；样本 \(\hat\gamma_k = \hat\Gamma_k + \hat\Gamma_{k+1}\)。IPS（Initial Positive Sequence）按 \(\hat\gamma_k>0\) 的最长正序窗口截断求和。
- 无偏样本方差：\(s^2 = \frac{1}{N-1}\sum_{t=1}^{N}(E_t-\bar E)^2\)。
- 方差膨胀因子（短窗/IPS 截断）：\(g = 1 + 2\sum_{k=1}^{K} \rho_k^{(+)}\)，其中 \(\rho_k^{(+)}\) 表示仅在 IPS 窗口内累计（或取非负段）。
- 复数情形：若 \(E_t\in\mathbb{C}\)，用于误差估计的统计量（如 \(s^2, \Gamma_k, \rho_k\)）基于 \(\operatorname{Re} E_t\) 计算；用于梯度/SR 的均值 \(\bar E\) 保留复数。

1) IID 情形
- 样本均值：\(\bar E = \frac{1}{N}\sum_{t=1}^N E_t\)
- 无偏方差：\(s^2 = \frac{1}{N-1}\sum_{t=1}^N (E_t - \bar E)^2\)
- 标准误：\(\operatorname{SE}_{\text{IID}}(\bar E) = \sqrt{\frac{s^2}{N}}\)

2) MCMC 自相关修正（方差膨胀因子法 + IPS 截断）
令归一化自相关系数 \(\rho_k\)，定义方差膨胀因子
\[ g = 1 + 2\sum_{k=1}^{K} \rho_k^{(+)} \]
其中 \(\rho_k^{(+)}\) 只在“初始正序列”（Geyer/Sokal）窗口内累加；K 可取固定小窗（如 10–20），或用 \(\gamma_k=\Gamma_k+\Gamma_{k+1}\) 首次非正处截断（IPS）。
于是
\[ \operatorname{SE}(\bar E) \approx \sqrt{\frac{s^2}{N}\, g}. \]
当 \(g \to 1\) 时退化为 IID-SE；当相关较强时 \(g>1\) 给出保守修正。

实践中可采用：
- 方差膨胀因子 g（IPS 截断），并行实现与通信最轻量。
- 或 blocking/batch means：将序列分为 \(B\) 个等长区块，\(\bar E_b\) 为每块均值。则
  \[ \widehat{\operatorname{Var}}(\bar E) \approx \frac{1}{B(B-1)} \sum_{b=1}^B (\bar E_b - \bar E)^2, \quad \operatorname{SE} = \sqrt{\widehat{\operatorname{Var}}(\bar E)}. \]

---

### 并行聚合（MPI）

在每个 rank 上，计算本地统计三元组：
- \(n_i = \) 本地样本数
- \(S_i = \sum_t E_t\)
- \(Q_i = \sum_t E_t^2\)

用 MPI_Allreduce 求全局：\(N = \sum_i n_i\), \(S = \sum_i S_i\), \(Q = \sum_i Q_i\)。

得到：
- 全局均值：\(\bar E = S / N\)
- 无偏方差：\(s^2 = \frac{Q - N\,\bar E^2}{N-1}\)（当 \(N>1\)）
- 标准误（IID）：\(\operatorname{SE}_{\text{IID}} = \sqrt{ s^2 / N }\)

数值稳定：推荐用 long double 累加，减少抵消误差；复数能量取实部统计（物理期望为实）。

#### 自相关修正的并行聚合（多链友好，推荐默认开启）
对每条链 i，本地估计 \(s_i^2\)、\(n_i\) 与方差膨胀因子 \(g_i\)（短窗 IPS）。
则该链均值的方差近似为
\[ \operatorname{Var}_i(\bar E_i) \approx \frac{s_i^2}{n_i}\, g_i. \]
全局均值 \(\bar E = \sum_i (n_i/N)\,\bar E_i\) 的方差可加权合并：
\[ \operatorname{Var}(\bar E) \approx \sum_i \left(\frac{n_i}{N}\right)^2 \operatorname{Var}_i(\bar E_i)
  = \frac{1}{N^2}\sum_i g_i\, s_i^2\, n_i, \quad \operatorname{SE}=\sqrt{\operatorname{Var}(\bar E)}. \]
通信只需 Allreduce 两个标量：\(\sum_i g_i s_i^2 n_i\) 与 \(N\)。

---

### 设计与接口

1) 计算路径
- 在 `MCEnergyGradEvaluator::Evaluate` 中收集 `energy_samples`。
- 单/多进程统一：
  - 必选：在所有 rank 计算本地 \(n_i, S_i, Q_i\)，Allreduce 得到 \(N,S,Q\)，计算 \(\bar E, s^2\)。
  - 默认：本地估短窗 IPS 的 \(g_i\)，按 \(\sum_i g_i s_i^2 n_i\) 与 \(N\) 聚合得到全局 SE。
  - 兜底：若 \(g\le 1.1\)（或本地估计不稳），退化为 IID-SE。
- 结果：
  - `Result::energy`：全局均值（TenElemT，可保留复数但实部来自统计）。
  - `Result::energy_error`：`SE = max(SE_IID, SE_g)`，其中 `SE_g` 由上式给出。

2) 自相关修正（可选互证）
- blocking/batch means/OBM：各 rank 形成本地 block 均值向量，或直接输出 `Var_i(\bar E_i)`；
  根/Allreduce 以 \((n_i/N)^2\) 做加权合并，得到全局 SE，作为 IPS 的交叉验证。

3) 类型与数值注意
- 对复数 `TenElemT`：E的统计保持complex，有利于后续Grad/SR稳定；最终 `Result::energy` 保持原类型。
- 使用 `long double` 存储 `S_i`、`Q_i` 与 Allreduce，减小舍入误差。(这里可能还需要考虑改进)
 - ρ_k 估计用 FFT 自协方差与窗函数（或直接 \(\gamma_k\) 正序准则）；不做 thinning。

#### 复数处理与 SR 稳定性
- 能量统计：用于均值/方差/SE 的样本序列取 \(\operatorname{Re} E_t\)（物理期望为实），从而 `energy_error` 基于实部；`Result::energy` 仍保留 `TenElemT` 原类型（含可能极小的虚部以便溯源）。

---

### 变更影响
- 单进程：不再返回 `inf`，得到有限标准误，测试阈值可设为 `k_sigma * SE`。
- 多进程：由“各进程均值的方差”改为“全样本标准误”，更符合统计意义。
- 行为：此变更为“hard 修改”，会影响现有误差数值与部分测试门限。

---

### 迁移与测试
- 将现有测试中的经验容忍度替换为：`tol = max(tol_min, k_sigma * energy_error)`，建议 `k_sigma = 3`。
- 新增单测：
  - 对合成 IID 数据验证 Allreduce 结果与单机一致（g≈1 时与 IID-SE 一致）。
  - 对 AR(1)/短相关数据验证短窗 IPS 得到 g≈1–1.2 区间，SE 不被低估。
  - 对中等相关数据（g 较大）验证 IPS 与 blocking 平台法一致性。
 - 集成测试：固定种子，比较旧/新误差行为并更新基线。

---

### 实施步骤
1) 在 `MCEnergyGradEvaluator::Evaluate` 内实现 `N/S/Q` 全局规约与 `SE_IID` 兜底输出。
2) 在每个 rank 本地估计 `g_i`（短窗 IPS），Allreduce 聚合 `sum(g_i*s_i^2*n_i)` 与 `N`，得到全局 SE。
3) 可选：实现 blocking/OBM 平台检测与合并，用作互证；提供配置开关（默认关闭互证，仅开启 IPS）。
4) 文档：在用户手册中说明误差定义变更、`g` 的物理意义与参数建议（K、阈值 1.1、最小块数 30–50 等）。

---

### 风险与对策
- 误差变大：更保守但更真实；同步更新阈值策略。
- 自相关估计不稳：默认仍提供 IID-SE；Blocking 作为增强互证选项。

---

### 实用参数建议（便于落地）
- 窗口/截断：K=10–20 或首次 \(\gamma_k\le 0\) 截断（IPS）。
- 判据：若本地 \(g_i\le 1.1\)，可直接使用 IID-SE；否则使用膨胀后的 SE。
- 并行通信：仅规约两个标量（\(\sum g_i s_i^2 n_i\)、\(N\)），对尴尬并行友好。
- Binning：在需要互证时选 b≈8–16（需 \(m=N/b\ge 30\)），做平台检查；常规生产不必强制启用。



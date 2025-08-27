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
2) 在 IID (Independent and Identically Distributed) 假设下给出稳健实现；可选启用 MCMC 自相关修正（IAT (Integrated Autocorrelation Time) 或 blocking）。

---

### 数学基础

设能量样本为 \(\{E_t\}_{t=1}^N\)。

1) IID 情形
- 样本均值：\(\bar E = \frac{1}{N}\sum_{t=1}^N E_t\)
- 无偏方差：\(s^2 = \frac{1}{N-1}\sum_{t=1}^N (E_t - \bar E)^2\)
- 标准误：\(\operatorname{SE}_{\text{IID}}(\bar E) = \sqrt{\frac{s^2}{N}}\)

2) MCMC 自相关修正
令积分自相关时间（integrated autocorrelation time）为 \(\tau_\text{int}\)：
\[ \tau_\text{int} = 1 + 2\sum_{k=1}^{\infty} \rho_k, \quad \rho_k = \frac{\operatorname{Cov}(E_t,E_{t+k})}{\operatorname{Var}(E_t)}. \]
则有效样本数 \(N_\text{eff} = \frac{N}{2\,\tau_\text{int}}\)（当 \(\tau_\text{int}\ge 0.5\)）。
据此，
\[ \operatorname{SE}(\bar E) \approx \sqrt{\frac{s^2}{N_\text{eff}}} = \sqrt{\frac{2\,\tau_\text{int}\, s^2}{N}}. \]

实践中可采用：
- 初始正序列/自适应窗口估计 \(\tau_\text{int}\)（Sokal 方法）。
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

---

### 设计与接口

1) 计算路径
- 在 `MCEnergyGradEvaluator::Evaluate` 中收集 `energy_samples`。
- 单/多进程统一：在所有 rank 计算本地 \(n_i, S_i, Q_i\)，Allreduce 得到 \(N,S,Q\)，计算 \(\bar E, s^2, SE\)。
- 结果：
  - `Result::energy` 设置为全局均值（TenElemT，可保留复数但实部来自统计）。
  - `Result::energy_error` 设置为 `SE_IID`（默认）或 `SE_with_IAT`（可选）。

2) 自相关修正（可选）
- 方案 A：IAT 估计（初始正序列/自适应窗口），每 rank 估计 \(\tau_{i}\)、\(s_i^2\)，近似合并：
  \[ \operatorname{Var}(\bar E) \approx \frac{1}{N^2} \sum_i 2\,\tau_i\, s_i^2\, n_i. \]
- 方案 B：blocking/batch means：各 rank 形成本地 block 均值向量，root 收集后统一估计 SE。

3) 类型与数值注意
- 对复数 `TenElemT`：E的统计保持complex，有利于后续Grad/SR稳定；最终 `Result::energy` 保持原类型。
- 使用 `long double` 存储 `S_i`、`Q_i` 与 Allreduce，减小舍入误差。(这里可能还需要考虑改进)

---

### 变更影响
- 单进程：不再返回 `inf`，得到有限标准误，测试阈值可设为 `k_sigma * SE`。
- 多进程：由“各进程均值的方差”改为“全样本标准误”，更符合统计意义。
- 行为：此变更为“hard 修改”，会影响现有误差数值与部分测试门限。

---

### 迁移与测试
- 将现有测试中的经验容忍度替换为：`tol = max(tol_min, k_sigma * energy_error)`，建议 `k_sigma = 3`。
- 新增单测：对合成 IID 数据验证 Allreduce 结果与单机一致；对 AR(1) 数据验证 blocking SE 趋于 IAT 修正。
- 集成测试：固定种子，比较旧/新误差行为并更新基线。

---

### 实施步骤
1) 在 `MCEnergyGradEvaluator::Evaluate` 内实现 `N/S/Q` 全局规约与 `SE_IID` 输出。
2) 保持现有能量均值广播逻辑，误差从本地/旧接口迁移至新实现。
3) 可选：实现 IAT 或 blocking；提供开关参数（默认关闭）。
4) 文档：在用户手册中说明误差定义变更及其物理含义。

---

### 风险与对策
- 误差变大：更保守但更真实；同步更新阈值策略。
- 自相关估计不稳：默认仍提供 IID SE；IAT/Blocking 作为增强选项。



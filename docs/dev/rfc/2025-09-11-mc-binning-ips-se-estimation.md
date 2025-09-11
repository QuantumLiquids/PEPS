---
title: Monte Carlo Standard Error via Power-of-Two Binning Scan and IPS Fallback
date: 2025-09-11
status: draft
owners: [PEPS Core]
---

## Motivation
Monte Carlo/VMC 序列存在显著自相关，直接按样本独立估计标准误差 (SE) 会严重低估不确定度。我们提出：
- 使用 b ∈ {1,2,4,8,...,b_max} 的分箱扫描，寻找 SE(b) 的稳定平台，选择推荐 bin 尺度 b*；
- 当样本不足形成平台时，以 IPS (Initial Positive Sequence) 膨胀因子作为兜底并给出 τ 的粗估。

## Definitions
对每个 MPI rank 的时间序列 X_r = {x_r(t)}_{t=1..N_r}，给定 bin 大小 b：
- 形成 K_r(b)=floor(N_r/b) 个完整 bin，丢弃尾部不足样本；
- 每 bin 的均值 m_r^{(i)}(b) 视为近独立样本；
- 通过 MPI Gatherv 汇总所有 rank 的 bin-mean 成 {m^{(j)}(b)}，在 master 上计算
  - μ(b) = mean({m^{(j)}(b)})
  - SE(b) = sqrt(Var({m^{(j)}(b)}) / (K(b)-1))，其中 K(b)=∑_r K_r(b)

## Binning Scan
- b 取幂序列：b=2^k，k=0,1,...，直到 b_max；默认 b_max = max_r floor(N_r/20)。
- 仅当 K(b)≥2 时 SE(b) 有效。
- 平台判据：设相邻相对变化 Δ(b_i) = |SE(b_{i+1})-SE(b_i)|/max(SE(b_i),tiny)。
  - 选择最小的 b_s 使得连续 L 个点满足 Δ<ε（默认 ε=0.05，L=2 或 3）。
  - 推荐 b* = b_s，SE* = SE(b*)。
- 若未找到平台，则取最大有效 b 的 SE 作为保守估计，并标记 unstable=true。

## IPS Fallback and τ Estimate
当有效 b 点数不足或 K(b)<2：
- 计算本地 IPS 膨胀因子 g ≥ 1（短窗，IPS 截断）：
  - g = 1 + 2∑_{k≤K}^{IPS} ρ_k^{(+)}，ρ_k 为归一化自相关；
  - 返回 SE ≈ sqrt(Var(X)/N_total) * sqrt(g)。
- 给出 τ_est ≈ g/2 的粗估（经典关系 τ_int ≈ ∑ρ_k）。

## Dump Policy
- 缺省仅输出 {μ, SE*, b*, plateau_found, τ_est(optional), g_ips(optional)}。
- 可选开启详细 dump：
  - stats/<key>_bin_scan.csv: 列 b,num_bins,mean,stderr
  - stats/<key>_autocorr.csv: 短窗自相关/IPS 计算所用 γ_k（可选）
- 复数 CSV 以两列 re,im 表示；header 写清列名与 shape。

## API Proposal
统计层（可独立复用）：
```c++
template<typename T>
struct BinningPoint { size_t b; size_t num_bins; T mean; double stderr; double tau_est; bool valid; };

template<typename T>
struct BinningScan { std::vector<BinningPoint<T>> curve; size_t chosen_index; bool plateau_found; double g_ips; };

template<typename T>
BinningScan<T> BinningSEScanMPI(const std::vector<T>& local_series,
                                MPI_Comm comm,
                                size_t b_max = 0,
                                double epsilon = 0.05,
                                int plateau_len = 2,
                                bool enable_ips = true);
```

调用层（`MCPEPSMeasurer`）：对每个观测量的每个分量调用扫描函数，汇总 μ、SE、b*，可选 dump。

## Implementation Notes
- 分箱丢尾样本，rank 间 K_r(b) 不同，使用 Gatherv。
- 复数自相关使用共轭；方差/SE 使用平方范数。
- 避免过多 I/O：仅 master 输出；使用 '\n' 替换 std::endl。

## Defaults
- epsilon=0.05，plateau_len=2，b_max= floor(N/20)（按最小 N_r 估计）；可由参数覆盖。

## Backward Compatibility
现有 energy/one-point/two-point 输出保持不变；新增统计结果通过新接口与 dump 暴露。

## Relation to RFC: Observable Registry
本 RFC 与 “Observable 注册表” 配套：每个 key/分量得到独立 SE 估计与 b 扫描曲线。



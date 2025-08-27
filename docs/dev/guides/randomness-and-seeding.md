### 随机数与种子（RNG）设置与作用域摘要

本指南总结 include/ 下 Monte Carlo 相关代码的随机数来源、作用域与复现性特征，供开发与测试参考。

### 当前结构（按组件）

- Configuration 随机初始化（函数局部 RNG）
  - 接口：`Configuration::Random(dim)`、`Random(occupancy_num)`、`Random(map)`。
  - 实现：函数内创建 `std::random_device` → `std::mt19937`，使用该临时引擎生成或打乱配置；调用结束即销毁。
  - 特性：每次调用重新播种；平台相关非确定性；MPI 各进程独立；当前使用 `rand_num_gen() % dim` 存在取模偏差风险。

- MonteCarloSweepUpdater（对象成员 RNG）
  - 基类：`MonteCarloSweepUpdaterBase` 内部持有 `std::mt19937 random_engine_`，构造时以 `std::random_device` 播种；还持有 `std::uniform_real_distribution<double> u_double_`。
  - 生命周期：与 Updater 对象一致；由 `MonteCarloEngine` 成员持有。
  - 复现性：默认无；对象复制会复制 RNG 状态（可能导致流复制）。

- 非详细平衡 MCMC 工具（调用方注入 RNG）
  - 模板函数：`NonDBMCMCStateUpdate(..., RandGenerator &generator)`。
  - 特性：不自持 RNG；由调用方传入引擎，最易实现端到端可复现。

- MonteCarloEngine（分布成员，无自有引擎）
  - 成员 `u_double_` 为分布对象；真正的随机引擎由 `mc_sweep_updater_` 内部持有。

### MPI 语义

- 现状主要依赖各进程独立 `std::random_device` 播种，通常互不相关，但不可严格复现，且平台差异存在。
- 建议在需要可复现实验/测试时，显式设定“全局基种子”，并按 rank 派生子种子。

### 现存问题（风险）

- 缺少统一的显式种子 API（配置/优化/采样）；不利于复现实验与回归测试。
- Updater/Optimizer 可复制引擎状态，若不注意会造成“相同随机流”复制。
- `MonteCarloEngine` 的分布成员与引擎分离易引起误解，应统一风格或移除冗余分布对象。

### 快速结论（Where is RNG owned?）

- Configuration::Random(...)：函数局部 RNG；每次重播种；不可复现。
- MonteCarloSweepUpdaterBase：成员 RNG；构造播种；生命周期=对象；复制会复制状态。
- NonDBMCMCStateUpdate：不自带 RNG；由调用方注入；最易复现。
- MonteCarloEngine：不自持引擎（依赖 Updater）；分布对象不产生熵。

### 最佳实践（保持兼容）

- 有显式种子时优先使用显式种子（按 rank 派生），无显式种子时退回 `std::random_device`。
- 避免复制带状态的 RNG 容器对象；如必须复制，复制后立即重播种。
- 多线程时每线程独立引擎，不共享一个引擎实例。

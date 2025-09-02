---
title: Slow Test Manifest (@testing/)
last_updated: 2025-09-02
---

## 目标
- 建立慢测清单与运行规范，统一 MPI 配置、样本规模分档与超时预期。
- 默认跑集测的 MPI 版本，支持本地低核数（如 4 ranks）与 CI 高核数（如 56 ranks）。

## 全局约定
- 所有慢测均以 MPI 方式运行，所有 rank 同步读取参数与状态文件。
- 种子策略遵循 `reproducibility-policy.md`（待补全）：
  - 固定主种子 `SEED_BASE`，各 rank 派生 `seed = SEED_BASE + rank`。
- 统计门槛：能量误差 `1e-3`（平方格子 3x4 Heisenberg 基准）。

## 运行配置
- CMake 选项：
  - `-DBUILD_SLOW_TESTS=ON` 开启慢测目标构建。
  - `-DSLOW_TESTS_MPI_PROCS=<int>` 指定慢测 MPI 进程数（默认 56）。

### 本地开发推荐（Mac，11 代 CPU，内存受限）
- 建议 `SLOW_TESTS_MPI_PROCS=4`。
- 采样规模分档：
  - VMC 优化：预热 200，采样 1k，block 1，迭代 30-50（SR 可适当减半）。
  - 测量：预热 2k，采样 4k，block 1。
- OpenMP 线程：`hp_numeric::SetTensorManipulationThreads(1)` 保守设置避免过度 oversubscription。

### CI/大机推荐
- `SLOW_TESTS_MPI_PROCS=56` 或由机群分配决定。
- 采样规模：至少本地的 2-4 倍；确保测量误差 < 1e-3。

## 用例清单

### Square Heisenberg 3x4, D=6（SR + 测量）
- 入口：`tests/integration_tests/test_square_heisenberg.cpp`
- 状态生成：SimpleUpdate（主 rank），规范化后保存至 `final_config/` 路径（`GenTPSPath`）。
- 优化：Stochastic Reconfiguration（SR）通过 `api/vmc_api.h` 包装调用。
- 测量：`MonteCarloMeasure` 包装调用。
- 断言：`EXPECT_NEAR(real(E), E_ref, 1e-3)`。

#### 运行与过滤示例
```bash
ctest -V -R test_square_heisenberg_.*_mpi --test-dir build
# 单条：
ctest -V -R test_square_heisenberg_double_mpi --test-dir build
```

#### 结果记录模板
```text
test: test_square_heisenberg_double_mpi
date: 2025-09-02
mpi_procs: 4
mc_opt: warmup=100, sample=100, blocks=1, iters=40, lr=0.3
mc_measure: warmup=1000, sample=1000, blocks=1
threads: 1
energy: -6.6921 +- 0.0008
pass: true
notes: 参数收敛稳定；无 Message truncated；Gflops ~ 1.2
```

## 运行示例
```bash
cmake -S .. -B build -DBUILD_SLOW_TESTS=ON -DSLOW_TESTS_MPI_PROCS=4
cmake --build build -j
ctest -V -R test_square_heisenberg_.*_mpi --test-dir build
```

#### 运行记录位置

- 实际运行记录统一保存在：`docs/dev/testing/runs/`（按日期与测试名命名）。


## 诊断建议（MPI Message truncated）
- 确保所有 rank 加载一致的 TPS（路径正确、文件完整）。
- 参数在各 rank 完全一致（优化器、MC、模型）。
- 不要向 `MPI_COMM_WORLD` 混入非参与 rank；确保所有 rank 进入同一个测试分支。
- 若仍出现 `Message truncated`，启用更详细日志并缩小样本规模以定位来源。

## 日志与归档策略
- 不引用 `build/*.log`（该目录被 `.gitignore` 忽略）。
- 以“运行记录”形式长期保存关键指标，路径：`docs/dev/testing/runs/`。
- 命名规范：`YYYY-MM-DD-<ctest-name>.md`，如：`2025-09-02-test_square_heisenberg_double_mpi.md`。
- 必填字段：日期、测试名、MPI ranks、线程数、采样配置、关键用时（阶段与总时长）、能量及误差、通过性（GTest/CTest）。



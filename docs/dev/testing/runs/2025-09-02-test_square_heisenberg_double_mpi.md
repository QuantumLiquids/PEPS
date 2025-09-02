---
title: 2025-09-02 Heisenberg (double) MPI run record
date: 2025-09-02
test: test_square_heisenberg_double_mpi
---

环境
- OS: macOS 15.6.1 (Apple M3 Pro, RAM 18 GB)
- MPI ranks: 4
- OpenMP threads: 1

命令
```bash
ctest -V -R test_square_heisenberg_double_mpi --test-dir build
```

采样配置
- MC优化（SR）：warmup=100, sample=100, blocks=1, iters=40, lr=0.3
- MC测量：warmup=1000, sample=1000, blocks=1

要点
- SimpleUpdate 用时: 约见 GTest 输出（包含在总时长内）
- SGDWithZeroLR 用时: 7361 ms
- SR 优化 + 测量用时: 159505 ms
- 整体 GTest 用时: 267029 ms
- 真实总用时 (ctest real): 267.84 s
- 配置校验: All 4 processes have valid configurations
- MC接受率: 0.27
- 能量: -6.69142 ± 6.43203e-04
- 误差估计：优化阶段bin size使用 sqrt(sample)；测量bin size = sample

通过性
- GTest: 3 tests passed
- CTest: 标记 Failed（MPI 报告 non-zero exit；需后续诊断）

备注
- 原始日志在本地 build 目录（未纳入 git）：heisenberg_test_20250902_155753.log
  本记录为精简提要，便于追踪。



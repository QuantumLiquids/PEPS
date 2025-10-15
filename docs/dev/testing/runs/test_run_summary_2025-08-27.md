### 测试运行摘要（2025-08-27）

- 执行者环境
  - OS: macOS 15.6.1 (24G90)
  - 机型: MacBook Pro (`Mac15,6`)
  - 芯片: Apple M3 Pro（总 11 核：5 性能核 + 6 效率核）
  - 内存: 18 GB

- 构建与执行
  - CMake 配置: Release，关闭慢测（`-DBUILD_SLOW_TESTS=OFF`）
  - 命令（供复现）：
    - 配置: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SLOW_TESTS=OFF`
    - 构建: `cmake --build build -j$(sysctl -n hw.ncpu)`
    - 测试: `cd build && ctest --output-on-failure -j2`
  - 备注: ctest 并行度为 2（`-j2`）。

- 结果汇总（完整通过的一次运行）
  - 通过: 49 / 49
  - 失败: 0
  - 总耗时: 160.37 秒（real）

- Top 慢测（取该次运行打印的主要耗时项）
  - test_optimizer_sgd_exact_sum_complex_mpi: 48.19 s
  - test_optimizer_sgd_exact_sum_double_mpi: 47.54 s
  - test_mc_energy_grad_evaluator_spinless_fermion_2x2_complex: 41.13 s
  - test_mc_energy_grad_evaluator_spinless_fermion_2x2_double: 40.11 s
  - test_optimizer_adagrad_exact_sum_double_mpi: 7.47 s
  - test_optimizer_adagrad_exact_sum_complex_mpi: 7.78 s
  - test_optimizer_adagrad_exact_sum_double: 8.18 s
  - test_optimizer_adagrad_exact_sum_complex: 7.36 s
  - test_boson_simple_update_complex: 7.36 s
  - test_tn2d: 6.26 s
  - test_boson_simple_update_double: 6.14 s
  - test_arnoldi: 2.69 s
  - 其余单测多数 < 2 s

- 观测与建议
  - 总耗时主要由 MPI 版本的优化器/评估器用例与 2x2 自由费米子 MC 评估用例主导。
  - 若需缩短回归时间：
    - 将 `ctest -j` 并行度提升（受限于机器核心与内存）。
    - 对最慢的 4 个用例提供一个 “快速配置” 开关（降低 sample 数、缩短 sweep），集成到 CI 日常跑；保留完整配置给 nightly/weekly。



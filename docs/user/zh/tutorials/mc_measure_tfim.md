# 蒙特卡洛测量（TFIM）——测量 VMC 得到的 TPS

本教程展示如何对横场 Ising 模型（TFIM）的优化后态进行**蒙特卡洛采样测量**。我们从 VMC 优化教程导出的 `SplitIndexTPS` 开始，加载到内存后调用 `MonteCarloMeasure(...)` 得到能量与基础可观测量。

本教程衔接：

- `examples/transverse_field_ising_vmc_optimize.cpp`

## 前置条件

1. 你已经运行过：
   - `examples/transverse_field_ising_simple_update.cpp`（导出 `peps/`）
   - `examples/transverse_field_ising_vmc_optimize.cpp`（导出 `optimized_tps/`）
2. 你的环境可以编译并运行 MPI 程序。

## 你会得到什么（输出文件）

测量器会把 CSV 结果写到：

- `./mc_measure_output/stats/`

对于 TFIM，内置求解器通常会注册（至少）这些 key：

- `energy` → `stats/energy.csv`（标量，以“扁平表”形式写出）
- `spin_z` → `stats/spin_z_mean.csv`, `stats/spin_z_stderr.csv`（Ly×Lx）
- `sigma_x` → `stats/sigma_x_mean.csv`, `stats/sigma_x_stderr.csv`（Ly×Lx）
- `SzSz_row` → `stats/SzSz_row.csv`（扁平数组；中间一行的关联）

> 说明：registry key 是下游工具的“权威接口”。可参考 `../reference/model_observables_registry.md`。

## 示例程序

仓库中提供了对应的示例代码：

- `examples/transverse_field_ising_mc_measure.cpp`

它会：
- 从 `./optimized_tps`（或你传入的路径）加载 `SplitIndexTPS`；
- 构造 `MCMeasurementParams`；
- 使用 `TransverseFieldIsingSquareOBC(h)` + `MCUpdateSquareNNFullSpaceUpdate` 调用 `MonteCarloMeasure(...)`；
- 把结果导出到 `./mc_measure_output/`。

## 编译

把 `examples/` 当成一个独立的小工程来编译：

```bash
cd examples
mkdir -p build && cd build
cmake ..
make -j
```

## 运行

使用 MPI 运行（每个 rank 独立跑一条马尔可夫链；最终统计会聚合）：

```bash
cd examples/build
mpirun -n 4 ./transverse_field_ising_mc_measure
```

### 测量哪一个态？

如果你的 VMC 同时保存了 “best state”（比如用优化器的 `DumpData()` 导出 `*_lowest`），建议先测量 best state。

对仓库自带的 VMC 示例，默认导出目录是：

- `./optimized_tps`

如需测量别的目录，把路径作为第一个参数传入：

```bash
mpirun -n 4 ./transverse_field_ising_mc_measure /path/to/your/sitps_dir
```

## 常见坑

- **格点尺寸必须匹配**：`Configuration(Ly,Lx)` 必须和加载到的 TPS 尺寸一致。
- **预热/样本数不够**：观测量噪声很大时，增大 `num_warmup_sweeps` 和/或 `total_samples`。
- **输出位置**：聚合后的统计结果在 `mc_measure_output/stats/`（一般由 master rank 写出）。

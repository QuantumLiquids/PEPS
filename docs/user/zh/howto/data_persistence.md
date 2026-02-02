# 数据落盘与目录结构（VMC/测量）

## 概览

统一的持久化体系覆盖 VMC 优化与测量，提供显式可控、语义清晰、无魔法路径的 I/O。

## 设计原则
- 显式用户控制：路径为空则不落盘（不推荐），不做隐式回退
- 按数据类型分离：配置、TPS、测量数据、优化轨迹彼此独立
- 合理默认：避免意外占盘，同时贴合常见科研工作流

## 组件与职责

### MonteCarloParams — 配置快照
```cpp
struct MonteCarloParams {
  // ...
  std::string config_dump_path; // 为空则不保存最终配置
};
```
- 推荐：`MonteCarloParams(..., "./final_config")`
- 关闭：`MonteCarloParams(..., "")`（不推荐）

### VMCPEPSOptimizerParams — TPS 基名
```cpp
struct VMCPEPSOptimizerParams {
  // ...
  std::string tps_dump_base_name; // 自动生成 "final"/"lowest"
};
```
- 行为：若非空，生成 `base+"final"` 与 `base+"lowest"`；为空则不导出
- 示例：`"expA" → "expAfinal", "expAlowest"`

### MCMeasurementParams — 测量数据路径
```cpp
struct MCMeasurementParams {
  MonteCarloParams mc_params;
  PEPSParams peps_params;
  std::string measurement_data_dump_path; // 为空=当前目录
};
```
- 目录结构：
```
{path}/
├── energy_sample_data/
├── wave_function_amplitudes/
├── one_point_function_samples/
└── two_point_function_samples/
```

## 优化器落盘
- TPS：输出末态与最低能量两份
- 能量轨迹：固定写入 `./energy/`
- 配置快照：由 `mc_params.config_dump_path` 控制

典型配置：
```cpp
// 生产（完整保存）
params.tps_dump_base_name = "production_run_tps";
params.mc_params.config_dump_path = "production_run_final_config";

// 快速测试（不落盘）
params.tps_dump_base_name = "";
params.mc_params.config_dump_path = "";
```

## 最佳实践
- 使用可读性的基名：包含模型、D、LxLy、run id
- 大规模参数扫描：可关闭 TPS/配置导出，仅保留能量轨迹
- 优化-测量串联：优化导出的配置用于测量初始化

## 小结
- 职责分离与命名清晰
- 落盘由用户显式控制
- 统一且可预期的目录结构，便于长期管理与复现

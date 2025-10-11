# 自定义 Measurement Solver 开发指南（注册表版）

本文面向需要在 VMC/PEPS 框架中新增/改造观测量（observable）的开发者，说明注册表接口、返回数据的形状规范、以及统计与导出格式。

## 1. 设计理念
- 不再使用固定的 energy/一体/二体分类；改为“可注册的观测量（key+meta）”。
- 每个观测量独立统计，独立导出，用户无需了解模型内部分类。

## 2. 必要接口
在模型类（继承 `ModelMeasurementSolver<YourModel>`）中实现：

1) 元数据描述
```c++
std::vector<ObservableMeta> DescribeObservables(size_t ly, size_t lx) const;
```
- `key`: 唯一字符串，如 "energy"、"spin_z"、"psi_mean"；
- `description`: 简短英文说明；
- `shape`: 数据形状；标量用 `{}`，site 级 `{ly, lx}`，横向 bond `{ly, lx-1}`，纵向 bond `{ly-1, lx}`，对角 `{ly-1, lx-1}` 等。
- `index_labels`: 可选轴标签；使用 `{"bond_y", "bond_x"}` 表示键起点坐标；如使用上三角打包，首元素填 `"pair_packed_upper_tri"`。

2) 单次样本计算
```c++
template<typename TenElemT, typename QNT>
ObservableMap<TenElemT> EvaluateObservables(
    const SplitIndexTPS<TenElemT, QNT>* sitps,
    TPSWaveFunctionComponent<TenElemT, QNT>* tps_sample);
```
- 返回 `key -> flat vector`；如果是矩阵，按行主序展平，长度等于 `shape` 各维乘积。
- 如无可计算观测量可返回空（框架将回退老式路径）。

建议：如需要 `psi_list` 一致性检查，可在模型内部收集并返回 `psi_list`，框架将计算 `psi_rel_err` 并按阈值告警。

## 3. 统计与导出
- 统计：框架对每个 `key` 的每个分量在 MPI master 汇总并计算“均值 + 旧式标准误差（忽略自相关）”。
- 导出目录：`stats/`
  - 统一 CSV：`stats/<key>.csv`，表头：`index,mean,stderr`
- 对于二维形状，仍以扁平索引导出，shape 信息可从 `DescribeObservables(ly,lx)` 的返回值获得
- 类型：由模板参数 TenElemT 决定（double/complex）。复数以 C++ 默认格式输出（"(re,im)"）。如需拆分列，可在后续版本加入 `_re/_im` 双列支持。

## 4. `psi_consistency` 建议
- 物理含义：单层张量网络不同收缩路径的振幅 `psi(S)` 差异度量。
- 建议以两个 key 输出：
  - `psi_mean`: 平均振幅
  - `psi_rel_err`: 相对半径，定义为 `max(|psi_i - psi_mean|) / |psi_mean| / 2`
- 运行时提醒：当 `psi_rel_err > 1e-4` 时在标准错误输出一次警告（每 rank 上限 50 次），以提示调节 BMPS 裁剪参数。

## 5. 性能与内存
- 请只注册必要的观测量。高维观测量（如两点函数场）建议通过参数开关控制是否启用。
- 样本原始数据（samples）默认不导出；如需调试，可在模型内局部输出。

## 6. 迁移建议
- 旧模型可先保留原有 `SampleMeasureImpl`，再逐步实现注册表接口。
- key 命名尽量语义清晰和稳定，便于用户脚本对接。

## 7. 常见问题
- Q: 复数观测量如何导出？
  - A: 当前为 "(re,im)" 文本。需要 CSV 双列时可提交需求，我们会统一支持。

## 8. 命名规范（强制）
- 不使用泛化的 `one_point`、`two_point` 键。必须使用语义明确的键：`spin_z`、`sigma_x`、`charge`、`SzSz_row`、`SC_bond_singlet` 等。
- `bond_energy` 默认不导出；如需，可按几何拆分：`bond_energy_h`、`bond_energy_v`、`bond_energy_diag`（为 PBC/三角格子预留）。
 - 超导相关：若仅有键-单线态幅度，命名为 `SC_bond_singlet`；含相位结构需另行命名（如确认 d-wave 再命名）。

### 上三角打包（all-to-all 配对量节省内存约定）
- 对于配对观测量（如 `SzSz_all2all`），可使用上三角打包（仅存 i≤j 的分量），长度为 L*(L+1)/2。
- 在 `DescribeObservables()` 中将 `index_labels` 的首元素设置为 `pair_packed_upper_tri` 以标记该约定。
- 导出时会额外写出 `stats/<key>_index_map.txt`，说明索引到 (i,j) 的映射规则。
## 9. 统计FAQ
- Q: 误差估计是否考虑自相关？
  - A: 当前为旧式 SE。见 RFC《Monte Carlo Standard Error via Power-of-Two Binning Scan and IPS Fallback》，后续会切换到分箱扫描。



# 内置模型可观测量支持（registry key）

本页汇总当前各个蒙特卡洛测量求解器会输出哪些 registry key，并记录 shape/索引约定。
同时也会对照旧的 “one-point/two-point” 管线指出缺口，方便在 registry API 稳定期间追踪回归。

> **注意**
> 下表中的每个 key 都是下游工具的“权威接口”。
> 如果某个求解器返回了额外的数据向量，但没有通过 `DescribeObservables()` 声明元数据，
> 那么 CSV 导出与可视化工具只能把它当作“匿名的一维数组”处理。
> 请扩展元数据，而不是依赖隐式约定。

## 汇总表

| 模型类 | 当前 registry key | Shape / 索引说明 | 相对 legacy 的缺口 |
|---|---|---|---|
| `TransverseFieldIsingSquareOBC`（以及 `TransverseFieldIsingSquarePBC`） | `energy`, `spin_z`, `sigma_x`, `SzSz_row` | `energy`：标量；站点量 `{Ly,Lx}`，索引 `{y,x}`；`SzSz_row`：扁平数组 | 缺口已补齐——元数据与返回 key 一致。 |
| `SquareSpinOneHalfXXZModelOBC` | `energy`, `spin_z`, `SzSz_all2all`, `bond_energy_h`, `bond_energy_v` | 站点量 `{Ly,Lx}`（`{y,x}`）；键能量按 `{bond_id}` 标号；`SzSz_all2all` 为上三角打包 | 无——legacy 的两点关联直接映射为 `SzSz_all2all`。 |
| `SquareSpinOneHalfJ1J2XXZModelOBC` | `energy`, `spin_z`, `bond_energy_h`, `bond_energy_v`, `bond_energy_dr`, `bond_energy_ur` | 站点量 `{Ly,Lx}`；键能量 `{bond_id}` | 与 legacy 的 NN/NNN 键能量输出一致。 |
| `SquareHubbardModel` | `energy`, `spin_z`, `charge`, `double_occupancy`, `bond_energy_h`, `bond_energy_v` | 站点量 `{Ly,Lx}`；键能量 `{bond_id}` | 缺口已补齐——`double_occupancy` 显式输出。 |
| `SquareSpinlessFermion` | `energy`, `charge`, `bond_energy_h`, `bond_energy_v`, `bond_energy_dr`, `bond_energy_ur` | 站点电荷 `{Ly,Lx}`；键能量 `{bond_id}` | legacy 把费米子动能按方向拆分；这里统一表述为键能量。 |
| `SquaretJNNModel`, `SquaretJNNNModel`, `SquaretJVModel` | `energy`, `spin_z`, `charge`, `SC_bond_singlet_h`, `SC_bond_singlet_v`, `bond_energy_h`, `bond_energy_v`（NNN 启用时包含 `bond_energy_dr/ur`） | 站点量 `{Ly,Lx}`；键能量 `{bond_id}`；SC key `{Ly,Lx}` | legacy 将 singlet 关联存放在 two-point bins；这里保留映射并用显式 key 表达。 |
| `SpinOneHalfTriHeisenbergSqrPEPS` | `energy`, `spin_z`, `bond_energy_h`, `bond_energy_v`, `bond_energy_ur`, `SzSz_row`, `SmSp_row`, `SpSm_row`, `SzSz_all2all` | 三角键会扁平化；row correlators 长度约为 `Lx/2` | 通道与 legacy 相同，但现在用显式 key 表示。 |
| `SpinOneHalfTriJ1J2HeisenbergSqrPEPS` | 同上（triangular 模型） | 同上 | 同上 |

补充说明：

- 通用基类 `SquareNNNModelMeasurementSolver` 仍会输出 `energy`、`bond_energy_*`，并依据具体模型的静态能力标志输出 `spin_z` 和/或 `charge`。
- 任何需要额外通道的求解器都应通过 `DescribeObservables()` 声明；否则执行器无法附加 shape 元数据，下游只能把数据当 1D 数组处理。
- `psi_list` 是内部工作缓冲区。全局统计（`psi_mean`, `psi_rel_err`）由执行器写出，刻意不属于 per-model metadata。

## 相对 main 分支的差异（历史说明）

重构移除了单体的 `Result` 结构体（`energy`, `one_point_functions`, `two_point_functions`, …）。
与当前 `main` 对比时，我们观察到：

1. 能量与键能量仍然存在，对应 `energy`、`bond_energy_h/v/(dr/ur)`。
2. 站点局域量（`spin_z`, `charge`）仍以显式 key 保留。
3. 一些特化的关联仍会被计算，但缺少元数据（例如 TFIM 求解器中的 `sigma_x`, `SzSz_row`）。这些 key 曾经会导出到 CSV；应通过 `DescribeObservables()` 补齐元数据，或调整导出逻辑。
4. legacy 的 “psi consistency” 字段不再属于 public struct；`MCPEPSMeasurer::GetEnergyEstimate()` 提供有限的兼容层，其余信息全部走 registry。
5. Hubbard 的 double-occupancy 直方图与原始 `psi_list` 导出是设计上被移除的；若需要回归，请在 RFC 中记录跟进任务。

如果迁移过程中出现新的缺口，请记录到：

- `docs/dev/rfc/2025-09-11-observable-registry-and-results-organization.md`

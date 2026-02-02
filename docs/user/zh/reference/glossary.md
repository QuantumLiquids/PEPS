# 术语表

本表统一用户文档中的术语（英文 `docs/user/en/` 为权威版本）。

| 术语 | 含义 |
|---|---|
| PEPS | Projected Entangled Pair State。本库中常指带有 bond weight（例如 Simple Update 输出）的 PEPS 形式。 |
| TPS | Tensor Product State。本库中指不显式保存 bond weight、格点张量直接相连的形式。 |
| SplitIndexTPS | 预先把物理指标拆分的 TPS 变体，用于加速蒙特卡洛投影/振幅评估。 |
| Configuration | 经典组态（每个格点的自旋/占据），用于蒙特卡洛采样。 |
| 蒙特卡洛更新器 | 在采样中更新 `Configuration` 的函子，需要满足正确性（balance / detailed balance）与遍历性要求。 |
| 能量求解器 | 在 VMC 中计算局域能量以及（可选）梯度“洞”张量。与模型相关。 |
| 测量求解器 | 在测量中计算 registry 形式的可观测量。与模型相关。 |
| OBC / PBC | 开放/周期边界条件。VMC/测量后端分别为 BMPS（OBC）或 TRG（PBC），并与 `PEPSParams` 交叉检查。 |
| SR | Stochastic reconfiguration（随机重构/自然梯度）方法。 |


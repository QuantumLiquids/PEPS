### 开发者文档索引（dev/）

本目录按“文档类型”为主轴组织，并提供“按模块”的导航入口。

- 文档类型
  - RFC/提案：`docs/dev/rfc/`
  - 架构/设计：`docs/dev/design/`
    - 架构：`docs/dev/design/arch/`
    - 数学：`docs/dev/design/math/`
  - 测试：`docs/dev/testing/`
  - 实践规范：`docs/dev/practices/`
- 按模块导航：`docs/dev/modules/`
  - `vmc_update/`、`optimizer/`、`peps/`

现有文档：
- RFC：`rfc/2025-08-21-energy-evaluator-concept.md`、`rfc/2025-01-29-missing-sgd-variants.md`
- 设计：`design/arch/overview_cn.md`、`design/arch/mpi-contracts.md`、`design/math/exact-summation.md`
- 测试：`testing/optimizer-testing-strategy.md`、`testing/monte-carlo-testing-best-practices.md`
- 规范：`practices/coding-standards.md`、`practices/code-review-findings.md`

约定（简要）：
- 文件命名：`YYYY-MM-DD-title.md` 用于 RFC；其余用主题名。
- 元信息（可选 YAML）：`title`、`status`、`last_updated`、`tags`、`applies_to`。

### 目录结构（约定）
```text
docs/
  dev/
    index.md                      # 目录/约定/地图
    adr/                           # Architecture Decision Records（决策记录，带日期）
    rfc/                           # 提案/RFC（含评审、状态）
    design/                        # 设计文档（系统&模块设计、算法、数据流）
      math/                        # 数学原理与推导
      arch/                        # 架构总览/边界/约束
    guides/                        # 开发者指南/上手/实践手册
    testing/                       # 测试策略、基准、覆盖面、慢测清单
    ops/                           # CI/CD、发布流程、版本策略、应急手册
    practices/                     # 编码规范、评审规范、分支策略
    postmortems/                   # 重大问题复盘/根因分析（Issue类）
    modules/                       # 按模块的汇总入口（仅“索引”，避免重复内容）
```
说明：若目录暂缺，将随首篇文档自动创建；无需预先占位。



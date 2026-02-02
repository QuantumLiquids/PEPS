# 历史：`VMCPEPSExecutor` → `VMCPEPSOptimizer`

本页是 executor → optimizer 重构系列的**历史记录**。

如果你现在要写新代码，请从当前用户文档入口开始：

- Tutorials：`../../tutorials/index.md`
- How-to：`../../howto/index.md`
- Explanation：`../../explanation/index.md`
- Reference：`../index.md`

## 范围与版本（tag + 锚点提交）

这次重构并非单一提交完成：它在 v0.0.1 tag 之前就开始，并在多个后续提交中持续推进。

- 重构起点（第一步大改动）：`fd401d0`（Major VMC PEPS refactoring: extract optimizer and create independent executor）
- 本仓库中的 release tag：
  - v0.0.1：`24936cb`
  - v0.0.2：`7f49e27`
  - v0.0.3：`4cf92e5`

## 概念层面的变化

总体上，代码从“executor 一体化承担所有职责”的结构，逐步迁移为更显式的组合式设计：

- **Configuration 成为一等输入**（不再把 “occupancy arrays + (ly,lx)” 分散在各处构造器中）。
- **参数按职责拆分**：optimizer 参数、Monte Carlo 参数、收缩参数各自独立。
- **高层 API 更清晰**（显式状态转换 helper、更清楚的构造模式、统一的 measurement registry 接口）。

## 提交列表（重构系列）

### 基础重构

```
fd401d0 Major VMC PEPS refactoring: Extract optimizer and create independent executor
```

### 直到 v0.0.1 的后续提交

```
8ffec1e Add test data for heisenberg complex number
f275ac3 Fix several unit test bugs
cc9161b Fix bug
0314f5f fix header messy
b215d15 Make simple update and NDB MC tests work
0c2ca59 Major test suite reorganization and API improvements
829313a Complete test suite reorganization - remove old directories
f7f317b Update according Tensor API udpate
465fcfe feat: Complete test suite reorganization and establish new test architecture
b0bb83d Update test_exact_sum_optimization.cpp
c20766e setup Doc
c375116 feat: add documentation infrastructure and build system improvements
f848d5b Proposal for params refactor and harmless change
24936cb Add .cursor/ to .gitignore to ignore Cursor IDE files across all branches (tag: v0.0.1)
```

### v0.0.1 与 v0.0.2 之间的 API/optimizer 工作

```
852b2d4 refactor: Update Optimizer to use new parameter structures with variant dispatch
f5210a6 feat: complete optimizer params refactor implementation
a4765ac feat: Complete legacy VMC PEPS executor cleanup and modernization
a0daad3 refactor: unify Monte Carlo PEPS API parameters
5eadbca Refactor executors constructor to factory function and add Chinese documentation
e0f48cd refactor: complete optimizer params refactor with MPI-aware exact sum evaluator
cb798c3 refactor: inline UpdateTPSByGradient function
49fdc2e chore(headers, docs): cleanup imports; sync guides; add MC updater tutorial
845a98e docs,test: restructure docs; move test_data; update code/tests
dd6ee66 Rename TOP LEVEL APIs
044ee31 refactor: switch MonteCarloEngine (originally MonteCarloPEPSBaseExecutor) from inherit to composition
0c437f1 update docs; rename header; separate declare and impl
da40bc8 docs: migrate tutorials; optimizer: add lr schedulers and SGD exact-sum tests; stop tracking .cursor; ignore Testing/
69d9dd5 optimizer: gradient clipping + tests; migrate to qlten::hp_numeric::kMPIMasterRank; docs: coding standards placeholder
748eb40 update documents
44e6452 add API/tests; prune optimizer; add RFCs; WIP: extract MC energy evaluator
8e0ed20 vmc: use MCEnergyGradEvaluator; remove legacy accumulators/paths; update Doxygen
bf88b9f Update std err estimation in energy evaluator; Real->std::real();
6db53d9 minor update
2e68cf8 Fix Intel icpx compiler crash
85597a6 feat: add MC update instantiation params; improve state conversion API; unify Transverse Field Ising naming; add VMC Chinese tutorial and examples
ccbbb57 2 rfc; remove normalization in optimization; clean up
881debc chore: checkpoint messy edits before redesign
2ec6d76 Registry refactor WIP: metadata overhaul; SquaretJ energy still zero
892ff49 Complete tJ integration test and helper; eliminate the warning for legacy api.
d38abc8 fix bugs
a970c67 Improve DescribeObservables
c644615 More type gymnastics; extend TPS to PBC
2d76900 feat: add TFIM PBC TRG solver, rename OBC solver, and add PBC simple-update test data
7f49e27 docs: release v0.0.2 (tag: v0.0.2)
```

### v0.0.2 之后的变化（v0.0.3 线）

```
24de948 Unify psi-consistency warnings; Fix UpdateSingleSite_ Bug; Add Hubbard simple update test
fc7eaa6 feat: Add WaveFunctionSum and configurable configuration rescue
24b178c feat(optimizer): add Adam/AdamW optimizer
2a55994 vmc: add TRG/PBC support and explicit trial API
3dcf6ee feat: add TRG read-only replacement evaluation and Heisenberg PBC test
bf81c49 api: unify OBC/PBC dispatch; rebaseline XXZ structure-factor regression
648bf4c bmps/vmc/tests: update heisenberg baseline; unify file layout; fix PBC code paths
8863be8 fix: use ElementWiseSquaredNorm for Adam/AdaGrad complex correctness
4cf92e5 Release v0.0.3 (tag: v0.0.3)
```

## 如何复现/查看（命令）

```
git show fd401d0
git show v0.0.1 v0.0.2 v0.0.3
git log --oneline fd401d0..v0.0.1
git log --oneline v0.0.1..v0.0.2 -- include/qlpeps/api/vmc_api.h include/qlpeps/algorithm/vmc_update include/qlpeps/optimizer
```


title: VMC/PEPS 能量优化中的 spike 处理与 Checkpoint 方案（自动识别、重采样、回退）
author: Hao-Xin Wang
date: 2025-09-01
status: implemented

### 背景与目标
VMC/PEPS 优化过程中，偶发的 Monte Carlo 估计失稳或线性代数病态会导致“spike”（能量或梯度相关统计量的异常跳变）。本 RFC 旨在：
- 提供用户可控的 checkpoint 能力，便于失败恢复与事后分析；
- 在不破坏用户空间的前提下，自动识别并处理 spike：优先“重采样”，必要时“回退”；
- 将阈值类 magic numbers 以“后门”参数暴露，默认行为尽量保守，确保旧脚本与结果分布不被显著改变。

---

### 推进计划（三步走）
- 第一步（优先上线）：仅实现 Checkpoint 能力（默认关闭），不改变优化逻辑；用于失败恢复与可重复性。
- 第二步（稳定后开启）：实现 S1–S3 自动处置（仅重采样，默认开启；不做回退）。
- 第三步（观察生产数据后）：如有必要，再加入 S4（EMA+σ）回退逻辑，默认关闭以守住兼容性。

上述顺序确保：最低复杂度先行（Never break userspace），逐步引入自动防护，避免一次性引入复杂的回退路径。

---

### 需求确认（Linus 三问）
1) 这是真问题：是。实测中偶发 errorbar 飙升、SR 仅 1 次迭代“收敛”、或自然梯度范数异常跳变均会发生，且会污染轨迹。
2) 更简单的方法：先给出统一的“重采样”与“单步回退”机制，不做复杂鲁棒估计器改造；先用 EMA+sigma 的简单检测。
3) 会破坏什么：自动回退可能改变优化轨迹；因此默认仅启用“自动重采样”，将“回退”默认关闭，用户可显式开启。

---

### 数据结构与现状
- 轨迹与误差：`VMCPEPSOptimizer` 和 `Optimizer` 均维护能量与误差轨迹（如 `energy_trajectory_`, `energy_error_traj_`）。
- SR/CG 信息：`optimizer_impl.h` 中记录 `sr_iterations`, `sr_natural_grad_norm` 并打印；`SRSMatrix` 通过 `O^*` 样本构建。
- Dump 能力：`VMCPEPSOptimizer::DumpData()` 已能导出 `final/lowest` TPS 与能量轨迹 CSV；`SplitIndexTPS::Dump()` 已健全。

---

### Spike 识别信号与动作
信号 S 与动作 A 按优先级从高到低触发；触发即执行对应动作并短路（除非另有说明）。

- S1: errorbar 突然飙升（当前 `error_t` 相对平滑基线 > factor_err，默认 100x）
  - A1: 对当前态（时刻 t）“重做一次 MC 采样”（不回到 t-1）。若连续 `redo_mc_max_retries` 次仍异常，转入 S4 处置。

- S2: 梯度范数突然增大（欧式梯度范数相对基线 > factor_grad，默认 1e10）
  - A2: 同 A1 重采样（梯度的协方差/除 ψ 操作致噪声偏大，重采样更安全）。

- S3: 自然梯度范数突然增大（仅 SR 有）：相对基线 > factor_ngrad（默认 10x），或 SR 的 CG 仅 1 次迭代“收敛”
  - A3: 同 A1 重采样；若 `sr_iterations <= 1` 直接判为可疑并重采样。

- S4: EMA + 10σ 规则：用 EMA 平滑能量；采用“单边向上”检测：若 E_t − EMA(E) > k·σ 且 ΔE = E_t − E_{t-1} > 0（默认 k=10）。
  - A4: 回退到 t-1 并重做第 t 步（master 需备份 t-1 态；恢复后先在 t-1 上重新评估 E/grad/SR，再执行第 t 步更新）。此规则在 S1–S3 均未触发且通过 errorbar 合理性检验时判定，避免重复处置。

为什么用单边检测：1. 早期优化阶段的能量下降较快，向下变化很容易超过 k sigma(这一点也可以通过warm up识别来规避) 
2. 屏蔽downward spike可能会导致本可跳出local minimal的尝试错误。
3 downward spike如果不是跳出local minimal的优化，往往可以通过S1-S3检测出来。
综合1-3 选择单边检测。
说明：
- “基线”采用 EMA 或窗口均值（实现上统一使用 EMA，窗口等效参数通过 α 折算）。
- 所有判定在 master rank 计算，结果通过 MPI 广播统一执行。
- 为避免无限循环，所有“重采样/回退”的单步重试次数受 `redo_mc_max_retries` 限制（默认 2）。

---

### 参数设计（后门配置，默认保守）
新增两类参数，分别挂到现有参数体系，避免污染构造函数：

1) Checkpoint（在 `OptimizerParams` 下新增 `CheckpointParams`）
- `enable`（默认 false）：是否启用 checkpoint。
- `every_n_steps`（默认 0=禁用）：每 n 步保存一次。
- `base_path`（默认空=不写）：基路径，实际生成 `base_path/step_{k}/`。
  说明：最低能态/最终态的导出沿用既有机制；Checkpoint 仅保存“过程快照”，不与 best/final 混用。

2) Spike Handling（在 `OptimizerParams` 下新增 `SpikeRecoveryParams`）
- `enable_auto_recover`（默认 true）：启用自动处置。
- `redo_mc_max_retries`（默认 2）：单步最大重采样/回退重试次数。
- `factor_err`（默认 100.0）：S1 阈值。
- `factor_grad`（默认 1e10）：S2 阈值。
- `factor_ngrad`（默认 10.0）：S3 阈值。
- `sr_min_iters_suspicious`（默认 1）：CG 迭代不超过此值视为可疑。
- `ema_window`（默认 50）：供文档显示；内部用 `alpha = 2/(window+1)` 实现 EMA。
- `sigma_k`（默认 10.0）：S4 的 σ 倍数。
- `rollback_on_sigma_rule`（默认 false）：是否对 S4 启用回退（默认关闭以保持轨迹兼容性）。
- `log_trigger_csv_path`（默认空）：若非空，追加记录触发历史。

S4 的默认固定策略（无额外参数）：
- 仅单边向上检测：`E - EMA_E > sigma_k * std_E` 且 `ΔE = E_t - E_{t-1} > 0`。

兼容性主张（Never break userspace）：默认仅启用“自动重采样”（S1–S3），S4 回退默认关闭。即：默认配置对最终轨迹影响极小（仅在异常采样那一步重做），避免改变典型收敛路径。

---

### 算法流程（伪代码）
```cpp
for (int step = 0; step < max_steps; ++step) {
  bool step_completed = false;
  int attempts = 0;

  while (!step_completed) {
    // 1) Evaluate (no side-effects on EMA/trajectory)
    auto eval = evaluator(state, /*collect_sr_buffers_if_needed=*/false);
    double E = Re(eval.energy);
    double err = eval.energy_error;  // master only
    double grad_norm = L2Norm(eval.grad); // master computes norm

    // 2) S1/S2 detection first (cheap checks, master-only) → broadcast action
    Trigger trigger12 = DetectS1S2(E, err, grad_norm, ema_stats);
    Decision decision = DecideAction(trigger12, attempts, params); // RESAMPLE / ROLLBACK / ACCEPT
    Broadcast(decision);

    if (decision == Decision::RESAMPLE) {
      if (++attempts <= redo_mc_max_retries) {
        continue; // redo evaluator; NO EMA/trajectory update
      }
      if (params.rollback_on_sigma_rule && step > 0) {
        RestorePrevState();   // snapshot of step-1
        attempts = 0;
        continue;             // redo step
      }
      WarnAndAcceptSpikeSample(); // policy: accept or early-stop
      // fallthrough to ACCEPT
    }

    if (decision == Decision::ROLLBACK) {
      if (step == 0) {
        // No snapshot to roll back; downgrade to resample
        if (++attempts <= redo_mc_max_retries) continue;
        WarnAndAcceptSpikeSample(); // fallthrough to ACCEPT
      } else {
        RestorePrevState(); // back to step-1 state
        attempts = 0;
        continue; // redo step
      }
    }

    // 3) Compute update candidate (SR or first-order). SR yields sr_iters/ngrad_norm
    SRResult sr_result{}; double ngrad_norm = 0.0; int sr_iters = -1;
    SITPST state_candidate;
    if (is_sr) {
      sr_result = SolveSR(eval); // uses distributed CG inside
      ngrad_norm = sr_result.natural_grad_norm;
      sr_iters = sr_result.iters;
      state_candidate = ApplyUpdate(state, eval, sr_result);
    } else {
      state_candidate = ApplyUpdate(state, eval, sr_result);
    }

    // 4) S3 detection (SR-only signals), master-only → broadcast action
    Trigger trigger3 = DetectS3(ngrad_norm, sr_iters, ema_stats);
    decision = DecideAction(trigger3, attempts, params);
    Broadcast(decision);

    if (decision == Decision::RESAMPLE || decision == Decision::ROLLBACK) {
      // Discard candidate, then resample or rollback per decision
      if (decision == Decision::ROLLBACK && step > 0) {
        RestorePrevState();
        attempts = 0;
      }
      else {
        if (++attempts > redo_mc_max_retries && params.rollback_on_sigma_rule && step > 0) {
          RestorePrevState();
          attempts = 0;
        }
      }
      continue; // redo step
    }

    // 5) ACCEPT: commit side-effects (master only where applicable)
    UpdateEMA(ema_stats, E, err, grad_norm, ngrad_norm); // master only
    AppendTrajectories(E, err, /*other stats*/);         // master only

    // 6) Apply accepted update
    state = state_candidate;

    // 7) Save snapshot for potential rollback of next step
    SavePrevState(state);

    // 8) Checkpoint at ACCEPT (optional; avoid step==0 burst)
    if (checkpoint.enable && checkpoint.every_n_steps > 0
        && step > 0 && (step % checkpoint.every_n_steps == 0)) {
      SaveCheckpoint(state, step, traj, rng);
    }

    step_completed = true;
  }
}
```

---

### 检测实现细节
- EMA 维护：对 E 与 error、grad_norm、ngrad_norm 分别维护独立 EMA 与方差（Welford+EMA 变体）。
- S1：`err > factor_err * max(eps, EMA_err)` 触发。
- S2：`grad_norm > factor_grad * max(eps, EMA_grad)` 触发（非 SR 路径）。
- S3：`ngrad_norm > factor_ngrad * max(eps, EMA_ngrad)` 或 `sr_iters <= sr_min_iters_suspicious` 触发（SR 路径）。
- S4：单边向上检测：`E - EMA_E > sigma_k * std_E` 且 `ΔE > 0`。

MPI 语义：检测仅 master 计算；将触发类型、重试/回退决策广播；各 rank 同步执行 evaluator 或恢复本地状态。状态/轨迹更新由 master 负责聚合与持久化（沿用现有约定）。

---

### Checkpoint 设计
- 落点：Optimizer 层（无需回调）。理由：Optimizer 掌握 ACCEPT 的提交点、能量/误差/EMA 统计与 `prev_state`，能形成一致快照；避免在上层通过回调拼装状态。
- 时机：仅在 ACCEPT 之后、`SavePrevState(state)` 之后立刻执行（即“update 完成后”的状态）。避免在 evaluate 前或 update 前产生不一致快照。
- 内容：TPS（当前态）、优化步号、能量/误差轨迹 CSV、（可选）RNG 状态与优化器内部 EMA/方差统计。最低能态导出沿用既有 `base+lowest` 目录，不在 checkpoint 重复保存。
- 频率：`every_n_steps` 控制。
- 路径：`base_path/step_{k}/`；沿用 `SplitIndexTPS::Dump()` 的并行 I/O 与 `MPI_Barrier` 协作。

---

### 实现落点（文件与改动点）
- `include/qlpeps/optimizer/optimizer_params.h`
  - 增加 `struct CheckpointParams` 与存取器；默认值如上；不改变既有构造器签名。
- `include/qlpeps/optimizer/optimizer_params.h`
  - 增加 `struct SpikeRecoveryParams` 与存取器；默认值如上；不改变既有构造器签名。
- `include/qlpeps/optimizer/optimizer_impl.h`
  - 在能量评估与更新的同一逻辑域中插入 spike 检测与动作（重采样/回退），S1/S2 在更新前，S3 在 SR 更新后、提交前；提交点一次化（仅 ACCEPT 分支更新 EMA/轨迹/状态）。
  - 维护上一时刻 `state` 的快照（轻量复制或 move 语义初值）以支持单步回退。
  - 通过广播统一各 rank 的动作决策；限制重试次数。
  - 在 ACCEPT 后按频率执行 `SaveCheckpoint(...)`（避免 step==0 I/O 峰值）。
- `include/qlpeps/vmc_basic/monte_carlo_tools/statistics.h`
  - 若需要，补充 EMA/Welford 工具（或在 Optimizer 内部提供轻量私有实现）。

测试与文档：
- 单元测试：构造受控噪声触发 S1–S4；验证“重采样降低方差”“S4 回退后轨迹恢复”；MPI 多进程一致性。
- 集成测试：在典型模型上启/禁用自动处置；比对能量曲线与收敛迭代数。
- 用户文档：`docs/user/zh/explanation/optimizer_algorithms.md` 增补章节，说明默认行为与可调阈值。

---

### 风险与兼容性
- 性能：EMA/检测开销可忽略；重采样会提高单步成本，但仅在异常时触发。
- 兼容性：默认不开启 S4 回退；仅在 S1–S3 异常时重采样，尽量不改变总体轨迹分布。
- I/O：checkpoint 默认关闭；开启时注意大规模并行时的 I/O 峰值（建议 `max_to_keep` 小值）。

---

### 对方案的评价与可选增强
- 现实有效性：S1–S3 基本覆盖大部分“糟糕样本”与 SR 病态，优先重采样即可修复；S4 作为兜底的“策略回退”。
- 更鲁棒但更复杂的替代：Median-of-Means、Hampel/winsor 去极值、对 SR 做预条件或谱裁剪等。由于复杂度与维护成本高，先不纳入本提案的默认实现。

---

### 结论（决策输出）
【核心判断】
✅ 值得做：在真实训练中 spike 确实出现；“重采样+单步回退+checkpoint”是低成本且实用的组合。

【关键洞察】
- 数据结构：能量/误差/（自）梯度范数的 EMA 与方差即可支撑检测；SR 迭代步数是强信号。
- 复杂度：避免多层嵌套与多分支，通过“触发→动作”有限状态机扁平化控制流。
- 风险点：回退改变轨迹；默认仅重采样，回退需显式开启，以守住兼容性底线。

【Linus式方案】
1) 先把数据结构简化：EMA 统计独立、触发枚举统一；
2) 消除特殊情况：S1–S3 同一动作（重采样），S4 单一策略（回退）；
3) 用最笨但清晰的方法：重采样重做，不去"修样本"；
4) 确保零破坏性：默认只重采样，回退关闭；checkpoint 默认关闭。

---

### 实施状态

已实施（v0.0.4-dev）。所有三个阶段（Checkpoint、S1-S3 重采样、S4 回退）均已合入 `optimizer_impl.h` 主循环。

### 已知限制

1. **AdaGrad/Adam 与 S3 重采样的交互**：S3 检测位于算法 dispatch 之后（dispatch 已更新 accumulator），若对非 SR 算法启用 S3，
   重采样会导致 accumulator 被 double-update。当前实现通过 `is_sr` 守卫避免此问题（S3 仅对 SR 生效），
   但如果未来将 S3 扩展到其他算法，需在 dispatch 前检测或保存/恢复 accumulator 快照。

2. **S4 回退仅在 master rank 恢复 `current_state`**：非 master rank 的 `current_state` 在回退后过期，
   依赖 `energy_evaluator` 在下次调用时从 master 广播最新状态。这是设计上的不变量，非缺陷。


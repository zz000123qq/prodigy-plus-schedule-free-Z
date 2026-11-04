# 改进计划（Improvement Plan）

> 本文件是**唯一权威的计划书**，用于跨会话续接。任何后续工作以本文件为准。
> 上游基线：`LoganBooker/prodigy-plus-schedule-free` @ `338cf40`（v2.0.1，2025-09-27 停更）
> 分支：`audit-fix`（fork 后自建，不向上游提 PR）
> 制定日期：2026-11-04

---

## 一、背景与目标

上游最后一次提交是 2025-09-27，此后无更新。经三轮源码审计（含与两份外部 AI 复审的交叉核对），结论是：

- **代码没有腐烂**：纯 PyTorch、零外部依赖，今天仍可运行。
- **有 1 个确认缺陷 + 4 个 P1 级隐患**，全部可静态定位。
- **算法侧落后一代半**：停在 2025 年中的 Refined SF-Adam（arXiv 2507.09846），未跟进 2026-08 的 ScheduleFree+（arXiv 2605.19095）。

本计划的目标分两层：

| 层 | 目标 | 是否本次执行 |
|---|---|---|
| L1 正确性/工程 | 修掉确认缺陷、消除性能隐患、补齐工程元数据 | **本次执行** |
| L2 算法 | 对齐 ScheduleFree+ 六项修正 | 仅记录，单独排期 |

---

## 二、本次要做的修复（L1）

### T1 `use_cautious` 掩码是死代码 【P0，必修】

- **位置**：`prodigyplus/prodigy_plus_schedulefree.py:329-333`
- **现状**：
  ```python
  mask = grad.mul(exp_avg).sign_().clamp_min_(0)
  mask.div_(mask.mean().clamp(min=1e-3))
  grad.mul_(exp_avg)      # ← mask 算完从未被使用
  del mask
  ```
- **判据**：`mask` 在 330-331 完整计算并归一化，332 未引用，333 直接 `del`。同函数 `else` 分支是 `grad.copy_(exp_avg)`，SF 路径对应写法是 `u.mul_(mask)`。这是死代码，不可能是设计意图。
- **影响**：更新量级从 `O(1)` 变成 `O(|g|)`。典型 `|g|~1e-2` 时步长塌成 1/100 → Prodigy 误判"进展不足" → `d` 过估。仅在 `use_schedulefree=False` + `use_cautious=True` 时触发。
- **修法**：`grad.copy_(exp_avg).mul_(mask)`
  - ⚠️ **不是** `grad.mul_(mask)`——此时 `grad` 仍是原始梯度，那样写会丢掉一阶动量。
- **验收**：`use_cautious=True` 下 `update` 的 RMS ≈ `use_cautious=False` 时的 RMS（同一组随机输入，误差 <5%）。

### T2 SPEED 模式的 `.item()` 主机同步 【P1，必修】

- **位置**：`prodigyplus/core_optimiser.py:400-420`（`use_speed` 分支）、`287-288`
- **现状**：`state['exp_avg_x0_dot'].item()`、`max_x0_minus_l1_norm.item()`、`max_x0_minus_l2_norm.item()` 三个标量以 Python float 形式存回 state，每参数每步触发 3 次 GPU→CPU 同步。普通模式另有 2 次/step。
- **影响**：几百个参数 = 每步上千次阻塞同步，摧毁流水线重叠，且使 CUDA Graph 捕获不可能。
- **修法**：三个标量改为 device 上的 0-dim tensor（与 `running_d_numerator` 一致）；`update_d_stats_and_reset` 里把两次 `.item()` 合并为一次 `torch.stack(...).tolist()`。
- **兼容性**：`load_state_dict` 旧 checkpoint 时这些键是 float，`clamp_min(float)` 依然可用，仅首次 step 后自动转回 tensor。
- **验收**：`use_speed=True` 时每步 `.item()` / `.tolist()` 调用次数为 1（可用 monkeypatch 计数）。

### T3 `fused_back_pass` 下参数计数器错位 【P1，已修】

- **位置**：`prodigyplus/core_optimiser.py` 的 `on_start_step` / `on_end_step`
- **背景**：计数器按**全部参数**（含冻结层）初始化，但 `fused_back_pass=True` 时 `step()` 直接 `break`，靠反向钩子逐个触发 `step_param`。若驱动方只为有梯度的参数调钩子，每轮的实际调用次数就少于计数器初值。
- **实际症状（已实测，修正早期的"永久冻结"描述）**：计数器归零后，会被**下一个参数的 `on_start_step` 重新初始化**，因此不是冻结，而是**步边界落在任意位置、`k` 的推进速率与真实步数脱钩**。n=10、12 轮的实测：

  | 有梯度参数 | 修复前 k | 修复后 k | 修复前 k/轮 |
  |---|---|---|---|
  | 10/10 | 12 | 12 | 1.00 |
  | 9/10 | 10 | 11 | 0.83 |
  | 5/10 | 6 | 11 | 0.50 |
  | 2/10 | 2 | 11 | 0.17 |

- **后果**：`ckp1 = 1/k` 变慢 → Schedule-Free 的平均窗口被拉长；`d` 的统计在错误的参数集上结算；`prodigy_steps` 的触发点整体偏移。**静默失效，训练能跑完但结果不对。**
- **修法**：引入"本 step 已见参数集合"。`on_start_step(p)` 发现 `id(p)` 已在集合里 → 说明上一步没正常收尾 → 强制收尾并重新上膛。无梯度参数本来就不贡献统计，因此提前收尾是安全的。
- **验收**：`audit/sim_step_counter.py` —— 任意冻结比例下 `k` 都稳定每轮 +1。

### T4 `orthograd_` 的数值边界与非连续张量 【P2，必修】

- **位置**：`prodigyplus/core_optimiser.py:123-136`
- **问题**：① `g_orth.norm(2).add_(1e-30)` 在 `g ∥ w` 时会让缩放因子达到 `g_norm·1e30`，fp32 下可溢出成 `inf` → `0*inf = nan`；② `p.view(-1)` 对非连续参数（channels_last / `expand`）直接抛错。
- **修法**：`add_(1e-30)` → `clamp_min_(1e-30)`；`view(-1)` → `reshape(-1)`。
- **验收**：构造 `g = k·w` 的退化输入，断言输出无 NaN；对 channels_last 张量断言不抛错。

### T5 分布式：FSDP/ZeRO 分片下的 `d` 统计规约 【P1，实现但不验证】

- **位置**：`prodigyplus/core_optimiser.py:271-302`（`update_d_stats_and_reset`）
- **背景**：仓库内**零** `torch.distributed` 引用。
  - **DDP 不需要**：backward 已把梯度 all-reduce 平均，各卡 `p.grad` 相同 → `d` 天然一致。
  - **FSDP/ZeRO 需要**：各卡只持有不同参数分片，本地统计不同 → `d` 逐卡不同 → 各分片有效 LR 不一致。
- **修法**：在 `update_d_stats_and_reset` 里对 `running_d_numerator/denom` 做 `all_reduce(SUM)` 后除以 world_size；仅当 `torch.distributed.is_initialized()` 且新开关 `sync_d_stats=True` 时启用。
- **验收**：**本机无多卡，不做验证**，代码路径用 `dist.is_initialized()` 保护，单卡不受影响。

### T6 工程元数据与坏味道 【P2，必修】

| 项 | 位置 | 修法 |
|---|---|---|
| `VERSION=(2,0,0)` 与包版本 2.0.1 不一致 | `core_optimiser.py:5` / `setup.py:9` | 统一为单一来源；本次因 T1 改变数值行为，升到 `2.1.0` |
| 裸 `except:` | `core_optimiser.py:554, 567` | → `except Exception:` |
| `__init__.py` 导入路径错（靠碰巧可用） | `prodigyplus/__init__.py` | → `from .core_optimiser import CoreOptimiser`，并导出 `__version__` |
| `python_requires='>=3.4'` | `setup.py:29` | → `>=3.9` |
| `pyproject.toml` 无 `[project]` 元数据 | `pyproject.toml` | 补最小可用的 `[project]` 表 |
| README 中 `eps=1e8` | `README.md:82` | → `1e-8` |

---

## 三、明确不做的事

| 项 | 理由 |
|---|---|
| 多卡 / FSDP 实测 | 当前无多卡环境，短期用不到。T5 只做代码路径，不做验证 |
| 重新引入 Muon | 正交化输出谱范数固定，与 Prodigy 的 `d` 口径冲突；且 LLM 预训练赛道已被 NeMo Emerging-Optimizers 等占住。属研究项目 |
| GaLore / APOLLO | 范畴错误：本库二阶矩已压到 O(n+m)，GaLore 针对的是低秩梯度，省不到关键处 |
| WSD 调度支持 | 引入 WSD 等于承认"还是需要 schedule"，与项目定位矛盾 |
| 引入 `foreach` / `fused` CUDA kernel | 收益不确定（step 仅占 e2e 5~10%），改造成本高。先做 T2 的同步消除 |
| 向上游提 PR | 上游已停更，自建分支维护 |

---

## 四、已核实为「非缺陷」的项（防止重复劳动）

| 曾经的怀疑 | 核实结论 |
|---|---|
| `weight_sum` 存在 `param_group` 上导致 `ckp1 ∝ 1/张量数` | **误报，不存在**。`weight_sum = group['running_weight_sum'] = group.get('weight_sum', 0) + weight` 是链式赋值，右侧读的是 **step 内不变**的 `group['weight_sum']`，`running_weight_sum` 只是暂存被重复覆盖。实测 `ckp1 = 1/k`，与 n 无关，与 per-param state 等价。**不要重构进 `state[p]`** |
| DDP 缺 `all_reduce` 导致各卡 `d` 分歧 | **对 DDP 不成立**。DDP 的梯度已在 backward 平均，`d` 天然一致。只有 FSDP/ZeRO 分片需要（见 T5） |
| `d = max(d, ...)` 是"单调自锁缺陷" | **不成立**。这是 Prodigy 的理论前提（`d` 非递减才有无调参保证），LR 下降由 Schedule-Free 的平均承担。属设计取舍 |
| `[::11]` 切片破坏合并访存、与隐藏维共振 | **两条论据均不成立**。读 1/11 元素但 cache line 利用率同比例下降，实际取回字节 ≈ 全量；切片在扁平化后按固定步长取，与张量形状无关。代价只是方差变大 |
| `p.float()` 造成显存碎片 | **不成立**。caching allocator 对同尺寸块复用高效；真实开销是峰值和分配器调用次数 |

---

## 五、执行阶段与提交粒度

每个修复一个 commit，便于单独回滚：

| # | commit message | 涉及文件 |
|---|---|---|
| 0 | `docs: add improvement plan` | `docs/IMPROVEMENT-PLAN.md` |
| 1 | `fix(cautious): apply the mask instead of discarding it` | `prodigy_plus_schedulefree.py` |
| 2 | `perf(speed): keep d-stat scalars on device, drop per-param syncs` | `core_optimiser.py` |
| 3 | `fix(fbp): self-heal parameter counter when params have no grad` | `core_optimiser.py` |
| 4 | `fix(orthograd): clamp instead of add epsilon, use reshape for non-contiguous` | `core_optimiser.py` |
| 5 | `feat(dist): optional all_reduce for d stats under sharded parallelism` | `core_optimiser.py` |
| 6 | `chore: unify version, fix imports, metadata, readme typo` | `setup.py` / `pyproject.toml` / `__init__.py` / `README.md` |

全部改完后打 tag `v2.1.0-audit`。

---

## 六、L2 待办：对齐 ScheduleFree+（不在本次范围）

来源：Aaron Defazio, *ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to LLMs*, arXiv 2605.19095（2026-08）。

| # | 修正 | 本库现状 | 改造量 |
|---|---|---|---|
| 1 | 恢复内层动量（β₁=0.75~0.9） | SF 路径无 `exp_avg`，`step_param_schedulefree` 不调 `update_first_moment` | 大：需新增 `inner_beta` 并给 SF 路径建一阶动量（+50% 状态显存） |
| 2 | Polyak 步长 + 平滑 L1 梯度范数 | 用 Prodigy 的 `num/denom` 比率 | 大：新增 `lr_mode='polyak'` 平行分支 |
| 3 | c-warmup（延迟启用平均） | 从第 1 步就平均 | 小：`k <= c_warmup` 时 `ckp1 = 1.0` |
| 4 | 外层 β 退火（0.8/0.9 → 0.965） | **结构上做不到**：`beta1` 同时是内层动量系数与 SF 平均系数 | 依赖 #1 解耦后才能做 |
| 5 | r=1 幂加权平均（`w = t^r · γ_max^p`） | 只有 `w = dlr²`，无 `t^r` 项 | 小：`weight = (k ** r) * dlr ** p` |
| 6 | 全解耦 AdamC WD（γ²λ） | `weight_decay_by_lr=True` 时是 γλ | 小：改成 `decay * dlr` → `decay * dlr**2`，但 λ 需放大到 5~50 |

**依赖顺序**：#1 必须先做（#4 依赖它）。#3 / #5 / #6 互不依赖，可并行。

**收益（论文口径）**：长跑 1000 tokens/参数下，相对最强 Linear Decay 基线训练时间 −31%。

---

## 七、分支与流程

```
main（上游基线，只读）
└── audit-fix   ← 本计划的所有改动都在这个分支
```

- 仓库：fork 后自建，`origin` 指向上游，新增 `fork` remote 指向自己的仓库。
- 提交规范：`fix/` `perf/` `feat/` `chore/` `docs/` 前缀。
- 不 rebase、不 force push，保持线性历史。
- 不向上游提 PR。

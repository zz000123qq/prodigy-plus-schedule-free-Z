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
  - 注意：不是 `grad.mul_(mask)`——此时 `grad` 仍是原始梯度，那样写会丢掉一阶动量。
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

### T4 `orthograd_` 的非连续张量崩溃 【P2，已修；NaN 一说已证伪】

- **位置**：`prodigyplus/core_optimiser.py` 的 `orthograd_`
- **真问题（已复现）**：`p.view(-1)` 对非连续参数抛 `RuntimeError: view size is not compatible...`。`channels_last` 的卷积权重、`expand` 出来的张量都会踩到。实测 v2.0.1 直接崩，改 `reshape(-1)` 后正常。
- **附带清理**：`sub_` / `mul_` 原地改写会污染调用方传入的 `update` 张量（当前调用点恰好安全），改为非原地。
- **已证伪**：原以为 `g_orth.norm(2).add_(1e-30)` 在 `g ∥ w` 时会溢出成 `inf` 导致 `0*inf = nan`。实测 scale 从 1e4 到 1e9，输出全部有限——浮点减法留下 ~1e-16 的相对残差，缩放因子上限约 1e16，碰不到 `inf`。且 `add_(1e-30)` 与 `clamp_min(1e-30)` 在 `norm=0` 时**完全等价**，这条改动不改变任何行为，仅统一写法。
- **验收**：`tests/test_audit_fixes.py::test_orthograd_accepts_non_contiguous_params`（旧版 RuntimeError，新版通过）。

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
| `VERSION=(2,0,0)` 与包版本 2.0.1 不一致 | `core_optimiser.py:5` / `setup.py:9` | 新增 `prodigyplus/_version.py` 作为唯一来源，`VERSION` 由它推导；本次因 T1 改变数值行为，升到 `2.1.0` |
| 裸 `except:` | `core_optimiser.py:554, 567` | → `except Exception:` |
| `__init__.py` 导入路径错（靠碰巧可用） | `prodigyplus/__init__.py` | → `from .core_optimiser import CoreOptimiser`，并导出 `__version__` |
| `python_requires='>=3.4'` | `setup.py:29` | → `>=3.9` |
| `pyproject.toml` 无 `[project]` 元数据 | `pyproject.toml` | **已补全 PEP 621 `[project]` 表**，version / readme 走 `dynamic`（`version` 从 `prodigyplus/_version.py` AST 静态解析，构建时不会 import 包、不需要 torch）。`setup.py` 退化为 `setup()` 空壳以兼容旧调用。实测 `pip wheel` 构建通过，产物 `prodigy_plus_schedule_free-2.1.0-py3-none-any.whl` |
| README 中 `eps=1e8` | `README.md` | → `1e-8`；补 v2.1.0 变更日志 |

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

## 五、实际提交记录

每个修复一个 commit，便于单独回滚。分支 `audit-fix`，tag `v2.1.0-audit`。

| commit | 内容 |
|---|---|
| `e68f51f` | `docs:` 计划书 + `.gitignore` |
| `8a12259` | `fix(cautious):` 死掩码 → `grad.copy_(exp_avg).mul_(mask)` |
| `c380b19` | `perf(speed):` d-stat 标量留在 device，去掉每参数同步 |
| `69d9e0d` | `fix(orthograd):` `reshape` 替代 `view`（NaN 一说已证伪，见 §八） |
| `52125e3` | `fix(fbp):` 步边界自愈 |
| `86a9dc2` | `feat(dist):` `sync_d_stats` |
| `da626c4` | `chore:` 版本单一来源、导入、元数据、README |
| `0085c6b` | `test:` 9 条真实张量回归 + orthograd 结论更正 |
| 本次 | `fix(packaging):` 补全 PEP 621 `[project]`，修复上一提交造成的构建失败 |

> 注：`da626c4` 里的 `[tool.setuptools]` 在缺少 `[project]` 表时会触发 setuptools 82 的
> `ValueError: project must contain ['version']`，构建直接失败。补全 `[project]` 后
> `pip wheel` 实测通过。**这是先改后验的代价，记在这里当教训。**

## 六、L2 待办：对齐 ScheduleFree+（不在本次范围）

来源：Aaron Defazio, *ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to LLMs*, arXiv 2605.19095（2026-08）。

| # | 修正 | 本库现状 | 改造量 |
|---|---|---|---|
| 1 | 恢复内层动量（β1=0.75~0.9） | SF 路径无 `exp_avg`，`step_param_schedulefree` 不调 `update_first_moment` | 大：需新增 `inner_beta` 并给 SF 路径建一阶动量（+50% 状态显存） |
| 2 | Polyak 步长 + 平滑 L1 梯度范数 | 用 Prodigy 的 `num/denom` 比率 | 大：新增 `lr_mode='polyak'` 平行分支 |
| 3 | c-warmup（延迟启用平均） | 从第 1 步就平均 | **已完成**：`c_warmup=N` 参数，默认 0（行为不变） |
| 4 | 外层 β 退火（0.8/0.9 → 0.965） | **结构上做不到**：`beta1` 同时是内层动量系数与 SF 平均系数 | 依赖 #1 解耦后才能做 |
| 5 | 幂加权平均（`w = t^r · γ^p`） | 只有 `w = dlr^2` | **已完成**：`schedulefree_r=r`，默认 0。论文说 r=1 仅适合长跑 |
| 6 | 全解耦 AdamC WD（γ^2λ） | `weight_decay_by_lr=True` 时是 γλ | 小：改成 `decay * dlr` → `decay * dlr**2`，但 λ 需放大到 5~50 |

**依赖顺序**：#1 必须先做（#4 依赖它）。#6 独立。

**状态**：#3 和 #5 已作为可选开关实现（默认关闭，不影响现有数值）。#1 / #2 / #4 / #6 未做。
#1 需要给 SF 路径补 `exp_avg`，实测会多占 85% 的状态显存（`exp_avg` 与 `z` 同尺寸），对显存紧张的 LoRA 场景要权衡。

**收益（论文口径）**：长跑 1000 tokens/参数下，相对最强 Linear Decay 基线训练时间 -31%。

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

---

## 八、执行状态（2026-11-04）

| 项 | 计划 | 状态 | 证据 |
|---|---|---|---|
| T1 `use_cautious` 死掩码 | 修 | 已修，已实测 | `zero_frac` 0.000 → 0.679；步长 5.48e-4 → 1.74e-2（旧版仅为 unmasked 对照的 1/20） |
| T2 SPEED 每参数 `.item()` | 合并 | 已修，已实测 | 单步同步次数 14 → 1（4 个参数） |
| T3 FBP 参数计数器 | 修 | 已修，已实测 | 9/10 有梯度时 `k` 每轮 +1 |
| T4 `orthograd` 非连续张量 | 修 | 已修，已实测 | `channels_last` 权重：`RuntimeError` → 正常。（原列的 NaN 一说不成立，见下表） |
| T5 `sync_d_stats` | 实现、不验证 | 已实现，仅单卡冒烟（未验证） | 单卡 `dist` 未初始化时为空操作，整步跑通；按约定不做多卡验证 |
| T6 工程元数据 | 全部 | 已完成 | `pip wheel` 构建通过；import 正常；version 2.1.0 |
| 回归测试 | 建 `tests/` | 14 条 | `python tests/test_audit_fixes.py` → 14/14，且逐条对照 v2.0.1 验过判别力 |
| L2 ScheduleFree+ 六项 | 不做 | 未开始 | 见 §六，需单独立项 |

**执行中修正的计划内容**（原计划有误，已按实测改写）：

| 原计划 | 实际情况 |
|---|---|
| §2.1 `weight_sum` 作用域错 → `ckp1 ∝ 1/(n·k)`，要重构进 `state[p]` | **误报**。链式赋值右侧读的是本 step 内不变的 `group['weight_sum']`，实测 `ckp1 = 1/k`、与参数个数无关。**未做，也不该做** |
| T4 `add_(1e-30)` 会溢出成 `inf` → NaN | **不成立**。scale 1e0~1e9 输出全部有限；且 `add_` 与 `clamp_min` 在 `norm=0` 时完全等价，该改动仅为写法统一 |
| `setup.py` 作为元数据唯一来源 | **已反转**。PEP 621 要求 `[project]` 表存在，`setup.py` 改为兼容空壳 |


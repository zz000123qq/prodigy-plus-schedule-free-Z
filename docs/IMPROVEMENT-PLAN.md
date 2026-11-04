# 改进计划（权威版）

> 本文是唯一权威版本，取代此前的逐项审计记录。审计过程见 `README.md` 附录与 git log，这里只保留"已经做完的"和"还值得做的"。
>
> 基线：上游 `LoganBooker/prodigy-plus-schedule-free` v2.0.1（`338cf40`，2025-09 停更）。
> 本分支：`audit-fix`，版本 `2.1.0`，远端 `zz000123qq/prodigy-plus-schedule-free-Z`。
> 测试：`python tests/test_audit_fixes.py`，15 条，本机 Python 3.13 + torch 2.9.0；CI 见 `.github/workflows/tests.yml`。

---

## 一、已完成（不再讨论）

| 项 | 内容 | 实测证据 |
|---|---|---|
| `use_cautious` 死掩码 | 算出的 mask 从未使用，更新被乘上一阶动量。改为 `grad.copy_(exp_avg).mul_(mask)` | 掩码置零比例 0.000 到 0.679；步长 5.5e-4 到 1.7e-2（旧版只有 unmasked 对照的 1/20） |
| SPEED 模式 `.item()` | 每参数 3 次主机同步改为设备端标量，两步 `.item()` 合并为一次 `stack().tolist()` | 4 参数单步同步次数 14 到 1 |
| `fused_back_pass` 计数器 | 无梯度参数不触发钩子，导致步边界错位、`k` 推进速率脱钩。改为"本步已见参数集合"自愈 | 10 参数里 2 个有梯度：`k` 从 2（12 轮）修正为每轮 +1 |
| `orthograd_` 非连续张量 | `view(-1)` 对 `channels_last` 权重直接抛 `RuntimeError`，改 `reshape(-1)` | 同一权重：异常到正常 |
| `sync_d_stats` | 新增开关（默认开），FSDP/ZeRO 分片下规约 `d` 统计 | 单卡冒烟通过；多卡未验证 |
| `betas[1]=None` | 原来构造通过、第一步 `TypeError`，改为构造时 `ValueError` | 用例覆盖 |
| `d_prev` | 写 3 次读 0 次，删除 | 用例覆盖 |
| `out=` 缓冲复用 | 纯 Prodigy 路径把更新直接写进梯度缓冲，省一次全量读写 | fp32 step 10.82 到 10.01 ms（7.5%）；13 种配置 25 步逐位一致 |
| `c_warmup` / `schedulefree_r` | ScheduleFree+ 的 #3 和 #5，可选开关，默认关闭 | `c_warmup=5` 时前 5 步 y 等于 z；`schedulefree_r=1` 在 k=10 把 ckp1 从 0.100 提到 0.182 |
| 工程元数据 | 版本单一来源、PEP 621 `[project]`、`python_requires>=3.9`、裸 `except`、`__init__` 导入 | `pip wheel` 构建通过 |

版本号仍为 `2.1.0`，不再随文档改动上调。tag `v2.1.0-audit` 是发布快照，不跟着每次提交走。

---

## 二、剩余可升级项

### A 档：已完成（2026-11-04）

| # | 项 | 结果 |
|---|---|---|
| A1 | 端到端收敛对比（迷你 UNet 去噪，600 步 x 3 种子） | 已完成，表在 README 5.4。三个实质结论：cautious 的 bug 会让训练完全失效（`d` 卡在 `d0`）；`c_warmup=100` 稳定好 16% 到 18%；cautious 本身在这个任务上无可测收益 |
| A2 | CI | 已完成：`.github/workflows/tests.yml`，Python 3.10/3.12/3.13 + CPU torch，跑测试、两个逻辑脚本和构建。无 GPU 环境下 15/15 通过已验证 |
| A3 | SF 路径 cautious / grams 只作用于 y | **不是缺陷**。上游参考实现（facebookresearch/schedule_free#54）明确写了"z 的更新不含动量项，掩码它没有意义"，本库与之一致。仅更正文档 |
| A4 | `use_adopt` 在 SF 路径缺 clamp | 改为构造时打印警告（那条路径没有一阶动量，硬补 clamp 等于猜语义）。含测试 |
| A5 | 文档补数 | 已完成：`prodigy_steps` 省 15%、step 新表、论文对应附录 |

### B 档：收益明确，但要动结构

| # | 项 | 收益 | 代价 |
|---|---|---|---|
| B1 | ScheduleFree+ #1：给 SF 路径补内层动量 | 解决大 batch 的 scaling cliff；**解锁 #4 的 β 退火**（现在 `beta1` 一个旋钮两用，退火做不了） | 状态显存 +85%（75.7 到 139.7 MB）。LoRA 场景要权衡 |
| B2 | ScheduleFree+ #6：权重衰减改全解耦 `γ^2λ` | 长跑数值更稳 | 小改动，但 λ 要放大 5 到 50 倍重标定；`weight_decay=0` 时完全无感，扩散微调基本不触发 |
| B3 | ScheduleFree+ #2：Polyak 步长平行分支 | 不依赖 Prodigy 的 `d` | 研究性。与 `d` 二选一，需独立验证在噪声梯度下是否更稳 |
| B4 | step 的 kernel 化（`foreach` / `fused` / Triton） | 当前 FP32 33.6M 参数下 step 是 AdamW 的 1.50 倍（11.09 对 7.38 ms），理论上限接近 fused AdamW（0.54 倍） | 大。且**先要测端到端占比**：小模型上 step 占 76%，那是 Windows WDDM 每 kernel 开销放大的结果，真实 UNet 训练里低得多。占比不高时做这个是净亏 |
| B5 | SF 路径 `p.grad.to(float32, copy=True)` 的冗余拷贝 | fp32 参数下这次拷贝是纯开销，可省一次全量读写 | 中高风险：跳过会改变 `p.grad` 的别名关系，若调用方复用梯度会静默出错。需先确认驱动方行为 |

### C 档：环境依赖或暂缓

| # | 项 | 卡在哪 |
|---|---|---|
| C1 | 多卡 `sync_d_stats` 验证 | 没有多卡环境。代码路径已就位，`dist` 未初始化时是空操作 |
| C2 | CUDA Graph | 还差属性赋值与剩余 `.item()`；且只有 step 占比高的场景才有意义 |
| C3 | `torch.compile` | 本机缺 MSVC，Triton 编不出 kernel；且对 `Optimizer` 实例做属性赋值会直接 `AssertionError`，不是 graph break。要支持得先重构成无状态函数 |

---

## 三、明确不做（附理由，防止重复提议）

| 项 | 理由 |
|---|---|
| 二阶矩 8-bit 量化 | 默认 `factored=True` 下二阶矩只占状态的 0.04%（32 KB / 75.7 MB），大头是 `z`（85%）。只有主动关掉 `factored` 才省 23% 到 36%，而关掉它的理由（画质）没有证据。纯 PyTorch 量化还会往 memory-bound 的 step 上加几趟遍历 |
| 重新引入 Muon | 正交化输出谱范数固定，与 Prodigy 的 `d` 口径冲突；LLM 预训练赛道已被有工程团队的实现占住 |
| GaLore / APOLLO | 范畴错误：本库二阶矩已压到 O(n+m)；GaLore 针对低秩梯度，省不到关键处；SF 本身无一阶动量可省 |
| WSD 调度 | 引入 WSD 等于承认"还是需要 schedule"，与项目定位矛盾 |
| 恢复"重置探索"机制 | Prodigy 论文 v4 的 Alg.3 第 10 行本身就是 `d = max(d, d_hat)`，单调是理论前提，不是遗漏。本库还额外加了 `d_limiter`（每步最多涨 19%），比论文更保守 |
| 向上游提 PR | 上游停更一年多，自建分支维护 |

---

## 四、已核实为「非缺陷」（防止重复劳动）

| 曾经的怀疑 | 结论 |
|---|---|
| `weight_sum` 存在 `param_group` 上，导致 `ckp1` 与张量数成反比 | **误报**。右侧读的是 step 内不变的 `group['weight_sum']`，实测 `ckp1 = 1/k`，与参数个数无关 |
| DDP 缺 `all_reduce` 导致各卡 `d` 分歧 | **对 DDP 不成立**，梯度已在 backward 平均。只有 FSDP/ZeRO 分片需要 |
| `d = max(d, ...)` 是"单调自锁缺陷" | 设计取舍，论文原样，见第三节 |
| `[::11]` 切片破坏合并访存、与隐藏维共振 | 两条论据都不成立。cache line 利用率同比例下降，取回字节约等于全量；切片在扁平化后按固定步长取，与张量形状无关。代价只是估计方差变大 |
| `p.float()` 造成显存碎片 | 不成立。caching allocator 对同尺寸块复用高效；真实开销是峰值 |
| `orthograd` 的 `add_(1e-30)` 会溢出成 inf 产生 NaN | **不成立**。scale 从 1e0 到 1e9 输出全部有限，浮点减法留下约 1e-16 相对残差，缩放因子上限约 1e16。且 `add_` 与 `clamp_min` 在 norm=0 时完全等价 |
| 按模块拆组、每组独立 `d` 能让不同深度/模块各走各的步长，是"结构感知"的关键 | **实测负收益**（README 5.4）：三组独立 `d` 让步长差 4 倍，eval loss 反而 +11.5%、方差翻倍；加 `split_groups_mean=True` 才回到持平。`d` 估计的是到最优点的距离，不是模块敏感度，拆开只会让每个估计更噪 |
| SF 模式 cautious 只过滤 y 不过滤 z，是语义不一致 | **不是**。上游 PR #54 的原话是 z 更新不含动量、掩码它没意义，cautious 操作应放在 y 更新上。本库照此实现 |
| 论文有"指数周期重置"，本项目删掉了 | **论文 v4 里 "resetting" 出现 0 次**。全文唯一一次 "restarting" 在 Related Work 讲 Polyak 步长 |
| 论文预警了扩散/噪声梯度下内积震荡 | 论文没有出现过 "nois" 词根，没有 SNR、没有 diffusion；理论前提是凸 + G-Lipschitz |

---

## 五、验证入口

```bash
# 回归测试（需要 torch；CUDA 用例应能跳过或用 CPU 版）
python tests/test_audit_fixes.py

# 不依赖 torch 的逻辑复现
python tools/logic_checks/sim_step_counter.py
python tools/logic_checks/sim_weight_sum.py

# 构建检查
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/pkgtest

# 端到端对比（迷你 UNet 去噪，约 15 分钟）
python tools/experiments/a1_denoise.py --steps 600 --seeds 3 --batch 8
```

改任何影响数值的代码前，先按"逐位对照旧实现"的方式验等价：用 `git show <commit>:prodigyplus/xxx.py` 导出旧版到临时目录，两版各跑 25 步比对参数张量，FP32 / BF16 与各种开关组合都要覆盖。

---

## 六、接续这份文档时

1. 先跑测试，确认基线是绿的。
2. `git log --oneline main..audit-fix` 看做到哪。
3. 新增结论只改本文，不要再写第二份计划书。
4. 任何关于"数值行为"的结论必须有实测，脚本要逐字复刻源码表达式。第四节里的四个误报全是凭语义直觉下判断造成的。

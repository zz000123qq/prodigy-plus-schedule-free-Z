# Prodigy + Schedule-Free（中文说明）

把 **Prodigy**（自动估计学习率）和 **Schedule-Free**（免学习率调度器）合在一起的 PyTorch 优化器。

- **上游项目**：https://github.com/LoganBooker/prodigy-plus-schedule-free（Apache-2.0，最后更新 2025-09）
- **本仓库**：个人维护的 fork，分支 `audit-fix`，版本 `2.1.0`。**不打算向上游提 PR**，也不发布到 PyPI。
- **状态**：实验性。下面的说明按"本分支实际代码能做什么"写，与上游 README 的措辞有出入的地方以本文为准。

---

## 一、本分支（2.1.0）相对上游 2.0.1 改了什么

| 改动 | 说明 |
|---|---|
| 修 `use_cautious` | 掩码算出来后被丢弃，实际执行的是 `grad × 一阶动量`，步长量级从 `O(1)` 变成 `O(|grad|)`。只在 `use_schedulefree=False` 时触发 |
| 修 `fused_back_pass` 计数器 | 有参数永远拿不到梯度时，步边界会落在任意位置，`k` 的推进速度与真实步数脱钩 |
| 减少主机同步 | `use_speed=True` 时每参数每步 3 次 GPU→CPU 同步，现在整步只剩 1 次（实测 4 参数场景 14 → 1） |
| 修 `use_orthograd` | `view(-1)` 对非连续张量（`channels_last` 卷积权重）会直接 `RuntimeError`，改 `reshape(-1)` |
| 新增 `sync_d_stats` | 分布式下把 d 的统计量跨卡平均。FSDP/ZeRO 分片必需；DDP 下是空操作 |
| 工程清理 | 版本号单一来源（`prodigyplus/_version.py`）、`python_requires>=3.9`、`[project]` 表、`__init__` 导入、裸 `except` |
| `betas[1]=None` 提前报错 | 原来能构造成功，第一次 `step()` 才在 `get_bias_correction` 里抛 `TypeError`。现在构造时就给出明确 `ValueError` |
| 删除死变量 `d_prev` | 原来写 3 次、读 0 次 |
| 新增测试 | `tests/test_audit_fixes.py`，9 条真实张量回归 |

完整审计、改动理由、以及**已经论证过但决定不做的事**见 `docs/IMPROVEMENT-PLAN.md`。

---

## 二、安装

```bash
pip install -e .
```

只依赖 `torch>=2.0`，Python 3.9+。设计上只支持 **bf16 / fp32**，其他 dtype 会打印警告且不保证正确。

---

## 三、最小用法

```python
from prodigyplus import ProdigyPlusScheduleFree

opt = ProdigyPlusScheduleFree(model.parameters(), lr=1.0)

for epoch in range(epochs):
    model.train()
    opt.train()                       # 必须：训练前切到 train
    for x, y in loader:
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    model.eval()
    opt.eval()                        # 必须：评估前切到 eval，否则指标不对
    # ... 评估 ...
```

**三条铁律**：

1. **`lr` 保持 1.0**。它只是外部缩放系数，真正的学习率由 Prodigy 算出的 `d` 决定。
2. **`train()` / `eval()` 必须配对调用**，且要在 `model.train()` / `model.eval()` 旁边。漏掉 `eval()` 会让评估结果失真（看到的是 y 点而不是平均点 x）。
3. **不要用外部梯度裁剪**。内部已经做了更新量缩放，外面再裁会打乱 Prodigy 对步长的估计。想让内部缩放更温和就调 `use_stableadamw` 或 `eps`。

---

## 四、参数速查

### 4.1 常用（一般只调这几个）

| 参数 | 默认 | 说明 |
|---|---|---|
| `lr` | `1.0` | 外部缩放系数，**保持 1** |
| `betas` | `(0.9, 0.99)` | 动量系数。Schedule-Free 下社区建议 `(0.95, 0.99)`，作者仍保留 0.9 作默认 |
| `weight_decay` | `0.0` | 解耦权重衰减 |
| `weight_decay_by_lr` | `True` | WD 是否乘自适应学习率（PyTorch AdamW 的行为）。不懂就别动 |
| `d0` | `1e-6` | Prodigy 的初始学习率猜测。LR 涨不上去就试 `1e-5`/`1e-4`；一直偏大就试 `1e-7` |
| `d_coef` | `1.0` | 调 LR 的首选旋钮，`0.5` 和 `2.0` 都常用 |
| `d_limiter` | `True` | 限制 `d` 每步涨幅（不超过 2^0.25 倍），避免早期高估 |
| `prodigy_steps` | `0` | 大于 0 时在第 N 步**冻结** `d` 并释放 Prodigy 专用显存。长跑或扩散模型训练常用；短跑不需要 |
| `schedulefree_c` | `0` | Refined SF-Adam 的平均强度。`0` = 关闭（用原始 Schedule-Free 规则）。小 batch 短跑试 6~12，长跑试 50~200 |
| `eps` | `1e-8` | 分母稳定项。设成 `None` 则改用 **Adam-atan2**（尺度不变、免调 eps），会顺带关掉 `use_stableadamw` |
| `use_schedulefree` | `True` | 设 `False` 退回纯 Prodigy，此时**需要自己配学习率调度器**（推荐 cosine） |

### 4.2 按场景才需要动

| 参数 | 默认 | 什么时候用 |
|---|---|---|
| `split_groups` | `True` | 每个 param_group 单独算 `d`。同时训多个网络（如 text encoder + UNet）时有用。只有一个 param_group 时自动禁用 |
| `split_groups_mean` | `False` | 各组 `d` 取调和平均后再乘各组自己的 lr。想复刻老版本行为时开 |
| `factored` | `True` | Adafactor 式二阶矩分解，省显存。出现 NaN 或 LR 涨不动时关掉 |
| `factored_fp32` | `True` | 分解后的二阶矩强制 fp32。只在显存极度紧张且短训时设 `False` |
| `stochastic_rounding` | `True` | bf16 权重写回时用随机舍入，效果更接近 fp32。**只对 bf16 参数生效** |
| `use_stableadamw` | `True` | 按更新量的 RMS 做软缩放（只压不放）。LR 一直高估时关掉，改用外部裁剪 |
| `use_bias_correction` | `False` | RAdam 式偏差修正 + 自动 warmup。**会显著拖慢 `d` 的收敛**（可能慢 10 倍），可用 `use_speed` 部分抵消 |
| `fused_back_pass` | `False` | 用 fused backward pass 时必须置 `True`，让优化器跳过常规 `step()`。需要训练脚本/UI 侧支持 |
| `sync_d_stats` | `True` | 分布式下跨卡平均 d 统计。FSDP/ZeRO-2/3 必需；纯 DDP 下多余但无害 |

### 4.3 实验性功能 —— 在本库中收益有限，多数默认关闭

这几个都来自 2024~2025 年的论文小技巧，作者自己在 docstring 里写了 "may have a limited effect"。**建议先不用**。

| 参数 | 在本库的真实状态 |
|---|---|
| `use_speed` | 作者自创（Simplified Prodigy with rElativE D），用动量式方向进展替代 num/denom。省显存、尺度不敏感，但**与 weight_decay 同用容易失稳**，超长训练也容易飘 |
| `use_orthograd` | 用正交于权重方向的梯度分量更新。两种模式都可用。注意：与权重平行的分量会被完全丢弃 |
| `use_focus` | FOCUS 更新规则。**强制关闭 `factored`（显存上升）并禁用 Adam-atan2** |
| `use_cautious` | C-Optim。⚠️ **Schedule-Free 模式下是半实现**：掩码只作用于 y，z 仍然吃未过滤的更新（见第五节） |
| `use_grams` | Grams，问题同 `use_cautious`（与它互斥，同开会自动关掉 grams） |
| `use_adopt` | ADOPT 的**残缺实现**。Schedule-Free 模式没有一阶动量，只剩"分母延后一步 + 首步跳过参数更新" |

---

## 五、这些功能其实"用不上"——按代码实际行为说明

写这份文档时逐项核对过源码，以下几点与上游 README 给人的印象不同：

1. **默认配置下根本没有一阶动量。**
   `use_schedulefree=True` 时 `initialise_state` 只建 `z`，不建 `exp_avg`。所以 `use_cautious` / `use_grams` / `use_adopt` 这些**依赖动量的技巧全部退化**——作者在 docstring 里也承认 "we do not have access to a first moment"。要用这些技巧，实际得设 `use_schedulefree=False`。

2. **Schedule-Free 模式下 `cautious` / `grams` 只过滤了一半。**
   `update_params` 里先 `z.sub_(update)`，之后才给 `u` 上掩码。结果是 **z 吃未过滤更新、y 吃过滤后更新**，与 C-Optim 的本意不一致。纯 Prodigy 模式反而是完整的。

3. **`use_adopt` 的 clamp 只在纯 Prodigy 模式存在**（`clamp_range = k**0.25`），SF 模式没有。

4. **README 里 `prodigy_steps` 建议取总步数 5~25% 的说法已作废**（上游自己在新版里写了"现在不需要了"）。

5. **MNIST 那组 98.5%~99.3% 的准确率是 v1.x 时代的数据**。v2.0 把内部实现整个重写过（上游明确写了"不要用新版续训老版本"），这组数字**不代表 2.x 的效果**，我也没复现。

6. **Muon 已被移除**，且不建议加回来：正交化输出谱范数固定，与 Prodigy 依赖二阶矩口径的 `d` 直接冲突。

---

## 六、与 ScheduleFree+（2026）的差距

上游停在 2025 年中的 *Refined SF-Adam*（arXiv 2507.09846）。2026-08 Defazio 本人的 **ScheduleFree+**（arXiv 2605.19095）给出 6 项让它扩展到 LLM 的修正，本库**一项都没有**：

| # | 修正 | 本库现状 |
|---|---|---|
| 1 | 恢复内层动量（β₁≈0.75~0.9） | SF 路径无 `exp_avg`，大 batch 会有 scaling cliff |
| 2 | Polyak 步长 + 平滑 L1 梯度范数 | 走的是 Prodigy 的 num/denom 比率 |
| 3 | c-warmup（延迟启用平均） | 从第 1 步就平均 |
| 4 | 外层 β 退火（0.8/0.9 → 0.965） | **结构上做不到**：`beta1` 同时是内层动量系数和 SF 平均系数 |
| 5 | r=1 幂加权平均（`w = t^r·γ^p`） | 只有 `w = dlr²` |
| 6 | 全解耦 AdamC 权重衰减（γ²λ） | 是 γλ |

论文口径的收益：长跑 1000 tokens/参数下比最强 Linear Decay 基线省 31% 训练时间。**这六项没做，是下一步的主要工作**，改造顺序和依赖关系见计划书第六节。

---

## 七、适用 / 不适用

**适合**：扩散模型微调、LoRA、小到中等规模任务的"不想调 LR 和 scheduler"场景。这类任务里"随时停训、随时存盘"是硬需求，Schedule-Free 的价值最大。

**不适合**：

- **LLM 预训练**。2026 年这个领域的主流是 Muon / SOAP（NVIDIA 在数万亿 token 上实测优于 AdamW），本项目主动放弃了这条线。
- 需要精确复现某个已知 LR 曲线的工作。
- 追求单步速度极限的场景：本优化器是逐参数 Python 循环，没有 `foreach`／`fused` kernel 实现。

---

## 八、测试

```bash
python tests/test_audit_fixes.py
```

不需要 pytest。9 条用例覆盖：cautious 掩码、SPEED 同步次数、FBP 计数器、orthograd 非连续张量、ckp1 = 1/k、train/eval 往返、单卡无进程组、CUDA bf16 整步。

每条都对照上游 v2.0.1 验证过判别力（例如 cautious 的步长在旧版是对照组的 1/20）。**多卡 `sync_d_stats` 没有验证过**——本机无多卡环境。

---

## 九、致谢与许可

Apache-2.0，见 `LICENSE`。核心算法来自：

- https://github.com/konstmish/prodigy
- https://github.com/facebookresearch/schedule_free
- Adafactor 分解参考 Ross Wightman 在 `pytorch-image-models` 的实现
- OrthoGrad / C-Optim / Grams / ADOPT / FOCUS 各自的原论文与上游 PR（见源码注释）

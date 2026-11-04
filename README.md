# Prodigy + Schedule-Free（中文说明）

把 **Prodigy**（自动估计学习率）和 **Schedule-Free**（免学习率调度器）组合起来的 PyTorch 优化器。

- **上游**：[LoganBooker/prodigy-plus-schedule-free](https://github.com/LoganBooker/prodigy-plus-schedule-free)，Apache-2.0，最后更新于 2025-09
- **本仓库**：[zz000123qq/prodigy-plus-schedule-free-Z](https://github.com/zz000123qq/prodigy-plus-schedule-free-Z)（fork，分支 `audit-fix`）
- **状态**：实验性。本文按**本分支代码的实际行为**写，与上游 README 措辞不一致处，以本文为准
- **不做**：不发布到 PyPI，不向上游提 PR

版本号只在 `prodigyplus/_version.py` 里维护（当前 `2.1.0`），`setup.py` / `pyproject.toml` 都从它读，不随每次提交变动。

> 编码说明：本文为 UTF-8（无 BOM），且刻意避开 GBK 覆盖范围之外的符号，中文 Windows 编辑器打开不会出现方框。

---

## 一、安装

```bash
pip install -e .
```

只依赖 `torch>=2.0`，Python 3.9+。**只支持 bf16 / fp32**，其他 dtype 会打印警告且不保证正确。

---

## 二、最小用法

```python
from prodigyplus import ProdigyPlusScheduleFree

opt = ProdigyPlusScheduleFree(model.parameters(), lr=1.0)

for epoch in range(epochs):
    model.train()
    opt.train()                       # 必须
    for x, y in loader:
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    model.eval()
    opt.eval()                        # 必须
    # ... 评估 ...
```

**三条铁律**

1. **`lr` 保持 1.0**。它只是外部缩放系数，真正的学习率是 Prodigy 算出的 `d`。
2. **`train()` / `eval()` 必须和 `model.train()` / `model.eval()` 配对调用**。漏掉 `eval()` 会让评估失真（看到的是 y 点而不是平均点 x）。
3. **不要用外部梯度裁剪**。内部已经按更新量 RMS 做缩放，外面再裁会打乱 Prodigy 对步长的估计。要更温和就调 `use_stableadamw` 或 `eps`。

---

## 三、参数速查

完整 29 个参数的逐条说明在 `prodigy_plus_schedulefree.py` 的 docstring 里。这里只列实际会动的。

### 3.1 常用

| 参数 | 默认 | 说明 |
|---|---|---|
| `lr` | `1.0` | 保持 1，别填学习率 |
| `betas` | `(0.9, 0.99)` | Schedule-Free 场景社区更推荐 `(0.95, 0.99)`，上游仍保留 0.9 作默认 |
| `weight_decay` | `0.0` | 解耦权重衰减 |
| `weight_decay_by_lr` | `True` | WD 是否乘自适应学习率（PyTorch AdamW 的行为） |
| `d0` | `1e-6` | Prodigy 的初始学习率猜测。LR 涨不上去试 `1e-5` / `1e-4`，一直偏大试 `1e-7` |
| `d_coef` | `1.0` | 调 LR 的首选旋钮，`0.5` / `2.0` 都常用 |
| `d_limiter` | `True` | 限制 `d` 每步涨幅（不超过 2 的 0.25 次方倍） |
| `prodigy_steps` | `0` | 大于 0 时在第 N 步**冻结** `d` 并释放 Prodigy 专用显存。长跑 / 扩散模型常用 |
| `schedulefree_c` | `0` | Refined SF-Adam 的平均强度。`0` = 关闭（走原始 SF 规则）。小 batch 短跑试 6 到 12，长跑试 50 到 200 |
| `c_warmup` | `0` | ScheduleFree+ 的延迟平均：前 N 步不平均（`y` 直接跟随 `z`）。论文建议取约 2 倍 LR warmup 长度。只多一个计数器，零显存 |
| `schedulefree_r` | `0.0` | 平均权重改成 `k 的 r 次方 乘 dlr^2`。**论文只在超过 30B token 的长跑用 r=1，短跑用 r=1 反而略差**，所以默认 0 |
| `eps` | `1e-8` | 设成 `None` 改用 **Adam-atan2**（尺度不变、免调 eps），会顺带关掉 `use_stableadamw` |
| `use_schedulefree` | `True` | 设 `False` 退回纯 Prodigy，此时**必须自己配调度器**（cosine 推荐） |

### 3.2 按场景才动

| 参数 | 默认 | 什么时候用 |
|---|---|---|
| `split_groups` | `True` | 每个 param_group 独立算 `d`。同时训多个网络时有用；只有一个组时自动禁用 |
| `split_groups_mean` | `False` | 各组 `d` 取调和平均再乘各组自己的 lr。想复刻老版本行为时开 |
| `factored` | `True` | Adafactor 式二阶矩分解，省显存。出现 NaN 或 LR 涨不动时关掉 |
| `factored_fp32` | `True` | 分解后的二阶矩强制 fp32。只在显存极紧且短训时设 `False` |
| `stochastic_rounding` | `True` | bf16 权重写回用随机舍入。**只对 bf16 参数生效** |
| `use_stableadamw` | `True` | 按更新量 RMS 软缩放（只压不放）。LR 持续高估时关掉，改外部裁剪 |
| `use_bias_correction` | `False` | RAdam 式偏差修正 + 自动 warmup。**会显著拖慢 `d` 收敛**（可能慢 10 倍） |
| `fused_back_pass` | `False` | 用 fused backward pass 时必须置 `True`。需要训练脚本侧配合 |
| `sync_d_stats` | `True` | 跨卡平均 `d` 统计。FSDP / ZeRO-2/3 必需；纯 DDP 下多余但无害 |

### 3.3 实验性 —— 建议先不用

都来自 2024 到 2025 年的论文小技巧，作者自己在 docstring 里写了 "may have a limited effect"。

| 参数 | 在本库的真实状态 |
|---|---|
| `use_speed` | 作者自创。省显存、尺度不敏感，但**与 weight_decay 同用易失稳**，超长训练也容易飘 |
| `use_orthograd` | 用正交于权重方向的梯度分量更新。与权重平行的分量会被完全丢弃 |
| `use_focus` | **强制关闭 `factored`（显存上升）并禁用 Adam-atan2** |
| `use_cautious` | C-Optim。注意：**SF 模式下是半实现**（见第四节第 2 条） |
| `use_grams` | Grams，问题同 `use_cautious`（二者互斥，同开会自动关 grams） |
| `use_adopt` | ADOPT 的**残缺实现**（见第四节第 3 条） |

---

## 四、已知限制

逐条对着源码核过。

1. **默认配置下没有一阶动量。**
   `use_schedulefree=True` 时 `initialise_state` 只建 `z`，不建 `exp_avg`。所以 `use_cautious` / `use_grams` / `use_adopt` 这些**依赖动量的技巧全部退化**——docstring 里也写了 "we do not have access to a first moment"。要用它们，实际得设 `use_schedulefree=False`。

2. **SF 模式下 `cautious` / `grams` 只过滤了一半。**
   `update_params` 先 `z.sub_(update)`，之后才给 `u` 上掩码。结果是 **z 吃未过滤更新、y 吃过滤后更新**，与 C-Optim 本意不符。纯 Prodigy 模式是完整的。
   *没有修*：修它需要决定不同系数下 `z` 和 `y` 如何重组掩码后的方向，手上没有参考实现可对照，不改就是在猜。

3. **`use_adopt` 的 clamp 只在纯 Prodigy 模式存在**（`clamp_range` 是 k 的 0.25 次方），SF 模式没有。

4. **bf16 参数不走 `out=` 优化**，仍走原来的拷贝路径。原因见第五节 5.2。

5. **`prodigy_steps` 取总步数 5% 到 25% 的老建议已作废**（上游新版自己说"现在不需要了"）。

6. **MNIST 那组 98.5% 到 99.3% 的准确率是 v1.x 时代的数据**。v2.0 把内部整个重写过（上游明确写了"不要用新版续训老版本"），这组数字不代表 2.x 效果，本分支也没复现。

7. **Muon 已被上游移除**，且不建议加回来：正交化输出谱范数固定，与 Prodigy 依赖二阶矩口径的 `d` 直接冲突。

---

## 五、性能

测试机：RTX 5060 Ti / Windows / torch 2.9.0+cu130 / Python 3.13.5。

### 5.1 step 到底多贵

33.6M 参数（两个 4096 x 4096 的 Linear），只测 `opt.step()`：

| 优化器 | step 耗时 |
|---|---|
| `torch.optim.AdamW` | 7.36 ms |
| `torch.optim.AdamW(foreach=True)` | 7.49 ms |
| `torch.optim.AdamW(fused=True)` | 3.84 ms |
| **本库（SF 默认）** | **10.57 ms** |

约为普通 AdamW 的 **1.44 倍**、fused AdamW 的 **2.75 倍**。差距来自每参数约 20 次全量张量遍历（AdamW 约 5 次），这是算法本身（SF 的 y/z 双缓冲 + Prodigy 统计量）决定的。

> 注意：小模型上 `step` 可能占到总时间 70% 以上，那是 Windows WDDM 每 kernel 开销高放大的结果。真实 UNet / SDXL LoRA 训练里 fwd+bwd 会占绝对多数，别拿这个比例外推。

### 5.2 本分支做过的优化

| 优化 | 效果 |
|---|---|
| `use_speed` 去掉每参数 3 次 `.item()` | 单步同步次数从 14 降到 1（4 参数场景） |
| `update_()` 支持 `out=`，纯 Prodigy 路径直接写进梯度缓冲 | fp32 实测 step 从 10.82 降到 10.01 ms（快 7.5%）；bf16 不变 |

`out=` 只在 `exp_avg.dtype` 与梯度缓冲的 dtype 一致（即 fp32 参数）时启用。因为 `torch.div(a, b, out=c)` 按**输入的公共 dtype** 计算：1D 参数不参与分解，`exp_avg_sq` 保持 bf16，此时会在 bf16 里除完再转 fp32，比"先转 fp32 再除"少一个最小精度单位。

### 5.3 评估过但没做的

| 想法 | 否决原因 |
|---|---|
| `torch.compile(opt.step)` | 三种模式全部直接抛 `AssertionError: Attempted setattr on a user-defined object...`，位置在 `on_start_step` 的属性赋值。不是 graph break，是硬报错。另外本机 Triton 编译不出 kernel（MSVC 那步失败） |
| 把 RMS 缩放折进 `y.sub_` 的 `alpha` | `alpha` 传 0 维 tensor 会触发主机同步，实测每次调用多 24 微秒，参数一多就是净亏 |
| `y.mul_(1-decay)` 与 `y.sub_` 合成 `addcmul_` | `addcmul_` 只能做一次乘积；且 `weight_decay=0` 时代码本来就跳过那一步 |
| 二阶矩做 8-bit 量化 | 默认配置下解耦后的二阶矩只占状态 0.04%（32KB / 75.7MB），大头是 `z`（85%）。只有主动关掉 `factored` 时量化二阶矩才有意义 |

---

## 六、与 ScheduleFree+（2026）的差距

上游停在 2025 年中的 *Refined SF-Adam*（arXiv 2507.09846）。2026-08 Defazio 本人的 **ScheduleFree+**（arXiv 2605.19095）给出 6 项让它扩展到 LLM 的修正：

| # | 修正 | 本库现状 |
|---|---|---|
| 1 | 恢复内层动量（beta1 约 0.75 到 0.9） | 未做。SF 路径无 `exp_avg`，大 batch 会出现 scaling cliff；补上会多占 85% 的状态显存 |
| 2 | Polyak 步长 + 平滑 L1 梯度范数 | 未做。走的是 Prodigy 的 num/denom 比率 |
| 3 | c-warmup（延迟启用平均） | **已实现** `c_warmup=N`，默认 0（关闭，行为不变） |
| 4 | 外层 beta 退火（0.8/0.9 转到 0.965） | 未做，且**结构上做不到**：`beta1` 同时是内层动量系数和 SF 平均系数 |
| 5 | 幂加权平均（`w = t 的 r 次方`） | **已实现** `schedulefree_r=r`，默认 0。注意论文说 r=1 只适合长跑，短跑反而略差 |
| 6 | 全解耦 AdamC 权重衰减 | 未做。现在是 gamma 乘 lambda，论文要 gamma 平方乘 lambda |

#3 和 #5 是纯加法、不改默认行为，已做成可选开关。实测：`c_warmup=5` 时前 5 步 `y` 与 `z` 完全相等、第 6 步恢复平均；`schedulefree_r=1` 时 `ckp1` 在 k=10 从 0.100 升到 0.182。

剩下 4 项未做。#1 和 #4 是结构性问题（`beta1` 一个旋钮两用），#1 还要多占 85% 的状态显存（`exp_avg` 与 `z` 同尺寸）。

论文口径的收益（省 31% 训练时间）来自 **LLM 长跑**场景，是否适用于 1000 到 3000 步的 LoRA 微调没有证据，别直接套。

---

## 七、适用与不适用

**适合**：扩散模型微调、LoRA、小到中等规模任务——"不想调 LR 和 scheduler"且需要"随时停训、随时存盘"的场景。

**不适合**：

- **LLM 预训练**。2026 年这个领域主流是 Muon / SOAP，本项目主动放弃了这条线。
- 需要精确复现某条已知 LR 曲线的工作。
- 追求单步速度极限：逐参数 Python 循环，没有 `foreach` / `fused` kernel 实现。

---

## 八、开发

```bash
python tests/test_audit_fixes.py      # 14 条用例，不需要 pytest
```

覆盖：cautious 掩码、SPEED 同步次数、FBP 计数器、orthograd 非连续张量、`out=` 与原地路径等价、ckp1 = 1/k、`c_warmup` 期间 y 等于 z、`schedulefree_r` 的权重变化、train/eval 往返、单卡无进程组、CUDA bf16 整步、参数校验。

`tools/logic_checks/` 下是两份不依赖 torch 的逻辑复现脚本（步计数器、`weight_sum`），用来在没装 torch 的环境里验证行为契约。

**未验证**：多卡 `sync_d_stats`——没有多卡环境。

---

## 九、致谢与许可

Apache-2.0，见 `LICENSE`。算法来自：

- [konstmish/prodigy](https://github.com/konstmish/prodigy)
- [facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free)
- Adafactor 分解参考 Ross Wightman 在 `pytorch-image-models` 的实现
- OrthoGrad / C-Optim / Grams / ADOPT / FOCUS 各自的原论文与上游 PR（见源码注释）

---

## 附录：与原始论文的对应（核对用）

本地 `docs/papers/` 存放的是 **arXiv v4（2024-03-19）**，与 ICML 2024 版一致。核对基准就按这一版，注意比它更早的修订里算法编号不同（某些在线渲染器仍显示旧编号，`Algorithm 3` 在那里是 D-Adaptation with Resetting，v4 已删除该内容）。

| 项 | 论文（Alg.3, v4） | 本库实现 | 判定 |
|---|---|---|---|
| `d` 更新 | 第 10 行 `d = max(d, d_hat)` | `core_optimiser.py` 的 `d = max(d, d_hat * d_coef)` | 原样。`d` 单调不减是论文的写法，不是本库的简化 |
| `d` 增长上限 | 无 | `d_limiter=True` 时 `d_hat <= d * 2**0.25` | 本库额外加的，每步最多涨 19%，比论文更保守 |
| `s` 衰减 | `√beta2` | `beta3 = beta2 ** 0.5` | 原样 |
| `v` 衰减 | `beta2` | `beta2`（偏差校正可选） | 原样 |
| `d_hat` 分母 | `‖s‖_1` | `.abs().sum()` | 原样。L1 来自附录 A.5 的坐标级分析（L-inf 距离配 L1 梯度，对偶配对），论文自己的说法是"在理论的其他部分自然出现" |
| 内积项 | `<g, x0 - x>` | `torch.dot(sliced_grad, x0_minus)` | 原样 |
| `gamma_k` | 默认 1，**配合 cosine annealing** | 无 `gamma`，用 Schedule-Free 的平均替代衰减 | 有意偏离：本库的目的就是不要 scheduler |
| `d^2` 缩放 | `gamma * d^2` | `d^2 / sqrt(d0)` | 常数缩放，分子分母同乘，在比值里抵消 |
| 逐元素采样 | 全量 | `[::11]` 切片 | 有意偏离：省统计开销，代价是估计方差变大 |
| 二阶矩形状 | 满尺寸 | `factored=True` 时行列分解 | 有意偏离：省显存，见第四节 |

论文把理论建立在**凸 + G-Lipschitz** 假设上，正文没有讨论梯度噪声、信噪比、扩散模型，也没有对非凸情形下内积项的震荡做分析；深度网络部分只做了经验验证。所以"论文预警了扩散训练的噪声问题"这类说法在原文里找不到依据。

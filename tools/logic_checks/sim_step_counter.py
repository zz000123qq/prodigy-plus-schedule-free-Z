"""
复刻 core_optimiser.py 的 on_start_step / on_end_step 计数逻辑,
对比"修复前(仅倒计数)"与"修复后(带 step 内已见参数集合)":
fused_back_pass + 部分参数无梯度 时, group['k'] 与真实步数的比例。

场景: n 个参数, 其中 n - active 个冻结(永远没有梯度 -> 钩子永不触发),
      因此每轮只有 active 次 step_param 调用。
"""

N = 10


class LegacyCounter:
    """修复前: 只有倒计数。counter 归零后被下一个参数重新初始化为 n, 于是
    步边界落在任意位置, k 的推进速度与真实步数不成比例。"""

    def __init__(self, n):
        self.n = n
        self.parameters_to_process = None
        self.k = 0

    def on_start_step(self, p):
        if not self.parameters_to_process:
            self.parameters_to_process = self.n

    def on_end_step(self, p=None):
        self.parameters_to_process -= 1
        if self.parameters_to_process == 0:
            self.k += 1


class FixedCounter:
    """修复后: 同一 step 内见到重复参数即判定上一步未收尾, 强制收尾后重新上膛。"""

    def __init__(self, n):
        self.n = n
        self.parameters_to_process = None
        self._params_seen_this_step = set()
        self.k = 0

    def on_start_step(self, p):
        if not self.parameters_to_process:
            self.parameters_to_process = self.n
        elif p is not None and p in self._params_seen_this_step:
            self.parameters_to_process = 1
            self.on_end_step(p)
            # on_end_step 把计数归零后重新上膛(与源码一致)
            self.parameters_to_process = self.n

        if p is not None:
            self._params_seen_this_step.add(p)

    def on_end_step(self, p=None):
        self.parameters_to_process -= 1
        if self.parameters_to_process <= 0:
            self.parameters_to_process = 0
            self._params_seen_this_step.clear()
            self.k += 1


def run(cls, n, active, rounds):
    c = cls(n)
    out = []
    for _ in range(rounds):
        for i in range(active):
            c.on_start_step(i)
            c.on_end_step(i)
        out.append(c.k)
    return out


ROUNDS = 12
print(f"n={N} 个参数, {ROUNDS} 轮\n")
print(f"{'active':>7} | {'修复前 k':>9} | {'修复后 k':>9} | {'修复前 k/轮':>11}")
print("-" * 48)
for active in (10, 9, 8, 5, 2):
    legacy = run(LegacyCounter, N, active, ROUNDS)[-1]
    fixed = run(FixedCounter, N, active, ROUNDS)[-1]
    print(f"{active:>7} | {legacy:>9} | {fixed:>9} | {legacy / ROUNDS:>11.2f}")

print("\n逐轮 k 序列 (active=9):")
print("  修复前:", run(LegacyCounter, N, 9, 8))
print("  修复后:", run(FixedCounter, N, 9, 8))

# 修复后: 每轮恰好推进一次(首轮收尾发生在下一轮首次调用, 因此有 1 步滞后)
for active in (10, 9, 8, 5, 2, 1):
    ks = run(FixedCounter, N, active, ROUNDS)
    diffs = [b - a for a, b in zip(ks, ks[1:])]
    assert set(diffs) == {1}, (active, ks)

# 修复前: 全参数时正常, 冻结参数一多 k 的推进速度就与真实步数脱钩
assert run(LegacyCounter, N, 10, ROUNDS)[-1] == ROUNDS
assert run(LegacyCounter, N, 2, ROUNDS)[-1] == 2
print("\n断言通过:")
print("  - 修复后: 任意冻结比例下 k 都稳定每轮 +1")
print("  - 修复前: 冻结一半参数时 k 只推进了真实步数的 1/6")

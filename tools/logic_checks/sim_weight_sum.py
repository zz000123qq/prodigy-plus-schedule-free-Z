"""
按源码原文逐行复刻 Schedule-Free 平均权重(ckp1)的计算, 验证 group 级 weight_sum 是否等价于 per-param state。

源码(prodigy_plus_schedulefree.py:248-250, 每个参数调用一次):
    weight = dlr ** 2
    weight_sum = group['running_weight_sum'] = group.get('weight_sum', 0) + weight
    ckp1 = weight / weight_sum if weight_sum else 0

源码(core_optimiser.py:379, 每个 step 调用一次):
    group['weight_sum'] = group.get('running_weight_sum', 0)

关键: 链式赋值右侧读的是 group.get('weight_sum') —— 这个值在 **本 step 内不变**,
      不是 running_weight_sum 自己。所以 step 内所有参数的 ckp1 完全相同,
      on_end_step 之后 weight_sum 只增加了一个 weight, 不是 n 个。
      => ckp1 = 1/k, 与 per-param state 等价(前提: 组内 dlr 恒定, 本库成立)。

注: 本文件初版曾误把 `running += weight` 写进参数循环, 从而得出 "ckp1 ∝ 1/(n*k)"
    的错误结论。已按源码重写。
"""

BETA1 = 0.9


def run(n_params, steps, scheme="repo", dlr=1.0):
    group = {}                      # 真实 param_group dict
    y = [0.0] * n_params
    z = [0.0] * n_params
    state_ws = [0.0] * n_params     # 参考实现: state[p]['weight_sum']
    ckp1_per_step = []

    for _ in range(steps):
        step_ckp1 = []
        for i in range(n_params):
            weight = dlr ** 2
            if scheme == "repo":
                # 逐字复刻链式赋值
                group['running_weight_sum'] = group.get('weight_sum', 0) + weight
                weight_sum = group['running_weight_sum']
            else:
                state_ws[i] += weight
                weight_sum = state_ws[i]
            ckp1 = weight / weight_sum if weight_sum else 0.0
            step_ckp1.append(ckp1)

            xy_step = 1 - BETA1 * (1 - ckp1)
            u = 1.0
            y[i] = y[i] + ckp1 * (z[i] - y[i]) - dlr * xy_step * u
            z[i] = z[i] - dlr * u

        if scheme == "repo":
            # 逐字复刻 on_end_step
            group['weight_sum'] = group.get('running_weight_sum', 0)
        ckp1_per_step.append(step_ckp1)

    return y, z, ckp1_per_step


def main():
    print(f"{'n_params':>9} | {'scheme':>6} | {'ckp1@k=1':>9} | {'ckp1@k=10':>10} | "
          f"{'ckp1@k=1000':>11} | {'y_T':>10}")
    print("-" * 76)
    for n in (1, 4, 64, 1024):
        for scheme in ("repo", "ref"):
            y, z, c = run(n, 1000, scheme)
            print(f"{n:>9} | {scheme:>6} | {c[0][0]:>9.6f} | {c[9][0]:>10.6f} | "
                  f"{c[-1][0]:>11.6f} | {y[0]:>10.3f}")
        spread = max(max(s) - min(s) for s in c)
        print(f"{'':>9} | {'':>6} | step 内 ckp1 极差 = {spread:.2e}  (0 = step 内恒定)")
        print("-" * 76)


if __name__ == "__main__":
    main()

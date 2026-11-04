"""A3: sweep the two knobs the docstring recommends for short, noisy, small-batch
runs -- schedulefree_c and beta1 -- starting from the c_warmup=100 baseline that
a1_denoise.py found best.

    python tools/experiments/a3_sweep.py --steps 600 --seeds 3 --batch 8
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from a1_denoise import MiniUNet, NoiseSchedule, make_dataset, run  # noqa: E402
from prodigyplus import ProdigyPlusScheduleFree  # noqa: E402


ARMS = [
    ("默认（c_warmup=0）", dict()),
    ("c_warmup=100", dict(c_warmup=100)),
    ("+ schedulefree_c=10", dict(c_warmup=100, schedulefree_c=10)),
    ("+ beta1=0.95", dict(c_warmup=100, betas=(0.95, 0.99))),
    ("+ use_cautious", dict(c_warmup=100, use_cautious=True)),
    ("+ c=10 + beta1=0.95", dict(c_warmup=100, schedulefree_c=10, betas=(0.95, 0.99))),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    sched = NoiseSchedule(100, "cuda")
    train_x = make_dataset(768, seed=1).cuda()
    eval_x = make_dataset(64, seed=12345).cuda()

    print(f"steps={args.steps} seeds={args.seeds} batch={args.batch}")
    print(f"{'配置':<24}{'eval loss':>12}{'±std':>9}{'相对 baseline':>14}")
    print("-" * 62)

    base = None
    for name, kw in ARMS:
        res = [run(ProdigyPlusScheduleFree, {"kw": kw}, s, args.steps,
                   sched, train_x, eval_x, batch=args.batch) for s in range(args.seeds)]
        ev = [r["eval"] for r in res]
        mean = sum(ev) / len(ev)
        sd = (sum((x - mean) ** 2 for x in ev) / len(ev)) ** 0.5
        if base is None:
            base = mean
            rel = "-"
        else:
            rel = f"{(mean / base - 1) * 100:+.1f}%"
        print(f"{name:<24}{mean:>12.5f}{sd:>9.5f}{rel:>14}")


if __name__ == "__main__":
    main()

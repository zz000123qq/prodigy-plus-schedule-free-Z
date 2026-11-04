"""A2: does per-module parameter grouping (one d per group) actually help?

The optimiser already gives every param_group its own d when more than one group
is passed (see CoreOptimiser.on_end_step). This measures whether that helps on
the same mini UNet denoising task used in a1_denoise.py.

    python tools/experiments/a2_param_groups.py --steps 600 --seeds 3

Three arms share the model, the data and the seed:
  single   one group, one d for the whole net      (the default usage)
  split    three groups, three independent d       (modulation / convs / io)
  harmonic three groups, effective d = harmonic mean of the three
"""

import argparse
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from a1_denoise import MiniUNet, NoiseSchedule, make_dataset  # noqa: E402
from prodigyplus import ProdigyPlusScheduleFree  # noqa: E402


def groups_of(model):
    """Split by role: the timestep MLP, the conv blocks, and the in/out convs."""
    mod, blocks, io = [], [], []
    for name, p in model.named_parameters():
        if name.startswith("tmlp"):
            mod.append(p)
        elif name.startswith(("inp", "out")):
            io.append(p)
        else:
            blocks.append(p)
    return {"mod": mod, "blocks": blocks, "io": io}


def run(arm, seed, steps, sched, train_x, eval_x, batch):
    torch.manual_seed(seed)
    model = MiniUNet().cuda()
    g = groups_of(model)

    if arm == "single":
        params = model.parameters()
        kw = {}
    elif arm == "split":
        params = [
            {"params": g["mod"], "lr": 1.0},
            {"params": g["blocks"], "lr": 1.0},
            {"params": g["io"], "lr": 1.0},
        ]
        kw = {}
    elif arm == "harmonic":
        params = [
            {"params": g["mod"], "lr": 1.0},
            {"params": g["blocks"], "lr": 1.0},
            {"params": g["io"], "lr": 1.0},
        ]
        kw = {"split_groups_mean": True}
    else:
        raise ValueError(arm)

    opt = ProdigyPlusScheduleFree(params, **kw)

    gen = torch.Generator(device="cuda").manual_seed(seed + 999)
    for _ in range(steps):
        model.train()
        opt.train()
        idx = torch.randint(0, train_x.shape[0], (batch,), device="cuda", generator=gen)
        xt, eps, t = sched.sample(train_x[idx], generator=gen)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(xt, t), eps).backward()
        opt.step()

    model.eval()
    opt.eval()
    with torch.no_grad():
        g2 = torch.Generator(device="cuda").manual_seed(4242)
        xt, eps, t = sched.sample(eval_x, generator=g2)
        ev = F.mse_loss(model(xt, t), eps).item()

    ds = [gr["d"] for gr in opt.param_groups]
    return ev, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    sched = NoiseSchedule(100, "cuda")
    train_x = make_dataset(768, seed=1).cuda()
    eval_x = make_dataset(64, seed=12345).cuda()

    print(f"steps={args.steps} seeds={args.seeds} batch={args.batch}")
    print(f"{'分组方式':<14}{'eval loss':>12}{'±std':>9}   d 值")
    print("-" * 78)
    for arm in ("single", "split", "harmonic"):
        res = [run(arm, s, args.steps, sched, train_x, eval_x, args.batch) for s in range(args.seeds)]
        ev = [r[0] for r in res]
        mean = sum(ev) / len(ev)
        sd = (sum((x - mean) ** 2 for x in ev) / len(ev)) ** 0.5
        ds = res[0][1]
        ds_txt = "  ".join(f"{d:.2e}" for d in ds)
        print(f"{arm:<14}{mean:>12.5f}{sd:>9.5f}   {ds_txt}")


if __name__ == "__main__":
    main()

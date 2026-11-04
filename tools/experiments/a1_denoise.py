"""A1: end-to-end comparison on a mini UNet denoising task.

Why not MNIST: classification has a high gradient signal-to-noise ratio and a
smooth parameter trajectory. Diffusion training predicts noise from noise, so
the gradient is high-variance by construction -- which is the regime where
cautious masking and c_warmup are supposed to matter. This script therefore
trains a small UNet to predict the noise added to a procedurally generated
image, and compares optimiser variants on held-out denoising loss.

    python tools/experiments/a1_denoise.py            # full run, 3 seeds
    python tools/experiments/a1_denoise.py --quick    # 1 seed, 150 steps

The old version is exported on demand with
    git show 338cf40:prodigyplus/<file>
so both implementations can be loaded in the same process.
"""

import argparse
import math
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OLD_REF = "338cf40"          # upstream v2.0.1
OLD_DIR = os.path.join(REPO, ".oldver")


# --------------------------------------------------------------------- data

def make_dataset(n, size=32, seed=0):
    """Procedural images: coloured blobs, stripes and ramps.

    No download, and enough spatial structure that a conv net has something to
    learn. Values in [-1, 1].
    """
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    imgs = torch.empty(n, 3, size, size)
    for i in range(n):
        chans = []
        for _ in range(3):
            kind = torch.randint(0, 3, (1,), generator=g).item()
            if kind == 0:                                  # blob
                cy, cx = (torch.rand(2, generator=g) * 2 - 1).tolist()
                r = 0.2 + 0.5 * torch.rand(1, generator=g).item()
                chans.append(torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
            elif kind == 1:                                # stripes
                freq = 1.0 + 4.0 * torch.rand(1, generator=g).item()
                ang = math.pi * torch.rand(1, generator=g).item()
                chans.append(0.5 + 0.5 * torch.sin(freq * (xx * math.cos(ang) + yy * math.sin(ang)) * math.pi))
            else:                                          # ramp
                a, b = (torch.rand(2, generator=g) * 2 - 1).tolist()
                v = a * xx + b * yy
                chans.append((v - v.min()) / (v.max() - v.min() + 1e-6))
        imgs[i] = torch.stack(chans) * 2 - 1
    return imgs


class NoiseSchedule:
    """Cosine schedule, same shape as the DDPM paper."""

    def __init__(self, steps=100, device="cpu"):
        s = 0.008
        t = torch.linspace(0, 1, steps + 1)
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = f / f[0]
        self.alpha_bar = ab.clamp(1e-5, 1.0).to(device)
        self.steps = steps

    def sample(self, x0, generator=None):
        b = x0.shape[0]
        t = torch.randint(0, self.steps, (b,), device=x0.device, generator=generator)
        ab = self.alpha_bar[t].view(b, 1, 1, 1)
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps, eps, t


# -------------------------------------------------------------------- model

def temb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float().view(-1, 1) * f.view(1, -1)
    return torch.cat([a.sin(), a.cos()], dim=1)


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout * 2)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.temb(t).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class MiniUNet(nn.Module):
    """Two down/up levels, ~1.5M parameters at base=32."""

    def __init__(self, base=32, tdim=64):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(3, base, 3, padding=1)
        self.d1 = Block(base, base, tdim)
        self.d2 = Block(base, base * 2, tdim)
        self.mid = Block(base * 2, base * 2, tdim)
        # up(m) is base*2 wide at 32x32, h1 is base: 3*base into u2
        self.u2 = Block(base * 3, base, tdim)
        self.u1 = Block(base * 2, base, tdim)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x, t):
        te = self.tmlp(temb(t, self.tdim))
        h1 = self.d1(self.inp(x), te)
        h2 = self.d2(F.avg_pool2d(h1, 2), te)
        m = self.mid(h2, te)
        u2 = self.u2(torch.cat([F.interpolate(m, scale_factor=2), h1], 1), te)
        u1 = self.u1(torch.cat([u2, h1], 1), te)
        return self.out(u1)


# ---------------------------------------------------------------- versions

def export_old():
    """Materialise v2.0.1 next to the repo so both can be imported."""
    pkg = os.path.join(OLD_DIR, "prodigyplus")
    os.makedirs(pkg, exist_ok=True)
    for f in ("core_optimiser.py", "prodigy_plus_schedulefree.py", "_version.py", "__init__.py"):
        out = subprocess.run(
            ["git", "-C", REPO, "show", f"{OLD_REF}:prodigyplus/{f}"],
            capture_output=True,
        )
        if out.returncode == 0:
            open(os.path.join(pkg, f), "wb").write(out.stdout)
    # v2.0.1 has no _version module
    init = os.path.join(pkg, "__init__.py")
    if not os.path.exists(init) or not open(init, "rb").read().strip():
        open(init, "w").write(
            "from .core_optimiser import CoreOptimiser\n"
            "from .prodigy_plus_schedulefree import ProdigyPlusScheduleFree\n"
        )


def load(root):
    for m in [k for k in list(sys.modules) if k.startswith("prodigyplus")]:
        del sys.modules[m]
    sys.path.insert(0, root)
    try:
        import prodigyplus
        assert prodigyplus.__file__.startswith(root), prodigyplus.__file__
        return prodigyplus.ProdigyPlusScheduleFree
    finally:
        sys.path.pop(0)


# ------------------------------------------------------------------- runner

def run(cls, arm, seed, steps, sched, train_x, eval_x, batch=32):
    torch.manual_seed(seed)
    model = MiniUNet().cuda()
    opt = cls(model.parameters(), lr=1.0, **arm["kw"])

    g = torch.Generator(device="cuda").manual_seed(seed + 999)
    losses, t0 = [], time.time()

    for step in range(steps):
        model.train()
        opt.train()
        idx = torch.randint(0, train_x.shape[0], (batch,), device="cuda", generator=g)
        x0 = train_x[idx]
        xt, eps, t = sched.sample(x0, generator=g)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(xt, t), eps)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Final evaluation uses the averaged weights (SF) on a fixed held-out batch.
    model.eval()
    opt.eval()
    with torch.no_grad():
        g2 = torch.Generator(device="cuda").manual_seed(4242)
        xt, eps, t = sched.sample(eval_x, generator=g2)
        eval_loss = F.mse_loss(model(xt, t), eps).item()
    opt.train()

    tail = losses[-100:]
    return {
        "d": opt.param_groups[0]["d"],
        "eval": eval_loss,
        "train_tail": sum(tail) / len(tail),
        "train_std": float(torch.tensor(tail).std()),
        "secs": time.time() - t0,
        "losses": losses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    steps = 150 if args.quick else args.steps
    seeds = [0] if args.quick else list(range(args.seeds))

    export_old()
    new_cls = load(REPO)
    old_cls = load(OLD_DIR)

    device = "cuda"
    sched = NoiseSchedule(100, device)
    train_x = make_dataset(768, seed=1).to(device)
    eval_x = make_dataset(64, seed=12345).to(device)

    SF = {}
    PROD = {"use_schedulefree": False}

    arms = [
        ("v2.0.1  SF 默认", old_cls, {"kw": dict(SF)}),
        ("当前    SF 默认", new_cls, {"kw": dict(SF)}),
        ("当前    SF 默认 (重复)", new_cls, {"kw": dict(SF)}),   # run-to-run floor
        ("v2.0.1  纯Prodigy+cautious", old_cls, {"kw": dict(PROD, use_cautious=True)}),
        ("当前    纯Prodigy+cautious", new_cls, {"kw": dict(PROD, use_cautious=True)}),
        ("当前    SF+cautious", new_cls, {"kw": dict(SF, use_cautious=True)}),
        ("当前    SF+c_warmup=100", new_cls, {"kw": dict(SF, c_warmup=100)}),
    ]
    if args.quick:
        arms = arms[:2]

    print(f"steps={steps} seeds={len(seeds)} batch={args.batch} params={sum(p.numel() for p in MiniUNet().parameters())/1e6:.2f}M")
    print(f"{'配置':<28}{'eval loss':>12}{'±std':>9}{'train tail':>12}{'loss std':>10}{'d':>10}{'秒':>7}")
    print("-" * 92)

    for name, cls, arm in arms:
        res = [run(cls, arm, s, steps, sched, train_x, eval_x, batch=args.batch) for s in seeds]
        ev = [r["eval"] for r in res]
        mean = sum(ev) / len(ev)
        sd = (sum((x - mean) ** 2 for x in ev) / len(ev)) ** 0.5
        tt = sum(r["train_tail"] for r in res) / len(res)
        ls = sum(r["train_std"] for r in res) / len(res)
        secs = sum(r["secs"] for r in res) / len(res)
        dd = sum(r["d"] for r in res) / len(res)
        print(f"{name:<28}{mean:>12.5f}{sd:>9.5f}{tt:>12.5f}{ls:>10.4f}{dd:>10.3e}{secs:>7.1f}")

    # Gradient SNR: how noisy is this objective, really?
    torch.manual_seed(0)
    probe = MiniUNet().cuda()
    pg = torch.Generator(device="cuda").manual_seed(7)
    norms = []
    for _ in range(40):
        idx = torch.randint(0, train_x.shape[0], (32,), device="cuda", generator=pg)
        xt, eps, t = sched.sample(train_x[idx], generator=pg)
        probe.zero_grad(set_to_none=True)
        F.mse_loss(probe(xt, t), eps).backward()
        norms.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in probe.parameters() if p.grad is not None))))
    mu = sum(norms) / len(norms)
    sd = (sum((x - mu) ** 2 for x in norms) / len(norms)) ** 0.5
    print()
    print(f"梯度范数（初始化，40 个 batch）: 均值 {mu:.3e}  标准差 {sd:.3e}  信噪比 {mu/sd:.2f}")

    # Reference: what a pure-noise predictor scores.
    with torch.no_grad():
        g2 = torch.Generator(device="cuda").manual_seed(4242)
        xt, eps, t = sched.sample(eval_x, generator=g2)
        print(f"\n零预测基线（输出全 0）: {F.mse_loss(torch.zeros_like(eps), eps).item():.4f}")
        print(f"输入本身当预测    : {F.mse_loss(xt, eps).item():.4f}")


if __name__ == "__main__":
    main()

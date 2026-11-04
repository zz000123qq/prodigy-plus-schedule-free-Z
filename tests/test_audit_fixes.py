"""
Regression tests for the audit fixes (see docs/IMPROVEMENT-PLAN.md, T1-T6).

Runs without pytest:  python tests/test_audit_fixes.py
Also collected by pytest if installed.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree  # noqa: E402

torch.manual_seed(0)


def _make_model(dtype=torch.float32):
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    ).to(dtype)


def _rms(t):
    return t.norm(2).div(t.numel() ** 0.5).item()


# ---------------------------------------------------------------- T1 cautious

def test_cautious_applies_the_mask():
    """The mask must reach the update, and must not change its scale.

    The mask is `sign(grad * exp_avg) > 0`. Momentum is a smoothed gradient, so it
    usually agrees with the current gradient: only a small fraction of entries get
    zeroed (and on the very first step, where exp_avg = (1-beta1)*d*grad, none do).
    Before the fix the mask was computed and discarded, so *no* entry was ever
    zeroed and the update became grad * exp_avg instead of exp_avg * mask.
    """
    stats = {}
    for cautious in (False, True):
        torch.manual_seed(1234)
        model = _make_model()
        opt = ProdigyPlusScheduleFree(
            model.parameters(), lr=1.0, use_schedulefree=False,
            use_cautious=cautious, use_stableadamw=False, d0=1e-2,
        )

        def step():
            opt.zero_grad()
            torch.manual_seed(7)
            model(torch.randn(32, 8)).pow(2).mean().backward()
            opt.step()

        for _ in range(5):                      # warm up the momentum
            step()

        # Measure single-step deltas, not the cumulative one: a zeroed entry in one
        # step is hidden by the other steps in a cumulative sum.
        zero_fracs, rms = [], []
        for _ in range(3):
            before = [p.detach().clone() for p in model.parameters()]
            step()
            delta = torch.cat([(p.detach() - b).reshape(-1) for p, b in zip(model.parameters(), before)])
            zero_fracs.append((delta == 0).float().mean().item())
            rms.append(_rms(delta))

        stats[cautious] = {"zero_frac": max(zero_fracs), "rms": sum(rms) / len(rms)}

    print(f"    cautious=False: zero_frac={stats[False]['zero_frac']:.3f} rms={stats[False]['rms']:.3e}")
    print(f"    cautious=True : zero_frac={stats[True]['zero_frac']:.3f} rms={stats[True]['rms']:.3e}")

    assert stats[False]["zero_frac"] < 0.01, "unmasked run should have no zeroed entries"
    assert stats[True]["zero_frac"] > 0.02, "mask never applied -- update has no zeroed entries"

    ratio = stats[True]["rms"] / stats[False]["rms"]
    print(f"    rms ratio cautious/base = {ratio:.2f}")
    assert 0.2 < ratio < 5.0, f"cautious update scale off by {ratio:.1f}x"


# --------------------------------------------------------- T2 host sync count

def test_speed_has_no_per_parameter_item_calls():
    """use_speed used to call .item() three times per parameter per step."""
    n_params = len(list(_make_model().parameters()))
    model = _make_model()
    opt = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, use_speed=True, d0=1e-2)

    # Warm up so state is initialised; then measure one clean step.
    opt.zero_grad(); model(torch.randn(8, 8)).pow(2).mean().backward(); opt.step()

    calls = {"n": 0}
    orig_item, orig_tolist = torch.Tensor.item, torch.Tensor.tolist

    def counted_item(self):
        calls["n"] += 1
        return orig_item(self)

    def counted_tolist(self):
        calls["n"] += 1
        return orig_tolist(self)

    torch.Tensor.item, torch.Tensor.tolist = counted_item, counted_tolist
    try:
        opt.zero_grad()
        model(torch.randn(8, 8)).pow(2).mean().backward()
        calls["n"] = 0          # ignore anything the loss/grad path did
        opt.step()
    finally:
        torch.Tensor.item, torch.Tensor.tolist = orig_item, orig_tolist

    budget = 2
    print(f"    syncs in one step (SPEED, {n_params} params): {calls['n']} (budget {budget})")
    assert calls["n"] <= budget, f"expected <= {budget} syncs, got {calls['n']}"


# ------------------------------------------------------------ T3 fbp counter

def test_step_counter_survives_frozen_parameters():
    """Drive step_param directly (as a fused-backward hook does) for params that
    have a gradient only. group['k'] must advance once per round."""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(10)]
    opt = ProdigyPlusScheduleFree(params, lr=1.0, d0=1e-2)
    group = opt.param_groups[0]

    rounds, active = 6, 9          # params[9] never receives a gradient
    ks = []
    for _ in range(rounds):
        for p in params[:active]:
            p.grad = torch.randn_like(p)
            opt.step_param(p, group)
        ks.append(group["k"])

    print(f"    k per round: {ks}")
    diffs = [b - a for a, b in zip(ks, ks[1:])]
    assert set(diffs) == {1}, f"k must advance exactly once per round, got {ks}"


# --------------------------------------------------------------- T4 orthograd

def test_orthograd_stays_finite_across_scales():
    """Guard, not a regression test: the parallel-gradient case is finite in v2.0.1
    too, because float subtraction leaves a residual so the scale factor never
    reaches inf. Kept so a future rewrite cannot introduce a NaN unnoticed.
    """
    for scale in (1e0, 1e4, 1e9):
        p = torch.nn.Parameter(torch.randn(64) * scale)
        g = p.detach().clone() * scale          # exactly parallel to w
        opt = ProdigyPlusScheduleFree([p], lr=1.0, use_orthograd=True)
        out = opt.orthograd_(p.detach(), g)
        print(f"    scale={scale:.0e}: finite={bool(torch.isfinite(out).all())} "
              f"norm={out.norm().item():.3e} (|g|={g.norm().item():.3e})")
        assert torch.isfinite(out).all(), f"non-finite orthograd output at scale {scale}"


def test_orthograd_accepts_non_contiguous_params():
    conv = torch.nn.Conv2d(4, 4, 3).to(memory_format=torch.channels_last)
    w = conv.weight
    assert not w.is_contiguous()
    opt = ProdigyPlusScheduleFree([w], lr=1.0, use_orthograd=True)

    out = opt.orthograd_(w.detach(), torch.randn_like(w))
    print(f"    non-contiguous weight ok, shape={tuple(out.shape)}")
    assert out.shape == w.shape


# --------------------------------------------------------------- T5 distributed

def test_runs_without_process_group():
    """sync_d_stats must be a no-op when torch.distributed is not initialised."""
    model = _make_model()
    opt = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, sync_d_stats=True)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 8)).pow(2).mean().backward()
        opt.step()
    k = opt.param_groups[0]["k"]
    print(f"    ok without process group, k={k} (k starts at 1)")
    assert k == 4


# ------------------------------------------------------- update buffer reuse

def test_update_out_matches_inplace_path():
    """update_(..., out=buf) must be bit-identical to update_(num, ...) in place.

    The fused path exists purely to skip a full copy of the momentum; if the two
    ever disagree the shortcut has to go.
    """
    for eps in (1e-8, None):
        torch.manual_seed(0)
        model = _make_model()
        opt = ProdigyPlusScheduleFree(
            model.parameters(), lr=1.0, d0=1e-2, eps=eps,
            use_schedulefree=False, use_stableadamw=False,
        )
        opt.zero_grad()
        model(torch.randn(8, 8)).pow(2).mean().backward()
        opt.step()
        group = opt.param_groups[0]

        num = torch.randn(64, 64, device=next(model.parameters()).device)
        den = torch.rand(64, 64, device=num.device) + 0.05

        ref = opt.update_(num.clone(), den.clone(), group, None)
        buf = torch.empty_like(num)
        out = opt.update_(num.clone(), den.clone(), group, None, out=buf)
        print(f"    eps={eps}: identical={torch.equal(ref, out)}")
        assert out is buf, "out= must write into the supplied buffer"
        assert torch.equal(ref, out), "out= path diverged from the in-place path"


# ------------------------------------------------------------ validation

def test_betas_index1_none_is_rejected_up_front():
    """betas[1] used to pass validation and then explode mid-training in
    get_bias_correction(), where beta2 ** k raises TypeError."""
    model = _make_model()
    try:
        ProdigyPlusScheduleFree(model.parameters(), lr=1.0, betas=(0.9, None))
    except ValueError as exc:
        print(f"    ValueError at construction: {str(exc).splitlines()[0][:60]}...")
        assert "betas[1]" in str(exc)
    else:
        raise AssertionError("betas[1]=None was accepted")


def test_no_dead_d_prev_key():
    """d_prev was written in three places and never read."""
    model = _make_model()
    opt = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, d0=1e-2)
    opt.zero_grad()
    model(torch.randn(8, 8)).pow(2).mean().backward()
    opt.step()
    assert "d_prev" not in opt.param_groups[0]
    print("    d_prev absent from param_group")


# ------------------------------------------------------- schedule-free sanity

def test_ckp1_matches_one_over_k():
    """weight_sum must advance once per *step*, not once per parameter."""
    model = _make_model()
    # prodigy_steps=1 freezes d after the first step, so dlr stays constant and
    # ckp1 should be exactly 1/k.
    opt = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, d0=1e-2, prodigy_steps=1)

    seen = []
    orig = ProdigyPlusScheduleFree.update_params

    def spy(self, y, z, update, group, dlr):
        w = dlr ** 2
        seen.append(w / (group.get("weight_sum", 0) + w))
        return orig(self, y, z, update, group, dlr)

    ProdigyPlusScheduleFree.update_params = spy
    try:
        for _ in range(8):
            opt.zero_grad()
            model(torch.randn(8, 8)).pow(2).mean().backward()
            opt.step()
    finally:
        ProdigyPlusScheduleFree.update_params = orig

    print(f"    ckp1 per param-step (first 6): {[round(v, 6) for v in seen[:6]]}")
    # every parameter in a given step must see the same ckp1 as the first one
    n_params = len(list(model.parameters()))
    for step in range(6):
        chunk = seen[step * n_params:(step + 1) * n_params]
        assert max(chunk) - min(chunk) < 1e-9, f"ckp1 varied within step {step}: {chunk}"
        expected = 1.0 / (step + 1)
        assert abs(chunk[0] - expected) < 1e-6, f"step {step}: ckp1={chunk[0]}, expected {expected}"


def test_train_eval_roundtrip_restores_weights():
    model = _make_model()
    opt = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, d0=1e-2)
    for _ in range(5):
        opt.zero_grad()
        model(torch.randn(8, 8)).pow(2).mean().backward()
        opt.step()

    opt.eval()
    opt.train()
    for _ in range(2):
        opt.zero_grad()
        model(torch.randn(8, 8)).pow(2).mean().backward()
        opt.step()
    k = opt.param_groups[0]["k"]
    print(f"    train/eval roundtrip ok, k={k} (7 steps, k starts at 1)")
    assert k == 8


def test_cuda_step_if_available():
    if not torch.cuda.is_available():
        print("    skipped (no CUDA)")
        return
    model = _make_model(torch.bfloat16).cuda()
    opt = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, d0=1e-2)
    for _ in range(5):
        opt.zero_grad()
        model(torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)).pow(2).mean().backward()
        opt.step()
    d = opt.param_groups[0]["d"]
    print(f"    cuda bf16 ok, d={d:.3e}")
    assert d > 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        try:
            fn()
            print("  PASS")
        except Exception as exc:                      # noqa: BLE001
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            failed.append((fn.__name__, exc))

    print("\n" + "=" * 60)
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    for name, exc in failed:
        print(f"  FAILED {name}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

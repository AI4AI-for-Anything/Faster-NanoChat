"""Check that the XLA-safe optimizer rewrites match upstream numerically.

The rewrites in xla_optim_patch.py change two things: hyperparameters arrive as
Python numbers instead of 0-D tensors, and nothing else. That claim is testable
on CPU, without a TPU: run the upstream function and the rewrite on identical
inputs and compare.

This does not prove anything about XLA behaviour. It only rules out transcription
errors in the rewrite, so a later TPU failure can be attributed to the platform
rather than to this file.

Usage (from the repo root):  python -m tpu_smoke.test_xla_optim_patch
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))    # repo root: nanochat.*, tpu_smoke.*

from nanochat import optim as O

from tpu_smoke.xla_optim_patch import adamw_step_xla, muon_step_xla


def check_adamw():
    torch.manual_seed(0)
    shape = (64, 32)
    p0 = torch.randn(shape)
    g0 = torch.randn(shape)
    m0 = torch.randn(shape) * 0.1
    v0 = torch.rand(shape).abs() + 0.01
    step, lr, b1, b2, eps, wd = 3, 0.01, 0.8, 0.96, 1e-10, 0.01

    up = [t.clone() for t in (p0, g0, m0, v0)]
    O.adamw_step_fused.__wrapped__(
        up[0], up[1], up[2], up[3],
        torch.tensor(float(step)), torch.tensor(lr), torch.tensor(b1),
        torch.tensor(b2), torch.tensor(eps), torch.tensor(wd))

    new = [t.clone() for t in (p0, g0, m0, v0)]
    adamw_step_xla(new[0], new[1], new[2], new[3], step, lr, b1, b2, eps, wd)

    diffs = [float((a - b).abs().max()) for a, b in zip(up, new)]
    names = ["p", "exp_avg", "exp_avg_sq"]
    worst = max(diffs[:1] + diffs[2:])  # grad is not modified by either
    print(f"adamw  max|diff| p={diffs[0]:.3e} m={diffs[2]:.3e} v={diffs[3]:.3e}", flush=True)
    return worst


def check_muon():
    torch.manual_seed(1)
    chunks, m, n = 4, 16, 24
    grads0 = torch.randn(chunks, m, n)
    params0 = torch.randn(chunks, m, n)
    mom0 = torch.zeros_like(grads0)
    smom0 = torch.zeros(chunks, 1, n)
    # wd must be non-zero: with wd=0 the cautious-decay branch is dead code and a
    # transcription error there would pass unnoticed (found by mutation check).
    momentum, lr, wd, beta2, ns_steps, red_dim = 0.95, 0.02, 0.1, 0.9, 5, -2

    ug, up_, um = grads0.clone(), params0.clone(), mom0.clone()
    us = smom0.clone()
    O.muon_step_fused.__wrapped__(
        ug, up_, um, us,
        torch.tensor(momentum), torch.tensor(lr), torch.tensor(wd),
        torch.tensor(beta2), ns_steps, red_dim)

    ng, np_, nm = grads0.clone(), params0.clone(), mom0.clone()
    ns = smom0.clone()
    muon_step_xla(ng, np_, nm, ns, momentum, lr, wd, beta2, ns_steps, red_dim)

    dp = float((up_ - np_).abs().max())
    dm = float((um - nm).abs().max())
    ds = float((us - ns).abs().max())
    print(f"muon   max|diff| params={dp:.3e} momentum={dm:.3e} second={ds:.3e}", flush=True)
    return max(dp, dm, ds)


def main():
    worst = max(check_adamw(), check_muon())
    # bfloat16-free fp32 path: exact agreement to float rounding is expected.
    ok = worst < 1e-6
    print(f"RESULT worst={worst:.3e} equivalent={ok}", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

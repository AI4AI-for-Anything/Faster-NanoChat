"""TPU check: nanochat_splash.flash_attn_func == nanochat's SDPA fallback.

Locks the property the splash shim must have before any timing counts: for the
shapes NanoChat d24 trains with (B, T=2048, H=12, D=128, bf16), causal, both a
full-context layer (window 2048) and a short layer (window 768), forward output
and q/k/v gradients agree with nanochat.flash_attention's own SDPA path within
a bf16 tolerance. Also checks the mask boundary: a window off by one (767) MUST
be detected as different, otherwise the comparison would not see a wrong window.

Run on one host (4 chips, SPMD), Pallas environment (tpu_smoke/README.md):
  python -m tpu_smoke.test_splash_equiv
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)                     # repo root: nanochat.*, tpu_smoke.*

import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

xr.use_spmd()

from nanochat import flash_attention as fa          # noqa: E402
from tpu_smoke import nanochat_splash as ns         # noqa: E402

sdpa_ref = fa.flash_attn.flash_attn_func


def run(fn, q, k, v, window, g):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o = fn(q, k, v, causal=True, window_size=(window, 0))
    (o.float() * g).sum().backward()
    torch_xla.sync()
    return [t.cpu().float() for t in (o, q.grad, k.grad, v.grad)]


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


def main():
    n = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(n), (n, 1), ("data", "fsdp"))
    ns.set_mesh(mesh, ("data", "fsdp"))
    dev = torch_xla.device()
    B, T, H, D = n, 2048, 12, 128
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(B, T, H, D, generator=gen).to(torch.bfloat16).to(dev) for _ in range(3))
    g = torch.randn(B, T, H, D, generator=gen).to(dev)
    for t in (q, k, v, g):
        xs.mark_sharding(t, mesh, (("data", "fsdp"), None, None, None))
    TOL = 2e-2           # bf16 inputs, online vs materialised softmax
    worst = {}
    for window in (T, 768):
        ref = run(sdpa_ref, q, k, v, window, g)
        got = run(ns.flash_attn_func, q, k, v, window, g)
        errs = [rel(a, b) for a, b in zip(got, ref)]
        worst[window] = errs
        names = ("out", "dq", "dk", "dv")
        print(f"window {window}: " + " ".join(f"{nm}={e:.2e}" for nm, e in zip(names, errs)))
        assert max(errs) < TOL, f"window {window} exceeds {TOL}"
    # Discriminator: which window does splash actually apply? Compare its output
    # with SDPA at 768 (intended) and at 767 / 769 (off by one each way). The
    # intended one MUST be the closest by a clear margin; a bound like "10x the
    # numerical error" was tried first and is the wrong test (11-06): an
    # off-by-one window changes only boundary rows, so its max error is small.
    got768 = run(ns.flash_attn_func, q, k, v, 768, g)[0]
    dists = {w: rel(got768, run(sdpa_ref, q, k, v, w, g)[0]) for w in (767, 768, 769)}
    print("discriminator: splash(768) vs sdpa(w): " + " ".join(f"{w}={d:.2e}" for w, d in dists.items()))
    assert dists[768] * 3 < min(dists[767], dists[769]), "splash window is not 768"
    print("PASS splash == sdpa (fwd + grads), both window types; off-by-one detected")


if __name__ == "__main__":
    main()

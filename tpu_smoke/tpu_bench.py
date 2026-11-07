"""Timing harness for the NanoChat XLA port.

An earlier timing loop was invalid. It timed
`float(loss.detach())` and then `torch_xla.sync()`, but torch_xla.sync defaults to
wait=False -- it launches the pending graph and returns. So the optimizer step,
including any SPMD gradient reduction, was never inside the measured interval;
part of it surfaced during the next step's materialization. Fixed costs and
deferred work made batch size look free.

This harness follows the acceptance protocol instead:

  * warmup runs until compilation is done, at least --warmup steps, and is
    excluded from the measurement;
  * a window is --window consecutive steps, timed per step, with the device
    allowed to run ahead inside the window and a single blocking sync at the
    window boundary -- blocking every step would measure a different pipeline;
  * --windows independent windows are reported individually, plus the median;
    a window whose (max-min)/median exceeds 10% marks the run INVALID rather
    than being averaged in;
  * no gc.collect() inside the window (it is host work and it varies).

Anything reported from this script states which windows it came from, so a
regression can be checked against the run rather than trusted from a summary.
"""
import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)                     # repo root: nanochat.*, tpu_smoke.*

import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.core import xla_model as xm


def build(mesh, depth, seq, prod_shape=False):
    os.environ["NANOCHAT_DTYPE"] = "bfloat16"
    for m in [m for m in list(sys.modules) if m.startswith("nanochat")]:
        del sys.modules[m]
    from nanochat.gpt import GPT, GPTConfig
    from tpu_smoke.xla_optim_patch import (adamw_step_xla, muon_step_xla,
                                           patch_optimizer_for_xla)
    from nanochat import optim as O
    O.adamw_step_fused = adamw_step_xla
    O.muon_step_fused = muon_step_xla

    device = torch_xla.device()
    if prod_shape:
        # The upstream recipe's shape rule (scripts/base_train.py build_model_meta,
        # aspect 64, head_dim 128, SSSL, vocab 32768 as in GPTConfig): d24 is
        # n_embd 1536 with 12 heads. Not the tokenizer, data or schedule.
        dim = ((depth * 64 + 127) // 128) * 128
        cfg = GPTConfig(sequence_len=seq, vocab_size=32768, n_layer=depth,
                        n_head=dim // 128, n_kv_head=dim // 128, n_embd=dim,
                        window_pattern="SSSL")
    else:
        cfg = GPTConfig(sequence_len=seq, vocab_size=256, n_layer=depth,
                        n_head=2, n_kv_head=2, n_embd=256, window_pattern="SSL")
    torch.manual_seed(42)
    model = GPT(cfg)
    # Initialise once on CPU with the same seed in each independent process.
    model.init_weights()
    model.to(device)
    opt = model.setup_optimizer(matrix_lr=0.02)
    patch_optimizer_for_xla(opt)
    return model, opt, device


def sync_blocking():
    """Block until the device has finished every queued execution.

    torch_xla.sync(wait=True) is NOT this barrier. Measured on v6e (torch_xla 2.9.0):
    a 32-matmul 4096^2 bf16 chain
    timed 0.18 ms with sync(wait=True) -- 24,000 TFLOP/s, 26x the chip's peak --
    and 5.4 ms with wait_device_ops (817 TFLOP/s). wait=True waits for the host
    to finish handing the graph over, not for the device to run it. A window
    timed with it measures host submission only.
    """
    torch_xla.sync()
    xm.wait_device_ops()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spmd", "single"], default="spmd")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--window", type=int, default=35)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--compile", action="store_true",
                    help="wrap the whole step in torch_xla.compile(full_graph=True)")
    ap.add_argument("--skip-opt", action="store_true",
                    help="ablation: omit the optimizer step, to attribute the fixed "
                         "per-step cost. The trajectory is meaningless in this mode.")
    ap.add_argument("--gc", choices=["off", "window"], default="window",
                    help="where to collect Python garbage. Inside a step it is host "
                         "work that lands in the timing; off lets the known XLA "
                         "ref-cycle growth drift the numbers.")
    args = ap.parse_args()

    mesh = None
    if args.mode == "spmd":
        xr.use_spmd()
        count_fn = getattr(xr, "global_runtime_device_count", None) or xr.global_device_count
        n = count_fn()
        mesh = xs.Mesh(np.arange(n), (n,), ("data",))
        devices = n
    else:
        devices = 1

    model, opt, device = build(mesh, args.depth, args.seq)
    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 256, (args.batch, args.seq + 1), generator=gen)
    x, y = idx[:, :-1].to(device), idx[:, 1:].to(device)
    if mesh is not None:
        x = xs.mark_sharding(x, mesh, ("data", None))
        y = xs.mark_sharding(y, mesh, ("data", None))

    def one_step():
        loss = model(x, y)
        loss.backward()
        if args.skip_opt:
            # Ablation: leaves the optimizer out so the fixed per-step cost can
            # be attributed. Trajectory is meaningless in this mode.
            return loss.detach()
        opt.step()
        model.zero_grad(set_to_none=True)
        return loss.detach()

    step_fn = torch_xla.compile(one_step, full_graph=True) if args.compile else one_step

    def maybe_gc():
        if args.gc == "window":
            gc.collect()

    # Warmup: run until compiles settle, then keep going to --warmup steps.
    t_compile = time.time()
    for _ in range(args.warmup):
        loss = step_fn()
        sync_blocking()
    compile_s = time.time() - t_compile
    first_loss = float(loss)
    maybe_gc()

    windows = []
    for w in range(args.windows):
        sync_blocking()
        started = time.perf_counter()
        for _ in range(args.window):
            loss = step_fn()
            # A lazy step boundary is REQUIRED even when execution is asynchronous.
            torch_xla.sync(wait=False)
        sync_blocking()
        elapsed = time.perf_counter() - started
        windows.append(dict(window=w, elapsed_s=elapsed,
                            mean_ms=1000 * elapsed / args.window))
        maybe_gc()

    medians = [w["mean_ms"] for w in windows]
    spread = (max(medians) - min(medians)) / statistics.median(medians)
    valid = spread <= 0.10
    result = dict(
        mode=args.mode, devices=devices, depth=args.depth, seq=args.seq,
        batch=args.batch, warmup=args.warmup, window=args.window,
        windows=windows, median_of_medians_ms=round(statistics.median(medians), 1),
        valid=valid, compile_warmup_s=round(compile_s, 1),
        first_loss=round(first_loss, 4),
        torch=torch.__version__, torch_xla=torch_xla.__version__,
        # Which switches actually took effect, so two runs can be diffed before
        # their numbers are compared.
        env={k: os.environ.get(k) for k in
             ("PJRT_DEVICE", "XLA_SYNC_WAIT", "TORCHDYNAMO_DISABLE",
              "NANOCHAT_DTYPE", "TPU_VISIBLE_CHIPS")},
    )
    print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Does NanoChat's SPMD data-parallel step scale across hosts of a v6e-64 pod?

On one host, four-chip SPMD keeps DEVICE time equal to the single-chip time at
the same per-chip batch, so the compute is really split and the gradient
reduction costs under 1%. That says nothing about the inter-host links, which
is where data parallelism on a pod usually stops scaling. This runs the same
step with one process per host, every process launched with the same command,
and the global batch sharded over all chips of the slice.

Input path: every process builds the SAME global batch from the same seed and
transfers it whole, then mark_sharding splits it. That is wasteful at scale but
keeps the data identical to the single-host run, so DEVICE can be compared
directly with it. A real loader would feed only the local rows.

Timing: TRACE / SUBMIT / DEVICE per step, each step from an
idle device, 6 windows, window 0 tainted by compilation. Every process prints its
own SUMMARY; the step is as slow as the slowest host, so compare the max.

Launch on all workers at once (gcloud ... --worker=all), WITHOUT the single-host
isolation variables (TPU_SKIP_MDS_QUERY, TPU_*_BOUNDS, TPU_VISIBLE_CHIPS): the
runtime must read the slice topology from metadata to see every host.
"""
import argparse
import json
import os
import socket
import statistics
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch_xla
import torch_xla.debug.metrics as tmetrics
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.core import xla_model as xm

from tpu_smoke.tpu_bench import build


def uncached():
    names = tmetrics.counter_names()
    return tmetrics.counter_value("UncachedCompile") if "UncachedCompile" in names else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--per-chip-batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--window", type=int, default=35)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--prod-shape", action="store_true",
                    help="upstream recipe shape (d24 -> 1536 dim, vocab 32768) instead "
                         "of the 256-dim smoke model")
    ap.add_argument("--vocab", type=int, default=256,
                    help="token id range of the synthetic batch; must not exceed the "
                         "model vocab")
    ap.add_argument("--fsdp", action="store_true",
                    help="shard every parameter with a dim divisible by the chip count "
                         "along that dim (ZeRO-3 style through SPMD annotations). The "
                         "optimizer state is created lazily from the parameters on the "
                         "first step, so it is checked after warmup, not assumed.")
    ap.add_argument("--fsdp-size", type=int, default=0,
                    help="2-D mesh (data = chips/fsdp_size, fsdp = fsdp_size): parameters "
                         "shard over the fsdp axis only, the batch over both. 0 = 1-D, "
                         "parameters over all chips. fsdp_size 4 keeps the "
                         "parameter all-gathers inside one host's chips.")
    ap.add_argument("--skip-opt", action="store_true",
                    help="attribution only: omit opt.step(). The pipelined step minus this "
                         "is the optimizer's share. Trajectory meaningless.")
    ap.add_argument("--splash", action="store_true",
                    help="route nanochat's flash_attn_func to Pallas splash attention "
                         "(tpu_smoke/nanochat_splash.py; N1, checked by "
                         "test_splash_equiv.py). Needs the Pallas environment, see "
                         "tpu_smoke/README.md.")
    ap.add_argument("--pin-activations", action="store_true",
                    help="force block outputs to batch sharding (refuted OOM fix)")
    ap.add_argument("--dynamo", action="store_true",
                    help="wrap the forward in torch.compile(backend='openxla'). "
                         "N1 (not bitwise), diagnostic only. "
                         "Needs TORCHDYNAMO_DISABLE unset.")
    ap.add_argument("--traj", type=int, default=0,
                    help="statistical-gate mode: N steps from init, a FRESH batch per step from a "
                         "fixed sparse Markov chain (learnable, unlike uniform tokens), print TRAJ")
    ap.add_argument("--seed", type=int, default=0, help="traj mode: data seed (init stays seed 42)")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="traj mode: multiply every group's lr (sensitivity control 1.02)")
    args = ap.parse_args()

    xr.use_spmd()
    n = xr.global_runtime_device_count()
    if args.fsdp_size:
        assert n % args.fsdp_size == 0, (n, args.fsdp_size)
        mesh = xs.Mesh(np.arange(n), (n // args.fsdp_size, args.fsdp_size), ("data", "fsdp"))
        batch_axes, param_axis, param_ways = ("data", "fsdp"), "fsdp", args.fsdp_size
    else:
        mesh = xs.Mesh(np.arange(n), (n,), ("data",))
        batch_axes, param_axis, param_ways = "data", "data", n
    proc, nproc = xr.process_index(), xr.process_count()
    batch = args.per_chip_batch * n

    model, opt, device = build(None, args.depth, args.seq, args.prod_shape)
    if args.splash:
        # build() re-imports nanochat, so the patch MUST go on the fresh module.
        from nanochat import flash_attention as nfa
        from tpu_smoke import nanochat_splash
        if not args.fsdp_size:
            raise SystemExit("--splash needs --fsdp-size: splash shards q/k/v on (data, fsdp)")
        nanochat_splash.set_mesh(mesh, batch_axes)
        nanochat_splash.install(nfa)
    sharded = replicated = 0
    if args.fsdp:
        for prm in model.parameters():
            dims = [d for d in range(prm.dim()) if prm.shape[d] % param_ways == 0]
            if dims:
                spec = [None] * prm.dim()
                spec[dims[0]] = param_axis
                xs.mark_sharding(prm, mesh, tuple(spec))
                sharded += prm.numel()
            else:
                replicated += prm.numel()
    if args.pin_activations:
        # Hypothesis tested and REFUTED: that the 64-chip per-chip-batch-4 OOM
        # came from GSPMD laying activations out along a parameter's sharded dim
        # (batch replicated). Pinning every block output to batch sharding left the
        # OOM byte-for-byte the same (121.6M asked, 104.7M free). Kept opt-in only:
        # the hook's mark_sharding inside forward breaks dynamo tracing
        # ('XLAShardedTensor' object has no attribute '_local_tensor').
        def pin_batch(_mod, _inp, out):
            if torch.is_tensor(out) and out.dim() == 3:
                xs.mark_sharding(out, mesh, ("data", None, None))
            return out
        model.transformer.wte.register_forward_hook(pin_batch)
        for blk in model.transformer.h:
            blk.register_forward_hook(pin_batch)
        model.lm_head.register_forward_hook(pin_batch)
    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(0, args.vocab, (batch, args.seq + 1), generator=gen)
    x, y = idx[:, :-1].to(device), idx[:, 1:].to(device)
    xs.mark_sharding(x, mesh, (batch_axes, None))
    xs.mark_sharding(y, mesh, (batch_axes, None))

    fwd = torch.compile(model, backend="openxla") if args.dynamo else model

    def one_step():
        model.zero_grad(set_to_none=True)
        loss = fwd(x, y)
        loss.backward()
        if not args.skip_opt:
            opt.step()
        return loss.detach()

    if args.traj:
        for group in opt.param_groups:
            group["lr"] *= args.lr_scale
        # Each token has 4 possible successors, so the chain's entropy floor is
        # log(4) and lr x1.02 moves the trajectory measurably (same chain as llava_step).
        succ = np.random.default_rng(1234).integers(0, args.vocab, (args.vocab, 4))
        traj = []
        for step in range(args.traj):
            r = np.random.default_rng(args.seed * 1000003 + step)
            tok = np.empty((batch, args.seq + 1), dtype=np.int64)
            tok[:, 0] = r.integers(0, args.vocab, batch)
            pick = r.integers(0, 4, (batch, args.seq))
            for t in range(args.seq):
                tok[:, t + 1] = succ[tok[:, t], pick[:, t]]
            idx = torch.from_numpy(tok)
            x, y = idx[:, :-1].to(device), idx[:, 1:].to(device)
            xs.mark_sharding(x, mesh, (batch_axes, None))
            xs.mark_sharding(y, mesh, (batch_axes, None))
            loss = one_step()
            torch_xla.sync()
            traj.append([float(loss), None])
        print("TRAJ " + json.dumps(dict(seed=args.seed, lr_scale=args.lr_scale,
                                        splash=args.splash, traj=traj)), flush=True)
        return

    t0 = time.time()
    for _ in range(args.warmup):
        loss = one_step()
        torch_xla.sync()
        xm.wait_device_ops()
    warm_s = time.time() - t0

    # Optimizer state sharding, read off the real tensors after the first steps.
    state_shard = {}
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v) and v.device.type == "xla" and v.dim() >= 2:
                spec = torch_xla._XLAC._get_xla_sharding_spec(v) or "replicated"
                state_shard[spec.split("}")[0][:24]] = state_shard.get(spec.split("}")[0][:24], 0) + v.numel()
    try:
        hbm_gib = round(xm.get_memory_info(device)["bytes_used"] / 2**30, 2)
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        hbm_gib = f"ERR:{type(exc).__name__}"

    windows = []
    for w in range(args.windows):
        u0 = uncached()
        trace, submit, dev = [], [], []
        for _ in range(args.window):
            t0 = time.perf_counter()
            loss = one_step()
            t1 = time.perf_counter()
            torch_xla.sync()
            t2 = time.perf_counter()
            xm.wait_device_ops()
            t3 = time.perf_counter()
            trace.append(1000 * (t1 - t0))
            submit.append(1000 * (t2 - t1))
            dev.append(1000 * (t3 - t2))
        last = float(loss)
        u1 = uncached()
        windows.append(dict(
            trace_ms=round(statistics.median(trace), 2),
            submit_ms=round(statistics.median(submit), 2),
            device_ms=round(statistics.median(dev), 2),
            total_ms=round(statistics.median(
                [a + b + c for a, b, c in zip(trace, submit, dev)]), 2),
            tainted=bool(u1 - u0) if None not in (u0, u1) else None))

    # Pipelined window: what a real training loop pays. Every window above
    # blocks per step (TRACE+SUBMIT+DEVICE, each step from an idle device);
    # here the host only cuts the graph with sync() and traces step k+1 while the
    # device runs step k, so the step costs about max(host, device).
    # The result is kept live and the device drained once at the end.
    t_a = time.perf_counter()
    for _ in range(args.window):
        res = one_step()
        torch_xla.sync()
    xm.wait_device_ops()
    pipelined_ms = round(1000 * (time.perf_counter() - t_a) / args.window, 2)
    _ = [float(r) for r in (res if isinstance(res, tuple) else (res,))]
    clean = [w for w in windows if not w["tainted"]] or windows
    tot = [w["total_ms"] for w in clean]
    print("SUMMARY " + json.dumps(dict(
        host=socket.gethostname(), process=proc, processes=nproc, chips=int(n),
        depth=args.depth, prod_shape=args.prod_shape, n_params=sum(p.numel() for p in model.parameters()),
        seq=args.seq, per_chip_batch=args.per_chip_batch,
        global_batch=batch, tokens=batch * args.seq, warmup_s=round(warm_s, 1),
        fsdp=args.fsdp, fsdp_size=args.fsdp_size, splash=args.splash, skip_opt=args.skip_opt, dynamo=args.dynamo, sharded_params=sharded, replicated_params=replicated,
        opt_state_sharding=state_shard, hbm_used_gib=hbm_gib,
        tainted=[w["tainted"] for w in windows], pipelined_ms=pipelined_ms,
        trace_ms=round(statistics.median(w["trace_ms"] for w in clean), 2),
        submit_ms=round(statistics.median(w["submit_ms"] for w in clean), 2),
        device_ms=round(statistics.median(w["device_ms"] for w in clean), 2),
        total_ms=round(statistics.median(tot), 2),
        spread=round((max(tot) - min(tot)) / statistics.median(tot), 3),
        last_loss=repr(last))), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # The runtime's status text (e.g. RESOURCE_EXHAUSTED with the HBM figures)
        # was lost once: the traceback printed but its message line did not reach
        # the log. Print it explicitly, then re-raise so the exit code stays nonzero.
        print("FAILED " + type(exc).__name__ + ": " + str(exc)[:3000], flush=True)
        raise

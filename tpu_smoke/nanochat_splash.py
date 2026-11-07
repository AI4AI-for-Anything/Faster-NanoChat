"""Pallas flash attention for NanoChat on TPU: same function, fewer FLOPs?

NanoChat's attention on non-CUDA hardware is its SDPA fallback
(nanochat/flash_attention.py:_sdpa_attention): for sliding-
window layers it builds a dense [T, T] boolean mask and calls SDPA, so on XLA
every layer computes the full T x T score matrix regardless of the window. At
d24 / seq 2048 that is about 3x the attention FLOPs the windows need (FLOP
arithmetic), plus materialised fp32 scores for backward.

This module provides the drop-in that nanochat's `flash_attn.flash_attn_func`
interface expects ([B, T, H, D] layout, causal, window_size=(left, 0)) on top
of torch_xla's Pallas splash attention, which takes a LocalMask for sliding
windows and never builds the dense matrix. It is installed by monkeypatching
the module attribute from outside (`install(nanochat.flash_attention)`), so the
upstream file is unchanged.

Numerical tier: N1 at best (same dtype, different accumulation order, online
softmax). tpu_smoke/test_splash_equiv.py compares it with the SDPA fallback on
TPU before any timing is taken.

Requires the pallas extra (jax 0.7.1) -- run with the Pallas environment
described in tpu_smoke/README.md.
"""
import torch

from torch_xla.experimental.splash_attention import SplashAttentionConfig, splash_attention


def _pin_mosaic_ir_version(version=7):
    """Emit Mosaic IR the installed libtpu can read.

    torch_xla's pallas extra pins jax 0.7.1, which serialises Mosaic kernels at
    IR version 8 unless it detects an older Cloud TPU runtime. Under call_jax it
    does not (jax has no TPU backend of its own here), so libtpu 0.0.21 rejected
    the kernel: "Failed to deserialize the Mosaic module: Unsupported version:
    expected <= 7 but got 8" (11-06). jax's own rule maps runtimes built before
    2025-08-14 to version 7; this applies that rule explicitly instead of
    editing site-packages.
    """
    from jax._src import tpu_custom_call as tcc
    tcc.get_ir_version = lambda ctx: version


_pin_mosaic_ir_version()


def _config(T, window, mesh_name, qkv_spec):
    blk = min(512, T)
    return SplashAttentionConfig(
        sa_block_q=blk, sa_block_kv=blk, sa_block_kv_compute=blk,
        sa_block_q_dkv=blk, sa_block_kv_dkv=blk, sa_block_kv_dkv_compute=blk,
        sa_block_q_dq=blk, sa_block_kv_dq=blk,
        mesh=mesh_name, qkv_partition_spec=qkv_spec,
        segment_ids_partition_spec=(qkv_spec[0], None),
        attentiontype_local_sliding=window is not None,
        slide_window_size=window).to_json()


_MESH = {"name": None, "spec": (("data",), None, None, None)}


def set_mesh(mesh, batch_axes):
    """Splash runs under shard_map; it needs the mesh and how q/k/v are sharded."""
    _MESH["name"] = str(mesh)
    ax = batch_axes if isinstance(batch_axes, tuple) else (batch_axes,)
    _MESH["spec"] = (ax, None, None, None)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """nanochat FA3 signature: q,k,v [B, T, H, D] -> [B, T, H, D]."""
    if not causal:
        raise NotImplementedError("splash shim: nanochat training is always causal")
    B, T, H, D = q.shape
    if k.shape[2] != H:
        k = k.repeat_interleave(H // k.shape[2], dim=2)
        v = v.repeat_interleave(H // v.shape[2], dim=2)
    left = window_size[0]
    window = None if (left < 0 or left >= T) else left
    cfg = _config(T, window, _MESH["name"], _MESH["spec"])
    # splash wants [B, H, T, D] and applies no 1/sqrt(D) itself.
    scale = D ** -0.5
    o = splash_attention((q * scale).transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                         cfg, None, True)
    return o.transpose(1, 2)


def install(fa_module):
    """Point nanochat.flash_attention.flash_attn.flash_attn_func at splash."""
    fa_module.flash_attn.flash_attn_func = flash_attn_func

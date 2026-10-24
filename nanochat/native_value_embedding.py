"""Native FP32 backward accumulation for explicitly selected value embeddings.

The module inherits nn.Embedding construction and weight registration. Its
ordinary traceable autograd Function keeps lookup, promotion, padding mask and
final BF16 conversion visible to compilation. Only native accumulation is
opaque; the token embedding and global decomposition table are unchanged.
"""
import atexit
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

_OP = None
_COUNTS = Counter()


def forward_reason(module, indices):
    weight = module.weight
    if (not module.training or not torch.is_grad_enabled()
            or not weight.requires_grad):
        return 'training_state'
    if (module.padding_idx is not None or module.max_norm is not None
            or module.scale_grad_by_freq is not False or module.sparse is not False
            or module.num_embeddings != 32768 or module.embedding_dim != 1536):
        return 'options'
    if (weight.device.type != 'cuda' or indices.device != weight.device
            or weight.layout != torch.strided or indices.layout != torch.strided):
        return 'device_or_layout'
    if weight.dtype != torch.bfloat16 or indices.dtype != torch.int64:
        return 'dtype'
    shapes = (tuple(weight.shape), tuple(indices.shape))
    strides = (tuple(weight.stride()), tuple(indices.stride()))
    if not all(type(x) is int for xs in (*shapes, *strides) for x in xs):
        return 'symbolic_metadata'
    if shapes != ((32768, 1536), (16, 2048)):
        return 'shape'
    if strides != ((1536, 1), (2048, 1)):
        return 'stride'
    # These real dtypes cannot carry a conjugate bit. ATen's functional
    # dispatch handles logical negative-bit values; do not trace unsupported
    # bool-returning Tensor.is_conj/is_neg methods here.
    return None


def accumulation_reason(grad, indices, num_weights, padding_idx):
    if (type(num_weights) is not int or num_weights != 32768
            or type(padding_idx) is not int or padding_idx != -1):
        return 'arguments'
    if (grad.device.type != 'cuda' or indices.device != grad.device
            or grad.layout != torch.strided or indices.layout != torch.strided):
        return 'device_or_layout'
    if grad.dtype != torch.float32 or indices.dtype != torch.int64:
        return 'dtype'
    shapes = (tuple(grad.shape), tuple(indices.shape))
    strides = (tuple(grad.stride()), tuple(indices.stride()))
    if not all(type(x) is int for xs in (*shapes, *strides) for x in xs):
        return 'symbolic_metadata'
    if shapes != ((16, 2048, 1536), (16, 2048)):
        return 'shape'
    if strides != ((3145728, 1536, 1), (2048, 1)):
        return 'stride'
    return None


class _ValueEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, indices):
        ctx.save_for_backward(indices)
        ctx.num_weights = weight.shape[0]
        ctx.result_dtype = weight.dtype
        return torch.ops.aten.embedding.default(weight, indices, -1, False, False)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        (indices,) = ctx.saved_tensors
        # Fixed BF16 module contract: upstream opmath promotes to FP32, masks
        # padding and only rounds back to BF16 after embedding accumulation.
        grad_output = grad_output.to(torch.float32)
        indices = indices.to(torch.long)
        mask = (indices == -1).unsqueeze(-1)
        grad = grad_output.masked_fill(mask, 0)
        return _OP(grad, indices, ctx.num_weights, -1).to(ctx.result_dtype), None


class ValueEmbedding(nn.Embedding):
    def forward(self, indices):
        # No Python counters or logging in the traced forward/backward. Missing
        # native call counts are checked against all 12 expected module roles.
        if forward_reason(self, indices) is not None:
            return super().forward(indices)
        return _ValueEmbeddingFunction.apply(self.weight, indices)


def _setup_context(ctx, inputs, output):
    _, indices, _, padding_idx = inputs
    ctx.save_for_backward(indices)
    ctx.padding_idx = padding_idx


def _backward(ctx, grad_weight):
    if grad_weight is None:
        return None, None, None, None
    (indices,) = ctx.saved_tensors
    grad = grad_weight.index_select(0, indices.reshape(-1))
    if ctx.padding_idx >= 0:
        grad = grad.masked_fill((indices == ctx.padding_idx).reshape(-1, 1), 0)
    return grad.reshape(*indices.shape, grad_weight.shape[-1]), None, None, None


def enable():
    global _OP
    import torch.compiler.config as compiler_config
    import torch._decomp.decompositions as base
    import torch.nn.functional as functional
    import torch.nn.modules.sparse as sparse
    from torch._inductor import config, decomposition
    from torch._inductor.virtualized import V

    if _OP is not None:
        raise RuntimeError('install native value embedding only once')
    if (config.cpp_wrapper or config.profile_bandwidth or V.aot_compilation
            or config.triton.cudagraphs):
        raise RuntimeError('native value embedding supports ordinary Python JIT')
    key = torch.ops.aten.embedding_dense_backward.default
    if decomposition.decompositions.get(key) is not decomposition._embedding_dense_backward:
        raise RuntimeError('native value embedding requires the original global decomposition')
    for module, digest in {
        base: '4061f0dd317b66822c2a324824110063cc262635e44ae7996f84538ce4f4d463',
        decomposition: 'ba6ec417bac0b61d5617f3ee44f64c098cb3e0ac24edd8580d4d4384e2d30bdc',
        sparse: '9efa8d236f5b29c42ffb2ed2822c01d8b255c54bb5853a8f919e70c06d101fab',
        functional: '5e96bb0880a4c6f4c0ae34c00a894e152412e68f6ffd073d2b442907eedeb5fa',
    }.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError('native value embedding requires reviewed Torch sources')

    @torch.library.custom_op('nanochat::value_embedding_accumulate_fp32',
                             mutates_args=(), device_types='cuda')
    def accumulate(grad: torch.Tensor, indices: torch.Tensor,
                   num_weights: int, padding_idx: int) -> torch.Tensor:
        reason = accumulation_reason(grad, indices, num_weights, padding_idx)
        if reason is not None:
            raise RuntimeError('unexpected native value embedding input: ' + reason)
        _COUNTS['native_calls'] += 1
        return torch.ops.aten.embedding_dense_backward.default(
            grad, indices, num_weights, padding_idx, False)

    @accumulate.register_fake
    def fake(grad, indices, num_weights, padding_idx):
        reason = accumulation_reason(grad, indices, num_weights, padding_idx)
        if reason is not None:
            raise RuntimeError('unexpected fake value embedding input: ' + reason)
        return grad.new_empty((num_weights, grad.shape[-1]))

    accumulate.register_autograd(_backward, setup_context=_setup_context)
    _OP = accumulate
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    compiler_config.cache_key_tag += ':nanochat-native-value-embedding-' + version
    print('NANOCHAT_OPT_NATIVE_VALUE: installed ' + json.dumps({
        'rank': os.environ.get('RANK'), 'pid': os.getpid(), 'helper': version,
        'scope': 'GPT.value_embeds only', 'input_dtype': 'bfloat16',
        'accumulation_dtype': 'float32', 'output_dtype': 'bfloat16',
        'cache_key_tag': compiler_config.cache_key_tag,
    }, sort_keys=True), flush=True)

    def report():
        print('NANOCHAT_OPT_NATIVE_VALUE: counters ' + json.dumps({
            'rank': os.environ.get('RANK'), 'pid': os.getpid(),
            'helper': version, 'counts': dict(_COUNTS),
        }, sort_keys=True), flush=True)
    atexit.register(report)

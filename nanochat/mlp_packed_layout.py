"""Pair the two BF16 operands for one possible shared layout conversion.

Transport raw bits in uint32, then retain native BF16-to-FP32 conversions.
All arithmetic, factory results, launch settings and stores stay unchanged.
"""
import ast
import copy
import functools
import hashlib
import json
import os
from pathlib import Path

from nanochat.scalar_scale_split import _fields

# Complete post-split source contract. Only name, rank and backend hash vary.
_EXPECTED = r"""
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'y': 32768, 'x': 8192}, tile_hint=TileHint.DEFAULT, filename=__file__, triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp8e4nv', 'out_ptr1': '*fp8e5', 'out_ptr2': '*fp8e5', 'ynumel': 'i32', 'xnumel': 'i32', '_nanochat_scale_input': '*fp32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': '_kernel', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '_backend', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 805306368, 'x': 1207959552}}, min_elem_per_thread=4)
@triton.jit
def _kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, _nanochat_scale_input, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 32768
    xnumel = 6144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 6144 * y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr1 + 0)
    tmp6 = tl.broadcast_to(tmp5, [YBLOCK, XBLOCK])
    tmp15 = tl.load(in_ptr2 + (x1 + 6144 * y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tl.load(in_ptr3 + 0)
    tmp22 = tmp21
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp7 = tmp4 * tmp6
    tmp8 = -448.0
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = 448.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tmp11.to(tl.float8e4nv)
    tmp13 = 0.0
    tmp14 = tmp2 <= tmp13
    tmp16 = 2.0
    tmp17 = tmp2 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tl.where(tmp14, tmp13, tmp18)
    tmp20 = tmp19.to(tl.float32)
    tmp23 = tmp22.to(tl.float64)
    tmp24 = tl.full([1, 1], 1e-12, tl.float64)
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp26 = tl.full([1, 1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = tl.full([1, 1], 57344.0, tl.float64)
    tmp29 = tmp27 * tmp28
    _nanochat_singleton_tmp30 = tl.load(_nanochat_scale_input + 0)
    tmp30 = tl.broadcast_to(_nanochat_singleton_tmp30, [YBLOCK, XBLOCK])
    tmp31 = tmp20 * tmp30
    tmp32 = -57344.0
    tmp33 = triton_helpers.maximum(tmp31, tmp32)
    tmp34 = 57344.0
    tmp35 = triton_helpers.minimum(tmp33, tmp34)
    tmp36 = tmp35.to(tl.float8e5)
    tl.store(out_ptr0 + (y0 + 32768 * x1), tmp12, xmask)
    tl.store(out_ptr1 + (x1 + 6144 * y0), tmp36, xmask)
    tl.store(out_ptr2 + (y0 + 32768 * x1), tmp36, xmask)

"""
_EXPECTED_AST = ast.dump(ast.parse(_EXPECTED))


def rewrite(source, kernel_name):
    """Bundle two co-indexed BF16 loads; leave unsupported sources unchanged."""
    # AST columns are UTF-8 byte offsets; captured sources use ASCII and LF.
    # Reject other separators that splitlines and Python tokenize differently.
    if (not isinstance(source, str) or not source.isascii()
            or any(c in source for c in '\r\v\f\x1c\x1d\x1e')):
        return source, False
    try:
        tree = ast.parse(source)
        normalized = copy.deepcopy(tree)
        functions = [n for n in normalized.body if isinstance(n, ast.FunctionDef)]
        if len(functions) != 1 or functions[0].name != kernel_name:
            return source, False
        fn = functions[0]
        fn.name = '_kernel'
        dec = fn.decorator_list[0]
        kw = {k.arg: k.value for k in dec.keywords}
        if len(kw) != len(dec.keywords):
            return source, False
        tm = _fields(kw['triton_meta'])
        device = tm['device']
        props = {k.arg: ast.literal_eval(k.value) for k in device.keywords}
        if (len(props) != len(device.keywords) or type(props.get('index')) is not int
                or not 0 <= props['index'] < 8):
            return source, False
        next(k for k in device.keywords if k.arg == 'index').value = ast.Constant(0)
        im = _fields(kw['inductor_meta'])
        if (ast.literal_eval(im['kernel_name']) != kernel_name
                or not isinstance(ast.literal_eval(im['backend_hash']), str)):
            return source, False
        im['kernel_name'].value = '_kernel'
        im['backend_hash'].value = '_backend'
        if ast.dump(normalized) != _EXPECTED_AST:
            return source, False
        from torch._inductor import config
        if (config.autotune_lookup_table or config.coordinate_descent_tuning
                or config.max_autotune or config.max_autotune_pointwise):
            return source, False
    except (SyntaxError, ValueError, TypeError, KeyError, IndexError,
            AttributeError, StopIteration):
        return source, False

    # The complete AST guard fixes both load addresses, masks, dtypes and users.
    # Move the second readonly load before its first consumer, pair the raw bits,
    # and keep the same native cvt semantics for both halves (including NaNs).
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    assignments = {n.targets[0].id: n for n in fn.body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)}
    first, second = assignments['tmp0'], assignments['tmp15']
    lines = source.splitlines(keepends=True)
    for node in (first, second):
        # Inserting multiline statements requires this exact body indentation.
        # Never delete half of a semicolon-separated line or adjacent statement.
        tail = lines[node.end_lineno - 1][node.end_col_offset:].strip()
        if (node.col_offset != 4 or lines[node.lineno - 1][:node.col_offset] != '    '
                or (tail and not tail.startswith('#'))):
            return source, False
    load_a = ast.get_source_segment(source, first.value.func.value)
    load_b = ast.get_source_segment(source, second.value.func.value)
    asm = ("{ .reg .b16 a, b; mov.b32 {a, b}, $2; "
           "cvt.f32.bf16 $0, a; cvt.f32.bf16 $1, b; }")
    replacement = (
        '_nanochat_pair_a = ' + load_a + '.to(tl.uint16, bitcast=True).to(tl.uint32)\n'
        '    _nanochat_pair_b = ' + load_b + '.to(tl.uint16, bitcast=True).to(tl.uint32)\n'
        '    _nanochat_pair = _nanochat_pair_a | (_nanochat_pair_b << 16)\n'
        '    tmp0, tmp15 = tl.inline_asm_elementwise(' + repr(asm) +
        ', constraints="=f,=f,r", args=[_nanochat_pair], '
        'dtype=(tl.float32, tl.float32), is_pure=True, pack=1)')
    def span(node):
        return (sum(map(len, lines[:node.lineno - 1])) + node.col_offset,
                sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset)
    edits = [(span(first), replacement), (span(second), '')]
    rewritten = source
    for (begin, end), text in sorted(edits, reverse=True):
        rewritten = rewritten[:begin] + text + rewritten[end:]
    try:
        ast.parse(rewritten)
    except SyntaxError:
        return source, False
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return '# nanochat_mlp_packed_source=' + version + '\n' + rewritten, True


def enable():
    import torch.compiler.config as compiler_config
    from torch._inductor import config
    from torch._inductor.async_compile import AsyncCompile
    from torch._inductor.virtualized import V

    def supported():
        if (config.cpp_wrapper or config.profile_bandwidth or V.aot_compilation
                or config.triton.cudagraphs or config.triton.multi_kernel
                or config.autotune_lookup_table or config.coordinate_descent_tuning
                or config.max_autotune or config.max_autotune_pointwise):
            raise RuntimeError('MLP packed layout requires ordinary fixed-config Python JIT')

    supported()
    original = AsyncCompile.triton
    if any(getattr(original, flag, False) for flag in (
        '_nanochat_mlp_packed', '_nanochat_persistent_scale', '_nanochat_max_rowpack',
        '_nanochat_norm_rowpack', '_nanochat_split_scale'
    )):
        raise RuntimeError('install MLP packed layout once, before the existing split/rowpack chain')
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    compiler_config.cache_key_tag += ':nanochat-mlp-packed-' + version

    @functools.wraps(original)
    def compile_triton(self, kernel_name, source_code, device_str='cuda'):
        supported()
        rewritten, matched = (rewrite(source_code, kernel_name) if device_str == 'cuda'
                              else (source_code, False))
        if matched:
            print('NANOCHAT_OPT_MLP_PACKED: ' + json.dumps({
                'rank': os.environ.get('RANK'), 'kernel': kernel_name,
                'input': hashlib.sha256(source_code.encode()).hexdigest(),
                'output': hashlib.sha256(rewritten.encode()).hexdigest(),
                'helper': version
            }, sort_keys=True), flush=True)
        return original(self, kernel_name, rewritten, device_str)

    compile_triton._nanochat_mlp_packed = True
    AsyncCompile.triton = compile_triton

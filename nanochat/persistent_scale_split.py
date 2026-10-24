"""Extract the singleton FP64 scale from one exact persistent reduction.

The main reduction, gating math and writes stay intact. The original kernel is
retained for unsupported runtime layouts, aliases or streams. N1 validation is
required; moving this chain can alter compiler scheduling and rounding.
"""
import ast
import atexit
import copy
import functools
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from nanochat import scalar_scale_split as split

# Full captured source selector, with only kernel name, rank and backend hash
# normalized. Constants here describe generated code; they replace no data.
_EXPECTED = r"""
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(size_hints={'x': 524288, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=__file__, triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'out_ptr0': '*fp8e5', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': '_kernel', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '_backend', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296, 'r0_': 805306368}})
@triton.jit
def _kernel(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 393216
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x3 = xindex // 12
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128 * x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 128 * x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + 0)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp23 = tl.load(in_ptr3 + x3, None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr4 + x0, None).to(tl.float32)
    tmp36 = tl.load(in_ptr4 + x0, None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tmp0.to(tl.float32)
    tmp9 = tmp8.to(tl.float64)
    tmp10 = tl.full([1, 1], 1e-12, tl.float64)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = tl.full([1, 1], 57344.0, tl.float64)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp6 * tmp16
    tmp18 = -57344.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 57344.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp21.to(tl.float8e5)
    tmp24 = tl.full([1, 1], -1, tl.int64)
    tmp25 = tmp23 == tmp24
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = 3.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp0 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = 0.0
    tmp33 = tl.where(tmp25, tmp32, tmp31)
    tmp34 = tmp5 * tmp28
    tmp35 = tmp34.to(tl.float32)
    tmp37 = tl.sigmoid(tmp36)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = 1.0
    tmp40 = tmp39 - tmp38
    tmp41 = tmp38 * tmp40
    tmp42 = tmp35 * tmp41
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr0 + (r0_1 + 128 * x0), tmp22, None)
    tl.store(out_ptr1 + (r0_1 + 128 * x0), tmp33, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp43, None)
"""
_EXPECTED_AST = ast.dump(ast.parse(_EXPECTED))


def split_source(source, kernel_name):
    """Accept the captured AST only; never execute generated source to inspect it."""
    if 'triton_heuristics.persistent_reduction' not in source or '_nanochat_' in source:
        return None
    try:
        tree = ast.parse(source)
        normalized = copy.deepcopy(tree)
        funcs = [n for n in normalized.body if isinstance(n, ast.FunctionDef)]
        if len(funcs) != 1 or funcs[0].name != kernel_name:
            return None
        fn = funcs[0]
        fn.name = '_kernel'
        dec = fn.decorator_list[0]
        kw = {k.arg: k.value for k in dec.keywords}
        if len(kw) != len(dec.keywords):
            return None
        meta = split._fields(kw['triton_meta'])
        device = meta['device']
        props = {k.arg: ast.literal_eval(k.value) for k in device.keywords}
        if (len(props) != len(device.keywords) or type(props.get('index')) is not int
                or not 0 <= props['index'] < 8):
            return None
        next(k for k in device.keywords if k.arg == 'index').value = ast.Constant(0)
        imeta = split._fields(kw['inductor_meta'])
        if (ast.literal_eval(imeta['kernel_name']) != kernel_name
                or not isinstance(ast.literal_eval(imeta['backend_hash']), str)):
            return None
        imeta['kernel_name'].value = '_kernel'
        imeta['backend_hash'].value = '_backend'
        if ast.dump(normalized) != _EXPECTED_AST:
            return None
        from torch._inductor import config
        if config.autotune_lookup_table or config.coordinate_descent_tuning:
            return None
    except (SyntaxError, ValueError, TypeError, KeyError, IndexError, AttributeError, StopIteration):
        return None

    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    scalar_tree = copy.deepcopy(tree)
    scalar_fn = next(n for n in scalar_tree.body if isinstance(n, ast.FunctionDef))
    scalar_name = kernel_name + '_nanochat_persistent_scale'
    scalar_fn.name = scalar_name
    scalar_fn.args = ast.parse(
        'def f(in_ptr2, _nanochat_scale_out, xnumel, XBLOCK: tl.constexpr): pass'
    ).body[0].args
    selected = {'tmp7', 'tmp8', *(f'tmp{i}' for i in range(9, 17))}
    scalar_fn.body = [n for n in scalar_fn.body if isinstance(n, ast.Assign)
                      and n.targets[0].id in selected]
    scalar_fn.body[1].value = split._expr('tmp7')  # Broadcast only after the FP32 cast.
    scalar_fn.body.append(ast.parse(
        'tl.store(_nanochat_scale_out + tl.full([1], 0, tl.int32), tl.reshape(tmp16, [1]))'
    ).body[0])
    scalar_dec = scalar_fn.decorator_list[0]
    scalar_kw = {k.arg: k.value for k in scalar_dec.keywords}
    scalar_tm = split._fields(scalar_kw['triton_meta'])
    scalar_tm['signature'] = split._expr(repr({
        'in_ptr2': '*fp32', '_nanochat_scale_out': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'
    }))
    scalar_tm['configs'] = split._expr('[{}]')
    scalar_im = split._fields(scalar_kw['inductor_meta'])
    scalar_im.update({k: split._expr(repr(v)) for k, v in {
        'kernel_name': scalar_name, 'mutated_arg_names': [], 'optimize_mem': False,
        'num_load': 1, 'num_reduction': 0,
    }.items()})
    for key in ('no_x_dim', 'tiling_scores', 'dynamic_scale_rblock', 'min_split_scan_rblock'):
        scalar_im.pop(key)
    scalar_dec.func = split._expr('triton_heuristics.pointwise')
    scalar_dec.keywords = [ast.keyword(arg=k, value=v) for k, v in {
        'size_hints': split._expr("{'x': 1}"), 'filename': split._expr('__file__'),
        'triton_meta': split._dictionary(scalar_tm),
        'inductor_meta': split._dictionary(scalar_im), 'min_elem_per_thread': ast.Constant(0)
    }.items()]

    # Keep the ten original runtime positions and every existing config index.
    fn.args.args.insert(10, ast.arg(arg='_nanochat_scale_input'))
    dec = fn.decorator_list[0]
    kw = {k.arg: k.value for k in dec.keywords}
    tm = split._fields(kw['triton_meta'])
    signature = split._fields(tm['signature'])
    signature = {**{k: v for k, v in signature.items() if k != 'XBLOCK'},
                 '_nanochat_scale_input': ast.Constant('*fp32'), 'XBLOCK': signature['XBLOCK']}
    tm['signature'] = split._dictionary(signature)
    kw['triton_meta'] = split._dictionary(tm)
    dec.keywords = [ast.keyword(arg=k, value=v) for k, v in kw.items()]
    # Replace one scalar load and its chain; retain all other statements, including
    # constexpr AnnAssigns, FP32 sum, sigmoid, barrier and the three writes.
    fn.body = [n for n in fn.body if not (isinstance(n, ast.Assign)
               and n.targets[0].id in {f'tmp{i}' for i in range(9, 16)})]
    for n in fn.body:
        if isinstance(n, ast.Assign) and n.targets[0].id == 'tmp7':
            n.value = split._expr('tl.load(_nanochat_scale_input + 0)')
        elif isinstance(n, ast.Assign) and n.targets[0].id == 'tmp16':
            n.value = split._expr('tmp8')
    identity = hashlib.sha256(Path(__file__).read_bytes() + Path(split.__file__).read_bytes()).hexdigest()
    prefix = f'# nanochat_persistent_scale_helpers={identity}\n'
    large = prefix + ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    scalar = prefix + ast.unparse(ast.fix_missing_locations(scalar_tree)) + '\n'
    return split.SplitSource(large, scalar, scalar_name, 3, 10)


def _runtime_reason(args, stream, kwargs):
    import torch
    if kwargs or len(args) != 10:
        return 'run_abi'
    if (type(args[8]) is not int or type(args[9]) is not int
            or args[8:] != (393216, 128)):
        return 'extents'
    source = args[3]
    tensors = [(source, torch.float32, 1), (args[0], torch.bfloat16, 393216),
               (args[6], torch.float8_e5m2, 50331648), (args[7], torch.float32, 50331648)]
    for tensor, dtype, numel in tensors:
        if (type(tensor) is not torch.Tensor or not tensor.is_cuda or tensor.layout != torch.strided
                or tensor.dtype != dtype or tensor.numel() != numel or not tensor.is_contiguous()
                or tensor.is_conj() or tensor.is_neg()):
            return 'tensor_layout'
        if tensor.device != source.device:
            return 'device'
    if (torch.cuda.current_device() != source.device.index
            or torch.cuda.current_stream(source.device).cuda_stream != stream):
        return 'stream'
    if torch.cuda.is_current_stream_capturing():
        return 'capture'
    lo = source.data_ptr()
    for tensor, _, numel in tensors[1:]:
        start = tensor.data_ptr()
        if lo < start + numel * tensor.element_size() and start < lo + 4:
            return 'source_output_alias'
    return None


_COUNTS = Counter()


class _GuardedKernel:
    def __init__(self, baseline, large, scalar, plan):
        self.baseline = baseline
        self.split = split._SplitKernel(large, scalar, plan)

    def run(self, *args, stream, **kwargs):
        reason = _runtime_reason(args, stream, kwargs)
        if reason is not None:
            _COUNTS['fallback_' + reason] += 1
            return self.baseline.run(*args, stream=stream, **kwargs)
        # No retry after either child has executed. Allocation ownership and
        # current-stream ordering are supplied by the existing validated wrapper.
        _COUNTS['fast'] += 1
        return self.split.run(*args, stream=stream)


def _combine(baseline, large, scalar, plan, future_types, make_future):
    def resolve(value):
        return value.result() if isinstance(value, future_types) else value
    def result():
        return _GuardedKernel(resolve(baseline), resolve(large), resolve(scalar), plan)
    return make_future(result) if any(isinstance(v, future_types)
                                     for v in (baseline, large, scalar)) else result()


def enable():
    from concurrent.futures import Future
    import torch.compiler.config as compiler_config
    from torch._inductor import config
    from torch._inductor.async_compile import AsyncCompile
    from torch._inductor.codecache import CodeCacheFuture, LambdaFuture
    from torch._inductor.virtualized import V

    def supported():
        if (config.cpp_wrapper or config.profile_bandwidth or V.aot_compilation
                or config.triton.cudagraphs or config.triton.multi_kernel):
            raise RuntimeError('persistent scale split requires ordinary uncaptured Python JIT')
    supported()
    original = AsyncCompile.triton
    if (getattr(original, '_nanochat_persistent_scale', False)
            or getattr(original, '_nanochat_split_scale', False)):
        raise RuntimeError('install persistent scale once, before pointwise split scale')
    identity = hashlib.sha256(Path(__file__).read_bytes() + Path(split.__file__).read_bytes()).hexdigest()
    compiler_config.cache_key_tag += ':nanochat-persistent-scale-' + identity

    @functools.wraps(original)
    def compile_triton(self, kernel_name, source_code, device_str='cuda'):
        supported()
        plan = split_source(source_code, kernel_name) if device_str == 'cuda' else None
        if plan is None:
            return original(self, kernel_name, source_code, device_str)
        _COUNTS['matched'] += 1
        print('NANOCHAT_OPT_PERSISTENT_SCALE_SPLIT: ' + json.dumps({
            'rank': os.environ.get('RANK'), 'kernel': kernel_name,
            'original': hashlib.sha256(source_code.encode()).hexdigest(),
            'large': hashlib.sha256(plan.large_source.encode()).hexdigest(),
            'scalar': hashlib.sha256(plan.scalar_source.encode()).hexdigest(), 'helpers': identity
        }, sort_keys=True), flush=True)
        baseline = original(self, kernel_name, source_code, device_str)
        large = original(self, kernel_name, plan.large_source, device_str)
        scalar = original(self, plan.scalar_name, plan.scalar_source, device_str)
        return _combine(baseline, large, scalar, plan, (Future, CodeCacheFuture), LambdaFuture)

    compile_triton._nanochat_persistent_scale = True
    AsyncCompile.triton = compile_triton
    def report():
        print('NANOCHAT_OPT_PERSISTENT_SCALE_COUNTERS: ' + json.dumps({
            'pid': os.getpid(), 'rank': os.environ.get('RANK'), **_COUNTS
        }, sort_keys=True), flush=True)
    atexit.register(report)

"""Pack independent row maxima; preserve the generated NaN-propagating max2."""
import ast
import copy
import functools
import hashlib
from pathlib import Path


# A conservative structural selector, not a replacement kernel. Only the row
# extent is normalized. Any different indexing or arithmetic uses stock code.
_BODY = """
xnumel = 0
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
tmp0 = r0_1
tmp1 = tl.full([1, 1], 64, tl.int64)
tmp2 = tmp0 >= tmp1
tmp3 = tl.load(in_ptr0 + ((-64) + r0_1 + 64*x0), tmp2, other=0.0).to(tl.float32)
tmp4 = 0.0
tmp5 = tl.where(tmp2, tmp3, tmp4)
tmp6 = tmp0 < tmp1
tmp7 = tl.load(in_ptr1 + (r0_1 + 64*x0), tmp6, other=0.0).to(tl.float32)
tmp8 = tl.where(tmp6, tmp7, tmp4)
tmp9 = tmp5 + tmp8
tmp10 = tmp9.to(tl.float32)
tmp11 = tl_math.abs(tmp10)
tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
tmp14 = triton_helpers.max2(tmp12, 1)[:, None].to(tl.float32)
tl.store(out_ptr0 + (x0), tmp14, None)
"""
_BODY_AST = ast.dump(ast.parse(_BODY))
_SIGNATURE = {'in_ptr0': '*bf16', 'in_ptr1': '*bf16',
              'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32',
              'XBLOCK': 'constexpr'}


def _mapping(node):
    if not isinstance(node, ast.Dict):
        raise ValueError('expected literal dictionary')
    keys = [ast.literal_eval(k) for k in node.keys]
    if any(type(k) is not str for k in keys) or len(set(keys)) != len(keys):
        raise ValueError('expected unique string keys')
    return dict(zip(keys, node.values))


def _metadata_ok(meta):
    return (meta.get('grid_type') == 'Grid1D'
            and meta.get('no_x_dim') in (None, False)
            and meta.get('num_load') == 2 and meta.get('num_reduction') == 1
            and meta.get('mutated_arg_names') == []
            and not meta.get('autotune_hints')
            and not meta.get('coordinate_descent_tuning')
            and not meta.get('tma_min_block_sizes')
            and not meta.get('profile_bandwidth'))


def rewrite(source, kernel_name):
    """Change only the decorator of the proven independent-row kernel."""
    if 'triton_heuristics.persistent_reduction' not in source:
        return source, False
    try:
        tree = ast.parse(source)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(funcs) != 1 or funcs[0].name != kernel_name:
            return source, False
        fn = funcs[0]
        if (len(fn.decorator_list) != 2
                or ast.unparse(fn.decorator_list[1]) != 'triton.jit'):
            return source, False
        dec = fn.decorator_list[0]
        if (not isinstance(dec, ast.Call) or dec.args
                or ast.unparse(dec.func) != 'triton_heuristics.persistent_reduction'):
            return source, False
        kw = {k.arg: k.value for k in dec.keywords}
        if len(kw) != len(dec.keywords) or set(kw) != {
            'size_hints', 'reduction_hint', 'filename', 'triton_meta', 'inductor_meta'
        }:
            return source, False
        if ast.unparse(kw['reduction_hint']) != 'ReductionHint.INNER':
            return source, False
        hints = ast.literal_eval(kw['size_hints'])
        if (set(hints) != {'x', 'r0_'} or hints['r0_'] != 128
                or type(hints['x']) is not int or hints['x'] < 8):
            return source, False
        meta = ast.literal_eval(kw['inductor_meta'])
        if not _metadata_ok(meta):
            return source, False
        tm = _mapping(kw['triton_meta'])
        if (set(tm) != {'signature', 'device', 'constants', 'configs'}
                or ast.literal_eval(tm['signature']) != _SIGNATURE
                or ast.literal_eval(tm['constants'])):
            return source, False
        device = tm['device']
        if (not isinstance(device, ast.Call) or device.args
                or ast.unparse(device.func) != 'DeviceProperties'):
            return source, False
        props = {k.arg: ast.literal_eval(k.value) for k in device.keywords}
        if props.get('type') != 'cuda' or props.get('cc') != 90 or props.get('warp_size') != 32:
            return source, False
        expected_args = ast.parse('def f(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr): pass').body[0].args
        if ast.dump(fn.args) != ast.dump(expected_args):
            return source, False
        body = copy.deepcopy(fn.body)
        if not (isinstance(body[0], ast.Assign) and len(body[0].targets) == 1
                and ast.unparse(body[0].targets[0]) == 'xnumel'):
            return source, False
        rows = ast.literal_eval(body[0].value)
        if type(rows) is not int or rows <= 0 or rows % 8 or rows > hints['x']:
            return source, False
        body[0].value = ast.Constant(0)
        if ast.dump(ast.Module(body=body, type_ignores=[])) != _BODY_AST:
            return source, False
        from torch._inductor import config
        if config.autotune_lookup_table:
            return source, False
        # Preserve every byte of the body and every original decorator argument.
        lines = source.splitlines(keepends=True)
        node = dec.func
        begin = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
        end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
        replacement = 'nanochat_max_rowpack.persistent_reduction'
        rewritten = source[:begin] + replacement + source[end:]
        version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        prefix = (f'# nanochat_max_rowpack_source={version}\n'
                  'from nanochat import max_rowpack as nanochat_max_rowpack\n')
        return prefix + rewritten, True
    except (SyntaxError, ValueError, TypeError, KeyError, IndexError):
        return source, False


def persistent_reduction(size_hints, reduction_hint=False, triton_meta=None,
                         filename=None, inductor_meta=None):
    from torch._inductor import config
    from torch._inductor.runtime import triton_heuristics as h
    from torch._inductor.runtime.hints import ReductionHint
    import triton

    meta = {} if inductor_meta is None else dict(inductor_meta)
    meta['reduction_hint'] = reduction_hint
    configs = h._persistent_reduction_configs(size_hints, reduction_hint, meta)
    baseline = triton.Config({'XBLOCK': 1}, num_warps=2, num_stages=1)
    baseline.dynamic_scale_rblock = True
    if (not _metadata_ok(meta) or config.autotune_lookup_table
            or reduction_hint != ReductionHint.INNER
            or len(configs) != 1 or configs[0].__dict__ != baseline.__dict__):
        print('NANOCHAT_OPT_MAX_ROWPACK: stock configuration fallback', flush=True)
        return h.persistent_reduction(size_hints, reduction_hint, triton_meta,
                                      filename, inductor_meta)
    packed = copy.deepcopy(configs[0])
    # Triton Config.__setstate__ omits dynamically attached fields on deepcopy.
    packed.__dict__ = copy.deepcopy(configs[0].__dict__)
    packed.kwargs['XBLOCK'] = 8
    packed.num_warps = 4
    print(f'NANOCHAT_OPT_MAX_ROWPACK: {meta.get("kernel_name")} XBLOCK=8 warps=4', flush=True)
    return h.cached_autotune(size_hints, [packed], triton_meta=triton_meta,
                            inductor_meta=meta, filename=filename,
                            heuristic_type=h.HeuristicType.PERSISTENT_REDUCTION)


def enable():
    import torch.compiler.config as compiler_config
    from torch._inductor import config
    from torch._inductor.async_compile import AsyncCompile
    from torch._inductor.virtualized import V

    def supported():
        if config.cpp_wrapper or config.profile_bandwidth or V.aot_compilation:
            raise RuntimeError('max row packing supports ordinary Python JIT only')

    supported()
    original = AsyncCompile.triton
    if getattr(original, '_nanochat_split_scale', False):
        raise RuntimeError('install max row packing before scale splitting')
    if getattr(original, '_nanochat_max_rowpack', False):
        raise RuntimeError('install max row packing only once')
    compiler_config.cache_key_tag += ':nanochat-max-rowpack-' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    @functools.wraps(original)
    def compile_triton(self, kernel_name, source_code, device_str='cuda'):
        supported()
        rewritten, matched = rewrite(source_code, kernel_name) if device_str == 'cuda' else (source_code, False)
        if matched:
            print(f'NANOCHAT_OPT_MAX_ROWPACK: rewrite {kernel_name}', flush=True)
        return original(self, kernel_name, rewritten, device_str)

    compile_triton._nanochat_max_rowpack = True
    AsyncCompile.triton = compile_triton

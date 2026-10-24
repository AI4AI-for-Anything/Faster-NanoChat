"""Bounded launch search for the unchanged accepted packed MLP kernel."""
import ast
import copy
import functools
import hashlib
import json
import os
from pathlib import Path

from nanochat import mlp_packed_layout as packed
from nanochat.scalar_scale_split import _fields

_PACKED_HASH = 'c08d198478dd0d8582575406fd1aa391574b941415043dc2614e50d5a9115a6f'
_HINTS = {'y': 32768, 'x': 8192}
_TILES = ((32, 32, 4), (64, 32, 4), (32, 64, 4),
          (64, 64, 4), (32, 32, 8), (64, 64, 8))


def _version():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _emit(event, **fields):
    print('NANOCHAT_OPT_MLP_PACKED_TILES: ' + json.dumps({
        'event': event, 'rank': os.environ.get('RANK'), 'pid': os.getpid(),
        **fields}, sort_keys=True), flush=True)


def _normalized(source, kernel_name):
    if (not isinstance(source, str) or not source.isascii()
            or any(c in source for c in '\r\v\f\x1c\x1d\x1e')
            or not source.startswith('# nanochat_mlp_packed_source=' + _PACKED_HASH + '\n')):
        return None
    try:
        tree = copy.deepcopy(ast.parse(source))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(functions) != 1 or functions[0].name != kernel_name:
            return None
        fn = functions[0]
        fn.name = '_kernel'
        dec = fn.decorator_list[0]
        kw = {k.arg: k.value for k in dec.keywords}
        if len(kw) != len(dec.keywords):
            return None
        tm = _fields(kw['triton_meta'])
        device = tm['device']
        props = {k.arg: ast.literal_eval(k.value) for k in device.keywords}
        if (len(props) != len(device.keywords) or type(props.get('index')) is not int
                or not 0 <= props['index'] < 8):
            return None
        next(k for k in device.keywords if k.arg == 'index').value = ast.Constant(0)
        im = _fields(kw['inductor_meta'])
        if (ast.literal_eval(im['kernel_name']) != kernel_name
                or not isinstance(ast.literal_eval(im['backend_hash']), str)):
            return None
        im['kernel_name'].value = '_kernel'
        im['backend_hash'].value = '_backend'
        return ast.dump(tree)
    except (SyntaxError, ValueError, TypeError, KeyError, IndexError,
            AttributeError, StopIteration):
        return None


def _oracle():
    source, matched = packed.rewrite(packed._EXPECTED, '_kernel')
    expected = _normalized(source, '_kernel')
    if not matched or expected is None:
        raise RuntimeError('Unable to construct packed MLP source contract')
    return expected


def rewrite(source, kernel_name, expected):
    if _normalized(source, kernel_name) != expected:
        return source, False
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    dec = fn.decorator_list[0]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:dec.func.lineno - 1])) + dec.func.col_offset
    end = sum(map(len, lines[:dec.func.end_lineno - 1])) + dec.func.end_col_offset
    last = dec.keywords[-1]
    insert = sum(map(len, lines[:last.end_lineno - 1])) + last.end_col_offset
    # Only the decorator changes. Preserve every byte of the executable body.
    digest = hashlib.sha256(source.encode()).hexdigest()
    rewritten = source[:insert] + ', source_hash=' + repr(digest) + source[insert:]
    rewritten = rewritten[:start] + 'nanochat_mlp_packed_tiles.pointwise' + rewritten[end:]
    try:
        ast.parse(rewritten)
    except SyntaxError:
        return source, False
    return ('# nanochat_mlp_packed_tiles_source=' + _version() + '\n'
            'from nanochat import mlp_packed_tiles as nanochat_mlp_packed_tiles\n'
            + rewritten), True


def _configs(size_hints, minimum):
    from torch._inductor.runtime import triton_heuristics as h
    choices = []
    for x, y, warps in _TILES:
        c = h.triton_config(size_hints, x, y, num_elements_per_warp=x*y//warps,
                            min_elem_per_thread=minimum)
        if (c.kwargs != {'XBLOCK': x, 'YBLOCK': y} or c.num_warps != warps
                or c.num_stages != 1 or c.num_ctas != 1 or c.pre_hook is not None
                or c.maxnreg is not None or c.ir_override is not None
                or 6144 % x or 32768 % y):
            raise ValueError('Installed helper changed requested MLP configuration')
        choices.append(c)
    return choices


def _configuration(c):
    return dict(kwargs=dict(c.kwargs), num_warps=c.num_warps,
                num_stages=c.num_stages, num_ctas=c.num_ctas)


class _SelectionObserver:
    """Picklable observation of the normal autotuner's selected launcher."""
    def __init__(self, original, identity):
        self.original = original
        self.identity = identity

    def __call__(self, config, time_taken_ns, **kwargs):
        result = (self.original(config, time_taken_ns, **kwargs)
                  if self.original is not None else None)
        # Called after selection, before its subsequent launch; no dispatch claim.
        _emit('selected', **self.identity, configuration=_configuration(config),
              triton_cache_hash=kwargs.get('triton_cache_hash'),
              precompile_and_autotune_time_ns=time_taken_ns)
        return result


def _metadata_ok(triton_meta, inductor_meta):
    try:
        fn = next(n for n in ast.parse(packed._EXPECTED).body if isinstance(n, ast.FunctionDef))
        kw = {k.arg: k.value for k in fn.decorator_list[0].keywords}
        expected = ast.literal_eval(kw['inductor_meta'])
        actual = copy.deepcopy(inductor_meta)
        if not isinstance(actual['kernel_name'], str) or not isinstance(actual['backend_hash'], str):
            return False
        actual['kernel_name'] = '_kernel'; actual['backend_hash'] = '_backend'
        if actual != expected:
            return False
        tm = _fields(kw['triton_meta'])
        if set(triton_meta) != set(tm):
            return False
        for key in ('signature', 'constants', 'configs'):
            if triton_meta[key] != ast.literal_eval(tm[key]):
                return False
        device = triton_meta['device']
        if type(device.index) is not int or not 0 <= device.index < 8:
            return False
        return all(getattr(device, k.arg) == ast.literal_eval(k.value)
                   for k in tm['device'].keywords if k.arg != 'index')
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def pointwise(size_hints, triton_meta, tile_hint=None, filename=None,
              min_elem_per_thread=0, inductor_meta=None, source_hash=None):
    from torch._inductor.runtime import triton_heuristics as h
    from torch._inductor.runtime.hints import TileHint

    def stock(reason):
        _emit('fallback', reason=reason)
        return h.pointwise(size_hints, triton_meta, tile_hint, filename,
                           min_elem_per_thread, inductor_meta)

    if (size_hints != _HINTS or tile_hint != TileHint.DEFAULT or min_elem_per_thread != 4
            or not filename or not _metadata_ok(triton_meta, inductor_meta)
            or not isinstance(source_hash, str) or len(source_hash) != 64
            or any(c not in '0123456789abcdef' for c in source_hash)):
        return stock('metadata')
    try:
        choices = _configs(size_hints, min_elem_per_thread)
    except (ValueError, AssertionError):
        return stock('configuration')
    identity = dict(kernel=inductor_meta['kernel_name'], source=source_hash,
                    helper=_version(), configurations=[_configuration(c) for c in choices])
    decorator = h.cached_autotune(size_hints, choices,
                                  triton_meta=copy.deepcopy(triton_meta),
                                  inductor_meta=copy.deepcopy(inductor_meta),
                                  filename=filename, heuristic_type=h.HeuristicType.POINTWISE)

    def decorate(fn):
        tuner = decorator(fn)
        _emit('constructed', **identity, cache_info=tuner.autotune_cache_info,
              remaining_configurations=[_configuration(c) for c in tuner.configs])
        tuner.save_cache_hook = _SelectionObserver(tuner.save_cache_hook, identity)
        return tuner
    return decorate


def enable():
    import torch.compiler.config as compiler_config
    from torch._inductor import config
    from torch._inductor.async_compile import AsyncCompile
    from torch._inductor.virtualized import V

    def supported():
        if (config.cpp_wrapper or config.profile_bandwidth or V.aot_compilation
                or config.triton.cudagraphs or config.triton.multi_kernel
                or config.autotune_lookup_table or config.coordinate_descent_tuning
                or config.max_autotune or config.max_autotune_pointwise
                or os.environ.get('TRITON_INTERPRET', '0') == '1'):
            raise RuntimeError('Packed MLP tile search requires ordinary Python JIT')
    supported()
    if hashlib.sha256(Path(packed.__file__).read_bytes()).hexdigest() != _PACKED_HASH:
        raise RuntimeError('Packed MLP tile search requires the reviewed predecessor')
    original = AsyncCompile.triton
    if any(getattr(original, flag, False) for flag in (
        '_nanochat_mlp_packed_tiles', '_nanochat_mlp_packed', '_nanochat_persistent_scale',
        '_nanochat_max_rowpack', '_nanochat_norm_rowpack', '_nanochat_split_scale')):
        raise RuntimeError('Install packed MLP tile search once, before the packed/split chain')
    expected = _oracle()
    version = _version()
    compiler_config.cache_key_tag += ':nanochat-mlp-packed-tiles-' + version + '-' + _PACKED_HASH

    @functools.wraps(original)
    def compile_triton(self, kernel_name, source_code, device_str='cuda'):
        supported()
        rewritten, matched = (rewrite(source_code, kernel_name, expected) if device_str == 'cuda'
                              else (source_code, False))
        if matched:
            _emit('rewrite', kernel=kernel_name, helper=version,
                  input=hashlib.sha256(source_code.encode()).hexdigest(),
                  output=hashlib.sha256(rewritten.encode()).hexdigest())
        elif device_str == 'cuda' and isinstance(source_code, str) and source_code.startswith('# nanochat_mlp_packed_source='):
            _emit('fallback', reason='source_contract', kernel=kernel_name)
        return original(self, kernel_name, rewritten, device_str)

    compile_triton._nanochat_mlp_packed_tiles = True
    AsyncCompile.triton = compile_triton

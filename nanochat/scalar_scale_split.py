"""Extract singleton FP8 scale calculations after graph fusion, on one stream.

Original arithmetic and autograd stay intact; frozen N1 validation is required.
"""

import ast
import copy
import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path


def _name(node):
    return ast.unparse(node)


def _numeric(node):
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, float)
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and _numeric(node.operand))


def _constant_rank(node, definitions):
    if _numeric(node):
        return 0
    if isinstance(node, ast.Name):
        definition = definitions.get(node.id)
        if definition is None:
            return None
        value = definition.value
        if _numeric(value):
            return 0
        if (isinstance(value, ast.Call) and _name(value.func) == "tl.full"
                and len(value.args) == 3 and not value.keywords
                and isinstance(value.args[0], (ast.List, ast.Tuple))
                and all(isinstance(x, ast.Constant) and type(x.value) is int and x.value == 1
                        for x in value.args[0].elts)
                and _numeric(value.args[1])
                and _name(value.args[2]) in {"tl.float64", "tl.float32", "tl.int32", "tl.int64"}):
            return len(value.args[0].elts)
    return None


def _cast_of(node, source, dtype):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == source
            and node.func.attr == "to" and len(node.args) == 1 and not node.keywords
            and _name(node.args[0]) == dtype)


def _delay_source(source, kernel_name):
    """Return unchanged bytes on a miss; never import or execute the source."""
    if "tl.float64" not in source or not any(x in source for x in ("tl.float8e5", "tl.float8e4nv")):
        return source, 0
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == kernel_name]
    if len(functions) != 1:
        return source, 0
    fn = functions[0]
    if not any(isinstance(d, ast.Call) and _name(d.func) == "triton_heuristics.pointwise"
               for d in fn.decorator_list):
        return source, 0
    if any(not isinstance(n, (ast.Assign, ast.Expr)) for n in fn.body):
        return source, 0
    if any(isinstance(n, ast.NamedExpr) for statement in fn.body for n in ast.walk(statement)):
        return source, 0

    definitions, uses, duplicates = {}, {}, set()
    for statement in fn.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                return source, 0
            name = statement.targets[0].id
            if name in definitions:
                duplicates.add(name)
            definitions[name] = statement
        for node in ast.walk(statement.value):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                uses.setdefault(node.id, []).append(statement)
    if duplicates:
        return source, 0

    arguments = {a.arg for a in fn.args.args}
    constexprs = {a.arg for a in fn.args.args
                  if a.annotation is not None and _name(a.annotation) == "tl.constexpr"}

    def owns_line(statement):
        return (statement.lineno == statement.end_lineno
                and all(other is statement or not (other.lineno <= statement.lineno <= other.end_lineno)
                        for other in fn.body))

    lines, edits, occupied = source.splitlines(keepends=True), {}, set()
    matches = 0
    for name, broadcast in definitions.items():
        call = broadcast.value
        if (not isinstance(call, ast.Call) or _name(call.func) != "tl.broadcast_to"
                or len(call.args) != 2 or call.keywords
                or not isinstance(call.args[0], ast.Name)
                or not isinstance(call.args[1], (ast.List, ast.Tuple))):
            continue
        rank = len(call.args[1].elts)
        if rank == 0 or not all(
                (isinstance(d, ast.Constant) and type(d.value) is int and d.value > 0)
                or (isinstance(d, ast.Name) and d.id in constexprs and d.id not in definitions)
                for d in call.args[1].elts):
            continue
        load = definitions.get(call.args[0].id)
        if (load is None or not isinstance(load.value, ast.Call)
                or _name(load.value.func) != "tl.load" or len(load.value.args) != 1
                or load.value.keywords):
            continue
        pointer = load.value.args[0]
        if not (isinstance(pointer, ast.BinOp) and isinstance(pointer.op, ast.Add)
                and isinstance(pointer.left, ast.Name) and pointer.left.id in arguments
                and pointer.left.id not in definitions
                and isinstance(pointer.right, ast.Constant) and type(pointer.right.value) is int):
            continue
        consumers = uses.get(name, [])
        if len(consumers) != 1 or not isinstance(consumers[0], ast.Assign):
            continue
        first = consumers[0]
        if not _cast_of(first.value, name, "tl.float64"):
            continue

        current, singleton_rank = first.targets[0].id, 0
        chain, terminal, has_division = {name, current}, None, False
        while True:
            consumers = uses.get(current, [])
            if len(consumers) != 1 or not isinstance(consumers[0], ast.Assign):
                break
            statement = consumers[0]
            if _cast_of(statement.value, current, "tl.float32"):
                terminal = statement
                break
            expression = statement.value
            operands = None
            if isinstance(expression, ast.BinOp) and isinstance(expression.op, (ast.Mult, ast.Div)):
                operands = (expression.left, expression.right)
                has_division |= isinstance(expression.op, ast.Div)
            elif (isinstance(expression, ast.Call)
                  and _name(expression.func) in {"triton_helpers.maximum", "triton_helpers.minimum"}
                  and len(expression.args) == 2 and not expression.keywords):
                operands = expression.args
            if operands is None:
                break
            other = [x for x in operands if not (isinstance(x, ast.Name) and x.id == current)]
            if len(other) != 1:
                break
            other_rank = _constant_rank(other[0], definitions)
            if other_rank is None or (singleton_rank and other_rank and singleton_rank != other_rank):
                break
            singleton_rank = max(singleton_rank, other_rank)
            current = statement.targets[0].id
            if current in chain:
                break
            chain.add(current)
        if (terminal is None or not has_division or singleton_rank not in (0, rank)
                or chain & occupied or not owns_line(broadcast) or not owns_line(terminal)):
            continue
        output = terminal.targets[0].id
        temporary = "_nanochat_singleton_" + output
        if temporary in definitions or temporary in arguments or temporary in uses:
            continue
        indent = " " * broadcast.col_offset
        edits[broadcast.lineno - 1] = indent + name + " = " + call.args[0].id + "\n"
        indent = " " * terminal.col_offset
        edits[terminal.lineno - 1] = (
            indent + temporary + " = " + _name(terminal.value) + "\n"
            + indent + output + " = tl.broadcast_to(" + temporary + ", "
            + _name(call.args[1]) + ")\n")
        occupied.update(chain | {output})
        matches += 1
    if not matches:
        return source, 0
    result = "".join(edits.get(i, line) for i, line in enumerate(lines))
    ast.parse(result)
    return result, matches


def _fields(node):
    if not isinstance(node, ast.Dict) or any(
            not isinstance(k, ast.Constant) or not isinstance(k.value, str)
            for k in node.keys):
        return None
    keys = [k.value for k in node.keys]
    if len(set(keys)) != len(keys):
        return None
    return dict(zip(keys, node.values))


def _dictionary(fields):
    return ast.Dict(keys=[ast.Constant(k) for k in fields], values=list(fields.values()))


def _expr(text):
    return ast.parse(text, mode="eval").body


@dataclass(frozen=True)
class SplitSource:
    large_source: str
    scalar_source: str
    scalar_name: str
    input_index: int
    runtime_args: int


def split_source(source, kernel_name):
    """Extract one proven singleton chain, or return None without executing source."""
    if "_nanochat_" in source:
        return None
    delayed, count = _delay_source(source, kernel_name)
    if count != 1:
        return None
    tree = ast.parse(delayed)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1:
        return None
    fn = functions[0]
    if (fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg
            or fn.args.defaults):
        return None
    args = fn.args.args
    names = [a.arg for a in args]
    runtime = [a for a in args if a.annotation is None]
    constexprs = args[len(runtime):]
    if args[:len(runtime)] != runtime or any(
            a.annotation is None or _name(a.annotation) != "tl.constexpr" for a in constexprs):
        return None
    pointwise = next(d for d in fn.decorator_list
                     if isinstance(d, ast.Call) and _name(d.func) == "triton_heuristics.pointwise")
    keywords = {k.arg: k.value for k in pointwise.keywords}
    if pointwise.args or len(keywords) != len(pointwise.keywords):
        return None
    meta = _fields(keywords.get("triton_meta"))
    imeta = _fields(keywords.get("inductor_meta"))
    if meta is None or imeta is None:
        return None
    signature = _fields(meta.get("signature"))
    if signature is None or list(signature) != names:
        return None
    if _fields(meta.get("constants")) != {}:
        return None
    if "extra_launcher_args" in imeta and ast.literal_eval(imeta["extra_launcher_args"]):
        return None
    configs = meta.get("configs")
    if not isinstance(configs, ast.List) or not configs.elts:
        return None
    for cfg in configs.elts:
        if not isinstance(cfg, ast.Dict):
            return None
        for key in cfg.keys:
            index = ast.literal_eval(key)
            if (not isinstance(index, tuple) or len(index) != 1
                    or type(index[0]) is not int or not 0 <= index[0] < len(runtime)):
                return None

    definitions = {n.targets[0].id: n for n in fn.body if isinstance(n, ast.Assign)}
    terminals = [n for name, n in definitions.items() if name.startswith("_nanochat_singleton_")]
    if len(terminals) != 1:
        return None
    terminal = terminals[0]
    selected = set()

    def dependencies(statement):
        name = statement.targets[0].id
        if name in selected:
            return
        selected.add(name)
        for node in ast.walk(statement.value):
            if isinstance(node, ast.Name) and node.id in definitions:
                dependencies(definitions[node.id])

    dependencies(terminal)
    body = [n for n in fn.body if isinstance(n, ast.Assign) and n.targets[0].id in selected]
    loads = [n for n in body if isinstance(n.value, ast.Call) and _name(n.value.func) == "tl.load"]
    if len(loads) != 1:
        return None
    load = loads[0]
    pointer = load.value.args[0].left.id
    if pointer not in names[:len(runtime)] or _name(signature[pointer]) != repr("*fp32"):
        return None
    # Moving the read ahead of the original kernel must not cross a memory write
    # or an opaque operation. These are the pure address/load operations emitted
    # before scalar loads in the supported Inductor pointwise path.
    pure = {"tl.load", "tl.full", "tl.program_id", "tl.arange", "tl.broadcast_to"}
    for statement in fn.body:
        if statement.lineno > load.lineno:
            break
        for call in (n for n in ast.walk(statement) if isinstance(n, ast.Call)):
            if _name(call.func) in pure:
                continue
            if isinstance(call.func, ast.Attribute) and call.func.attr == "to":
                continue
            return None
    if "mutated_arg_names" not in imeta:
        return None
    if pointer in ast.literal_eval(imeta["mutated_arg_names"]):
        return None

    scalar_name = kernel_name + "_nanochat_scale"
    out_name = "_nanochat_scale_out"
    in_name = "_nanochat_scale_input"
    scalar_fn = copy.deepcopy(fn)
    scalar_fn.name = scalar_name
    scalar_fn.args.args = [ast.arg(arg=pointer), ast.arg(arg=out_name), ast.arg(arg="xnumel"),
                           ast.arg(arg="XBLOCK", annotation=_expr("tl.constexpr"))]
    scalar_fn.body = copy.deepcopy(body)
    result = terminal.targets[0].id
    # A shape-only reshape gives the output pointer and value the same rank.
    scalar_fn.body.append(ast.parse(f"tl.store({out_name} + tl.full([1], 0, tl.int32), tl.reshape({result}, [1]))").body[0])
    scalar_meta = copy.deepcopy(meta)
    scalar_meta["signature"] = _expr(repr({pointer: "*fp32", out_name: "*fp32", "xnumel": "i32", "XBLOCK": "constexpr"}))
    scalar_meta["configs"] = _expr("[{}]")
    scalar_imeta = copy.deepcopy(imeta)
    scalar_imeta.update({"grid_type": ast.Constant("Grid1D"), "kernel_name": ast.Constant(scalar_name),
                         "mutated_arg_names": _expr("[]"), "num_load": ast.Constant(1),
                         "num_reduction": ast.Constant(0), "autotune_hints": _expr("set()")})
    scalar_imeta.pop("tiling_scores", None)
    scalar_keywords = copy.deepcopy(keywords)
    scalar_keywords.update({"size_hints": _expr("{'x': 1}"), "triton_meta": _dictionary(scalar_meta),
                            "inductor_meta": _dictionary(scalar_imeta), "min_elem_per_thread": ast.Constant(0)})
    scalar_keywords.pop("tile_hint", None)
    scalar_pointwise = next(d for d in scalar_fn.decorator_list
                            if isinstance(d, ast.Call) and _name(d.func) == "triton_heuristics.pointwise")
    scalar_pointwise.keywords = [ast.keyword(arg=k, value=v) for k, v in scalar_keywords.items()]
    scalar_tree = copy.deepcopy(tree)
    scalar_tree.body = [scalar_fn if isinstance(n, ast.FunctionDef) else n for n in scalar_tree.body]

    # Existing runtime argument positions and their config indices stay intact.
    fn.args.args.insert(len(runtime), ast.arg(arg=in_name))
    large_signature = {a.arg: signature[a.arg] for a in runtime}
    large_signature[in_name] = ast.Constant("*fp32")
    large_signature.update({a.arg: signature[a.arg] for a in constexprs})
    meta["signature"] = _dictionary(large_signature)
    keywords["triton_meta"] = _dictionary(meta)
    imeta["num_load"] = ast.Constant(ast.literal_eval(imeta["num_load"]) + 1)
    keywords["inductor_meta"] = _dictionary(imeta)
    pointwise.keywords = [ast.keyword(arg=k, value=v) for k, v in keywords.items()]
    terminal.value = _expr(f"tl.load({in_name} + (0))")
    large_source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    scalar_source = ast.unparse(ast.fix_missing_locations(scalar_tree)) + "\n"
    ast.parse(large_source)
    ast.parse(scalar_source)
    return SplitSource(large_source, scalar_source, scalar_name, names.index(pointer), len(runtime))


class _SplitKernel:
    """Only the ordinary Python wrapper's run ABI is supported."""

    def __init__(self, large, scalar, plan):
        self.large = large
        self.scalar = scalar
        self.plan = plan

    def run(self, *args, stream, **kwargs):
        import torch

        if kwargs or len(args) != self.plan.runtime_args:
            raise RuntimeError("split scale requires the ordinary Python JIT run ABI")
        source = args[self.plan.input_index]
        if (not source.is_cuda or source.dtype != torch.float32
                or torch.cuda.current_device() != source.device.index
                or torch.cuda.current_stream(source.device).cuda_stream != stream):
            raise RuntimeError("split scale requires FP32 input and matching current CUDA device/stream")
        # Private allocation owned until both launches have been enqueued on its
        # allocation stream. The ordinary caching allocator then preserves lifetime.
        scale = torch.empty((), device=source.device, dtype=torch.float32)
        self.scalar.run(source, scale, 1, stream=stream)
        return self.large.run(*args, scale, stream=stream)


def _combine(large, scalar, plan, future_types, make_future):
    def resolve(value):
        return value.result() if isinstance(value, future_types) else value

    if isinstance(large, future_types) or isinstance(scalar, future_types):
        return make_future(lambda: _SplitKernel(resolve(large), resolve(scalar), plan))
    return _SplitKernel(large, scalar, plan)


def enable():
    from concurrent.futures import Future
    import torch.compiler.config as compiler_config
    from torch._inductor import config
    from torch._inductor.async_compile import AsyncCompile
    from torch._inductor.codecache import CodeCacheFuture, LambdaFuture
    from torch._inductor.virtualized import V

    def supported():
        if config.cpp_wrapper or config.profile_bandwidth or V.aot_compilation:
            raise RuntimeError("split scale supports ordinary Python JIT wrappers only")

    supported()
    original = AsyncCompile.triton
    if getattr(original, "_nanochat_split_scale", False):
        raise RuntimeError("split scale must be installed only once")
    compiler_config.cache_key_tag += ":nanochat-split-scale-" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    @functools.wraps(original)
    def compile_triton(self, kernel_name, source_code, device_str="cuda"):
        supported()
        plan = split_source(source_code, kernel_name) if device_str == "cuda" else None
        if plan is None:
            return original(self, kernel_name, source_code, device_str)
        print(f"NANOCHAT_OPT_SPLIT_SCALE: {kernel_name} input={plan.input_index} "
              f"large={hashlib.sha256(plan.large_source.encode()).hexdigest()} "
              f"scalar={hashlib.sha256(plan.scalar_source.encode()).hexdigest()}", flush=True)
        large = original(self, kernel_name, plan.large_source, device_str)
        scalar = original(self, plan.scalar_name, plan.scalar_source, device_str)
        return _combine(large, scalar, plan, (Future, CodeCacheFuture), LambdaFuture)

    compile_triton._nanochat_split_scale = True
    AsyncCompile.triton = compile_triton


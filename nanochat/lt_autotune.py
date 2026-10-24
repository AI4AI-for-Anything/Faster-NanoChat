"""Guarded native cuBLASLt algorithm selection without changing the AOT graph."""
import atexit
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

_CFLAGS = ["-O3", "-std=c++17"]
# Only a C++ source is compiled, but with_cuda also prepares an unused NVCC
# rule. Explicit architecture prevents that helper from initializing CUDA.
_CUDA_CFLAGS = ["-gencode=arch=compute_90,code=sm_90"]


def build_spec():
    from torch.utils.cpp_extension import CUDA_HOME, get_cxx_compiler

    here = Path(__file__).resolve()
    cpp = here.with_suffix(".cpp")
    cublas = Path(torch.__file__).resolve().parents[1] / "nvidia" / "cublas"
    compiler = get_cxx_compiler()
    ninja = shutil.which("ninja") or "/usr/local/bin/ninja"
    if not os.access(ninja, os.X_OK):
        raise RuntimeError("An existing Ninja executable is required for this CPU build")
    info = {
        "python_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "cpp_sha256": hashlib.sha256(cpp.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cublas": importlib.metadata.version("nvidia-cublas-cu12"),
        "cxx_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "compiler": compiler,
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True),
        "ninja": str(Path(ninja).resolve()),
        "ninja_version": subprocess.check_output([ninja, "--version"], text=True),
        "cuda_home": str(CUDA_HOME),
        "cflags": _CFLAGS,
        "cuda_cflags": _CUDA_CFLAGS,
        "cublas_header_sha256": hashlib.sha256(
            (cublas / "include" / "cublasLt.h").read_bytes()).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return digest, info, cpp, cublas


def load_extension():
    from torch.utils.cpp_extension import load

    root = os.environ.get("NANOCHAT_OPT_LT_BUILD_DIR")
    if not root or not Path(root).is_absolute():
        raise RuntimeError("NANOCHAT_OPT_LT_BUILD_DIR must name an absolute build-cache directory")
    digest, info, cpp, cublas = build_spec()
    directory = Path(root).resolve() / digest
    directory.mkdir(parents=True, exist_ok=True)
    # Startup-only build, before compiler workers/training. Restore the process
    # path even when compilation fails; no package installation is required.
    old_path = os.environ.get("PATH")
    os.environ["PATH"] = str(Path(info["ninja"]).parent) + os.pathsep + (old_path or "")
    try:
        module = load(
            name="nanochat_lt_" + digest[:24],
            sources=[str(cpp)],
            extra_cflags=_CFLAGS,
            extra_cuda_cflags=_CUDA_CFLAGS,
            extra_include_paths=[str(cublas / "include")],
            extra_ldflags=["-L" + str(cublas / "lib"),
                           "-Wl,-rpath," + str(cublas / "lib"), "-lcublasLt"],
            with_cuda=True,
            build_directory=str(directory),
            verbose=False,
        )
    finally:
        if old_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old_path
    print("NANOCHAT_OPT_LT_SEARCH: build " + json.dumps(
        {"rank": os.environ.get("RANK"), "pid": os.getpid(),
         "digest": digest, "directory": str(directory), "versions": info}, sort_keys=True),
        flush=True)
    return module, digest


def matches(args, kwargs):
    if (len(args) != 4 or set(kwargs) != {"out_dtype", "use_fast_accum", "out"}
            or kwargs["out_dtype"] is not torch.bfloat16
            or kwargs["use_fast_accum"] is not False):
        return False
    tensors = (*args, kwargs["out"])
    if any(type(t) is not torch.Tensor or not t.is_cuda for t in tensors):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return False
    a, b, sa, sb, out = tensors
    if torch.is_inference(out) and not torch.is_inference_mode_enabled():
        return False
    return (a.dtype == torch.float8_e5m2 and b.dtype == torch.float8_e4m3fn
            and sa.dtype == sb.dtype == torch.float32 and out.dtype == torch.bfloat16
            and tuple(a.shape) == (1536, 32768) and tuple(b.shape) == (32768, 1536)
            and tuple(out.shape) == (1536, 1536)
            and sa.ndim == sb.ndim == 0
            and all(t.device == a.device for t in tensors))


def enable():
    import torch.compiler.config as compiler_config
    from torch._inductor import config
    from torch._inductor.select_algorithm import ExternKernelChoice, extern_kernels
    from torch._inductor.virtualized import V

    if config.cpp_wrapper or config.profile_bandwidth or V.aot_compilation:
        raise RuntimeError("native Lt search supports ordinary Python JIT only")
    original = extern_kernels._scaled_mm
    if getattr(original, "_nanochat_lt_search", False) or original is not torch._scaled_mm:
        raise RuntimeError("install native Lt search once over the original extern callable")
    native, digest = load_extension()
    compiler_config.cache_key_tag += ":nanochat-lt-search-" + digest
    counts = {"calls": 0, "unmatched": 0, "native_fallback": 0, "dispatched": 0}

    @functools.wraps(original)
    def scaled_mm(*args, **kwargs):
        counts["calls"] += 1
        if not matches(args, kwargs):
            counts["unmatched"] += 1
            return original(*args, **kwargs)
        if native.run(*args, kwargs["out"]):
            torch.autograd.graph.increment_version(kwargs["out"])
            counts["dispatched"] += 1
            return kwargs["out"]
        counts["native_fallback"] += 1
        return original(*args, **kwargs)

    def report():
        print("NANOCHAT_OPT_LT_SEARCH: counters " + json.dumps(
            {"rank": os.environ.get("RANK"), "pid": os.getpid(),
             "python": counts, "cpp": dict(native.counters())}, sort_keys=True), flush=True)

    scaled_mm._nanochat_lt_search = True
    # Extern-choice timing caches use callable identity independently of the
    # compiler cache tag; wraps() alone would retain the original builtin name.
    scaled_mm.__name__ = "_scaled_mm_nanochat_lt_" + digest
    extern_kernels._scaled_mm = scaled_mm
    ExternKernelChoice.hash_key.cache_clear()
    atexit.register(report)

"""O(log n) hull-based KV cache wrapper using a lazy pybind11 extension."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

import mlx.core as mx
import numpy as np

_MODULE_NAME = "_mlx_tvm_hull_ext"
_hull_ext = None


def _cache_root():
    base = os.environ.get("MLX_TRANSFORMER_VM_CACHE")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "mlx-transformer-vm"


def _source_hash():
    digest = hashlib.sha256()
    for name in ("hull_cache.py", "hull_ext.cpp", "hull2d_cht.h"):
        digest.update((Path(__file__).with_name(name)).read_bytes())
    digest.update(sys.version.encode())
    return digest.hexdigest()[:16]


def _extension_path():
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out_dir = _cache_root() / "extensions" / _MODULE_NAME / _source_hash()
    return out_dir / f"{_MODULE_NAME}{ext_suffix}"


def _compiler_flags():
    flags = shlex.split(sysconfig.get_config_var("CFLAGS") or "")
    return [flag for flag in flags if flag != "-Wstrict-prototypes"]


def _link_flags():
    ldflags = shlex.split(sysconfig.get_config_var("LDFLAGS") or "")
    if sys.platform == "darwin":
        return ["-bundle", "-undefined", "dynamic_lookup", *ldflags]
    return ["-shared", *(shlex.split(sysconfig.get_config_var("CCSHARED") or "")), *ldflags]


def _build_ext(target):
    import pybind11

    source = Path(__file__).with_name("hull_ext.cpp")
    include_dirs = [
        sysconfig.get_paths()["include"],
        pybind11.get_include(),
        pybind11.get_include(user=True),
        np.get_include(),
    ]
    compiler = shlex.split(sysconfig.get_config_var("CXX") or "c++")
    cmd = [
        *compiler,
        *_compiler_flags(),
        "-O3",
        "-std=c++17",
        *(f"-I{path}" for path in include_dirs),
        str(source),
        "-o",
        str(target.with_suffix(target.suffix + ".tmp")),
        *_link_flags(),
    ]

    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "failed to build hull extension:\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    target.with_suffix(target.suffix + ".tmp").replace(target)


def _load_ext():
    global _hull_ext
    if _hull_ext is not None:
        return _hull_ext

    target = _extension_path()
    if not target.exists():
        _build_ext(target)

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load hull extension from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _hull_ext = module
    return module


def has_hull_extension():
    """Return whether the hull extension can be built and loaded."""

    try:
        _load_ext()
    except Exception:
        return False
    return True


def _as_head_pairs(array, n_heads):
    values = np.asarray(array).reshape(-1)
    if values.shape[0] % n_heads != 0:
        raise ValueError("tensor width must be divisible by n_heads")
    head_dim = values.shape[0] // n_heads
    if head_dim != 2:
        raise ValueError(f"HullKVCache requires head_dim=2, got {head_dim}")
    return values.reshape(n_heads, 2)


class HullKVCache:
    """O(log n) hard-attention KV cache using a 2D convex hull extension."""

    def __init__(self, n_layers, n_heads):
        ext = _load_ext()
        self._cache = ext.HullKVCache(n_layers, n_heads)
        self._n_heads = n_heads
        self._seq = -1

    def clear(self):
        self._cache.clear()
        self._seq = -1

    def set_tiebreak(self, layer, head, latest):
        self._cache.set_tiebreak(layer, head, 1 if latest else 0)

    def layer_step(self, layer, keys, queries, values):
        self._seq += 1
        out_np = self._cache.layer_step(
            layer,
            _as_head_pairs(keys, self._n_heads),
            _as_head_pairs(queries, self._n_heads),
            _as_head_pairs(values, self._n_heads),
            self._seq,
        )
        return mx.array(out_np, dtype=values.dtype).reshape((-1,))

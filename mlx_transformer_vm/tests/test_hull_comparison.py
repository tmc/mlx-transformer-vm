"""Correctness and performance comparison across all cache implementations.

Four KV cache implementations exist, serving two different code paths:

Evaluator path (exact arithmetic, Python float64):
  - PythonHullKVCache: O(n log n), pure Python, no build deps
  - HullKVCache: O(n log n), C++ pybind11, ~9x faster than Python hull

Runner path (MLX float32, transformer inference):
  - OptimizedKVCache: pre-allocated tensors, vectorized hardmax
  - StandardKVCache: list-append + stack, per-head loop
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from mlx_transformer_vm.attention.hull_cache import HullKVCache, has_hull_extension
from mlx_transformer_vm.attention.hull_python import PythonHullKVCache
from mlx_transformer_vm.attention.optimized_cache import OptimizedKVCache
from mlx_transformer_vm.attention.standard_cache import StandardKVCache

N_LAYERS = 2
N_HEADS = 18
HEAD_DIM = 2
D_MODEL = N_HEADS * HEAD_DIM

skip_no_ext = pytest.mark.skipif(
    not has_hull_extension(),
    reason="C++ hull extension not available",
)


@pytest.fixture
def seed():
    np.random.seed(42)


def _random_vectors(n):
    """Return n tuples of (keys, queries, values) as mx.array."""
    vecs = []
    for _ in range(n):
        k = mx.array(np.random.randn(D_MODEL).astype(np.float32))
        q = mx.array(np.random.randn(D_MODEL).astype(np.float32))
        v = mx.array(np.random.randn(D_MODEL).astype(np.float32))
        vecs.append((k, q, v))
    return vecs


def _setup_tiebreak(cache, layer, latest_heads):
    for h in latest_heads:
        cache.set_tiebreak(layer, h, True)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@skip_no_ext
def test_hull_python_cpp_correctness(seed):
    """PythonHullKVCache and C++ HullKVCache produce identical outputs."""
    py_cache = PythonHullKVCache(N_LAYERS, N_HEADS)
    cpp_cache = HullKVCache(N_LAYERS, N_HEADS)

    latest_heads = [0, 3, 7, 12, 15]
    for layer in range(N_LAYERS):
        _setup_tiebreak(py_cache, layer, latest_heads)
        _setup_tiebreak(cpp_cache, layer, latest_heads)

    vecs = _random_vectors(200)
    for step, (k, q, v) in enumerate(vecs):
        for layer in range(N_LAYERS):
            out_py = np.array(py_cache.layer_step(layer, k, q, v))
            out_cpp = np.array(cpp_cache.layer_step(layer, k, q, v))
            assert np.allclose(out_py, out_cpp, atol=1e-6), (
                f"mismatch at step {step} layer {layer}: "
                f"max diff {np.max(np.abs(out_py - out_cpp)):.2e}"
            )


def test_optimized_standard_correctness(seed):
    """OptimizedKVCache and StandardKVCache produce identical outputs."""
    opt = OptimizedKVCache(N_LAYERS, N_HEADS)
    std = StandardKVCache(N_LAYERS, N_HEADS)

    latest_heads = [0, 3, 7, 12, 15]
    for layer in range(N_LAYERS):
        _setup_tiebreak(opt, layer, latest_heads)
        _setup_tiebreak(std, layer, latest_heads)

    vecs = _random_vectors(200)
    for step, (k, q, v) in enumerate(vecs):
        for layer in range(N_LAYERS):
            out_opt = np.array(opt.layer_step(layer, k, q, v))
            out_std = np.array(std.layer_step(layer, k, q, v))
            assert np.allclose(out_opt, out_std, atol=1e-6), (
                f"mismatch at step {step} layer {layer}: "
                f"max diff {np.max(np.abs(out_opt - out_std)):.2e}"
            )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def _bench_cache(cache, vecs, n_layers):
    """Run insert+query sequence, return elapsed seconds."""
    start = time.perf_counter()
    for k, q, v in vecs:
        for layer in range(n_layers):
            cache.layer_step(layer, k, q, v)
    return time.perf_counter() - start


def _make_cache(cls, n_layers, n_heads, latest_heads):
    cache = cls(n_layers, n_heads)
    for layer in range(n_layers):
        _setup_tiebreak(cache, layer, latest_heads)
    return cache


@skip_no_ext
def test_hull_python_cpp_performance(seed, capsys):
    """Benchmark Python hull vs C++ hull (evaluator path)."""
    seq_lengths = [100, 500, 1000, 5000]
    latest = [0, 7, 15]
    results = []

    for n in seq_lengths:
        np.random.seed(42)
        vecs = _random_vectors(n)

        py_cache = _make_cache(PythonHullKVCache, 1, N_HEADS, latest)
        t_py = _bench_cache(py_cache, vecs, 1)

        np.random.seed(42)
        vecs = _random_vectors(n)

        cpp_cache = _make_cache(HullKVCache, 1, N_HEADS, latest)
        t_cpp = _bench_cache(cpp_cache, vecs, 1)

        ratio = t_py / t_cpp if t_cpp > 0 else float("inf")
        results.append((n, t_py, t_cpp, ratio))

    with capsys.disabled():
        print("\n")
        print("Evaluator hull cache (float64, exact arithmetic):")
        print(f"{'Seq Len':>8}  {'Python (s)':>11}  {'C++ (s)':>11}  {'Py/C++':>7}")
        print(f"{'-------':>8}  {'-----------':>11}  {'-----------':>11}  {'------':>7}")
        for n, t_py, t_cpp, ratio in results:
            print(f"{n:>8}  {t_py:>11.4f}  {t_cpp:>11.4f}  {ratio:>6.1f}x")


def test_mlx_cache_performance(seed, capsys):
    """Benchmark OptimizedKVCache vs StandardKVCache (runner path)."""
    seq_lengths = [100, 500, 1000, 5000]
    latest = [0, 7, 15]
    results = []

    for n in seq_lengths:
        np.random.seed(42)
        vecs = _random_vectors(n)

        std = _make_cache(StandardKVCache, 1, N_HEADS, latest)
        t_std = _bench_cache(std, vecs, 1)

        np.random.seed(42)
        vecs = _random_vectors(n)

        opt = _make_cache(OptimizedKVCache, 1, N_HEADS, latest)
        t_opt = _bench_cache(opt, vecs, 1)

        speedup = t_std / t_opt if t_opt > 0 else float("inf")
        results.append((n, t_std, t_opt, speedup))

    with capsys.disabled():
        print("\n")
        print("Runner MLX cache (float32, transformer inference):")
        print(f"{'Seq Len':>8}  {'Standard (s)':>12}  {'Optimized (s)':>14}  {'Speedup':>8}")
        print(f"{'-------':>8}  {'------------':>12}  {'--------------':>14}  {'-------':>8}")
        for n, t_std, t_opt, speedup in results:
            print(f"{n:>8}  {t_std:>12.4f}  {t_opt:>14.4f}  {speedup:>7.1f}x")

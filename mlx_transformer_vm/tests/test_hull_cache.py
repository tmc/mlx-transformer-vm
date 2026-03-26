from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_transformer_vm.attention import HullKVCache, StandardKVCache, has_hull_extension
from mlx_transformer_vm.model import VanillaTransformer


def _require_hull_extension():
    if not has_hull_extension():
        pytest.skip("hull extension build unavailable")


def test_hull_cache_matches_standard_cache():
    _require_hull_extension()

    standard = StandardKVCache(n_layers=1, n_heads=2)
    hull = HullKVCache(n_layers=1, n_heads=2)
    standard.set_tiebreak(0, 0, True)
    hull.set_tiebreak(0, 0, True)

    first_keys = mx.array([1.0, 0.0, 1.0, 0.0], dtype=mx.float32)
    first_queries = mx.array([1.0, 0.0, 1.0, 0.0], dtype=mx.float32)
    first_values = mx.array([3.0, 1.0, 2.0, 4.0], dtype=mx.float32)
    second_keys = mx.array([1.0, 0.0, 1.0, 0.0], dtype=mx.float32)
    second_queries = mx.array([1.0, 0.0, 1.0, 0.0], dtype=mx.float32)
    second_values = mx.array([5.0, 7.0, 11.0, 13.0], dtype=mx.float32)

    standard.layer_step(0, first_keys, first_queries, first_values)
    hull.layer_step(0, first_keys, first_queries, first_values)

    want = np.array(standard.layer_step(0, second_keys, second_queries, second_values))
    got = np.array(hull.layer_step(0, second_keys, second_queries, second_values))

    assert np.allclose(got, want)


def test_transformer_generate_with_hull_cache_stops_on_stop_token():
    _require_hull_extension()

    model = VanillaTransformer(vocab=2, d_model=4, n_heads=2, n_layers=1, d_ffn=2, stop_token_id=0)
    model.tok.weight = mx.zeros((2, 4), dtype=mx.float32)
    model.attn[0].in_proj_weight = mx.zeros((12, 4), dtype=mx.float32)
    model.attn[0].out_proj.weight = mx.zeros((4, 4), dtype=mx.float32)
    model.ff_in[0].weight = mx.zeros((4, 4), dtype=mx.float32)
    model.ff_out[0].weight = mx.zeros((4, 2), dtype=mx.float32)
    model.head.weight = mx.zeros((2, 4), dtype=mx.float32)

    out = model.generate_with_cache(
        mx.array([[1]], dtype=mx.int32),
        max_new_tokens=3,
        cache_class=HullKVCache,
    )

    assert np.array_equal(np.array(out), np.array([[1, 0]], dtype=np.int32))

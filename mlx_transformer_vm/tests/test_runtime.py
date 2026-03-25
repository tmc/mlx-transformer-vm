from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from mlx_transformer_vm.attention import StandardKVCache
from mlx_transformer_vm.attention.standard_cache import exact_hardmax_attention
from mlx_transformer_vm.model import VanillaTransformer, add_position_encoding


def test_add_position_encoding_matches_upstream_formula():
    x = mx.zeros((4,), dtype=mx.float32)
    out = add_position_encoding(x, 3)

    assert np.isclose(np.array(out[0]), 3.0)
    assert np.isclose(np.array(out[1]), 1.0 / math.log(2) - 1.0 / math.log(5))
    assert np.isclose(np.array(out[2]), 9.0)


def test_exact_hardmax_attention_averages_ties():
    out = exact_hardmax_attention(
        mx.array([[1.0, 0.0]], dtype=mx.float32),
        mx.array(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ],
            dtype=mx.float32,
        ),
        mx.array(
            [
                [[3.0, 1.0]],
                [[5.0, 7.0]],
            ],
            dtype=mx.float32,
        ),
    )

    assert np.allclose(np.array(out), np.array([4.0, 4.0], dtype=np.float32))


def test_exact_hardmax_attention_latest_picks_latest_tie():
    out = exact_hardmax_attention(
        mx.array([[1.0, 0.0]], dtype=mx.float32),
        mx.array(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ],
            dtype=mx.float32,
        ),
        mx.array(
            [
                [[3.0, 1.0]],
                [[5.0, 7.0]],
            ],
            dtype=mx.float32,
        ),
        prefer_latest=True,
    )

    assert np.allclose(np.array(out), np.array([5.0, 7.0], dtype=np.float32))


def test_standard_cache_matches_manual_hardmax():
    cache = StandardKVCache(n_layers=1, n_heads=2)
    cache.layer_step(
        0,
        mx.array([1.0, 0.0, 0.0, 1.0], dtype=mx.float32),
        mx.array([1.0, 0.0, 0.0, 1.0], dtype=mx.float32),
        mx.array([3.0, 1.0, 2.0, 4.0], dtype=mx.float32),
    )
    out = cache.layer_step(
        0,
        mx.array([0.0, 1.0, 1.0, 0.0], dtype=mx.float32),
        mx.array([1.0, 0.0, 0.0, 1.0], dtype=mx.float32),
        mx.array([5.0, 7.0, 11.0, 13.0], dtype=mx.float32),
    )

    expected = np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float32)

    assert np.allclose(np.array(out), expected)


def test_standard_cache_set_tiebreak_picks_latest():
    cache = StandardKVCache(n_layers=1, n_heads=1)
    cache.set_tiebreak(0, 0, True)
    cache.layer_step(
        0,
        mx.array([1.0, 0.0], dtype=mx.float32),
        mx.array([1.0, 0.0], dtype=mx.float32),
        mx.array([3.0, 1.0], dtype=mx.float32),
    )
    out = cache.layer_step(
        0,
        mx.array([1.0, 0.0], dtype=mx.float32),
        mx.array([1.0, 0.0], dtype=mx.float32),
        mx.array([5.0, 7.0], dtype=mx.float32),
    )

    assert np.allclose(np.array(out), np.array([5.0, 7.0], dtype=np.float32))


def test_transformer_generate_with_cache_stops_on_stop_token():
    model = VanillaTransformer(vocab=2, d_model=4, n_heads=2, n_layers=1, d_ffn=2, stop_token_id=0)
    model.tok.weight = mx.zeros((2, 4), dtype=mx.float32)
    model.attn[0].in_proj_weight = mx.zeros((12, 4), dtype=mx.float32)
    model.attn[0].out_proj.weight = mx.zeros((4, 4), dtype=mx.float32)
    model.ff_in[0].weight = mx.zeros((4, 4), dtype=mx.float32)
    model.ff_out[0].weight = mx.zeros((4, 2), dtype=mx.float32)
    model.head.weight = mx.zeros((2, 4), dtype=mx.float32)

    out = model.generate_with_cache(mx.array([[1]], dtype=mx.int32), max_new_tokens=3)

    assert np.array_equal(np.array(out), np.array([[1, 0]], dtype=np.int32))

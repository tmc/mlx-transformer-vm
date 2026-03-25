"""O(n) reference KV cache for MLX attention."""

from __future__ import annotations

import mlx.core as mx


class StandardKVCache:
    """Standard softmax KV cache with the same API as the upstream cache."""

    def __init__(self, n_layers, n_heads):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self._keys = [[] for _ in range(n_layers)]
        self._vals = [[] for _ in range(n_layers)]

    def clear(self):
        self._keys = [[] for _ in range(self.n_layers)]
        self._vals = [[] for _ in range(self.n_layers)]

    def layer_step(self, layer, keys, queries, values):
        self._keys[layer].append(mx.array(keys))
        self._vals[layer].append(mx.array(values))

        head_dim = keys.shape[0] // self.n_heads
        keys_stacked = mx.stack(self._keys[layer], axis=0).reshape((-1, self.n_heads, head_dim))
        vals_stacked = mx.stack(self._vals[layer], axis=0).reshape((-1, self.n_heads, head_dim))
        query_heads = queries.reshape((self.n_heads, head_dim))

        scores = mx.einsum("thi,hi->th", keys_stacked, query_heads)
        weights = mx.softmax(scores, axis=0)
        out = mx.einsum("th,thi->hi", weights, vals_stacked)
        return out.reshape((-1,))

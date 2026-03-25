"""O(n) exact hard-attention cache for MLX."""

from __future__ import annotations

import mlx.core as mx


def exact_hardmax_attention(queries, keys, values, prefer_latest=False):
    """Return exact hard-attention output with explicit tie handling.

    Args:
        queries: ``(H, Dh)`` query vectors.
        keys: ``(T, H, Dh)`` cached key vectors.
        values: ``(T, H, Dh)`` cached value vectors.
        prefer_latest: when True, choose the latest tied key per head.
            Otherwise average all exact ties.
    """

    scores = mx.einsum("thi,hi->th", keys, queries)
    max_scores = mx.max(scores, axis=0, keepdims=True)
    is_max = scores == max_scores

    if prefer_latest:
        indices = mx.arange(scores.shape[0], dtype=mx.int32).reshape((-1, 1))
        masked = mx.where(is_max, indices, mx.full(indices.shape, -1, dtype=mx.int32))
        latest = mx.max(masked, axis=0, keepdims=True)
        weights = (indices == latest).astype(values.dtype)
    else:
        weights = is_max.astype(values.dtype)
        weights = weights / mx.sum(weights, axis=0, keepdims=True)

    out = mx.einsum("th,thi->hi", weights, values)
    return out.reshape((-1,))


class StandardKVCache:
    """Exact hard-attention KV cache with upstream-compatible tie handling."""

    def __init__(self, n_layers, n_heads):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self._keys = [[] for _ in range(n_layers)]
        self._vals = [[] for _ in range(n_layers)]
        self._prefer_latest = [[False for _ in range(n_heads)] for _ in range(n_layers)]

    def clear(self):
        self._keys = [[] for _ in range(self.n_layers)]
        self._vals = [[] for _ in range(self.n_layers)]

    def set_tiebreak(self, layer, head, latest):
        """Set the explicit tie mode for one head.

        The graph geometry already biases ``latest`` lookups. This is kept for
        API compatibility and as a fallback for any residual exact ties.
        """

        self._prefer_latest[layer][head] = bool(latest)

    def layer_step(self, layer, keys, queries, values):
        self._keys[layer].append(mx.array(keys))
        self._vals[layer].append(mx.array(values))

        head_dim = keys.shape[0] // self.n_heads
        keys_stacked = mx.stack(self._keys[layer], axis=0).reshape((-1, self.n_heads, head_dim))
        vals_stacked = mx.stack(self._vals[layer], axis=0).reshape((-1, self.n_heads, head_dim))
        query_heads = queries.reshape((self.n_heads, head_dim))
        outputs = []
        for head in range(self.n_heads):
            outputs.append(
                exact_hardmax_attention(
                    query_heads[head : head + 1],
                    keys_stacked[:, head : head + 1, :],
                    vals_stacked[:, head : head + 1, :],
                    prefer_latest=self._prefer_latest[layer][head],
                )
            )
        return mx.concatenate(outputs, axis=0)

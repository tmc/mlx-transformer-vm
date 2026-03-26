"""Pre-allocated exact hard-attention KV cache for MLX."""

from __future__ import annotations

import mlx.core as mx


class OptimizedKVCache:
    """Exact hard-attention KV cache with pre-allocated tensors and vectorized hardmax."""

    def __init__(self, n_layers, n_heads, head_dim=2, max_seq=65536):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq = max_seq
        self._keys = [mx.zeros((max_seq, n_heads, head_dim)) for _ in range(n_layers)]
        self._vals = [mx.zeros((max_seq, n_heads, head_dim)) for _ in range(n_layers)]
        self._pos = [0] * n_layers
        # Per-head tiebreak: True means LATEST, False means AVERAGE.
        # Stored as (n_heads,) bool mask per layer for vectorized mx.where.
        self._latest_mask = [mx.zeros((n_heads,), dtype=mx.bool_) for _ in range(n_layers)]

    def clear(self):
        for layer in range(self.n_layers):
            self._keys[layer] = mx.zeros((self.max_seq, self.n_heads, self.head_dim))
            self._vals[layer] = mx.zeros((self.max_seq, self.n_heads, self.head_dim))
            self._pos[layer] = 0

    def set_tiebreak(self, layer, head, latest):
        """Set the explicit tie mode for one head.

        Pre-builds an mx.array mask so layer_step avoids per-head branching.
        """
        mask = list(self._latest_mask[layer].tolist())
        mask[head] = bool(latest)
        self._latest_mask[layer] = mx.array(mask, dtype=mx.bool_)

    def layer_step(self, layer, keys, queries, values):
        pos = self._pos[layer]
        k_vec = mx.array(keys).reshape((self.n_heads, self.head_dim))
        v_vec = mx.array(values).reshape((self.n_heads, self.head_dim))
        self._keys[layer][pos] = k_vec
        self._vals[layer][pos] = v_vec
        self._pos[layer] = pos + 1

        seq_len = pos + 1
        k_slice = self._keys[layer][:seq_len]   # (T, H, Dh)
        v_slice = self._vals[layer][:seq_len]    # (T, H, Dh)
        q_heads = mx.array(queries).reshape((self.n_heads, self.head_dim))  # (H, Dh)

        # Vectorized scores across all heads at once.
        scores = mx.einsum("thi,hi->th", k_slice, q_heads)  # (T, H)

        max_scores = mx.max(scores, axis=0, keepdims=True)  # (1, H)
        is_max = scores == max_scores  # (T, H)

        # AVERAGE path: weight all tied positions equally.
        avg_weights = is_max.astype(mx.float32)
        avg_weights = avg_weights / mx.sum(avg_weights, axis=0, keepdims=True)

        # LATEST path: pick the last tied position per head.
        indices = mx.arange(seq_len, dtype=mx.int32).reshape((-1, 1))  # (T, 1)
        masked_idx = mx.where(is_max, indices, mx.full(indices.shape, -1, dtype=mx.int32))
        latest_idx = mx.max(masked_idx, axis=0, keepdims=True)  # (1, H)
        latest_weights = (indices == latest_idx).astype(mx.float32)  # (T, H)

        # Select per-head via pre-built mask: (H,) broadcast over (T, H).
        mask = self._latest_mask[layer]  # (H,)
        weights = mx.where(mask, latest_weights, avg_weights)  # (T, H)

        out = mx.einsum("th,thi->hi", weights, v_slice)  # (H, Dh)
        return out.reshape((-1,))

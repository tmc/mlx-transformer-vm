"""VanillaTransformer with ReGLU FFN and standard-cache generation in MLX."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_transformer_vm.attention import StandardKVCache

DEFAULT_DTYPE = mx.float32


def add_position_encoding(x, pos):
    """Add the upstream deterministic position features in place."""

    x[0] += pos
    x[1] += 1.0 / math.log(2) - 1.0 / math.log(pos + 2)
    x[2] += pos * pos
    return x


class TokenEmbedding(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.weight = mx.zeros((vocab, d_model), dtype=DEFAULT_DTYPE)

    def __call__(self, idx):
        return self.weight[idx]


class WeightOnlyLinear(nn.Module):
    def __init__(self, out_dims, in_dims):
        super().__init__()
        self.weight = mx.zeros((out_dims, in_dims), dtype=DEFAULT_DTYPE)

    def __call__(self, x):
        return self.weight @ x


class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_proj_weight = mx.zeros((3 * d_model, d_model), dtype=DEFAULT_DTYPE)
        self.out_proj = WeightOnlyLinear(d_model, d_model)


class VanillaTransformer(nn.Module):
    def __init__(self, vocab, d_model=36, n_heads=18, n_layers=7, d_ffn=36, stop_token_id=0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.stop_token_id = stop_token_id
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ffn = d_ffn

        self.tok = TokenEmbedding(vocab, d_model)
        self.attn = [AttentionLayer(d_model) for _ in range(n_layers)]
        self.ff_in = [WeightOnlyLinear(2 * d_ffn, d_model) for _ in range(n_layers)]
        self.ff_out = [WeightOnlyLinear(d_model, d_ffn) for _ in range(n_layers)]
        self.head = WeightOnlyLinear(vocab, d_model)

    def step_logits(self, token_id, pos, cache):
        x = mx.array(self.tok.weight[token_id])
        add_position_encoding(x, pos)

        for layer_idx in range(self.n_layers):
            qkv = self.attn[layer_idx].in_proj_weight @ x
            q, k, v = mx.split(qkv, 3)
            attn_out = cache.layer_step(layer_idx, k, q, v)
            x = x + self.attn[layer_idx].out_proj(attn_out)

            gate_and_value = self.ff_in[layer_idx](x)
            gate, value = mx.split(gate_and_value, 2)
            x = x + self.ff_out[layer_idx](nn.relu(gate) * value)

        return self.head(x)

    def generate_with_cache(self, idx, max_new_tokens=5000, cache_class=StandardKVCache):
        cache = cache_class(self.n_layers, self.n_heads)
        if hasattr(self, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
            for layer_idx in range(self.n_layers):
                for head_idx in range(self.n_heads):
                    if self.head_tiebreak[layer_idx][head_idx]:
                        cache.set_tiebreak(layer_idx, head_idx, True)

        idx_list = np.array(idx).reshape(-1).tolist()
        limit = len(idx_list) + max_new_tokens

        for pos in range(limit):
            token_id = idx_list[pos]
            logits = self.step_logits(token_id, pos, cache)
            if pos + 1 == len(idx_list):
                next_id = int(mx.argmax(logits).item())
                idx_list.append(next_id)
                if next_id == self.stop_token_id:
                    break

        return mx.array([idx_list], dtype=mx.int32)

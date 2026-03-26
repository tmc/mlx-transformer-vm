#!/usr/bin/env python3
"""Evaluate token programs through the computation graph with exact arithmetic."""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np

from mlx_transformer_vm.graph.core import (
    CumSumDimension,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
)

logger = logging.getLogger(__name__)


def _build_default_graph():
    from mlx_transformer_vm.graph.core import _all_dims, _all_lookups
    from mlx_transformer_vm.wasm.interpreter import build

    input_tokens, output_tokens = build()
    return input_tokens, output_tokens, list(_all_dims), list(_all_lookups)


_default_graph = None


def _get_default_graph():
    global _default_graph
    if _default_graph is None:
        _default_graph = _build_default_graph()
    return _default_graph


def _load_hull():
    try:
        from mlx_transformer_vm.attention.hull_cache import _load_ext

        return _load_ext()
    except Exception as e:
        logger.warning("could not load hull extension: %s", e)
        return None


class HullAttention:
    """Per-lookup O(log n) attention using pybind11 convex hull."""

    def __init__(self, lookup, ext):
        self.lookup = lookup
        nv = len(lookup.value_exprs)
        self.num_value_pairs = (nv + 1) // 2
        self.nv = nv
        self.cache = ext.HullKVCache(1, self.num_value_pairs)
        self.seq = -1

    def clear(self):
        self.cache.clear()
        self.seq = -1

    def insert_and_query(self, vals, seq):
        lu = self.lookup
        kx = lu.key_exprs_2d[0].evaluate(vals)
        ky = lu.key_exprs_2d[1].evaluate(vals)
        qx = lu.query_exprs_2d[0].evaluate(vals)
        qy = lu.query_exprs_2d[1].evaluate(vals)

        nvp = self.num_value_pairs
        raw_vals = [v.evaluate(vals) for v in lu.value_exprs]
        while len(raw_vals) < nvp * 2:
            raw_vals.append(0.0)

        keys = np.zeros((nvp, 2))
        values = np.zeros((nvp, 2))
        queries = np.zeros((nvp, 2))
        for p in range(nvp):
            keys[p, 0] = kx
            keys[p, 1] = ky
            values[p, 0] = raw_vals[p * 2]
            values[p, 1] = raw_vals[p * 2 + 1]
            queries[p, 0] = qx
            queries[p, 1] = qy

        self.seq += 1
        out = self.cache.layer_step(0, keys, queries, values, self.seq)
        out_flat = out.reshape(-1).tolist()
        return out_flat[: self.nv]


class PythonHullAttention:
    """Per-lookup O(log n) attention using pure Python convex hull."""

    def __init__(self, lookup):
        from mlx_transformer_vm.attention.hull_python import HardAttentionHead, AVERAGE, LATEST

        self.lookup = lookup
        self.nv = len(lookup.value_exprs)
        self.num_value_pairs = (self.nv + 1) // 2
        self._heads = [HardAttentionHead() for _ in range(self.num_value_pairs)]
        self._tb = LATEST if lookup.tie_break == "latest" else AVERAGE
        self.seq = -1

    def clear(self):
        for h in self._heads:
            h.clear()
        self.seq = -1

    def insert_and_query(self, vals, seq):
        lu = self.lookup
        kx = lu.key_exprs_2d[0].evaluate(vals)
        ky = lu.key_exprs_2d[1].evaluate(vals)
        qx = lu.query_exprs_2d[0].evaluate(vals)
        qy = lu.query_exprs_2d[1].evaluate(vals)

        nvp = self.num_value_pairs
        raw_vals = [v.evaluate(vals) for v in lu.value_exprs]
        while len(raw_vals) < nvp * 2:
            raw_vals.append(0.0)

        self.seq += 1
        results = []
        for p in range(nvp):
            h = self._heads[p]
            h.insert(kx, ky, raw_vals[p * 2], raw_vals[p * 2 + 1], self.seq)
            o0, o1 = h.query(qx, qy, self._tb)
            results.extend([o0, o1])
        return results[: self.nv]


class BruteAttention:
    """Per-lookup exact hard-attention cache."""

    def __init__(self, lookup):
        self.lookup = lookup
        self.entries = []

    def clear(self):
        self.entries.clear()

    def insert_and_query(self, values, seq):
        lookup = self.lookup
        key_x = lookup.key_exprs_2d[0].evaluate(values)
        key_y = lookup.key_exprs_2d[1].evaluate(values)
        raw_values = [expr.evaluate(values) for expr in lookup.value_exprs]
        self.entries.append((seq, key_x, key_y, raw_values))

        query_x = lookup.query_exprs_2d[0].evaluate(values)
        query_y = lookup.query_exprs_2d[1].evaluate(values)

        best_score = -1e300
        for _stored_seq, entry_key_x, entry_key_y, _entry_values in self.entries:
            score = query_x * entry_key_x + query_y * entry_key_y
            if score > best_score + 1e-9:
                best_score = score

        if lookup.tie_break == "average":
            total = [0.0] * len(lookup.value_exprs)
            count = 0
            for _stored_seq, entry_key_x, entry_key_y, entry_values in self.entries:
                score = query_x * entry_key_x + query_y * entry_key_y
                if abs(score - best_score) <= 1e-9:
                    for index, value in enumerate(entry_values):
                        total[index] += value
                    count += 1
            return [value / count for value in total] if count else total

        best_seq = -1
        best_values = None
        for stored_seq, entry_key_x, entry_key_y, entry_values in self.entries:
            score = query_x * entry_key_x + query_y * entry_key_y
            if abs(score - best_score) <= 1e-9 and stored_seq > best_seq:
                best_seq = stored_seq
                best_values = entry_values
        return list(best_values)


class Runtime:
    """Exact graph evaluator for a :class:`ProgramGraph`."""

    def __init__(self, use_hull=True, force_python=False, program_graph=None):
        self.use_hull = use_hull
        self.use_python_hull = False
        self.hull_ext = None
        if use_hull:
            if not force_python:
                self.hull_ext = _load_hull()
            if self.hull_ext is not None:
                logger.debug("using C++ hull extension")
            else:
                try:
                    from mlx_transformer_vm.attention.hull_python import HardAttentionHead  # noqa: F401

                    self.use_python_hull = True
                    logger.debug("using pure Python hull")
                except ImportError:
                    logger.warning("hull unavailable; falling back to brute-force")
                    self.use_hull = False

        if program_graph is not None:
            self.input_tokens = program_graph.input_tokens
            self.output_tokens = program_graph.output_tokens
            self.all_dims = program_graph.all_dims
            self.all_lookups = program_graph.all_lookups
            self._one = program_graph.one
            self._position = program_graph.position
            self._position_sq = program_graph.position_sq
            self._inv_log_pos = program_graph.inv_log_pos
        else:
            input_tokens, output_tokens, all_dims, all_lookups = _get_default_graph()
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.all_dims = all_dims
            self.all_lookups = all_lookups
            from mlx_transformer_vm.graph.core import inv_log_pos, one, position, position_sq

            self._one = one
            self._position = position
            self._position_sq = position_sq
            self._inv_log_pos = inv_log_pos

        self.reset()

    def reset(self):
        self.pos = 0
        self.cumsum_accum = {}
        for dimension in self.all_dims:
            if isinstance(dimension, CumSumDimension):
                self.cumsum_accum[dimension] = 0.0

        self.attention = {}
        for lookup in self.all_lookups:
            if self.use_python_hull:
                self.attention[lookup.id] = PythonHullAttention(lookup)
            elif self.use_hull:
                self.attention[lookup.id] = HullAttention(lookup, self.hull_ext)
            else:
                self.attention[lookup.id] = BruteAttention(lookup)

    def step(self, token_name):
        values = {}

        embedding = self.input_tokens.get(token_name)
        if embedding is None:
            raise ValueError(f"unknown token: {token_name}")
        for dimension, coefficient in embedding.terms.items():
            values[dimension] = coefficient
        values[self._position] = float(self.pos)
        values[self._position_sq] = float(self.pos) ** 2
        values[self._inv_log_pos] = 1.0 / math.log(2) - 1.0 / math.log(self.pos + 2)

        processed_lookups = {}

        for dimension in self.all_dims:
            if dimension in values:
                continue

            if isinstance(dimension, CumSumDimension):
                self.cumsum_accum[dimension] += dimension.value_expr.evaluate(values)
                values[dimension] = self.cumsum_accum[dimension]

            elif isinstance(dimension, ReGLUDimension):
                a_value = dimension.a_expr.evaluate(values)
                b_value = dimension.b_expr.evaluate(values)
                values[dimension] = a_value * max(0.0, b_value)

            elif isinstance(dimension, PersistDimension):
                values[dimension] = dimension.expr.evaluate(values)

            elif isinstance(dimension, LookUpDimension):
                lookup = dimension.lookup
                if lookup.id not in processed_lookups:
                    processed_lookups[lookup.id] = self.attention[lookup.id].insert_and_query(
                        values, self.pos
                    )
                values[dimension] = processed_lookups[lookup.id][dimension.value_index]

        self.pos += 1
        return values

    def predict_next(self, values):
        best_score = -1e300
        best_name = None
        for token_name, score_expr in self.output_tokens.items():
            score = score_expr.evaluate(values)
            if score > best_score:
                best_score = score
                best_name = token_name
        return best_name


def generate_trace(tokens, runtime=None, max_steps=50000):
    """Return the full predicted token trace for ``tokens``."""

    if runtime is None:
        runtime = Runtime()

    program_end_idx = None
    for index, token in enumerate(tokens):
        if token == "}":
            program_end_idx = index
            break
    if program_end_idx is None:
        raise ValueError("no closing '}' found in token stream")

    for index in range(program_end_idx + 1):
        values = runtime.step(tokens[index])

    input_end_idx = program_end_idx
    if program_end_idx + 1 < len(tokens):
        for index in range(program_end_idx + 1, len(tokens)):
            values = runtime.step(tokens[index])
            input_end_idx = index

    predicted = list(tokens[: input_end_idx + 1])
    for _step in range(max_steps):
        next_token = runtime.predict_next(values)
        predicted.append(next_token)
        if next_token == "halt":
            break
        values = runtime.step(next_token)

    return predicted


def run_program(program_file, ref_file=None, use_hull=True, force_python=False, verbose=False):
    """Run a compiled token program and optionally compare with a reference trace."""

    with open(program_file) as handle:
        tokens = handle.read().split()

    ref_tokens = None
    if ref_file and os.path.exists(ref_file):
        with open(ref_file) as handle:
            ref_tokens = handle.read().split()

    predicted = generate_trace(
        tokens, runtime=Runtime(use_hull=use_hull, force_python=force_python)
    )
    if verbose:
        logger.info("Tokens: %s", " ".join(predicted))

    if ref_tokens is None:
        return True
    if predicted == ref_tokens:
        return True

    for index in range(max(len(predicted), len(ref_tokens))):
        predicted_token = predicted[index] if index < len(predicted) else "<END>"
        ref_token = ref_tokens[index] if index < len(ref_tokens) else "<END>"
        if predicted_token != ref_token:
            logger.warning(
                "mismatch at position %d: predicted=%s expected=%s",
                index,
                predicted_token,
                ref_token,
            )
            break
    return False


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Evaluate token programs through the computation graph."
    )
    parser.add_argument("files", nargs="+", help="Program .txt files to evaluate")
    parser.add_argument("--nohull", action="store_true", help="Use brute-force O(n) attention")
    parser.add_argument(
        "--python",
        action="store_true",
        help="Force pure Python hull (skip C++ extension even if available)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print the full token trace")
    args = parser.parse_args()

    any_failed = False
    for program_file in args.files:
        ref_file = str(Path(program_file).with_name(Path(program_file).stem + "_ref.txt"))
        has_ref = os.path.exists(ref_file)
        ok = run_program(
            program_file,
            ref_file if has_ref else None,
            use_hull=not args.nohull,
            force_python=args.python,
            verbose=args.verbose,
        )
        logger.info("%s: %s", program_file, "PASS" if ok else "FAIL")
        any_failed = any_failed or not ok
    if any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

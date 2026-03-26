"""Analytical weight construction for the MLX transformer runtime."""

from __future__ import annotations

import logging
from collections import defaultdict

import mlx.core as mx
import numpy as np
import yaml
from mlx.utils import tree_unflatten

from mlx_transformer_vm.graph.core import (
    Expression,
    InputDimension,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
    _all_dims,
)
from mlx_transformer_vm.model.transformer import DEFAULT_DTYPE, VanillaTransformer
from mlx_transformer_vm.scheduler import milp_schedule
from mlx_transformer_vm.wasm.interpreter import WASMMachine

logger = logging.getLogger(__name__)


def _load_plan(path, all_dims):
    with open(path) as handle:
        plan = yaml.safe_load(handle)

    name_to_dim = {dimension.name: dimension for dimension in all_dims}

    def resolve(name):
        dimension = name_to_dim.get(name)
        if dimension is None:
            raise KeyError(f"plan references unknown dim {name!r}")
        return dimension

    std_layers = []
    alive_after = {}

    for entry in plan["layers"]:
        layer = entry["layer"]

        seen_lookups = {}
        for dimension_name in entry["attention"]:
            dimension = resolve(dimension_name)
            if not isinstance(dimension, LookUpDimension):
                raise TypeError(f"{dimension_name!r} is not a lookup dimension")
            lookup = dimension.lookup
            if lookup.id not in seen_lookups:
                seen_lookups[lookup.id] = lookup

        persist1 = [resolve(name) for name in entry["persist1"]]
        ffn = [resolve(name) for name in entry["ffn"]]
        persist2 = [resolve(name) for name in entry["persist2"]]
        std_layers.append((list(seen_lookups.values()), persist1, ffn, persist2))

        if "after_persist1" in entry:
            alive_after[4 * layer + 1] = frozenset(
                resolve(name) for name in entry["after_persist1"]["dims"]
            )
        if "after_persist2" in entry:
            alive_after[4 * layer + 3] = frozenset(
                resolve(name) for name in entry["after_persist2"]["dims"]
            )

    return plan["summary"]["layers"], std_layers, alive_after


def _expr_dims(output_tokens):
    dims = set()
    for expr in output_tokens.values():
        if isinstance(expr, Expression):
            dims.update(expr.terms)
    return dims


def _build_default_graph():
    pg = WASMMachine().build()
    return pg.input_tokens, pg.output_tokens, pg, list(_all_dims)


def build_model(
    use_erase=True,
    _shared=None,
    program_graph=None,
    plan_path=None,
    max_layers=None,
    no_reuse=False,
    max_ffn=None,
):
    """Build an MLX transformer analytically from a computation graph."""

    if _shared is not None:
        (
            all_dims,
            input_tokens,
            output_tokens,
            n_layers,
            std_layers,
            alive_after,
            slot_of,
            reused_at,
            erased_at,
            d_model,
            internal_reglus,
            internal_lookups,
            internal_persists,
            internal_dims,
            one_expr,
            pos_expr,
            erase_q2d,
            erase_k2d,
            expr_to_vector,
            max_heads,
            n_heads,
            d_ffn,
            max_ffn_used,
            all_tokens,
            tok_to_idx_map,
            vocab_size,
            input_dims,
        ) = _shared
    else:
        if program_graph is not None:
            pg = program_graph
            input_tokens = pg.input_tokens
            output_tokens = pg.output_tokens
            all_dims = pg.all_dims
            one_dim = pg.one
            pos_dim = pg.position
            ilp_dim = pg.inv_log_pos
            psq_dim = pg.position_sq
        else:
            input_tokens, output_tokens, pg, all_dims = _build_default_graph()
            one_dim = pg.one
            pos_dim = pg.position
            ilp_dim = pg.inv_log_pos
            psq_dim = pg.position_sq

        if plan_path:
            n_layers, std_layers, alive_after = _load_plan(plan_path, all_dims)
        else:
            schedule = milp_schedule(
                input_tokens,
                output_tokens,
                max_layers=max_layers,
                max_ffn=max_ffn,
                program_graph=program_graph or pg,
                log=logger.info,
            )
            std_layers = schedule["std_layers"]
            n_layers = schedule["num_layers"]
            alive_after = schedule.get("alive_after", {})

        input_dims = [dimension for dimension in all_dims if isinstance(dimension, InputDimension)]
        fixed = {pos_dim: 0, ilp_dim: 1, psq_dim: 2}
        protected_slots = set(fixed.values())
        slot_of = dict(fixed)
        next_slot = 3
        for dimension in input_dims:
            if dimension not in slot_of:
                slot_of[dimension] = next_slot
                next_slot += 1

        free_slots = []
        pending_free = []
        reused_at = {}
        erased_at = {}

        current = frozenset(input_dims)
        for layer in range(n_layers):
            for half in (1, 3):
                boundary = 4 * layer + half
                nxt = alive_after.get(boundary, current)
                dying = current - nxt
                born = nxt - current

                if not no_reuse:
                    free_slots.extend(pending_free)
                    free_slots.sort()
                    pending_free = []

                erased = set()
                if not no_reuse:
                    for dimension in dying:
                        slot = slot_of[dimension]
                        if slot in protected_slots:
                            continue
                        pending_free.append(slot)
                        erased.add(slot)

                reused = set()
                for dimension in sorted(born, key=lambda item: item.id):
                    if free_slots:
                        slot = free_slots.pop(0)
                        slot_of[dimension] = slot
                        reused.add(slot)
                    else:
                        slot_of[dimension] = next_slot
                        next_slot += 1

                reused_at[(layer, half)] = reused
                erased_at[(layer, half)] = erased
                current = nxt

        output_dims_need = set()
        for expr in output_tokens.values():
            if not isinstance(expr, Expression):
                continue
            for dimension in expr.terms:
                if dimension not in slot_of and dimension not in input_dims:
                    output_dims_need.add(dimension)

        for dimension in sorted(output_dims_need, key=lambda item: item.id):
            if free_slots:
                slot_of[dimension] = free_slots.pop(0)
            else:
                slot_of[dimension] = next_slot
                next_slot += 1

        d_model = next_slot
        d_model += d_model % 2

        all_ever_alive = set()
        for dims in alive_after.values():
            all_ever_alive.update(dims)
        output_dims_set = _expr_dims(output_tokens)
        non_internal = all_ever_alive | set(input_dims) | output_dims_set
        internal_reglus = {
            dimension
            for dimension in all_dims
            if isinstance(dimension, ReGLUDimension) and dimension not in non_internal
        }
        internal_lookups = {
            dimension
            for dimension in all_dims
            if isinstance(dimension, LookUpDimension) and dimension not in non_internal
        }
        internal_persists = {
            dimension
            for dimension in all_dims
            if isinstance(dimension, PersistDimension) and dimension not in non_internal
        }
        internal_dims = internal_reglus | internal_lookups | internal_persists

        one_expr = Expression({one_dim: 1})
        pos_expr = Expression({pos_dim: 1})
        erase_q2d = [pos_expr, one_expr]
        erase_k2d = [pos_expr * 2, one_expr]

        def expr_to_vector(expr):
            vec = np.zeros(d_model, dtype=np.float32)
            for dimension, coeff in expr.terms.items():
                if dimension in internal_dims:
                    continue
                if dimension in slot_of:
                    vec[slot_of[dimension]] += coeff
            return vec

        max_heads = 0
        max_ffn_used = 1

        current = frozenset(input_dims)
        for layer in range(n_layers):
            attn, persist1, ffn, persist2 = std_layers[layer]

            boundary1 = 4 * layer + 1
            nxt1 = alive_after.get(boundary1, current)
            erased1 = erased_at[(layer, 1)]

            lookup_dims = set()
            for lookup in attn:
                lookup_dims.update(lookup.dims)
            n_lookup_heads = sum((len(lookup.value_exprs) + 1) // 2 for lookup in attn)

            passthrough_sources = set(erased1)
            for persist_dim in persist1:
                for dimension in persist_dim.expr.terms:
                    if dimension not in slot_of or dimension in internal_dims:
                        continue
                    if not (isinstance(dimension, LookUpDimension) and dimension in lookup_dims):
                        passthrough_sources.add(slot_of[dimension])

            max_heads = max(max_heads, n_lookup_heads + (len(passthrough_sources) + 1) // 2)
            current = nxt1

            boundary2 = 4 * layer + 3
            nxt2 = alive_after.get(boundary2, current)
            erased2 = erased_at[(layer, 3)]

            ffn_dims = set(ffn)
            passthrough_ffn = set(erased2)
            for persist_dim in persist2:
                for dimension in persist_dim.expr.terms:
                    if dimension not in slot_of or dimension in internal_dims:
                        continue
                    if dimension not in ffn_dims:
                        passthrough_ffn.add(slot_of[dimension])

            max_ffn_used = max(max_ffn_used, len(ffn) + len(passthrough_ffn))
            current = nxt2

        if 2 * max_heads > d_model:
            d_model = 2 * max_heads
            d_model += d_model % 2
        n_heads = d_model // 2
        d_ffn = max_ffn_used

        all_tokens = sorted(set(input_tokens) | set(output_tokens))
        tok_to_idx_map = {token: index for index, token in enumerate(all_tokens)}
        vocab_size = len(all_tokens)

    model = VanillaTransformer(
        vocab=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        stop_token_id=tok_to_idx_map.get("halt", 0),
    )

    weights = []

    def set_weight(path, value):
        weights.append((path, mx.array(value, dtype=DEFAULT_DTYPE)))

    embedding = np.zeros((vocab_size, d_model), dtype=np.float32)
    for token_name, expr in input_tokens.items():
        row = expr_to_vector(expr)
        row[0] = 0.0
        row[1] = 0.0
        row[2] = 0.0
        embedding[tok_to_idx_map[token_name]] = row
    set_weight("tok.weight", embedding)

    head = np.zeros((vocab_size, d_model), dtype=np.float32)
    for token_name, expr in output_tokens.items():
        head[tok_to_idx_map[token_name]] = expr_to_vector(expr)
    set_weight("head.weight", head)

    current = frozenset(input_dims)
    all_tiebreak = []
    lookup_heads_total = 0
    passthrough_heads_total = 0
    passthrough_ffn_total = 0

    # The MLX runtime uses exact hard attention, so the upstream softmax
    # temperature scaling is unnecessary and would overflow float32.
    query_scale = 1.0

    for layer_idx in range(n_layers):
        attn, persist1, ffn, persist2 = std_layers[layer_idx]

        in_proj = np.zeros((3 * d_model, d_model), dtype=np.float32)
        out_proj = np.zeros((d_model, d_model), dtype=np.float32)
        ff_in = np.zeros((2 * d_ffn, d_model), dtype=np.float32)
        ff_out = np.zeros((d_model, d_ffn), dtype=np.float32)

        boundary1 = 4 * layer_idx + 1
        nxt1 = alive_after.get(boundary1, current)
        head_idx = 0
        lookup_dim_to_head = {}
        layer_tiebreak = []

        for lookup in attn:
            n_values = len(lookup.value_exprs)
            for pair_idx in range((n_values + 1) // 2):
                head = head_idx
                head_idx += 1
                lookup_heads_total += 1
                layer_tiebreak.append(lookup.tie_break == "latest")

                in_proj[head * 2] = expr_to_vector(lookup.query_exprs_2d[0]) * query_scale
                in_proj[head * 2 + 1] = expr_to_vector(lookup.query_exprs_2d[1]) * query_scale
                in_proj[d_model + head * 2] = expr_to_vector(lookup.key_exprs_2d[0])
                in_proj[d_model + head * 2 + 1] = expr_to_vector(lookup.key_exprs_2d[1])
                in_proj[2 * d_model + head * 2] = expr_to_vector(lookup.value_exprs[pair_idx * 2])
                if pair_idx * 2 + 1 < n_values:
                    in_proj[2 * d_model + head * 2 + 1] = expr_to_vector(
                        lookup.value_exprs[pair_idx * 2 + 1]
                    )

                dimension0 = lookup.dims[pair_idx * 2]
                lookup_dim_to_head[dimension0] = (head, 0)
                if dimension0 not in internal_lookups:
                    out_proj[slot_of[dimension0], head * 2] = 1.0

                if pair_idx * 2 + 1 < n_values:
                    dimension1 = lookup.dims[pair_idx * 2 + 1]
                    lookup_dim_to_head[dimension1] = (head, 1)
                    if dimension1 not in internal_lookups:
                        out_proj[slot_of[dimension1], head * 2 + 1] = 1.0

        passthrough = defaultdict(lambda: defaultdict(float))
        for persist_dim in persist1:
            if persist_dim not in slot_of:
                continue
            for dimension, coeff in persist_dim.expr.terms.items():
                if dimension in lookup_dim_to_head:
                    head, component = lookup_dim_to_head[dimension]
                    out_proj[slot_of[persist_dim], head * 2 + component] += coeff
                elif dimension in slot_of:
                    passthrough[slot_of[dimension]][slot_of[persist_dim]] += coeff

        if use_erase:
            for slot in erased_at[(layer_idx, 1)]:
                passthrough[slot][slot] -= 1.0

        passthrough_items = list(passthrough.items())
        for pair_idx in range(0, len(passthrough_items), 2):
            head = head_idx
            head_idx += 1
            in_proj[head * 2] = expr_to_vector(erase_q2d[0]) * query_scale
            in_proj[head * 2 + 1] = expr_to_vector(erase_q2d[1]) * query_scale
            in_proj[d_model + head * 2] = expr_to_vector(erase_k2d[0])
            in_proj[d_model + head * 2 + 1] = expr_to_vector(erase_k2d[1])

            src0, dsts0 = passthrough_items[pair_idx]
            in_proj[2 * d_model + head * 2, src0] = 1.0
            for dst, coeff in dsts0.items():
                out_proj[dst, head * 2] += coeff

            if pair_idx + 1 < len(passthrough_items):
                src1, dsts1 = passthrough_items[pair_idx + 1]
                in_proj[2 * d_model + head * 2 + 1, src1] = 1.0
                for dst, coeff in dsts1.items():
                    out_proj[dst, head * 2 + 1] += coeff

        passthrough_heads_total += len(passthrough_items)
        while len(layer_tiebreak) < head_idx:
            layer_tiebreak.append(False)
        if head_idx > n_heads:
            raise ValueError(f"layer {layer_idx} needs {head_idx} heads, model only has {n_heads}")
        while len(layer_tiebreak) < n_heads:
            layer_tiebreak.append(False)
        all_tiebreak.append(layer_tiebreak)
        current = nxt1

        boundary2 = 4 * layer_idx + 3
        nxt2 = alive_after.get(boundary2, current)
        reglu_to_gate = {}
        gate_idx = 0

        for reglu_dim in ffn:
            ff_in[gate_idx] = expr_to_vector(reglu_dim.b_expr)
            ff_in[d_ffn + gate_idx] = expr_to_vector(reglu_dim.a_expr)
            reglu_to_gate[reglu_dim] = gate_idx
            if reglu_dim not in internal_reglus:
                ff_out[slot_of[reglu_dim], gate_idx] = 1.0
            gate_idx += 1

        passthrough_ffn = defaultdict(lambda: defaultdict(float))
        for persist_dim in persist2:
            if persist_dim not in slot_of:
                continue
            for dimension, coeff in persist_dim.expr.terms.items():
                if dimension in reglu_to_gate:
                    ff_out[slot_of[persist_dim], reglu_to_gate[dimension]] += coeff
                elif dimension in slot_of:
                    passthrough_ffn[slot_of[dimension]][slot_of[persist_dim]] += coeff

        if use_erase:
            for slot in erased_at[(layer_idx, 3)]:
                passthrough_ffn[slot][slot] -= 1.0

        for src, dsts in passthrough_ffn.items():
            ff_in[gate_idx] = expr_to_vector(one_expr)
            ff_in[d_ffn + gate_idx, src] = 1.0
            for dst, coeff in dsts.items():
                ff_out[dst, gate_idx] += coeff
            gate_idx += 1
            passthrough_ffn_total += 1

        if gate_idx > d_ffn:
            raise ValueError(f"layer {layer_idx} needs {gate_idx} FFN neurons, model only has {d_ffn}")

        current = nxt2

        set_weight(f"attn.{layer_idx}.in_proj_weight", in_proj)
        set_weight(f"attn.{layer_idx}.out_proj.weight", out_proj)
        set_weight(f"ff_in.{layer_idx}.weight", ff_in)
        set_weight(f"ff_out.{layer_idx}.weight", ff_out)

    model.update(tree_unflatten(weights))
    model.attn_erase = [sorted(erased_at[(layer_idx, 1)]) for layer_idx in range(n_layers)]
    model.ffn_erase = [sorted(erased_at[(layer_idx, 3)]) for layer_idx in range(n_layers)]
    model.head_tiebreak = all_tiebreak

    n_persist = sum(len(persist1) + len(persist2) for _, persist1, _, persist2 in std_layers)
    logger.info(
        "built model d_model=%d n_layers=%d n_heads=%d d_ffn=%d erase=%s",
        d_model,
        n_layers,
        n_heads,
        d_ffn,
        use_erase,
    )
    logger.info(
        "vocab=%d hulls=%d persist=%d pt_h=%d pt_g=%d",
        vocab_size,
        lookup_heads_total,
        n_persist,
        passthrough_heads_total,
        passthrough_ffn_total,
    )

    shared = (
        all_dims,
        input_tokens,
        output_tokens,
        n_layers,
        std_layers,
        alive_after,
        slot_of,
        reused_at,
        erased_at,
        d_model,
        internal_reglus,
        internal_lookups,
        internal_persists,
        internal_dims,
        one_expr,
        pos_expr,
        erase_q2d,
        erase_k2d,
        expr_to_vector,
        max_heads,
        n_heads,
        d_ffn,
        max_ffn_used,
        all_tokens,
        tok_to_idx_map,
        vocab_size,
        input_dims,
    )
    return model, all_tokens, tok_to_idx_map, shared

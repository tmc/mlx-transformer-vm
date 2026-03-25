"""Deterministic scheduler compatible with the upstream layer layout.

This is a conservative first pass: it preserves the 4-phase layer structure
and dependency ordering, but it does not attempt MILP optimization.
"""

from __future__ import annotations

import heapq
from collections import defaultdict

from mlx_transformer_vm.graph.core import (
    Expression,
    InputDimension,
    LookUp,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
    _all_dims,
    _all_lookups,
)
from mlx_transformer_vm.graph.core import (
    inv_log_pos as _default_inv_log_pos,
)
from mlx_transformer_vm.graph.core import (
    position as _default_position,
)
from mlx_transformer_vm.graph.core import (
    position_sq as _default_position_sq,
)


def _build_graph(all_dims=None, all_lookups=None, ilp=None):
    if all_dims is None:
        all_dims = _all_dims
    if all_lookups is None:
        all_lookups = _all_lookups
    if ilp is None:
        ilp = _default_inv_log_pos

    inputs = [dimension for dimension in all_dims if isinstance(dimension, InputDimension)]
    reglus = [dimension for dimension in all_dims if isinstance(dimension, ReGLUDimension)]
    persists = [dimension for dimension in all_dims if isinstance(dimension, PersistDimension)]
    lookups = list(all_lookups)
    ops = reglus + persists + lookups

    produced = {}
    for reglu in reglus:
        produced[reglu] = {reglu}
    for persist in persists:
        produced[persist] = {persist}
    for lookup in lookups:
        produced[lookup] = set(lookup.dims)

    def expr_deps(expr):
        return set(expr.terms.keys()) if isinstance(expr, Expression) else set()

    deps_cache = {}
    for reglu in reglus:
        deps_cache[reglu] = expr_deps(reglu.a_expr) | expr_deps(reglu.b_expr)
    for persist in persists:
        deps_cache[persist] = expr_deps(persist.expr)
    for lookup in lookups:
        deps = set()
        for expr in lookup.query_exprs_2d + lookup.key_exprs_2d + lookup.value_exprs:
            deps |= expr_deps(expr)
        deps.add(ilp)
        deps_cache[lookup] = deps

    dim_to_op = {}
    for op in ops:
        for dimension in produced[op]:
            dim_to_op[dimension] = op

    op_deps = defaultdict(set)
    children = defaultdict(set)
    consumers = defaultdict(set)
    for op in ops:
        for dimension in deps_cache[op]:
            consumers[dimension].add(op)
            if dimension in dim_to_op and dim_to_op[dimension] != op:
                predecessor = dim_to_op[dimension]
                op_deps[op].add(predecessor)
                children[predecessor].add(op)

    avg_lookups = {lookup for lookup in lookups if lookup.tie_break == "average"}
    tight_to = defaultdict(set)
    for op in reglus + persists:
        for dimension in deps_cache[op]:
            if isinstance(dimension, LookUpDimension) and dimension in dim_to_op:
                lookup = dim_to_op[dimension]
                if lookup in avg_lookups:
                    tight_to[op].add(lookup)

    return dict(
        ops=ops,
        reglus=reglus,
        persists=persists,
        lookups=lookups,
        inputs=inputs,
        produced=produced,
        deps_cache=deps_cache,
        dim_to_op=dim_to_op,
        op_deps=op_deps,
        children=children,
        consumers=consumers,
        tight_to=dict(tight_to),
    )


def _all_result_dims(graph):
    dims = list(graph["inputs"])
    dim_set = set(dims)
    for op in graph["ops"]:
        for dimension in graph["produced"][op]:
            if dimension not in dim_set:
                dim_set.add(dimension)
                dims.append(dimension)
    return dims


def _type_priority(op):
    if isinstance(op, LookUp):
        return 0
    if isinstance(op, PersistDimension):
        return 1
    return 2


def _aligned_phase(base_phase, phase_mod):
    phase = max(base_phase, 0)
    return phase + ((phase_mod - phase % 4) % 4)


def _candidate_persist_phase(base_phase, forced_layer=None):
    candidates = [_aligned_phase(base_phase, 1), _aligned_phase(base_phase, 3)]
    if forced_layer is not None:
        adjusted = []
        for candidate in candidates:
            layer = candidate // 4
            if layer < forced_layer:
                candidate = 4 * forced_layer + (candidate % 4)
                if candidate < base_phase:
                    candidate += 4
            adjusted.append(candidate)
        candidates = adjusted
    return min(candidates)


def _earliest_phase(op, phase_assign, graph, max_ffn=None, layer_ffn_counts=None):
    deps = graph["op_deps"].get(op, set())
    base_phase = max((phase_assign[pred] for pred in deps), default=-1) + 1
    forced_layer = None
    if op in graph["tight_to"]:
        forced_layer = max(phase_assign[lookup] // 4 for lookup in graph["tight_to"][op])

    if isinstance(op, LookUp):
        phase = _aligned_phase(base_phase, 0)
    elif isinstance(op, ReGLUDimension):
        phase = _aligned_phase(base_phase, 2)
        if forced_layer is not None and phase // 4 < forced_layer:
            phase = max(phase, 4 * forced_layer + 2)
        if max_ffn is not None and layer_ffn_counts is not None:
            while layer_ffn_counts.get(phase // 4, 0) >= max_ffn:
                phase = 4 * (phase // 4 + 1) + 2
    else:
        phase = _candidate_persist_phase(base_phase, forced_layer=forced_layer)
    return phase


def _schedule_asap(graph, max_ffn=None):
    phase_assign = {}
    layer_ffn_counts = defaultdict(int)
    indegree = {op: len(graph["op_deps"].get(op, set())) for op in graph["ops"]}

    ready = []
    seq = 0
    for op in graph["ops"]:
        if indegree[op] == 0:
            earliest = _earliest_phase(op, phase_assign, graph, max_ffn, layer_ffn_counts)
            heapq.heappush(ready, (earliest, _type_priority(op), getattr(op, "id", seq), seq, op))
            seq += 1

    scheduled = 0
    while ready:
        _earliest, _prio, _opid, _seq, op = heapq.heappop(ready)
        phase = _earliest_phase(op, phase_assign, graph, max_ffn, layer_ffn_counts)
        phase_assign[op] = phase
        if isinstance(op, ReGLUDimension):
            layer_ffn_counts[phase // 4] += 1
        scheduled += 1

        for child in graph["children"].get(op, set()):
            indegree[child] -= 1
            if indegree[child] == 0:
                earliest = _earliest_phase(child, phase_assign, graph, max_ffn, layer_ffn_counts)
                heapq.heappush(
                    ready,
                    (earliest, _type_priority(child), getattr(child, "id", seq), seq, child),
                )
                seq += 1

    if scheduled != len(graph["ops"]):
        raise RuntimeError("cycle in graph dependencies")
    return phase_assign


def _extract_std_layers(phase_assign):
    if not phase_assign:
        return [], 0
    max_phase = max(phase_assign.values())
    num_layers = max_phase // 4 + 1
    by_phase = defaultdict(list)
    for op, phase in phase_assign.items():
        by_phase[phase].append(op)

    std_layers = []
    for layer in range(num_layers):
        std_layers.append(
            (
                [op for op in by_phase.get(4 * layer, []) if isinstance(op, LookUp)],
                [op for op in by_phase.get(4 * layer + 1, []) if isinstance(op, PersistDimension)],
                [op for op in by_phase.get(4 * layer + 2, []) if isinstance(op, ReGLUDimension)],
                [op for op in by_phase.get(4 * layer + 3, []) if isinstance(op, PersistDimension)],
            )
        )
    return std_layers, num_layers


def _output_dims(output_tokens):
    dims = set()
    for expr in output_tokens.values():
        if isinstance(expr, Expression):
            dims |= set(expr.terms.keys())
    return dims


def _birth_death(graph, phase_assign, output_tokens, pos, ilp, psq):
    all_dims = _all_result_dims(graph)
    dim_to_op = graph["dim_to_op"]
    consumers = graph["consumers"]
    output_dims = _output_dims(output_tokens)

    dim_birth = {}
    for dimension in all_dims:
        if isinstance(dimension, InputDimension):
            dim_birth[dimension] = -1
        else:
            producer = dim_to_op.get(dimension)
            if producer in phase_assign:
                dim_birth[dimension] = phase_assign[producer]

    last_boundary = 4 * (max(phase_assign.values()) // 4 + 1) - 1 if phase_assign else -1
    protected_post = {pos, ilp, psq}
    dim_death = {}
    for dimension in all_dims:
        if dimension in output_dims or dimension in protected_post:
            dim_death[dimension] = last_boundary + 1
            continue
        last_phase = -1
        for consumer in consumers.get(dimension, set()):
            if consumer in phase_assign:
                last_phase = max(last_phase, phase_assign[consumer])
        if last_phase >= 0:
            dim_death[dimension] = last_phase
        elif dimension in dim_birth:
            dim_death[dimension] = dim_birth[dimension]

    return all_dims, dim_birth, dim_death


def _alive_after(num_layers, dim_birth, dim_death):
    def alive_at(boundary):
        return frozenset(
            dimension
            for dimension in dim_birth
            if dimension in dim_death
            and dim_birth[dimension] <= boundary
            and dim_death[dimension] > boundary
        )

    alive = {}
    for layer in range(num_layers):
        for phase in (1, 3):
            boundary = 4 * layer + phase
            alive[boundary] = alive_at(boundary)
    return alive


def interval_coloring(all_dims, dim_birth, dim_death, fixed=None):
    fixed = fixed or {}
    remaining = [dimension for dimension in all_dims if dimension not in fixed]
    items = sorted(
        (dim_birth[dimension], dim_death[dimension], index, dimension)
        for index, dimension in enumerate(remaining)
        if dimension in dim_birth and dimension in dim_death and dim_death[dimension] > dim_birth[dimension]
    )
    slot_of = dict(fixed)
    free = []
    next_slot = max(fixed.values(), default=-1) + 1
    for dimension, slot in fixed.items():
        if dimension in dim_death:
            heapq.heappush(free, (dim_death[dimension], slot))

    for birth, death_phase, _index, dimension in items:
        available = []
        while free and free[0][0] <= birth:
            available.append(heapq.heappop(free)[1])
        if available:
            slot = min(available)
            for free_slot in available:
                if free_slot != slot:
                    heapq.heappush(free, (birth, free_slot))
        else:
            slot = next_slot
            next_slot += 1
        slot_of[dimension] = slot
        heapq.heappush(free, (death_phase, slot))

    return slot_of


def deterministic_schedule(
    input_tokens,
    output_tokens,
    max_layers=None,
    max_ffn=None,
    program_graph=None,
):
    del input_tokens

    if program_graph is not None:
        pos = program_graph.position
        ilp = program_graph.inv_log_pos
        psq = program_graph.position_sq
        graph = _build_graph(program_graph.all_dims, program_graph.all_lookups, ilp)
    else:
        pos = _default_position
        ilp = _default_inv_log_pos
        psq = _default_position_sq
        graph = _build_graph()

    phase_assign = _schedule_asap(graph, max_ffn=max_ffn)
    std_layers, num_layers = _extract_std_layers(phase_assign)
    if max_layers is not None and num_layers > max_layers:
        raise RuntimeError(f"deterministic schedule needs {num_layers} layers, exceeds max_layers={max_layers}")

    all_dims, dim_birth, dim_death = _birth_death(graph, phase_assign, output_tokens, pos, ilp, psq)
    alive_after = _alive_after(num_layers, dim_birth, dim_death)
    dep_widths = {boundary: len(dims) for boundary, dims in alive_after.items()}
    width = 2 * max((((len(dims) + 1) // 2) for dims in alive_after.values()), default=1)

    return dict(
        phase_assign=phase_assign,
        std_layers=std_layers,
        num_layers=num_layers,
        dim_birth=dim_birth,
        dim_death=dim_death,
        alive_after=alive_after,
        lin_widths=dep_widths,
        width=width,
        all_dims=all_dims,
    )

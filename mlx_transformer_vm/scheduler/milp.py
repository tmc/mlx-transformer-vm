"""MILP scheduler minimizing transformer width under VM phase constraints."""

from __future__ import annotations

import argparse
import heapq
import logging
from collections import defaultdict

import numpy as np
import yaml
from pulp import LpBinary, LpInteger, LpMinimize, LpProblem, LpVariable, lpSum, value

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

logger = logging.getLogger(__name__)


def _build_graph(all_dims=None, all_lookups=None, ilp=None):
    """Build dependency metadata from graph dimensions and lookups."""

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


def _min_layers(ops, op_deps):
    """Return the critical path length subject to phase alignment."""

    phase = {}
    remaining = set(ops)
    while remaining:
        progress = False
        for op in list(remaining):
            if not all(pred in phase for pred in op_deps[op]):
                continue
            lo = max((phase[pred] for pred in op_deps[op]), default=-1) + 1
            if isinstance(op, LookUp):
                lo += (-lo) % 4
            elif isinstance(op, ReGLUDimension):
                lo += (2 - lo % 4 + 4) % 4
            else:
                lo += 0 if lo % 2 == 1 else 1
            phase[op] = lo
            remaining.discard(op)
            progress = True
        if not progress:
            raise RuntimeError("cycle in dependencies")
    return max(phase.values()) // 4 + 1


def _all_result_dims(graph):
    dims = list(graph["inputs"])
    dim_set = set(dims)
    for op in graph["ops"]:
        for dimension in graph["produced"][op]:
            if dimension not in dim_set:
                dim_set.add(dimension)
                dims.append(dimension)
    return dims


def milp_schedule(
    input_tokens,
    output_tokens,
    max_layers=None,
    max_ffn=None,
    log=None,
    program_graph=None,
    plan_path=None,
):
    """Compute an optimized 4-phase schedule using MILP."""

    del input_tokens
    log_fn = log or logger.info

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

    ops = graph["ops"]
    op_deps = graph["op_deps"]
    tight_to = graph.get("tight_to", {})
    produced = graph["produced"]
    consumers = graph["consumers"]
    dim_to_op = graph["dim_to_op"]
    all_dims = _all_result_dims(graph)

    output_dims = set()
    for expr in output_tokens.values():
        if isinstance(expr, Expression):
            output_dims |= set(expr.terms.keys())

    n_layers = max_layers or _min_layers(ops, op_deps)
    n_phases = 4 * n_layers
    log_fn(f"MILP: {len(ops)} ops, {len(all_dims)} dims, {n_layers} layers, {n_phases} phases")

    problem = LpProblem("schedule", LpMinimize)
    d_half = LpVariable("D_half", 0, cat="Integer")
    problem += d_half

    k = {op: LpVariable(f"k_{idx}", 0, n_layers - 1, LpInteger) for idx, op in enumerate(ops)}
    z = {
        op: LpVariable(f"z_{idx}", 0, 1, LpBinary)
        for idx, op in enumerate(ops)
        if isinstance(op, PersistDimension)
    }

    def phase_of(op):
        if isinstance(op, LookUp):
            return 4 * k[op]
        if isinstance(op, ReGLUDimension):
            return 4 * k[op] + 2
        return 4 * k[op] + 1 + 2 * z[op]

    for op in ops:
        for pred in op_deps.get(op, set()):
            if pred in k:
                problem += phase_of(op) >= phase_of(pred) + 1

    for op, lookups in tight_to.items():
        for lookup in lookups:
            if lookup in k and op in k:
                problem += k[op] == k[lookup]

    death = {}
    for dimension in all_dims:
        if dimension in output_dims:
            continue
        cons = [consumer for consumer in consumers.get(dimension, set()) if consumer in k]
        if not cons and dimension is not pos:
            continue
        death_var = LpVariable(f"d_{id(dimension)}", 0, n_phases - 1, LpInteger)
        for consumer in cons:
            problem += death_var >= phase_of(consumer)
        death[dimension] = death_var

    if pos in death:
        for op in ops:
            if isinstance(op, PersistDimension) and op in z:
                problem += death[pos] >= phase_of(op) - n_phases * z[op]

    if max_ffn is not None:
        reglus = graph["reglus"]
        ff_bin = {}
        for index, reglu in enumerate(reglus):
            for layer in range(n_layers):
                ff_bin[index, layer] = LpVariable(f"fb_{index}_{layer}", 0, 1, LpBinary)
            problem += lpSum(ff_bin[index, layer] for layer in range(n_layers)) == 1
            problem += k[reglu] == lpSum(layer * ff_bin[index, layer] for layer in range(n_layers))
        for layer in range(n_layers):
            problem += lpSum(ff_bin[index, layer] for index in range(len(reglus))) <= max_ffn

    dim_index = {dimension: idx for idx, dimension in enumerate(all_dims)}
    lookups = graph["lookups"]
    persists = graph["persists"]
    deps_cache = graph["deps_cache"]

    lu_at = {}
    for lookup in lookups:
        for layer in range(n_layers):
            lu_at[lookup, layer] = LpVariable(f"la_{id(lookup)}_{layer}", 0, 1, LpBinary)
        problem += lpSum(lu_at[lookup, layer] for layer in range(n_layers)) == 1
        problem += k[lookup] == lpSum(layer * lu_at[lookup, layer] for layer in range(n_layers))

    p_layer = {}
    p1_at = {}
    for persist in persists:
        for layer in range(n_layers):
            p_layer[persist, layer] = LpVariable(f"pl_{id(persist)}_{layer}", 0, 1, LpBinary)
        problem += lpSum(p_layer[persist, layer] for layer in range(n_layers)) == 1
        problem += k[persist] == lpSum(layer * p_layer[persist, layer] for layer in range(n_layers))
        for layer in range(n_layers):
            p1_var = LpVariable(f"p1_{id(persist)}_{layer}", 0, 1, LpBinary)
            problem += p1_var <= p_layer[persist, layer]
            problem += p1_var <= 1 - z[persist]
            problem += p1_var >= p_layer[persist, layer] + (1 - z[persist]) - 1
            p1_at[persist, layer] = p1_var

    p1_deps = defaultdict(set)
    for persist in persists:
        for dimension in deps_cache[persist]:
            if isinstance(
                dimension,
                (InputDimension, LookUpDimension, ReGLUDimension, PersistDimension),
            ):
                p1_deps[dimension].add(persist)

    passthrough_var = {}
    for dimension, persist_set in p1_deps.items():
        lookup_of_dim = dim_to_op.get(dimension) if isinstance(dimension, LookUpDimension) else None
        for layer in range(n_layers):
            var = LpVariable(f"pd_{id(dimension)}_{layer}", 0, 1, LpBinary)
            problem += var <= lpSum(p1_at[persist, layer] for persist in persist_set)
            if lookup_of_dim is not None and lookup_of_dim in lu_at:
                problem += var <= 1 - lu_at[lookup_of_dim, layer]
                for persist in persist_set:
                    problem += var >= p1_at[persist, layer] - lu_at[lookup_of_dim, layer]
            else:
                for persist in persist_set:
                    problem += var >= p1_at[persist, layer]
            passthrough_var[dimension, layer] = var

    lookup_heads = {lookup: (len(lookup.value_exprs) + 1) // 2 for lookup in lookups}
    lookup_dims = {lookup: len(lookup.dims) for lookup in lookups}

    needs_slot = {}
    for dimension in all_dims:
        if dimension in output_dims or dimension not in death:
            continue
        producer = dim_to_op.get(dimension)
        if producer is None or producer not in k:
            continue
        if isinstance(dimension, (LookUpDimension, ReGLUDimension)):
            needs = LpVariable(f"ns_{dim_index[dimension]}", 0, 1, LpBinary)
            problem += death[dimension] >= phase_of(producer) + 2 - n_phases * (1 - needs)
            problem += death[dimension] <= phase_of(producer) + 1 + n_phases * needs
            needs_slot[dimension] = needs

    protected_dims = {pos, ilp, psq}
    alive_sum = {}
    n_inputs = sum(1 for dimension in all_dims if isinstance(dimension, InputDimension))

    for boundary in range(n_phases):
        if boundary % 2 == 0:
            continue
        effective_width = []
        alive = []
        for dimension in all_dims:
            producer = dim_to_op.get(dimension)
            is_input = isinstance(dimension, InputDimension)

            if dimension in output_dims or dimension in protected_dims:
                if is_input:
                    effective_width.append(1)
                    alive.append(1)
                elif producer in k:
                    born = LpVariable(f"b_{dim_index[dimension]}_{boundary}", 0, 1, LpBinary)
                    problem += phase_of(producer) <= boundary + n_phases * (1 - born)
                    problem += phase_of(producer) >= (boundary + 1) - n_phases * born
                    effective_width.append(born)
                    alive.append(born)
                continue

            if dimension not in death:
                continue

            if is_input:
                alive_for_width = LpVariable(f"ew_{dim_index[dimension]}_{boundary}", 0, 1, LpBinary)
                problem += death[dimension] >= (boundary - 1) - n_phases * (1 - alive_for_width)
                problem += death[dimension] <= (boundary - 2) + n_phases * alive_for_width
                effective_width.append(alive_for_width)

                alive_after = LpVariable(f"a_{dim_index[dimension]}_{boundary}", 0, 1, LpBinary)
                problem += death[dimension] >= (boundary + 1) - n_phases * (1 - alive_after)
                problem += death[dimension] <= boundary + n_phases * alive_after
                alive.append(alive_after)
            elif producer in k:
                born = LpVariable(f"b_{dim_index[dimension]}_{boundary}", 0, 1, LpBinary)
                problem += phase_of(producer) <= boundary + n_phases * (1 - born)
                problem += phase_of(producer) >= (boundary + 1) - n_phases * born

                alive_for_width = LpVariable(
                    f"eu_{dim_index[dimension]}_{boundary}",
                    0,
                    1,
                    LpBinary,
                )
                problem += death[dimension] >= (boundary - 1) - n_phases * (1 - alive_for_width)
                problem += death[dimension] <= (boundary - 2) + n_phases * alive_for_width

                occupied = LpVariable(f"ew_{dim_index[dimension]}_{boundary}", 0, 1, LpBinary)
                if dimension in needs_slot:
                    problem += occupied <= born
                    problem += occupied <= alive_for_width
                    problem += occupied <= needs_slot[dimension]
                    problem += occupied >= born + alive_for_width + needs_slot[dimension] - 2
                else:
                    problem += occupied <= born
                    problem += occupied <= alive_for_width
                    problem += occupied >= born + alive_for_width - 1
                effective_width.append(occupied)

                alive_future = LpVariable(f"au_{dim_index[dimension]}_{boundary}", 0, 1, LpBinary)
                problem += death[dimension] >= (boundary + 1) - n_phases * (1 - alive_future)
                problem += death[dimension] <= boundary + n_phases * alive_future

                alive_out = LpVariable(f"a_{dim_index[dimension]}_{boundary}", 0, 1, LpBinary)
                problem += alive_out <= born
                problem += alive_out <= alive_future
                problem += alive_out >= born + alive_future - 1
                alive.append(alive_out)

        problem += 2 * d_half >= lpSum(effective_width)
        alive_sum[boundary] = lpSum(alive)

    for layer in range(n_layers):
        boundary = 4 * layer + 1
        prev_boundary = 4 * layer - 1
        n_lookup = lpSum(lookup_heads[lookup] * lu_at[lookup, layer] for lookup in lookups)
        passthrough = lpSum(
            passthrough_var[dimension, layer]
            for dimension in p1_deps
            if (dimension, layer) in passthrough_var
        )
        born = lpSum(lookup_dims[lookup] * lu_at[lookup, layer] for lookup in lookups) + lpSum(
            p1_at[persist, layer] for persist in persists
        )
        prev_alive = alive_sum.get(prev_boundary, n_inputs)
        cur_alive = alive_sum[boundary]
        dying = prev_alive - cur_alive + born
        problem += 2 * d_half >= 2 * n_lookup + dying + passthrough

    log_fn("Solving MILP...")
    try:
        from pulp import HiGHS

        solver = HiGHS(timeLimit=3600)
        if not solver.available():
            raise RuntimeError("HiGHS unavailable")
    except Exception:
        from pulp import PULP_CBC_CMD

        solver = PULP_CBC_CMD(msg=0, timeLimit=3600)
    problem.solve(solver)

    if problem.status != 1:
        raise RuntimeError(f"MILP infeasible (status={problem.status}); try more layers")

    opt_d_half = int(round(value(d_half)))
    opt_d_model = 2 * opt_d_half
    log_fn(f"MILP optimal d_model: {opt_d_model}")

    phase_assign = {}
    for op in ops:
        layer = int(round(value(k[op])))
        if isinstance(op, LookUp):
            phase_assign[op] = 4 * layer
        elif isinstance(op, ReGLUDimension):
            phase_assign[op] = 4 * layer + 2
        else:
            phase_assign[op] = 4 * layer + 1 + 2 * int(round(value(z[op])))

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

    dim_birth = {}
    for dimension in all_dims:
        if isinstance(dimension, InputDimension):
            dim_birth[dimension] = -1
        else:
            producer = dim_to_op.get(dimension)
            if producer and producer in phase_assign:
                dim_birth[dimension] = phase_assign[producer]

    last_boundary = 4 * num_layers - 1
    protected_post = {pos, ilp, psq}
    dim_death = {}
    for dimension in all_dims:
        if dimension in output_dims or dimension in protected_post:
            dim_death[dimension] = last_boundary + 1
            continue
        last = -1
        for consumer in consumers.get(dimension, set()):
            if consumer in phase_assign:
                last = max(last, phase_assign[consumer])
        if last >= 0:
            dim_death[dimension] = last
        elif dimension in dim_birth:
            dim_death[dimension] = dim_birth[dimension]

    def alive_at(boundary):
        return frozenset(
            dimension
            for dimension in all_dims
            if dimension in dim_birth
            and dimension in dim_death
            and dim_birth[dimension] <= boundary
            and dim_death[dimension] > boundary
        )

    alive_after = {}
    for layer in range(num_layers):
        for half in (1, 3):
            boundary = 4 * layer + half
            alive_after[boundary] = alive_at(boundary)

    dim_col = {dimension: idx for idx, dimension in enumerate(all_dims)}
    n_dims = len(all_dims)

    def expr_vec(expr):
        vec = np.zeros(n_dims)
        if isinstance(expr, Expression):
            for dimension, coeff in expr.terms.items():
                if dimension in dim_col:
                    vec[dim_col[dimension]] = coeff
        return vec

    op_vecs = {}
    for op in ops:
        if isinstance(op, PersistDimension):
            op_vecs[op] = [expr_vec(op.expr)]
        elif isinstance(op, ReGLUDimension):
            op_vecs[op] = [expr_vec(op.a_expr), expr_vec(op.b_expr)]
        elif isinstance(op, LookUp):
            op_vecs[op] = [expr_vec(expr) for expr in op.query_exprs_2d + op.key_exprs_2d + op.value_exprs]
    out_vecs = [expr_vec(expr) for expr in output_tokens.values()]

    lin_widths = {}
    for boundary in sorted(alive_after):
        past = np.array([dim_birth.get(dimension, max_phase + 1) <= boundary for dimension in all_dims], dtype=bool)
        rows = []
        for op in ops:
            if phase_assign[op] > boundary:
                for vec in op_vecs.get(op, []):
                    masked = vec * past
                    if np.any(masked != 0):
                        rows.append(masked)
        for vec in out_vecs:
            masked = vec * past
            if np.any(masked != 0):
                rows.append(masked)
        if rows:
            matrix = np.vstack(rows)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            lin_widths[boundary] = int(np.linalg.matrix_rank(matrix / norms))
        else:
            lin_widths[boundary] = 0

    expand_cache = {}

    def expand_dim(dimension):
        if dimension in expand_cache:
            return expand_cache[dimension]
        if isinstance(dimension, PersistDimension):
            vec = np.zeros(n_dims)
            for dep, coeff in dimension.expr.terms.items():
                vec += coeff * expand_dim(dep)
        else:
            vec = np.zeros(n_dims)
            if dimension in dim_col:
                vec[dim_col[dimension]] = 1.0
        expand_cache[dimension] = vec
        return vec

    def expr_vec_expanded(expr):
        vec = np.zeros(n_dims)
        if isinstance(expr, Expression):
            for dimension, coeff in expr.terms.items():
                vec += coeff * expand_dim(dimension)
        return vec

    op_vecs_expanded = {}
    for op in ops:
        if isinstance(op, PersistDimension):
            op_vecs_expanded[op] = [expr_vec_expanded(op.expr)]
        elif isinstance(op, ReGLUDimension):
            op_vecs_expanded[op] = [expr_vec_expanded(op.a_expr), expr_vec_expanded(op.b_expr)]
        elif isinstance(op, LookUp):
            op_vecs_expanded[op] = [
                expr_vec_expanded(expr) for expr in op.query_exprs_2d + op.key_exprs_2d + op.value_exprs
            ]
    out_vecs_expanded = [expr_vec_expanded(expr) for expr in output_tokens.values()]

    exp_lin_widths = {}
    for boundary in sorted(alive_after):
        past = np.array([dim_birth.get(dimension, max_phase + 1) <= boundary for dimension in all_dims], dtype=bool)
        rows = []
        for op in ops:
            if phase_assign[op] > boundary:
                for vec in op_vecs_expanded.get(op, []):
                    masked = vec * past
                    if np.any(masked != 0):
                        rows.append(masked)
        for vec in out_vecs_expanded:
            masked = vec * past
            if np.any(masked != 0):
                rows.append(masked)
        if rows:
            matrix = np.vstack(rows)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            exp_lin_widths[boundary] = int(np.linalg.matrix_rank(matrix / norms))
        else:
            exp_lin_widths[boundary] = 0

    max_dep = max((len(dims) for dims in alive_after.values()), default=0)
    max_lin = max(lin_widths.values(), default=0)
    max_exp_lin = max(exp_lin_widths.values(), default=0)
    log_fn(
        f"\nSchedule: {num_layers} layers, d_model={opt_d_model}, "
        f"max_dep={max_dep}, max_lin={max_lin}, max_exp_lin={max_exp_lin}"
    )

    if plan_path is not None:
        _write_plan(
            plan_path,
            std_layers,
            num_layers,
            opt_d_model,
            max_dep,
            max_lin,
            alive_after,
            lin_widths,
            produced,
            max_phase,
        )
        log_fn(f"Schedule written to {plan_path}")

    return dict(
        phase_assign=phase_assign,
        std_layers=std_layers,
        num_layers=num_layers,
        dim_birth=dim_birth,
        dim_death=dim_death,
        alive_after=alive_after,
        lin_widths=lin_widths,
        width=opt_d_model,
    )


class _InlineList(list):
    """List marker for inline YAML rendering."""


def _compact_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _write_plan(
    path,
    std_layers,
    num_layers,
    width,
    max_dep,
    max_lin,
    alive_after,
    lin_widths,
    produced,
    max_phase,
):
    yaml.add_representer(_InlineList, _compact_representer)

    plan = {
        "summary": {
            "layers": num_layers,
            "milp_d_model": width,
            "max_dep_width": max_dep,
            "max_lin_width": max_lin,
        },
        "layers": [],
    }
    for layer in range(num_layers):
        attn, persist1, ffn, persist2 = std_layers[layer]
        boundary1 = 4 * layer + 1
        boundary2 = 4 * layer + 3
        attn_dims = []
        for lookup in attn:
            attn_dims.extend(dimension.name for dimension in produced.get(lookup, []))
        entry = {
            "layer": layer,
            "attention": _InlineList(attn_dims),
            "persist1": _InlineList(persist.name for persist in persist1),
            "after_persist1": {
                "dep_width": len(alive_after.get(boundary1, set())),
                "lin_width": lin_widths.get(boundary1, 0),
                "dims": _InlineList(sorted(dimension.name for dimension in alive_after.get(boundary1, set()))),
            },
            "ffn": _InlineList(reglu.name for reglu in ffn),
            "persist2": _InlineList(persist.name for persist in persist2),
        }
        if boundary2 <= max_phase:
            entry["after_persist2"] = {
                "dep_width": len(alive_after.get(boundary2, set())),
                "lin_width": lin_widths.get(boundary2, 0),
                "dims": _InlineList(sorted(dimension.name for dimension in alive_after.get(boundary2, set()))),
            }
        plan["layers"].append(entry)

    with open(path, "w") as handle:
        yaml.dump(plan, handle, default_flow_style=False, sort_keys=False, width=200)


def interval_coloring(all_dims, dim_birth, dim_death, fixed=None):
    """Assign slots greedily to interval lifetimes."""

    fixed = fixed or {}
    remaining = [dimension for dimension in all_dims if dimension not in fixed]
    items = sorted(
        (dim_birth[dimension], dim_death[dimension], idx, dimension)
        for idx, dimension in enumerate(remaining)
        if dimension in dim_birth and dimension in dim_death and dim_death[dimension] > dim_birth[dimension]
    )
    slot_of = dict(fixed)
    free = []
    next_slot = max(fixed.values(), default=-1) + 1
    for dimension, slot in fixed.items():
        if dimension in dim_death:
            heapq.heappush(free, (dim_death[dimension], slot))

    for birth, death_phase, _idx, dimension in items:
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


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run the MILP scheduler for the WASM interpreter.")
    parser.add_argument("--max-layers", type=int, default=None, help="Max transformer layers")
    parser.add_argument("--max-ffn", type=int, default=None, help="Max FFN neurons per layer")
    parser.add_argument(
        "--plan",
        default="plan.yaml",
        help="Path to write the schedule plan (default: plan.yaml)",
    )
    args = parser.parse_args()

    from mlx_transformer_vm.wasm.interpreter import build

    input_tokens, output_tokens = build()
    milp_schedule(
        input_tokens,
        output_tokens,
        max_layers=args.max_layers,
        max_ffn=args.max_ffn,
        plan_path=args.plan,
    )


if __name__ == "__main__":
    main()

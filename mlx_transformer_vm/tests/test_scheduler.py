from __future__ import annotations

from mlx_transformer_vm.graph import core as graph
from mlx_transformer_vm.graph.core import (
    InputDimension,
    ProgramGraph,
    fetch,
    persist,
    reglu,
    reset_graph,
)
from mlx_transformer_vm.scheduler.deterministic import _build_graph, deterministic_schedule
from mlx_transformer_vm.scheduler.milp import milp_schedule
from mlx_transformer_vm.wasm.interpreter import WASMMachine


def test_deterministic_schedule_places_lookup_before_persist():
    reset_graph()
    x = InputDimension("x")
    fetched = fetch(x, query=graph.one, key=graph.one)
    kept = persist(fetched + 1, name="kept")

    program_graph = ProgramGraph({"tok": x + 1}, {"out": kept})
    schedule = deterministic_schedule(
        program_graph.input_tokens,
        program_graph.output_tokens,
        program_graph=program_graph,
    )

    lookup_phase = schedule["phase_assign"][fetched.lookup]
    [persist_dim] = list(kept.terms)
    persist_phase = schedule["phase_assign"][persist_dim]

    assert lookup_phase == 0
    assert persist_phase in {1, 3}
    assert lookup_phase < persist_phase


def test_deterministic_schedule_places_reglu_before_persist2():
    reset_graph()
    x = InputDimension("x")
    gated = reglu(x, graph.one)
    kept = persist(gated + x, name="kept")

    program_graph = ProgramGraph({"tok": x + 1}, {"out": kept})
    schedule = deterministic_schedule(
        program_graph.input_tokens,
        program_graph.output_tokens,
        program_graph=program_graph,
    )

    [reglu_dim] = list(gated.terms)
    [persist_dim] = list(kept.terms)
    assert schedule["phase_assign"][reglu_dim] == 2
    assert schedule["phase_assign"][persist_dim] == 3


def test_milp_schedule_places_lookup_before_persist():
    reset_graph()
    x = InputDimension("x")
    fetched = fetch(x, query=graph.one, key=graph.one)
    kept = persist(fetched + 1, name="kept")

    program_graph = ProgramGraph({"tok": x + 1}, {"out": kept})
    schedule = milp_schedule(
        program_graph.input_tokens,
        program_graph.output_tokens,
        program_graph=program_graph,
    )

    lookup_phase = schedule["phase_assign"][fetched.lookup]
    [persist_dim] = list(kept.terms)
    persist_phase = schedule["phase_assign"][persist_dim]

    assert lookup_phase == 0
    assert persist_phase in {1, 3}
    assert lookup_phase < persist_phase


def test_milp_schedule_respects_max_ffn():
    reset_graph()
    x = InputDimension("x")
    left = reglu(x, graph.one)
    right = reglu(x + 1, graph.one)
    out = persist(left + right, name="out")

    program_graph = ProgramGraph({"tok": x + 1}, {"out": out})
    schedule = milp_schedule(
        program_graph.input_tokens,
        program_graph.output_tokens,
        program_graph=program_graph,
        max_layers=2,
        max_ffn=1,
    )

    assert schedule["num_layers"] == 2
    assert all(len(ffn) <= 1 for _, _, ffn, _ in schedule["std_layers"])


def test_wasm_machine_schedule_respects_dependencies():
    program_graph = WASMMachine().build()
    schedule = deterministic_schedule(
        program_graph.input_tokens,
        program_graph.output_tokens,
        program_graph=program_graph,
        max_ffn=256,
    )
    graph_meta = _build_graph(program_graph.all_dims, program_graph.all_lookups, program_graph.inv_log_pos)

    assert schedule["num_layers"] > 0
    assert schedule["width"] % 2 == 0

    for op, deps in graph_meta["op_deps"].items():
        for dep in deps:
            assert schedule["phase_assign"][dep] < schedule["phase_assign"][op]

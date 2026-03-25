from __future__ import annotations

from mlx_transformer_vm.wasm.interpreter import WASMMachine


def test_wasm_machine_builds_universal_graph():
    program_graph = WASMMachine().build()

    assert "{" in program_graph.input_tokens
    assert "halt" in program_graph.output_tokens
    assert len(program_graph.all_dims) > 0
    assert len(program_graph.all_lookups) > 0

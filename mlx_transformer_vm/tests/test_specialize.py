from __future__ import annotations

from pathlib import Path

import pytest

from mlx_transformer_vm.attention import StandardKVCache
from mlx_transformer_vm.compilation.compile_wasm import compile_program, find_clang
from mlx_transformer_vm.graph import core as graph
from mlx_transformer_vm.graph.core import (
    InputDimension,
    ProgramGraph,
    fetch,
    persist,
    reglu,
    reset_graph,
)
from mlx_transformer_vm.model import build_model, load_weights, save_weights
from mlx_transformer_vm.runner import run_model_tokens
from mlx_transformer_vm.specialize import spec_input_from_txt, specialize
from mlx_transformer_vm.wasm.reference import generate_ref


def _build_custom_graph():
    reset_graph()
    x = InputDimension("x")
    fetched = fetch(x, query=graph.one, key=graph.one)
    mix = persist(fetched + x, name="mix")
    gated = reglu(graph.one, mix)
    out = persist(2 * gated + mix, name="out")
    return ProgramGraph(
        {
            "neg": graph.one - 4 * x,
            "pos": graph.one + x,
        },
        {
            "neg": -out,
            "pos": out,
            "halt": -100 * graph.one,
        },
    )


def test_save_and_load_weights_round_trip(tmp_path):
    program_graph = _build_custom_graph()
    model, all_tokens, tok_to_idx_map, _shared = build_model(program_graph=program_graph)

    path = tmp_path / "model.bin"
    save_weights(model, all_tokens, str(path))
    loaded_model, loaded_tokens, loaded_map = load_weights(str(path))

    tokens = ["neg", "pos"]
    expected = run_model_tokens(
        model,
        all_tokens,
        tok_to_idx_map,
        tokens,
        max_new_tokens=2,
        cache_class=StandardKVCache,
    )
    actual = run_model_tokens(
        loaded_model,
        loaded_tokens,
        loaded_map,
        tokens,
        max_new_tokens=2,
        cache_class=StandardKVCache,
    )

    assert actual == expected


def test_specialize_matches_reference_on_constant_program(tmp_path):
    try:
        find_clang()
    except Exception:
        pytest.skip("wasm32-capable clang not available")

    source = tmp_path / "hello.c"
    source.write_text(
        "\n".join(
            [
                "void compute(const char *input) {",
                "    putchar('H');",
                "    putchar('i');",
                "    putchar('\\n');",
                "}",
                "",
            ]
        )
    )

    out_base = tmp_path / "hello"
    txt_path, spec_path, _input_base = compile_program(str(source), out_base=str(out_base))
    ref_path = tmp_path / "hello_ref.txt"
    generate_ref(str(txt_path), str(ref_path))

    model, all_tokens, tok_to_idx_map, _instructions = specialize(txt_path)
    predicted = run_model_tokens(
        model,
        all_tokens,
        tok_to_idx_map,
        spec_input_from_txt(txt_path),
        cache_class=StandardKVCache,
    )
    ref_tokens = ref_path.read_text().split()
    ref_exec = ref_tokens[ref_tokens.index("}") + 1 :]

    assert predicted[0] == "start"
    assert predicted[1:] == ref_exec
    assert Path(spec_path).read_text().split() == ["start"]


def test_compile_program_smoke(tmp_path):
    try:
        find_clang()
    except Exception:
        pytest.skip("wasm32-capable clang not available")

    source = tmp_path / "hello.c"
    source.write_text(
        "\n".join(
            [
                "void compute(const char *input) {",
                "    putchar('O');",
                "    putchar('K');",
                "    putchar('\\n');",
                "}",
                "",
            ]
        )
    )

    out_base = tmp_path / "hello"
    txt_path, spec_path, input_base = compile_program(str(source), out_base=str(out_base))

    assert Path(txt_path).exists()
    assert Path(spec_path).exists()
    assert input_base > 0
    assert spec_input_from_txt(txt_path) == ["start"]

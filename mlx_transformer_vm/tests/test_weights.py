from __future__ import annotations

import mlx.core as mx

from mlx_transformer_vm.attention import StandardKVCache
from mlx_transformer_vm.evaluator import Runtime
from mlx_transformer_vm.graph import core as graph
from mlx_transformer_vm.graph.core import (
    InputDimension,
    ProgramGraph,
    fetch,
    persist,
    reglu,
    reset_graph,
)
from mlx_transformer_vm.model import build_model


def _predict_tokens(model, all_tokens, tok_to_idx_map, tokens):
    cache = StandardKVCache(model.n_layers, model.n_heads)
    predicted = []
    for pos, token in enumerate(tokens):
        logits = model.step_logits(tok_to_idx_map[token], pos, cache)
        predicted.append(all_tokens[int(mx.argmax(logits).item())])
    return predicted


def test_build_model_matches_exact_runtime_on_custom_graph():
    reset_graph()

    x = InputDimension("x")
    fetched = fetch(x, query=graph.one, key=graph.one)
    mix = persist(fetched + x, name="mix")
    gated = reglu(graph.one, mix)
    out = persist(2 * gated + mix, name="out")

    program_graph = ProgramGraph(
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

    model, all_tokens, tok_to_idx_map, _shared = build_model(program_graph=program_graph)
    runtime = Runtime(program_graph=program_graph)
    tokens = ["neg", "pos", "neg"]

    expected = []
    for token in tokens:
        values = runtime.step(token)
        expected.append(runtime.predict_next(values))

    assert _predict_tokens(model, all_tokens, tok_to_idx_map, tokens) == expected
    assert any(any(layer) for layer in model.head_tiebreak)


def test_build_model_generate_with_cache_uses_built_weights():
    reset_graph()

    x = InputDimension("x")
    fetched = fetch(x, query=graph.one, key=graph.one)
    mix = persist(fetched + x, name="mix")
    gated = reglu(graph.one, mix)
    out = persist(2 * gated + mix, name="out")

    program_graph = ProgramGraph(
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

    model, all_tokens, tok_to_idx_map, _shared = build_model(program_graph=program_graph)
    idx = mx.array([[tok_to_idx_map["neg"], tok_to_idx_map["pos"]]], dtype=mx.int32)
    out = model.generate_with_cache(idx, max_new_tokens=1)

    predicted = [all_tokens[index] for index in out[0].tolist()]
    assert predicted == ["neg", "pos", "pos"]

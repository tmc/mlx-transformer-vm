from __future__ import annotations

from mlx_transformer_vm.graph import core as graph
from mlx_transformer_vm.graph.core import (
    Expression,
    InputDimension,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
    auto_name,
    fetch,
    fetch_sum,
    persist,
    reglu,
    stepglu,
)


def test_expression_arithmetic():
    x = InputDimension("x")
    y = InputDimension("y")
    expr = x * 3 + y - 2

    assert isinstance(expr, Expression)
    assert expr[x] == 3
    assert expr[y] == 1
    assert expr[graph.one] == -2
    assert expr.evaluate({x: 4, y: 5, graph.one: 1}) == 15


def test_reglu_is_cached():
    x = InputDimension("x")
    gate = InputDimension("gate")

    first = reglu(x, gate)
    second = reglu(x, gate)

    [dimension] = list(first.terms)
    [dimension_again] = list(second.terms)
    assert isinstance(dimension, ReGLUDimension)
    assert dimension is dimension_again


def test_stepglu_materializes_to_persist():
    x = InputDimension("x")
    gate = InputDimension("gate")

    expr = stepglu(x, gate)

    [dimension] = list(expr.terms)
    assert isinstance(dimension, PersistDimension)


def test_fetch_returns_lookup_dimensions():
    x = InputDimension("x")
    lookup = fetch([x, x + 1], query=graph.position, key=graph.position)

    assert isinstance(lookup, tuple)
    assert len(lookup) == 2
    assert all(isinstance(item, LookUpDimension) for item in lookup)


def test_fetch_sum_builds_reglu_expression():
    x = InputDimension("x")
    expr = fetch_sum(x)

    [dimension] = list(expr.terms)
    assert isinstance(dimension, ReGLUDimension)


def test_auto_name_applies_variable_names():
    x = InputDimension("x")
    cached = persist(x + 1)
    fetched = fetch(x, query=graph.position, key=graph.position)

    auto_name(locals())

    [persist_dim] = list(cached.terms)
    assert persist_dim.name == "cached"
    assert fetched.name == "fetched"

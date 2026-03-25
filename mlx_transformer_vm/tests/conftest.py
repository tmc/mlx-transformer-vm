from __future__ import annotations

import pytest

import mlx_transformer_vm.evaluator as evaluator_mod
from mlx_transformer_vm.graph.core import reset_graph


@pytest.fixture(autouse=True)
def clean_graph():
    reset_graph()
    evaluator_mod._default_graph = None
    yield
    reset_graph()
    evaluator_mod._default_graph = None

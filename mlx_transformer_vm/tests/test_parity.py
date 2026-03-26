from __future__ import annotations

import os

import pytest

from mlx_transformer_vm.compilation.compile_wasm import has_wasm_toolchain
from mlx_transformer_vm.parity import (
    DEFAULT_EXAMPLES,
    compare_example,
)

TEST_EXAMPLES = (
    list(DEFAULT_EXAMPLES.items())
    if os.environ.get("MLX_TRANSFORMER_VM_FULL_PARITY") == "1"
    else [(name, DEFAULT_EXAMPLES[name]) for name in ("hello", "addition")]
)


@pytest.mark.slow
@pytest.mark.parametrize("example,args", TEST_EXAMPLES)
def test_example_parity(example, args, tmp_path):
    if not has_wasm_toolchain():
        pytest.skip("local wasm toolchain not available for C example compilation")
    result = compare_example(example, args, tmp_path)

    assert result["reference_ok"], example
    assert result["evaluator_ok"], example
    if result["weighted_available"]:
        assert result["weighted_ok"], example

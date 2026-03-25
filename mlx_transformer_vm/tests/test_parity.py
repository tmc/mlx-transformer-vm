from __future__ import annotations

import pytest

from mlx_transformer_vm.parity import (
    DEFAULT_EXAMPLES,
    compare_example,
    has_upstream_wasm_toolchain,
    upstream_root,
)


@pytest.mark.slow
@pytest.mark.parametrize("example,args", list(DEFAULT_EXAMPLES.items()))
def test_example_parity(example, args, tmp_path):
    upstream_root()
    if not has_upstream_wasm_toolchain():
        pytest.skip("wasm32-capable clang not available for upstream example compilation")
    result = compare_example(example, args, tmp_path)

    assert result["reference_ok"], example
    assert result["evaluator_ok"], example

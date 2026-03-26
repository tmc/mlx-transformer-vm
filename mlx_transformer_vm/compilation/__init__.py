"""WASM compilation pipeline: decode, lower, and compile."""

from mlx_transformer_vm.compilation.compile_wasm import (
    compile_program,
    compile_wasm_to_prefix,
    find_clang,
)
from mlx_transformer_vm.compilation.decoder import decode
from mlx_transformer_vm.compilation.lower import lower_hard_ops

__all__ = [
    "compile_program",
    "compile_wasm_to_prefix",
    "decode",
    "find_clang",
    "lower_hard_ops",
]

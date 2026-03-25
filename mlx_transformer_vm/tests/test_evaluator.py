from __future__ import annotations

from mlx_transformer_vm.evaluator import run_program
from mlx_transformer_vm.wasm.reference import generate_ref


def _write_program(path, instructions):
    lines = ["{"]
    for opcode, immediate in instructions:
        bytes_ = [(immediate >> (8 * offset)) & 0xFF for offset in range(4)]
        lines.append(f"{opcode} {' '.join(f'{byte:02x}' for byte in bytes_)}")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def test_evaluator_matches_reference_on_constant_output_program(tmp_path):
    program_path = tmp_path / "hello.txt"
    ref_path = tmp_path / "hello_ref.txt"
    _write_program(
        program_path,
        [
            ("i32.const", ord("H")),
            ("output", 0),
            ("i32.const", ord("i")),
            ("output", 0),
            ("i32.const", ord("\n")),
            ("output", 0),
            ("halt", 0),
        ],
    )

    generate_ref(str(program_path), str(ref_path))

    assert run_program(str(program_path), str(ref_path)) is True


def test_evaluator_matches_reference_on_arithmetic_program(tmp_path):
    program_path = tmp_path / "add.txt"
    ref_path = tmp_path / "add_ref.txt"
    _write_program(
        program_path,
        [
            ("i32.const", 65),
            ("i32.const", 1),
            ("i32.add", 0),
            ("output", 0),
            ("halt", 0),
        ],
    )

    generate_ref(str(program_path), str(ref_path))

    assert run_program(str(program_path), str(ref_path)) is True

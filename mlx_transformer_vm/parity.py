"""Helpers for comparing this port against the upstream repository."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import tempfile
from pathlib import Path

from mlx_transformer_vm.evaluator import generate_trace
from mlx_transformer_vm.wasm.reference import generate_ref

logger = logging.getLogger(__name__)

DEFAULT_UPSTREAM_ROOT = Path("/Users/tmc/go/src/github.com/Percepta-Core/transformer-vm")
DEFAULT_EXAMPLES = {
    "hello": "World",
    "addition": "12345+6789",
    "collatz": "7",
    "fibonacci": "10",
}


def upstream_root():
    root = Path(os.environ.get("TRANSFORMER_VM_UPSTREAM_ROOT", DEFAULT_UPSTREAM_ROOT))
    if not root.exists():
        raise FileNotFoundError(
            f"upstream repo not found at {root}; set TRANSFORMER_VM_UPSTREAM_ROOT to override"
        )
    return root


def _load_upstream_modules():
    root = upstream_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    compile_wasm = importlib.import_module("transformer_vm.compilation.compile_wasm")
    reference = importlib.import_module("transformer_vm.wasm.reference")
    return compile_wasm, reference


def has_upstream_wasm_toolchain():
    try:
        compile_wasm, _reference = _load_upstream_modules()
        compile_wasm.find_clang()
    except Exception:
        return False
    return True


def compile_example(example, args, out_dir):
    compile_wasm, reference = _load_upstream_modules()
    root = upstream_root()
    out_base = out_dir / example
    source = root / "transformer_vm" / "examples" / f"{example}.c"
    if not source.exists():
        raise FileNotFoundError(f"missing upstream example source: {source}")
    compile_wasm.compile_program(str(source), args, str(out_base))
    program_path = out_base.with_suffix(".txt")
    reference.generate_ref(str(program_path))
    return program_path, program_path.with_name(program_path.stem + "_ref.txt")


def compare_example(example, args, out_dir):
    program_path, upstream_ref = compile_example(example, args, out_dir)

    with open(program_path) as handle:
        tokens = handle.read().split()
    predicted = generate_trace(tokens)
    with open(upstream_ref) as handle:
        expected = handle.read().split()

    local_ref = program_path.with_name(program_path.stem + "_local_ref.txt")
    generate_ref(str(program_path), str(local_ref))
    with open(local_ref) as handle:
        local_ref_tokens = handle.read().split()

    evaluator_ok = predicted == expected
    reference_ok = local_ref_tokens == expected
    return {
        "example": example,
        "program_path": program_path,
        "ref_path": upstream_ref,
        "evaluator_ok": evaluator_ok,
        "reference_ok": reference_ok,
        "predicted": predicted,
        "expected": expected,
        "local_ref": local_ref_tokens,
    }


def compare_examples(examples=None, workdir=None):
    if examples is None:
        examples = list(DEFAULT_EXAMPLES.items())
    if isinstance(examples, dict):
        examples = list(examples.items())
    if workdir is None:
        workdir_obj = tempfile.TemporaryDirectory(prefix="mlx-transformer-vm-parity-")
        out_dir = Path(workdir_obj.name)
    else:
        workdir_obj = None
        out_dir = Path(workdir)
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = []
        for example, args in examples:
            logger.info("Checking %s", example)
            results.append(compare_example(example, args, out_dir))
        return results
    finally:
        if workdir_obj is not None:
            workdir_obj.cleanup()


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Compare this port against the upstream examples.")
    parser.add_argument(
        "--examples",
        nargs="*",
        default=list(DEFAULT_EXAMPLES),
        help="Example names to compile and compare",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory for compiled parity fixtures (defaults to a temporary directory)",
    )
    args = parser.parse_args()

    selected = [(name, DEFAULT_EXAMPLES[name]) for name in args.examples]
    results = compare_examples(selected, workdir=args.workdir)

    failed = False
    for result in results:
        ok = result["evaluator_ok"] and result["reference_ok"]
        logger.info(
            "%s: %s",
            result["example"],
            "PASS" if ok else "FAIL",
        )
        failed = failed or not ok
        if not ok:
            logger.info("program: %s", result["program_path"])
            logger.info("reference: %s", result["ref_path"])
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

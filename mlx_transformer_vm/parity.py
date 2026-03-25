"""Helpers for comparing this port against the upstream repository."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
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


def has_upstream_weighted_runtime():
    root = upstream_root()
    try:
        result = subprocess.run(
            ["uv", "run", "--python", "3.11", "python", "-c", "import torch, yaml, pulp"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


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


def _run_upstream_weighted(program_infos, workdir):
    """Run upstream weighted execution in the upstream uv environment."""

    root = upstream_root()
    build_dir = Path(workdir) / ".upstream-weighted"
    build_dir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        [
            {
                "program_path": str(info["program_path"]),
                "ref_path": str(info["ref_path"]),
            }
            for info in program_infos
        ]
    )
    script = """
import json
import os
import sys

import torch

ROOT = sys.argv[1]
BUILD_DIR = sys.argv[2]
sys.path.insert(0, ROOT)
os.chdir(BUILD_DIR)

from transformer_vm.attention import StandardKVCache
from transformer_vm.model.weights import build_model

programs = json.loads(sys.stdin.read())
model, all_tokens, tok_to_idx_map, _ = build_model(plan_path=None)

results = {}
for program in programs:
    with open(program["program_path"]) as handle:
        tokens = handle.read().split()
    max_new_tokens = 50000
    ref_path = program.get("ref_path")
    if ref_path and os.path.exists(ref_path):
        with open(ref_path) as handle:
            ref_tokens = handle.read().split()
        max_new_tokens = max(len(ref_tokens) - len(tokens) + 4, 1)
    idx = torch.tensor([[tok_to_idx_map[token] for token in tokens]], dtype=torch.long)
    out = model.generate_with_cache(
        idx,
        max_new_tokens=max_new_tokens,
        cache_class=StandardKVCache,
    )
    results[program["program_path"]] = [all_tokens[i] for i in out[0].tolist()]

print(json.dumps(results))
""".strip()
    result = subprocess.run(
        ["uv", "run", "--python", "3.11", "python", "-c", script, str(root), str(build_dir)],
        cwd=root,
        input=payload,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "upstream weighted runtime failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def compare_examples(examples=None, workdir=None, include_weighted=True):
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
        compiled = []
        for example, args in examples:
            logger.info("Checking %s", example)
            program_path, ref_path = compile_example(example, args, out_dir)
            compiled.append(
                {
                    "example": example,
                    "program_path": program_path,
                    "ref_path": ref_path,
                }
            )

        weighted_traces = {}
        if include_weighted and has_upstream_weighted_runtime():
            weighted_traces = _run_upstream_weighted(compiled, out_dir)

        results = []
        for item in compiled:
            with open(item["program_path"]) as handle:
                tokens = handle.read().split()
            predicted = generate_trace(tokens)
            with open(item["ref_path"]) as handle:
                expected = handle.read().split()

            local_ref = item["program_path"].with_name(item["program_path"].stem + "_local_ref.txt")
            generate_ref(str(item["program_path"]), str(local_ref))
            with open(local_ref) as handle:
                local_ref_tokens = handle.read().split()

            weighted_trace = weighted_traces.get(str(item["program_path"]))
            results.append(
                {
                    "example": item["example"],
                    "program_path": item["program_path"],
                    "ref_path": item["ref_path"],
                    "evaluator_ok": predicted == expected,
                    "reference_ok": local_ref_tokens == expected,
                    "weighted_available": weighted_trace is not None,
                    "weighted_ok": weighted_trace == expected if weighted_trace is not None else None,
                    "predicted": predicted,
                    "expected": expected,
                    "local_ref": local_ref_tokens,
                    "weighted": weighted_trace,
                }
            )
        return results
    finally:
        if workdir_obj is not None:
            workdir_obj.cleanup()


def compare_example(example, args, out_dir, include_weighted=True):
    return compare_examples([(example, args)], workdir=out_dir, include_weighted=include_weighted)[0]


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
    parser.add_argument(
        "--no-weighted",
        action="store_true",
        help="Skip the upstream weighted-runtime comparison",
    )
    args = parser.parse_args()

    selected = [(name, DEFAULT_EXAMPLES[name]) for name in args.examples]
    results = compare_examples(selected, workdir=args.workdir, include_weighted=not args.no_weighted)

    failed = False
    for result in results:
        ok = result["evaluator_ok"] and result["reference_ok"]
        if result["weighted_available"]:
            ok = ok and result["weighted_ok"]
        logger.info("%s: %s", result["example"], "PASS" if ok else "FAIL")
        failed = failed or not ok
        if not ok:
            logger.info("program: %s", result["program_path"])
            logger.info("reference: %s", result["ref_path"])
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

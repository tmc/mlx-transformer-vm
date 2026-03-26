#!/usr/bin/env python3
"""Compile C/WASM programs to token-prefix format.

Usage:
    python -m mlx_transformer_vm.compilation.compile_wasm examples/hello.c
    python -m mlx_transformer_vm.compilation.compile_wasm examples/collatz.c --args 7
    python -m mlx_transformer_vm.compilation.compile_wasm examples/hello.wasm -o data/hello

Outputs:
    <name>.txt       — full token prefix with { program } input (universal model input)
    <name>_spec.txt  — start + input tokens (specialized model input)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path

from .decoder import (
    OP_BLOCK,
    OP_BR,
    OP_BR_IF,
    OP_CALL,
    OP_DROP,
    OP_ELSE,
    OP_END,
    OP_GLOBAL_GET,
    OP_GLOBAL_SET,
    OP_I32_ADD,
    OP_I32_CONST,
    OP_I32_EQ,
    OP_I32_EQZ,
    OP_I32_GE_S,
    OP_I32_GE_U,
    OP_I32_GT_S,
    OP_I32_GT_U,
    OP_I32_LE_S,
    OP_I32_LE_U,
    OP_I32_LOAD,
    OP_I32_LOAD8_S,
    OP_I32_LOAD8_U,
    OP_I32_LOAD16_S,
    OP_I32_LOAD16_U,
    OP_I32_LT_S,
    OP_I32_LT_U,
    OP_I32_NE,
    OP_I32_STORE,
    OP_I32_STORE8,
    OP_I32_STORE16,
    OP_I32_SUB,
    OP_IF,
    OP_LOCAL_GET,
    OP_LOCAL_SET,
    OP_LOCAL_TEE,
    OP_LOOP,
    OP_NOP,
    OP_RETURN,
    OP_SELECT,
    OP_UNREACHABLE,
    FuncBody,
    WasmModule,
    decode,
)
from .lower import lower_hard_ops

logger = logging.getLogger(__name__)

MASK32 = 0xFFFFFFFF
GLOBAL_BASE = 8  # memory address for globals (shared across call depths)

DEFAULT_EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2]
    / os.environ.get("MLX_TRANSFORMER_VM_EXAMPLES_DIR", "examples")
)
DEFAULT_MANIFEST = Path(
    os.environ.get("MLX_TRANSFORMER_VM_MANIFEST", DEFAULT_EXAMPLES_ROOT / "manifest.yaml")
)

WASM_TO_NAME = {
    OP_I32_CONST: "i32.const",
    OP_LOCAL_GET: "local.get",
    OP_LOCAL_SET: "local.set",
    OP_LOCAL_TEE: "local.tee",
    OP_DROP: "drop",
    OP_SELECT: "select",
    OP_I32_ADD: "i32.add",
    OP_I32_SUB: "i32.sub",
    OP_I32_EQ: "i32.eq",
    OP_I32_NE: "i32.ne",
    OP_I32_LT_S: "i32.lt_s",
    OP_I32_LT_U: "i32.lt_u",
    OP_I32_GT_S: "i32.gt_s",
    OP_I32_GT_U: "i32.gt_u",
    OP_I32_LE_S: "i32.le_s",
    OP_I32_LE_U: "i32.le_u",
    OP_I32_GE_S: "i32.ge_s",
    OP_I32_GE_U: "i32.ge_u",
    OP_I32_EQZ: "i32.eqz",
    OP_I32_LOAD: "i32.load",
    OP_I32_LOAD8_S: "i32.load8_s",
    OP_I32_LOAD8_U: "i32.load8_u",
    OP_I32_LOAD16_S: "i32.load16_s",
    OP_I32_LOAD16_U: "i32.load16_u",
    OP_I32_STORE: "i32.store",
    OP_I32_STORE8: "i32.store8",
    OP_I32_STORE16: "i32.store16",
    OP_BR: "br",
    OP_BR_IF: "br_if",
    OP_UNREACHABLE: "halt",
    OP_RETURN: "halt",
}


# ── C -> WASM compilation ──────────────────────────────────────────


def find_clang() -> str:
    """Find a clang binary with wasm32 target support.

    Resolution order:
      1. ``CLANG_PATH`` environment variable (explicit override)
      2. ``clang`` on ``$PATH`` (via ``shutil.which``)
      3. Platform-specific fallback locations (Homebrew on macOS, common Linux paths)
    """
    env_path = os.environ.get("CLANG_PATH")
    candidates: list[str] = []
    if env_path:
        candidates.append(env_path)

    which_clang = shutil.which("clang")
    if which_clang:
        candidates.append(which_clang)

    candidates.extend([
        # macOS (Homebrew)
        "/opt/homebrew/opt/llvm/bin/clang",
        "/usr/local/opt/llvm/bin/clang",
        # Linux
        "/usr/lib/llvm-18/bin/clang",
        "/usr/lib/llvm-17/bin/clang",
        "/usr/lib/llvm-16/bin/clang",
        "/usr/bin/clang",
    ])

    for cc in candidates:
        try:
            out = subprocess.check_output(
                [cc, "--print-targets"], stderr=subprocess.DEVNULL, text=True
            )
            if "wasm32" in out:
                return cc
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError(
        "No clang with wasm32 target found. "
        "Install LLVM with wasm32 support or set the CLANG_PATH environment variable."
    )


def compile_c_to_wasm(c_path: str) -> str:
    wasm_path = c_path.rsplit(".", 1)[0] + ".wasm"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_h = os.path.join(script_dir, "runtime.h")
    if not os.path.exists(runtime_h):
        raise FileNotFoundError(f"runtime.h not found at {runtime_h}")
    cc = find_clang()
    cmd = [
        cc,
        "--target=wasm32",
        "-nostdlib",
        "-O2",
        "-fno-builtin",
        "-fno-jump-tables",
        "-mllvm",
        "--combiner-store-merging=false",
        "-Wl,--no-entry",
        "-Wl,--export=compute",
        "-Wl,--export=__heap_base",
        "-Wl,-z,stack-size=4096",
        "-Wl,--initial-memory=10485760",
        f"-include{runtime_h}",
        "-o",
        wasm_path,
        c_path,
    ]
    logger.info("Compiling %s -> %s", c_path, wasm_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"clang failed (exit {result.returncode}) compiling {c_path}:\n"
            f"{result.stderr.strip()}"
        )
    return wasm_path


# ── WASM -> dispatch table ──────────────────────────────────────────


def int_to_bytes(v):
    v = v & MASK32
    return [(v >> (8 * i)) & 0xFF for i in range(4)]


def compile_function(
    func: FuncBody,
    mod: WasmModule,
    local_func_idx: int = 0,
    is_main: bool = True,
    global_temp_local: int | None = None,
) -> list[tuple[str, list[int]]]:
    """Compile WASM instructions to a flat dispatch table."""
    import_map: dict[int, str] = {}
    num_imports = 0
    for imp in mod.imports:
        if imp.kind == 0:
            import_map[num_imports] = imp.name
            num_imports += 1

    entries: list[tuple[str, int]] = []
    label_stack: list[dict] = []
    PLACEHOLDER = 0xDEAD

    for instr in func.instructions:
        op = instr.opcode

        if op == OP_NOP:
            continue

        if op == OP_BLOCK:
            label_stack.append(
                {
                    "kind": "block",
                    "start_pc": len(entries),
                    "end_pc": None,
                    "patches": [],
                }
            )
            continue

        if op == OP_LOOP:
            label_stack.append(
                {
                    "kind": "loop",
                    "start_pc": len(entries),
                    "end_pc": None,
                    "patches": [],
                }
            )
            continue

        if op == OP_IF:
            entries.append(("i32.eqz", 0))
            br_idx = len(entries)
            entries.append(("br_if", PLACEHOLDER))
            label_stack.append(
                {
                    "kind": "if",
                    "start_pc": len(entries) - 2,
                    "end_pc": None,
                    "else_pc": None,
                    "patches": [br_idx],
                    "if_entry": br_idx,
                }
            )
            continue

        if op == OP_ELSE:
            frame = label_stack[-1]
            assert frame["kind"] == "if"
            else_br_idx = len(entries)
            entries.append(("br", PLACEHOLDER))
            frame["else_pc"] = len(entries)
            frame["patches"] = [else_br_idx]
            entries[frame["if_entry"]] = ("br_if", len(entries))
            continue

        if op == OP_END:
            if not label_stack:
                entries.append(("halt" if is_main else "return", 0))
                continue
            frame = label_stack.pop()
            end_pc = len(entries)
            for idx in frame["patches"]:
                name, _ = entries[idx]
                entries[idx] = (name, end_pc)
            continue

        if op == OP_BR:
            label_idx = instr.immediates[0]
            target_frame = label_stack[-(label_idx + 1)]
            if target_frame["kind"] == "loop":
                entries.append(("br", target_frame["start_pc"]))
            else:
                idx = len(entries)
                entries.append(("br", PLACEHOLDER))
                target_frame["patches"].append(idx)
            continue

        if op == OP_BR_IF:
            label_idx = instr.immediates[0]
            target_frame = label_stack[-(label_idx + 1)]
            if target_frame["kind"] == "loop":
                entries.append(("br_if", target_frame["start_pc"]))
            else:
                idx = len(entries)
                entries.append(("br_if", PLACEHOLDER))
                target_frame["patches"].append(idx)
            continue

        if op == OP_RETURN:
            entries.append(("halt" if is_main else "return", 0))
            continue
        if op == OP_UNREACHABLE:
            entries.append(("halt", 0))
            continue

        if op == OP_CALL:
            fi = instr.immediates[0]
            if fi in import_map and import_map[fi] == "output_byte":
                entries.append(("output", 0))
            elif fi >= num_imports:
                entries.append(("call", fi))
            else:
                raise ValueError(f"Unsupported CALL to import {fi}")
            continue

        if op == OP_GLOBAL_GET:
            gidx = instr.immediates[0]
            entries.append(("i32.const", GLOBAL_BASE + 4 * gidx))
            entries.append(("i32.load", 0))
            continue
        if op == OP_GLOBAL_SET:
            gidx = instr.immediates[0]
            assert global_temp_local is not None, "global.set requires a temp local"
            entries.append(("local.set", global_temp_local))
            entries.append(("i32.const", GLOBAL_BASE + 4 * gidx))
            entries.append(("local.get", global_temp_local))
            entries.append(("i32.store", 0))
            continue

        if op in (OP_LOCAL_GET, OP_LOCAL_SET, OP_LOCAL_TEE):
            name = WASM_TO_NAME[op]
            entries.append((name, instr.immediates[0]))
            continue

        if op == OP_I32_CONST:
            entries.append(("i32.const", instr.immediates[0] & MASK32))
            continue

        if op in (
            OP_I32_LOAD,
            OP_I32_LOAD8_S,
            OP_I32_LOAD8_U,
            OP_I32_LOAD16_S,
            OP_I32_LOAD16_U,
            OP_I32_STORE,
            OP_I32_STORE8,
            OP_I32_STORE16,
        ):
            name = WASM_TO_NAME[op]
            offset = instr.immediates[1]
            entries.append((name, offset))
            continue

        if op in WASM_TO_NAME:
            entries.append((WASM_TO_NAME[op], 0))
            continue

        from .decoder import WASM_OP_NAMES

        name = WASM_OP_NAMES.get(op, f"0x{op:02x}")
        raise ValueError(f"Unsupported wasm opcode: {name}")

    return [(name, int_to_bytes(imm)) for name, imm in entries]


def _adjust_branches(body, offset):
    """Adjust BR/BR_IF targets in a compiled function body by offset."""
    result = []
    for op, imm_bytes in body:
        if op in ("br", "br_if"):
            target = sum(imm_bytes[j] * (256**j) for j in range(4))
            result.append((op, int_to_bytes(target + offset)))
        else:
            result.append((op, imm_bytes))
    return result


def _func_global_temp(func: FuncBody, mod: WasmModule, local_func_idx: int) -> int | None:
    """Return the temp local index for global.set, or None if not needed."""
    uses_global_set = any(ins.opcode == OP_GLOBAL_SET for ins in func.instructions)
    if not uses_global_set:
        return None
    type_idx = mod.func_type_indices[local_func_idx]
    param_count = len(mod.types[type_idx].params)
    return param_count + func.num_locals


def _compute_input_base(mod: WasmModule) -> int:
    """Compute a safe address for input data, using __heap_base from the linker."""
    for exp in mod.exports:
        if exp.name == "__heap_base" and exp.kind == 3:
            return (mod.globals[exp.index]["init"] + 15) & ~15
    raise ValueError("__heap_base not exported — add -Wl,--export=__heap_base to linker flags")


def build_program(mod: WasmModule) -> tuple[list[tuple[str, list[int]]], int]:
    """Build the full dispatch table including local initialization prologue.

    Returns (program, input_base).
    Input data is NOT baked in — it's provided at runtime via input_base.
    """
    num_imports = sum(1 for imp in mod.imports if imp.kind == 0)

    main_local_idx = 0
    for exp in mod.exports:
        if exp.kind == 0 and exp.name == "compute":
            main_local_idx = exp.index - num_imports
            break

    func = mod.functions[main_local_idx]
    type_idx = mod.func_type_indices[main_local_idx]
    param_count = len(mod.types[type_idx].params)

    num_locals = param_count + func.num_locals

    input_base = _compute_input_base(mod) if param_count > 0 else 0
    entry_args = [input_base] if param_count > 0 else []

    prologue: list[tuple[str, list[int]]] = []

    # Emit input_base as the first instruction (tells runtime where to load input)
    if param_count > 0:
        prologue.append(("input_base", int_to_bytes(input_base)))

    # Part 1: main function local variable initialization
    for k in reversed(range(num_locals)):
        init_val = 0
        if k < len(entry_args):
            init_val = entry_args[k] & MASK32
        prologue.append(("i32.const", int_to_bytes(init_val)))
        prologue.append(("local.set", int_to_bytes(k)))

    # Part 2: memory initialization (data segments + globals only, NO input)
    initial_memory: dict[int, int] = {}

    used_globals: set[int] = set()
    for fn in mod.functions:
        for ins in fn.instructions:
            if ins.opcode in (OP_GLOBAL_GET, OP_GLOBAL_SET):
                used_globals.add(ins.immediates[0])

    for gidx in range(len(mod.globals)):
        if gidx not in used_globals:
            continue
        gval = mod.globals[gidx]["init"] & MASK32
        addr = GLOBAL_BASE + 4 * gidx
        for b in range(4):
            initial_memory[addr + b] = (gval >> (8 * b)) & 0xFF

    for seg in mod.data_segments:
        for i, byte_val in enumerate(seg.data):
            initial_memory[seg.offset + i] = byte_val

    for addr in sorted(initial_memory):
        byte_val = initial_memory[addr]
        if byte_val == 0:
            continue  # memory is already zero-initialized
        prologue.append(("i32.const", int_to_bytes(addr)))
        prologue.append(("i32.const", int_to_bytes(byte_val)))
        prologue.append(("i32.store8", int_to_bytes(0)))

    # Part 3: compile main function body
    gt = _func_global_temp(func, mod, main_local_idx)
    body0 = compile_function(
        func, mod, local_func_idx=main_local_idx, is_main=True, global_temp_local=gt
    )
    program = list(prologue) + _adjust_branches(body0, len(prologue))

    # Part 4: compile called functions with parameter prologues
    func_addresses: dict[int, int] = {}
    for fi in range(len(mod.functions)):
        if fi == main_local_idx:
            continue
        func_fi = mod.functions[fi]
        ti = mod.func_type_indices[fi]
        n_params = len(mod.types[ti].params)

        func_start = len(program)
        func_addresses[num_imports + fi] = func_start

        for k in reversed(range(n_params)):
            program.append(("local.set", int_to_bytes(k)))

        gt_fi = _func_global_temp(func_fi, mod, fi)
        body_fi = compile_function(
            func_fi, mod, local_func_idx=fi, is_main=False, global_temp_local=gt_fi
        )
        program.extend(_adjust_branches(body_fi, func_start + n_params))

        for j in range(func_start, len(program)):
            if program[j][0] == "return":
                d_local = j - func_start
                program[j] = ("return", int_to_bytes((~d_local) & MASK32))

    # Part 5: resolve CALL targets (func_idx -> absolute cursor position)
    for j, (op, imm_bytes) in enumerate(program):
        if op == "call":
            fi = sum(imm_bytes[i] * (256**i) for i in range(4))
            if fi not in func_addresses:
                raise ValueError(f"call to unknown function index {fi}")
            program[j] = ("call", int_to_bytes(func_addresses[fi]))

    # Convert branch/call targets from absolute cursor to relative offset
    for i, (op, imm_bytes) in enumerate(program):
        if op in ("br", "br_if", "call"):
            target = sum(imm_bytes[j] * (256**j) for j in range(4))
            offset = target - i - 1
            program[i] = (op, int_to_bytes(offset))

    return program, input_base


def format_prefix(program: list[tuple[str, list[int]]]) -> str:
    """Convert dispatch table to prefix string with { } delimiters."""
    lines = ["{"]
    for op, imm_bytes in program:
        hex_bytes = " ".join(f"{b:02x}" for b in imm_bytes)
        lines.append(f"{op} {hex_bytes}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def format_input_section(input_str: str) -> str:
    """Format input bytes + commit token for appending after the program."""
    data = input_str.encode("utf-8") + b"\x00"
    tokens = []
    for b in data:
        if 0x20 < b < 0x7F and chr(b) not in ("{", "}"):
            tokens.append(chr(b))
        else:
            tokens.append(f"{b:02x}")
    tokens.append("commit(+0,sts=0,bt=0)")
    return " ".join(tokens) + "\n"


def format_spec_input(input_str: str = "") -> str:
    """Format the specialized model input (start + optional input tokens)."""
    tokens = ["start"]
    if input_str:
        data = input_str.encode("utf-8") + b"\x00"
        for b in data:
            if 0x20 < b < 0x7F and chr(b) not in ("{", "}"):
                tokens.append(chr(b))
            else:
                tokens.append(f"{b:02x}")
        tokens.append("commit(+0,sts=0,bt=0)")
    return " ".join(tokens) + "\n"


def compile_wasm_to_prefix(wasm_path: str) -> tuple[str, int]:
    """Full pipeline: WASM -> lower -> prefix string.

    Returns (prefix_string, input_base).
    Program does NOT contain input data — use input_base to load it at runtime.
    """
    with open(wasm_path, "rb") as f:
        mod = decode(f.read())

    for fi, func in enumerate(mod.functions):
        type_idx = mod.func_type_indices[fi]
        num_params = len(mod.types[type_idx].params)
        mod.functions[fi] = lower_hard_ops(func, num_params)

    program, input_base = build_program(mod)
    return format_prefix(program), input_base


# ── Compile one program end-to-end ────────────────────────────────


def compile_program(input_path: str, args_str: str = "", out_base: str | None = None):
    """Compile a C or WASM file to token-prefix format.

    Args:
        input_path: Path to .c or .wasm file.
        args_str: Input string for the program.
        out_base: Output base path (default: next to the input file).
    """
    wasm_path = input_path
    if wasm_path.endswith(".c"):
        wasm_path = compile_c_to_wasm(wasm_path)

    name = os.path.splitext(os.path.basename(wasm_path))[0]
    if out_base is None:
        out_base = os.path.join(os.path.dirname(os.path.abspath(input_path)), name)
    out_dir = os.path.dirname(out_base)

    prefix, input_base = compile_wasm_to_prefix(wasm_path)
    program = prefix.split("\n")
    program_body = [line for line in program if line not in ("{", "}", "")]

    os.makedirs(out_dir, exist_ok=True)

    txt_path = out_base + ".txt"
    with open(txt_path, "w") as f:
        f.write(prefix)
        if args_str and input_base:
            f.write(format_input_section(args_str))

    spec_path = out_base + "_spec.txt"
    with open(spec_path, "w") as f:
        f.write(format_spec_input(args_str if input_base else ""))

    # Clean up intermediate .wasm file
    if input_path.endswith(".c") and os.path.exists(wasm_path):
        os.remove(wasm_path)

    n_instrs = len(program_body)
    logger.info("%s: %d instructions", txt_path, n_instrs)
    if input_base:
        logger.info("  input_base = %d (0x%x)", input_base, input_base)
    return txt_path, spec_path, input_base


# ── Compile all from manifest ────────────────────────────────────


def load_manifest():
    """Load the examples manifest. Returns list of dicts with 'name' and 'args'."""
    import yaml

    if not DEFAULT_MANIFEST.exists():
        raise FileNotFoundError(
            "manifest.yaml not found; set MLX_TRANSFORMER_VM_MANIFEST "
            "or create examples/manifest.yaml"
        )
    with open(DEFAULT_MANIFEST) as f:
        return yaml.safe_load(f)["programs"]


def compile_all():
    """Compile all programs listed in the manifest."""
    manifest = load_manifest()
    for entry in manifest:
        name = entry["name"]
        args_str = entry.get("args", "")
        c_path = os.path.join(DEFAULT_EXAMPLES_ROOT, f"{name}.c")
        if not os.path.exists(c_path):
            logger.warning("Skipping %s: %s not found", name, c_path)
            continue
        compile_program(c_path, args_str)
    logger.info("Compiled %d programs from manifest", len(manifest))


def ensure_data(generate_refs: bool = True):
    """Compile all programs from the manifest and generate reference traces if missing."""
    from mlx_transformer_vm.wasm.reference import generate_ref

    manifest = load_manifest()
    for entry in manifest:
        name = entry["name"]
        txt_path = os.path.join(DEFAULT_EXAMPLES_ROOT, f"{name}.txt")
        if not os.path.exists(txt_path):
            c_path = os.path.join(DEFAULT_EXAMPLES_ROOT, f"{name}.c")
            if os.path.exists(c_path):
                logger.info("Compiling missing program: %s", name)
                compile_program(c_path, entry.get("args", ""))
            else:
                logger.warning("Skipping %s: %s not found", name, c_path)

    if generate_refs:
        for entry in manifest:
            txt_path = os.path.join(DEFAULT_EXAMPLES_ROOT, f"{entry['name']}.txt")
            ref_path = os.path.join(DEFAULT_EXAMPLES_ROOT, f"{entry['name']}_ref.txt")
            if os.path.exists(txt_path) and not os.path.exists(ref_path):
                generate_ref(txt_path, ref_path)


# ── Main ──────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", nargs="?", help="Path to .c or .wasm file")
    parser.add_argument("--args", default="", help="Input string for the program")
    parser.add_argument(
        "--output", "-o", default=None, help="Output base path (default: data/<name>)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Compile all programs from manifest.yaml"
    )
    args = parser.parse_args()

    if args.all:
        compile_all()
        return

    if not args.input:
        parser.error("input is required (or use --all)")

    compile_program(args.input, args.args, args.output)


if __name__ == "__main__":
    main()

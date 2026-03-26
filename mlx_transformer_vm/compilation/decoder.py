# Copyright 2026 Percepta
# Licensed under the Apache License, Version 2.0.
# Obtained from https://github.com/Percepta-Core/transformer-vm
# SPDX-License-Identifier: Apache-2.0

"""Wasm MVP binary decoder.

Parses a .wasm file into structured sections and decoded instruction
lists.  Only the MVP subset is supported (no extensions).

Reference: https://webassembly.github.io/spec/core/binary/
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# ===================================================================== #
#  LEB128 helpers                                                        #
# ===================================================================== #


def _read_unsigned_leb128(data: bytes, pos: int) -> tuple[int, int]:
    """Decode an unsigned LEB128 integer.  Returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
    return result, pos


def _read_signed_leb128(data: bytes, pos: int, bits: int = 32) -> tuple[int, int]:
    """Decode a signed LEB128 integer.  Returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
    # Sign extend
    if shift < bits and (b & 0x40):
        result |= -(1 << shift)
    # Truncate to the specified bit width (two's complement)
    mask = (1 << bits) - 1
    result &= mask
    return result, pos


# ===================================================================== #
#  Wasm types                                                            #
# ===================================================================== #

# Value types
VALTYPE_I32 = 0x7F
VALTYPE_I64 = 0x7E
VALTYPE_F32 = 0x7D
VALTYPE_F64 = 0x7C

# Section IDs
SEC_CUSTOM = 0
SEC_TYPE = 1
SEC_IMPORT = 2
SEC_FUNCTION = 3
SEC_TABLE = 4
SEC_MEMORY = 5
SEC_GLOBAL = 6
SEC_EXPORT = 7
SEC_START = 8
SEC_ELEMENT = 9
SEC_CODE = 10
SEC_DATA = 11
SEC_DATACOUNT = 12


# ===================================================================== #
#  Wasm instruction representation                                       #
# ===================================================================== #


@dataclass
class WasmInstr:
    """A single decoded wasm instruction."""

    opcode: int
    immediates: tuple = ()

    def __repr__(self) -> str:
        name = WASM_OP_NAMES.get(self.opcode, f"0x{self.opcode:02x}")
        if self.immediates:
            args = ", ".join(str(a) for a in self.immediates)
            return f"{name}({args})"
        return name


# Wasm MVP opcode constants (only the ones we need)
OP_UNREACHABLE = 0x00
OP_NOP = 0x01
OP_BLOCK = 0x02
OP_LOOP = 0x03
OP_IF = 0x04
OP_ELSE = 0x05
OP_END = 0x0B
OP_BR = 0x0C
OP_BR_IF = 0x0D
OP_BR_TABLE = 0x0E
OP_RETURN = 0x0F
OP_CALL = 0x10
OP_CALL_INDIRECT = 0x11
OP_DROP = 0x1A
OP_SELECT = 0x1B
OP_LOCAL_GET = 0x20
OP_LOCAL_SET = 0x21
OP_LOCAL_TEE = 0x22
OP_GLOBAL_GET = 0x23
OP_GLOBAL_SET = 0x24

# Memory instructions
OP_I32_LOAD = 0x28
OP_I64_LOAD = 0x29
OP_F32_LOAD = 0x2A
OP_F64_LOAD = 0x2B
OP_I32_LOAD8_S = 0x2C
OP_I32_LOAD8_U = 0x2D
OP_I32_LOAD16_S = 0x2E
OP_I32_LOAD16_U = 0x2F
OP_I64_LOAD8_S = 0x30
OP_I64_LOAD8_U = 0x31
OP_I64_LOAD16_S = 0x32
OP_I64_LOAD16_U = 0x33
OP_I64_LOAD32_S = 0x34
OP_I64_LOAD32_U = 0x35
OP_I32_STORE = 0x36
OP_I64_STORE = 0x37
OP_F32_STORE = 0x38
OP_F64_STORE = 0x39
OP_I32_STORE8 = 0x3A
OP_I32_STORE16 = 0x3B
OP_I64_STORE8 = 0x3C
OP_I64_STORE16 = 0x3D
OP_I64_STORE32 = 0x3E
OP_MEMORY_SIZE = 0x3F
OP_MEMORY_GROW = 0x40

# Constants
OP_I32_CONST = 0x41
OP_I64_CONST = 0x42
OP_F32_CONST = 0x43
OP_F64_CONST = 0x44

# Comparison
OP_I32_EQZ = 0x45
OP_I32_EQ = 0x46
OP_I32_NE = 0x47
OP_I32_LT_S = 0x48
OP_I32_LT_U = 0x49
OP_I32_GT_S = 0x4A
OP_I32_GT_U = 0x4B
OP_I32_LE_S = 0x4C
OP_I32_LE_U = 0x4D
OP_I32_GE_S = 0x4E
OP_I32_GE_U = 0x4F

# Arithmetic / bitwise
OP_I32_CLZ = 0x67
OP_I32_CTZ = 0x68
OP_I32_POPCNT = 0x69
OP_I32_ADD = 0x6A
OP_I32_SUB = 0x6B
OP_I32_MUL = 0x6C
OP_I32_DIV_S = 0x6D
OP_I32_DIV_U = 0x6E
OP_I32_REM_S = 0x6F
OP_I32_REM_U = 0x70
OP_I32_AND = 0x71
OP_I32_OR = 0x72
OP_I32_XOR = 0x73
OP_I32_SHL = 0x74
OP_I32_SHR_S = 0x75
OP_I32_SHR_U = 0x76
OP_I32_ROTL = 0x77
OP_I32_ROTR = 0x78

# Sign-extension (post-MVP but very common)
OP_I32_EXTEND8_S = 0xC0
OP_I32_EXTEND16_S = 0xC1

WASM_OP_NAMES: dict[int, str] = {
    OP_UNREACHABLE: "unreachable",
    OP_NOP: "nop",
    OP_BLOCK: "block",
    OP_LOOP: "loop",
    OP_IF: "if",
    OP_ELSE: "else",
    OP_END: "end",
    OP_BR: "br",
    OP_BR_IF: "br_if",
    OP_BR_TABLE: "br_table",
    OP_RETURN: "return",
    OP_CALL: "call",
    OP_CALL_INDIRECT: "call_indirect",
    OP_DROP: "drop",
    OP_SELECT: "select",
    OP_LOCAL_GET: "local.get",
    OP_LOCAL_SET: "local.set",
    OP_LOCAL_TEE: "local.tee",
    OP_GLOBAL_GET: "global.get",
    OP_GLOBAL_SET: "global.set",
    # Memory
    OP_I32_LOAD: "i32.load",
    OP_I32_LOAD8_S: "i32.load8_s",
    OP_I32_LOAD8_U: "i32.load8_u",
    OP_I32_LOAD16_S: "i32.load16_s",
    OP_I32_LOAD16_U: "i32.load16_u",
    OP_I32_STORE: "i32.store",
    OP_I32_STORE8: "i32.store8",
    OP_I32_STORE16: "i32.store16",
    OP_MEMORY_SIZE: "memory.size",
    OP_MEMORY_GROW: "memory.grow",
    # Constants
    OP_I32_CONST: "i32.const",
    # Comparison
    OP_I32_EQZ: "i32.eqz",
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
    # Arithmetic / bitwise
    OP_I32_CLZ: "i32.clz",
    OP_I32_CTZ: "i32.ctz",
    OP_I32_POPCNT: "i32.popcnt",
    OP_I32_ADD: "i32.add",
    OP_I32_SUB: "i32.sub",
    OP_I32_MUL: "i32.mul",
    OP_I32_DIV_S: "i32.div_s",
    OP_I32_DIV_U: "i32.div_u",
    OP_I32_REM_S: "i32.rem_s",
    OP_I32_REM_U: "i32.rem_u",
    OP_I32_AND: "i32.and",
    OP_I32_OR: "i32.or",
    OP_I32_XOR: "i32.xor",
    OP_I32_SHL: "i32.shl",
    OP_I32_SHR_S: "i32.shr_s",
    OP_I32_SHR_U: "i32.shr_u",
    OP_I32_ROTL: "i32.rotl",
    OP_I32_ROTR: "i32.rotr",
    # Sign-extension
    OP_I32_EXTEND8_S: "i32.extend8_s",
    OP_I32_EXTEND16_S: "i32.extend16_s",
}


# ===================================================================== #
#  Module representation                                                 #
# ===================================================================== #


@dataclass
class FuncType:
    """Wasm function type: params -> results."""

    params: list[int]  # list of VALTYPE_*
    results: list[int]  # list of VALTYPE_*


@dataclass
class Import:
    """Wasm import entry."""

    module: str
    name: str
    kind: int  # 0=func, 1=table, 2=memory, 3=global
    index: int  # type index for func, etc.


@dataclass
class Export:
    """Wasm export entry."""

    name: str
    kind: int  # 0=func, 1=table, 2=memory, 3=global
    index: int


@dataclass
class FuncBody:
    """Decoded function body."""

    locals: list[tuple[int, int]]  # (count, valtype) pairs
    num_locals: int  # total local count (sum of counts)
    instructions: list[WasmInstr]


@dataclass
class DataSegment:
    """Wasm data segment (active, memory 0)."""

    offset: int  # constant offset expression value
    data: bytes


@dataclass
class WasmModule:
    """Parsed wasm module."""

    types: list[FuncType] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    func_type_indices: list[int] = field(default_factory=list)
    exports: list[Export] = field(default_factory=list)
    functions: list[FuncBody] = field(default_factory=list)
    data_segments: list[DataSegment] = field(default_factory=list)
    globals: list[dict] = field(default_factory=list)

    @property
    def num_imported_funcs(self) -> int:
        return sum(1 for imp in self.imports if imp.kind == 0)


# ===================================================================== #
#  Decoder                                                               #
# ===================================================================== #


def decode(data: bytes) -> WasmModule:
    """Decode a wasm binary into a WasmModule."""
    if len(data) < 8:
        raise ValueError("File too short to be a wasm module")
    magic = data[0:4]
    if magic != b"\x00asm":
        raise ValueError(f"Bad magic: {magic.hex()}")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != 1:
        raise ValueError(f"Unsupported wasm version: {version}")

    mod = WasmModule()
    pos = 8

    while pos < len(data):
        section_id = data[pos]
        pos += 1
        section_size, pos = _read_unsigned_leb128(data, pos)
        section_end = pos + section_size

        if section_id == SEC_TYPE:
            _decode_type_section(data, pos, section_end, mod)
        elif section_id == SEC_IMPORT:
            _decode_import_section(data, pos, section_end, mod)
        elif section_id == SEC_FUNCTION:
            _decode_function_section(data, pos, section_end, mod)
        elif section_id == SEC_EXPORT:
            _decode_export_section(data, pos, section_end, mod)
        elif section_id == SEC_CODE:
            _decode_code_section(data, pos, section_end, mod)
        elif section_id == SEC_DATA:
            _decode_data_section(data, pos, section_end, mod)
        elif section_id == SEC_GLOBAL:
            _decode_global_section(data, pos, section_end, mod)
        # Skip other sections (custom, table, memory, start, element, datacount)

        pos = section_end

    return mod


def _decode_type_section(data: bytes, pos: int, end: int, mod: WasmModule):
    count, pos = _read_unsigned_leb128(data, pos)
    for _ in range(count):
        form = data[pos]
        pos += 1
        assert form == 0x60, f"Expected functype 0x60, got 0x{form:02x}"
        # Params
        num_params, pos = _read_unsigned_leb128(data, pos)
        params = []
        for _ in range(num_params):
            params.append(data[pos])
            pos += 1
        # Results
        num_results, pos = _read_unsigned_leb128(data, pos)
        results = []
        for _ in range(num_results):
            results.append(data[pos])
            pos += 1
        mod.types.append(FuncType(params, results))


def _decode_import_section(data: bytes, pos: int, end: int, mod: WasmModule):
    count, pos = _read_unsigned_leb128(data, pos)
    for _ in range(count):
        mod_len, pos = _read_unsigned_leb128(data, pos)
        module_name = data[pos : pos + mod_len].decode("utf-8")
        pos += mod_len
        name_len, pos = _read_unsigned_leb128(data, pos)
        field_name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        kind = data[pos]
        pos += 1
        if kind == 0:  # function
            type_idx, pos = _read_unsigned_leb128(data, pos)
            mod.imports.append(Import(module_name, field_name, kind, type_idx))
        elif kind == 1:  # table
            # elem_type + limits
            pos += 1  # elem_type
            flags = data[pos]
            pos += 1
            _, pos = _read_unsigned_leb128(data, pos)  # min
            if flags & 1:
                _, pos = _read_unsigned_leb128(data, pos)  # max
            mod.imports.append(Import(module_name, field_name, kind, 0))
        elif kind == 2:  # memory
            flags = data[pos]
            pos += 1
            _, pos = _read_unsigned_leb128(data, pos)  # min
            if flags & 1:
                _, pos = _read_unsigned_leb128(data, pos)  # max
            mod.imports.append(Import(module_name, field_name, kind, 0))
        elif kind == 3:  # global
            pos += 1  # valtype
            pos += 1  # mutability
            mod.imports.append(Import(module_name, field_name, kind, 0))


def _decode_function_section(data: bytes, pos: int, end: int, mod: WasmModule):
    count, pos = _read_unsigned_leb128(data, pos)
    for _ in range(count):
        type_idx, pos = _read_unsigned_leb128(data, pos)
        mod.func_type_indices.append(type_idx)


def _decode_export_section(data: bytes, pos: int, end: int, mod: WasmModule):
    count, pos = _read_unsigned_leb128(data, pos)
    for _ in range(count):
        name_len, pos = _read_unsigned_leb128(data, pos)
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        kind = data[pos]
        pos += 1
        index, pos = _read_unsigned_leb128(data, pos)
        mod.exports.append(Export(name, kind, index))


def _decode_global_section(data: bytes, pos: int, end: int, mod: WasmModule):
    count, pos = _read_unsigned_leb128(data, pos)
    for _ in range(count):
        valtype = data[pos]
        pos += 1
        mutability = data[pos]
        pos += 1
        # Decode init expression (simplified: expect i32.const + end)
        init_val = 0
        op = data[pos]
        pos += 1
        if op == OP_I32_CONST:
            init_val, pos = _read_signed_leb128(data, pos, 32)
        # Read end
        while data[pos] != OP_END:
            pos += 1
        pos += 1  # skip END
        mod.globals.append({"valtype": valtype, "mutable": mutability, "init": init_val})


def _decode_code_section(data: bytes, pos: int, end: int, mod: WasmModule):
    count, pos = _read_unsigned_leb128(data, pos)
    for _ in range(count):
        body_size, pos = _read_unsigned_leb128(data, pos)
        body_end = pos + body_size

        # Locals
        num_local_decls, pos = _read_unsigned_leb128(data, pos)
        locals_list = []
        total_locals = 0
        for _ in range(num_local_decls):
            lcount, pos = _read_unsigned_leb128(data, pos)
            ltype = data[pos]
            pos += 1
            locals_list.append((lcount, ltype))
            total_locals += lcount

        # Instructions
        instructions = []
        while pos < body_end:
            instr, pos = _decode_instruction(data, pos)
            instructions.append(instr)

        mod.functions.append(FuncBody(locals_list, total_locals, instructions))


def _decode_data_section(data: bytes, pos: int, end: int, mod: WasmModule):
    count, pos = _read_unsigned_leb128(data, pos)
    for _ in range(count):
        seg_flags, pos = _read_unsigned_leb128(data, pos)
        if seg_flags == 0:
            # Active segment, memory 0, with offset expr
            # Decode offset expression: expect i32.const N, end
            op = data[pos]
            pos += 1
            if op != OP_I32_CONST:
                raise ValueError(f"Expected i32.const in data offset, got 0x{op:02x}")
            offset_val, pos = _read_signed_leb128(data, pos, 32)
            end_op = data[pos]
            pos += 1
            assert end_op == OP_END
            # Data bytes
            byte_count, pos = _read_unsigned_leb128(data, pos)
            seg_data = data[pos : pos + byte_count]
            pos += byte_count
            mod.data_segments.append(DataSegment(offset_val, seg_data))
        else:
            raise ValueError(f"Unsupported data segment flags: {seg_flags}")


def _decode_instruction(data: bytes, pos: int) -> tuple[WasmInstr, int]:
    """Decode a single wasm instruction. Returns (WasmInstr, new_pos)."""
    opcode = data[pos]
    pos += 1

    # No immediates
    if opcode in (
        OP_UNREACHABLE,
        OP_NOP,
        OP_END,
        OP_ELSE,
        OP_RETURN,
        OP_DROP,
        OP_SELECT,
        OP_I32_EQZ,
        OP_I32_EQ,
        OP_I32_NE,
        OP_I32_LT_S,
        OP_I32_LT_U,
        OP_I32_GT_S,
        OP_I32_GT_U,
        OP_I32_LE_S,
        OP_I32_LE_U,
        OP_I32_GE_S,
        OP_I32_GE_U,
        OP_I32_CLZ,
        OP_I32_CTZ,
        OP_I32_POPCNT,
        OP_I32_ADD,
        OP_I32_SUB,
        OP_I32_MUL,
        OP_I32_DIV_S,
        OP_I32_DIV_U,
        OP_I32_REM_S,
        OP_I32_REM_U,
        OP_I32_AND,
        OP_I32_OR,
        OP_I32_XOR,
        OP_I32_SHL,
        OP_I32_SHR_S,
        OP_I32_SHR_U,
        OP_I32_ROTL,
        OP_I32_ROTR,
        OP_I32_EXTEND8_S,
        OP_I32_EXTEND16_S,
    ):
        return WasmInstr(opcode), pos

    # Block type
    if opcode in (OP_BLOCK, OP_LOOP, OP_IF):
        block_type = data[pos]
        pos += 1
        # 0x40 = void, or a valtype
        return WasmInstr(opcode, (block_type,)), pos

    # Branch
    if opcode in (OP_BR, OP_BR_IF):
        label_idx, pos = _read_unsigned_leb128(data, pos)
        return WasmInstr(opcode, (label_idx,)), pos

    # br_table
    if opcode == OP_BR_TABLE:
        num_targets, pos = _read_unsigned_leb128(data, pos)
        targets = []
        for _ in range(num_targets):
            t, pos = _read_unsigned_leb128(data, pos)
            targets.append(t)
        default, pos = _read_unsigned_leb128(data, pos)
        return WasmInstr(opcode, (tuple(targets), default)), pos

    # Call
    if opcode == OP_CALL:
        func_idx, pos = _read_unsigned_leb128(data, pos)
        return WasmInstr(opcode, (func_idx,)), pos

    if opcode == OP_CALL_INDIRECT:
        type_idx, pos = _read_unsigned_leb128(data, pos)
        table_idx = data[pos]
        pos += 1  # always 0x00 in MVP
        return WasmInstr(opcode, (type_idx, table_idx)), pos

    # Local/global variable access
    if opcode in (OP_LOCAL_GET, OP_LOCAL_SET, OP_LOCAL_TEE, OP_GLOBAL_GET, OP_GLOBAL_SET):
        idx, pos = _read_unsigned_leb128(data, pos)
        return WasmInstr(opcode, (idx,)), pos

    # Memory instructions: alignment + offset
    if opcode in (
        OP_I32_LOAD,
        OP_I64_LOAD,
        OP_F32_LOAD,
        OP_F64_LOAD,
        OP_I32_LOAD8_S,
        OP_I32_LOAD8_U,
        OP_I32_LOAD16_S,
        OP_I32_LOAD16_U,
        OP_I64_LOAD8_S,
        OP_I64_LOAD8_U,
        OP_I64_LOAD16_S,
        OP_I64_LOAD16_U,
        OP_I64_LOAD32_S,
        OP_I64_LOAD32_U,
        OP_I32_STORE,
        OP_I64_STORE,
        OP_F32_STORE,
        OP_F64_STORE,
        OP_I32_STORE8,
        OP_I32_STORE16,
        OP_I64_STORE8,
        OP_I64_STORE16,
        OP_I64_STORE32,
    ):
        align, pos = _read_unsigned_leb128(data, pos)
        offset, pos = _read_unsigned_leb128(data, pos)
        return WasmInstr(opcode, (align, offset)), pos

    # memory.size / memory.grow: 1-byte reserved index (always 0x00 in MVP)
    if opcode in (OP_MEMORY_SIZE, OP_MEMORY_GROW):
        reserved = data[pos]
        pos += 1
        return WasmInstr(opcode, (reserved,)), pos

    # f32.const: 4-byte IEEE 754 literal
    if opcode == OP_F32_CONST:
        val = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        return WasmInstr(opcode, (val,)), pos

    # f64.const: 8-byte IEEE 754 literal
    if opcode == OP_F64_CONST:
        val = struct.unpack_from("<d", data, pos)[0]
        pos += 8
        return WasmInstr(opcode, (val,)), pos

    # i32.const
    if opcode == OP_I32_CONST:
        val, pos = _read_signed_leb128(data, pos, 32)
        return WasmInstr(opcode, (val,)), pos

    # i64.const
    if opcode == OP_I64_CONST:
        val, pos = _read_signed_leb128(data, pos, 64)
        return WasmInstr(opcode, (val,)), pos

    raise ValueError(f"Unsupported wasm opcode: 0x{opcode:02x} at position {pos - 1}")

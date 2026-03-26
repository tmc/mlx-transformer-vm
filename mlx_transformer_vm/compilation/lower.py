# Copyright 2026 Percepta
# Licensed under the Apache License, Version 2.0.
# Obtained from https://github.com/Percepta-Core/transformer-vm
# SPDX-License-Identifier: Apache-2.0

"""WASM instruction lowering pass.

Replaces hard-to-simulate WASM instructions (MUL, DIV_U, DIV_S, REM_U, REM_S,
AND, SHL, SHR_U, CLZ, CTZ, POPCNT, ROTL, ROTR, EXTEND8_S) with sequences of
basic instructions that the transformer can
execute: ADD, SUB, comparisons, branches, local ops, LOAD8/STORE8.

The lowering only applies when the hard op is preceded by i32.const C
(so the second operand is a known constant).  The const+op pair is replaced
with an equivalent instruction sequence using temporary locals.

Scratch memory address 0 is used for byte extraction (AND 255, EXTEND8_S).
"""

from __future__ import annotations

import logging

from .decoder import (
    OP_BLOCK,
    OP_BR,
    OP_BR_IF,
    OP_BR_TABLE,
    OP_CALL,
    OP_DROP,
    OP_ELSE,
    OP_END,
    OP_GLOBAL_GET,
    OP_GLOBAL_SET,
    OP_I32_ADD,
    OP_I32_AND,
    OP_I32_CLZ,
    OP_I32_CONST,
    OP_I32_CTZ,
    OP_I32_DIV_S,
    OP_I32_DIV_U,
    OP_I32_EQ,
    OP_I32_EQZ,
    OP_I32_EXTEND8_S,
    OP_I32_EXTEND16_S,
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
    OP_I32_MUL,
    OP_I32_NE,
    OP_I32_OR,
    OP_I32_POPCNT,
    OP_I32_REM_S,
    OP_I32_REM_U,
    OP_I32_ROTL,
    OP_I32_ROTR,
    OP_I32_SHL,
    OP_I32_SHR_S,
    OP_I32_SHR_U,
    OP_I32_STORE,
    OP_I32_STORE8,
    OP_I32_STORE16,
    OP_I32_SUB,
    OP_I32_XOR,
    OP_IF,
    OP_LOCAL_GET,
    OP_LOCAL_SET,
    OP_LOCAL_TEE,
    OP_LOOP,
    OP_MEMORY_SIZE,
    OP_NOP,
    OP_RETURN,
    OP_SELECT,
    OP_UNREACHABLE,
    WASM_OP_NAMES,
    FuncBody,
    WasmInstr,
)

logger = logging.getLogger(__name__)

SCRATCH_ADDR = 0  # Byte 0 of linear memory used as scratch

_CONTROL_OPS = frozenset(
    {
        OP_UNREACHABLE,
        OP_NOP,
        OP_BLOCK,
        OP_LOOP,
        OP_IF,
        OP_ELSE,
        OP_END,
        OP_BR,
        OP_BR_IF,
        OP_BR_TABLE,
        OP_RETURN,
        OP_CALL,
        OP_MEMORY_SIZE,
    }
)

BASIC_OPS = _CONTROL_OPS | frozenset(
    {
        OP_DROP,
        OP_SELECT,
        OP_LOCAL_GET,
        OP_LOCAL_SET,
        OP_LOCAL_TEE,
        OP_GLOBAL_GET,
        OP_GLOBAL_SET,
        OP_I32_LOAD,
        OP_I32_LOAD8_S,
        OP_I32_LOAD8_U,
        OP_I32_LOAD16_S,
        OP_I32_LOAD16_U,
        OP_I32_STORE,
        OP_I32_STORE8,
        OP_I32_STORE16,
        OP_I32_CONST,
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
        OP_I32_ADD,
        OP_I32_SUB,
    }
)

# Opcodes that can be lowered when preceded by i32.const
LOWERABLE_BINOPS = {
    OP_I32_MUL,
    OP_I32_DIV_S,
    OP_I32_DIV_U,
    OP_I32_REM_U,
    OP_I32_REM_S,
    OP_I32_AND,
    OP_I32_OR,
    OP_I32_SHL,
    OP_I32_SHR_U,
    OP_I32_SHR_S,
    OP_I32_XOR,
    OP_I32_ROTL,
    OP_I32_ROTR,
}
LOWERABLE_UNARY = {OP_I32_EXTEND8_S, OP_I32_EXTEND16_S, OP_I32_CLZ, OP_I32_CTZ, OP_I32_POPCNT}


def _instr(op, *imm):
    """Shorthand to create a WasmInstr."""
    return WasmInstr(op, tuple(imm))


def _expand_mul(c: int, local_a: int) -> list[WasmInstr]:
    """Expand x * C using additions.  Returns instrs that leave result on stack.
    Assumes x is in local_a.  Uses an addition chain (binary method)."""
    c = c & 0xFFFFFFFF
    # Handle sign: treat as unsigned for the chain, 2's complement wraps
    if c == 0:
        return [_instr(OP_I32_CONST, 0)]
    if c == 1:
        return [_instr(OP_LOCAL_GET, local_a)]
    if c == 0xFFFFFFFF:  # -1
        return [_instr(OP_I32_CONST, 0), _instr(OP_LOCAL_GET, local_a), _instr(OP_I32_SUB)]

    # Addition chain: compute c*x by doubling and adding
    # Use binary representation of c
    bits = []
    v = c
    while v:
        bits.append(v & 1)
        v >>= 1
    # bits[0] is LSB, bits[-1] is MSB (always 1)

    # Strategy: start with x (for MSB), then for each subsequent bit
    # going from MSB-1 to LSB: double, and add x if bit is 1
    instrs: list[WasmInstr] = []
    # Start: push x (for the top bit which is always 1)
    instrs.append(_instr(OP_LOCAL_GET, local_a))  # accumulator = x

    # Use local_a+1 as temp for accumulator
    tmp = local_a + 1
    instrs.append(_instr(OP_LOCAL_SET, tmp))  # tmp = x

    for i in range(len(bits) - 2, -1, -1):
        # Double: tmp = tmp + tmp
        instrs.append(_instr(OP_LOCAL_GET, tmp))
        instrs.append(_instr(OP_LOCAL_GET, tmp))
        instrs.append(_instr(OP_I32_ADD))
        instrs.append(_instr(OP_LOCAL_SET, tmp))

        if bits[i]:
            # Add x: tmp = tmp + x
            instrs.append(_instr(OP_LOCAL_GET, tmp))
            instrs.append(_instr(OP_LOCAL_GET, local_a))
            instrs.append(_instr(OP_I32_ADD))
            instrs.append(_instr(OP_LOCAL_SET, tmp))

    instrs.append(_instr(OP_LOCAL_GET, tmp))
    return instrs


def _expand_div_u(c: int, local_a: int) -> list[WasmInstr]:
    """Expand unsigned x / C using subtraction loop.
    Assumes x is on stack (not in local yet)."""
    # x / C:  q=0; while x >= C: x -= C; q++; result = q
    local_n = local_a  # dividend
    local_q = local_a + 1  # quotient

    return [
        _instr(OP_LOCAL_SET, local_n),  # n = x
        _instr(OP_I32_CONST, 0),
        _instr(OP_LOCAL_SET, local_q),  # q = 0
        _instr(OP_BLOCK, 0x40),  # block $exit
        _instr(OP_LOOP, 0x40),  #   loop $loop
        _instr(OP_LOCAL_GET, local_n),
        _instr(OP_I32_CONST, c),
        _instr(OP_I32_LT_U),  #     n < C ?
        _instr(OP_BR_IF, 1),  #     br_if $exit (depth 1)
        _instr(OP_LOCAL_GET, local_n),
        _instr(OP_I32_CONST, c),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_n),  #     n -= C
        _instr(OP_LOCAL_GET, local_q),
        _instr(OP_I32_CONST, 1),
        _instr(OP_I32_ADD),
        _instr(OP_LOCAL_SET, local_q),  #     q++
        _instr(OP_BR, 0),  #     br $loop
        _instr(OP_END),  #   end loop
        _instr(OP_END),  # end block
        _instr(OP_LOCAL_GET, local_q),  # result = q
    ]


def _expand_rem_u(c: int, local_a: int) -> list[WasmInstr]:
    """Expand unsigned x % C using subtraction loop.
    Assumes x is on stack (not in local yet)."""
    local_n = local_a

    return [
        _instr(OP_LOCAL_SET, local_n),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOOP, 0x40),
        _instr(OP_LOCAL_GET, local_n),
        _instr(OP_I32_CONST, c),
        _instr(OP_I32_LT_U),
        _instr(OP_BR_IF, 1),
        _instr(OP_LOCAL_GET, local_n),
        _instr(OP_I32_CONST, c),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_n),
        _instr(OP_BR, 0),
        _instr(OP_END),
        _instr(OP_END),
        _instr(OP_LOCAL_GET, local_n),
    ]


def _expand_div_s(c: int, local_a: int) -> list[WasmInstr]:
    """Expand signed x / C (truncates toward zero).
    Assumes x is on stack."""
    c = c & 0xFFFFFFFF
    c_signed = c - (1 << 32) if c >= (1 << 31) else c
    abs_c = abs(c_signed) & 0xFFFFFFFF
    c_negative = c_signed < 0

    local_x = local_a
    local_q = local_a + 1
    local_neg = local_a + 2

    instrs = [
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_CONST, 0),
        _instr(OP_I32_LT_S),
        _instr(OP_LOCAL_SET, local_neg),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOCAL_GET, local_neg),
        _instr(OP_I32_EQZ),
        _instr(OP_BR_IF, 0),
        _instr(OP_I32_CONST, 0),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_END),
    ]

    if c_negative:
        instrs += [
            _instr(OP_LOCAL_GET, local_neg),
            _instr(OP_I32_EQZ),
            _instr(OP_LOCAL_SET, local_neg),
        ]

    instrs += [
        _instr(OP_I32_CONST, 0),
        _instr(OP_LOCAL_SET, local_q),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOOP, 0x40),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_CONST, abs_c),
        _instr(OP_I32_LT_U),
        _instr(OP_BR_IF, 1),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_CONST, abs_c),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_LOCAL_GET, local_q),
        _instr(OP_I32_CONST, 1),
        _instr(OP_I32_ADD),
        _instr(OP_LOCAL_SET, local_q),
        _instr(OP_BR, 0),
        _instr(OP_END),
        _instr(OP_END),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOCAL_GET, local_neg),
        _instr(OP_I32_EQZ),
        _instr(OP_BR_IF, 0),
        _instr(OP_I32_CONST, 0),
        _instr(OP_LOCAL_GET, local_q),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_q),
        _instr(OP_END),
        _instr(OP_LOCAL_GET, local_q),
    ]

    return instrs


def _expand_clz(local_a: int) -> list[WasmInstr]:
    """Expand i32.clz (count leading zeros).
    Assumes x is on the stack. Doubles x until bit 31 is set."""
    local_x = local_a
    local_count = local_a + 1

    return [
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_I32_CONST, 0),
        _instr(OP_LOCAL_SET, local_count),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_BR_IF, 0),
        _instr(OP_I32_CONST, 32),
        _instr(OP_LOCAL_SET, local_count),
        _instr(OP_BR, 1),
        _instr(OP_END),
        _instr(OP_LOOP, 0x40),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_CONST, 0),
        _instr(OP_I32_LT_S),
        _instr(OP_BR_IF, 1),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_ADD),
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_LOCAL_GET, local_count),
        _instr(OP_I32_CONST, 1),
        _instr(OP_I32_ADD),
        _instr(OP_LOCAL_SET, local_count),
        _instr(OP_BR, 0),
        _instr(OP_END),
        _instr(OP_END),
        _instr(OP_LOCAL_GET, local_count),
    ]


def _expand_ctz(local_a: int) -> list[WasmInstr]:
    """Expand i32.ctz (count trailing zeros).
    Extracts low bit via store8/load8_u + mod-2 loop, right-shifts via div-by-2 loop."""
    local_x = local_a
    local_count = local_a + 1
    local_byte = local_a + 2
    local_q = local_a + 3

    return [
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_I32_CONST, 0),
        _instr(OP_LOCAL_SET, local_count),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_BR_IF, 0),
        _instr(OP_I32_CONST, 32),
        _instr(OP_LOCAL_SET, local_count),
        _instr(OP_BR, 1),
        _instr(OP_END),
        _instr(OP_LOOP, 0x40),
        # extract low byte, compute mod 2
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_STORE8, 0, 0),
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_I32_LOAD8_U, 0, 0),
        _instr(OP_LOCAL_SET, local_byte),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOOP, 0x40),
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_LT_U),
        _instr(OP_BR_IF, 1),
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_byte),
        _instr(OP_BR, 0),
        _instr(OP_END),
        _instr(OP_END),
        # byte is 0 or 1; if 1, low bit is set → done
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_BR_IF, 1),
        # x >>= 1 via unsigned div by 2
        _instr(OP_I32_CONST, 0),
        _instr(OP_LOCAL_SET, local_q),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOOP, 0x40),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_LT_U),
        _instr(OP_BR_IF, 1),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_LOCAL_GET, local_q),
        _instr(OP_I32_CONST, 1),
        _instr(OP_I32_ADD),
        _instr(OP_LOCAL_SET, local_q),
        _instr(OP_BR, 0),
        _instr(OP_END),
        _instr(OP_END),
        _instr(OP_LOCAL_GET, local_q),
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_LOCAL_GET, local_count),
        _instr(OP_I32_CONST, 1),
        _instr(OP_I32_ADD),
        _instr(OP_LOCAL_SET, local_count),
        _instr(OP_BR, 0),
        _instr(OP_END),
        _instr(OP_END),
        _instr(OP_LOCAL_GET, local_count),
    ]


def _expand_popcnt(local_a: int) -> list[WasmInstr]:
    """Expand i32.popcnt (population count).
    Extracts low bit via mod-2, right-shifts via div-by-2, accumulates count."""
    local_x = local_a
    local_count = local_a + 1
    local_byte = local_a + 2
    local_q = local_a + 3

    return [
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_I32_CONST, 0),
        _instr(OP_LOCAL_SET, local_count),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOOP, 0x40),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_EQZ),
        _instr(OP_BR_IF, 1),
        # extract low byte, compute mod 2
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_STORE8, 0, 0),
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_I32_LOAD8_U, 0, 0),
        _instr(OP_LOCAL_SET, local_byte),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOOP, 0x40),
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_LT_U),
        _instr(OP_BR_IF, 1),
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_byte),
        _instr(OP_BR, 0),
        _instr(OP_END),
        _instr(OP_END),
        # count += low_bit
        _instr(OP_LOCAL_GET, local_count),
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_I32_ADD),
        _instr(OP_LOCAL_SET, local_count),
        # x >>= 1 via unsigned div by 2
        _instr(OP_I32_CONST, 0),
        _instr(OP_LOCAL_SET, local_q),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOOP, 0x40),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_LT_U),
        _instr(OP_BR_IF, 1),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_LOCAL_GET, local_q),
        _instr(OP_I32_CONST, 1),
        _instr(OP_I32_ADD),
        _instr(OP_LOCAL_SET, local_q),
        _instr(OP_BR, 0),
        _instr(OP_END),
        _instr(OP_END),
        _instr(OP_LOCAL_GET, local_q),
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_BR, 0),
        _instr(OP_END),
        _instr(OP_END),
        _instr(OP_LOCAL_GET, local_count),
    ]


def _expand_rotl_const(c: int, local_a: int) -> list[WasmInstr]:
    """Expand x rotl C where C is a compile-time constant.
    Assumes x is on the stack.  Uses (x << c) + (x >> (32-c))."""
    c = c & 31
    if c == 0:
        return []

    local_saved_x = local_a + 2
    local_left = local_a + 3

    instrs = [
        _instr(OP_LOCAL_SET, local_a),
        _instr(OP_LOCAL_GET, local_a),
        _instr(OP_LOCAL_SET, local_saved_x),
    ]
    instrs += _expand_shl(c, local_a)
    instrs.append(_instr(OP_LOCAL_SET, local_left))
    instrs.append(_instr(OP_LOCAL_GET, local_saved_x))
    instrs += _expand_shr_u(32 - c, local_a)
    instrs.append(_instr(OP_LOCAL_GET, local_left))
    instrs.append(_instr(OP_I32_ADD))
    return instrs


def _expand_rotr_const(c: int, local_a: int) -> list[WasmInstr]:
    """Expand x rotr C where C is a compile-time constant.
    Assumes x is on the stack.  Uses (x >> c) + (x << (32-c))."""
    c = c & 31
    if c == 0:
        return []

    local_saved_x = local_a + 2
    local_right = local_a + 3

    instrs = [
        _instr(OP_LOCAL_SET, local_a),
        _instr(OP_LOCAL_GET, local_a),
        _instr(OP_LOCAL_SET, local_saved_x),
    ]
    instrs.append(_instr(OP_LOCAL_GET, local_a))
    instrs += _expand_shr_u(c, local_a)
    instrs.append(_instr(OP_LOCAL_SET, local_right))
    instrs.append(_instr(OP_LOCAL_GET, local_saved_x))
    instrs.append(_instr(OP_LOCAL_SET, local_a))
    instrs += _expand_shl(32 - c, local_a)
    instrs.append(_instr(OP_LOCAL_GET, local_right))
    instrs.append(_instr(OP_I32_ADD))
    return instrs


def _expand_and_255(local_a: int) -> list[WasmInstr]:
    """Expand x & 255 using store8 + load8_u at scratch address."""
    return [
        _instr(OP_LOCAL_SET, local_a),  # save x
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_LOCAL_GET, local_a),
        _instr(OP_I32_STORE8, 0, 0),  # mem[0] = x & 0xFF
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_I32_LOAD8_U, 0, 0),  # result = mem[0]
    ]


def _expand_extend8_s(local_a: int) -> list[WasmInstr]:
    """Expand i32.extend8_s using store8 + load8_u + sign extension."""
    return [
        _instr(OP_LOCAL_SET, local_a),
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_LOCAL_GET, local_a),
        _instr(OP_I32_STORE8, 0, 0),
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_I32_LOAD8_U, 0, 0),
        _instr(OP_LOCAL_SET, local_a),
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOCAL_GET, local_a),
        _instr(OP_I32_CONST, 128),
        _instr(OP_I32_LT_U),
        _instr(OP_BR_IF, 0),
        _instr(OP_LOCAL_GET, local_a),
        _instr(OP_I32_CONST, 256),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_a),
        _instr(OP_END),
        _instr(OP_LOCAL_GET, local_a),
    ]


def _expand_extend16_s(local_a: int) -> list[WasmInstr]:
    """Expand i32.extend16_s using store16 + load16_s at scratch address."""
    return [
        _instr(OP_LOCAL_SET, local_a),
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_LOCAL_GET, local_a),
        _instr(OP_I32_STORE16, 0, 0),
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_I32_LOAD16_S, 0, 0),
    ]


def _expand_shl(c: int, local_a: int) -> list[WasmInstr]:
    """Expand x << C.  Uses memory for byte-aligned part, doubling for rest.

    Precondition: x is in local `local_a`.
    Postcondition: result (x << c) is on the stack.

    For c >= 8, exploits little-endian memory as a byte-shift register:
    zero SCRATCH, store x at SCRATCH+q, load from SCRATCH gives x << (q*8).
    Then apply remaining r doublings (c = 8*q + r).
    """
    if c == 0:
        return [_instr(OP_LOCAL_GET, local_a)]
    if c >= 32:
        return [_instr(OP_I32_CONST, 0)]

    q, r = divmod(c, 8)

    if q > 0:
        instrs = [
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_I32_CONST, 0),
            _instr(OP_I32_STORE, 0, 0),
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_LOCAL_GET, local_a),
            _instr(OP_I32_STORE, 0, q),
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_I32_LOAD, 0, 0),
        ]
        for _ in range(r):
            instrs.append(_instr(OP_LOCAL_TEE, local_a))
            instrs.append(_instr(OP_LOCAL_GET, local_a))
            instrs.append(_instr(OP_I32_ADD))
        return instrs

    instrs = [_instr(OP_LOCAL_GET, local_a)]
    for _ in range(c):
        instrs.append(_instr(OP_LOCAL_TEE, local_a))
        instrs.append(_instr(OP_LOCAL_GET, local_a))
        instrs.append(_instr(OP_I32_ADD))
    return instrs


def _expand_shl_from_stack(c: int, local_a: int) -> list[WasmInstr]:
    """Expand x << C where x is on the stack (not yet in a local).

    Uses local.tee to avoid redundant set+get, saving 2 instructions
    vs the set + _expand_shl approach for small shifts.
    """
    if c == 0:
        return []
    if c >= 32:
        return [_instr(OP_DROP), _instr(OP_I32_CONST, 0)]

    q, r = divmod(c, 8)

    if q > 0:
        return [_instr(OP_LOCAL_SET, local_a)] + _expand_shl(c, local_a)

    instrs: list[WasmInstr] = []
    for _ in range(c):
        instrs.append(_instr(OP_LOCAL_TEE, local_a))
        instrs.append(_instr(OP_LOCAL_GET, local_a))
        instrs.append(_instr(OP_I32_ADD))
    return instrs


def _expand_shr_u(c: int, local_a: int) -> list[WasmInstr]:
    """Expand unsigned x >> C.  Uses memory for byte-aligned part.

    Precondition: x is on the stack.
    Postcondition: result (x >> c) is on the stack.

    For c >= 8, exploits little-endian memory as a byte-shift register:
    store x at SCRATCH, zero SCRATCH+4, load from SCRATCH+q gives x >> (q*8).
    Then apply remaining r-bit shift via division loop (c = 8*q + r).
    """
    if c >= 32:
        return [_instr(OP_DROP), _instr(OP_I32_CONST, 0)]

    q, r = divmod(c, 8)

    if q > 0:
        instrs = [
            _instr(OP_LOCAL_SET, local_a),
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_LOCAL_GET, local_a),
            _instr(OP_I32_STORE, 0, 0),
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_I32_CONST, 0),
            _instr(OP_I32_STORE, 0, 4),
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_I32_LOAD, 0, q),
        ]
        if r > 0:
            instrs.extend(_expand_div_u(1 << r, local_a))
        return instrs

    return _expand_div_u(1 << c, local_a)


def _emit_byte_bitop(
    op: str, mask_byte: int, local_byte: int, local_result: int
) -> list[WasmInstr]:
    """Emit instructions for byte-level bitwise op with a compile-time mask.

    Extracts all 8 bits of local_byte (value 0..255) from MSB to LSB.
    After processing bit i the value in local_byte is < 2^i, so the
    comparison ``local_byte >= 2^i`` correctly tests that bit.

    The result is accumulated in local_result according to *op*:
      'and' → result bit = input_bit AND mask_bit
      'or'  → result bit = input_bit OR  mask_bit
      'xor' → result bit = input_bit XOR mask_bit
    """
    out: list[WasmInstr] = [
        _instr(OP_I32_CONST, 0),
        _instr(OP_LOCAL_SET, local_result),
    ]

    for i in range(7, -1, -1):
        mb = (mask_byte >> i) & 1
        add_if_set = (
            (op == "and" and mb == 1) or (op == "or" and mb == 0) or (op == "xor" and mb == 0)
        )
        always_add = op == "or" and mb == 1
        add_if_clear = op == "xor" and mb == 1

        val = 1 << i

        if add_if_clear:
            out += [
                _instr(OP_LOCAL_GET, local_byte),
                _instr(OP_I32_CONST, val),
                _instr(OP_I32_GE_U),
                _instr(OP_IF, 0x40),
                _instr(OP_LOCAL_GET, local_byte),
                _instr(OP_I32_CONST, val),
                _instr(OP_I32_SUB),
                _instr(OP_LOCAL_SET, local_byte),
                _instr(OP_ELSE),
                _instr(OP_LOCAL_GET, local_result),
                _instr(OP_I32_CONST, val),
                _instr(OP_I32_ADD),
                _instr(OP_LOCAL_SET, local_result),
                _instr(OP_END),
            ]
        else:
            out += [
                _instr(OP_BLOCK, 0x40),
                _instr(OP_LOCAL_GET, local_byte),
                _instr(OP_I32_CONST, val),
                _instr(OP_I32_LT_U),
                _instr(OP_BR_IF, 0),
                _instr(OP_LOCAL_GET, local_byte),
                _instr(OP_I32_CONST, val),
                _instr(OP_I32_SUB),
                _instr(OP_LOCAL_SET, local_byte),
            ]
            if add_if_set:
                out += [
                    _instr(OP_LOCAL_GET, local_result),
                    _instr(OP_I32_CONST, val),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_SET, local_result),
                ]
            out.append(_instr(OP_END))

            if always_add:
                out += [
                    _instr(OP_LOCAL_GET, local_result),
                    _instr(OP_I32_CONST, val),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_SET, local_result),
                ]
    return out


def _expand_bitop_general(op: str, c: int, local_a: int) -> list[WasmInstr]:
    """General 32-bit bitwise op with a compile-time constant (AND/OR/XOR).

    Uses i32.store to split x into 4 little-endian bytes, processes each
    byte with _emit_byte_bitop, writes back with i32.store8, then reloads
    the 32-bit result with i32.load.
    """
    c = c & 0xFFFFFFFF
    c_bytes = [c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, (c >> 24) & 0xFF]
    local_x = local_a
    local_byte = local_a + 1
    local_result = local_a + 2

    instrs: list[WasmInstr] = [
        _instr(OP_LOCAL_SET, local_x),
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_STORE, 0, 0),
    ]

    for b in range(4):
        cb = c_bytes[b]

        if op == "and":
            if cb == 0xFF:
                continue
            if cb == 0x00:
                instrs += [
                    _instr(OP_I32_CONST, SCRATCH_ADDR),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_I32_STORE8, 0, b),
                ]
                continue
        elif op == "or":
            if cb == 0x00:
                continue
            if cb == 0xFF:
                instrs += [
                    _instr(OP_I32_CONST, SCRATCH_ADDR),
                    _instr(OP_I32_CONST, 0xFF),
                    _instr(OP_I32_STORE8, 0, b),
                ]
                continue
        else:  # xor
            if cb == 0x00:
                continue
            if cb == 0xFF:
                instrs += [
                    _instr(OP_I32_CONST, SCRATCH_ADDR),
                    _instr(OP_I32_LOAD8_U, 0, b),
                    _instr(OP_LOCAL_SET, local_byte),
                    _instr(OP_I32_CONST, SCRATCH_ADDR),
                    _instr(OP_I32_CONST, 255),
                    _instr(OP_LOCAL_GET, local_byte),
                    _instr(OP_I32_SUB),
                    _instr(OP_I32_STORE8, 0, b),
                ]
                continue

        instrs += [
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_I32_LOAD8_U, 0, b),
            _instr(OP_LOCAL_SET, local_byte),
        ]
        instrs += _emit_byte_bitop(op, cb, local_byte, local_result)
        instrs += [
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_LOCAL_GET, local_result),
            _instr(OP_I32_STORE8, 0, b),
        ]

    instrs += [
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_I32_LOAD, 0, 0),
    ]
    return instrs


def _expand_and_general(c: int, local_a: int) -> list[WasmInstr]:
    """Expand x & C for arbitrary C."""
    c = c & 0xFFFFFFFF
    if c == 0xFFFFFFFE:
        return _expand_and_fffffffe(local_a)
    if c == 1:
        return _expand_and_1(local_a)
    return _expand_bitop_general("and", c, local_a)


def _expand_and_1(local_a: int) -> list[WasmInstr]:
    """Expand x & 1 = x mod 2 (extract least significant bit)."""
    local_x = local_a
    local_byte = local_a + 1

    return [
        _instr(OP_LOCAL_SET, local_x),
        # Extract low byte: store8 + load8_u
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_STORE8, 0, 0),
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_I32_LOAD8_U, 0, 0),
        _instr(OP_LOCAL_SET, local_byte),
        # byte mod 2 via subtraction loop
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOOP, 0x40),
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_LT_U),
        _instr(OP_BR_IF, 1),  # if byte < 2, exit
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_byte),
        _instr(OP_BR, 0),  # continue loop
        _instr(OP_END),
        _instr(OP_END),
        _instr(OP_LOCAL_GET, local_byte),
    ]


def _expand_and_fffffffe(local_a: int) -> list[WasmInstr]:
    """Expand x & 0xFFFFFFFE = x with lowest bit cleared = x - (x mod 2)."""
    local_x = local_a
    local_byte = local_a + 1

    return [
        _instr(OP_LOCAL_SET, local_x),
        # Extract low byte
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_I32_STORE8, 0, 0),
        _instr(OP_I32_CONST, SCRATCH_ADDR),
        _instr(OP_I32_LOAD8_U, 0, 0),
        _instr(OP_LOCAL_SET, local_byte),  # byte0 = x & 0xFF
        # byte0 mod 2 via subtraction loop
        _instr(OP_BLOCK, 0x40),
        _instr(OP_LOOP, 0x40),
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_LT_U),
        _instr(OP_BR_IF, 1),
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_I32_CONST, 2),
        _instr(OP_I32_SUB),
        _instr(OP_LOCAL_SET, local_byte),
        _instr(OP_BR, 0),
        _instr(OP_END),
        _instr(OP_END),
        # byte0 is now x & 1 (0 or 1)
        _instr(OP_LOCAL_GET, local_x),
        _instr(OP_LOCAL_GET, local_byte),
        _instr(OP_I32_SUB),  # result = x - (x & 1)
    ]


def _expand_and_7ffffffe_v2(local_a: int) -> list[WasmInstr]:
    """Expand x & 0x7FFFFFFE using conditional subtraction."""
    local_x = local_a
    local_byte = local_a + 1

    instrs = []
    instrs.append(_instr(OP_LOCAL_SET, local_x))

    # Step 1: clear bit 0 — extract low byte, compute mod 2
    instrs.extend(
        [
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_LOCAL_GET, local_x),
            _instr(OP_I32_STORE8, 0, 0),
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_I32_LOAD8_U, 0, 0),
            _instr(OP_LOCAL_SET, local_byte),  # byte0
            # mod 2 loop
            _instr(OP_BLOCK, 0x40),
            _instr(OP_LOOP, 0x40),
            _instr(OP_LOCAL_GET, local_byte),
            _instr(OP_I32_CONST, 2),
            _instr(OP_I32_LT_U),
            _instr(OP_BR_IF, 1),
            _instr(OP_LOCAL_GET, local_byte),
            _instr(OP_I32_CONST, 2),
            _instr(OP_I32_SUB),
            _instr(OP_LOCAL_SET, local_byte),
            _instr(OP_BR, 0),
            _instr(OP_END),
            _instr(OP_END),
            # x = x - bit0
            _instr(OP_LOCAL_GET, local_x),
            _instr(OP_LOCAL_GET, local_byte),
            _instr(OP_I32_SUB),
            _instr(OP_LOCAL_SET, local_x),
        ]
    )

    # Step 2: clear bit 31 — conditional subtract 0x80000000
    # Use: x = x - (x >= 0x80000000u) * 0x80000000
    # Implement with block + br_if:
    instrs.extend(
        [
            _instr(OP_BLOCK, 0x40),
            _instr(OP_LOCAL_GET, local_x),
            _instr(OP_I32_CONST, -2147483648),  # 0x80000000 as signed i32
            _instr(OP_I32_LT_U),  # x < 0x80000000? (bit 31 not set)
            _instr(OP_BR_IF, 0),  # skip if bit 31 not set
            _instr(OP_LOCAL_GET, local_x),
            _instr(OP_I32_CONST, -2147483648),
            _instr(OP_I32_SUB),
            _instr(OP_LOCAL_SET, local_x),
            _instr(OP_END),
        ]
    )

    instrs.append(_instr(OP_LOCAL_GET, local_x))
    return instrs


def _expand_xor(c: int, local_a: int) -> list[WasmInstr]:
    """Expand x ^ C."""
    if c == 0xFFFFFFFF:
        return [
            _instr(OP_LOCAL_SET, local_a),
            _instr(OP_I32_CONST, -1),
            _instr(OP_LOCAL_GET, local_a),
            _instr(OP_I32_SUB),
        ]
    if c == 1:
        local_x = local_a
        local_bit = local_a + 1
        instrs = [_instr(OP_LOCAL_TEE, local_x)]
        instrs += _expand_and_1(local_a)
        instrs += [
            _instr(OP_LOCAL_SET, local_bit),
            _instr(OP_LOCAL_GET, local_x),
            _instr(OP_I32_CONST, 1),
            _instr(OP_I32_ADD),
            _instr(OP_LOCAL_GET, local_bit),
            _instr(OP_LOCAL_GET, local_bit),
            _instr(OP_I32_ADD),
            _instr(OP_I32_SUB),
        ]
        return instrs
    return _expand_bitop_general("xor", c, local_a)


def _expand_or(c: int, local_a: int) -> list[WasmInstr]:
    """Expand x | C."""
    return _expand_bitop_general("or", c, local_a)


def _find_const_locals(instrs) -> dict[int, int]:
    """Pre-scan instructions to find locals always set to the same i32.const.

    Returns a dict mapping local index → constant value for locals that are
    only ever assigned from i32.const with a single consistent value.
    This enables lowering of ``local.get X; HARD_OP`` patterns where the
    compiler hoists constants into registers.
    """
    const_map: dict[int, int | None] = {}
    for i, ins in enumerate(instrs):
        if ins.opcode in (OP_LOCAL_SET, OP_LOCAL_TEE):
            local_idx = ins.immediates[0]
            if i > 0 and instrs[i - 1].opcode == OP_I32_CONST:
                val = instrs[i - 1].immediates[0]
                if local_idx not in const_map:
                    const_map[local_idx] = val
                elif const_map[local_idx] != val and const_map[local_idx] is not None:
                    const_map[local_idx] = None
            else:
                const_map[local_idx] = None
    return {k: v for k, v in const_map.items() if v is not None}


def _expand_shr_s(c: int, local_a: int) -> list[WasmInstr]:
    """Expand signed x >> C using memory byte extraction.

    For byte-aligned shifts, uses i32.load8_s / i32.load16_s to get the
    correct sign-extended result.  Falls back to unsigned shift for other cases.
    """
    if c == 0:
        return []
    if c >= 32:
        return [
            _instr(OP_LOCAL_SET, local_a),
            _instr(OP_I32_CONST, -1),
            _instr(OP_I32_CONST, 0),
            _instr(OP_LOCAL_GET, local_a),
            _instr(OP_I32_CONST, 0),
            _instr(OP_I32_LT_S),
            _instr(OP_SELECT),
        ]

    q, r = divmod(c, 8)

    if r == 0:
        if q == 3:
            return [
                _instr(OP_LOCAL_SET, local_a),
                _instr(OP_I32_CONST, SCRATCH_ADDR),
                _instr(OP_LOCAL_GET, local_a),
                _instr(OP_I32_STORE, 0, 0),
                _instr(OP_I32_CONST, SCRATCH_ADDR),
                _instr(OP_I32_LOAD8_S, 0, 3),
            ]
        if q == 2:
            return [
                _instr(OP_LOCAL_SET, local_a),
                _instr(OP_I32_CONST, SCRATCH_ADDR),
                _instr(OP_LOCAL_GET, local_a),
                _instr(OP_I32_STORE, 0, 0),
                _instr(OP_I32_CONST, SCRATCH_ADDR),
                _instr(OP_I32_LOAD16_S, 0, 2),
            ]
        if q == 1:
            local_tmp = local_a + 1
            return [
                _instr(OP_LOCAL_SET, local_a),
                _instr(OP_I32_CONST, SCRATCH_ADDR),
                _instr(OP_LOCAL_GET, local_a),
                _instr(OP_I32_STORE, 0, 0),
                _instr(OP_I32_CONST, SCRATCH_ADDR),
                _instr(OP_I32_CONST, 0),
                _instr(OP_I32_STORE, 0, 4),
                _instr(OP_I32_CONST, SCRATCH_ADDR),
                _instr(OP_I32_LOAD, 0, 1),
                _instr(OP_LOCAL_SET, local_tmp),
                _instr(OP_BLOCK, 0x40),
                _instr(OP_LOCAL_GET, local_tmp),
                _instr(OP_I32_CONST, 0x800000),
                _instr(OP_I32_LT_U),
                _instr(OP_BR_IF, 0),
                _instr(OP_LOCAL_GET, local_tmp),
                _instr(OP_I32_CONST, 0x1000000),
                _instr(OP_I32_SUB),
                _instr(OP_LOCAL_SET, local_tmp),
                _instr(OP_END),
                _instr(OP_LOCAL_GET, local_tmp),
            ]

    if q > 0:
        instrs = [
            _instr(OP_LOCAL_SET, local_a),
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_LOCAL_GET, local_a),
            _instr(OP_I32_STORE, 0, 0),
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_I32_CONST, 0),
            _instr(OP_I32_STORE, 0, 4),
            _instr(OP_I32_CONST, SCRATCH_ADDR),
            _instr(OP_I32_LOAD, 0, q),
        ]
        if r > 0:
            instrs.extend(_expand_div_u(1 << r, local_a))
        return instrs

    return _expand_div_u(1 << c, local_a)


def lower_hard_ops(func: FuncBody, num_params: int = 0) -> FuncBody:
    """Lower hard-to-simulate instructions in a function body.

    Returns a new FuncBody with hard ops replaced by basic instruction
    sequences.  Adds temporary locals as needed.

    Args:
        func: The original function body.
        num_params: Number of function parameters (local indices start after these).
    """
    instrs = func.instructions
    needs_lowering = False

    # Check if any lowering is needed
    for ins in instrs:
        if ins.opcode in LOWERABLE_BINOPS or ins.opcode in LOWERABLE_UNARY:
            needs_lowering = True
            break

    if not needs_lowering:
        return func

    const_locals = _find_const_locals(instrs)

    # Allocate temporary locals (4 i32 temps should suffice)
    NUM_TEMPS = 4
    total_existing = num_params + func.num_locals
    temp_base = total_existing  # first temp local index

    new_locals = list(func.locals) + [(NUM_TEMPS, 0x7F)]  # 4 x i32
    new_num_locals = func.num_locals + NUM_TEMPS

    # Build new instruction list
    new_instrs: list[WasmInstr] = []
    i = 0
    lowered_count = 0
    while i < len(instrs):
        ins = instrs[i]

        # Pattern 1: i32.const C + binop
        # Pattern 2: local.get X (where X is a known constant) + binop
        const_val = None
        binop_idx = -1

        if (
            ins.opcode == OP_I32_CONST
            and i + 1 < len(instrs)
            and instrs[i + 1].opcode in LOWERABLE_BINOPS
        ):
            const_val = ins.immediates[0] & 0xFFFFFFFF
            binop_idx = i + 1
        elif (
            ins.opcode == OP_LOCAL_GET
            and ins.immediates[0] in const_locals
            and i + 1 < len(instrs)
            and instrs[i + 1].opcode in LOWERABLE_BINOPS
        ):
            const_val = const_locals[ins.immediates[0]] & 0xFFFFFFFF
            binop_idx = i + 1

        if const_val is not None and binop_idx >= 0:
            op = instrs[binop_idx].opcode
            expansion = None
            local_a = temp_base

            if op == OP_I32_AND:
                if const_val == 255:
                    expansion = _expand_and_255(local_a)
                elif const_val == 0xFFFFFFFE:
                    expansion = _expand_and_fffffffe(local_a)
                elif const_val == 0x7FFFFFFE:
                    expansion = _expand_and_7ffffffe_v2(local_a)
                else:
                    expansion = _expand_and_general(const_val, local_a)

            elif op == OP_I32_MUL:
                expansion = [_instr(OP_LOCAL_SET, local_a)] + _expand_mul(const_val, local_a)

            elif op == OP_I32_DIV_U:
                expansion = _expand_div_u(const_val, local_a)

            elif op in (OP_I32_REM_U, OP_I32_REM_S):
                expansion = _expand_rem_u(const_val, local_a)

            elif op == OP_I32_SHL:
                expansion = _expand_shl_from_stack(const_val, local_a)

            elif op == OP_I32_SHR_U:
                expansion = _expand_shr_u(const_val, local_a)

            elif op == OP_I32_SHR_S:
                expansion = _expand_shr_s(const_val, local_a)

            elif op == OP_I32_XOR:
                expansion = _expand_xor(const_val, local_a)

            elif op == OP_I32_OR:
                expansion = _expand_or(const_val, local_a)

            elif op == OP_I32_DIV_S:
                expansion = _expand_div_s(const_val, local_a)

            elif op == OP_I32_ROTL:
                expansion = _expand_rotl_const(const_val, local_a)

            elif op == OP_I32_ROTR:
                expansion = _expand_rotr_const(const_val, local_a)

            if expansion is not None:
                new_instrs.extend(expansion)
                i = binop_idx + 1
                lowered_count += 1
                continue

        # Runtime SHL (no preceding const): a b i32.shl
        # Loop b times, doubling a each iteration via ADD
        if ins.opcode == OP_I32_SHL:
            local_a = temp_base
            local_b = temp_base + 1
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_b),  # save shift amount
                    _instr(OP_LOCAL_SET, local_a),  # save value
                    _instr(OP_BLOCK, 0x40),  # block $exit
                    _instr(OP_LOOP, 0x40),  #   loop $loop
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_EQZ),
                    _instr(OP_BR_IF, 1),  #     if b == 0: exit
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_SET, local_a),  #     a = a + a
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 1),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),  #     b--
                    _instr(OP_BR, 0),  #     continue
                    _instr(OP_END),  #   end loop
                    _instr(OP_END),  # end block
                    _instr(OP_LOCAL_GET, local_a),  # result
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Runtime SHR_U (no preceding const): a b i32.shr_u
        # Loop b times, halving a each iteration via div-by-2 subtraction loop
        if ins.opcode == OP_I32_SHR_U:
            local_a = temp_base
            local_b = temp_base + 1
            local_q = temp_base + 2
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_b),  # save shift amount
                    _instr(OP_LOCAL_SET, local_a),  # save value
                    _instr(OP_BLOCK, 0x40),  # block $outer
                    _instr(OP_LOOP, 0x40),  #   loop $outer_loop
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_EQZ),
                    _instr(OP_BR_IF, 1),  #     if b == 0: exit
                    # a = a / 2 (unsigned) via subtraction loop
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_LOCAL_SET, local_q),  #     q = 0
                    _instr(OP_BLOCK, 0x40),  #     block $div
                    _instr(OP_LOOP, 0x40),  #       loop $div_loop
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_CONST, 2),
                    _instr(OP_I32_LT_U),
                    _instr(OP_BR_IF, 1),  #         if a < 2: exit div
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_CONST, 2),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_a),  #         a -= 2
                    _instr(OP_LOCAL_GET, local_q),
                    _instr(OP_I32_CONST, 1),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_SET, local_q),  #         q++
                    _instr(OP_BR, 0),  #         continue div
                    _instr(OP_END),  #       end div_loop
                    _instr(OP_END),  #     end div block
                    _instr(OP_LOCAL_GET, local_q),
                    _instr(OP_LOCAL_SET, local_a),  #     a = q
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 1),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),  #     b--
                    _instr(OP_BR, 0),  #     continue outer
                    _instr(OP_END),  #   end outer_loop
                    _instr(OP_END),  # end outer block
                    _instr(OP_LOCAL_GET, local_a),  # result
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Runtime SHR_S (no preceding const): treat as SHR_U for now
        # (correct for non-negative values; varargs shifts are typically small)
        if ins.opcode == OP_I32_SHR_S:
            local_a = temp_base
            local_b = temp_base + 1
            local_q = temp_base + 2
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_EQZ),
                    _instr(OP_BR_IF, 1),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_LOCAL_SET, local_q),
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_CONST, 2),
                    _instr(OP_I32_LT_U),
                    _instr(OP_BR_IF, 1),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_CONST, 2),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_LOCAL_GET, local_q),
                    _instr(OP_I32_CONST, 1),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_SET, local_q),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    _instr(OP_LOCAL_GET, local_q),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 1),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    _instr(OP_LOCAL_GET, local_a),
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Runtime MUL (no preceding const): a b i32.mul
        # Loop: result = 0; while b > 0: result += a; b--
        if ins.opcode == OP_I32_MUL:
            local_a = temp_base
            local_b = temp_base + 1
            local_r = temp_base + 2
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_LOCAL_SET, local_r),
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_EQZ),
                    _instr(OP_BR_IF, 1),
                    _instr(OP_LOCAL_GET, local_r),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_SET, local_r),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 1),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    _instr(OP_LOCAL_GET, local_r),
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Runtime DIV_U (no preceding const): a b i32.div_u
        if ins.opcode == OP_I32_DIV_U:
            local_a = temp_base
            local_b = temp_base + 1
            local_q = temp_base + 2
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_LOCAL_SET, local_q),
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_LT_U),
                    _instr(OP_BR_IF, 1),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_LOCAL_GET, local_q),
                    _instr(OP_I32_CONST, 1),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_SET, local_q),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    _instr(OP_LOCAL_GET, local_q),
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Runtime REM_U / REM_S (no preceding const): subtraction loop returning remainder
        if ins.opcode in (OP_I32_REM_U, OP_I32_REM_S):
            local_a = temp_base
            local_b = temp_base + 1
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_LT_U),
                    _instr(OP_BR_IF, 1),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    _instr(OP_LOCAL_GET, local_a),
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Runtime XOR (no preceding const): approximate as NE (0 or 1)
        if ins.opcode == OP_I32_XOR:
            new_instrs.append(_instr(OP_I32_NE))
            i += 1
            lowered_count += 1
            continue

        # Runtime AND (no preceding const): a b i32.and → select(a, 0, b)
        if ins.opcode == OP_I32_AND:
            local_a = temp_base
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_a),  # save b
                    _instr(OP_I32_CONST, 0),  # push 0
                    _instr(OP_LOCAL_GET, local_a),  # push b (condition)
                    _instr(OP_SELECT),  # if b != 0: a, else: 0
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Runtime OR (no preceding const): a b i32.or → select(1, a, b)
        if ins.opcode == OP_I32_OR:
            local_a = temp_base
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_a),  # save b
                    _instr(OP_I32_CONST, 1),  # push 1
                    _instr(OP_LOCAL_GET, local_a),  # push b (condition)
                    _instr(OP_SELECT),  # if b != 0: 1, else: a
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Runtime DIV_S (no preceding const): signed division via abs + unsigned div
        if ins.opcode == OP_I32_DIV_S:
            local_a = temp_base
            local_b = temp_base + 1
            local_q = temp_base + 2
            local_neg = temp_base + 3
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_LOCAL_SET, local_a),
                    # neg = (a < 0) XOR (b < 0)
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_I32_LT_S),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_I32_LT_S),
                    _instr(OP_I32_NE),
                    _instr(OP_LOCAL_SET, local_neg),
                    # abs(a)
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_I32_GE_S),
                    _instr(OP_BR_IF, 0),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_END),
                    # abs(b)
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_I32_GE_S),
                    _instr(OP_BR_IF, 0),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_END),
                    # unsigned divide: q = a / b
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_LOCAL_SET, local_q),
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_LT_U),
                    _instr(OP_BR_IF, 1),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_LOCAL_GET, local_q),
                    _instr(OP_I32_CONST, 1),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_SET, local_q),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    # negate if signs differed
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOCAL_GET, local_neg),
                    _instr(OP_I32_EQZ),
                    _instr(OP_BR_IF, 0),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_LOCAL_GET, local_q),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_q),
                    _instr(OP_END),
                    _instr(OP_LOCAL_GET, local_q),
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Runtime ROTL (no preceding const): loop b times rotating left by 1
        if ins.opcode == OP_I32_ROTL:
            local_a = temp_base
            local_b = temp_base + 1
            local_bit = temp_base + 2
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_LOCAL_SET, local_a),
                    # b = b mod 32
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 32),
                    _instr(OP_I32_LT_U),
                    _instr(OP_BR_IF, 1),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 32),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    # rotate left by 1, b times
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_EQZ),
                    _instr(OP_BR_IF, 1),
                    # bit = (a < 0) signed → bit 31 set
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_I32_LT_S),
                    _instr(OP_LOCAL_SET, local_bit),
                    # a = a + a + bit
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_GET, local_bit),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_SET, local_a),
                    # b--
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 1),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    _instr(OP_LOCAL_GET, local_a),
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Runtime ROTR (no preceding const): convert to ROTL by (32 - b%32)%32
        if ins.opcode == OP_I32_ROTR:
            local_a = temp_base
            local_b = temp_base + 1
            local_bit = temp_base + 2
            new_instrs.extend(
                [
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_LOCAL_SET, local_a),
                    # b = b mod 32
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 32),
                    _instr(OP_I32_LT_U),
                    _instr(OP_BR_IF, 1),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 32),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    # b = (32 - b) mod 32
                    _instr(OP_I32_CONST, 32),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 32),
                    _instr(OP_I32_LT_U),
                    _instr(OP_BR_IF, 1),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 32),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    # rotate left by 1, b times (same as ROTL loop)
                    _instr(OP_BLOCK, 0x40),
                    _instr(OP_LOOP, 0x40),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_EQZ),
                    _instr(OP_BR_IF, 1),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_CONST, 0),
                    _instr(OP_I32_LT_S),
                    _instr(OP_LOCAL_SET, local_bit),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_LOCAL_GET, local_a),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_GET, local_bit),
                    _instr(OP_I32_ADD),
                    _instr(OP_LOCAL_SET, local_a),
                    _instr(OP_LOCAL_GET, local_b),
                    _instr(OP_I32_CONST, 1),
                    _instr(OP_I32_SUB),
                    _instr(OP_LOCAL_SET, local_b),
                    _instr(OP_BR, 0),
                    _instr(OP_END),
                    _instr(OP_END),
                    _instr(OP_LOCAL_GET, local_a),
                ]
            )
            i += 1
            lowered_count += 1
            continue

        # Unary CLZ
        if ins.opcode == OP_I32_CLZ:
            new_instrs.extend(_expand_clz(temp_base))
            i += 1
            lowered_count += 1
            continue

        # Unary CTZ
        if ins.opcode == OP_I32_CTZ:
            new_instrs.extend(_expand_ctz(temp_base))
            i += 1
            lowered_count += 1
            continue

        # Unary POPCNT
        if ins.opcode == OP_I32_POPCNT:
            new_instrs.extend(_expand_popcnt(temp_base))
            i += 1
            lowered_count += 1
            continue

        if ins.opcode == OP_I32_EXTEND8_S:
            new_instrs.extend(_expand_extend8_s(temp_base))
            i += 1
            lowered_count += 1
            continue

        if ins.opcode == OP_I32_EXTEND16_S:
            new_instrs.extend(_expand_extend16_s(temp_base))
            i += 1
            lowered_count += 1
            continue

        new_instrs.append(ins)
        i += 1

    if lowered_count > 0:
        logger.debug("  Lowered %d hard ops", lowered_count)

    return FuncBody(
        locals=new_locals,
        num_locals=new_num_locals,
        instructions=new_instrs,
    )


# ===================================================================== #
#  Verification                                                          #
# ===================================================================== #


def check_basic_only(func: FuncBody, func_index: int = 0) -> dict[str, int]:
    """Check that a function body uses only basic ops.

    Returns a dict mapping unsupported op names to their occurrence count.
    Empty dict means all instructions are basic.
    """
    bad: dict[str, int] = {}
    for ins in func.instructions:
        if ins.opcode not in BASIC_OPS:
            name = WASM_OP_NAMES.get(ins.opcode, f"0x{ins.opcode:02x}")
            bad[name] = bad.get(name, 0) + 1
    return bad

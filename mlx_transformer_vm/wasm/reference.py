#!/usr/bin/env python3
"""Reference WASM interpreter for exact token-trace generation."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

MASK32 = 0xFFFFFFFF


def to_signed(value):
    return value - (1 << 32) if value >= (1 << 31) else value


def _add_carries(a, b):
    carries = []
    carry = 0
    for index in range(4):
        total = ((a >> (8 * index)) & 0xFF) + ((b >> (8 * index)) & 0xFF) + carry
        carry = 1 if total >= 256 else 0
        carries.append(carry)
    return carries


def _sub_borrows(a, b):
    borrows = []
    borrow = 0
    for index in range(4):
        total = ((a >> (8 * index)) & 0xFF) - ((b >> (8 * index)) & 0xFF) - borrow
        borrow = 1 if total < 0 else 0
        borrows.append(borrow)
    return borrows


def _byte_tokens(value, num_bytes, carries=None):
    tokens = []
    for index in range(num_bytes):
        byte_value = (value >> (8 * index)) & 0xFF
        carry = carries[index] if carries else 0
        tokens.append(f"{byte_value:02x}'" if carry else f"{byte_value:02x}")
    return tokens


def _out_token(byte_value):
    byte_value &= 0xFF
    if 0x20 < byte_value < 0x7F:
        return f"out({chr(byte_value)})"
    return f"out({byte_value:02x})"


def _commit(stack_delta, store_to_stack, branch_taken):
    return f"commit({stack_delta:+d},sts={store_to_stack},bt={branch_taken})"


def load_program(path):
    with open(path) as handle:
        tokens = handle.read().split()
    program = _parse_program_tokens(tokens)
    input_str = _extract_input(tokens)
    return program, input_str


def load_program_from_string(prefix_str):
    return _parse_program_tokens(prefix_str.split())


def _parse_program_tokens(tokens):
    assert tokens[0] == "{"
    try:
        end = len(tokens) - 1 - tokens[::-1].index("}")
    except ValueError as err:
        raise ValueError("no closing '}' found in program") from err
    body = tokens[1:end]
    program = []
    index = 0
    while index < len(body):
        opcode = body[index]
        bytes_ = [int(body[index + 1 + offset], 16) for offset in range(4)]
        immediate = bytes_[0] | (bytes_[1] << 8) | (bytes_[2] << 16) | (bytes_[3] << 24)
        program.append((opcode, immediate))
        index += 5
    return program


def _extract_input(tokens):
    try:
        end = len(tokens) - 1 - tokens[::-1].index("}")
    except ValueError:
        return ""
    input_tokens = tokens[end + 1 :]
    if not input_tokens:
        return ""
    if input_tokens and input_tokens[-1].startswith("commit("):
        input_tokens = input_tokens[:-1]

    chars = []
    for token in input_tokens:
        if len(token) == 1:
            chars.append(token)
        elif len(token) == 2:
            byte_value = int(token, 16)
            if byte_value == 0:
                break
            chars.append(chr(byte_value))
        else:
            break
    return "".join(chars)


def run(program, input_str="", max_tokens=1_000_000, input_base=None, trace=False):
    mem = bytearray(10 * 1024 * 1024)

    if input_base is None and program and program[0][0] == "input_base":
        input_base = program[0][1]

    if input_base is not None and input_str:
        for index, byte_value in enumerate(input_str.encode("utf-8") + b"\x00"):
            mem[input_base + index] = byte_value

    stack = []
    locals_ = [0] * 256
    call_stack = []
    pc = 0
    instr_count = 0
    token_count = 0
    output = []
    trace_tokens = [] if trace else None

    while pc < len(program) and token_count < max_tokens:
        opcode, immediate = program[pc]
        instr_count += 1

        if opcode == "input_base":
            input_bytes = input_str.encode("utf-8") + b"\x00" if input_str else b"\x00"
            token_count += len(input_bytes) + 1
            if trace:
                for byte_value in input_bytes:
                    if 0x20 < byte_value < 0x7F:
                        trace_tokens.append(chr(byte_value))
                    else:
                        trace_tokens.append(f"{byte_value:02x}")
                trace_tokens.append(_commit(0, 0, 0))
            pc += 1

        elif opcode == "halt":
            token_count += 1
            if trace:
                trace_tokens.append("halt")
            break

        elif opcode == "i32.const":
            result = immediate & MASK32
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(+1, 1, 0))
            pc += 1

        elif opcode == "local.get":
            result = locals_[immediate] & MASK32
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(+1, 1, 0))
            pc += 1

        elif opcode == "local.set":
            value = stack.pop() & MASK32
            locals_[immediate] = value
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(value, 4))
                trace_tokens.append(_commit(-1, 0, 0))
            pc += 1

        elif opcode == "local.tee":
            value = stack[-1] & MASK32
            locals_[immediate] = value
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(value, 4))
                trace_tokens.append(_commit(0, 0, 0))
            pc += 1

        elif opcode == "global.get":
            result = locals_[immediate] & MASK32
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(+1, 1, 0))
            pc += 1

        elif opcode == "global.set":
            locals_[immediate] = stack.pop() & MASK32
            token_count += 1
            if trace:
                trace_tokens.append(_commit(-1, 0, 0))
            pc += 1

        elif opcode == "drop":
            stack.pop()
            token_count += 1
            if trace:
                trace_tokens.append(_commit(-1, 0, 0))
            pc += 1

        elif opcode == "select":
            condition = stack.pop()
            b_value = stack.pop()
            a_value = stack.pop()
            result = a_value if condition != 0 else b_value
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-2, 1, 0))
            pc += 1

        elif opcode == "i32.add":
            b_value = stack.pop()
            a_value = stack.pop()
            result = (a_value + b_value) & MASK32
            stack.append(result)
            carries = _add_carries(a_value, b_value)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4, carries))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.sub":
            b_value = stack.pop()
            a_value = stack.pop()
            result = (a_value - b_value) & MASK32
            stack.append(result)
            borrows = _sub_borrows(a_value, b_value)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4, borrows))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.eqz":
            value = stack.pop()
            result = 1 if value == 0 else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif opcode == "i32.eq":
            b_value = stack.pop()
            a_value = stack.pop()
            result = 1 if a_value == b_value else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.ne":
            b_value = stack.pop()
            a_value = stack.pop()
            result = 1 if a_value != b_value else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.lt_s":
            b_value = to_signed(stack.pop())
            a_value = to_signed(stack.pop())
            result = 1 if a_value < b_value else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.lt_u":
            b_value = stack.pop()
            a_value = stack.pop()
            result = 1 if a_value < b_value else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.gt_s":
            b_value = to_signed(stack.pop())
            a_value = to_signed(stack.pop())
            result = 1 if a_value > b_value else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.gt_u":
            b_value = stack.pop()
            a_value = stack.pop()
            result = 1 if a_value > b_value else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.le_s":
            b_value = to_signed(stack.pop())
            a_value = to_signed(stack.pop())
            result = 1 if a_value <= b_value else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.le_u":
            b_value = stack.pop()
            a_value = stack.pop()
            result = 1 if a_value <= b_value else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.ge_s":
            b_value = to_signed(stack.pop())
            a_value = to_signed(stack.pop())
            result = 1 if a_value >= b_value else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.ge_u":
            b_value = stack.pop()
            a_value = stack.pop()
            result = 1 if a_value >= b_value else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif opcode == "i32.load":
            addr = (stack.pop() + immediate) & MASK32
            value = mem[addr] | (mem[addr + 1] << 8) | (mem[addr + 2] << 16) | (mem[addr + 3] << 24)
            result = value & MASK32
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif opcode == "i32.load8_u":
            addr = (stack.pop() + immediate) & MASK32
            result = mem[addr]
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif opcode == "i32.load8_s":
            addr = (stack.pop() + immediate) & MASK32
            value = mem[addr]
            result = (value - 256) & MASK32 if value >= 128 else value
            stack.append(result)
            sign = 1 if value >= 128 else 0
            carries = [sign, sign, sign, sign]
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4, carries))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif opcode == "i32.load16_u":
            addr = (stack.pop() + immediate) & MASK32
            result = mem[addr] | (mem[addr + 1] << 8)
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif opcode == "i32.load16_s":
            addr = (stack.pop() + immediate) & MASK32
            value = mem[addr] | (mem[addr + 1] << 8)
            result = (value - 65536) & MASK32 if value >= 32768 else value
            stack.append(result)
            sign = 1 if value >= 32768 else 0
            carries = [0, sign, sign, sign]
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4, carries))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif opcode == "i32.store":
            value = stack.pop()
            addr = (stack.pop() + immediate) & MASK32
            mem[addr] = value & 0xFF
            mem[addr + 1] = (value >> 8) & 0xFF
            mem[addr + 2] = (value >> 16) & 0xFF
            mem[addr + 3] = (value >> 24) & 0xFF
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(value, 4))
                trace_tokens.append(_commit(-2, 0, 0))
            pc += 1

        elif opcode == "i32.store8":
            value = stack.pop()
            addr = (stack.pop() + immediate) & MASK32
            mem[addr] = value & 0xFF
            token_count += 2
            if trace:
                trace_tokens.extend(_byte_tokens(value, 1))
                trace_tokens.append(_commit(-2, 0, 0))
            pc += 1

        elif opcode == "i32.store16":
            value = stack.pop()
            addr = (stack.pop() + immediate) & MASK32
            mem[addr] = value & 0xFF
            mem[addr + 1] = (value >> 8) & 0xFF
            token_count += 3
            if trace:
                trace_tokens.extend(_byte_tokens(value, 2))
                trace_tokens.append(_commit(-2, 0, 0))
            pc += 1

        elif opcode == "br":
            offset = to_signed(immediate)
            token_count += 6
            if trace:
                trace_tokens.append("branch_taken")
                trace_tokens.extend(_byte_tokens(immediate & MASK32, 4))
                trace_tokens.append(_commit(0, 0, 1))
            pc = pc + 1 + offset

        elif opcode == "br_if":
            condition = stack.pop()
            if condition != 0:
                offset = to_signed(immediate)
                pc = pc + 1 + offset
                token_count += 6
                if trace:
                    trace_tokens.append("branch_taken")
                    trace_tokens.extend(_byte_tokens(immediate & MASK32, 4))
                    trace_tokens.append(_commit(-1, 0, 1))
            else:
                pc += 1
                token_count += 1
                if trace:
                    trace_tokens.append(_commit(-1, 0, 0))

        elif opcode == "call":
            offset = to_signed(immediate)
            call_stack.append((pc + 1, list(locals_), immediate & MASK32))
            locals_ = [0] * 256
            token_count += 6
            if trace:
                trace_tokens.append("branch_taken")
                trace_tokens.extend(_byte_tokens(immediate & MASK32, 4))
                trace_tokens.append("call_commit")
            pc = pc + 1 + offset

        elif opcode == "return":
            ret_pc, ret_locals, call_imm = call_stack.pop()
            ret_offset = (ret_pc - pc - 1) & MASK32
            borrows = _sub_borrows(immediate & MASK32, call_imm)
            token_count += 6
            if trace:
                trace_tokens.append("branch_taken")
                trace_tokens.extend(_byte_tokens(ret_offset, 4, borrows))
                trace_tokens.append("return_commit")
            pc = ret_pc
            locals_ = ret_locals

        elif opcode == "output":
            value = stack.pop() & 0xFF
            output.append(chr(value))
            token_count += 1
            if trace:
                trace_tokens.append(_out_token(value))
            pc += 1

        else:
            raise RuntimeError(f"unknown op: {opcode} at pc={pc}")

    result = (instr_count, token_count, "".join(output))
    if trace:
        return result + (trace_tokens,)
    return result


def format_trace(program_path, trace_tokens):
    with open(program_path) as handle:
        raw = handle.read().split()
    assert raw[0] == "{"
    end = len(raw) - 1 - raw[::-1].index("}")

    lines = ["{"]
    for index in range(1, end):
        group_idx = (index - 1) % 5
        if group_idx == 0:
            current = [raw[index]]
        else:
            current.append(raw[index])
        if group_idx == 4:
            lines.append(" ".join(current))
    lines.append("}")

    current = []
    for token in trace_tokens:
        is_terminal = (
            token.startswith("commit(")
            or token.startswith("out(")
            or token == "halt"
            or token == "branch_taken"
            or token == "call_commit"
            or token == "return_commit"
        )
        current.append(token)
        if is_terminal:
            lines.append(" ".join(current))
            current = []

    if current:
        lines.append(" ".join(current))
    return "\n".join(lines) + "\n"


def generate_ref(prog_path, ref_path=None, max_tokens=100_000_000):
    if ref_path is None:
        ref_path = prog_path.replace(".txt", "_ref.txt")
    program, input_str = load_program(prog_path)
    _instrs, token_count, output, trace_tokens = run(
        program, input_str, max_tokens=max_tokens, trace=True
    )
    formatted = format_trace(prog_path, trace_tokens)
    with open(ref_path, "w") as handle:
        handle.write(formatted)
    logger.info("%s: %d tokens, output=%r", ref_path, token_count, output)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Generate reference token traces from tokenized WASM programs."
    )
    parser.add_argument("files", nargs="+", help="Program .txt files")
    args = parser.parse_args()

    for program_path in args.files:
        generate_ref(program_path)


if __name__ == "__main__":
    main()

"""WASM machine construction expressed in the graph DSL."""

from __future__ import annotations

from mlx_transformer_vm.graph import core as _graph
from mlx_transformer_vm.graph.core import (
    Expression,
    InputDimension,
    ReGLUDimension,
    auto_name,
    fetch,
    fetch_sum,
    persist,
    reglu,
    stepglu,
)

pointsR2 = 32045
points = [
    (179, 2),
    (179, -2),
    (-179, 2),
    (-179, -2),
    (2, 179),
    (2, -179),
    (-2, 179),
    (-2, -179),
    (178, 19),
    (178, -19),
    (-178, 19),
    (-178, -19),
    (19, 178),
    (19, -178),
    (-19, 178),
    (-19, -178),
    (173, 46),
    (173, -46),
    (-173, 46),
    (-173, -46),
    (46, 173),
    (46, -173),
    (-46, 173),
    (-46, -173),
    (166, 67),
    (166, -67),
    (-166, 67),
    (-166, -67),
    (67, 166),
    (67, -166),
    (-67, 166),
    (-67, -166),
    (163, 74),
    (163, -74),
    (-163, 74),
    (-163, -74),
    (74, 163),
    (74, -163),
    (-74, 163),
    (-74, -163),
    (157, 86),
    (157, -86),
    (-157, 86),
    (-157, -86),
    (86, 157),
    (86, -157),
    (-86, 157),
    (-86, -157),
    (142, 109),
    (142, -109),
    (-142, 109),
    (-142, -109),
    (109, 142),
    (109, -142),
    (-109, 142),
    (-109, -142),
    (131, 122),
    (131, -122),
    (-131, 122),
    (-131, -122),
    (122, 131),
    (122, -131),
    (-122, 131),
    (-122, -131),
]

OPCODES = {
    "halt": 0x00,
    "return": 0x0F,
    "call": 0x10,
    "br": 0x0C,
    "br_if": 0x0D,
    "drop": 0x1A,
    "select": 0x1B,
    "local.get": 0x20,
    "local.set": 0x21,
    "local.tee": 0x22,
    "global.get": 0x23,
    "global.set": 0x24,
    "i32.load": 0x28,
    "i32.load8_s": 0x2C,
    "i32.load8_u": 0x2D,
    "i32.load16_s": 0x2E,
    "i32.load16_u": 0x2F,
    "i32.store": 0x36,
    "i32.store8": 0x3A,
    "i32.store16": 0x3B,
    "i32.const": 0x41,
    "i32.eqz": 0x45,
    "i32.eq": 0x46,
    "i32.ne": 0x47,
    "i32.lt_s": 0x48,
    "i32.lt_u": 0x49,
    "i32.gt_s": 0x4A,
    "i32.gt_u": 0x4B,
    "i32.le_s": 0x4C,
    "i32.le_u": 0x4D,
    "i32.ge_s": 0x4E,
    "i32.ge_u": 0x4F,
    "i32.add": 0x6A,
    "i32.sub": 0x6B,
    "output": 0xFF,
    "input_base": 0xFE,
}
OPCODE_POINT = {opcode: points[index] for index, opcode in enumerate(OPCODES)}


def get_byte_value(byte_value, index, signed=False):
    return (byte_value - 256 if signed and byte_value >= 128 else byte_value) * (1 << (8 * index))


STACK_DELTA = {
    "i32.const": +1,
    "drop": -1,
    "i32.add": -1,
    "i32.sub": -1,
    "i32.eqz": 0,
    "i32.gt_s": -1,
    "i32.gt_u": -1,
    "i32.le_s": -1,
    "i32.le_u": -1,
    "i32.ge_s": -1,
    "i32.ge_u": -1,
    "i32.lt_s": -1,
    "i32.lt_u": -1,
    "i32.eq": -1,
    "i32.ne": -1,
    "select": -2,
    "local.set": -1,
    "local.get": +1,
    "local.tee": 0,
    "global.get": +1,
    "global.set": -1,
    "i32.store8": -2,
    "i32.store16": -2,
    "i32.store": -2,
    "i32.load8_s": 0,
    "i32.load8_u": 0,
    "i32.load16_s": 0,
    "i32.load16_u": 0,
    "i32.load": 0,
    "br_if": -1,
    "br": 0,
    "output": -1,
    "halt": 0,
    "call": 0,
    "return": 0,
    "input_base": 0,
}
STS_OPS = {
    "i32.const",
    "i32.add",
    "i32.sub",
    "i32.eqz",
    "select",
    "i32.gt_s",
    "i32.gt_u",
    "i32.le_s",
    "i32.le_u",
    "i32.ge_s",
    "i32.ge_u",
    "i32.lt_s",
    "i32.lt_u",
    "i32.eq",
    "i32.ne",
    "local.get",
    "global.get",
    "i32.load8_s",
    "i32.load8_u",
    "i32.load16_s",
    "i32.load16_u",
    "i32.load",
}
LOCAL_STRIDE = 256


def build(program=None):
    one = _graph.one
    position = _graph.position

    _op_dot_cache = {}

    def op_dot(opcode):
        if opcode not in _op_dot_cache:
            point_x, point_y = OPCODE_POINT[opcode]
            _op_dot_cache[opcode] = (
                point_x * fetched_opcode_x + point_y * fetched_opcode_y - pointsR2 * one + 1
            )
        return _op_dot_cache[opcode]

    _is_op_cache = {}

    def is_op(opcode):
        if opcode not in _is_op_cache:
            if program is not None:
                _is_op_cache[opcode] = stepglu(one, op_dot(opcode))
            else:
                _is_op_cache[opcode] = reglu(one, op_dot(opcode))
        return _is_op_cache[opcode]

    byte_number = InputDimension("byte_number")
    carry = InputDimension("carry")
    delta_cursor = InputDimension("delta_cursor")
    delta_stack = InputDimension("delta_stack")
    is_jump = InputDimension("is_jump")
    store_to_stack = InputDimension("store_to_stack")
    is_branch_taken = InputDimension("is_branch_taken")
    delta_call_depth = InputDimension("delta_call_depth")
    is_return_commit = InputDimension("is_return_commit")

    if program is None:
        delta_stack_prefix = InputDimension("delta_stack_prefix")
        store_to_stack_prefix = InputDimension("store_to_stack_prefix")
        opcode_x = InputDimension("opcode_x")
        opcode_y = InputDimension("opcode_y")
        is_write = InputDimension("is_write")

    input_tokens = (
        {
            (f"{byte_value:02x}'" if carry_bit else f"{byte_value:02x}"): (byte_value + 1)
            * byte_number
            + carry_bit * carry
            for byte_value in range(256)
            for carry_bit in range(2)
        }
        | {
            f"commit({stack_delta:+d},sts={store_stack},bt={branch_taken})": delta_cursor
            + stack_delta * delta_stack
            + store_stack * store_to_stack
            + branch_taken * is_jump
            for stack_delta, store_stack in {
                (STACK_DELTA[opcode], 1 if opcode in STS_OPS else 0) for opcode in OPCODES
            }
            for branch_taken in range(2)
        }
        | {
            (
                f"out({chr(byte_value)})" if 0x20 < byte_value < 0x7F else f"out({byte_value:02x})"
            ): delta_cursor * 1
            for byte_value in range(256)
        }
        | {
            "branch_taken": 1 * is_branch_taken,
            "call_commit": delta_cursor + delta_call_depth + is_jump,
            "return_commit": delta_cursor - delta_call_depth + is_return_commit + is_jump,
        }
    )

    if program is None:
        input_tokens["{"] = 0 * one
        input_tokens["}"] = 3 * delta_stack
        for opcode in OPCODES:
            stack_delta = STACK_DELTA[opcode]
            store_stack = 1 if opcode in STS_OPS else 0
            point_x, point_y = OPCODE_POINT[opcode]
            embedding = (
                point_x * opcode_x
                + point_y * opcode_y
                + stack_delta * delta_stack_prefix
                + store_stack * store_to_stack_prefix
                + (
                    1
                    if opcode in (
                        "local.set",
                        "local.tee",
                        "i32.store8",
                        "i32.store16",
                        "i32.store",
                    )
                    else 0
                )
                * is_write
            )
            input_tokens[opcode] = embedding
    else:
        input_tokens["start"] = 3 * delta_stack

    for byte_value in range(0x21, 0x7F):
        char = chr(byte_value)
        if char in input_tokens:
            continue
        input_tokens[char] = input_tokens[f"{byte_value:02x}"]

    start_token = "start" if program is not None else "{"
    for token_name in input_tokens:
        if token_name != start_token:
            input_tokens[token_name][one] = 1

    store_bytes = [fetch(byte_number - 1, query=position - i, key=position) for i in range(1, 5)]
    store_value = sum((1 << (8 * (4 - i))) * store_bytes[i - 1] for i in range(1, 5))
    store_value = persist(store_value)
    msb = store_bytes[0]
    unsigned_branch = reglu(store_value, is_jump)
    jump_sign = stepglu(one, msb + 128 * is_jump - 256)
    delta_cursor_expr = delta_cursor + unsigned_branch - jump_sign * (1 << 32)

    byte_index = position - fetch(position, query=one, key=one, clear_key=byte_number)
    is_boundary = stepglu(one, -byte_number)

    stack_depth, cursor, call_depth = fetch_sum([delta_stack, delta_cursor_expr, delta_call_depth])

    if program is None:
        instruction_position = 5 * cursor + 1
        (
            fetched_opcode_x,
            fetched_opcode_y,
            fetched_stack_delta,
            fetched_store_to_stack,
            fetched_is_write,
        ) = fetch(
            [opcode_x, opcode_y, delta_stack_prefix, store_to_stack_prefix, is_write],
            query=instruction_position,
            key=position,
        )
        immediate = sum(
            (1 << (8 * (i - 1)))
            * fetch(byte_number - 1, query=instruction_position + i, key=position)
            for i in range(1, 5)
        )
        immediate = persist(immediate)
    else:
        num_instructions = len(program)
        one_expr = Expression({one: 1})
        cursor_expr = cursor if isinstance(cursor, Expression) else Expression({cursor: 1})

        reglu_pos = []
        reglu_neg = []
        for index in range(1, num_instructions):
            reglu_pos.append(
                ReGLUDimension(one_expr, cursor_expr - index + 1, name=f"step_pos_{index}")
            )
            reglu_neg.append(
                ReGLUDimension(one_expr, cursor_expr - index, name=f"step_neg_{index}")
            )

        def _cursor_lookup(values, name=None):
            expr = Expression({one: values[0]})
            for index in range(1, num_instructions):
                diff = values[index] - values[index - 1]
                if diff == 0:
                    continue
                expr[reglu_pos[index - 1]] = diff
                expr[reglu_neg[index - 1]] = -diff
            return persist(expr, name=name)

        write_ops = {"local.set", "local.tee", "i32.store8", "i32.store16", "i32.store"}
        opx_vals = [OPCODE_POINT[instruction["opcode"]][0] for instruction in program]
        opy_vals = [OPCODE_POINT[instruction["opcode"]][1] for instruction in program]
        stack_delta_vals = [STACK_DELTA[instruction["opcode"]] for instruction in program]
        store_stack_vals = [1 if instruction["opcode"] in STS_OPS else 0 for instruction in program]
        is_write_vals = [1 if instruction["opcode"] in write_ops else 0 for instruction in program]

        fetched_opcode_x = _cursor_lookup(opx_vals, "fetched_opcode_x")
        fetched_opcode_y = _cursor_lookup(opy_vals, "fetched_opcode_y")
        fetched_stack_delta = _cursor_lookup(stack_delta_vals, "fetched_stack_delta")
        fetched_store_to_stack = _cursor_lookup(store_stack_vals, "fetched_store_to_stack")
        fetched_is_write = _cursor_lookup(is_write_vals, "fetched_is_write")

        imm_vals = [
            sum(instruction["bytes"][j] * (1 << (8 * j)) for j in range(4))
            for instruction in program
        ]
        immediate = _cursor_lookup(imm_vals, "immediate")

    is_output = is_op("output")
    memory_write_gate = persist(
        is_op("i32.store") + is_op("i32.store8") + is_op("i32.store16") + is_op("input_base")
    )
    uses_top_byte = fetched_is_write + is_output
    is_producing_bytes = fetched_store_to_stack + fetched_is_write

    local_write_key_dim = LOCAL_STRIDE * call_depth + 4 * immediate + byte_index
    not_local_write = 1 - is_op("local.set") - is_op("local.tee") + is_boundary
    local_byte = fetch(
        byte_number - 1,
        query=local_write_key_dim + 1,
        key=local_write_key_dim,
        clear_key=not_local_write,
    )

    not_store_to_stack = 1 - store_to_stack
    stack_top_value, stack_top_position = fetch(
        [store_value, position - 4],
        query=stack_depth,
        key=stack_depth,
        clear_key=not_store_to_stack,
    )
    stack_second_value, stack_second_position = fetch(
        [store_value, position - 4],
        query=stack_depth - 1,
        key=stack_depth,
        clear_key=not_store_to_stack,
    )
    stack_third_position = fetch(
        position - 4,
        query=stack_depth - 2,
        key=stack_depth,
        clear_key=not_store_to_stack,
    )
    top_byte = fetch(byte_number - 1, query=stack_top_position + byte_index, key=position)
    second_byte = fetch(byte_number - 1, query=stack_second_position + byte_index, key=position)
    third_byte = fetch(byte_number - 1, query=stack_third_position + byte_index, key=position)

    memory_read_address = stack_top_value + immediate + byte_index
    memory_write_address = stack_second_value + immediate + byte_index - 1
    not_memory_write_byte = 1 + is_boundary - memory_write_gate
    memory_byte_dirty, memory_byte_dirty_position = fetch(
        [byte_number - 1, memory_write_address],
        query=memory_read_address,
        key=memory_write_address,
        clear_key=not_memory_write_byte,
    )
    diff = memory_byte_dirty_position - memory_read_address
    memory_byte = (
        reglu(memory_byte_dirty, diff + 1)
        - 2 * reglu(memory_byte_dirty, diff)
        + reglu(memory_byte_dirty, diff - 1)
    )
    memory_sign = stepglu(one, memory_byte - 128)

    carry_late = persist(carry)

    add_value = second_byte + top_byte + carry_late
    add_carry = stepglu(one, add_value - 256)
    add_byte = add_value - 256 * add_carry

    sub_value = second_byte - top_byte - carry_late
    sub_borrow = 1 - stepglu(one, sub_value)
    sub_byte = sub_value + 256 * sub_borrow

    a_gt_b_u = stepglu(one, stack_second_value - stack_top_value - 1)
    a_lt_b_u = stepglu(one, stack_top_value - stack_second_value - 1)
    a_eq_b = one - a_gt_b_u - a_lt_b_u

    sign_diff = persist(
        reglu(one, stack_top_value - (1 << 31) + 1)
        - reglu(one, stack_top_value - (1 << 31))
        - reglu(one, stack_second_value - (1 << 31) + 1)
        + reglu(one, stack_second_value - (1 << 31))
    )
    a_gt_b_s = stepglu(one, sign_diff + a_gt_b_u - 1)
    a_lt_b_s = stepglu(one, -sign_diff + a_lt_b_u - 1)

    cond_nonzero = stepglu(one, stack_top_value - 1)

    call_stack_write_key = call_depth * 4 + byte_index - 1
    call_stack_read_key = (call_depth - 1) * 4 + byte_index
    if program is not None:
        not_call_byte = 1 - stepglu(one - is_boundary, op_dot("call"))
    else:
        not_call_byte = 1 - reglu(one - is_boundary, op_dot("call"))
    call_stack_byte = fetch(
        byte_number - 1,
        query=call_stack_read_key,
        key=call_stack_write_key,
        clear_key=not_call_byte,
    )

    if program is None:
        const_byte = fetch(byte_number - 1, query=instruction_position + byte_index + 1, key=position)
    else:
        const_bytes = []
        for byte in range(4):
            byte_vals = [instruction["bytes"][byte] for instruction in program]
            const_bytes.append(_cursor_lookup(byte_vals, f"const_byte_{byte}"))
        const_byte = const_bytes[0]
        for byte in range(1, 4):
            const_byte = const_byte + stepglu(const_bytes[byte] - const_bytes[byte - 1], byte_index - byte)

    call_stack_byte_gated = reglu(call_stack_byte, op_dot("return"))
    carry_gated = reglu(carry_late, op_dot("return"))
    branch_sub_val = const_byte - call_stack_byte_gated - carry_gated
    branch_sub_val = persist(branch_sub_val)
    branch_sub_borrow = 1 - stepglu(one, branch_sub_val)
    branch_byte = branch_sub_val + 256 * branch_sub_borrow
    branch_carry = branch_sub_borrow

    byte_at_2 = stepglu(one, byte_index - 2)

    result_byte_early = persist(
        reglu(local_byte, op_dot("local.get"))
        + reglu(top_byte, uses_top_byte)
        + reglu(third_byte, op_dot("select") + cond_nonzero - 1)
        + reglu(second_byte, op_dot("select") - cond_nonzero)
    )
    result_byte = persist(
        reglu(branch_sub_val, op_dot("i32.const"))
        + reglu(add_byte, op_dot("i32.add"))
        + reglu(sub_byte, op_dot("i32.sub"))
        + result_byte_early
        + reglu(memory_byte, op_dot("i32.load"))
        + reglu(memory_byte, op_dot("i32.load8_u") + is_boundary - 1)
        + reglu(memory_byte, op_dot("i32.load8_s") + is_boundary - 1)
        + 255 * reglu(carry_late, op_dot("i32.load8_s") - is_boundary)
        + reglu(memory_byte, op_dot("i32.load16_u") - byte_at_2)
        + reglu(memory_byte, op_dot("i32.load16_s") - byte_at_2)
        + 255 * reglu(carry_late, op_dot("i32.load16_s") + byte_at_2 - 1)
        + reglu(branch_sub_val, op_dot("br"))
        + reglu(branch_sub_val, op_dot("br_if"))
        + reglu(branch_sub_val, op_dot("call"))
        + reglu(branch_byte, op_dot("return"))
    ) + persist(
        reglu(a_eq_b, op_dot("i32.eq") + is_boundary - 1)
        + reglu(1 - a_eq_b, op_dot("i32.ne") + is_boundary - 1)
        + reglu(a_gt_b_u, op_dot("i32.gt_u") + is_boundary - 1)
        + reglu(1 - a_gt_b_u, op_dot("i32.le_u") + is_boundary - 1)
        + reglu(a_lt_b_u, op_dot("i32.lt_u") + is_boundary - 1)
        + reglu(1 - a_lt_b_u, op_dot("i32.ge_u") + is_boundary - 1)
        + reglu(a_gt_b_s, op_dot("i32.gt_s") + is_boundary - 1)
        + reglu(1 - a_gt_b_s, op_dot("i32.le_s") + is_boundary - 1)
        + reglu(a_lt_b_s, op_dot("i32.lt_s") + is_boundary - 1)
        + reglu(1 - a_lt_b_s, op_dot("i32.ge_s") + is_boundary - 1)
        + reglu(1 - cond_nonzero, op_dot("i32.eqz") + is_boundary - 1)
    )

    result_carry = persist(
        reglu(add_carry, op_dot("i32.add"))
        + reglu(sub_borrow, op_dot("i32.sub"))
        + reglu(memory_sign, op_dot("i32.load8_s") + is_boundary - 1)
        + reglu(carry_late, op_dot("i32.load8_s") - is_boundary)
        + reglu(memory_sign, op_dot("i32.load16_s") - is_boundary)
        + reglu(carry_late - memory_sign, op_dot("i32.load16_s") + byte_at_2 - 1)
        + reglu(branch_carry, op_dot("return"))
    )

    byte_index_4 = stepglu(one, byte_index - 4)
    early_done = reglu(1 - is_boundary, op_dot("i32.store8")) + reglu(byte_at_2, op_dot("i32.store16"))
    early_done = persist(early_done)
    byte_done = byte_index_4 + early_done
    is_byte_seq = 1 - is_boundary - byte_done

    emit_halt = reglu(is_boundary, op_dot("halt"))
    emit_branch_taken = (
        reglu(is_boundary, op_dot("br") - is_branch_taken)
        + reglu(is_boundary, cond_nonzero + op_dot("br_if") - is_branch_taken - 1)
        + reglu(is_boundary, op_dot("return") - is_branch_taken)
        + reglu(is_boundary, op_dot("call") - is_branch_taken)
    )
    emit_return_commit = reglu(byte_index_4, op_dot("return"))
    emit_out = reglu(is_boundary, op_dot("output"))
    emit_byte_start = reglu(is_producing_bytes, is_boundary)
    emit_byte = emit_byte_start + is_byte_seq + is_branch_taken
    emit_call_commit = reglu(byte_index_4, op_dot("call"))
    emit_bt = reglu(byte_done, op_dot("br")) + reglu(byte_done, op_dot("br_if"))
    emit_commit = (
        byte_done
        + is_boundary
        - emit_halt
        - emit_branch_taken
        - emit_return_commit
        - emit_out
        - emit_byte_start
        - is_branch_taken
        - emit_call_commit
    )

    H = 1e5
    output_tokens = {}
    output_tokens["halt"] = H * emit_halt
    output_tokens["branch_taken"] = H * emit_branch_taken
    output_tokens["call_commit"] = H * emit_call_commit
    output_tokens["return_commit"] = H * emit_return_commit

    for byte_value in range(256):
        token_name = f"out({chr(byte_value)})" if 0x20 < byte_value < 0x7F else f"out({byte_value:02x})"
        output_tokens[token_name] = H * emit_out + (2 * byte_value) * top_byte - byte_value * byte_value

    for stack_delta, store_stack in {
        (STACK_DELTA[opcode], 1 if opcode in STS_OPS else 0) for opcode in OPCODES
    }:
        for branch_taken in range(2):
            output_tokens[f"commit({stack_delta:+d},sts={store_stack},bt={branch_taken})"] = (
                H * emit_commit
                + (2 * stack_delta) * fetched_stack_delta
                - stack_delta * stack_delta
                + (2 * store_stack) * fetched_store_to_stack
                - store_stack * store_stack
                + (2 * branch_taken) * emit_bt
                - branch_taken * branch_taken
            )

    for byte_value in range(256):
        byte_base = H * emit_byte + (2 * byte_value) * result_byte - byte_value * byte_value
        for carry_bit in range(2):
            score = byte_base + (2 * carry_bit) * result_carry - carry_bit * carry_bit
            output_tokens[f"{byte_value:02x}'" if carry_bit else f"{byte_value:02x}"] = score

    auto_name(locals())
    return input_tokens, output_tokens


class WASMMachine:
    """WASM interpreter graph builder.

    With no program argument, builds the universal interpreter graph.
    With a program argument, builds the specialized graph shape.
    """

    def __init__(self, program=None):
        self.program = program

    def build(self):
        from mlx_transformer_vm.graph.core import ProgramGraph, reset_graph

        reset_graph()
        input_tokens, output_tokens = build(program=self.program)
        return ProgramGraph(input_tokens, output_tokens)

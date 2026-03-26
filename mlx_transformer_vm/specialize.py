#!/usr/bin/env python3
"""Specialize the WASM interpreter for one compiled token program."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from mlx.utils import tree_flatten

from mlx_transformer_vm.model.weights import build_model, save_weights
from mlx_transformer_vm.wasm.interpreter import WASMMachine

logger = logging.getLogger(__name__)


def _parameter_count(model):
    return sum(math.prod(value.shape) for _name, value in tree_flatten(model.parameters()))


def parse_program(filepath):
    """Parse a compiled token program into instruction dictionaries."""

    content = Path(filepath).read_text()
    tokens = content.split()

    if "{" in tokens:
        start = tokens.index("{")
        try:
            end = tokens.index("}")
        except ValueError as err:
            raise ValueError(f"found '{{' but no '}}' in {filepath}") from err
        program_tokens = tokens[start + 1 : end]
    else:
        program_tokens = content.split()

    instructions = []
    for index in range(0, len(program_tokens), 5):
        chunk = program_tokens[index : index + 5]
        if not chunk:
            continue
        if len(chunk) != 5:
            raise ValueError(f"incomplete instruction at token {index}: {chunk}")
        instructions.append(
            {
                "opcode": chunk[0],
                "bytes": [int(chunk[1 + offset], 16) for offset in range(4)],
            }
        )
    return instructions


def spec_input_from_txt(filepath):
    """Extract the specialized-model input sequence from a compiled ``.txt`` file."""

    tokens = Path(filepath).read_text().split()
    try:
        end = tokens.index("}")
    except ValueError as err:
        raise ValueError(f"no '}}' found in {filepath}") from err
    return ["start", *tokens[end + 1 :]]


def specialize(filepath):
    """Return a model specialized to the program in ``filepath``."""

    instructions = parse_program(filepath)
    logger.info("parsed %d instructions from %s", len(instructions), filepath)

    program_graph = WASMMachine(program=instructions).build()
    logger.info(
        "  %d dims, %d lookups, %d input tokens, %d output tokens",
        len(program_graph.all_dims),
        len(program_graph.all_lookups),
        len(program_graph.input_tokens),
        len(program_graph.output_tokens),
    )

    model, all_tokens, tok_to_idx_map, _shared = build_model(program_graph=program_graph)
    return model, all_tokens, tok_to_idx_map, instructions


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Specialize the WASM interpreter for a compiled .txt program."
    )
    parser.add_argument("program", help="Path to a compiled .txt program")
    parser.add_argument(
        "--save-weights",
        type=str,
        default=None,
        help="Output path for the specialized model (defaults to <program>.bin)",
    )
    args = parser.parse_args()

    program_path = Path(args.program)
    save_path = Path(args.save_weights) if args.save_weights else program_path.with_suffix(".bin")

    model, all_tokens, _tok_to_idx_map, instructions = specialize(program_path)
    logger.info("Specialized model for %s:", program_path.name)
    logger.info(
        "  d_model=%d, n_layers=%d, n_heads=%d, d_ffn=%d",
        model.d_model,
        model.n_layers,
        model.n_heads,
        model.d_ffn,
    )
    logger.info("  vocab=%d, params=%s", len(all_tokens), f"{_parameter_count(model):,}")
    logger.info("  instructions baked: %d", len(instructions))

    save_weights(
        model,
        all_tokens,
        str(save_path),
        metadata={
            "model_kind": "specialized",
            "program_path": str(program_path),
            "instruction_count": len(instructions),
        },
    )


if __name__ == "__main__":
    main()

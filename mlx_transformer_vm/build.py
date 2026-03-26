#!/usr/bin/env python3
"""Build the universal WASM transformer with analytically derived weights."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from mlx.utils import tree_flatten

from mlx_transformer_vm.model.weights import build_model, save_weights

logger = logging.getLogger(__name__)


def _parameter_count(model):
    return sum(math.prod(value.shape) for _name, value in tree_flatten(model.parameters()))


def build(plan_path=None, max_layers=None, no_reuse=False, max_ffn=None):
    """Build the universal WASM transformer."""

    model, all_tokens, tok_to_idx_map, _shared = build_model(
        plan_path=plan_path,
        max_layers=max_layers,
        no_reuse=no_reuse,
        max_ffn=max_ffn,
    )
    return model, all_tokens, tok_to_idx_map


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Build the universal WASM transformer with analytically derived weights."
    )
    parser.add_argument(
        "--plan", type=str, default=None, help="Schedule plan YAML path (omit to run MILP solver)"
    )
    parser.add_argument(
        "--milp",
        action="store_true",
        help="Generate the schedule via the MILP solver (ignores --plan)",
    )
    parser.add_argument(
        "--layers",
        "--max-layers",
        type=int,
        default=None,
        dest="max_layers",
        help="Max transformer layers",
    )
    parser.add_argument("--max-ffn", type=int, default=None, help="Max FFN neurons per layer")
    parser.add_argument("--no-reuse", action="store_true", help="Disable slot reuse")
    parser.add_argument(
        "--save-weights",
        type=str,
        default=None,
        help="Save weights to file",
    )
    args = parser.parse_args()

    plan = None if args.milp else args.plan
    model, all_tokens, _tok_to_idx_map = build(
        plan_path=plan,
        max_layers=args.max_layers,
        no_reuse=args.no_reuse,
        max_ffn=args.max_ffn,
    )

    logger.info("Universal model:")
    logger.info(
        "  d_model=%d, n_layers=%d, n_heads=%d, d_ffn=%d",
        model.d_model,
        model.n_layers,
        model.n_heads,
        model.d_ffn,
    )
    logger.info("  vocab=%d, params=%s", len(all_tokens), f"{_parameter_count(model):,}")

    if args.save_weights:
        save_weights(
            model,
            all_tokens,
            str(Path(args.save_weights)),
            metadata={"model_kind": "universal"},
        )


if __name__ == "__main__":
    main()

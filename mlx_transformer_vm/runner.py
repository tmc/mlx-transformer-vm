#!/usr/bin/env python3
"""Run compiled token programs through the MLX transformer runtime."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import mlx.core as mx

from mlx_transformer_vm.attention import OptimizedKVCache, StandardKVCache
from mlx_transformer_vm.build import build
from mlx_transformer_vm.compilation.compile_wasm import compile_program
from mlx_transformer_vm.model.weights import flops_per_token, load_weights, save_weights

logger = logging.getLogger(__name__)

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "model.bin"


def _ensure_model(model_path):
    model_path = Path(model_path)
    if model_path.exists():
        logger.info("[model] loading weights from %s", model_path)
        return model_path

    logger.info("[model] weights not found at %s", model_path)
    logger.info("[model] solving MILP schedule and constructing weights")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model, all_tokens, _tok_to_idx_map = build()
    save_weights(model, all_tokens, str(model_path), metadata={"model_kind": "universal"})
    return model_path


def _configure_cache(model, cache_class):
    cache = cache_class(model.n_layers, model.n_heads)
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for layer_idx in range(model.n_layers):
            for head_idx in range(model.n_heads):
                if model.head_tiebreak[layer_idx][head_idx]:
                    cache.set_tiebreak(layer_idx, head_idx, True)
    return cache


def _compare_with_ref(predicted, ref_tokens):
    predicted_cmp = predicted
    ref_cmp = ref_tokens

    if predicted_cmp and predicted_cmp[0] == "start":
        predicted_cmp = predicted_cmp[1:]
        if "}" in ref_cmp:
            ref_cmp = ref_cmp[ref_cmp.index("}") + 1 :]

    if predicted_cmp == ref_cmp:
        return True, None

    max_len = max(len(predicted_cmp), len(ref_cmp))
    for index in range(max_len):
        predicted_token = predicted_cmp[index] if index < len(predicted_cmp) else "<END>"
        ref_token = ref_cmp[index] if index < len(ref_cmp) else "<END>"
        if predicted_token != ref_token:
            return False, (index, predicted_token, ref_token)
    return False, None


def run_model_tokens(
    model,
    all_tokens,
    tok_to_idx_map,
    tokens,
    max_new_tokens=50000,
    cache_class=OptimizedKVCache,
):
    cache = _configure_cache(model, cache_class)
    raw_tokens = list(tokens)
    prefill_tokens = list(getattr(model, "prefill_tokens", []))
    input_tokens = raw_tokens
    if prefill_tokens and raw_tokens and raw_tokens[0] == "start":
        input_tokens = raw_tokens[1:]

    token_ids = []
    for token in prefill_tokens + input_tokens:
        if token not in tok_to_idx_map:
            raise ValueError(f"token {token!r} not present in model vocabulary")
        token_ids.append(tok_to_idx_map[token])

    predicted = raw_tokens
    logits = None
    for pos, token_id in enumerate(token_ids):
        logits = model.step_logits(token_id, pos, cache)
        mx.eval(logits)

    if logits is None:
        return predicted

    for gen_idx in range(max_new_tokens):
        next_id = int(mx.argmax(logits).item())
        next_token = all_tokens[next_id]
        predicted.append(next_token)
        if next_id == model.stop_token_id:
            break
        logits = model.step_logits(next_id, len(token_ids) + gen_idx, cache)
        mx.async_eval(logits)

    return predicted


def run_model_program(
    model,
    all_tokens,
    tok_to_idx_map,
    program_file,
    ref_file=None,
    max_new_tokens=50000,
    verbose=False,
    cache_class=OptimizedKVCache,
):
    tokens = Path(program_file).read_text().split()
    predicted = run_model_tokens(
        model,
        all_tokens,
        tok_to_idx_map,
        tokens,
        max_new_tokens=max_new_tokens,
        cache_class=cache_class,
    )
    n_tok = len(predicted)
    n_ops = sum(1 for token in predicted if token == "branch_taken" or token.startswith("commit("))

    if verbose:
        logger.info("  tokens: %s", " ".join(predicted))

    if ref_file is None or not Path(ref_file).exists():
        return True, n_tok, n_ops

    ref_tokens = Path(ref_file).read_text().split()
    ok, mismatch = _compare_with_ref(predicted, ref_tokens)
    if not ok and mismatch is not None:
        index, predicted_token, ref_token = mismatch
        logger.warning(
            "  mismatch at position %d: predicted=%s expected=%s",
            index,
            predicted_token,
            ref_token,
        )
    elif ok and len(predicted) < len(ref_tokens):
        logger.info("  output truncated: %d/%d tokens", n_tok, len(ref_tokens))
    return ok, n_tok, n_ops


def _resolve_program_file(path):
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path
    if suffix in {".c", ".wasm"}:
        txt_path, _spec_path, _input_base = compile_program(str(file_path))
        return Path(txt_path)
    raise ValueError(f"unsupported input file {path!r}; expected .txt, .c, or .wasm")


def _default_ref_file(program_file):
    program_path = Path(program_file)
    if program_path.name.endswith("_spec.txt"):
        return program_path.with_name(program_path.name[:-9] + "_ref.txt")
    return program_path.with_name(program_path.stem + "_ref.txt")


def _cache_class(nohull):
    if nohull:
        return StandardKVCache
    return OptimizedKVCache


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run token programs through the MLX transformer.")
    parser.add_argument("files", nargs="+", help="Program files (.txt, .c, or .wasm)")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to model weights (.bin); auto-built if missing",
    )
    parser.add_argument(
        "--nohull",
        action="store_true",
        help="Use the O(n) StandardKVCache instead of the hull cache",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print the full token trace")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50000,
        help="Max tokens to generate per program",
    )
    args = parser.parse_args()

    model_path = _ensure_model(args.model)
    model, all_tokens, tok_to_idx_map = load_weights(str(model_path))
    cache_class = _cache_class(args.nohull)

    passed = failed = skipped = 0
    total_tokens = total_ops = 0
    total_time = 0.0

    for original in args.files:
        program_file = _resolve_program_file(original)
        ref_file = _default_ref_file(program_file)
        has_ref = ref_file.exists()
        name = program_file.name

        start = time.time()
        ok, n_tok, n_ops = run_model_program(
            model,
            all_tokens,
            tok_to_idx_map,
            program_file,
            str(ref_file) if has_ref else None,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
            cache_class=cache_class,
        )
        elapsed = time.time() - start

        total_tokens += n_tok
        total_ops += n_ops
        total_time += elapsed

        if has_ref:
            logger.info(
                "%s: %s  %d tok, %d ops in %.2fs (%.0f tok/s)",
                name,
                "PASS" if ok else "FAIL",
                n_tok,
                n_ops,
                elapsed,
                n_tok / max(elapsed, 1e-9),
            )
            if ok:
                passed += 1
            else:
                failed += 1
        else:
            logger.info(
                "%s: RAN   %d tok, %d ops in %.2fs (%.0f tok/s)",
                name,
                n_tok,
                n_ops,
                elapsed,
                n_tok / max(elapsed, 1e-9),
            )
            skipped += 1

    logger.info("%d passed, %d failed, %d no-ref", passed, failed, skipped)
    if total_time > 0:
        logger.info("Benchmark: %d tok, %d ops, %.2fs", total_tokens, total_ops, total_time)
        logger.info(
            "  %.0f tok/s, %.0f wasm-ops/s",
            total_tokens / total_time,
            total_ops / total_time if total_ops else 0,
        )
        logger.info("  %.1fM FLOPs/tok", flops_per_token(model) / 1e6)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

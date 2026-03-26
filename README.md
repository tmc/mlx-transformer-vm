# mlx-transformer-vm

`mlx-transformer-vm` is a standalone Python port of Percepta's
[`transformer-vm`](https://github.com/Percepta-Core/transformer-vm) with the
compiler stack kept in Python and the eventual model runtime targeted at MLX.

The current focus is semantic parity for the computation graph, exact
evaluator, and WASM machine construction.

## Status

Ported now:

- graph DSL in [`mlx_transformer_vm/graph/core.py`](mlx_transformer_vm/graph/core.py)
- exact graph evaluator in [`mlx_transformer_vm/evaluator.py`](mlx_transformer_vm/evaluator.py)
- WASM machine graph builder in [`mlx_transformer_vm/wasm/interpreter.py`](mlx_transformer_vm/wasm/interpreter.py)
- reference WASM interpreter in [`mlx_transformer_vm/wasm/reference.py`](mlx_transformer_vm/wasm/reference.py)
- deterministic scheduler in [`mlx_transformer_vm/scheduler/deterministic.py`](mlx_transformer_vm/scheduler/deterministic.py)
- MILP scheduler in [`mlx_transformer_vm/scheduler/milp.py`](mlx_transformer_vm/scheduler/milp.py)
- MLX standard-cache runtime skeleton in [`mlx_transformer_vm/model/transformer.py`](mlx_transformer_vm/model/transformer.py) and [`mlx_transformer_vm/attention/standard_cache.py`](mlx_transformer_vm/attention/standard_cache.py)
- hull KV cache bridge in [`mlx_transformer_vm/attention/hull_cache.py`](mlx_transformer_vm/attention/hull_cache.py)
- analytical weight construction in [`mlx_transformer_vm/model/weights.py`](mlx_transformer_vm/model/weights.py)
- local parity harness in [`mlx_transformer_vm/parity.py`](mlx_transformer_vm/parity.py), compiling vendored examples and optionally checking the upstream weighted runtime as an oracle
- local compiler fixtures in [`examples/hello.c`](examples/hello.c), [`examples/addition.c`](examples/addition.c), [`examples/collatz.c`](examples/collatz.c), and [`examples/fibonacci.c`](examples/fibonacci.c)
- CLI parity for `wasm-build`, `wasm-run`, `wasm-compile`, and `wasm-specialize`

## Upstream Mapping

The port stays structurally close to upstream:

| Upstream | This repo |
| --- | --- |
| `transformer_vm/graph/core.py` | `mlx_transformer_vm/graph/core.py` |
| `transformer_vm/evaluator.py` | `mlx_transformer_vm/evaluator.py` |
| `transformer_vm/wasm/interpreter.py` | `mlx_transformer_vm/wasm/interpreter.py` |
| `transformer_vm/wasm/reference.py` | `mlx_transformer_vm/wasm/reference.py` |
| `transformer_vm/model/weights.py` | `mlx_transformer_vm/model/weights.py` |
| `transformer_vm/model/transformer.py` | `mlx_transformer_vm/model/transformer.py` |
| `transformer_vm/scheduler/milp.py` | `mlx_transformer_vm/scheduler/milp.py` |
| `transformer_vm/compilation/*` | `mlx_transformer_vm/compilation/` |

## Development

Install dependencies:

```bash
uv sync --extra dev
```

Run the fast unit tests:

```bash
uv run pytest mlx_transformer_vm/tests/test_graph_core.py
```

Run the parity tests against the vendored local examples:

```bash
uv run pytest mlx_transformer_vm/tests/test_parity.py
```

Run the evaluator on a compiled token program:

```bash
uv run wasm-eval-mlx /path/to/program.txt
```

Diff this port against the local examples:

```bash
uv run tvm-mlx-parity --examples hello addition collatz fibonacci
```

Enable the optional upstream weighted oracle explicitly:

```bash
uv run tvm-mlx-parity --weighted --examples hello addition collatz fibonacci
```

The upstream repository is optional and is only used for the weighted-oracle
comparison path in [`mlx_transformer_vm/parity.py`](mlx_transformer_vm/parity.py).
Set `TRANSFORMER_VM_UPSTREAM_ROOT` to enable that path.

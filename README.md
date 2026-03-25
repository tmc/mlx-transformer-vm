# mlx-transformer-vm

`mlx-transformer-vm` is a standalone Python port of Percepta's
[`transformer-vm`](https://github.com/Percepta-Core/transformer-vm) with the
compiler stack kept in Python and the eventual model runtime targeted at MLX.

The current focus is semantic parity for the computation graph, exact
evaluator, and WASM machine construction.

## Status

Ported now:

- graph DSL in [`mlx_transformer_vm/graph/core.py`](/Volumes/tmc/go/src/github.com/tmc/mlx-transformer-vm/mlx_transformer_vm/graph/core.py)
- exact graph evaluator in [`mlx_transformer_vm/evaluator.py`](/Volumes/tmc/go/src/github.com/tmc/mlx-transformer-vm/mlx_transformer_vm/evaluator.py)
- WASM machine graph builder in [`mlx_transformer_vm/wasm/interpreter.py`](/Volumes/tmc/go/src/github.com/tmc/mlx-transformer-vm/mlx_transformer_vm/wasm/interpreter.py)
- reference WASM interpreter in [`mlx_transformer_vm/wasm/reference.py`](/Volumes/tmc/go/src/github.com/tmc/mlx-transformer-vm/mlx_transformer_vm/wasm/reference.py)
- upstream parity harness in [`mlx_transformer_vm/parity.py`](/Volumes/tmc/go/src/github.com/tmc/mlx-transformer-vm/mlx_transformer_vm/parity.py)

Still missing:

- scheduler/allocation parity
- analytical weight construction
- MLX transformer runtime
- standard and hull KV caches
- specialization path
- full CLI parity for build/run/compile/specialize

## Upstream Mapping

The port stays structurally close to upstream:

| Upstream | This repo |
| --- | --- |
| `transformer_vm/graph/core.py` | `mlx_transformer_vm/graph/core.py` |
| `transformer_vm/evaluator.py` | `mlx_transformer_vm/evaluator.py` |
| `transformer_vm/wasm/interpreter.py` | `mlx_transformer_vm/wasm/interpreter.py` |
| `transformer_vm/wasm/reference.py` | `mlx_transformer_vm/wasm/reference.py` |
| `transformer_vm/model/weights.py` | `mlx_transformer_vm/model/weights.py` (planned) |
| `transformer_vm/model/transformer.py` | `mlx_transformer_vm/model/transformer.py` (planned) |
| `transformer_vm/scheduler/milp.py` | `mlx_transformer_vm/scheduler/` (planned) |
| `transformer_vm/compilation/*` | `mlx_transformer_vm/compilation/` (planned) |

## Development

Install dependencies:

```bash
uv sync --extra dev
```

Run the fast unit tests:

```bash
uv run pytest mlx_transformer_vm/tests/test_graph_core.py
```

Run the parity tests against the upstream repository:

```bash
uv run pytest mlx_transformer_vm/tests/test_parity.py
```

Run the evaluator on a compiled token program:

```bash
uv run wasm-eval-mlx /path/to/program.txt
```

Diff this port against the upstream examples:

```bash
uv run tvm-mlx-parity --examples hello addition collatz fibonacci
```

The parity harness expects the upstream repository at
`/Users/tmc/go/src/github.com/Percepta-Core/transformer-vm`. Set
`TRANSFORMER_VM_UPSTREAM_ROOT` to override that path.

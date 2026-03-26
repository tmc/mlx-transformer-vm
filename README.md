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

## Architecture: Precision and the Float32 Boundary

The system has two distinct execution paths with different precision
requirements:

**MLX transformer runner** (`runner.py`, `model/transformer.py`): Runs the
learned transformer on Apple Silicon via MLX float32. This path computes
dot-product scores and argmax for hardmax attention — float32 is sufficient
for these operations at the model dimensions we use (d_model~36, n_heads~18).

**Exact graph evaluator** (`evaluator.py`): Evaluates the computation graph
with exact arithmetic using Python float64. The convex hull cache that
provides O(log n) attention lookups operates entirely in float64 (Python
floats or C++ double), **not** on the GPU.

This separation is load-bearing. The hull's key geometry uses parabolic
coordinates `(2k, -k^2)` where cross-product sign determines hull
membership. At sequence position k=4095, float32 cross-products lose exact
magnitude; at k=4097, the sign itself becomes unreliable (~58% error rate
above k=4096). A GPU-only float32 convex hull would silently corrupt
results for sequences longer than ~4k tokens.

The C++ pybind11 extension (`hull_cache.py`) is optional — a pure Python
hull (`hull_python.py`) provides identical correctness at ~9x slower
throughput. Both use float64. The C++ extension remains available for
performance-sensitive workloads but is not required.

See `tests/test_precision_limits.py` for empirical verification of these
bounds.

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

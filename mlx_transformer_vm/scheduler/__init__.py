"""Scheduling and allocation helpers.

The MILP scheduler is not ported yet. The deterministic scheduler provides a
compatible first-pass schedule shape for downstream weight construction.
"""

from mlx_transformer_vm.scheduler.deterministic import (
    deterministic_schedule,
    interval_coloring,
)

__all__ = [
    "deterministic_schedule",
    "interval_coloring",
]

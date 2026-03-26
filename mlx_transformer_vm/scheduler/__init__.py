"""Scheduling and allocation helpers.

The deterministic scheduler provides a conservative schedule shape. The MILP
scheduler matches the upstream width optimizer.
"""

from mlx_transformer_vm.scheduler.deterministic import (
    deterministic_schedule,
    interval_coloring,
)
from mlx_transformer_vm.scheduler.milp import milp_schedule

__all__ = [
    "deterministic_schedule",
    "interval_coloring",
    "milp_schedule",
]

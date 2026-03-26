"""Precision guardrail: document where float32 breaks convex hull geometry.

The evaluator's convex hull operates on Python float64 (or C++ double),
NOT MLX float32.  This test exists to prevent anyone from moving the hull
computation onto the GPU.

For our key geometry point(k) = (2k, -k^2), the 2D cross-product of three
consecutive points should always be -4.  Float32 loses exact magnitude at
k=4095 and first produces a sign error (0.0 instead of -4.0) at k=4097.
Above k=4096, ~58% of cross-product evaluations return the wrong sign,
which would silently corrupt the convex hull envelope.
"""

from __future__ import annotations

import mlx.core as mx
import pytest


def _cross_f64(k):
    """Cross-product of three consecutive parabolic key points in float64."""
    k1, k2, k3 = k, k + 1, k + 2
    ax, ay = 2.0 * k1, -(float(k1) ** 2)
    bx, by = 2.0 * k2, -(float(k2) ** 2)
    cx, cy = 2.0 * k3, -(float(k3) ** 2)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _cross_f32(k):
    """Same cross-product computed in MLX float32."""
    k1, k2, k3 = k, k + 1, k + 2
    ax = mx.array(2.0 * k1, dtype=mx.float32)
    ay = mx.array(-(float(k1) ** 2), dtype=mx.float32)
    bx = mx.array(2.0 * k2, dtype=mx.float32)
    by = mx.array(-(float(k2) ** 2), dtype=mx.float32)
    cx = mx.array(2.0 * k3, dtype=mx.float32)
    cy = mx.array(-(float(k3) ** 2), dtype=mx.float32)
    return ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)).item()


def test_float64_cross_product_always_minus_four():
    """Float64 produces exact -4 for all parabolic key triples."""
    for k in [100, 1000, 4096, 10000, 50000]:
        assert _cross_f64(k) == -4.0, f"float64 failed at k={k}"


@pytest.mark.parametrize("k", [100, 500, 1000, 2048, 4000, 4094])
def test_float32_exact_below_boundary(k):
    """Below k=4095, float32 still produces the exact -4 result."""
    assert _cross_f32(k) == -4.0


def test_float32_magnitude_error_at_4095():
    """At k=4095, float32 magnitude is wrong but sign is still negative."""
    result = _cross_f32(4095)
    assert result != -4.0, f"expected magnitude error, got {result}"
    assert result < 0, f"expected negative sign preserved, got {result}"


def test_float32_first_sign_failure_at_4097():
    """At k=4097, float32 cross-product is 0 — first sign failure."""
    result = _cross_f32(4097)
    assert result == 0.0, f"expected 0.0 at k=4097, got {result}"


def test_float32_sign_failure_rate():
    """Above k=4096, float32 sign errors are frequent (~58%).

    This quantifies why a GPU-only convex hull is infeasible for
    sequences longer than ~4k tokens with our key geometry.
    """
    sign_failures = 0
    total = 0
    for k in range(4096, 10001):
        total += 1
        if _cross_f32(k) >= 0:
            sign_failures += 1

    rate = sign_failures / total
    # Observed: ~57.7%.  Assert >40% to be robust to minor platform variance.
    assert rate > 0.40, f"sign failure rate {rate:.1%} lower than expected"
    # But also <80% — it's not universal failure, some values still work.
    assert rate < 0.80, f"sign failure rate {rate:.1%} higher than expected"

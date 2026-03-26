"""Pure Python convex hull KV cache -- drop-in replacement for the C++ extension.

Ports the CHT (convex hull trick) algorithm from hull2d_cht.h to Python.
Operates on Python floats for exact arithmetic; no MLX arrays internally.
"""

from __future__ import annotations

import bisect

import mlx.core as mx
import numpy as np

AVERAGE = 0
LATEST = 1

INF = float("inf")


class HullMeta:
    """Accumulates value metadata for tie-breaking resolution."""

    __slots__ = ("vsum0", "vsum1", "vlast0", "vlast1", "count", "last_seq")

    def __init__(self):
        self.vsum0 = 0.0
        self.vsum1 = 0.0
        self.vlast0 = 0.0
        self.vlast1 = 0.0
        self.count = 0
        self.last_seq = -1

    def add(self, v0: float, v1: float, seq: int = 0) -> None:
        self.vsum0 += v0
        self.vsum1 += v1
        self.count += 1
        if seq > self.last_seq:
            self.last_seq = seq
            self.vlast0 = v0
            self.vlast1 = v1

    def merge(self, other: HullMeta) -> None:
        self.vsum0 += other.vsum0
        self.vsum1 += other.vsum1
        self.count += other.count
        if other.last_seq > self.last_seq:
            self.last_seq = other.last_seq
            self.vlast0 = other.vlast0
            self.vlast1 = other.vlast1

    def resolve(self, tb: int) -> tuple[float, float]:
        if self.count == 0:
            return (0.0, 0.0)
        if tb == LATEST:
            return (self.vlast0, self.vlast1)
        inv = 1.0 / self.count
        return (self.vsum0 * inv, self.vsum1 * inv)

    def copy(self) -> HullMeta:
        m = HullMeta()
        m.vsum0 = self.vsum0
        m.vsum1 = self.vsum1
        m.vlast0 = self.vlast0
        m.vlast1 = self.vlast1
        m.count = self.count
        m.last_seq = self.last_seq
        return m


class PythonHullCHT:
    """Dynamic upper envelope of lines y = m*x + b.

    Faithful port of _HullCHT from hull2d_cht.h. Lines are stored sorted by
    slope (m). Each line has a breakpoint (p) indicating where it starts being
    optimal. The last line has p = +inf. Queries use bisect on p for O(log n).

    The C++ uses a multiset with heterogeneous lookup. Here we maintain parallel
    sorted lists for m, b, p, meta.
    """

    def __init__(self):
        self._m: list[float] = []
        self._b: list[float] = []
        self._p: list[float] = []
        self._meta: list[HullMeta] = []

    def empty(self) -> bool:
        return len(self._m) == 0

    def size(self) -> int:
        return len(self._m)

    def clear(self) -> None:
        self._m.clear()
        self._b.clear()
        self._p.clear()
        self._meta.clear()

    def _compute_p(self, i: int, j: int) -> float:
        """Intersection x of lines i and j. Does NOT store result."""
        if j >= len(self._m):
            return INF
        mi, bi = self._m[i], self._b[i]
        mj, bj = self._m[j], self._b[j]
        if mi == mj:
            return INF if bi >= bj else -INF
        return (bj - bi) / (mi - mj)

    def _set_p(self, i: int) -> None:
        """Compute and store p[i] = intersection of line i with line i+1."""
        self._p[i] = self._compute_p(i, i + 1)

    def _remove(self, i: int) -> None:
        self._m.pop(i)
        self._b.pop(i)
        self._p.pop(i)
        self._meta.pop(i)

    def add_line(self, m: float, b: float, meta: HullMeta) -> None:
        """Insert a line y = m*x + b with metadata, maintaining the upper envelope.

        Faithfully mirrors _HullCHT::add_line from hull2d_cht.h.
        """
        n = len(self._m)
        pos = bisect.bisect_left(self._m, m)

        # Handle existing line(s) with same slope.
        # C++ checks at lower_bound position and its predecessor.
        if pos < n and self._m[pos] == m:
            if self._b[pos] == b:
                self._meta[pos].merge(meta)
                return
            if self._b[pos] >= b:
                return
            self._remove(pos)
        elif pos > 0 and self._m[pos - 1] == m:
            pos -= 1
            if self._b[pos] == b:
                self._meta[pos].merge(meta)
                return
            if self._b[pos] >= b:
                return
            self._remove(pos)

        # Insert the new line at pos.
        self._m.insert(pos, m)
        self._b.insert(pos, b)
        self._p.insert(pos, 0.0)
        self._meta.insert(pos, meta)

        # y = pos (newly inserted line), z = y + 1.
        y = pos

        # --- Right removal ---
        # C++: while (isect(y, z)) { z = lines.erase(z); }
        # isect(y, z) sets y->p = intersection(y, z) and returns y->p >= z->p.
        while True:
            self._set_p(y)  # y->p = intersection with successor
            if y + 1 < len(self._m) and self._p[y] >= self._p[y + 1]:
                self._remove(y + 1)
            else:
                break

        # --- Left fix and left removal ---
        # C++ code:
        #   if (x != begin && isect(--x, y)) {
        #       isect(x, y = erase(y));
        #   }
        #   while ((y = x) != begin && (--x)->p >= y->p) {
        #       isect(x, erase(y));
        #   }
        #
        # x starts equal to y (the new line). --x moves to predecessor.
        # isect(x, y) sets x->p. If x->p >= y->p, y is dominated, erase y.
        # Then the while loop continues leftward: y becomes x (predecessor),
        # x becomes prepredecessor, checking x->p >= y->p.
        if y > 0:
            x = y - 1
            self._set_p(x)  # x->p = intersection(x, y)
            if self._p[x] >= self._p[y]:
                # y is dominated — erase it, recompute x->p with new successor.
                self._remove(y)
                y = x
                self._set_p(y)

            # Left removal: y = x, then check x's predecessor.
            # After the if block, regardless of whether it fired:
            #   - If it fired: y = x (predecessor of original insertion), x = y-1
            #   - If it didn't: y was unchanged, x = y-1 (predecessor)
            # In both cases, we now set y = x and continue leftward.
            # But in the "didn't fire" case, the C++ code still enters the
            # while loop with y = x (predecessor, whose p was just set).
            y = x  # y = x from above (the predecessor with updated p)
            while y > 0:
                x = y - 1
                # x->p was set by a previous isect or is from the hull invariant.
                if self._p[x] >= self._p[y]:
                    self._remove(y)
                    y = x
                    self._set_p(y)
                else:
                    break

    def argmax(self, x: float) -> int:
        """Return index of the line with maximum value at query point x.

        Uses lower_bound on breakpoints (same semantics as C++ argmax).
        lower_bound finds first i where p[i] >= x. That's the optimal line.
        If past end, use the last line.
        """
        if not self._m:
            raise ValueError("empty hull")
        idx = bisect.bisect_left(self._p, x)
        if idx >= len(self._m):
            idx = len(self._m) - 1
        return idx


class HullHalf:
    """Upper or lower half of hull for 2D hard attention."""

    def __init__(self, is_upper: bool = True):
        self.cht = PythonHullCHT()
        self.is_upper = is_upper

    def size(self) -> int:
        return self.cht.size()

    def clear(self) -> None:
        self.cht.clear()

    def insert(self, kx: float, ky: float, v0: float, v1: float, seq: int = 0) -> None:
        meta = HullMeta()
        meta.add(v0, v1, seq)
        if self.is_upper:
            self.cht.add_line(kx, ky, meta)
        else:
            self.cht.add_line(-kx, -ky, meta)

    def query(self, qx: float, qy: float, tb: int) -> tuple[bool, float, float]:
        if self.cht.empty():
            return (False, 0.0, 0.0)

        if qy == 0.0:
            x = INF if qx >= 0 else -INF
            idx = self.cht.argmax(x)
            out = self.cht._meta[idx].resolve(tb)
            return (True, out[0], out[1])

        m = qx / qy
        best_idx = self.cht.argmax(m)

        if self.is_upper:
            kx_best = self.cht._m[best_idx]
            ky_best = self.cht._b[best_idx]
        else:
            kx_best = -self.cht._m[best_idx]
            ky_best = -self.cht._b[best_idx]
        best_score = qx * kx_best + qy * ky_best

        combined = self.cht._meta[best_idx].copy()

        # Scan left neighbors for tied lines (same dot product score).
        i = best_idx - 1
        while i >= 0:
            if self.is_upper:
                kx_p = self.cht._m[i]
                ky_p = self.cht._b[i]
            else:
                kx_p = -self.cht._m[i]
                ky_p = -self.cht._b[i]
            s = qx * kx_p + qy * ky_p
            if s == best_score:
                combined.merge(self.cht._meta[i])
                i -= 1
            else:
                break

        # Scan right neighbors for tied lines.
        i = best_idx + 1
        while i < self.cht.size():
            if self.is_upper:
                kx_p = self.cht._m[i]
                ky_p = self.cht._b[i]
            else:
                kx_p = -self.cht._m[i]
                ky_p = -self.cht._b[i]
            s = qx * kx_p + qy * ky_p
            if s == best_score:
                combined.merge(self.cht._meta[i])
                i += 1
            else:
                break

        out = combined.resolve(tb)
        return (True, out[0], out[1])


class HardAttentionHead:
    """2D hard attention head with upper/lower half-hulls."""

    def __init__(self):
        self.upper = HullHalf(is_upper=True)
        self.lower = HullHalf(is_upper=False)
        self.global_meta = HullMeta()
        self.left_meta = HullMeta()
        self.right_meta = HullMeta()
        self.min_kx = INF
        self.max_kx = -INF
        self.n = 0

    def clear(self) -> None:
        self.upper.clear()
        self.lower.clear()
        self.global_meta = HullMeta()
        self.left_meta = HullMeta()
        self.right_meta = HullMeta()
        self.min_kx = INF
        self.max_kx = -INF
        self.n = 0

    def insert(self, kx: float, ky: float, v0: float, v1: float, seq: int = 0) -> None:
        self.global_meta.add(v0, v1, seq)

        if kx < self.min_kx:
            self.min_kx = kx
            self.left_meta = HullMeta()
        if kx == self.min_kx:
            self.left_meta.add(v0, v1, seq)

        if kx > self.max_kx:
            self.max_kx = kx
            self.right_meta = HullMeta()
        if kx == self.max_kx:
            self.right_meta.add(v0, v1, seq)

        self.upper.insert(kx, ky, v0, v1, seq)
        self.lower.insert(kx, ky, v0, v1, seq)
        self.n += 1

    def query(self, qx: float, qy: float, tb: int) -> tuple[float, float]:
        if qy == 0.0:
            if qx > 0.0:
                return self.right_meta.resolve(tb)
            elif qx < 0.0:
                return self.left_meta.resolve(tb)
            else:
                return self.global_meta.resolve(tb)

        if qy > 0.0:
            _, o0, o1 = self.upper.query(qx, qy, tb)
            return (o0, o1)
        _, o0, o1 = self.lower.query(qx, qy, tb)
        return (o0, o1)


class PythonHullKVCache:
    """Pure Python O(log n) hard-attention KV cache -- drop-in for HullKVCache."""

    def __init__(self, n_layers: int, n_heads: int):
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._heads = [HardAttentionHead() for _ in range(n_layers * n_heads)]
        self._tb = [AVERAGE] * (n_layers * n_heads)
        self._seq = -1

    def clear(self) -> None:
        for h in self._heads:
            h.clear()
        self._seq = -1

    def set_tiebreak(self, layer: int, head: int, latest: bool) -> None:
        self._tb[layer * self._n_heads + head] = LATEST if latest else AVERAGE

    def layer_step(self, layer: int, keys, queries, values):
        self._seq += 1
        kp = np.asarray(keys, dtype=np.float64).reshape(self._n_heads, 2)
        qp = np.asarray(queries, dtype=np.float64).reshape(self._n_heads, 2)
        vp = np.asarray(values, dtype=np.float64).reshape(self._n_heads, 2)

        out = np.empty((self._n_heads, 2), dtype=np.float64)
        base = layer * self._n_heads
        for head in range(self._n_heads):
            h = self._heads[base + head]
            h.insert(
                float(kp[head, 0]), float(kp[head, 1]),
                float(vp[head, 0]), float(vp[head, 1]),
                self._seq,
            )
            o0, o1 = h.query(
                float(qp[head, 0]), float(qp[head, 1]),
                self._tb[base + head],
            )
            out[head, 0] = o0
            out[head, 1] = o1

        return mx.array(out, dtype=mx.float32).reshape((-1,))

#pragma once
/*
 * hull2d_cht.h — Drop-in alternative to hull2d.h
 *
 * What it does:
 *   Incremental-only 2D point structure that answers hard-attention queries:
 *     argmax_k  q · k
 *   with the same public API as hull2d.h (HardAttentionHead / BruteAttentionHead).
 *
 * Key idea (duality / 1D reduction):
 *   For q=(qx,qy) with qy != 0, maximizing q·(kx,ky) is equivalent to
 *     maximizing (kx * m + ky) for m = qx/qy  if qy > 0
 *     minimizing (kx * m + ky)               if qy < 0
 *
 *   So we maintain two dynamic 1D convex envelopes (dynamic CHT):
 *     - upper:  max   (kx*m + ky)
 *     - lower:  min   (kx*m + ky)  implemented as max of (-(kx*m + ky))
 *
 * Performance profile vs hull2d.h:
 *   - Query: O(log h) in both.
 *   - Insert:
 *       hull2d.h  : geometric walk + vector erase/insert => can cost O(h) memmove.
 *       this file : dynamic CHT => amortized O(log h) with no bulk memmove.
 */

#include <algorithm>
#include <cassert>
#include <limits>
#include <set>
#include <vector>

enum class TieBreak { AVERAGE, LATEST };

struct HullMeta {
    double vsum[2] = {0, 0};
    double vlast[2] = {0, 0};
    int count = 0;
    int last_seq = -1;

    void add(const double v[2], int seq = 0) {
        vsum[0] += v[0];
        vsum[1] += v[1];
        count++;
        if (seq > last_seq) {
            last_seq = seq;
            vlast[0] = v[0];
            vlast[1] = v[1];
        }
    }

    void merge(const HullMeta& other) {
        vsum[0] += other.vsum[0];
        vsum[1] += other.vsum[1];
        count += other.count;
        if (other.last_seq > last_seq) {
            last_seq = other.last_seq;
            vlast[0] = other.vlast[0];
            vlast[1] = other.vlast[1];
        }
    }

    void resolve(TieBreak tb, double out[2]) const {
        if (count == 0) {
            out[0] = out[1] = 0;
            return;
        }
        if (tb == TieBreak::LATEST) {
            out[0] = vlast[0];
            out[1] = vlast[1];
            return;
        }
        double inv = 1.0 / count;
        out[0] = vsum[0] * inv;
        out[1] = vsum[1] * inv;
    }
};

struct _HullCHT {
    struct Line {
        double m = 0.0;
        double b = 0.0;
        mutable long double p = 0.0L;
        HullMeta meta;

        bool operator<(const Line& other) const { return m < other.m; }
        bool operator<(long double x) const { return p < x; }
    };

    using Set = std::multiset<Line, std::less<>>;
    using It = Set::iterator;

    Set lines;

    static constexpr long double INF = std::numeric_limits<long double>::infinity();

    bool isect(It x, It y) {
        if (y == lines.end()) {
            x->p = INF;
            return false;
        }

        if (x->m == y->m) {
            x->p = (x->b >= y->b ? INF : -INF);
        } else {
            x->p = ((long double)y->b - (long double)x->b) /
                   ((long double)x->m - (long double)y->m);
        }
        return x->p >= y->p;
    }

    void add_line(double m, double b, const HullMeta& meta) {
        Line nl;
        nl.m = m;
        nl.b = b;
        nl.meta = meta;

        auto it = lines.lower_bound(nl);
        if (it != lines.end() && it->m == m) {
            if (it->b == b) {
                Line merged = *it;
                merged.meta.merge(meta);
                lines.erase(it);
                nl = merged;
            } else if (it->b >= b) {
                return;
            } else {
                lines.erase(it);
            }
        } else if (it != lines.begin()) {
            auto it2 = std::prev(it);
            if (it2->m == m) {
                if (it2->b == b) {
                    Line merged = *it2;
                    merged.meta.merge(meta);
                    lines.erase(it2);
                    nl = merged;
                } else if (it2->b >= b) {
                    return;
                } else {
                    lines.erase(it2);
                }
            }
        }

        auto z = lines.insert(nl);
        auto y = z++;
        auto x = y;

        while (isect(y, z)) {
            z = lines.erase(z);
        }

        if (x != lines.begin() && isect(--x, y)) {
            isect(x, y = lines.erase(y));
        }

        while ((y = x) != lines.begin() && (--x)->p >= y->p) {
            isect(x, lines.erase(y));
        }
    }

    bool empty() const { return lines.empty(); }
    int size() const { return (int)lines.size(); }
    void clear() { lines.clear(); }

    It argmax(long double x) {
        assert(!lines.empty());
        auto it = lines.lower_bound(x);
        if (it == lines.end()) {
            it = std::prev(lines.end());
        }
        return it;
    }

    It argmax(long double x) const {
        assert(!lines.empty());
        auto it = lines.lower_bound(x);
        if (it == lines.end()) {
            it = std::prev(lines.end());
        }
        return it;
    }
};

struct HullHalf {
    _HullCHT cht;
    bool is_upper;

    explicit HullHalf(bool upper = true) : is_upper(upper) {}

    int size() const { return cht.size(); }
    void clear() { cht.clear(); }

    void insert(double kx, double ky, const double val[2], int seq = 0) {
        HullMeta meta;
        meta.add(val, seq);
        if (is_upper) {
            cht.add_line(kx, ky, meta);
        } else {
            cht.add_line(-kx, -ky, meta);
        }
    }

    bool query(double qx, double qy, TieBreak tb, double out[2], double* score_out = nullptr,
               double* best_kx_out = nullptr) const {
        if (cht.empty()) {
            return false;
        }

        if (qy == 0.0) {
            long double x = (qx >= 0 ? _HullCHT::INF : -_HullCHT::INF);
            auto it = cht.argmax(x);
            it->meta.resolve(tb, out);
            double kx_best = is_upper ? it->m : -it->m;
            if (score_out) {
                double ky_best = is_upper ? it->b : -it->b;
                *score_out = qx * kx_best + qy * ky_best;
            }
            if (best_kx_out) {
                *best_kx_out = kx_best;
            }
            return true;
        }

        long double m = (long double)qx / (long double)qy;
        auto best_it = cht.argmax(m);

        double kx_best = is_upper ? best_it->m : -best_it->m;
        double ky_best = is_upper ? best_it->b : -best_it->b;
        double best_score = qx * kx_best + qy * ky_best;

        HullMeta combined;
        combined.merge(best_it->meta);

        auto itL = best_it;
        while (itL != cht.lines.begin()) {
            auto prev = std::prev(itL);
            double kx_p = is_upper ? prev->m : -prev->m;
            double ky_p = is_upper ? prev->b : -prev->b;
            double s = qx * kx_p + qy * ky_p;
            if (s == best_score) {
                combined.merge(prev->meta);
                itL = prev;
            } else {
                break;
            }
        }

        auto itR = std::next(best_it);
        while (itR != cht.lines.end()) {
            double kx_p = is_upper ? itR->m : -itR->m;
            double ky_p = is_upper ? itR->b : -itR->b;
            double s = qx * kx_p + qy * ky_p;
            if (s == best_score) {
                combined.merge(itR->meta);
                ++itR;
            } else {
                break;
            }
        }

        combined.resolve(tb, out);
        if (score_out) {
            *score_out = best_score;
        }
        if (best_kx_out) {
            *best_kx_out = kx_best;
        }
        return true;
    }
};

struct HardAttentionHead {
    HullHalf upper{true};
    HullHalf lower{false};
    HullMeta global;
    HullMeta left_meta;
    HullMeta right_meta;
    double min_kx = std::numeric_limits<double>::infinity();
    double max_kx = -std::numeric_limits<double>::infinity();
    int n = 0;

    void clear() {
        upper.clear();
        lower.clear();
        global = {};
        left_meta = {};
        right_meta = {};
        min_kx = std::numeric_limits<double>::infinity();
        max_kx = -std::numeric_limits<double>::infinity();
        n = 0;
    }

    int size() const { return n; }

    void insert(const double key[2], const double val[2], int seq = 0) {
        global.add(val, seq);

        if (key[0] < min_kx) {
            min_kx = key[0];
            left_meta = {};
        }
        if (key[0] == min_kx) {
            left_meta.add(val, seq);
        }

        if (key[0] > max_kx) {
            max_kx = key[0];
            right_meta = {};
        }
        if (key[0] == max_kx) {
            right_meta.add(val, seq);
        }

        upper.insert(key[0], key[1], val, seq);
        lower.insert(key[0], key[1], val, seq);
        n++;
    }

    bool query(const double q[2], TieBreak tb, double out[2]) const {
        double qx = q[0];
        double qy = q[1];
        if (qy == 0.0) {
            if (qx > 0.0) {
                right_meta.resolve(tb, out);
            } else if (qx < 0.0) {
                left_meta.resolve(tb, out);
            } else {
                global.resolve(tb, out);
            }
            return true;
        }

        if (qy > 0.0) {
            return upper.query(qx, qy, tb, out);
        }
        return lower.query(qx, qy, tb, out);
    }
};

struct BruteAttentionHead {
    struct Entry {
        double kx;
        double ky;
        double vx;
        double vy;
        int seq;
    };
    std::vector<Entry> entries;

    void clear() { entries.clear(); }
    int size() const { return (int)entries.size(); }

    void insert(const double key[2], const double val[2], int seq) {
        entries.push_back({key[0], key[1], val[0], val[1], seq});
    }

    bool query(const double q[2], TieBreak tb, double out[2]) const {
        if (entries.empty()) {
            out[0] = out[1] = 0;
            return false;
        }

        double qx = q[0];
        double qy = q[1];
        double max_score = -std::numeric_limits<double>::infinity();
        for (const auto& entry : entries) {
            double score = qx * entry.kx + qy * entry.ky;
            if (score > max_score) {
                max_score = score;
            }
        }

        HullMeta meta;
        for (const auto& entry : entries) {
            double score = qx * entry.kx + qy * entry.ky;
            if (score == max_score) {
                double value[2] = {entry.vx, entry.vy};
                meta.add(value, entry.seq);
            }
        }
        meta.resolve(tb, out);

        return true;
    }
};

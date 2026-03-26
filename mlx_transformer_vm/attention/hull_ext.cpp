#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "hull2d_cht.h"

namespace py = pybind11;

class HullKVCache {
    std::vector<HardAttentionHead> hulls_;
    std::vector<TieBreak> head_tb_;
    int n_layers_;
    int n_heads_;

   public:
    HullKVCache(int n_layers, int n_heads)
        : hulls_(n_layers * n_heads),
          head_tb_(n_layers * n_heads, TieBreak::AVERAGE),
          n_layers_(n_layers),
          n_heads_(n_heads) {}

    void clear() {
        for (auto& hull : hulls_) {
            hull.clear();
        }
    }

    void set_tiebreak(int layer, int head, int latest) {
        head_tb_[layer * n_heads_ + head] = latest ? TieBreak::LATEST : TieBreak::AVERAGE;
    }

    py::array_t<double> layer_step(
        int layer,
        py::array_t<double, py::array::c_style | py::array::forcecast> keys,
        py::array_t<double, py::array::c_style | py::array::forcecast> queries,
        py::array_t<double, py::array::c_style | py::array::forcecast> values,
        int seq) {
        const double* kp = keys.data();
        const double* qp = queries.data();
        const double* vp = values.data();

        auto result = py::array_t<double>({n_heads_, 2});
        double* op = result.mutable_data();

        int base = layer * n_heads_;
        for (int head = 0; head < n_heads_; head++) {
            hulls_[base + head].insert(&kp[head * 2], &vp[head * 2], seq);
            hulls_[base + head].query(&qp[head * 2], head_tb_[base + head], &op[head * 2]);
        }
        return result;
    }
};

PYBIND11_MODULE(_mlx_tvm_hull_ext, m) {
    m.doc() = "2D convex hull KV cache for O(log n) hard attention";
    py::class_<HullKVCache>(m, "HullKVCache")
        .def(py::init<int, int>(), py::arg("n_layers"), py::arg("n_heads"))
        .def("clear", &HullKVCache::clear)
        .def("set_tiebreak", &HullKVCache::set_tiebreak, py::arg("layer"), py::arg("head"),
             py::arg("latest"))
        .def("layer_step", &HullKVCache::layer_step, py::arg("layer"), py::arg("keys"),
             py::arg("queries"), py::arg("values"), py::arg("seq"));
}

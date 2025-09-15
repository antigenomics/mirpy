#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <algorithm>

namespace py = pybind11;

static inline int norm_pos(int p, int m) { if (p >= 0) return p > m ? m : p; int q = m + p; return q < 0 ? 0 : q; }

static inline double seg_equal(const char* s1, const char* s2, int start, int end,
                               const double* mat, bool use_mat) {
    double x = 0.0;
    for (int i = start; i < end; ++i) {
        unsigned char c1 = (unsigned char)s1[i], c2 = (unsigned char)s2[i];
        x += use_mat ? mat[(size_t)c1 * 256 + c2] : (c1 == c2 ? 0.0 : 1.0);
    }
    return x;
}

static inline double score_with_gap(const char* s1, int n1, const char* s2, int n2,
                                    int p_raw, int start, int end,
                                    double gap_pen, const double* mat, bool use_mat) {
    if (n1 == n2) return seg_equal(s1, s2, start, end, mat, use_mat);

    if (n1 < n2) {
        int gap_len = n2 - n1, p = norm_pos(p_raw, n1);
        int g0 = std::max(start, p), g1 = std::min(end, p + gap_len);
        double x = 0.0;
        x += seg_equal(s1, s2, start, g0, mat, use_mat);
        if (g1 > g0) x += (g1 - g0) * gap_pen;
        for (int i = g1; i < end; ++i) {
            int j = i - gap_len;
            unsigned char c1 = (unsigned char)s1[j], c2 = (unsigned char)s2[i];
            x += use_mat ? mat[(size_t)c1 * 256 + c2] : (c1 == c2 ? 0.0 : 1.0);
        }
        return x;
    } else {
        int gap_len = n1 - n2, p = norm_pos(p_raw, n2);
        int g0 = std::max(start, p), g1 = std::min(end, p + gap_len);
        double x = 0.0;
        x += seg_equal(s1, s2, start, g0, mat, use_mat);
        if (g1 > g0) x += (g1 - g0) * gap_pen;
        for (int i = g1; i < end; ++i) {
            int j = i - gap_len;
            unsigned char c1 = (unsigned char)s1[i], c2 = (unsigned char)s2[j];
            x += use_mat ? mat[(size_t)c1 * 256 + c2] : (c1 == c2 ? 0.0 : 1.0);
        }
        return x;
    }
}

double score_max(const std::string& s1, const std::string& s2,
                 py::array_t<double, py::array::c_style | py::array::forcecast> mat256,
                 py::array_t<int,    py::array::c_style | py::array::forcecast> gaps,
                 double gap_pen, int v_off, int j_off, double factor, bool use_mat) {
    const double* mat = nullptr;
    if (use_mat) {
        auto mbuf = mat256.request();
        if (mbuf.ndim != 2 || mbuf.shape[0] != 256 || mbuf.shape[1] != 256)
            throw std::runtime_error("mat must be 256x256");
        mat = static_cast<double*>(mbuf.ptr);
    }
    auto gb = gaps.request();
    const int* gp = static_cast<int*>(gb.ptr);
    int ng = (int)gb.shape[0];

    int L = std::max((int)s1.size(), (int)s2.size());
    int start = v_off, end = L - j_off;
    if (end <= start) return 0.0;

    double best = -1e300;
    {
        py::gil_scoped_release release;
        for (int k = 0; k < ng; ++k) {
            double sc = score_with_gap(s1.data(), (int)s1.size(), s2.data(), (int)s2.size(),
                                       gp[k], start, end, gap_pen, mat, use_mat);
            if (sc > best) best = sc;
        }
    }
    return factor * best;
}

double selfscore(const std::string& s,
                 py::array_t<double, py::array::c_style | py::array::forcecast> mat256,
                 double factor, bool use_mat) {
    if (!use_mat) return 0.0;
    auto mbuf = mat256.request();
    const double* mat = static_cast<double*>(mbuf.ptr);
    double x = 0.0;
    for (unsigned char c : s) x += mat[(size_t)c * 256 + c];
    return factor * x;
}

PYBIND11_MODULE(cdrscore, m) {
    m.def("score_max", &score_max);
    m.def("selfscore", &selfscore);
}

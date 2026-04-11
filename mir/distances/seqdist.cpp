/*
 * seqdist — C-native sequence distance and scoring functions.
 *
 * Compiled as pybind11 module ``seqdist_c``.  Accepts Python str, bytes, or bytearray.
 *
 * Distance functions:
 *   hamming(a, b)      → int     Hamming distance (equal-length sequences)
 *   levenshtein(a, b)  → int     Levenshtein (edit) distance
 *
 * CDR3 alignment scoring (from former cdrscore module):
 *   score_max(s1, s2, mat256, gaps, gap_pen, v_off, j_off, factor, use_mat) → double
 *   selfscore(s, mat256, factor, use_mat) → double
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

/* ================================================================
 * Helper: extract raw pointer + length from str or bytes
 * ================================================================ */

struct SeqView {
    const char* data;
    size_t      len;
};

static SeqView to_view(const py::object& obj) {
    if (py::isinstance<py::str>(obj)) {
        Py_ssize_t sz = 0;
        const char* p = PyUnicode_AsUTF8AndSize(obj.ptr(), &sz);
        if (!p) throw py::error_already_set();
        return {p, (size_t)sz};
    }
    if (py::isinstance<py::bytes>(obj)) {
        char* buf = nullptr;
        Py_ssize_t sz = 0;
        PyBytes_AsStringAndSize(obj.ptr(), &buf, &sz);
        return {buf, (size_t)sz};
    }
    if (py::isinstance<py::bytearray>(obj)) {
        const char* buf = PyByteArray_AS_STRING(obj.ptr());
        size_t sz = (size_t)PyByteArray_GET_SIZE(obj.ptr());
        return {buf, sz};
    }
    throw py::type_error("expected str, bytes, or bytearray");
}

/* ================================================================
 * Hamming distance
 * ================================================================ */

static int c_hamming(const py::object& a, const py::object& b) {
    auto sa = to_view(a);
    auto sb = to_view(b);
    if (sa.len != sb.len)
        throw std::invalid_argument("sequences must have equal length for hamming distance");
    int d = 0;
    for (size_t i = 0; i < sa.len; ++i)
        d += (sa.data[i] != sb.data[i]);
    return d;
}

/* ================================================================
 * Levenshtein distance  (classic DP, two-row, O(min(m,n)) space)
 * ================================================================ */

static int c_levenshtein(const py::object& a, const py::object& b) {
    auto sa = to_view(a);
    auto sb = to_view(b);
    size_t m = sa.len, n = sb.len;
    const char* s = sa.data;
    const char* t = sb.data;
    if (m > n) { std::swap(s, t); std::swap(m, n); }
    std::vector<int> prev(m + 1), curr(m + 1);
    for (size_t i = 0; i <= m; ++i) prev[i] = (int)i;
    for (size_t j = 1; j <= n; ++j) {
        curr[0] = (int)j;
        for (size_t i = 1; i <= m; ++i) {
            int cost = (s[i-1] != t[j-1]) ? 1 : 0;
            int del_ = prev[i] + 1;
            int ins  = curr[i-1] + 1;
            int sub  = prev[i-1] + cost;
            curr[i] = std::min({del_, ins, sub});
        }
        std::swap(prev, curr);
    }
    return prev[m];
}

/* ================================================================
 * CDR3 alignment scoring (merged from cdrscore)
 * ================================================================ */

static inline int norm_pos(int p, int m) {
    if (p >= 0) return p > m ? m : p;
    int q = m + p;
    return q < 0 ? 0 : q;
}

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

static double c_score_max(const std::string& s1, const std::string& s2,
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

static double c_selfscore(const std::string& s,
                 py::array_t<double, py::array::c_style | py::array::forcecast> mat256,
                 double factor, bool use_mat) {
    if (!use_mat) return 0.0;
    auto mbuf = mat256.request();
    const double* mat = static_cast<double*>(mbuf.ptr);
    double x = 0.0;
    for (unsigned char c : s) x += mat[(size_t)c * 256 + c];
    return factor * x;
}

/* ================================================================
 * Module definition
 * ================================================================ */

PYBIND11_MODULE(seqdist_c, m) {
    m.doc() = "C-native sequence distance and CDR3 scoring functions";

    m.def("hamming", &c_hamming,
          py::arg("a"), py::arg("b"),
          "Hamming distance between two equal-length sequences");
    m.def("levenshtein", &c_levenshtein,
          py::arg("a"), py::arg("b"),
          "Levenshtein (edit) distance between two sequences");
    m.def("score_max", &c_score_max,
          "Best CDR3 alignment score over a set of gap positions");
    m.def("selfscore", &c_selfscore,
          "Self-alignment score (diagonal of substitution matrix)");
}

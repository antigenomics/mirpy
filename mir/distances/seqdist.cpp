/*
 * seqdist — C-native sequence distance functions.
 *
 * Compiled as a pybind11 module.  Accepts Python str, bytes, or bytearray.
 *
 * Functions:
 *   hamming(a, b)      → int   Hamming distance (equal-length sequences)
 *   levenshtein(a, b)  → int   Levenshtein (edit) distance
 */

#include <pybind11/pybind11.h>
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
 * Module definition
 * ================================================================ */

PYBIND11_MODULE(seqdist_c, m) {
    m.doc() = "C-native sequence distance functions (Hamming, Levenshtein)";

    m.def("hamming", &c_hamming,
          py::arg("a"), py::arg("b"),
          "Hamming distance between two equal-length sequences");
    m.def("levenshtein", &c_levenshtein,
          py::arg("a"), py::arg("b"),
          "Levenshtein (edit) distance between two sequences");
}

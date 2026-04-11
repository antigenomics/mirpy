/*
 * mirseq — C-native sequence translation, tokenization, and distances.
 *
 * Compiled as a pybind11 module.  All functions accept Python str or bytes
 * via py::bytes / std::string_view and return Python list[str] or list[bytes].
 *
 * Lookup tables are compile-time constants (constexpr arrays) — no heap
 * allocation, no GC interaction.
 *
 * Functions:
 *   translate_linear(nt_seq)         → str   amino-acid translation (linear)
 *   translate_bidi(nt_seq)           → str   amino-acid translation (bidirectional)
 *   aa_to_reduced(aa_seq)            → str   reduced amino-acid alphabet
 *   tokenize_bytes(seq, k)           → list[bytes]  sliding window k-mers
 *   tokenize_str(seq, k)             → list[str]    sliding window k-mers
 *   tokenize_gapped_bytes(seq,k,m)   → list[bytes]  gapped k-mers
 *   tokenize_gapped_str(seq,k,m)     → list[str]    gapped k-mers
 *   hamming(a, b)                    → int   hamming distance
 *   levenshtein(a, b)                → int   levenshtein distance
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

/* ================================================================
 * Codon table  (64 entries, indexed by 2-bit packed nucleotides)
 * A=0, T=1, G=2, C=3.  Any codon containing N → 'X'.
 * ================================================================ */

// Nucleotide to 2-bit index: A=0, T=1, G=2, C=3, else -1
struct NtIdx {
    int v[256];
    constexpr NtIdx() : v{} {
        for (int i = 0; i < 256; ++i) v[i] = -1;
        v['A'] = 0; v['T'] = 1; v['G'] = 2; v['C'] = 3;
    }
};
static constexpr NtIdx nt_idx{};

// Standard genetic code: index = n1*16 + n2*4 + n3
// Order: A=0 T=1 G=2 C=3
static constexpr char CODON_TABLE[64] = {
    // AAA AAT AAG AAC  ATA ATT ATG ATC  AGA AGT AGG AGC  ACA ACT ACG ACC
    'K','N','K','N',    'I','I','M','I',  'R','S','R','S',  'T','T','T','T',
    // TAA TAT TAG TAC  TTA TTT TTG TTC  TGA TGT TGG TGC  TCA TCT TCG TCC
    '*','Y','*','Y',    'L','F','L','F',  '*','C','W','C',  'S','S','S','S',
    // GAA GAT GAG GAC  GTA GTT GTG GTC  GGA GGT GGG GGC  GCA GCT GCG GCC
    'E','D','E','D',    'V','V','V','V',  'G','G','G','G',  'A','A','A','A',
    // CAA CAT CAG CAC  CTA CTT CTG CTC  CGA CGT CGG CGC  CCA CCT CCG CCC
    'Q','H','Q','H',    'L','L','L','L',  'R','R','R','R',  'P','P','P','P',
};

static inline char translate_codon(unsigned char n1, unsigned char n2, unsigned char n3) {
    int i1 = nt_idx.v[n1], i2 = nt_idx.v[n2], i3 = nt_idx.v[n3];
    if (i1 < 0 || i2 < 0 || i3 < 0) return 'X';
    return CODON_TABLE[i1 * 16 + i2 * 4 + i3];
}

/* ================================================================
 * Amino acid → reduced alphabet lookup table (256 entries)
 * ================================================================ */

struct ReducedLut {
    char v[256];
    constexpr ReducedLut() : v{} {
        for (int i = 0; i < 256; ++i) v[i] = 0;
        v['A'] = 'l'; v['R'] = 'b'; v['N'] = 'm'; v['D'] = 'c';
        v['C'] = 's'; v['Q'] = 'm'; v['E'] = 'c'; v['G'] = 'G';
        v['H'] = 'b'; v['I'] = 'l'; v['L'] = 'l'; v['K'] = 'b';
        v['M'] = 's'; v['F'] = 'F'; v['P'] = 'P'; v['S'] = 'h';
        v['T'] = 'h'; v['W'] = 'W'; v['Y'] = 'Y'; v['V'] = 'l';
        v['X'] = 'X'; v['*'] = '*'; v['_'] = '_';
    }
};
static constexpr ReducedLut reduced_lut{};

/* ================================================================
 * Helper: extract raw pointer + length from str or bytes
 * ================================================================ */

struct SeqView {
    const char* data;
    size_t      len;
};

static SeqView to_view(const py::object& obj) {
    if (py::isinstance<py::str>(obj)) {
        // str → UTF-8 (ASCII subset)
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
 * Translation: linear
 * ================================================================ */

static py::str translate_linear(const py::object& obj) {
    auto sv = to_view(obj);
    size_t n = sv.len;
    size_t full_codons = n / 3;
    bool incomplete = (n % 3) != 0;
    size_t out_len = full_codons + (incomplete ? 1 : 0);
    std::string result(out_len, '\0');
    const char* s = sv.data;
    for (size_t i = 0; i < full_codons; ++i) {
        result[i] = translate_codon((unsigned char)s[i*3],
                                     (unsigned char)s[i*3+1],
                                     (unsigned char)s[i*3+2]);
    }
    if (incomplete) result[full_codons] = '_';
    return py::str(result);
}

/* ================================================================
 * Translation: bidirectional
 * ================================================================ */

static py::str translate_bidi(const py::object& obj) {
    auto sv = to_view(obj);
    size_t n = sv.len;
    if (n == 0) return py::str("");

    size_t remainder = n % 3;
    if (remainder == 0) {
        // Exact multiple of 3: just translate linearly
        size_t n_codons = n / 3;
        std::string result(n_codons, '\0');
        const char* s = sv.data;
        for (size_t i = 0; i < n_codons; ++i)
            result[i] = translate_codon((unsigned char)s[i*3],
                                         (unsigned char)s[i*3+1],
                                         (unsigned char)s[i*3+2]);
        return py::str(result);
    }

    // Not multiple of 3: bidirectional with gap
    size_t n_codons = n / 3;  // full codons available
    // Determine forward and reverse codon counts
    // For long sequences (>= 9 codons worth = 27 nt): gap after 4th codon from start
    // For shorter sequences: gap in the middle
    size_t fwd_codons, rev_codons;
    if (n >= 9 * 3) {
        fwd_codons = 4;
        rev_codons = n_codons - 4;
    } else {
        fwd_codons = n_codons / 2;
        rev_codons = n_codons - fwd_codons;
    }

    // out_len = fwd + 1 (gap) + rev
    size_t out_len = fwd_codons + 1 + rev_codons;
    std::string result(out_len, '\0');
    const char* s = sv.data;

    // Forward codons from start
    for (size_t i = 0; i < fwd_codons; ++i)
        result[i] = translate_codon((unsigned char)s[i*3],
                                     (unsigned char)s[i*3+1],
                                     (unsigned char)s[i*3+2]);

    // Gap
    result[fwd_codons] = '_';

    // Reverse codons from end
    for (size_t i = 0; i < rev_codons; ++i) {
        size_t nt_pos = n - (rev_codons - i) * 3;
        result[fwd_codons + 1 + i] = translate_codon(
            (unsigned char)s[nt_pos],
            (unsigned char)s[nt_pos+1],
            (unsigned char)s[nt_pos+2]);
    }

    return py::str(result);
}

/* ================================================================
 * AA → reduced alphabet
 * ================================================================ */

static py::str c_aa_to_reduced(const py::object& obj) {
    auto sv = to_view(obj);
    std::string result(sv.len, '\0');
    for (size_t i = 0; i < sv.len; ++i) {
        char c = reduced_lut.v[(unsigned char)sv.data[i]];
        result[i] = c ? c : sv.data[i]; // pass-through unmapped
    }
    return py::str(result);
}

/* ================================================================
 * Tokenization: sliding window → list[bytes] / list[str]
 * ================================================================ */

static py::list c_tokenize_bytes(const py::object& obj, int k) {
    auto sv = to_view(obj);
    int n = (int)sv.len;
    if (k < 1 || k > n)
        throw std::invalid_argument("k must be between 1 and sequence length");
    int count = n - k + 1;
    py::list result(count);
    for (int i = 0; i < count; ++i)
        result[i] = py::bytes(sv.data + i, k);
    return result;
}

static py::list c_tokenize_str(const py::object& obj, int k) {
    auto sv = to_view(obj);
    int n = (int)sv.len;
    if (k < 1 || k > n)
        throw std::invalid_argument("k must be between 1 and sequence length");
    int count = n - k + 1;
    py::list result(count);
    for (int i = 0; i < count; ++i)
        result[i] = py::str(std::string(sv.data + i, k));
    return result;
}

/* ================================================================
 * Tokenization: sliding window + mask → list[bytes] / list[str]
 * ================================================================ */

static py::list c_tokenize_gapped_bytes(const py::object& obj, int k, int mask_byte) {
    auto sv = to_view(obj);
    int n = (int)sv.len;
    if (k < 1 || k > n)
        throw std::invalid_argument("k must be between 1 and sequence length");
    int n_windows = n - k + 1;
    int total = n_windows * k;
    py::list result(total);
    // Temporary buffer for each gapped k-mer
    char* buf = (char*)alloca(k);
    int idx = 0;
    for (int i = 0; i < n_windows; ++i) {
        for (int j = 0; j < k; ++j) {
            std::memcpy(buf, sv.data + i, k);
            buf[j] = (char)mask_byte;
            result[idx++] = py::bytes(buf, k);
        }
    }
    return result;
}

static py::list c_tokenize_gapped_str(const py::object& obj, int k, int mask_byte) {
    auto sv = to_view(obj);
    int n = (int)sv.len;
    if (k < 1 || k > n)
        throw std::invalid_argument("k must be between 1 and sequence length");
    int n_windows = n - k + 1;
    int total = n_windows * k;
    py::list result(total);
    char* buf = (char*)alloca(k);
    int idx = 0;
    for (int i = 0; i < n_windows; ++i) {
        for (int j = 0; j < k; ++j) {
            std::memcpy(buf, sv.data + i, k);
            buf[j] = (char)mask_byte;
            result[idx++] = py::str(std::string(buf, k));
        }
    }
    return result;
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
    // Ensure m <= n for space optimisation
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

PYBIND11_MODULE(mirseq, m) {
    m.doc() = "C-native sequence translation, tokenization, and distances";

    // Translation
    m.def("translate_linear", &translate_linear,
          py::arg("seq"),
          "Translate nucleotide sequence to amino acids (linear, incomplete codon → '_')");
    m.def("translate_bidi", &translate_bidi,
          py::arg("seq"),
          "Translate nucleotide sequence to amino acids (bidirectional, gap '_' inserted)");
    m.def("aa_to_reduced", &c_aa_to_reduced,
          py::arg("seq"),
          "Convert amino acid sequence to reduced alphabet");

    // Tokenization
    m.def("tokenize_bytes", &c_tokenize_bytes,
          py::arg("seq"), py::arg("k"),
          "Sliding window k-mers as list[bytes]");
    m.def("tokenize_str", &c_tokenize_str,
          py::arg("seq"), py::arg("k"),
          "Sliding window k-mers as list[str]");
    m.def("tokenize_gapped_bytes", &c_tokenize_gapped_bytes,
          py::arg("seq"), py::arg("k"), py::arg("mask_byte"),
          "Gapped k-mers (each position masked) as list[bytes]");
    m.def("tokenize_gapped_str", &c_tokenize_gapped_str,
          py::arg("seq"), py::arg("k"), py::arg("mask_byte"),
          "Gapped k-mers (each position masked) as list[str]");

    // Distances
    m.def("hamming", &c_hamming,
          py::arg("a"), py::arg("b"),
          "Hamming distance between two equal-length sequences");
    m.def("levenshtein", &c_levenshtein,
          py::arg("a"), py::arg("b"),
          "Levenshtein (edit) distance between two sequences");
}

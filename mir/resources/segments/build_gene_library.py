#!/usr/bin/env python3
"""Build gene libraries from OLGA models and IMGT V-QUEST.

Two output files are written next to this script:

* ``olga_gene_library.txt`` — alleles extracted from OLGA default
  ``model_params.txt`` files (no network required).
* ``imgt_gene_library.txt`` — alleles fetched from the IMGT V-QUEST
  reference directory (requires network access).

Both files share the same tab-separated schema::

    species   locus   gene   allele   sequence

where *species* is ``human`` or ``mouse``, *locus* is ``TRB`` / ``TRA`` /
etc., *gene* is ``V``, ``D``, or ``J``, *allele* is the full IMGT name
(e.g. ``TRBV3-1*02``), and *sequence* is the nucleotide sequence.

On every run a dated entry is appended to ``build_gene_library.log`` with the
build timestamp, current git commit hash, and per-(species, locus, gene) allele
counts.  When both libraries are built together (no ``--olga`` / ``--imgt``
flag), a consistency report comparing the two is appended as well.

Usage::

    python mir/resources/segments/build_gene_library.py           # build both + check
    python mir/resources/segments/build_gene_library.py --olga    # OLGA only
    python mir/resources/segments/build_gene_library.py --imgt    # IMGT only
    python mir/resources/segments/build_gene_library.py --check   # consistency only

Species aliases
---------------
The string ``"human"`` maps to ``Homo_sapiens`` in IMGT URLs;
``"mouse"`` maps to ``Mus_musculus``.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import olga


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

RESOURCES_DIR = Path(__file__).parent

#: Build log — one appended section per run.
LOG_PATH = RESOURCES_DIR / "build_gene_library.log"

#: Maps canonical species name → IMGT organism string used in URLs.
SPECIES_IMGT_MAP: dict[str, str] = {
    "human": "Homo_sapiens",
    "mouse": "Mus_musculus",
}

#: Maps OLGA model-directory name → (species, locus).
OLGA_MODEL_MAP: dict[str, tuple[str, str]] = {
    "human_T_beta":   ("human", "TRB"),
    "human_T_alpha":  ("human", "TRA"),
    "human_T_gamma":  ("human", "TRG"),
    "human_T_delta":  ("human", "TRD"),
    "human_B_heavy":  ("human", "IGH"),
    "human_B_kappa":  ("human", "IGK"),
    "human_B_lambda": ("human", "IGL"),
    "mouse_T_beta":   ("mouse", "TRB"),
    "mouse_T_alpha":  ("mouse", "TRA"),
    # mouse_B_* models use synthetic non-IMGT allele names (*00 suffix); excluded
}

#: Loci to fetch from IMGT, keyed by species.
IMGT_LOCI: dict[str, list[str]] = {
    "human": ["TRB", "TRA", "TRG", "TRD", "IGH", "IGK", "IGL"],
    "mouse": ["TRB", "TRA", "TRG", "TRD", "IGH", "IGK", "IGL"],
}

#: Loci that carry a D (Diversity) gene segment.
LOCI_WITH_D: set[str] = {"TRB", "TRD", "IGH"}

# Internal: IMGT URL gene-family prefix for each locus.
_LOCUS_FAMILY: dict[str, str] = {
    locus: locus[:2]
    for locus in ["TRB", "TRA", "TRG", "TRD", "IGH", "IGK", "IGL"]
}

# Internal: OLGA model_params.txt section-header keyword → gene label.
_OLGA_SEGMENT_KEYS: dict[str, str] = {
    "V_gene": "V",
    "D_gene": "D",
    "J_gene": "J",
}

#: Type alias for a single gene-library row.
Row = tuple[str, str, str, str, str]  # (species, locus, gene, allele, sequence)


# ---------------------------------------------------------------------------
# OLGA library
# ---------------------------------------------------------------------------

def _parse_olga_model_params(path: Path) -> list[tuple[str, str, str]]:
    """Parse an OLGA ``model_params.txt`` file.

    Args:
        path: Path to a ``model_params.txt`` file.

    Returns:
        List of ``(gene, allele, sequence)`` triples where *gene* is
        ``"V"``, ``"D"``, or ``"J"``.  Entries whose allele name does not
        contain ``*`` (e.g. combinatorial D-D numeric indices in TRD models)
        are silently skipped.
    """
    records: list[tuple[str, str, str]] = []
    current_gene: str | None = None

    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            if line.startswith("#GeneChoice"):
                current_gene = None
                for key, label in _OLGA_SEGMENT_KEYS.items():
                    if key in line:
                        current_gene = label
                        break
                continue

            # Gene entry lines start with '%'; D-gene entries have an extra space.
            if line.startswith("%") and current_gene is not None:
                body = line[1:].strip()
                parts = body.split(";")
                if len(parts) >= 2:
                    allele   = parts[0].strip()
                    sequence = parts[1].strip()
                    if allele and sequence and "*" in allele:
                        records.append((current_gene, allele, sequence))

    return records


def build_olga_library(models_dirs: list[Path] | None = None) -> list[Row]:
    """Build the OLGA gene library from ``model_params.txt`` files.

    Searches *models_dirs* in order; the first directory containing a given
    OLGA model subdirectory is used.  When *models_dirs* is ``None`` the
    default search path is used: the local ``olga/default_models/`` copy
    bundled with this package (prioritised, contains human TRG/TRD), then
    the installed ``olga`` package directory.

    Args:
        models_dirs: Ordered list of directories to search for OLGA model
            subdirectories.  ``None`` uses the default search paths.

    Returns:
        List of ``(species, locus, gene, allele, sequence)`` rows.
    """
    if models_dirs is None:
        models_dirs = _default_olga_dirs()

    rows: list[Row] = []
    for model_name, (species, locus) in OLGA_MODEL_MAP.items():
        params_file: Path | None = None
        for d in models_dirs:
            candidate = d / model_name / "model_params.txt"
            if candidate.exists():
                params_file = candidate
                break

        if params_file is None:
            print(f"  [skip] {model_name}: model_params.txt not found", file=sys.stderr)
            continue

        for gene, allele, sequence in _parse_olga_model_params(params_file):
            rows.append((species, locus, gene, allele, sequence))

    return rows


def _default_olga_dirs() -> list[Path]:
    """Return the default OLGA model search paths (local copy first)."""
    local   = RESOURCES_DIR / "olga" / "default_models"
    package = Path(olga.__file__).parent / "default_models"
    return [d for d in (local, package) if d.exists()]


# ---------------------------------------------------------------------------
# IMGT library
# ---------------------------------------------------------------------------

def _parse_imgt_fasta(text: str, species: str, locus: str, gene: str) -> list[Row]:
    """Parse IMGT V-QUEST FASTA text into gene-library rows.

    IMGT FASTA headers are pipe-separated; field[1] is the allele name and
    field[15] is the nucleotide sequence (gaps represented by ``.``).

    Args:
        text: Raw FASTA text downloaded from IMGT.
        species: Canonical species name (e.g. ``"human"``).
        locus: Locus identifier (e.g. ``"TRB"``).
        gene: Segment type: ``"V"``, ``"D"``, or ``"J"``.

    Returns:
        List of ``(species, locus, gene, allele, sequence)`` rows.
    """
    rows: list[Row] = []
    entries = text.replace("\n", "").split(">")
    for entry in entries[1:]:
        if not entry:
            continue
        fields = entry.split("|")
        if len(fields) < 16:
            continue
        allele = fields[1].strip()
        if not allele:
            continue
        if "*" not in allele:
            allele += "*01"
        sequence = fields[15].replace(".", "").upper()
        if sequence:
            rows.append((species, locus, gene, allele, sequence))
    return rows


def build_imgt_library(
    species_list: list[str] | None = None,
    loci_map: dict[str, list[str]] | None = None,
) -> list[Row]:
    """Fetch V/D/J alleles from the IMGT V-QUEST reference directory.

    Requires a live network connection.  Loci listed in :data:`LOCI_WITH_D`
    (``TRB``, ``TRD``, ``IGH``) are fetched for V, D, and J; all other loci
    are fetched for V and J only.  Failed requests are logged as warnings and
    skipped so a partial result is always returned.

    Args:
        species_list: Species to fetch.  Defaults to all keys in
            :data:`IMGT_LOCI`.
        loci_map: Override loci per species.  Defaults to :data:`IMGT_LOCI`.

    Returns:
        List of ``(species, locus, gene, allele, sequence)`` rows.
    """
    if species_list is None:
        species_list = list(IMGT_LOCI.keys())
    if loci_map is None:
        loci_map = IMGT_LOCI

    rows: list[Row] = []
    for species in species_list:
        imgt_organism = SPECIES_IMGT_MAP[species]
        for locus in loci_map.get(species, []):
            family     = _LOCUS_FAMILY[locus]
            gene_types = ["V", "J"] + (["D"] if locus in LOCI_WITH_D else [])
            for gene in gene_types:
                url = (
                    "https://www.imgt.org/download/V-QUEST/"
                    f"IMGT_V-QUEST_reference_directory/{imgt_organism}"
                    f"/{family}/{locus}{gene}.fasta"
                )
                try:
                    response = urllib.request.urlopen(url)
                    fetched  = _parse_imgt_fasta(
                        response.read().decode("utf-8"), species, locus, gene
                    )
                    print(f"  {species} {locus}{gene}: {len(fetched)} alleles")
                    rows.extend(fetched)
                except Exception as exc:
                    print(f"  [warn] {species} {locus}{gene}: {exc}", file=sys.stderr)

    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_library(rows: list[Row], output_path: Path) -> None:
    """Write *rows* to a tab-separated gene library file with a header row.

    Args:
        rows: Iterable of ``(species, locus, gene, allele, sequence)`` tuples.
        output_path: Destination file path (parent directory is created if
            needed).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        out.write("species\tlocus\tgene\tallele\tsequence\n")
        for row in rows:
            out.write("\t".join(row) + "\n")
    print(f"Wrote {len(rows)} allele entries → {output_path}")


def _load_rows(path: Path) -> list[Row]:
    """Load a gene library TSV file into a list of rows (header skipped)."""
    rows: list[Row] = []
    with path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 5:
                rows.append(tuple(parts))  # type: ignore[arg-type]
    return rows


def compute_stats(
    rows: list[Row],
) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
    """Compute allele counts per (species, locus, gene).

    Args:
        rows: Gene-library rows.

    Returns:
        Tuple ``(total, major)`` where both are dicts keyed by
        ``(species, locus, gene)``.  *total* counts all alleles; *major*
        counts only those whose allele name ends with ``*01``.
    """
    total: dict[tuple[str, str, str], int] = defaultdict(int)
    major: dict[tuple[str, str, str], int] = defaultdict(int)
    for species, locus, gene, allele, _ in rows:
        key = (species, locus, gene)
        total[key] += 1
        if allele.endswith("*01"):
            major[key] += 1
    return total, major


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=RESOURCES_DIR,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _format_stats_block(rows: list[Row]) -> str:
    """Return a fixed-width allele-count table string."""
    total, major = compute_stats(rows)
    col = (10, 8, 6, 12, 14)
    lines = [
        (
            f"{'species':<{col[0]}}"
            f"{'locus':<{col[1]}}"
            f"{'gene':<{col[2]}}"
            f"{'alleles':>{col[3]}}"
            f"{'major (*01)':>{col[4]}}"
        ),
        "-" * sum(col),
    ]
    for key in sorted(total):
        s, l, g = key
        lines.append(
            f"{s:<{col[0]}}{l:<{col[1]}}{g:<{col[2]}}"
            f"{total[key]:>{col[3]}}{major[key]:>{col[4]}}"
        )
    lines.append("-" * sum(col))
    lines.append(
        f"{'TOTAL':<{col[0]+col[1]+col[2]}}"
        f"{sum(total.values()):>{col[3]}}"
        f"{sum(major.values()):>{col[4]}}"
    )
    return "\n".join(lines)


def _log_section(title: str, body: str) -> str:
    """Wrap *body* in a dated log-section header."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    git_hash  = _get_git_hash()
    rule      = "=" * 78
    return f"\n{rule}\n{title}  |  {timestamp}  |  commit {git_hash}\n{rule}\n{body}\n"


def append_log(log_path: Path, section: str) -> None:
    """Append *section* to *log_path*, creating the file if needed."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(section)
    print(f"Appended log → {log_path}")


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

def check_library_consistency(
    olga_rows: list[Row],
    imgt_rows: list[Row],
) -> str:
    """Return a formatted consistency report comparing OLGA and IMGT libraries.

    The report lists:

    * ``(species, locus, gene)`` keys present in one library but not the
      other.
    * For every shared key: OLGA allele count, IMGT allele count, number of
      alleles present in both, number exclusive to each library.

    Args:
        olga_rows: Rows from the OLGA gene library.
        imgt_rows: Rows from the IMGT gene library.

    Returns:
        Multi-line string suitable for appending to the build log.
    """
    olga_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    imgt_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for s, l, g, allele, _ in olga_rows:
        olga_by_key[(s, l, g)].add(allele)
    for s, l, g, allele, _ in imgt_rows:
        imgt_by_key[(s, l, g)].add(allele)

    olga_keys = set(olga_by_key)
    imgt_keys = set(imgt_by_key)
    all_keys  = sorted(olga_keys | imgt_keys)

    lines: list[str] = []

    only_olga = sorted(olga_keys - imgt_keys)
    only_imgt = sorted(imgt_keys - olga_keys)

    if only_olga:
        lines.append("[keys in OLGA, absent in IMGT]")
        for s, l, g in only_olga:
            lines.append(f"  {s} {l} {g}: {len(olga_by_key[(s, l, g)])} alleles")
        lines.append("")

    if only_imgt:
        lines.append("[keys in IMGT, absent in OLGA]")
        for s, l, g in only_imgt:
            lines.append(f"  {s} {l} {g}: {len(imgt_by_key[(s, l, g)])} alleles")
        lines.append("")

    col = (10, 7, 5, 8, 8, 8, 10, 10)
    header = (
        f"{'species':<{col[0]}}{'locus':<{col[1]}}{'gene':<{col[2]}}"
        f"{'OLGA':>{col[3]}}{'IMGT':>{col[4]}}{'shared':>{col[5]}}"
        f"{'only-OLGA':>{col[6]}}{'only-IMGT':>{col[7]}}"
    )
    lines.append("[allele counts per (species, locus, gene)]")
    lines.append(header)
    lines.append("-" * len(header))
    for key in all_keys:
        s, l, g  = key
        o_set    = olga_by_key.get(key, set())
        i_set    = imgt_by_key.get(key, set())
        n_shared = len(o_set & i_set)
        n_only_o = len(o_set - i_set)
        n_only_i = len(i_set - o_set)
        lines.append(
            f"{s:<{col[0]}}{l:<{col[1]}}{g:<{col[2]}}"
            f"{len(o_set):>{col[3]}}{len(i_set):>{col[4]}}{n_shared:>{col[5]}}"
            f"{n_only_o:>{col[6]}}{n_only_i:>{col[7]}}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_stats_cli(rows: list[Row], label: str) -> None:
    print(f"\n{label}")
    print(_format_stats_block(rows))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    only_olga  = "--olga"  in sys.argv
    only_imgt  = "--imgt"  in sys.argv
    only_check = "--check" in sys.argv

    run_olga  = not only_imgt  and not only_check
    run_imgt  = not only_olga  and not only_check
    run_check = not only_olga  and not only_imgt

    olga_rows: list[Row] = []
    imgt_rows: list[Row] = []

    if run_olga:
        print("=== Building OLGA library ===")
        for d in _default_olga_dirs():
            print(f"  Search path: {d}")
        olga_rows = build_olga_library()
        write_library(olga_rows, RESOURCES_DIR / "olga_gene_library.txt")
        _print_stats_cli(olga_rows, "OLGA library")
        append_log(LOG_PATH, _log_section("OLGA gene library", _format_stats_block(olga_rows)))

    if run_imgt:
        print("\n=== Building IMGT library (requires network) ===")
        imgt_rows = build_imgt_library()
        write_library(imgt_rows, RESOURCES_DIR / "imgt_gene_library.txt")
        _print_stats_cli(imgt_rows, "IMGT library")
        append_log(LOG_PATH, _log_section("IMGT gene library", _format_stats_block(imgt_rows)))

    if run_check:
        if not olga_rows:
            p = RESOURCES_DIR / "olga_gene_library.txt"
            if p.exists():
                olga_rows = _load_rows(p)
                print(f"Loaded {len(olga_rows)} rows from {p.name}")
        if not imgt_rows:
            p = RESOURCES_DIR / "imgt_gene_library.txt"
            if p.exists():
                imgt_rows = _load_rows(p)
                print(f"Loaded {len(imgt_rows)} rows from {p.name}")

        if olga_rows and imgt_rows:
            print("\n=== Consistency check (OLGA vs IMGT) ===")
            report = check_library_consistency(olga_rows, imgt_rows)
            print(report)
            append_log(LOG_PATH, _log_section("Consistency: OLGA vs IMGT", report))
        else:
            print("[skip] consistency check: one or both libraries missing", file=sys.stderr)

"""Convert immune-repertoire clone tables to AIRR TSV format.

Supported input formats
-----------------------
old_mixcr   Legacy MiXCR clone tables (tab-delimited, plain or gzipped).
vdjdb_slim  VDJdb slim export (tab-delimited, plain or gzipped).
olga_gen    OLGA sequence-generation output (no header, plain or gzipped).

Usage examples
--------------
Single file (auto-detected as gzipped from extension):
    python old_mixcr2airr.py sample.txt.gz -o sample.airr.tsv.gz

Glob mask → folder, gzipped output, explicit format:
    python old_mixcr2airr.py "data/*.txt.gz" -o airr_out/ -f old_mixcr -z

VDJdb slim, human only:
    python old_mixcr2airr.py vdjdb.slim.txt.gz -o vdjdb.airr.tsv -f vdjdb_slim
"""

from __future__ import annotations

import argparse
import sys
import time
from glob import glob
from pathlib import Path

from mir.common.parser import AIRRWriter, OldMiXCRParser, OlgaParser, VDJdbSlimParser

_PARSERS = {
    "old_mixcr":  OldMiXCRParser,
    "vdjdb_slim": VDJdbSlimParser,
    "olga_gen":   OlgaParser,
}

_DEFAULT_EXT = ".airr.tsv"
_GZ_EXT      = ".airr.tsv.gz"


def _resolve_output(src: Path, output: Path, compress: bool) -> Path:
    """Return the destination path for *src* given the CLI *output* argument."""
    ext = _GZ_EXT if compress else _DEFAULT_EXT
    if output.is_dir() or str(output).endswith("/"):
        stem = src.stem                          # strips .gz
        if stem.endswith((".txt", ".tsv")):
            stem = Path(stem).stem
        return output / f"{stem}{ext}"
    return output


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Convert immune-repertoire clone tables to AIRR TSV format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[0],
    )
    ap.add_argument(
        "input",
        help="Input file path or glob mask (quote globs to prevent shell expansion).",
    )
    ap.add_argument(
        "-o", "--output",
        required=True,
        help="Output file (single input) or directory (multiple inputs).",
    )
    ap.add_argument(
        "-f", "--format",
        dest="fmt",
        default="old_mixcr",
        choices=list(_PARSERS),
        help="Input file format (default: old_mixcr).",
    )
    ap.add_argument(
        "-z", "--compress",
        action="store_true",
        default=False,
        help="Write gzip-compressed output (.airr.tsv.gz). "
             "Also inferred when --output ends in .gz.",
    )
    args = ap.parse_args(argv)

    sources = sorted(Path(p) for p in glob(args.input))
    if not sources:
        print(f"No files matched: {args.input}", file=sys.stderr)
        sys.exit(1)

    output  = Path(args.output)
    compress = args.compress or str(args.output).endswith(".gz")

    if len(sources) > 1:
        if output.exists() and not output.is_dir():
            print("Output must be a directory when multiple input files are given.",
                  file=sys.stderr)
            sys.exit(1)
        output.mkdir(parents=True, exist_ok=True)

    parser = _PARSERS[args.fmt]()
    writer = AIRRWriter(compress=compress)

    total_clonotypes = 0
    n_files = len(sources)

    for i, src in enumerate(sources, 1):
        dst = _resolve_output(src, output, compress)
        prefix = f"[{i}/{n_files}]"
        print(f"{prefix} {src.name} …", end="\r", flush=True)

        t0 = time.perf_counter()
        sample = parser.parse_file(src)
        writer.write(sample, dst)
        elapsed = time.perf_counter() - t0

        n = sum(lr.clonotype_count for lr in sample)
        loci = ", ".join(
            f"{locus}:{lr.clonotype_count}"
            for locus, lr in sample.loci.items()
        )
        print(f"{prefix} {src.name} → {dst.name}  ({n:,} clonotypes: {loci}) [{elapsed:.1f}s]")
        total_clonotypes += n

    if n_files > 1:
        print(f"Done: {total_clonotypes:,} clonotypes from {n_files} file(s)")


if __name__ == "__main__":
    main()

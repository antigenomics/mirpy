#!/usr/bin/env bash
# Fetch the latest VDJdb release and extract human TRB CDR3 sequences
# specific for HLA-A*02:01 restricted GILGFVFTL epitope (Influenza M1).
#
# Output: tests/assets/gilgfvftl_trb_cdr3.txt  (one CDR3aa per line, deduplicated)
#
# Usage:  bash tests/assets/fetch_vdjdb_gilgfvftl.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTFILE="$SCRIPT_DIR/gilgfvftl_trb_cdr3.txt"

# Skip download if output already exists
if [ -f "$OUTFILE" ]; then
    n=$(wc -l < "$OUTFILE")
    echo "Already have $OUTFILE ($n sequences), skipping download."
    exit 0
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# --- Fetch latest VDJdb release URL via GitHub API ---
echo "Querying GitHub for latest VDJdb release..."
ASSET_URL=$(curl -sL https://api.github.com/repos/antigenomics/vdjdb-db/releases/latest \
    | grep -o '"browser_download_url": *"[^"]*"' \
    | head -1 \
    | sed 's/"browser_download_url": *"//;s/"//')

if [ -z "$ASSET_URL" ]; then
    echo "ERROR: Could not determine download URL" >&2
    exit 1
fi

echo "Downloading $ASSET_URL ..."
curl -sL "$ASSET_URL" -o "$TMPDIR/vdjdb.zip"
unzip -q "$TMPDIR/vdjdb.zip" -d "$TMPDIR/vdjdb"

# --- Find the main database file ---
DBFILE=$(find "$TMPDIR/vdjdb" -name 'vdjdb.txt' -not -name 'vdjdb.slim.txt' | head -1)
if [ -z "$DBFILE" ]; then
    # Fall back to slim
    DBFILE=$(find "$TMPDIR/vdjdb" -name 'vdjdb.slim.txt' | head -1)
fi
if [ -z "$DBFILE" ]; then
    echo "ERROR: Could not find vdjdb.txt in archive" >&2
    exit 1
fi
echo "Using database file: $DBFILE"

# --- Filter: human TRB, HLA-A*02, GILGFVFTL epitope ---
# VDJdb columns (tab-separated):
#   0:complex.id  1:Gene  2:CDR3  3:V  4:J  5:Species  6:MHC A  7:MHC B
#   8:MHC class  9:Epitope  10:Epitope gene  11:Epitope species  ...
#
# Filter criteria:
#   Gene == TRB
#   Species == HomoSapiens
#   Epitope == GILGFVFTL
#   MHC A contains A*02
#   CDR3 starts with C and ends with F (canonical)
awk -F'\t' 'NR > 1 && $2 == "TRB" && $6 == "HomoSapiens" && $10 == "GILGFVFTL" && $7 ~ /A\*02/ && $3 ~ /^C[A-Z]+F$/ { print $3 }' "$DBFILE" \
    | sort -u > "$OUTFILE"

n=$(wc -l < "$OUTFILE")
echo "Wrote $n unique CDR3 sequences to $OUTFILE"

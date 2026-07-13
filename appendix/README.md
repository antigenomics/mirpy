# Appendix T — TCREMP embedding theory

`tcremp_theory.tex` is the research-level mathematical appendix behind the TCREMP prototype
embedding (Appendix T; same house style as `../../vdjtools/appendix`). Sections **T.1–T.7**:
foundations of prototype (landmark) embeddings → the alignment dissimilarity as a metric → the
V(D)J rearrangement model → TCRs & paired chains → IGH / somatic hypermutation → density geometry
& background subtraction → insights and open problems. Every quantitative claim is reproduced by
`mir/bench/theory.py` (Table T.1).

## Build

```
make            # -> tcremp_theory.pdf   (latexmk + lualatex + bibtex)
make clean
```

Requires **lualatex** (OldStandard OpenType math), **gnuplot**, **graphviz** (`dot`), and — only to
regenerate figure data — the conda `mirpy` env with the `[bench]` extra.

## Figures (data and plotting kept separate)

The pipeline writes plain-text data; gnuplot and dot render it (no matplotlib):

- `gen_theory_data.py` (conda `mirpy` env) reuses `mir.bench.theory` to write `data/*.tsv` +
  `data/theory_stats.txt` (S1–S3 dissimilarity/distance data, distribution fits, correlations).
- `fig_s2_correlation.gp`, `fig_s1_distributions.gp`, `fig_s3_prototype_source.gp` render the TSVs.
- `diagram_embedding.dot` is the pipeline schematic (`dot -Tpdf`).
- `make data` regenerates the text data; `make figures` re-renders the PDFs.

## gnuplot-palettes (not vendored, not a submodule)

The `.gp` scripts colour with ColorBrewer palettes from
[gnuplot-palettes](https://github.com/Gnuplotting/gnuplot-palettes). It is **not** committed here.
Clone it once and either put it at `../../gnuplot-palettes` (the default) or point `GNUPLOT_PALETTES`
at it:

```
git clone https://github.com/Gnuplotting/gnuplot-palettes.git ../../gnuplot-palettes
# or:  export GNUPLOT_PALETTES=/path/to/gnuplot-palettes
```

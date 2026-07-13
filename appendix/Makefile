# Compile the TCREMP embedding-theory appendix (Appendix T). Requires lualatex (OldStandard
# OpenType fonts; pinned in .latexmkrc) + bibtex, gnuplot, and graphviz (dot).
#
# Figures follow the group convention: the pipeline writes plain-text data (gen_theory_data.py,
# needs the conda `mirpy` env), then gnuplot + dot render it (data and plotting kept separate).
# gnuplot uses ColorBrewer palettes from gnuplot-palettes (github.com/Gnuplotting/gnuplot-palettes)
# -- NOT vendored here; set $GNUPLOT_PALETTES or clone it to ../gnuplot-palettes.

GP_FIGS  = fig_s2_correlation.pdf fig_s1_distributions.pdf fig_s3_prototype_source.pdf
DOT_FIGS = diagram_embedding.pdf
FIGS     = $(GP_FIGS) $(DOT_FIGS)

tcremp_theory.pdf: tcremp_theory.tex refs.bib $(FIGS)
	latexmk -lualatex -interaction=nonstopmode -halt-on-error tcremp_theory.tex

# Regenerate the text data (S1-S3) from mir.bench.theory; run inside the conda `mirpy` env.
data/theory_stats.txt: gen_theory_data.py
	python gen_theory_data.py

$(GP_FIGS): %.pdf: %.gp data/theory_stats.txt
	gnuplot $<

$(DOT_FIGS): %.pdf: %.dot
	dot -Tpdf -o $@ $<

.PHONY: data figures clean
data: data/theory_stats.txt
figures: $(FIGS)
clean:
	latexmk -C tcremp_theory.tex
